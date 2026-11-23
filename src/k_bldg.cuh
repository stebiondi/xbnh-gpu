// k_bldg.cuh: sequential building failure computed inside the time loop (Sanderson and Slinn, in prep.).
//
// The original workflow wrote zs over the whole domain every 0.5 s (about 60 GB per 15 min segment on the
// 8.7 M cell Estero deck), then a Julia script computed the wave impulse at every building perimeter, removed
// the buildings whose cumulative impulse passed their threshold and edited the hotstart files. Here the same
// quantities are computed on the GPU, on the perimeter cells only, while the model runs:
//
//   h(t)    = max(zs - zb, 0) at a perimeter cell, sampled every bldg_tint s (0.5 s)
//   hbar(t) = centred bldg_tavg s (120 s) moving average of h      (ring buffer of bldg_tavg/bldg_tint + 1 samples)
//   eta     = h - hbar
//   F(t)    = rho g / 2 (2 hbar eta + eta^2) = rho g / 2 (h^2 - hbar^2)        force per unit wall width, N/m
//   J_cell  = sum max(F, 0) dt over the check interval                          (bldg_fpos = 0 keeps the sign)
//   J_seg(b)= max over the perimeter cells of building b of J_cell
//   J(b)   += J_seg(b) at each check (every hotstart write, i.e. every 15 min)
// bldg_tcheck > 0 (continuous mode): J_cell is not reset, it is the impulse since the start of the storm, and
// every bldg_tcheck s a building's impulse is J(b) = max over its perimeter of J_cell; a building fails at the
// first check where J(b) >= threshold, during the run, and the solver sees the new bed from the next step.
// bldg_tcheck = 0 (segment mode, the original workflow): checks only at hotstart writes, J(b) += J_seg(b).
// A standing building with J(b) >= its threshold (kN hr/m, from bldgfile) is removed: zb := bare earth
// (bldgbarefile) on its cells, zs := mean zs of its wet perimeter cells (bare earth when none is wet); uu, vv,
// umean, vmean are left as they are. The removal goes into the hotstart written at the same time, so the next
// segment starts with the building gone, as in the original workflow.
//
// Recorded, not used for failure (bldg_impulse.txt, while the building stands): the peak over time and over the
// perimeter of (1) the wave force F above, (2) the flow drag 1/2 rho Cd hw |U|^2 with Cd = 2 (= rho hw |U|^2; U the
// cell-centre velocity), (3) the quasi-static water-level difference across the building rho g / 2 (hwbar_max^2 -
// hwbar_min^2); with the times of the peaks. hw is the water height against the wall: water level at the perimeter
// cell minus the building's ground elevation (mean bare earth under its cells), not the cell depth, so a canal or
// a lower lot next to the building does not count as water on the wall; hwbar uses the 2 min mean water level.
// Total lateral load (bldg_impulse.txt, last columns): the same three water terms and the wind, all at one instant (the
// centre of the 2 min window, where F is evaluated), per unit wall width:
//   F_total(t) = max over the perimeter of [F + rho hw |U|^2]  +  rho g / 2 (hwbar_max^2 - hwbar_min^2)
//              + 1/2 rho_air C_f V(t)^2 max(H - hwbar_mean, 0)
// V: wind speed from windfile (uniform), C_f: bldg_wind_cf (1.3 = 0.8 windward + 0.5 leeward wall pressure), rho_air:
// bldg_rhoa (1.25), H: building height from bldg_heightfile (lines "id height_m", id as in bldg_impulse.txt; buildings
// not listed, or no file: no wind term). The peak of F_total over the storm is recorded with its time and its three
// parts at that time (wave + drag, quasi-static, wind): the parts peak at different times, so their separate peaks
// do not add up to the peak of the total.
//
// Buildings: 4-connected cells with the same threshold value (> 0). Standing: any cell raised more than 1 m
// above bare earth in the current bed. Perimeter of b: the 4-neighbours of its cells that are not cells of b
// (cells of a neighbouring standing building are dry, so they give F = 0 until that building goes).
// The list is fixed for the whole storm, so the ring buffer and impulses carry over between segments through
// bldg_state (binary; read on a hotstart when present, rewritten at every check).
#pragma once
#include <fstream>
#include <deque>
#include <sstream>
#include <algorithm>

__global__ void k_bldg_sample(int npc, const int *pc, const real *zs, const real *zb, const real *u, const real *v,
                              float *ring, double *J, float *hb, float *Fc, float *m2, float *ringU, float *hcc, float *m2c,
                              int W, long ns, double rhog2, double tint, int fpos) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npc) return;
    int k = pc[p];
    float hnow = fmaxf((float)(zs[k] - zb[k]), 0.f);
    ring[(size_t)(ns % W) * npc + p] = hnow;
    m2[p] = hnow > 0.f ? (float)(u[k] * u[k] + v[k] * v[k]) : 0.f;
    ringU[(size_t)(ns % W) * npc + p] = m2[p];
    if (ns < W - 1) { hb[p] = 0.f; Fc[p] = 0.f; hcc[p] = 0.f; m2c[p] = 0.f; return; }   // window not full yet
    double s = 0;
    for (int q = 0; q < W; q++) s += ring[(size_t)q * npc + p];
    double hbar = s / W;
    double hc = ring[(size_t)((ns - (W - 1) / 2) % W) * npc + p];   // centre of the window
    double F = rhog2 * (hc * hc - hbar * hbar);
    J[p] += (fpos ? fmax(F, 0.0) : F) * tint;
    hb[p] = (float)hbar; Fc[p] = (float)F;
    hcc[p] = (float)hc; m2c[p] = ringU[(size_t)((ns - (W - 1) / 2) % W) * npc + p];   // depth and |U|^2 at the centre
}

// per standing building: peaks over its perimeter of F, h|U|^2 and hbar_max^2 - hbar_min^2 (pk[3b..3b+2], times pt)
__global__ void k_bldg_peaks(int nb, const int *ofs, const int *idx, const char *stand, const float *hb, const float *Fc,
                             const float *m2, const int *pc, const real *zs, const real *zb, const float *zg,
                             double *pk, double *pt, double tF, double tM, int full,
                             const float *hcc, const float *m2c, const float *Hb, double rho, double rhog2, double qwind,
                             double *pk2, double *pt2) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb || !stand[b]) return;
    float fmx = 0.f, mmx = 0.f, hmx = 0.f, hmn = 1e30f, g0 = zg[b];
    double fwd = -1e300;                                                            // max of wave + drag at the centre
    for (int q = ofs[b]; q < ofs[b + 1]; q++) {
        int p = idx[q], k = pc[p];
        float hw = fmaxf((float)zs[k] - g0, 0.f);                                   // water on the wall now
        float hwb = hb[p] > 0.f ? fmaxf(hb[p] + (float)zb[k] - g0, 0.f) : 0.f;      // from the 2 min mean level
        fmx = fmaxf(fmx, Fc[p]); mmx = fmaxf(mmx, hw * m2[p]);
        hmx = fmaxf(hmx, hwb); hmn = fminf(hmn, hwb);
        float hwc = hcc[p] > 0.f ? fmaxf(hcc[p] + (float)zb[k] - g0, 0.f) : 0.f;   // on the wall at the centre
        fwd = fmax(fwd, (double)Fc[p] + rho * hwc * m2c[p]);
    }
    if (mmx > pk[3 * b + 1]) { pk[3 * b + 1] = mmx; pt[3 * b + 1] = tM; }
    if (!full) return;
    if (fmx > pk[3 * b]) { pk[3 * b] = fmx; pt[3 * b] = tF; }
    float ds = hmx * hmx - hmn * hmn;
    if (ds > pk[3 * b + 2]) { pk[3 * b + 2] = ds; pt[3 * b + 2] = tF; }
    if (ofs[b + 1] == ofs[b]) return;
    double fh = rhog2 * ds, H = Hb ? (double)Hb[b] : 0.0;
    double fw = qwind * fmax(H - 0.5 * ((double)hmx + hmn), 0.0);
    double tot = fwd + fh + fw;
    if (tot > pk2[4 * b]) { pk2[4 * b] = tot; pk2[4 * b + 1] = fwd; pk2[4 * b + 2] = fh; pk2[4 * b + 3] = fw; pt2[b] = tF; }
}

struct Buildings {
    bool on = false;
    int W = 241, fpos = 1;
    double tint = 0.5, tavg = 120, rho = 1025, g = 9.81, tnext = 0, wind_cf = 1.3, rhoa = 1.25;
    std::vector<double> wind_t, wind_s;             // wind speed series (set by the caller when wind is on)
    float *d_ringU = nullptr, *d_hcc = nullptr, *d_m2c = nullptr, *d_H = nullptr; double *d_pk2 = nullptr, *d_pt2 = nullptr;
    int nH = 0;
    long ns = 0;                                    // samples taken since the cold start
    int nb = 0, npc = 0;
    std::vector<double> thr;                        // N s / m
    std::vector<std::vector<int>> cells;            // building cells (k)
    std::vector<std::vector<int>> per;              // perimeter entries (index into pc) of each building
    std::vector<int> pc;                            // unique perimeter cells (k)
    std::vector<double> Jcum;                       // N s / m
    std::vector<char> standing;
    std::vector<double> bare;
    std::vector<int> ic, jc;                        // building centroid (grid indices) for the tables
    int *d_pc = nullptr; float *d_ring = nullptr; double *d_J = nullptr;
    float *d_hb = nullptr, *d_Fc = nullptr, *d_m2 = nullptr; int *d_ofs = nullptr, *d_idx = nullptr; char *d_st = nullptr;
    double *d_pk = nullptr, *d_pt = nullptr; float *d_zg = nullptr;
    std::string statefile;
    int nchecks = 0; double tlast = -1; int dump = 0, NX1_ = 0;
    double tcheck = 0, tnextcheck = 0; bool changed = false; long nremoved_run = 0;

    template <class IDX>
    void init(ParFile &F, const Params &P, int NX1, int NY1, IDX I, const std::vector<double> &zb) {
        std::string tf = F.name("bldgfile");
        if (tf.empty()) return;
        on = true;
        std::string bf = F.name("bldgbarefile");
        if (bf.empty()) fail("bldgfile needs bldgbarefile (bare-earth bed)");
        tint = F.dbl("bldg_tint", 0.5); tavg = F.dbl("bldg_tavg", 120.0); rho = F.dbl("bldg_rho", 1025.0);
        fpos = F.integer("bldg_fpos", 1); statefile = F.name("bldg_state", "bldg_state.bin"); g = P.g;
        dump = F.integer("bldg_dump", 0); NX1_ = NX1; tcheck = F.dbl("bldg_tcheck", 0.0);
        wind_cf = F.dbl("bldg_wind_cf", 1.3); rhoa = F.dbl("bldg_rhoa", 1.25);
        std::string hf = F.name("bldg_heightfile");
        W = (int)lround(tavg / tint) + 1; if (W % 2 == 0) W++;
        const size_t N = (size_t)NX1 * NY1;
        std::vector<double> th; read_grid_file(tf, NX1, NY1, th, true);
        read_grid_file(bf, NX1, NY1, bare, true);
        double posdwn = P.posdwn * fabs(P.posdwn);
        for (auto &z : bare) z = -z * posdwn;
        if (NY1 > 1) for (int i = 1; i <= NX1; i++) { bare[I(i, 1)] = bare[I(i, 2)]; bare[I(i, NY1)] = bare[I(i, NY1 - 1)]; }
        // label buildings: 4-connected, same threshold
        std::vector<int> lab(N, -1);
        for (int j = 1; j <= NY1; j++) for (int i = 1; i <= NX1; i++) {
            size_t k0 = I(i, j);
            if (th[k0] <= 0 || lab[k0] >= 0) continue;
            int id = nb++; std::vector<int> cl; std::deque<std::pair<int, int>> q{{i, j}}; lab[k0] = id;
            while (!q.empty()) {
                auto [a, b] = q.front(); q.pop_front(); cl.push_back((int)I(a, b));
                const int di[4] = {1, -1, 0, 0}, dj[4] = {0, 0, 1, -1};
                for (int m = 0; m < 4; m++) {
                    int a2 = a + di[m], b2 = b + dj[m];
                    if (a2 < 1 || a2 > NX1 || b2 < 1 || b2 > NY1) continue;
                    size_t k2 = I(a2, b2);
                    if (lab[k2] < 0 && th[k2] == th[k0]) { lab[k2] = id; q.push_back({a2, b2}); }
                }
            }
            cells.push_back(cl); thr.push_back(th[k0] * 3.6e6);   // kN hr/m -> N s/m
            ic.push_back(i); jc.push_back(j);
        }
        // perimeter
        std::vector<int> pidx(N, -1);
        per.resize(nb);
        for (int b = 0; b < nb; b++) {
            std::vector<int> mine;
            for (int k : cells[b]) {
                int i = k % NX1 + 1, j = k / NX1 + 1;
                const int di[4] = {1, -1, 0, 0}, dj[4] = {0, 0, 1, -1};
                for (int m = 0; m < 4; m++) {
                    int a2 = i + di[m], b2 = j + dj[m];
                    if (a2 < 1 || a2 > NX1 || b2 < 1 || b2 > NY1) continue;
                    size_t k2 = I(a2, b2);
                    if (lab[k2] == b) continue;
                    if (pidx[k2] < 0) { pidx[k2] = (int)pc.size(); pc.push_back((int)k2); }
                    mine.push_back(pidx[k2]);
                }
            }
            std::sort(mine.begin(), mine.end()); mine.erase(std::unique(mine.begin(), mine.end()), mine.end());
            per[b] = mine;
        }
        npc = (int)pc.size();
        standing.assign(nb, 0); Jcum.assign(nb, 0);
        for (int b = 0; b < nb; b++) for (int k : cells[b]) if (zb[k] > bare[k] + 1.0) { standing[b] = 1; break; }
        CK(cudaMalloc(&d_pc, npc * sizeof(int))); CK(cudaMemcpy(d_pc, pc.data(), npc * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_ring, (size_t)W * npc * sizeof(float))); CK(cudaMemset(d_ring, 0, (size_t)W * npc * sizeof(float)));
        CK(cudaMalloc(&d_J, npc * sizeof(double))); CK(cudaMemset(d_J, 0, npc * sizeof(double)));
        for (float **a : {&d_hb, &d_Fc, &d_m2}) { CK(cudaMalloc(a, npc * sizeof(float))); CK(cudaMemset(*a, 0, npc * sizeof(float))); }
        {   std::vector<int> ofs(nb + 1, 0), idx;
            for (int b = 0; b < nb; b++) { for (int p : per[b]) idx.push_back(p); ofs[b + 1] = (int)idx.size(); }
            CK(cudaMalloc(&d_ofs, (nb + 1) * sizeof(int))); CK(cudaMemcpy(d_ofs, ofs.data(), (nb + 1) * sizeof(int), cudaMemcpyHostToDevice));
            CK(cudaMalloc(&d_idx, std::max<size_t>(1, idx.size()) * sizeof(int))); CK(cudaMemcpy(d_idx, idx.data(), idx.size() * sizeof(int), cudaMemcpyHostToDevice)); }
        CK(cudaMalloc(&d_ringU, (size_t)W * npc * sizeof(float))); CK(cudaMemset(d_ringU, 0, (size_t)W * npc * sizeof(float)));
        for (float **a : {&d_hcc, &d_m2c}) { CK(cudaMalloc(a, npc * sizeof(float))); CK(cudaMemset(*a, 0, npc * sizeof(float))); }
        CK(cudaMalloc(&d_pk2, 4 * nb * sizeof(double))); CK(cudaMalloc(&d_pt2, nb * sizeof(double)));
        {   std::vector<double> z4(4 * nb, -1e300); CK(cudaMemcpy(d_pk2, z4.data(), 4 * nb * sizeof(double), cudaMemcpyHostToDevice)); }
        CK(cudaMemset(d_pt2, 0, nb * sizeof(double)));
        if (!hf.empty()) {   // building heights for the wind term
            std::ifstream f(hf);
            if (!f) fail("bldg_heightfile " + hf + " not found");
            std::vector<float> H(nb, 0.f); std::string ln;
            while (std::getline(f, ln)) {
                size_t c = ln.find('%'); if (c != std::string::npos) ln.erase(c);
                std::istringstream s(ln); int id; double h;
                if (!(s >> id >> h)) continue;
                if (id < 1 || id > nb) fail("bldg_heightfile: building " + std::to_string(id) + " not in bldgfile");
                H[id - 1] = (float)h; nH++;
            }
            CK(cudaMalloc(&d_H, nb * sizeof(float))); CK(cudaMemcpy(d_H, H.data(), nb * sizeof(float), cudaMemcpyHostToDevice));
        }
        CK(cudaMalloc(&d_st, nb)); CK(cudaMalloc(&d_pk, 3 * nb * sizeof(double))); CK(cudaMalloc(&d_pt, 3 * nb * sizeof(double)));
        CK(cudaMemset(d_pk, 0, 3 * nb * sizeof(double))); CK(cudaMemset(d_pt, 0, 3 * nb * sizeof(double)));
        {   std::vector<float> zg(nb, 0.f);   // building ground elevation: mean bare earth under its cells
            for (int b = 0; b < nb; b++) { double sz = 0; for (int c : cells[b]) sz += bare[c]; zg[b] = (float)(sz / std::max<size_t>(1, cells[b].size())); }
            CK(cudaMalloc(&d_zg, nb * sizeof(float))); CK(cudaMemcpy(d_zg, zg.data(), nb * sizeof(float), cudaMemcpyHostToDevice)); }
        bool resumed = false;
        if (P.hotstart) resumed = load();
        CK(cudaMemcpy(d_st, standing.data(), nb, cudaMemcpyHostToDevice));
        int ns0 = 0; for (char s : standing) ns0 += s;
        printf("buildings: %d (%d standing), %d perimeter cells, window %d samples x %.2f s, state %s; heights for the wind term %d\n",
               nb, ns0, npc, W, tint, resumed ? ("resumed from " + statefile + ", sample " + std::to_string(ns)).c_str() : "new", nH);
        fflush(stdout);
        tnext = tint; tnextcheck = tcheck;
    }

    // called after every time step
    void step(double t, const real *zs, const real *zb, const real *u, const real *v) {
        if (!on) return;
        while (t >= tnext - 1e-9) {
            k_bldg_sample<<<(npc + 255) / 256, 256>>>(npc, d_pc, zs, zb, u, v, d_ring, d_J, d_hb, d_Fc, d_m2, d_ringU, d_hcc, d_m2c,
                                                      W, ns, 0.5 * rho * g, tint, fpos);
            double ts = (ns + 1) * tint;   // storm time of this sample (s since the cold start; ns carries over)
            double tc = tnext - (W - 1) / 2 * tint, V = 0;   // model time of the window centre: wind at that time
            if (!wind_t.empty()) V = wind_t.size() > 1 ? lin_interp(wind_t, wind_s, tc) : wind_s[0];
            k_bldg_peaks<<<(nb + 127) / 128, 128>>>(nb, d_ofs, d_idx, d_st, d_hb, d_Fc, d_m2, d_pc, zs, zb, d_zg, d_pk, d_pt,
                                                    ts - (W - 1) / 2 * tint, ts, ns >= W - 1 ? 1 : 0, d_hcc, d_m2c, d_H, rho,
                                                    0.5 * rho * g, 0.5 * rhoa * wind_cf * V * V, d_pk2, d_pt2);
            ns++; tnext += tint;
        }
    }

    // check: add this interval's perimeter-maximum impulse, remove failed buildings in zb/zs (device), write state
    template <class D2HF, class H2DF>
    void check(double t, real *d_zs, real *d_zb, size_t N, double eps, D2HF d2h, H2DF h2d, bool quick = false) {
        if (!on) return;
        if (!quick) { nchecks++; tlast = t; }
        std::vector<double> J(npc); CK(cudaMemcpy(J.data(), d_J, npc * sizeof(double), cudaMemcpyDeviceToHost));
        if (tcheck <= 0) CK(cudaMemset(d_J, 0, npc * sizeof(double)));
        if (dump && !quick) {   // per perimeter cell impulse of this interval (verification)
            FILE *f = fopen(("bldg_cells_" + std::to_string(nchecks) + ".txt").c_str(), "w");
            for (int p = 0; p < npc; p++) fprintf(f, "%d %d %d %.6e\n", p, pc[p] % NX1_ + 1, pc[p] / NX1_ + 1, J[p]);
            fclose(f);
        }
        std::vector<int> gone;
        for (int b = 0; b < nb; b++) {
            if (!standing[b]) continue;
            double m = 0; for (int p : per[b]) m = std::max(m, J[p]);
            if (tcheck > 0) Jcum[b] = m; else Jcum[b] += m;
            if (Jcum[b] >= thr[b]) gone.push_back(b);
        }
        long ncell = 0;
        if (!gone.empty()) {
            std::vector<real> zs = d2h(d_zs, N), zb = d2h(d_zb, N);
            for (int b : gone) {
                double s = 0; int n = 0;
                for (int p : per[b]) { int k = pc[p]; if (zs[k] - zb[k] > eps) { s += zs[k]; n++; } }
                for (int k : cells[b]) {
                    zb[k] = (real)bare[k];
                    zs[k] = (real)(n ? std::max(s / n, bare[k]) : bare[k]);
                }
                standing[b] = 0; ncell += (long)cells[b].size();
            }
            h2d(d_zs, zs); h2d(d_zb, zb); changed = true;
            CK(cudaMemcpy(d_st, standing.data(), nb, cudaMemcpyHostToDevice));
        }
        if (!gone.empty()) {
            FILE *g = fopen("bldg_removed.txt", "a");
            for (int b : gone) fprintf(g, "%10.3f %6d %5d %5d %6.1f %10.4f %6zu\n", t, b + 1, ic[b], jc[b], thr[b] / 3.6e6, Jcum[b] / 3.6e6, cells[b].size());
            fclose(g);
        }
        nremoved_run += (long)gone.size();
        if (quick) {
            if (!gone.empty()) { int left = 0; for (char c : standing) left += c;
                printf("buildings: t = %.3f s, %zu removed (%ld cells), %d standing\n", t, gone.size(), ncell, left); fflush(stdout); }
            return;
        }
        int left = 0; for (char s : standing) left += s;
        printf("buildings: check at t = %.3f s, %zu removed here, %ld in this run so far, %d standing\n", t, gone.size(), nremoved_run, left);
        FILE *f;
        fflush(stdout);
        save();
        f = fopen("bldg_impulse.txt", "w");
        std::vector<double> pk(3 * nb), pt(3 * nb);
        CK(cudaMemcpy(pk.data(), d_pk, 3 * nb * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pt.data(), d_pt, 3 * nb * sizeof(double), cudaMemcpyDeviceToHost));
        fprintf(f, "%% id i j threshold_kNhr_m impulse_kNhr_m standing ncells | peaks while standing, kN/m, and storm time s:"
                   " wave_F t drag_Cd2 t dhydrostatic t | total lateral load at one instant, kN/m: F_total t wave+drag quasistatic wind\n");
        std::vector<double> pk2(4 * nb), pt2(nb);
        CK(cudaMemcpy(pk2.data(), d_pk2, 4 * nb * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pt2.data(), d_pt2, nb * sizeof(double), cudaMemcpyDeviceToHost));
        for (int b = 0; b < nb; b++) {
            const double *q = &pk2[4 * b]; bool ok = q[0] > -1e299;   // never evaluated: 0
            fprintf(f, "%6d %5d %5d %6.1f %12.5f %d %6zu %9.3f %9.1f %9.3f %9.1f %9.3f %9.1f %9.3f %9.1f %9.3f %9.3f %9.3f\n",
                    b + 1, ic[b], jc[b], thr[b] / 3.6e6, Jcum[b] / 3.6e6, standing[b], cells[b].size(),
                    pk[3 * b] / 1e3, pt[3 * b], rho * pk[3 * b + 1] / 1e3, pt[3 * b + 1],
                    0.5 * rho * g * pk[3 * b + 2] / 1e3, pt[3 * b + 2],
                    ok ? q[0] / 1e3 : 0.0, pt2[b], ok ? q[1] / 1e3 : 0.0, ok ? q[2] / 1e3 : 0.0, ok ? q[3] / 1e3 : 0.0);
        }
        fclose(f);
    }

    void save() {
        std::vector<float> ring((size_t)W * npc); CK(cudaMemcpy(ring.data(), d_ring, ring.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::string tmp = statefile + ".tmp";
        std::ofstream o(tmp, std::ios::binary);
        int hdr[3] = {nb, npc, W}; o.write((char *)hdr, sizeof(hdr));
        o.write((char *)&ns, sizeof(ns));
        o.write((char *)Jcum.data(), nb * sizeof(double)); o.write(standing.data(), nb);
        o.write((char *)ring.data(), ring.size() * sizeof(float));
        std::vector<double> J(npc); CK(cudaMemcpy(J.data(), d_J, npc * sizeof(double), cudaMemcpyDeviceToHost));
        o.write((char *)J.data(), npc * sizeof(double));
        std::vector<double> pk(3 * nb), pt(3 * nb);
        CK(cudaMemcpy(pk.data(), d_pk, 3 * nb * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pt.data(), d_pt, 3 * nb * sizeof(double), cudaMemcpyDeviceToHost));
        o.write((char *)pk.data(), 3 * nb * sizeof(double)); o.write((char *)pt.data(), 3 * nb * sizeof(double));
        {   std::vector<double> pk2(4 * nb), pt2(nb); std::vector<float> rU((size_t)W * npc);
            CK(cudaMemcpy(pk2.data(), d_pk2, 4 * nb * sizeof(double), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(pt2.data(), d_pt2, nb * sizeof(double), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(rU.data(), d_ringU, rU.size() * sizeof(float), cudaMemcpyDeviceToHost));
            o.write((char *)pk2.data(), 4 * nb * sizeof(double)); o.write((char *)pt2.data(), nb * sizeof(double));
            o.write((char *)rU.data(), rU.size() * sizeof(float)); }
        o.close();
        std::rename(tmp.c_str(), statefile.c_str());
    }
    bool load() {
        std::ifstream f(statefile, std::ios::binary);
        if (!f) return false;
        int hdr[3]; f.read((char *)hdr, sizeof(hdr));
        if (hdr[0] != nb || hdr[1] != npc || hdr[2] != W) fail(statefile + ": building set or window differs from this deck");
        f.read((char *)&ns, sizeof(ns));
        std::vector<char> st(nb);
        f.read((char *)Jcum.data(), nb * sizeof(double)); f.read(st.data(), nb);
        std::vector<float> ring((size_t)W * npc); f.read((char *)ring.data(), ring.size() * sizeof(float));
        std::vector<double> J(npc); f.read((char *)J.data(), npc * sizeof(double));
        std::vector<double> pk(3 * nb), pt(3 * nb);
        f.read((char *)pk.data(), 3 * nb * sizeof(double)); f.read((char *)pt.data(), 3 * nb * sizeof(double));
        if (!f) fail(statefile + ": truncated");
        {   std::vector<double> pk2(4 * nb), pt2(nb); std::vector<float> rU((size_t)W * npc);
            f.read((char *)pk2.data(), 4 * nb * sizeof(double)); f.read((char *)pt2.data(), nb * sizeof(double));
            f.read((char *)rU.data(), rU.size() * sizeof(float));
            if (f) {
                CK(cudaMemcpy(d_pk2, pk2.data(), 4 * nb * sizeof(double), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d_pt2, pt2.data(), nb * sizeof(double), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d_ringU, rU.data(), rU.size() * sizeof(float), cudaMemcpyHostToDevice));
            } else printf("buildings: %s has no total-load state (older run): total load starts here\n", statefile.c_str());
        }
        CK(cudaMemcpy(d_pk, pk.data(), 3 * nb * sizeof(double), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pt, pt.data(), 3 * nb * sizeof(double), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_J, J.data(), npc * sizeof(double), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ring, ring.data(), ring.size() * sizeof(float), cudaMemcpyHostToDevice));
        // a building removed earlier stays removed (the hotstart bed already has it gone)
        for (int b = 0; b < nb; b++) if (!st[b] && standing[b]) fail(statefile + ": building " + std::to_string(b + 1) + " removed in the state but present in the hotstart bed");
        return true;
    }
};
