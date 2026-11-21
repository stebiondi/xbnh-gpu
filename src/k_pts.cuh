// k_pts.cuh: point gauges for buildings that are not in the bed (elevated on piles/columns: open underneath, so
// they do not block the flow and are not modeled as raised cells). Each gauge is one cell; nothing is removed,
// the gauge only records the forcing over the whole storm, for damage analysis after the run.
//
// ptsfile, one line per gauge (% starts a comment):   id  i  j  z_beam  z_floor      (i, j: 1-based cell; m, NAVD88)
//   z_beam : lowest horizontal structural member (underside of the floor system): uplift starts when zs > z_beam
//   z_floor: first-floor elevation; water above it is water inside the building
//   either may be nan when unknown (those columns stay empty)
//
// Every model step (phase resolved, so crests are included; no wave height is added on top):
//   zs, h = zs - zb, U = |cell-centre velocity| (only where h >= pts_hmin), Fr = U / sqrt(g h), h U^2, h U
//   e_b = zs - z_beam (wet cells only, h > 1 mm): peak, integral of max(e_b, 0) dt (m s; times rho g it is the uplift pressure impulse per
//         unit plan area), time above, first time above; the same for e_f = zs - z_floor
// Every pts_tint s (0.5): the trailing pts_tavg s (120) mean water level; its peak = surge + setup without crests.
// Times are storm seconds (since the cold start, carried over between segments through pts_state).
// Output pts_peaks.txt (rewritten at every hotstart write and at the end), state pts_state (binary).
#pragma once
#include <fstream>
#include <sstream>
#include <cmath>

enum { PQ_ZS, PQ_ZS_T, PQ_ZM, PQ_ZM_T, PQ_H, PQ_H_T, PQ_U, PQ_U_T, PQ_FR, PQ_FR_T, PQ_HU2, PQ_HU2_T, PQ_HU, PQ_HU_T,
       PQ_EB, PQ_EB_T, PQ_EB_INT, PQ_EB_DUR, PQ_EB_FIRST, PQ_EF, PQ_EF_T, PQ_EF_INT, PQ_EF_DUR, PQ_EF_FIRST, PQ_N };

__device__ inline void pts_mx(double *q, int a, double v, double ts) { if (v > q[a]) { q[a] = v; q[a + 1] = ts; } }

__global__ void k_pts_step(int np, const int *pk, const float *zl, const real *zs, const real *zb, const real *u,
                           const real *v, double *Q, double ts, double dt, double g, double hmin) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= np) return;
    int k = pk[p];
    double *q = Q + (size_t)p * PQ_N;
    double z = zs[k], h = fmax(z - (double)zb[k], 0.0);
    pts_mx(q, PQ_ZS, z, ts); pts_mx(q, PQ_H, h, ts);
    if (h >= hmin) {
        double U = sqrt((double)u[k] * u[k] + (double)v[k] * v[k]);
        pts_mx(q, PQ_U, U, ts); pts_mx(q, PQ_FR, U / sqrt(g * h), ts);
        pts_mx(q, PQ_HU2, h * U * U, ts); pts_mx(q, PQ_HU, h * U, ts);
    }
    if (h <= 1e-3) return;   // dry: zs is the bed there, not water
    for (int s = 0; s < 2; s++) {
        double zr = zl[2 * p + s];
        if (isnan(zr)) continue;
        int a = s ? PQ_EF : PQ_EB;
        double e = z - zr;
        pts_mx(q, a, e, ts);
        if (e > 0) { q[a + 2] += e * dt; q[a + 3] += dt; if (q[a + 4] < 0) q[a + 4] = ts; }
    }
}

// trailing mean of zs over W samples; its peak is stamped with the time of the window's last sample
__global__ void k_pts_sample(int np, const int *pk, const real *zs, float *ring, double *Q, int W, long ns, double ts) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= np) return;
    ring[(size_t)(ns % W) * np + p] = (float)zs[pk[p]];
    if (ns < W - 1) return;
    double s = 0;
    for (int w = 0; w < W; w++) s += ring[(size_t)w * np + p];
    pts_mx(Q + (size_t)p * PQ_N, PQ_ZM, s / W, ts);
}

struct Gauges {
    bool on = false;
    int np = 0, W = 0; long ns = 0;
    double tint = 0.5, tavg = 120, hmin = 0.05, g = 9.81, t0 = 0, tnext = 0, tlast = -1;
    std::vector<int> id, ii, jj, kk; std::vector<float> zl;
    int *d_k = nullptr; float *d_zl = nullptr, *d_ring = nullptr; double *d_Q = nullptr;
    std::string statefile;

    template <class IDX>
    void init(ParFile &F, const Params &P, int NX1, int NY1, IDX I) {
        std::string pf = F.name("ptsfile");
        if (pf.empty()) return;
        on = true;
        tint = F.dbl("pts_tint", 0.5); tavg = F.dbl("pts_tavg", 120.0); hmin = F.dbl("pts_hmin", 0.05);
        statefile = F.name("pts_state", "pts_state.bin"); g = P.g;
        W = (int)lround(tavg / tint) + 1;
        std::ifstream f(pf);
        if (!f) fail("ptsfile " + pf + " not found");
        std::string ln;
        while (std::getline(f, ln)) {
            size_t c = ln.find('%'); if (c != std::string::npos) ln.erase(c);
            std::istringstream s(ln); int a, i, j; std::string zb_, zf_;
            if (!(s >> a >> i >> j >> zb_ >> zf_)) continue;
            if (i < 1 || i > NX1 || j < 1 || j > NY1) fail("ptsfile: gauge " + std::to_string(a) + " outside the grid");
            id.push_back(a); ii.push_back(i); jj.push_back(j); kk.push_back((int)I(i, j));
            zl.push_back(std::strtof(zb_.c_str(), nullptr)); zl.push_back(std::strtof(zf_.c_str(), nullptr));   // "nan" -> NaN
        }
        np = (int)id.size();
        if (!np) fail("ptsfile " + pf + ": no gauges");
        CK(cudaMalloc(&d_k, np * sizeof(int))); CK(cudaMemcpy(d_k, kk.data(), np * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_zl, 2 * np * sizeof(float))); CK(cudaMemcpy(d_zl, zl.data(), 2 * np * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_ring, (size_t)W * np * sizeof(float))); CK(cudaMemset(d_ring, 0, (size_t)W * np * sizeof(float)));
        std::vector<double> Q((size_t)np * PQ_N, 0.0);
        for (int p = 0; p < np; p++) {
            double *q = &Q[(size_t)p * PQ_N];
            for (int a : {PQ_ZS, PQ_ZM, PQ_EB, PQ_EF}) q[a] = -1e30;
            q[PQ_EB_FIRST] = q[PQ_EF_FIRST] = -1;
        }
        CK(cudaMalloc(&d_Q, Q.size() * sizeof(double))); CK(cudaMemcpy(d_Q, Q.data(), Q.size() * sizeof(double), cudaMemcpyHostToDevice));
        bool resumed = P.hotstart && load();
        printf("gauges: %d points, mean window %d samples x %.2f s, state %s\n", np, W, tint,
               resumed ? ("resumed from " + statefile + ", storm time " + std::to_string(t0) + " s").c_str() : "new");
        fflush(stdout);
        tnext = tint;
    }

    // after every time step (t: model time of this run, dt: the step just taken)
    void step(double t, double dt, const real *zs, const real *zb, const real *u, const real *v) {
        if (!on) return;
        k_pts_step<<<(np + 127) / 128, 128>>>(np, d_k, d_zl, zs, zb, u, v, d_Q, t0 + t, dt, g, hmin);
        while (t >= tnext - 1e-9) {
            k_pts_sample<<<(np + 127) / 128, 128>>>(np, d_k, zs, d_ring, d_Q, W, ns, t0 + tnext);
            ns++; tnext += tint;
        }
    }

    // table and state, at every hotstart write and at the end of the run
    void write(double t) {
        if (!on) return;
        tlast = t;
        std::vector<double> Q((size_t)np * PQ_N);
        CK(cudaMemcpy(Q.data(), d_Q, Q.size() * sizeof(double), cudaMemcpyDeviceToHost));
        FILE *f = fopen("pts_peaks.txt", "w");
        fprintf(f, "%% storm time %.1f s. Peaks with their storm time (s); U, Fr, hU2, hU only where h >= %.3f m.\n", t0 + t, hmin);
        fprintf(f, "%% id i j z_beam z_floor | zs_max t zsmean_max t h_max t U_max t Fr_max t hU2_max t hU_max t"
                   " | beam: exc_max t int_m_s dur_s first_s | floor: exc_max t int_m_s dur_s first_s\n");
        auto v = [](double x) { return x < -1e29 ? NAN : x; };
        for (int p = 0; p < np; p++) {
            const double *q = &Q[(size_t)p * PQ_N];
            fprintf(f, "%6d %5d %5d %7.3f %7.3f", id[p], ii[p], jj[p], zl[2 * p], zl[2 * p + 1]);
            for (int a = PQ_ZS; a < PQ_EB; a += 2) fprintf(f, " %8.4f %8.1f", v(q[a]), q[a + 1]);
            for (int a : {PQ_EB, PQ_EF})
                fprintf(f, " %8.4f %8.1f %10.3f %8.1f %8.1f", v(q[a]), q[a + 1], q[a + 2], q[a + 3], q[a + 4]);
            fprintf(f, "\n");
        }
        fclose(f);
        std::vector<float> ring((size_t)W * np);
        CK(cudaMemcpy(ring.data(), d_ring, ring.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::string tmp = statefile + ".tmp";
        std::ofstream o(tmp, std::ios::binary);
        int hdr[3] = {np, W, PQ_N}; double ts = t0 + t;
        o.write((char *)hdr, sizeof(hdr)); o.write((char *)&ts, sizeof(ts)); o.write((char *)&ns, sizeof(ns));
        o.write((char *)id.data(), np * sizeof(int));
        o.write((char *)Q.data(), Q.size() * sizeof(double)); o.write((char *)ring.data(), ring.size() * sizeof(float));
        o.close();
        if (!o) fail("cannot write " + tmp);
        std::rename(tmp.c_str(), statefile.c_str());
    }

    bool load() {
        std::ifstream f(statefile, std::ios::binary);
        if (!f) return false;
        int hdr[3]; f.read((char *)hdr, sizeof(hdr));
        if (hdr[0] != np || hdr[1] != W || hdr[2] != PQ_N) fail(statefile + ": gauge set or window differs from this deck");
        f.read((char *)&t0, sizeof(t0)); f.read((char *)&ns, sizeof(ns));
        std::vector<int> id2(np); f.read((char *)id2.data(), np * sizeof(int));
        if (id2 != id) fail(statefile + ": gauge ids differ from ptsfile");
        std::vector<double> Q((size_t)np * PQ_N); f.read((char *)Q.data(), Q.size() * sizeof(double));
        std::vector<float> ring((size_t)W * np); f.read((char *)ring.data(), ring.size() * sizeof(float));
        if (!f) fail(statefile + ": truncated");
        CK(cudaMemcpy(d_Q, Q.data(), Q.size() * sizeof(double), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ring, ring.data(), ring.size() * sizeof(float), cudaMemcpyHostToDevice));
        return true;
    }
};
