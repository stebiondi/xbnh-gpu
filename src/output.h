// output.h: timestep.F90 output times (timestep_init, outputtimes_update), ncoutput.F90 (fortran and netCDF
// global, point, runup-gauge, mean and drifter output; dims.dat, xy.dat) and varianceupdate.F90 (time means).
#pragma once
#include "vartable.h"
#ifdef XB_NETCDF
#include <netcdf.h>
#define NC(x) do { int e_ = (x); if (e_ != NC_NOERR) { fprintf(stderr, "netCDF error %s at %s:%d\n", nc_strerror(e_), __FILE__, __LINE__); exit(1);} } while (0)
#endif

// ---- device helpers for output
__global__ void k_copy_out(const real *a, double *o, size_t n) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) o[k] = a[k];
}
__global__ void k_copy_out_int(const int *a, double *o, size_t n) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) o[k] = a[k];
}
// point output: the values at the point cells, so that only these are copied to the host
__global__ void k_gather_points(const double *f, const int *idx, int np, double *o) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < np) o[q] = f[idx[q]];
}
// derived output fields (code: 0 u, 1 v, 2 ue, 3 ve, 4 zs1, 5 qmag, 6 vmagu, 7 vmagv, 8 zswet, 9 umean, 10 vmean)
// with rotation to world axes for the vector components when rotate = 1 (postprocess.F90 gridrotate)
__device__ __forceinline__ real ucen(int i, int j) { return d.u[ID(i, j)]; }
__device__ __forceinline__ real vcen(int i, int j) { return d.v[ID(i, j)]; }
__device__ __forceinline__ real uecen(int i, int j) {   // ue(1) = ueu(1), ue(2:nx) = mean, ue(nx+1) never set (0)
    if (i == c.NX1) return 0;
    return i == 1 ? d.ueu[ID(1, j)] : (real)0.5 * (d.ueu[ID(i - 1, j)] + d.ueu[ID(i, j)]);
}
__device__ __forceinline__ real vecen(int i, int j) { return d.ve[ID(i, j)]; }   // k_ve_centre
__global__ void k_out_derived(int code, int rot, double *o) {
    IJ
    size_t k = ID(i, j);
    double v = 0, a = alfaz(i, j);
    switch (code) {
    case 0: v = rot ? ucen(i, j) * cos(a) - vcen(i, j) * sin(a) : ucen(i, j); break;
    case 1: v = rot ? ucen(i, j) * sin(a) + vcen(i, j) * cos(a) : vcen(i, j); break;
    case 2: v = rot ? uecen(i, j) * cos(a) - vecen(i, j) * sin(a) : uecen(i, j); break;
    case 3: v = rot ? uecen(i, j) * sin(a) + vecen(i, j) * cos(a) : vecen(i, j); break;
    case 4: v = d.zs[k] - d.zs0[k]; break;
    case 5: v = sqrt(d.qx[k] * d.qx[k] + d.qy[k] * d.qy[k]); break;
    case 6: v = sqrt(d.uu[k] * d.uu[k] + d.vu[k] * d.vu[k]); break;
    case 7: v = sqrt(d.uv[k] * d.uv[k] + d.vv[k] * d.vv[k]); break;
    case 8: v = d.wetz[k] == 1 ? d.zs[k] : DFILL; break;
    case 9: v = rot ? d.umean[k] * cos(a) - d.vmean[k] * sin(a) : d.umean[k]; break;
    case 10: v = rot ? d.umean[k] * sin(a) + d.vmean[k] * cos(a) : d.vmean[k]; break;
    }
    o[k] = v;
}
// postprocess.F90 gridrotate for the sediment fluxes (rotate = 1): fluxes averaged to cell centres, then rotated;
// code 0 Susg, 1 Svsg, 2 Subg, 3 Svbg (one fraction), 4 Sutot, 5 Svtot, 6 cctot; gw: 7 gwu, 8 gwv
__global__ void k_out_sedvec(int code, int jg, const real *Su, const real *Sv, const real *Bu, const real *Bv, const real *cc,
                             int ngd, double *o) {
    IJ
    size_t k = ID(i, j), NNo = (size_t)c.NX1 * c.NY1;
    int nx = c.nx, ny = c.ny;
    double a = alfaz(i, j);
    auto fu = [&](const real *A, int g, int ii) { return (double)A[(size_t)g * NNo + ID(ii, j)]; };
    auto fv = [&](const real *A, int g, int jj) { return (double)A[(size_t)g * NNo + ID(i, jj)]; };
    auto cu = [&](const real *A, int g) { return (i >= 2 && i <= nx) ? 0.5 * (fu(A, g, i - 1) + fu(A, g, i)) : fu(A, g, i); };
    auto cvv = [&](const real *A, int g) { return (!c.oned && j >= 2 && j <= ny) ? 0.5 * (fv(A, g, j - 1) + fv(A, g, j)) : fv(A, g, j); };
    double v = 0;
    if (code <= 3) {
        const real *U = code <= 1 ? Su : Bu, *V = code <= 1 ? Sv : Bv;
        double uc = cu(U, jg), vc = cvv(V, jg);
        v = (code % 2 == 0) ? uc * cos(a) - vc * sin(a) : uc * sin(a) + vc * cos(a);
    } else if (code <= 5) {
        auto tot = [&](const real *A, const real *B, size_t q) { double sa = 0, sb = 0; for (int g = 0; g < ngd; g++) { sa += A[(size_t)g * NNo + q]; sb += B[(size_t)g * NNo + q]; } return sa + sb; };
        double ut = (i >= 2 && i <= nx) ? 0.5 * (tot(Bu, Su, ID(i - 1, j)) + tot(Bu, Su, k)) : tot(Bu, Su, k);
        double vt = (!c.oned && j >= 2 && j <= ny) ? 0.5 * (tot(Bv, Sv, ID(i, j - 1)) + tot(Bv, Sv, k)) : tot(Bv, Sv, k);
        v = code == 4 ? ut * cos(a) - vt * sin(a) : ut * sin(a) + vt * cos(a);
    } else if (code == 6) {
        double sc = 0; for (int g = 0; g < ngd; g++) sc += cc[(size_t)g * NNo + k]; v = sc;
    } else {
        double gu = d.gwu[k], gv = d.gwv[k];
        v = code == 7 ? gu * cos(a) - gv * sin(a) : gu * sin(a) + gv * cos(a);
    }
    o[(size_t)jg * NNo + k] = v;
}
__global__ void k_remdry(const int *wetz, double *o, size_t n, double fill) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n && wetz[k] == 0) o[k] = fill;
}
// varianceupdate.F90 makeaverage, rank 2 (m: mean, cr: variancecrossterm, sq: variancesquareterm, va, mn, mx)
__global__ void k_mean_update(const double *x, double *m, double *cr, double *sq, double *va, double *mn, double *mx,
                              double mult, size_t n, int zswet) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const double numeps = 2.220446049250313e-16, dfill = -3.4028234663852886e38;
    double om = m[k];
    if (om < numeps && om >= 0) om = numeps;
    if (om > -numeps && om < 0) om = -numeps;
    double t = x[k];
    if (zswet && !(t > dfill + 1)) return;
    double nm = m[k] + mult * t;
    m[k] = nm;
    cr[k] = cr[k] / om * nm + mult * 2.0 * t * nm;
    sq[k] = sq[k] + mult * t * t;
    va[k] = sq[k] - cr[k] + nm * nm;
    mx[k] = fmax(mx[k], t);
    mn[k] = fmin(mn[k], t);
}

struct Output {
    Params *P = nullptr; const Grid *G = nullptr; const Dev *H = nullptr;
    size_t N = 0; int NX1 = 0, NY1 = 0;
    std::vector<double> tpg, tpp, tpm, tph;
    int itg = 0, itp = 0, itm = 0, ith = 0;          // timepars counters
    int nitg = 0, nitp = 0, nitm = 0;                // local counters of ncoutput (fortran records written)
    bool outputg = false, outputp = false, outputm = false, outputh = false, output = false;
    double tnext = 0;
    double *dout = nullptr;                          // device buffer, N doubles
    int *d_pidx = nullptr; double *d_pval = nullptr; // point output: cell indices and gathered values
    std::vector<double> hbuf;
    struct Field { std::string name; const VarInfo *vi; int nlay; };
    std::vector<Field> gf, pf, mf;
    std::vector<FILE *> gfiles, pfiles, mfiles[4], dfiles;
    struct Pt { int i, j, type; std::string id; double xw, yw; };
    std::vector<Pt> pts;
    // means (device)
    std::vector<double *> mm, mcr, msq, mva, mmn, mmx;
    int ngd = 1, nd = 3, tidelen = 2, windlen = 1, nveg = 1, nship = 0;
    const SedDev *sed = nullptr;
    const IntraDev *intra = nullptr;
    const Veg *veg = nullptr;
    std::vector<std::pair<std::string, std::string>> parfile_kv;
    std::map<std::string, std::vector<double>> shipv;   // per-ship global output (shipxCG ... shipMz), set by the caller
#ifdef XB_NETCDF
    int ncid = -1, ncdouble = 1;
    std::vector<int> gvid, pvid, mvid[4];
    int gtvid = -1, ptvid = -1, mtvid = -1, xpvid = -1, ypvid = -1, dtvid = -1, nitd = 0;
    std::vector<int> dvid;
#endif
    bool dofortran() const { return P->outputformat == OUT_FORTRAN || P->outputformat == OUT_DEBUG; }
    bool donetcdf() const { return P->outputformat == OUT_NETCDF || P->outputformat == OUT_DEBUG; }

    static const VarInfo *lookup(const std::string &n) {
        for (int q = 0; q < XB_NVARS; q++) if (n == XB_VARS[q].name) return &XB_VARS[q];
        return nullptr;
    }
    int layers(const VarInfo *v) const {
        if (v->rank <= 2) return 1;
        std::string s = v->shape; size_t c2 = s.find(',', s.find(',') + 1);
        std::string l = s.substr(c2 + 1);
        if (l == "ngd") return ngd;
        if (l == "nd") return nd;
        if (l == "ntheta" || l == "ntheta_s") return 1;
        if (l == "nd,ngd") return nd * ngd;
        if (l == "nsecvegmax") return nveg;
        if (l == "kmax") return std::max(P->kmax, 1);
        return -1;
    }
    Field field(const std::string &name, const char *what) const {
        const VarInfo *v = lookup(name);
        if (!v) fail(std::string("unknown ") + what + " output variable '" + name + "'");
        if (v->rank == 1 && strcmp(v->shape, "nship") == 0 && strcmp(what, "global") == 0) {   // shipxCG ... shipMz
            if (nship == 0) fail("global output variable '" + name + "' needs ships = 1");
            return {name, v, 0};
        }
        if (v->rank < 2 || (v->shape[0] != 'n' || strncmp(v->shape, "nx+1,ny+1", 9) != 0))
            fail(std::string(what) + " output variable '" + name + "' is not a grid field; not supported");
        int nl = layers(v);
        if (nl < 1) fail(std::string(what) + " output variable '" + name + "' has an unsupported shape " + v->shape);
        return {name, v, nl};
    }

    // timestep_init
    void init_times() {
        Params &p = *P;
        auto seq = [](double t0, double t1, double dt) { std::vector<double> v; int n = (int)floor((t1 - t0) / dt) + 1;
                                                         for (int q = 0; q < n; q++) v.push_back(t0 + dt * q); return v; };
        // time lists from files are given in morphological time (timestep_init divides by morfac)
        auto rtl = [&](const std::string &f) { auto v = read_time_list(f); for (auto &x : v) x /= std::max(p.morfac, 1.0); return v; };
        if (p.nglobalvar != 0) tpg = p.tsglobal.empty() ? seq(p.tstart, p.tstop, p.tintg) : rtl(p.tsglobal);
        if (p.npoints + p.nrugauge + p.ndrifter > 0) tpp = p.tspoints.empty() ? seq(p.tstart, p.tstop, p.tintp) : rtl(p.tspoints);
        if (p.nmeanvar > 0) {
            tpm = p.tsmean.empty() ? seq(p.tstart, p.tstop, p.tintm) : rtl(p.tsmean);
            if (tpm.size() <= 1) fail("tintm is larger than output simulation time");
        }
        if (p.writehotstart == 1) {
            if (!p.tshotstart.empty()) tph = rtl(p.tshotstart);
            else { int n = (int)floor(p.tstop / p.tinth); if (n <= 0) fail("tinth is larger than output simulation time");
                   for (int q = 1; q <= n; q++) tph.push_back(p.tinth * q); }
        }
    }
    // outputtimes_update
    void update_times(double t, double tstop) {
        auto hit = [&](const std::vector<double> &v, int it) { return !v.empty() && fabs(v[std::min(it + 1, (int)v.size()) - 1] - t) <= 0.0000001; };
        outputg = hit(tpg, itg); outputp = hit(tpp, itp); outputm = hit(tpm, itm); outputh = hit(tph, ith);
        output = outputg || outputp || outputm || outputh;
        if (outputg) itg++;
        if (outputp) itp++;
        if (outputm) itm++;
        if (outputh) ith++;
        double tn = 1e300;
        for (auto *v : {&tpg, &tpp, &tpm, &tph}) for (double x : *v) if (x > t + 0.0000001) tn = std::min(tn, x);
        tnext = std::min(tn, tstop);
    }

    void init(Params &p, const Grid &g, const Dev &h) {
        P = &p; G = &g; H = &h; NX1 = g.NX1; NY1 = g.NY1; N = (size_t)NX1 * NY1;
        size_t lmax = std::max({(size_t)nveg, (size_t)(ngd * nd), (size_t)std::max(p.kmax, 1)});
        CK(cudaMalloc(&dout, N * sizeof(double) * lmax));
        hbuf.resize(N * lmax);
        for (auto &n : p.globalvars) gf.push_back(field(n, "global"));
        for (auto &n : p.pointvars) pf.push_back(field(n, "point"));
        for (auto &n : p.meanvars) { mf.push_back(field(n, "mean")); if (mf.back().nlay != 1 || mf.back().vi->type != 'r') fail("mean output supports real 2D fields only: " + n); }
        // snappointstogrid: nearest cell centre (first minimum in Fortran order); runup gauges use the row only
        for (auto &q : p.points) {
            double best = 1e300; int bi = 1, bj = 1;
            for (int j = 1; j <= NY1; j++) for (int i = 1; i <= NX1; i++) {
                size_t k = g.I(i, j);
                double dd = sqrt((q.x - g.xz[k]) * (q.x - g.xz[k]) + (q.y - g.yz[k]) * (q.y - g.yz[k]));
                if (dd < best) { best = dd; bi = i; bj = j; }
            }
            pts.push_back({q.type == 1 ? 1 : bi, bj, q.type, q.id, q.x, q.y});
            if (q.type == 0) printf("output point %s (%.2f, %.2f) -> grid cell (%d, %d), %.2f m\n", q.id.c_str(), q.x, q.y, bi, bj, best);
            else printf("runup gauge %s at grid line iy = %d\n", q.id.c_str(), bj);
        }
        for (size_t q = 0; q < mf.size(); q++) {
            double *a[6];
            for (int r = 0; r < 6; r++) CK(cudaMalloc(&a[r], N * sizeof(double)));
            mm.push_back(a[0]); mcr.push_back(a[1]); msq.push_back(a[2]); mva.push_back(a[3]); mmn.push_back(a[4]); mmx.push_back(a[5]);
        }
        clear_means();
        if (dofortran()) init_fortran();
#ifdef XB_NETCDF
        if (donetcdf()) init_netcdf();
#endif
    }
    void clear_means() {
        std::vector<double> z(N, 0), lo(N, 3.4028234663852886e38), hi(N, -3.4028234663852886e38);
        for (size_t q = 0; q < mf.size(); q++) {
            for (double *a : {mm[q], mcr[q], msq[q], mva[q]}) CK(cudaMemcpy(a, z.data(), N * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(mmn[q], lo.data(), N * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(mmx[q], hi.data(), N * 8, cudaMemcpyHostToDevice));
        }
    }
    void init_fortran() {
        FILE *f = fopen("xy.dat", "wb");
        fwrite(G->xz.data(), 8, N, f); fwrite(G->yz.data(), 8, N, f); fwrite(G->x.data(), 8, N, f); fwrite(G->y.data(), 8, N, f);
        fclose(f);
        for (auto &q : gf) { FILE *g = fopen((q.name + ".dat").c_str(), "wb"); if (!g) fail("cannot write " + q.name + ".dat"); gfiles.push_back(g); }
        for (auto &q : pts) { FILE *g = fopen((q.id + ".dat").c_str(), "wb"); if (!g) fail("cannot write " + q.id + ".dat"); pfiles.push_back(g); }
        if (P->npoints > 0) { FILE *g = fopen("pointvars.idx", "w"); for (auto &q : pf) fprintf(g, " %s\n", q.name.c_str()); fclose(g); }
        const char *suf[4] = {"_mean", "_var", "_min", "_max"};
        for (auto &q : mf) for (int r = 0; r < 4; r++) mfiles[r].push_back(fopen((q.name + suf[r] + ".dat").c_str(), "wb"));
        for (int q = 1; q <= P->ndrifter; q++) { char b[32]; snprintf(b, sizeof b, "drifter%03d.dat", q); dfiles.push_back(fopen(b, "wb")); }
    }
    // fill dout with variable `name` (layers stacked); returns false when the variable is not computed (zeros)
    bool produce(const Field &f, int uvzero) {
        const std::string &n = f.name;
        size_t M = N * f.nlay; unsigned nb = (unsigned)((M + 255) / 256);
        int rot = P->rotate;
        static const std::map<std::string, int> derived = {{"u", 0}, {"v", 1}, {"ue", 2}, {"ve", 3}, {"zs1", 4}, {"qmag", 5},
            {"vmagu", 6}, {"vmagv", 7}, {"zswet", 8}, {"umean", 9}, {"vmean", 10}};
        auto it = derived.find(n);
        if (it != derived.end()) {
            if (uvzero && it->second <= 3) CK(cudaMemset(dout, 0, N * 8));
            else k_out_derived<<<L.G, L.B>>>(it->second, rot && it->second != 4, dout);
            return true;
        }
        const Dev &h = *H;
        {   // rotated vectors (postprocess.F90 gridrotate)
            static const std::map<std::string, int> vec = {{"Susg", 0}, {"Svsg", 1}, {"Subg", 2}, {"Svbg", 3}, {"Sutot", 4},
                                                           {"Svtot", 5}, {"cctot", 6}, {"gwu", 7}, {"gwv", 8}};
            auto iv = vec.find(n);
            if (iv != vec.end() && rot && (iv->second <= 6 ? sed != nullptr : h.gwu != nullptr)) {
                int cd = iv->second, nl = cd <= 3 ? f.nlay : 1;
                for (int g = 0; g < nl; g++)
                    k_out_sedvec<<<L.G, L.B>>>(cd, g, sed ? sed->Susg : nullptr, sed ? sed->Svsg : nullptr, sed ? sed->Subg : nullptr,
                                               sed ? sed->Svbg : nullptr, sed ? sed->ccg : nullptr, ngd, dout);
                return true;
            }
            if (iv != vec.end() && iv->second >= 4 && iv->second <= 6) { CK(cudaMemset(dout, 0, M * 8)); return true; }   // not set without rotation
        }
        if (sed) {
            std::map<std::string, const real *> SR = {
                {"ccg", sed->ccg}, {"ceqsg", sed->ceqsg}, {"ceqbg", sed->ceqbg}, {"Tsg", sed->Tsg}, {"Susg", sed->Susg},
                {"Svsg", sed->Svsg}, {"Subg", sed->Subg}, {"Svbg", sed->Svbg}, {"ero", sed->ero}, {"depo_ex", sed->depo_ex},
                {"pbbed", sed->pbbed}, {"dzbed", sed->dzbed}, {"z0bed", sed->z0bed}, {"structdepth", sed->structdepth},
                {"D50top", sed->D50top}, {"D90top", sed->D90top}, {"sedero", sed->sedero}, {"dzbdt", sed->dzbdt}, {"dzav", sed->dzav},
                {"dzbdx", sed->dzbdx}, {"dzbdy", sed->dzbdy}, {"kturb", sed->kturb}, {"rolthick", sed->rolthick}, {"kb", sed->kb},
                {"Dc", sed->Dc}, {"vmag", sed->vmag}, {"ureps", sed->ureps}, {"urepb", sed->urepb}, {"vreps", sed->vreps},
                {"vrepb", sed->vrepb}, {"dcsdx", sed->dcsdx}, {"dcsdy", sed->dcsdy}, {"hold", sed->hold}, {"vmagu", sed->vmagu},
                {"vmagv", sed->vmagv}, {"ccz", sed->ccz}, {"refA", sed->refA}, {"ca", sed->ca}};
            if (intra) { SR["taubxsed"] = intra->taubxsed; SR["taubysed"] = intra->taubysed; SR["taubx_turb"] = intra->taubx_turb;
                         SR["rouse"] = intra->rouse; SR["ca_nonh"] = intra->ca_nonh; }
            auto is_ = SR.find(n);
            if (is_ != SR.end()) { k_copy_out<<<nb, 256>>>(is_->second, dout, M); return true; }
        }
        std::map<std::string, const real *> R = {
            {"zs", h.zs}, {"zb", h.zb}, {"zs0", h.zs0}, {"hh", h.hh}, {"uu", h.uu}, {"vv", h.vv}, {"qx", h.qx}, {"qy", h.qy},
            {"hu", h.hu}, {"hv", h.hv}, {"hum", h.hum}, {"hvm", h.hvm}, {"ws", h.ws}, {"wb", h.wb}, {"pres", h.pres},
            {"nuh", h.nuh}, {"taubx", h.taubx}, {"tauby", h.tauby}, {"cfu", h.cfu}, {"cfv", h.cfv}, {"dzsdt", h.dzsdt},
            {"dzsdx", h.dzsdx}, {"dzsdy", h.dzsdy}, {"vu", h.vu}, {"uv", h.uv}, {"ueu", h.ueu}, {"vev", h.vev},
            {"vmageu", h.vmageu}, {"vmagev", h.vmagev}, {"maxzs", h.maxzs}, {"minzs", h.minzs}, {"dU", h.dU}, {"dV", h.dV},
            {"windsu", h.windsu}, {"windnv", h.windnv}, {"Fvegu", h.Fvegu}, {"Fvegv", h.Fvegv}, {"ph", h.ph},
            {"infil", h.infil}, {"rainfallrate", h.rainrate}, {"bedfriccoef", h.bfc}, {"viscu", h.viscu}, {"viscv", h.viscv},
            {"ududx", h.ududx}, {"vdudy", h.vdudy}, {"vdvdy", h.vdvdy}, {"udvdx", h.udvdx}, {"dzs0dn", h.dzs0dn},
            {"taubx_add", h.taubx_add}, {"tauby_add", h.tauby_add}, {"ucan", h.ucan}, {"vcan", h.vcan}, {"wscrit", h.wscrit},
            {"gwhead", h.gwhead}, {"gwheadb", h.gwheadb}, {"gwlevel", h.gwlevel}, {"gwheight", h.gwheight},
            {"gwbottom", h.gwbottom}, {"gwu", h.gwu}, {"gwv", h.gwv}, {"gwqx", h.gwqx}, {"gwqy", h.gwqy}, {"gww", h.gww},
            {"gwcurv", h.gwcurv}, {"dinfil", h.dinfil}, {"Kx", h.gKx}, {"Ky", h.gKy}, {"Kz", h.gKz}, {"Kzinf", h.gKzinf}};
        std::map<std::string, const int *> I = {{"wetz", h.wetz}, {"wetu", h.wetu}, {"wetv", h.wetv}, {"breaking", h.breaking},
                                                {"wete", h.wetz}};
        auto ir = R.find(n);
        if (ir != R.end() && ir->second) { k_copy_out<<<nb, 256>>>(ir->second, dout, N); return true; }
        auto ii = I.find(n);
        if (ii != I.end() && ii->second) { k_copy_out_int<<<nb, 256>>>(ii->second, dout, N); return true; }
        if (n == "bedfriccoef") {   // uniform coefficient
            std::vector<double> v(N, P->bedfriccoef); CK(cudaMemcpy(dout, v.data(), N * 8, cudaMemcpyHostToDevice)); return true;
        }
        std::map<std::string, const std::vector<double> *> S = {
            {"x", &G->x}, {"y", &G->y}, {"xz", &G->xz}, {"yz", &G->yz}, {"xu", &G->xu}, {"yu", &G->yu}, {"xv", &G->xv},
            {"yv", &G->yv}, {"dsu", &G->dsu}, {"dsv", &G->dsv}, {"dsz", &G->dsz}, {"dsc", &G->dsc}, {"dnu", &G->dnu},
            {"dnv", &G->dnv}, {"dnz", &G->dnz}, {"dnc", &G->dnc}, {"dsdnui", &G->dsdnui}, {"dsdnvi", &G->dsdnvi},
            {"dsdnzi", &G->dsdnzi}, {"alfaz", &G->alfaz}, {"alfau", &G->alfau}, {"alfav", &G->alfav},
            {"sdist", &G->sdist}, {"ndist", &G->ndist}};
        if (veg) {
            S.insert({{"vegtype", &veg->vegtype}, {"nsecveg", &veg->nsec}, {"isCanopy", &veg->isCanopy}, {"Cdveg", &veg->Cd},
                      {"ahveg", &veg->ah}, {"bveg", &veg->bv}, {"Nveg", &veg->Nv}, {"Kp_can", &veg->Kp}, {"beta_can", &veg->beta},
                      {"cf_can", &veg->cf}, {"cm_can", &veg->cm}, {"lp_can", &veg->lp}});
        } else if (n == "Kp_can" || n == "lp_can") {   // veggie_init without vegetation
            std::vector<double> v(N, n == "Kp_can" ? std::numeric_limits<double>::max() : 1e-12);
            CK(cudaMemcpy(dout, v.data(), N * 8, cudaMemcpyHostToDevice)); return true;
        }
        auto is = S.find(n);
        if (is != S.end()) { CK(cudaMemcpy(dout, is->second->data(), is->second->size() * 8, cudaMemcpyHostToDevice)); return true; }
        CK(cudaMemset(dout, 0, M * 8));
        return false;
    }
    void download(size_t M) { CK(cudaMemcpy(hbuf.data(), dout, M * 8, cudaMemcpyDeviceToHost)); }
    static bool remdry_var(const std::string &n) {
        static const std::set<std::string> s = {"zs", "u", "v", "H", "thetamean", "k", "sigm", "c", "cg", "hh", "E", "R",
                                                "urms", "D", "ue", "ve", "Sk", "As"};
        return s.count(n) > 0;
    }

    // makeaverage (called after each step when tpm(1) < t <= tpm(end))
    void accumulate(double t, double dt) {
        if (mf.empty() || !(t > tpm.front() && t <= tpm.back())) return;
        double mult = std::max(dt / P->tintm, 0.0);
        unsigned nb = (unsigned)((N + 255) / 256);
        for (size_t q = 0; q < mf.size(); q++) {
            produce(mf[q], 0);
            k_mean_update<<<nb, 256>>>(dout, mm[q], mcr[q], msq[q], mva[q], mmn[q], mmx[q], mult, N, mf[q].name == "zswet");
        }
    }

    // ncoutput: write everything due at time t
    void write(double t, int uvzero, const std::vector<int> &rowhh_dummy = {}) {
        (void)rowhh_dummy;
        bool dog = outputg && P->nglobalvar > 0, dop = outputp && P->npointvar > 0 && !pts.empty();
        bool dom = outputm && !mf.empty() && itm > 1;
        // written times: netCDF always in morphological time, Fortran point files and dims.dat when morfacopt = 1
        double mf = std::max(P->morfac, 1.0), tnc = t * mf, tout = P->morfacopt == 1 ? t * mf : t;
        if (dog) {
            nitg++;
            for (size_t q = 0; q < gf.size(); q++) {
                auto &f = gf[q];
                if (f.nlay == 0) {   // one value per ship
                    const std::vector<double> &sv = shipv[f.name];
                    if (dofortran()) { fwrite(sv.data(), 8, sv.size(), gfiles[q]); fflush(gfiles[q]); }
#ifdef XB_NETCDF
                    if (donetcdf()) { size_t s2[2] = {(size_t)itg - 1, 0}, c2[2] = {1, sv.size()};
                                      NC(nc_put_vara_double(ncid, gvid[q], s2, c2, sv.data())); }
#endif
                    continue;
                }
                size_t M = N * f.nlay;
                produce(f, uvzero);
                if (dofortran()) {
                    download(M);
                    if (f.vi->type == 'i') { std::vector<int> iv(M); for (size_t k = 0; k < M; k++) iv[k] = (int)hbuf[k]; fwrite(iv.data(), 4, M, gfiles[q]); }
                    else fwrite(hbuf.data(), 8, M, gfiles[q]);
                    fflush(gfiles[q]);
                }
#ifdef XB_NETCDF
                if (donetcdf()) {
                    if (P->remdryoutput == 1 && f.nlay == 1 && f.vi->type == 'r' && remdry_var(f.name))
                        k_remdry<<<(unsigned)((N + 255) / 256), 256>>>(H->wetz, dout, N, -3.4028234663852886e38);
                    download(M);
                    size_t st[4] = {(size_t)itg - 1, 0, 0, 0}, ct[4] = {1, (size_t)f.nlay, (size_t)NY1, (size_t)NX1};
                    if (f.vi->rank == 2) { size_t s3[3] = {(size_t)itg - 1, 0, 0}, c3[3] = {1, (size_t)NY1, (size_t)NX1};
                        if (f.vi->type == 'i') { std::vector<int> iv(M); for (size_t k = 0; k < M; k++) iv[k] = (int)hbuf[k]; NC(nc_put_vara_int(ncid, gvid[q], s3, c3, iv.data())); }
                        else NC(nc_put_vara_double(ncid, gvid[q], s3, c3, hbuf.data())); }
                    else if (f.vi->rank == 4) {   // (nx+1, ny+1, nd, ngd) -> (time, ngd, nd, ny, nx): same memory order
                        size_t s5[5] = {(size_t)itg - 1, 0, 0, 0, 0}, c5[5] = {1, (size_t)ngd, (size_t)nd, (size_t)NY1, (size_t)NX1};
                        NC(nc_put_vara_double(ncid, gvid[q], s5, c5, hbuf.data())); }
                    else { std::vector<double> tr(M); for (int l = 0; l < f.nlay; l++) for (size_t k = 0; k < N; k++) tr[l * N + k] = hbuf[l * N + k];
                        NC(nc_put_vara_double(ncid, gvid[q], st, ct, tr.data())); }
                }
#endif
            }
#ifdef XB_NETCDF
            if (donetcdf()) { size_t s1 = itg - 1; NC(nc_put_var1_double(ncid, gtvid, &s1, &tnc)); }
#endif
        }
        if (dop) write_points(tout, tnc);
        if (dom) write_means();
#ifdef XB_NETCDF
        if (donetcdf() && (dog || dop || dom)) NC(nc_sync(ncid));
#endif
        if (dofortran() && output) write_dims();
    }
    void write_points(double tout, double tnc) {
        nitp++;
        size_t np = pts.size(), nv = pf.size();
        std::vector<double> vals(np * nv, 0);
        // runup gauges first: their point variables are sampled at the runup location (ncoutput: xpoints1)
        int nrd = P->nrugdepth;
        std::vector<double> hh(N), zsh(N);
        std::vector<int> si(np), sj(np);
        bool anyrug = false; for (auto &q : pts) if (q.type == 1) anyrug = true;
        if (anyrug) {
            std::vector<real> hhr(N), zsr(N);
            CK(cudaMemcpy(hhr.data(), H->hh, N * sizeof(real), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(zsr.data(), H->zs, N * sizeof(real), cudaMemcpyDeviceToHost));
            for (size_t k = 0; k < N; k++) { hh[k] = hhr[k]; zsh[k] = zsr[k]; }
        }
        std::vector<std::vector<double>> runups(np);
        for (size_t q = 0; q < np; q++) {
            si[q] = pts[q].i; sj[q] = pts[q].j;
            if (pts[q].type == 0) continue;
            std::vector<double> rec(1 + 3 * nrd, 1.79769313486231571e308);
            rec[0] = tout;
            int jr = pts[q].j;
            si[q] = NX1;
            for (int r = 0; r < nrd; r++) {   // first j = 3..nx+1 with hh(j) <= rugdepth < hh(j-1) along the gauge row
                int idumhl = -1;
                for (int i = 3; i <= NX1; i++) if (hh[G->I(i, jr)] <= P->rugdepth[r] && hh[G->I(i - 1, jr)] > P->rugdepth[r]) { idumhl = i - 1; break; }
                if (r == 0 && idumhl > 0) si[q] = idumhl;
                if (idumhl > 0) { rec[1 + 3 * r] = G->xz[G->I(idumhl, jr)]; rec[2 + 3 * r] = G->yz[G->I(idumhl, jr)]; rec[3 + 3 * r] = zsh[G->I(idumhl, jr)]; }
            }
            runups[q] = rec;
        }
        {   // gather the point values on the device: np values per variable instead of the whole field
            std::vector<int> idx(np);
            for (size_t q = 0; q < np; q++) idx[q] = (int)G->I(si[q], sj[q]);
            if (!d_pidx) { CK(cudaMalloc(&d_pidx, std::max<size_t>(np, 1) * sizeof(int)));
                           CK(cudaMalloc(&d_pval, std::max<size_t>(np * nv, 1) * sizeof(double))); }
            CK(cudaMemcpy(d_pidx, idx.data(), np * sizeof(int), cudaMemcpyHostToDevice));
            for (size_t v = 0; v < nv; v++) {
                produce(pf[v], 0);
                k_gather_points<<<(unsigned)((np + 127) / 128), 128>>>(dout, d_pidx, (int)np, d_pval + v * np);
            }
            std::vector<double> g(np * nv);
            CK(cudaMemcpy(g.data(), d_pval, np * nv * sizeof(double), cudaMemcpyDeviceToHost));
            for (size_t v = 0; v < nv; v++) for (size_t q = 0; q < np; q++) vals[q * nv + v] = g[v * np + q];
        }
        for (size_t q = 0; q < np; q++) {
            if (!dofortran()) break;
            if (pts[q].type == 0) { fwrite(&tout, 8, 1, pfiles[q]); fwrite(&vals[q * nv], 8, nv, pfiles[q]); }
            else fwrite(runups[q].data(), 8, runups[q].size(), pfiles[q]);
            fflush(pfiles[q]);
        }
#ifdef XB_NETCDF
        if (donetcdf()) {
            size_t s1 = itp - 1; NC(nc_put_var1_double(ncid, ptvid, &s1, &tnc));
            for (size_t v = 0; v < nv; v++) {
                std::vector<double> c(np); for (size_t q = 0; q < np; q++) c[q] = vals[q * nv + v];
                size_t st[2] = {(size_t)itp - 1, 0}, ct[2] = {1, np};
                NC(nc_put_vara_double(ncid, pvid[v], st, ct, c.data()));
            }
        }
#endif
    }
    void write_means() {
        nitm++;
        for (size_t q = 0; q < mf.size(); q++) {
            double *src[4] = {mm[q], mva[q], mmn[q], mmx[q]};
            for (int r = 0; r < 4; r++) {
                CK(cudaMemcpy(hbuf.data(), src[r], N * 8, cudaMemcpyDeviceToHost));
                const std::string &n = mf[q].name;
                if (r == 0 && (n == "H" || n == "urms")) { std::vector<double> sq(N); CK(cudaMemcpy(sq.data(), msq[q], N * 8, cudaMemcpyDeviceToHost)); for (size_t k = 0; k < N; k++) hbuf[k] = sqrt(sq[k]); }
                if (dofortran()) {
                    std::vector<double> w(hbuf.begin(), hbuf.begin() + N);
                    if (r == 2) for (auto &x : w) if (x > 0.99 * 1.79769313486231571e308) x = -999;
                    if (r == 3) for (auto &x : w) if (x < -0.99 * 1.79769313486231571e308) x = -999;
                    fwrite(w.data(), 8, N, mfiles[r][q]); fflush(mfiles[r][q]);
                }
#ifdef XB_NETCDF
                if (donetcdf()) { size_t st[3] = {(size_t)itm - 2, 0, 0}, ct[3] = {1, (size_t)NY1, (size_t)NX1};
                                  NC(nc_put_vara_double(ncid, mvid[r][q], st, ct, hbuf.data())); }
#endif
            }
        }
#ifdef XB_NETCDF
        if (donetcdf()) { size_t s1 = itm - 2; double tm = tpm[itm - 1] * std::max(P->morfac, 1.0); NC(nc_put_var1_double(ncid, mtvid, &s1, &tm)); }
#endif
        P->tintm = std::max(tpm[std::min(nitm + 2, (int)tpm.size()) - 1] - tpm[nitm], 2.2250738585072014e-308);
        clear_means();
    }
    void write_dims() {
        std::vector<double> ot(tpg.size() + tpp.size() + tpm.size(), -999.0);
        for (int q = 0; q < nitg; q++) ot[q] = tpg[q];
        for (int q = 0; q < nitp; q++) ot[nitg + q] = tpp[q];
        for (int q = 0; q < nitm; q++) ot[nitg + nitp + q] = tpm[q + 1];
        if (P->morfacopt == 1) for (auto &x : ot) x *= std::max(P->morfac, 1.0);
        double h[10] = {(double)nitg, (double)P->nx, (double)P->ny, 1.0, (double)P->kmax, (double)ngd, (double)nd, (double)nitp, 0.0, (double)nitm};
        FILE *f = fopen("dims.dat", "r+b"); if (!f) f = fopen("dims.dat", "wb");
        fseek(f, 0, SEEK_SET);
        fwrite(h, 8, 10, f); fwrite(ot.data(), 8, ot.size(), f);
        fclose(f);
    }
    void write_drifters(double t, const std::vector<double> &idr, const std::vector<double> &jdr, const std::vector<double> &tb,
                        const std::vector<double> &te) {
        for (int q = 0; q < P->ndrifter; q++) {
            double dr[3];
            if (t >= tb[q] && t <= te[q] && idr[q] > 1 && idr[q] <= P->nx && jdr[q] > 1 && jdr[q] <= P->ny) {
                int iz = (int)idr[q], jz = (int)jdr[q];
                double di = fmod(idr[q], 1.0), dj = fmod(jdr[q], 1.0);
                size_t k = G->I(iz, jz);
                double dx = di * G->dsu[k] * cos(G->alfaz[k]) - dj * G->dnv[k] * sin(G->alfaz[k]);
                double dy = di * G->dsu[k] * sin(G->alfaz[k]) + dj * G->dnv[k] * cos(G->alfaz[k]);
                dr[0] = G->xz[k] + dx; dr[1] = G->yz[k] + dy; dr[2] = t;
            } else { dr[0] = -999; dr[1] = -999; dr[2] = t; }
            if (dofortran()) { fwrite(dr, 8, 3, dfiles[q]); fflush(dfiles[q]); }
#ifdef XB_NETCDF
            if (donetcdf()) { size_t st[2] = {(size_t)nitd, 0}, ct[2] = {1, 2}; NC(nc_put_vara_double(ncid, dvid[q], st, ct, dr)); }
#endif
        }
#ifdef XB_NETCDF
        if (donetcdf()) { size_t s1 = nitd; NC(nc_put_var1_double(ncid, dtvid, &s1, &t)); NC(nc_sync(ncid)); }
        nitd++;
#endif
    }
#ifdef XB_NETCDF
    void init_netcdf() {
        Params &p = *P;
        NC(nc_create(p.ncfilename.c_str(), NC_CLOBBER | NC_64BIT_OFFSET, &ncid));
        nc_type RT = p.outputdouble ? NC_DOUBLE : NC_FLOAT;
        int dx, dy, dth, dbl, dsc, dio, dgt, dpts = -1, dpt = -1, dpl = -1, dmt = -1;
        NC(nc_def_dim(ncid, "nx", NX1, &dx)); NC(nc_def_dim(ncid, "ny", NY1, &dy));
        NC(nc_def_dim(ncid, "wave_angle", 1, &dth)); NC(nc_def_dim(ncid, "bed_layers", nd, &dbl));
        NC(nc_def_dim(ncid, "sediment_classes", ngd, &dsc)); NC(nc_def_dim(ncid, "inout", 2, &dio));
        int ddr = -1, ddt = -1;
        if (p.ndrifter > 0) { NC(nc_def_dim(ncid, "ndrifter", p.ndrifter, &ddr)); NC(nc_def_dim(ncid, "drifterstime", tpp.size(), &ddt)); }
        int dveg = -1, dship = -1, dnz = -1;
        if (p.ships > 0) NC(nc_def_dim(ncid, "nship", nship, &dship));
        if (p.vegetation > 0) NC(nc_def_dim(ncid, "nsecvegmax", nveg, &dveg));
        if (p.nz > 1) NC(nc_def_dim(ncid, "nz", p.nz, &dnz));
        NC(nc_def_dim(ncid, "globaltime", NC_UNLIMITED, &dgt));
        bool outp = !pts.empty() && !tpp.empty();
        if (outp) { NC(nc_def_dim(ncid, "points", pts.size(), &dpts)); NC(nc_def_dim(ncid, "pointtime", tpp.size(), &dpt));
                    NC(nc_def_dim(ncid, "pointnamelength", 64, &dpl)); }
        if (!mf.empty()) NC(nc_def_dim(ncid, "meantime", tpm.size() - 1, &dmt));
        int dum;
        NC(nc_def_dim(ncid, "tidetime", tidelen, &dum));
        if (p.tideloc > 0) NC(nc_def_dim(ncid, "tidecorners", p.tideloc, &dum));
        NC(nc_def_dim(ncid, "windtime", windlen, &dum));
        int pv, xv, yv;
        NC(nc_def_var(ncid, "_parameters", RT, 0, nullptr, &pv));
        auto att = [&](int v, const char *k, const std::string &s) { NC(nc_put_att_text(ncid, v, k, s.size(), s.c_str())); };
        // _parameters: the keys of params.txt (numbers as doubles, the rest as text) and the main derived values
        for (auto &kv : parfile_kv) {
            const std::string &k = kv.first, &v = kv.second;
            char *e; double x = strtod(v.c_str(), &e);
            std::string key = k; for (auto &ch : key) if (!isalnum((unsigned char)ch) && ch != '_') ch = '_';
            if (key.empty() || isdigit((unsigned char)key[0])) continue;
            if (e != v.c_str() && trim(std::string(e)).empty()) NC(nc_put_att_double(ncid, pv, key.c_str(), NC_DOUBLE, 1, &x));
            else att(pv, key.c_str(), v);
        }
        {
            double vals_[] = {(double)p.nx, (double)p.ny, p.tstop, p.tstart, p.tintg, p.tintp, p.tintm, p.CFL, p.eps, p.g, p.rho, 1.0};
            const char *keys_[] = {"nx", "ny", "tstop", "tstart", "tintg", "tintp", "tintm", "CFL", "eps", "g", "rho", "morfac"};
            for (int q = 0; q < 12; q++) NC(nc_put_att_double(ncid, pv, keys_[q], NC_DOUBLE, 1, &vals_[q]));
            att(pv, "wavemodel_str", "nonh");
        }
        NC(nc_put_att_text(ncid, NC_GLOBAL, "Conventions", 6, "CF-1.4"));
        att(NC_GLOBAL, "Producer", "xbnh_gpu (GPU port of XBeach r6167 non-hydrostatic, http://www.xbeach.org)");
        int dims2[2] = {dy, dx};
        NC(nc_def_var(ncid, "globalx", RT, 2, dims2, &xv)); att(xv, "units", "m"); att(xv, "long_name", "local x coordinate");
        att(xv, "standard_name", "projection_x_coordinate"); att(xv, "axis", "X");
        NC(nc_def_var(ncid, "globaly", RT, 2, dims2, &yv)); att(yv, "units", "m"); att(yv, "long_name", "local y coordinate");
        att(yv, "standard_name", "projection_y_coordinate"); att(yv, "axis", "Y");
        if (!p.projection.empty()) { att(xv, "projection", p.projection); att(yv, "projection", p.projection); }
        NC(nc_def_var(ncid, "globaltime", RT, 1, &dgt, &gtvid)); att(gtvid, "units", p.tunits); att(gtvid, "axis", "T"); att(gtvid, "standard_name", "time");
        for (auto &f : gf) {
            int dims[4], nd_ = 0, id;
            std::string shp = f.vi->shape;
            dims[nd_++] = dgt;
            if (f.nlay == 0) {   // (globaltime, nship)
                dims[nd_++] = dship;
                NC(nc_def_var(ncid, f.name.c_str(), RT, nd_, dims, &id));
                att(id, "units", f.vi->units); att(id, "long_name", f.vi->description);
                if (p.outputdouble) { double fv = -3.4028234663852886e38; NC(nc_put_att_double(ncid, id, "_FillValue", NC_DOUBLE, 1, &fv)); }
                else { float fv = -3.4028234663852886e38f; NC(nc_put_att_float(ncid, id, "_FillValue", NC_FLOAT, 1, &fv)); }
                gvid.push_back(id);
                continue;
            }
            if (f.vi->rank == 3) {
                std::string l3 = shp.substr(shp.rfind(',') + 1);
                dims[nd_++] = l3 == "nsecvegmax" ? dveg : l3 == "ngd" ? dsc : (l3 == "ntheta" || l3 == "ntheta_s") ? dth : l3 == "kmax" ? dnz : dbl;
            } else if (f.vi->rank == 4) { dims[nd_++] = dsc; dims[nd_++] = dbl; }
            dims[nd_++] = dy; dims[nd_++] = dx;
            NC(nc_def_var(ncid, f.name.c_str(), f.vi->type == 'i' ? NC_INT : RT, nd_, dims, &id));
            att(id, "coordinates", "globalx globaly"); att(id, "units", f.vi->units);
            if (strlen(f.vi->standard_name)) att(id, "standard_name", f.vi->standard_name);
            att(id, "long_name", f.vi->description);
            if (f.vi->type == 'i') { int fv = IFILL; NC(nc_put_att_int(ncid, id, "_FillValue", NC_INT, 1, &fv)); }
            else if (p.outputdouble) { double fv = -3.4028234663852886e38; NC(nc_put_att_double(ncid, id, "_FillValue", NC_DOUBLE, 1, &fv)); }
            else { float fv = -3.4028234663852886e38f; NC(nc_put_att_float(ncid, id, "_FillValue", NC_FLOAT, 1, &fv)); }
            gvid.push_back(id);
        }
        int sid = -1, xi = -1, yi = -1, pt = -1;
        if (outp) {
            NC(nc_def_var(ncid, "pointtime", RT, 1, &dpt, &ptvid)); att(ptvid, "units", p.tunits); att(ptvid, "axis", "T"); att(ptvid, "standard_name", "time");
            NC(nc_def_var(ncid, "pointx", RT, 1, &dpts, &xpvid)); att(xpvid, "units", "m"); att(xpvid, "long_name", "local x coordinate");
            att(xpvid, "standard_name", "projection_x_coordinate"); att(xpvid, "axis", "X");
            NC(nc_def_var(ncid, "pointy", RT, 1, &dpts, &ypvid)); att(ypvid, "units", "m"); att(ypvid, "long_name", "local y coordinate");
            att(ypvid, "standard_name", "projection_y_coordinate"); att(ypvid, "axis", "Y");
            int d2[2] = {dpts, dpl};
            NC(nc_def_var(ncid, "station_id", NC_CHAR, 2, d2, &sid)); att(sid, "long_name", "station identification code");
            NC(nc_def_var(ncid, "xpointindex", NC_INT, 1, &dpts, &xi)); att(xi, "long_name", "nearest x grid cell");
            NC(nc_def_var(ncid, "ypointindex", NC_INT, 1, &dpts, &yi)); att(yi, "long_name", "nearest y grid cell");
            NC(nc_def_var(ncid, "pointtypes", NC_INT, 1, &dpts, &pt)); att(pt, "long_name", "type of point (0=point, 1=rugauge)");
            for (auto &f : pf) {
                int dd[2] = {dpt, dpts}, id;
                NC(nc_def_var(ncid, ("point_" + f.name).c_str(), RT, 2, dd, &id));
                att(id, "coordinates", "pointx pointy"); att(id, "units", f.vi->units);
                if (strlen(f.vi->standard_name)) att(id, "standard_name", f.vi->standard_name);
                att(id, "long_name", f.vi->description);
                pvid.push_back(id);
            }
        }
        if (p.ndrifter > 0) {   // driftertime and drifter_NNN (x, y) per point output time
            NC(nc_def_var(ncid, "driftertime", RT, 1, &ddt, &dtvid)); att(dtvid, "units", p.tunits); att(dtvid, "axis", "T"); att(dtvid, "standard_name", "time");
            int dd2; NC(nc_def_dim(ncid, "drifterstime2", 2, &dd2));
            for (int q = 1; q <= p.ndrifter; q++) {
                char b[32]; snprintf(b, sizeof b, "drifter_%03d", q);
                int dd[2] = {ddt, dd2}, id;
                NC(nc_def_var(ncid, b, RT, 2, dd, &id)); att(id, "coordinates", "pointx pointy"); att(id, "units", "m");
                dvid.push_back(id);
            }
        }
        if (!mf.empty()) {
            NC(nc_def_var(ncid, "meantime", RT, 1, &dmt, &mtvid)); att(mtvid, "units", p.tunits); att(mtvid, "axis", "T"); att(mtvid, "standard_name", "time");
            const char *suf[4] = {"mean", "var", "min", "max"};
            for (auto &f : mf) for (int r = 0; r < 4; r++) {
                int dd[3] = {dmt, dy, dx}, id;
                NC(nc_def_var(ncid, (f.name + "_" + suf[r]).c_str(), RT, 3, dd, &id));
                att(id, "coordinates", "globalx globaly");
                att(id, "units", r == 1 ? "(" + std::string(f.vi->units) + ")^2" : std::string(f.vi->units));
                if (strlen(f.vi->standard_name)) att(id, "standard_name", f.vi->standard_name);
                att(id, "long_name", f.vi->description);
                att(id, "cell_methods", std::string("meantime: ") + (r == 0 && (f.name == "H" || f.name == "urms") ? "rms" : suf[r]));
                double fv = (r == 2 && f.name == "zswet") ? 3.4028234663852886e38 : -3.4028234663852886e38;
                NC(nc_put_att_double(ncid, id, "_FillValue", RT, 1, &fv));
                mvid[r].push_back(id);
            }
        }
        NC(nc_enddef(ncid));
        NC(nc_put_var_double(ncid, xv, G->xz.data())); NC(nc_put_var_double(ncid, yv, G->yz.data()));
        if (outp) {
            std::vector<double> px, py; std::vector<int> xi_, yi_, ty; std::vector<char> names(pts.size() * 64, 0);
            for (size_t q = 0; q < pts.size(); q++) { px.push_back(pts[q].xw); py.push_back(pts[q].yw); xi_.push_back(pts[q].i);
                yi_.push_back(pts[q].j); ty.push_back(pts[q].type); strncpy(&names[q * 64], pts[q].id.c_str(), 63); }
            NC(nc_put_var_double(ncid, xpvid, px.data())); NC(nc_put_var_double(ncid, ypvid, py.data()));
            NC(nc_put_var_int(ncid, xi, xi_.data())); NC(nc_put_var_int(ncid, yi, yi_.data())); NC(nc_put_var_int(ncid, pt, ty.data()));
            NC(nc_put_var_text(ncid, sid, names.data()));
        }
        NC(nc_sync(ncid));
    }
#endif
    void close() {
        for (auto f : gfiles) fclose(f);
        for (auto f : pfiles) fclose(f);
        for (int r = 0; r < 4; r++) for (auto f : mfiles[r]) fclose(f);
        for (auto f : dfiles) fclose(f);
#ifdef XB_NETCDF
        if (ncid >= 0) NC(nc_close(ncid));
#endif
    }
};
