// xbnh_gpu: XBeach non-hydrostatic (wavemodel = nonh) on one NVIDIA GPU.
//
// CUDA port of XBeach trunk r6167 (Deltares; LGPL). The routines are translated line by line, keeping the order of
// operations of the Fortran code, including its quirks, so that a run reproduces XBeach to within the tolerance of
// the pressure solver. File map:
//   params.h      params.F90 (all_input) for wavemodel = nonh
//   grid.h        initialize.F90 grid_bathy, spaceparams.F90 gridprops
//   k_flow.cuh    wetcells.F90, timestep.F90 compute_dt, flow_timestep.F90, bedroughness.F90
//   k_secorder.cuh flow_secondorder.F90
//   k_nonh.cuh    nonh.F90 (one-layer and reduced two-layer schemes, 2D and 1D, nonh_break, zuzv)
//   solver.cuh    pressure solvers (BiCGStab instead of SIP in 2D; cyclic reduction instead of Thomas in 1D)
//   k_bc.cuh      boundaryconditions.F90 flow_bc, compute_tide_zs0.F90
//   wavebc.h      boundaryconditions.F90 wave_bc (nonh), velocity_Boundary
//   wavespec.h    waveparamsnew.F90 spectral_wave_bc (nonhspectrum = 1)
//   k_extra.cuh   discharges, rainfall, drifters
//   k_veg.cuh     vegetation.F90 (momeqveg, porcanflow)
//   k_gw.cuh      groundwater.F90 (gw_bc, gwflow), infiltration bed friction
//   k_ship.cuh    ship.F90 (shipwave, ship_force), interp.F90 mkmap/grmap
//   k_sed.cuh     morphevolution.F90 transus, sedtransform, waveturb; initialize.F90 sed_init
//   k_morph.cuh   morphevolution.F90 bed_update (struct, fractions, avalanching), setbathy_update
//   output.h      timestep.F90 output times, ncoutput.F90, varianceupdate.F90
// Arrays use the XBeach layout: index (i,j), i = 1..nx+1 fastest (cross-shore), j = 1..ny+1.
// Build: see Makefile (nvcc -O3 -std=c++20 -arch=native [-DXB_SINGLE] [-DXB_NETCDF -lnetcdf]).
// Run  : xbnh_gpu   (in a directory holding an XBeach params.txt and its input files)

#include <chrono>
#include <array>
#include "common.cuh"
#include "params.h"
#include "io.h"
#include "grid.h"
#include "k_gw.cuh"      // before k_flow.cuh: infiltration bed friction in k_rough
#include "k_flow.cuh"
#include "k_secorder.cuh"
#include "k_nonh.cuh"
#include "solver.cuh"
#include "k_bc.cuh"
#include "k_extra.cuh"
#include "wavebc.h"
#include "wavespec.h"
#include "k_veg.cuh"
#include "k_ship.cuh"
#include <functional>
#include "k_sed.cuh"
#include "k_morph.cuh"
#include "k_gravel.cuh"
#include "output.h"

static const char *XBNH_VERSION = "2.0.0";

template <typename T> static T *dalloc(size_t n, T val = 0) {
    T *p; CK(cudaMalloc(&p, n * sizeof(T)));
    std::vector<T> h(n, val); CK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <typename T> static void H2D(T *dst, const std::vector<T> &src) {
    CK(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T> static std::vector<T> D2H(const T *src, size_t n) {
    std::vector<T> h(n); CK(cudaMemcpy(h.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost)); return h;
}
static std::vector<real> toreal(const std::vector<double> &a) { return std::vector<real>(a.begin(), a.end()); }

int main(int argc, char **argv) {
    if (argc > 1 && (std::string(argv[1]) == "--version" || std::string(argv[1]) == "-v")) {
        printf("xbnh_gpu %s (%s)\n", XBNH_VERSION, sizeof(real) == 8 ? "fp64" : "fp32"); return 0;
    }
    auto wall0 = std::chrono::steady_clock::now();
    ParFile F("params.txt");
    Params P;
    read_params(F, P);
    if (P.gwflow && P.cyclic) fail("gwflow = 1 with cyclic = 1 is not supported");
    if (P.gwflow && P.ngd > 1) fail("gwflow = 1 with ngd > 1 needs the sediment module (D50top), not available yet");
    if (P.ships && P.cyclic) fail("ships = 1 with cyclic = 1 is not supported");
    if (P.hotstartflow) fail("hotstartflow = 1 is not supported");

    // ---------------------------------------------------------------------------------------------
    // grid and constants
    // ---------------------------------------------------------------------------------------------
    Grid G;
    grid_bathy(P, G);
    const int nx = G.nx, ny = G.ny, NX1 = G.NX1, NY1 = G.NY1;
    const size_t N = (size_t)NX1 * NY1;
    auto I = [&](int i, int j) { return G.I(i, j); };
    Const h{};
    h.nx = nx; h.ny = ny; h.NX1 = NX1; h.NY1 = NY1; h.oned = ny == 0; h.uni = G.uni;
    {
        size_t k = I(std::min(2, NX1), std::min(2, NY1));
        h.m_dsu = G.dsu[k]; h.m_dsv = G.dsv[k]; h.m_dsz = G.dsz[k]; h.m_dsc = G.dsc[k];
        h.m_dnu = G.dnu[k]; h.m_dnv = G.dnv[k]; h.m_dnz = G.dnz[k]; h.m_dnc = G.dnc[k];
        h.m_dsdnui = G.dsdnui[k]; h.m_dsdnvi = G.dsdnvi[k]; h.m_dsdnzi = G.dsdnzi[k];
        h.m_alfau = G.alfau[k]; h.m_alfav = G.alfav[k]; h.m_alfaz = G.alfaz[k];
    }
    h.par_dy = P.dy;
    h.g = P.g; h.rho = P.rho; h.eps = P.eps; h.eps_sd = P.eps_sd; h.umin = P.umin; h.px = M_PI;
    h.maxfacg = P.maxfacg; h.nuh = P.nuh; h.nuhv = P.nuhv; h.rhoa = P.rhoa; h.Cd = P.Cd;
    h.fc = 2 * P.wearth * sin(P.lat); h.lwave = P.lwave;
    h.smag = P.smag; h.viscosity = P.viscosity; h.secorder = P.secorder; h.oldhu = P.oldhu;
    h.bedfriction = P.bedfriction; h.bedfriccoef = P.bedfriccoef; h.maxcf = P.maxcf; h.mincf = P.mincf;
    h.nonhq3d = P.nonhq3d; h.nhbreaker = P.nhbreaker; h.nhlay = P.nhlay; h.breakviscfac = P.breakviscfac;
    h.maxbrsteep = P.maxbrsteep; h.reformsteep = P.reformsteep; h.kdmin = P.kdmin;
    h.xbg_brk = P.useXBeachG && P.nhbreaker >= 2 && !P.nonhq3d;   // nonh_cor only (one-layer)
    h.secbrsteep = P.secbrsteep; h.breakvisclen = P.breakvisclen;
    h.back = P.back; h.left = P.left; h.right = P.right; h.arc = P.ARC; h.cyclic = P.cyclic;
    h.wbcoff = P.wbctype == WBC_OFF; h.tidetype = P.tidetype; h.front_nonh = 1;
    h.order = P.order; h.theta0 = P.theta0;
    h.jlo = h.cyclic ? 3 : 2;
    h.uv_jmin = h.cyclic ? 2 : 3;  h.uv_jmax = h.cyclic ? ny : ny - 1;
    h.w_jmin = h.cyclic ? 1 : 2;   h.w_jmax = h.cyclic ? ny : ny - 1;
    h.ps_j0 = h.cyclic ? 3 : 2;    h.ps_j1 = h.cyclic ? ny - 1 : ny;
    h.wind = P.wind; h.vegetation = P.vegetation; h.veguntow = P.veguntow;
    h.gwflow = P.gwflow; h.gwnonh = P.gwnonh; h.gwscheme = P.gwscheme; h.gwheadmodel = P.gwheadmodel;
    h.gwland = P.tideloc == 4 || (P.tideloc == 2 && P.paulrevere == PR_LAND); h.gwfi = P.friction_infiltration;
    h.kx = P.kx; h.ky = P.ky; h.kz = P.kz; h.dwetlayer = P.dwetlayer; h.por = P.por;
    h.D50top = P.D50.empty() ? 0.0002 : P.D50[0]; h.gwReturb = P.gwReturb;
    h.gwvcr = P.gwReturb / (P.D50.empty() ? 0.0002 : P.D50[0]) * P.por * 1.0e-6;
    h.connectcrit = P.gwnonh == 0 ? P.dwetlayer : P.eps;
    h.fric_acc = P.friction_acceleration; h.fric_turb = P.friction_turbulence; h.dynamrough = P.dynamrough;
    h.ci = P.ci; h.phiacc = P.phit / 180 * M_PI; h.gamma_turb = P.gamma_turb; h.droot = P.droot; h.dstem = P.dstem;
    h.D90top = P.D90.empty() ? 0.0003 : P.D90[0];
    CK(cudaMemcpyToSymbol(c, &h, sizeof(Const)));
    const bool oned = ny == 0, q3 = P.nonhq3d == 1, q3lay = q3 && P.nhlay > 0;

    // launch configuration
    if (oned) { L.B = dim3(256, 1); L.G = dim3((NX1 + 255) / 256, 1); }
    else { L.B = dim3(32, 8); L.G = dim3((NX1 + 31) / 32, (NY1 + 7) / 8); }
    L.bi = (NX1 + 255) / 256; L.bj = (NY1 + 255) / 256;
    const dim3 B = L.B, GG = L.G; const unsigned bi = L.bi, bj = L.bj;

    // ---------------------------------------------------------------------------------------------
    // initial state (hotstart_init_1, readtide, readwind, flow_init, hotstart_init_2, nonh_init)
    // ---------------------------------------------------------------------------------------------
    std::vector<double> zb = G.zb, zs(N, 0), uu(N, 0), vv(N, 0);
    if (P.hotstart) {
        read_hotstart("zs", P.hotstartfileno, NX1, NY1, zs, false);
        read_hotstart("zb", P.hotstartfileno, NX1, NY1, zb, false);
        read_hotstart("uu", P.hotstartfileno, NX1, NY1, uu, false);
        read_hotstart("vv", P.hotstartfileno, NX1, NY1, vv, false);
        if (!P.gpu_hotstart_keepuv) { std::fill(uu.begin(), uu.end(), 0.0); std::fill(vv.begin(), vv.end(), 0.0); }
    }
    // setbathy_init: bed series; the bed at t = 0 replaces the depfile bed
    std::vector<double> sb_t; std::vector<std::vector<double>> sb_z;
    if (P.setbathy) {
        Tok tk(P.setbathyfile);
        for (int it = 0; it < P.nsetbathy; it++) {
            double tt; if (!tk.num(tt)) fail("cannot read setbathyfile"); tk.nextline();
            std::vector<double> z(N);
            for (int j = 0; j < NY1; j++) { for (int i = 0; i < NX1; i++) if (!tk.num(z[(size_t)j * NX1 + i])) fail("cannot read setbathyfile"); tk.nextline(); }
            sb_t.push_back(tt); sb_z.push_back(z);
        }
        for (size_t k = 0; k < N; k++) { std::vector<double> col(sb_t.size()); for (size_t q = 0; q < sb_t.size(); q++) col[q] = sb_z[q][k]; zb[k] = lin_interp(sb_t, col, 0.0); }
    }
    // readtide: tideloc corner series
    std::vector<double> tide_t; std::vector<std::vector<double>> tide_z;
    if (P.tideloc > 0) {
        auto cols = read_columns(P.zs0file, P.tideloc + 1);
        tide_t = cols[0]; for (int q = 1; q <= P.tideloc; q++) tide_z.push_back(cols[q]);
        if (P.morfacopt == 1) for (auto &x : tide_t) x /= std::max(P.morfac, 1.0);   // readtide: morphological time
        if (tide_t.empty() || tide_t.back() < P.tstop) fail("tide condition time series too short (zs0file)");
    }
    auto tide_at = [&](double tt) {
        TideNow T{};
        double z1 = P.zs0, z2 = P.zs0, z3 = P.zs0, z4 = P.zs0;
        if (P.tideloc == 1) { z1 = lin_interp(tide_t, tide_z[0], tt); z2 = z3 = z4 = z1; }
        else if (P.tideloc == 2 && P.paulrevere == PR_LAND) { z1 = lin_interp(tide_t, tide_z[0], tt); z2 = z1; z3 = lin_interp(tide_t, tide_z[1], tt); z4 = z3; }
        else if (P.tideloc == 2) { z1 = lin_interp(tide_t, tide_z[0], tt); z4 = z1; z2 = lin_interp(tide_t, tide_z[1], tt); z3 = z2; }
        else if (P.tideloc == 4) { z1 = lin_interp(tide_t, tide_z[0], tt); z2 = lin_interp(tide_t, tide_z[1], tt);
                                   z3 = lin_interp(tide_t, tide_z[2], tt); z4 = lin_interp(tide_t, tide_z[3], tt); }
        T.zs01 = z1; T.zs02 = z2; T.zs03 = z3; T.zs04 = z4;
        T.nd1 = G.ndist[I(1, 1)]; T.nd4 = G.ndist[I(NX1, 1)]; T.nd3 = G.ndist[I(NX1, NY1)]; T.nd2 = G.ndist[I(1, NY1)];
        T.sd1 = G.sdist[I(1, 1)]; T.sd4 = G.sdist[I(NX1, 1)]; T.sd3 = G.sdist[I(NX1, NY1)]; T.sd2 = G.sdist[I(1, NY1)];
        return T;
    };
    // readwind
    std::vector<double> wind_t, wind_x, wind_y;
    if (P.wind) {
        if (!P.windfile.empty()) {
            auto cols = read_columns(P.windfile, 3);
            wind_t = cols[0];
            if (P.morfacopt == 1) for (auto &x : wind_t) x /= std::max(P.morfac, 1.0);
            for (size_t q = 0; q < wind_t.size(); q++) {
                double th = (270.0 - cols[2][q] - G.alfa) * M_PI / 180.0;   // XBeach subtracts alfa in radians here
                wind_x.push_back(cos(th) * cols[1][q]); wind_y.push_back(sin(th) * cols[1][q]);
            }
            if (wind_t.empty() || wind_t.back() < P.tstop) fail("wind condition time series too short (windfile)");
        } else {
            double th = (270.0 - P.windth - G.alfa) * M_PI / 180.0;
            wind_t = {P.tstop}; wind_x = {cos(th) * P.windv}; wind_y = {sin(th) * P.windv};
        }
    }

    // ---------------------------------------------------------------------------------------------
    // device arrays
    // ---------------------------------------------------------------------------------------------
    Dev hd{};
    auto R = [&](real v = 0) { return dalloc<real>(N, v); };
    auto Iz = [&](int v = 0) { return dalloc<int>(N, v); };
    if (!G.uni) {
        hd.dsu = R(); H2D(hd.dsu, toreal(G.dsu)); hd.dsv = R(); H2D(hd.dsv, toreal(G.dsv));
        hd.dsz = R(); H2D(hd.dsz, toreal(G.dsz)); hd.dsc = R(); H2D(hd.dsc, toreal(G.dsc));
        hd.dnu = R(); H2D(hd.dnu, toreal(G.dnu)); hd.dnv = R(); H2D(hd.dnv, toreal(G.dnv));
        hd.dnz = R(); H2D(hd.dnz, toreal(G.dnz)); hd.dnc = R(); H2D(hd.dnc, toreal(G.dnc));
        hd.dsdnui = R(); H2D(hd.dsdnui, toreal(G.dsdnui)); hd.dsdnvi = R(); H2D(hd.dsdnvi, toreal(G.dsdnvi));
        hd.dsdnzi = R(); H2D(hd.dsdnzi, toreal(G.dsdnzi));
        hd.alfau = R(); H2D(hd.alfau, toreal(G.alfau)); hd.alfav = R(); H2D(hd.alfav, toreal(G.alfav));
        hd.alfaz = R(); H2D(hd.alfaz, toreal(G.alfaz));
    }
    {   // nonh_init row/column spacings
        std::vector<real> dxu(NX1), dxz(NX1), dyz(NY1), dyv(NY1), a(NX1), b(NX1), cc(NY1), dd(NY1);
        for (int i = 1; i <= NX1; i++) { dxu[i - 1] = G.dsu[I(i, 1)]; dxz[i - 1] = G.dsz[I(i, 1)]; a[i - 1] = 1 / (real)G.dsu[I(i, 1)]; b[i - 1] = 1 / (real)G.dsz[I(i, 1)]; }
        for (int j = 1; j <= NY1; j++) { dyz[j - 1] = G.dnz[I(1, j)]; dyv[j - 1] = G.dnv[I(1, j)]; cc[j - 1] = 1 / (real)G.dnz[I(1, j)]; dd[j - 1] = 1 / (real)G.dnv[I(1, j)]; }
        for (int i = 0; i < NX1; i++) { a[i] = (real)(1.0 / G.dsu[I(i + 1, 1)]); b[i] = (real)(1.0 / G.dsz[I(i + 1, 1)]); }
        for (int j = 0; j < NY1; j++) { cc[j] = (real)(1.0 / G.dnz[I(1, j + 1)]); dd[j] = (real)(1.0 / G.dnv[I(1, j + 1)]); }
        hd.dxu = dalloc<real>(NX1); H2D(hd.dxu, dxu); hd.dxz = dalloc<real>(NX1); H2D(hd.dxz, dxz);
        hd.dyz = dalloc<real>(NY1); H2D(hd.dyz, dyz); hd.dyv = dalloc<real>(NY1); H2D(hd.dyv, dyv);
        hd.ddxu = dalloc<real>(NX1); H2D(hd.ddxu, a); hd.ddxz = dalloc<real>(NX1); H2D(hd.ddxz, b);
        hd.ddyz = dalloc<real>(NY1); H2D(hd.ddyz, cc); hd.ddyv = dalloc<real>(NY1); H2D(hd.ddyv, dd);
    }
    hd.zb = R(); hd.zs = R(); hd.hh = R(); hd.uu = R(); hd.vv = R(); hd.hu = R(); hd.hv = R(); hd.hum = R(); hd.hvm = R();
    hd.qx = R(); hd.qy = R(); hd.wetu = Iz(); hd.wetv = Iz(); hd.wetz = Iz();
    hd.dzsdx = R(); hd.dzsdy = R(); hd.ududx = R(); hd.vdudy = R(); hd.vdvdy = R(); hd.udvdx = R(); hd.nuh = R();
    hd.viscu = R(); hd.viscv = R(); hd.taubx = R(); hd.tauby = R(); hd.cfu = R(); hd.cfv = R();
    hd.vu = R(); hd.uv = R(); hd.ueu = R(); hd.vev = R(); hd.vmageu = R(); hd.vmagev = R(); hd.uu_old = R(); hd.vv_old = R();
    hd.u = R(); hd.v = R(); hd.dzsdt = R(); hd.ve = R();
    hd.zs0 = R(); hd.dzs0dn = R(); hd.umean = R(); hd.vmean = R();
    hd.maxzs = R(DFILL); hd.minzs = R(DFILL);
    if (P.wind) { hd.windsu = R(); hd.windnv = R(); }
    hd.breaking = Iz(); hd.brk_new = Iz(); hd.nonhU = Iz(1); hd.nonhV = Iz(1); hd.nonhZ = Iz(1);
    hd.pres = R(); hd.ws = R(); hd.wb = R(); hd.Wm = R(); hd.Wm_old = R(); hd.wcoef = R(1);
    hd.au0 = R(); hd.au1 = R(); hd.av0 = R(); hd.av1 = R();
    for (int q = 0; q < 5; q++) { hd.awb[q] = R(); hd.aws[q] = R(); hd.mat[q] = R(q == 0 ? 1 : 0); }
    hd.awbr = R(); hd.awsr = R(); hd.rhs = R(); hd.dp = R();
    hd.zbu = R(); hd.zbv = R(); hd.zsu = R(); hd.zsv = R();
    if (q3) { hd.dU = R(); hd.dV = R(); hd.dU0 = R(); hd.dV0 = R(); hd.Wm0 = R(); hd.omega = R();
              hd.adu0 = R(); hd.adu1 = R(); hd.adv0 = R(); hd.adv1 = R(); }
    hd.w1 = R(); hd.w2 = R(); hd.w3 = R(); hd.w4 = R(); hd.wrk1p = R();
    hd.r = R(); hd.rh = R(); hd.p = R(); hd.v_ = R(); hd.s = R(); hd.t = R(); hd.y = R(); hd.z = R();
    hd.sc = dalloc<double>(16); hd.dtmin = dalloc<unsigned long long>(1);
    // offshore boundary series ui, vi, zi, wi, dui, dvi: one block, uploaded with one copy per step
    real *bc6 = dalloc<real>(6 * (size_t)NY1);
    hd.ui = bc6; hd.vi = bc6 + NY1; hd.zi = bc6 + 2 * NY1; hd.wi = bc6 + 3 * NY1; hd.dui = bc6 + 4 * NY1; hd.dvi = bc6 + 5 * NY1;
    hd.msum = dalloc<double>(4); hd.skey = dalloc<unsigned long long>(2); hd.stale = dalloc<double>(2);
    real *zs0old = (P.tidetype == TT_INSTANT) ? R() : nullptr;
    real *d_sdist = R(), *d_ndist = R();
    H2D(d_sdist, toreal(G.sdist)); H2D(d_ndist, toreal(G.ndist));
    // bed friction coefficient map
    if (!P.bedfricfile.empty()) {
        std::vector<double> bf; read_grid_file(P.bedfricfile, NX1, NY1, bf, true);
        hd.bfc = R(); H2D(hd.bfc, toreal(bf));
        printf("bedfricfile %s: %.4f to %.4f\n", P.bedfricfile.c_str(), *std::min_element(bf.begin(), bf.end()), *std::max_element(bf.begin(), bf.end()));
    }
    // rainfall
    std::vector<double> rain_t; std::vector<real *> rain_maps;
    if (P.rainfall) {
        hd.rainrate = R();
        if (P.rainfallratefile.empty()) { std::vector<real> r(N, (real)(P.rainfallrate / 1000.0 / 3600.0)); H2D(hd.rainrate, r); }
        else {
            Tok t(P.rainfallratefile);
            for (int q = 0; q < P.nrainfallrate; q++) {
                double tt; if (!t.num(tt)) fail("cannot read rainfallratefile");
                rain_t.push_back(tt);
                std::vector<real> m(N);
                for (size_t k = 0; k < N; k++) { double v; if (!t.num(v)) fail("cannot read rainfallratefile"); m[k] = (real)(v / 1000.0 / 3600.0); }
                real *dm = R(); H2D(dm, m); rain_maps.push_back(dm);
            }
        }
    }
    // vegetation (veggie_init)
    Veg V;
    auto upload_veg = [&]() { H2D(hd.Cdveg, toreal(V.Cd)); };
    if (P.vegetation) {
        V = veggie_init(P, G);
        const size_t M = N * V.nsecmax;
        std::vector<int> ns(N), ic(N);
        for (size_t k = 0; k < N; k++) { ns[k] = (int)V.nsec[k]; ic[k] = (int)V.isCanopy[k]; }
        hd.nsecveg = Iz(); H2D(hd.nsecveg, ns);
        hd.Cdveg = dalloc<real>(M); hd.ahveg = dalloc<real>(M); hd.bveg = dalloc<real>(M); hd.Nveg = dalloc<real>(M);
        upload_veg(); H2D(hd.ahveg, toreal(V.ah)); H2D(hd.bveg, toreal(V.bv)); H2D(hd.Nveg, toreal(V.Nv));
        hd.zb0 = R(); H2D(hd.zb0, toreal(zb));
        hd.Fvegu = R(); hd.Fvegv = R();
        if (P.porcanflow) {
            hd.isCanopy = Iz(); H2D(hd.isCanopy, ic);
            hd.Kp_can = R(); H2D(hd.Kp_can, toreal(V.Kp)); hd.beta_can = R(); H2D(hd.beta_can, toreal(V.beta));
            hd.cf_can = R(); H2D(hd.cf_can, toreal(V.cf)); hd.cm_can = R(); H2D(hd.cm_can, toreal(V.cm));
            hd.lp_can = R(); H2D(hd.lp_can, toreal(V.lp));
            hd.ucan = R(); hd.vcan = R(); hd.Fcanu = R(); hd.Fcanv = R();
        }
        size_t nv = 0; for (auto x : V.vegtype) nv += x > 0;
        printf("vegetation: %zu vegetated cells, nsecvegmax %d%s\n", nv, V.nsecmax, P.porcanflow ? ", porous canopy model" : "");
    }
    // ships (ship_init)
    std::vector<Ship> ships;
    double *d_xz = nullptr, *d_yz = nullptr, *d_zsv = nullptr, *d_sbuf = nullptr;
    int *d_pairs = nullptr, *d_count = nullptr, *d_sidx = nullptr, pair_cap = 0;
    if (P.ships) {
        ships = ship_init(P);
        hd.ph = R();
        d_xz = dalloc<double>(N); H2D(d_xz, G.xz); d_yz = dalloc<double>(N); H2D(d_yz, G.yz);
        d_zsv = dalloc<double>(N); d_count = dalloc<int>(1);
    }
    // groundwater (gw_init; hotstart_init_2 for the groundwater fields)
    double dtgw = 1e300;
    if (P.gwflow) {
        std::vector<double> bot(N, P.aquiferbot), head(N, P.gw0);
        if (!P.aquiferbotfile.empty()) read_grid_file(P.aquiferbotfile, NX1, NY1, bot, true);
        if (!P.gw0file.empty()) read_grid_file(P.gw0file, NX1, NY1, head, true);
        for (int j = 1; j <= NY1; j++) { bot[I(1, j)] = bot[I(2, j)]; bot[I(NX1, j)] = bot[I(nx, j)]; }
        if (ny > 0) for (int i = 1; i <= NX1; i++) { bot[I(i, 1)] = bot[I(i, 2)]; bot[I(i, NY1)] = bot[I(i, ny)]; }
        std::vector<double> lev(N), hgt(N), back(2 * NY1);
        for (size_t k = 0; k < N; k++) {
            bot[k] = std::min(bot[k], zb[k]);
            head[k] = std::max(head[k], bot[k]);
            lev[k] = std::max(std::min(zb[k], head[k]), bot[k]);
            hgt[k] = lev[k] - bot[k];
        }
        for (int j = 1; j <= NY1; j++) { back[2 * (j - 1)] = head[I(nx, j)]; back[2 * (j - 1) + 1] = head[I(NX1, j)]; }
        std::vector<double> curv(N, 0.0);
        if (P.hotstart) {
            read_hotstart("gwlevel", P.hotstartfileno, NX1, NY1, lev, false);
            read_hotstart("gwhead", P.hotstartfileno, NX1, NY1, head, false);
            if (P.gwnonh) read_hotstart("gwcurv", P.hotstartfileno, NX1, NY1, curv, false);
        }
        for (real **a : {&hd.gwheadb, &hd.gwu, &hd.gwv, &hd.gwqx, &hd.gwqy, &hd.gww, &hd.gKx, &hd.gKy, &hd.gKz, &hd.gKzinf,
                         &hd.gwhu, &hd.gwhv, &hd.gwheadtop, &hd.KxN, &hd.KyN, &hd.KzN, &hd.fracdt, &hd.gwratio, &hd.dynpres,
                         &hd.infiluncon, &hd.infilcon, &hd.infilhorgw, &hd.infilhorsw}) *a = R();
        hd.gwhead = R(); H2D(hd.gwhead, toreal(head)); hd.gwlevel = R(); H2D(hd.gwlevel, toreal(lev));
        hd.gwheight = R(); H2D(hd.gwheight, toreal(hgt)); hd.gwbottom = R(); H2D(hd.gwbottom, toreal(bot));
        hd.gwcurv = R(); H2D(hd.gwcurv, toreal(curv));
        hd.gw0back = dalloc<real>(2 * NY1); H2D(hd.gw0back, toreal(back));
        hd.dinfil = R((real)std::max(P.dwetlayer / 3.0, (double)0.02f));
        hd.Kzinf = R((real)P.kz); hd.Kxupd = R((real)P.kx); hd.Kyupd = R((real)P.ky); hd.Kzupd = R((real)P.kz);
        hd.gwconn = Iz();
        if (!hd.infil) hd.infil = R();
        if (P.friction_infiltration) { if (!hd.taubx_add) hd.taubx_add = R(); if (!hd.tauby_add) hd.tauby_add = R(); }
        // timestep.F90 groundwater stability limit (hydrostatic groundwater)
        if (P.gwnonh == 0) {
            int j1 = ny == 0 ? 1 : 2, j2 = std::max(ny, 1);
            for (int j = j1; j <= j2; j++) for (int i = 2; i <= nx; i++) {
                size_t k = I(i, j);
                dtgw = std::min(dtgw, P.CFL * G.dsc[k] * G.dsc[k] * P.por / (2 * P.kx));
                if (P.ny > 2) dtgw = std::min(dtgw, P.CFL * G.dnc[k] * G.dnc[k] * P.por / (2 * P.ky));
            }
        }
        printf("groundwater: %s head, %s flow, aquifer bottom %.2f to %.2f m%s\n", P.gwnonh ? "non-hydrostatic" : "hydrostatic",
               P.gwscheme ? "turbulent" : "laminar", *std::min_element(bot.begin(), bot.end()), *std::max_element(bot.begin(), bot.end()),
               P.gwhorinfil ? ", horizontal infiltration" : "");
    }
    // sediment (sed_init)
    SedHost SH;
    if (P.sedtrans) {
        sed_init(P, G, zb, SH, [&](real v) { return R(v); });
        hd.Susg = SH.h.Susg; hd.Subg = SH.h.Subg;
        h.sed_ngd = P.ngd; h.sed_morfac = (real)P.morfac;
        CK(cudaMemcpyToSymbol(c, &h, sizeof(Const)));
    }
    if (P.morphology || P.setbathy) { if (!hd.zb0) { hd.zb0 = R(); H2D(hd.zb0, toreal(zb)); } }
    if (P.useXBeachG && P.nhbreaker == 3) hd.wscrit = R();
    // bed friction add-ons (bedroughness.F90): grain-size roughness, dynamic roughness, acceleration, turbulence
    if (P.sedtrans) { hd.D50topA = SH.h.D50top; hd.D90topA = SH.h.D90top; hd.kbA = SH.h.kb; hd.sedero = SH.h.sedero; }
    if (P.bedfriction == BF_WCG && P.ngd > 1) {
        if (!P.sedtrans) fail("bedfriction = white-colebrook-grainsize with ngd > 1 needs sedtrans = 1 in this build");
        hd.kru = R(); hd.krv = R();
    }
    if (P.friction_acceleration || P.friction_turbulence) {
        if (!hd.taubx_add) hd.taubx_add = R();
        if (!hd.tauby_add) hd.tauby_add = R();
        hd.ueuf = R(); hd.vevf = R(); hd.ueuold = R(); hd.vevold = R();
    }
    if (P.dynamrough && P.bedfriction == BF_MANNING) {
        std::vector<double> n0(N, P.bedfriccoef);
        if (!P.bedfricfile.empty()) read_grid_file(P.bedfricfile, NX1, NY1, n0, true);
        double nmin = *std::min_element(n0.begin(), n0.end());
        std::vector<int> iv(N); for (size_t k = 0; k < N; k++) iv[k] = n0[k] > nmin ? 1 : 0;
        if (!hd.bfc) hd.bfc = R();
        H2D(hd.bfc, toreal(n0)); hd.nman0 = R(); H2D(hd.nman0, toreal(n0)); hd.isveggie = Iz(); H2D(hd.isveggie, iv);
        h.nmanmin = (real)nmin;
    }
    CK(cudaMemcpyToSymbol(c, &h, sizeof(Const)));
    // gravel formulations (XBeach-G): constants of the first call of Nielsen2006 / mccall_vanrijn
    const bool gravel = P.sedtrans && (P.form == 3 || P.form == 4);
    if (P.sedtrans && P.form >= 5 && P.form <= 10)
        printf("warning: XBeach r6167 transus never calls the formula of form %d (it sits in mccall_vanrijn, called for "
               "mccall_vanrijn only), so this run has no sediment transport, as in XBeach\n", P.form);
    if (P.sedtrans && P.form == 3 && P.streaming != 1)
        printf("warning: XBeach r6167 Nielsen2006 stops with a segmentation fault unless streaming = 1 (array fe not allocated); "
               "xbnh_gpu runs it\n");
    bool grav_first = true;
    GravDev GD{};
    if (gravel) {
        GravConst K{};
        K.sedfricfac = P.sedfricfac; K.phaselag = P.phaselag; K.streaming = P.streaming; K.slopecorr = P.slopecorr;
        K.incldzdx = P.incldzdx; K.inclrelweight = P.inclrelweight; K.gwflow = P.gwflow; K.lwt = P.lwt;
        K.Ctrans = P.Ctrans; K.phirad = P.phit / 180 * M_PI; K.Arms = P.Arms; K.fsed = P.fsed; K.uprushfac = P.uprushfac;
        K.backwashfac = P.backwashfac; K.yturb = P.yturb; K.facthr = P.facthr;
        double rr = P.reposeangle * M_PI / 180; K.reposerad = rr; K.reposedzdx = tan(rr);
        for (int jg = 0; jg < P.ngd; jg++) {
            double dstar = pow(P.g * (P.rhos / P.rho - 1) / 1e-6 / 1e-6, 1.0 / 3) * P.D50[jg];
            K.dstar[jg] = dstar;
            K.shieldscrit[jg] = P.thetcr < 0 ? 0.3 / (1 + 1.2 * dstar) + 0.055 * (1 - exp(-0.02 * dstar)) : P.thetcr;
        }
        CK(cudaMemcpyToSymbol(gc, &K, sizeof(GravConst)));
        for (real **a : {&GD.ulocal, &GD.ulocalold, &GD.umeanupd, &GD.uvarupd, &GD.dudts, &GD.dcfl, &GD.Arms, &GD.qtmp}) *a = R();
        CK(cudaMemcpyToSymbol(gd, &GD, sizeof(GravDev)));
    }
    IntraDev ID_{};
    if (P.sedtrans && P.form == 11) {   // intra_sedtr constants
        IntraConst K{};
        K.cf_transport = P.cf_transport; K.nonh_za = P.nonh_za; K.ws_nonh = P.ws_nonh; K.yturb = P.yturb; K.facthr = P.facthr;
        K.lwt = P.lwt; K.bed = P.bed; K.sus = P.sus; K.slopecorr = P.slopecorr;
        double ratio_s = (P.rhos - P.rho) / P.rho, rr = P.reposeangle * M_PI / 180.0;
        K.reposerad = rr; K.reposedzd = tan(rr);
        for (int jg = 0; jg < P.ngd; jg++) {
            double D50 = P.D50[jg], dstar = D50 * pow((ratio_s - 1.0) * P.g / pow(1e-6, 2.0), 1.0 / 3.0);
            double sc = P.thetcr < 0 ? 0.3 / (1 + 1.2 * dstar) + 0.055 * (1 - exp(-0.02 * dstar)) : P.thetcr;
            K.dstar[jg] = dstar; K.taucr[jg] = sc * (P.rhos - P.rho) * P.g * D50;
            K.za[jg] = P.nonh_za <= 0 ? std::max(20 * D50 / 2, 0.01) : P.nonh_za;
            if (P.ws_nonh > 0) K.ws[jg] = P.ws_nonh;
            else {
                double Te = 10.0, kvis = 4.0 / (20.0 + Te) * 1e-5;
                double Sster = D50 / (4 * kvis) * sqrt((P.rhos / P.rho - 1) * P.g * D50);
                double cc1 = 1.06 * tanh(0.064 * Sster * exp(-7.5 / (Sster * Sster)));
                double cc2 = 0.22 * tanh(2.34 * pow(Sster, -1.18) * exp(-0.0064 * (Sster * Sster)));
                K.ws[jg] = (cc1 + cc2 * Sster) * sqrt((P.rhos / P.rho - 1.0) * P.g * D50);
            }
        }
        CK(cudaMemcpyToSymbol(ic_, &K, sizeof(IntraConst)));
        for (real **a : {&ID_.taubxsed, &ID_.taubysed, &ID_.taubx_turb, &ID_.rouse, &ID_.c1mean}) *a = R();
        CK(cudaMalloc(&ID_.ca_nonh, N * P.ngd * sizeof(real))); CK(cudaMemset(ID_.ca_nonh, 0, N * P.ngd * sizeof(real)));
        CK(cudaMemcpyToSymbol(idv, &ID_, sizeof(IntraDev)));
    }
    CK(cudaMemcpyToSymbol(d, &hd, sizeof(Dev)));

    // cyclic halo helper
    auto cyc = [&](std::initializer_list<real *> fr, std::initializer_list<int *> fi = {}) {
        if (!h.cyclic) return;
        CycSet s{}; s.n = 0;
        for (auto p : fr) { if (p) { s.p[s.n] = p; s.isint[s.n++] = 0; } }
        for (auto p : fi) { if (p) { s.p[s.n] = p; s.isint[s.n++] = 1; } }
        if (s.n) k_cyc<<<bi, 256>>>(s);
    };

    // flow_init
    H2D(hd.zb, toreal(zb));
    // tide_init: zsinitfile, or the corner levels at t = 0 spread over the grid (fill_tide_grid)
    if (!P.zsinitfile.empty()) {
        std::vector<double> z0; read_grid_file(P.zsinitfile, NX1, NY1, z0, true); H2D(hd.zs0, toreal(z0));
    } else {
        TideNow T0 = tide_at(0.0);
        k_tide_edges<<<GG, B>>>(T0, P.tideloc, P.paulrevere, 1);
        if (P.tideloc == 2 && P.paulrevere == PR_LAND) { k_tide_complex<<<1, 1024>>>(T0, 3, d_sdist, d_ndist); if (ny > 0) k_tide_complex<<<1, 1024>>>(T0, 4, d_sdist, d_ndist); }
        if (P.tideloc == 2 && P.paulrevere == PR_SEA) { k_tide_complex<<<1, 1024>>>(T0, 1, d_sdist, d_ndist); k_tide_complex<<<1, 1024>>>(T0, 2, d_sdist, d_ndist); k_tide_copy_fb<<<bj, 256>>>(); }
        if (P.tideloc == 4) { for (int id = 1; id <= 4; id++) if (id < 4 || ny > 0) k_tide_complex<<<1, 1024>>>(T0, id, d_sdist, d_ndist); k_tide_copy_fb<<<bj, 256>>>(); }
        if (ny > 0) k_tide_fill<<<std::max(1, ny - 1), 256>>>(d_sdist);
    }
    std::vector<real> zs0h = D2H(hd.zs0, N);
    if (!P.hotstart) for (size_t k = 0; k < N; k++) zs[k] = std::max(zb[k] + 2.220446049250313e-16, (double)zs0h[k]);
    {
        std::vector<double> hh(N), hu(N), hv(N);
        for (size_t k = 0; k < N; k++) hh[k] = std::max(zs[k] - zb[k], P.eps);
        for (int j = 1; j <= NY1; j++) {
            for (int i = 1; i <= nx; i++) hu[I(i, j)] = std::max(zs[I(i, j)], zs[I(i + 1, j)]) - std::max(zb[I(i, j)], zb[I(i + 1, j)]);
            hu[I(NX1, j)] = hu[I(nx, j)];
        }
        if (ny > 0) for (int i = 1; i <= NX1; i++) {
            for (int j = 1; j <= ny; j++) hv[I(i, j)] = std::max(zs[I(i, j)], zs[I(i, j + 1)]) - std::max(zb[I(i, j)], zb[I(i, j + 1)]);
            hv[I(i, NY1)] = hv[I(i, ny)];
        } else for (int i = 1; i <= NX1; i++) hv[I(i, 1)] = zs[I(i, 1)] - zb[I(i, 1)];
        std::vector<int> wet(N);
        for (size_t k = 0; k < N; k++) wet[k] = zs[k] > zb[k] + P.eps ? 1 : 0;
        H2D(hd.zs, toreal(zs)); H2D(hd.hh, toreal(hh)); H2D(hd.hu, toreal(hu)); H2D(hd.hv, toreal(hv));
        H2D(hd.wetz, wet); H2D(hd.wetu, wet); H2D(hd.wetv, wet);
        H2D(hd.uu, toreal(uu)); H2D(hd.vv, toreal(vv));
        CK(cudaMemcpy(hd.uu_old, hd.uu, N * sizeof(real), cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(hd.vv_old, hd.vv, N * sizeof(real), cudaMemcpyDeviceToDevice));
    }
    // hotstart_init_2
    if (P.hotstart) {
        std::vector<int> brk; read_hotstart("breaking", P.hotstartfileno, NX1, NY1, brk, true); H2D(hd.breaking, brk);
        std::vector<double> a;
        read_hotstart("wb", P.hotstartfileno, NX1, NY1, a, false); H2D(hd.wb, toreal(a));
        read_hotstart("ws", P.hotstartfileno, NX1, NY1, a, false); H2D(hd.ws, toreal(a));
        if (q3) { read_hotstart("dU", P.hotstartfileno, NX1, NY1, a, false); H2D(hd.dU, toreal(a));
                  read_hotstart("dV", P.hotstartfileno, NX1, NY1, a, false); H2D(hd.dV, toreal(a)); }
        read_hotstart("umean", P.hotstartfileno, NX1, NY1, a, false); H2D(hd.umean, toreal(a));
        read_hotstart("vmean", P.hotstartfileno, NX1, NY1, a, false); H2D(hd.vmean, toreal(a));
        cyc({hd.zs, hd.uu, hd.vv, hd.ws, hd.wb}, {hd.breaking});
    }
    // nonh_init: wcoef (nonh_init_wcoef from zs0, or nhlay for nonhq3d), zuzv
    auto init_wcoef = [&]() {
        if (q3 || P.dispc > 0) { H2D(hd.wcoef, std::vector<real>(N, (real)(q3 ? P.nhlay : P.dispc))); return; }
        k_init_wcoef<<<GG, B>>>(2 * M_PI / P.Topt);
    };
    init_wcoef();
    k_zuzv<<<GG, B>>>();
    if (P.wind) k_wind<<<GG, B>>>((real)wind_x[0], (real)wind_y[0]);

    // discharges (discharge_init)
    std::vector<Discharge> disch;
    std::vector<double> disch_t;
    if (P.ndischarge > 0) {
        std::ifstream f(P.disch_loc_file);
        if (!f) fail("cannot open disch_loc_file");
        std::vector<std::array<double, 4>> loc(P.ndischarge);
        for (auto &l : loc) if (!(f >> l[0] >> l[1] >> l[2] >> l[3])) fail("cannot read disch_loc_file");
        std::vector<std::vector<double>> q(P.ndischarge);
        if (P.ntdischarge > 0) {
            auto cols = read_columns(P.disch_timeseries_file, P.ndischarge + 1);
            disch_t = cols[0];
            if (P.morfacopt == 1) for (auto &x : disch_t) x /= std::max(P.morfac, 1.0);
            for (int a = 0; a < P.ndischarge; a++) q[a] = cols[a + 1];
        }
        auto nearest = [&](double x, double y) {
            double best = 1e300; int bi_ = 1, bj_ = 1;
            for (int j = 1; j <= NY1; j++) for (int i = 1; i <= NX1; i++) {
                double dd = sqrt((G.xz[I(i, j)] - x) * (G.xz[I(i, j)] - x) + (G.yz[I(i, j)] - y) * (G.yz[I(i, j)] - y));
                if (dd < best) { best = dd; bi_ = i; bj_ = j; }
            }
            return std::make_pair(bi_, bj_);
        };
        for (int a = 0; a < P.ndischarge; a++) {
            Discharge D{}; D.q = q[a];
            D.pnt = (loc[a][0] == loc[a][2] && loc[a][1] == loc[a][3]) ? 1 : 0;
            auto mb = nearest(loc[a][0], loc[a][1]), me = nearest(loc[a][2], loc[a][3]);
            if (D.pnt) { D.m1 = mb.first; D.n1 = mb.second; D.m2 = 0; D.n2 = 0; }
            else if (mb.first != me.first) {
                D.m1 = std::min(mb.first, me.first); D.m2 = std::max(mb.first, me.first);
                D.n1 = (int)lround(0.5 * (mb.second + me.second)); D.n1 = std::min(std::max(D.n1, 1), std::max(ny, 1)); D.n2 = D.n1;
            } else {
                D.m1 = (int)lround(0.5 * (mb.first + me.first)); D.m1 = std::min(std::max(D.m1, 1), nx);
                D.n1 = std::min(mb.second, me.second); D.n2 = std::max(mb.second, me.second); D.m2 = D.m1;
            }
            disch.push_back(D);
        }
    }
    auto qdisch = [&](const Discharge &D, double tt) { return disch_t.empty() ? 0.0 : lin_interp(disch_t, D.q, tt); };

    // drifters (drifter_init)
    std::vector<double> idr, jdr, tdb, tde;
    double *d_idr = nullptr, *d_jdr = nullptr, *d_tdb = nullptr, *d_tde = nullptr;
    if (P.ndrifter > 0) {
        std::ifstream f(P.drifterfile);
        for (int q = 0; q < P.ndrifter; q++) {
            double xd, yd, tb, te; if (!(f >> xd >> yd >> tb >> te)) fail("cannot read drifterfile");
            double best = 1e300; int bi_ = 1, bj_ = 1;
            for (int j = 1; j <= NY1; j++) for (int i = 1; i <= NX1; i++) {
                double dd = sqrt((G.xz[I(i, j)] - xd) * (G.xz[I(i, j)] - xd) + (G.yz[I(i, j)] - yd) * (G.yz[I(i, j)] - yd));
                if (dd < best) { best = dd; bi_ = i; bj_ = j; }
            }
            size_t k = I(bi_, bj_);
            double ds = (xd - G.xz[k]) * cos(G.alfaz[k]) + (yd - G.yz[k]) * sin(G.alfaz[k]);
            double dn = -(xd - G.xz[k]) * sin(G.alfaz[k]) + (yd - G.yz[k]) * cos(G.alfaz[k]);
            idr.push_back(bi_ + ds / G.dsu[k]); jdr.push_back(bj_ + dn / G.dnv[k]); if (P.morfacopt == 1) { tb /= std::max(P.morfac, 1.0); te /= std::max(P.morfac, 1.0); }
            tdb.push_back(tb); tde.push_back(te);
        }
        d_idr = dalloc<double>(P.ndrifter); H2D(d_idr, idr); d_jdr = dalloc<double>(P.ndrifter); H2D(d_jdr, jdr);
        d_tdb = dalloc<double>(P.ndrifter); H2D(d_tdb, tdb); d_tde = dalloc<double>(P.ndrifter); H2D(d_tde, tde);
    }

    // wave boundary
    SpectralGen sgen;
    bool spectral = P.wbctype == WBC_PARAMETRIC || P.wbctype == WBC_JONSTABLE || P.wbctype == WBC_SWAN || P.wbctype == WBC_VARDENS;
    if (spectral) sgen.init(P, G, hd.zs0, hd.zb);
    WaveBC wbc; wbc.init(P, G, spectral ? &sgen : nullptr);
    WaveBCOut wo;
    // flow_bc first call: zsmean, umean = uu (no hotstart)
    if (!P.hotstart) { CK(cudaMemcpy(hd.umean, hd.uu, N * sizeof(real), cudaMemcpyDeviceToDevice)); CK(cudaMemcpy(hd.vmean, hd.vv, N * sizeof(real), cudaMemcpyDeviceToDevice)); }

    // output
    Output out;
    out.P = &P;
    out.parfile_kv = F.kv;
    out.tidelen = P.tideloc > 0 ? (int)tide_t.size() : 2;
    out.windlen = P.wind && !wind_t.empty() ? (int)wind_t.size() : 1;
    out.init_times();
    if (P.vegetation) { out.veg = &V; out.nveg = V.nsecmax; }
    out.nship = (int)ships.size();
    if (P.sedtrans) { out.sed = &SH.h; out.ngd = P.ngd; out.nd = P.nd; }
    if (P.sedtrans && P.form == 11) out.intra = &ID_;
    out.init(P, G, hd);
    auto set_shipv = [&]() {   // per-ship global output variables, in XBeach units
        if (ships.empty()) return;
        for (const char *nm : {"shipxCG", "shipyCG", "shipzCG", "shipFx", "shipFy", "shipFz", "shipMx", "shipMy", "shipMz"}) out.shipv[nm].clear();
        for (auto &S : ships) {
            double v[9] = {S.shipx, S.shipy, S.shipz, S.Fx, S.Fy, S.Fz, S.Mx, S.My, S.Mz};
            int q = 0;
            for (const char *nm : {"shipxCG", "shipyCG", "shipzCG", "shipFx", "shipFy", "shipFz", "shipMx", "shipMy", "shipMz"}) out.shipv[nm].push_back(v[q++]);
        }
    };
    double t = 0;
    out.update_times(t, P.tstop);
    set_shipv(); out.write(t, 1);
    if (out.outputp && P.ndrifter > 0) out.write_drifters(t, idr, jdr, tdb, tde);
    // vegatt, first call: bulk drag coefficient where Cd < 0
    if (P.vegetation && !P.porcanflow && veg_bulkdrag(V, G)) upload_veg();

    // hotstart writing (writehotstart): the variables hotstart = 1 reads back
    int hs_count = 0;
    auto write_hotstart_all = [&](double tt) {
        hs_count++;
        auto wr = [&](const char *nm, real *src) { std::vector<real> a = D2H(src, N); write_hotstart(nm, hs_count, NX1, NY1, a.data(), false); };
        wr("zs", hd.zs); wr("zb", hd.zb); wr("uu", hd.uu); wr("vv", hd.vv); wr("wb", hd.wb); wr("ws", hd.ws);
        std::vector<int> b = D2H(hd.breaking, N); write_hotstart("breaking", hs_count, NX1, NY1, b.data(), true);
        if (q3) { wr("dU", hd.dU); wr("dV", hd.dV); }
        if (P.gwflow) { wr("gwlevel", hd.gwlevel); wr("gwhead", hd.gwhead); wr("gwcurv", hd.gwcurv); }
        if (P.sedtrans && (P.sus == 1 || P.bulk == 1)) {
            std::vector<real> a = D2H(SH.h.ccg, N * P.ngd); write_hotstart("ccg", hs_count, NX1, NY1, a.data(), false, P.ngd);
        }
        if (P.struct_ == 1) wr("structdepth", SH.h.structdepth);
        wr("umean", hd.umean); wr("vmean", hd.vmean);
        printf("hotstart %06d written at t = %.3f s\n", hs_count, tt); fflush(stdout);
    };

    // ---------------------------------------------------------------------------------------------
    // time loop (libxbeach.F90 executestep)
    // ---------------------------------------------------------------------------------------------
    printf("xbnh_gpu %s %s: nx=%d ny=%d cells=%zu grid=%s nonhq3d=%d cyclic=%d back=%d wbctype=%d hotstart=%d points=%zu\n",
           XBNH_VERSION, sizeof(real) == 8 ? "fp64" : "fp32", nx, ny, N, G.uni ? "uniform" : "curvilinear/variable",
           P.nonhq3d, P.cyclic, P.back, P.wbctype, P.hotstart, P.points.size());
    fflush(stdout);
    FILE *flog = fopen("xbgpu_log.txt", "w");
    long nstep = 0, itsum = 0, itmax = 0, nsolve = 0, nonconv = 0;
    double tsolve = 0, dt = 0, dtref = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<double> hh1(NY1), hu1(NY1);
    std::vector<real> col(NY1);
    auto column1 = [&](const real *a, std::vector<double> &o) {   // a(1,:) (stride NX1)
        CK(cudaMemcpy2D(col.data(), sizeof(real), a, NX1 * sizeof(real), sizeof(real), NY1, cudaMemcpyDeviceToHost));
        for (int j = 0; j < NY1; j++) o[j] = col[j];
    };
    bool first_flow = true, aborted = false;
    // 2D BiCGStab on mat/rhs (inner cells); returns the iteration count, dp = 0 when the right-hand side is zero
    auto bicgstab = [&](int &it) {
        it = 0;
        CK(cudaMemset(hd.sc, 0, 16 * sizeof(double)));
        k_s_init<<<GG, B>>>();
        double sch[16]; CK(cudaMemcpy(sch, hd.sc, sizeof(sch), cudaMemcpyDeviceToHost));
        if (!(sch[8] > 0)) { CK(cudaMemset(hd.dp, 0, N * sizeof(real))); return; }
        double tgt = P.gpu_solver_acc * P.gpu_solver_acc * sch[8];
        CK(cudaMemcpy(hd.sc + 13, &tgt, 8, cudaMemcpyHostToDevice));
        const int batch = 4;
        for (it = 1; it <= P.gpu_maxit; it++) {
            k_s_beta<<<1, 32>>>(it == 1);
            k_s_p<<<GG, B>>>(); k_s_Ay<<<GG, B>>>(); k_s_s<<<GG, B>>>(); k_s_Az<<<GG, B>>>();
            k_s_omega<<<1, 32>>>(); k_s_x<<<GG, B>>>(); k_s_roll<<<1, 32>>>();
            if (it % batch == 0 || it == P.gpu_maxit) {
                double fl[2]; CK(cudaMemcpy(fl, hd.sc + 11, 16, cudaMemcpyDeviceToHost));
                if (fl[0] != 0) { it = (int)fl[1]; return; }
            }
        }
        nonconv++; it = P.gpu_maxit;
    };
    bool gw_initial = true;
    CK(cudaDeviceSynchronize());
    auto loop0 = std::chrono::steady_clock::now();
    while (t < P.tstop) {
        k_wetcells<<<GG, B>>>();
        // compute_dt
        if (nstep == 0) {
            std::vector<real> hh = D2H(hd.hh, N), zsn = D2H(hd.zs, N), uun = D2H(hd.uu, N), vvn = D2H(hd.vv, N);
            double mds = 1e300, mdn = 1e300, m = 0;
            int j0 = oned ? 1 : 2, j1 = oned ? 1 : ny;
            for (int j = j0; j <= j1; j++) for (int i = 2; i <= nx; i++) {
                size_t k = I(i, j);
                mds = std::min(mds, G.dsu[k]); mdn = std::min(mdn, G.dnv[k]);
                double v = P.hotstart ? sqrt(std::max((double)zsn[k] - zb[k], P.eps) * P.g) + sqrt((double)uun[k] * uun[k] + (double)vvn[k] * vvn[k])
                                      : (double)hh[k];
                m = std::max(m, v);
            }
            dt = P.hotstart ? P.CFL * std::min(mds, mdn) / m : P.CFL * std::min(mds, mdn) / sqrt(m * P.g);
            dtref = 0;
            if (out.tnext > dt) { long n = (long)ceil((out.tnext - t) / dt); dt = (out.tnext - t) / n; }
        } else {
            unsigned long long big = (unsigned long long)__builtin_bit_cast(long long, 1e300);
            CK(cudaMemcpy(hd.dtmin, &big, 8, cudaMemcpyHostToDevice));
            k_dt<<<GG, B>>>();
            unsigned long long bits; CK(cudaMemcpy(&bits, hd.dtmin, 8, cudaMemcpyDeviceToHost));
            dt = __builtin_bit_cast(double, (long long)bits) * P.CFL * 0.5;
            dt = std::min(dt, dtgw);
            if (out.tnext > t && (out.tnext - t) / dt < (double)(0.9f * 2147483647.0f)) {
                long n = (long)ceil((out.tnext - t) / dt); dt = (out.tnext - t) / n;
            }
        }
        if (dtref == 0) dtref = dt;
        // timestep
        t += dt;
        if (t >= out.tnext) { dt -= (t - out.tnext); t = out.tnext; }
        double fac = 1.0 / 5000;
        dtref = (1 - fac) * dtref + fac * dt;
        if (((dtref / dt > P.maxdtfac && t < out.tnext) || dt / dtref > P.maxdtfac) && P.defuse == 1) {
            fprintf(stderr, "xbnh_gpu: computational time step implodes/explodes (dt = %.3e s, dtref = %.3e s at t = %.3f s); "
                            "writing the last output and stopping\n", dt, dtref, t);
            aborted = true;
        }
        out.update_times(t, P.tstop);
        if (aborted) {   // timestep.F90: output everything at this time; executestep skips the physics
            out.outputg = !out.tpg.empty(); out.outputp = !out.tpp.empty(); out.outputm = !out.tpm.empty();
            out.outputh = !out.tph.empty(); out.output = true;
            set_shipv(); out.write(t, 0);
            break;
        }
        // wave_bc
        // hh(1,:) and hu(1,:) are read back only when the boundary needs them (params; ts_nonh discharge column),
        // because each read back stalls the GPU pipeline
        if (P.wbctype == WBC_PARAMS) column1(hd.hh, hh1);
        if (P.wbctype == WBC_TSNONH && wbc.bc.iQ >= 0) column1(hd.hu, hu1);
        wbc.step(t, dt, hh1, hu1, wo);
        {
            static std::vector<real> a;
            a.resize(6 * (size_t)NY1);
            const std::vector<double> *src[6] = {&wo.ui, &wo.vi, &wo.zi, &wo.wi, &wo.dui, &wo.dvi};
            for (int q = 0; q < 6; q++) for (int j = 0; j < NY1; j++) a[(size_t)q * NY1 + j] = (real)(*src[q])[j];
            H2D(bc6, a);
        }
        {   // nonh_init_wcoef depends only on zs0 and zb: skip the recomputation (exact) when neither can change
            static const bool varies = !P.zs0file.empty() || P.morphology || P.setbathy;
            if (wo.newwcoef && !q3 && P.dispc <= 0 && varies) init_wcoef();
        }
        // gw_bc
        if (P.gwflow) {
            k_gw_bc1<<<GG, B>>>(); k_gw_bc2<<<bj, 256>>>();
            if (ny > 0) k_gw_bc3<<<bi, 256>>>();
            k_gw_bc4<<<bj, 256>>>();
        }
        // flow_bc: tide
        TideNow T = tide_at(t);
        if (P.tidetype == TT_INSTANT) CK(cudaMemcpy(zs0old, hd.zs0, N * sizeof(real), cudaMemcpyDeviceToDevice));
        k_tide_edges<<<GG, B>>>(T, P.tideloc, P.paulrevere, 0);
        if (P.tideloc == 2 && P.paulrevere == PR_LAND && !h.cyclic) { k_tide_complex<<<1, 1024>>>(T, 3, d_sdist, d_ndist); if (ny > 0) k_tide_complex<<<1, 1024>>>(T, 4, d_sdist, d_ndist); }
        if (P.tideloc == 2 && P.paulrevere == PR_SEA) { k_tide_complex<<<1, 1024>>>(T, 1, d_sdist, d_ndist); k_tide_complex<<<1, 1024>>>(T, 2, d_sdist, d_ndist); k_tide_copy_fb<<<bj, 256>>>(); }
        if (P.tideloc == 4) { for (int id = 1; id <= 4; id++) if (id < 3 || (!h.cyclic && (id < 4 || ny > 0))) k_tide_complex<<<1, 1024>>>(T, id, d_sdist, d_ndist); k_tide_copy_fb<<<bj, 256>>>(); }
        k_tide_wetz<<<GG, B>>>();
        if (P.tidetype == TT_INSTANT) {
            if (ny > 0) k_tide_fill<<<std::max(1, ny - 1), 256>>>(d_sdist);
            TideNow Tp = tide_at(t - dt);
            double dz[4] = {T.zs01 - Tp.zs01, T.zs02 - Tp.zs02, T.zs03 - Tp.zs03, T.zs04 - Tp.zs04};
            k_tide_instant<<<NY1, 256>>>(zs0old, (real)*std::max_element(dz, dz + 4), (real)*std::min_element(dz, dz + 4));
        }
        real factime = (real)(P.epsi < 0 ? 1.0 / 60 * dt : P.epsi);
        int relax = factime > 0 && P.tidetype == TT_VELOCITY;
        if (relax) k_bc_msum<<<1, 1024>>>(0);
        k_bc_mapply<<<bj, 256>>>(factime, 0, relax);
        if (P.wbctype != WBC_OFF) {
            k_bc_front<<<bj, 256>>>(wo.zmode, wo.wmode, (real)wo.fz, wo.isSetV, q3lay ? 1 : 0);
            if (!oned && !h.cyclic) k_bc_front_corners<<<1, 32>>>();
        }
        if (relax) k_bc_msum<<<1, 1024>>>(1);
        k_bc_mapply<<<bj, 256>>>(factime, 1, relax);
        k_bc_back1<<<bj, 256>>>((real)dt);
        k_bc_back2<<<bj, 256>>>();
        if (!oned && !h.cyclic) { k_bc_back_corners<<<1, 32>>>(); k_bc_lat<<<bi, 256>>>(); }
        if (P.wind && wind_t.size() > 1) k_wind<<<GG, B>>>((real)lin_interp(wind_t, wind_x, t), (real)lin_interp(wind_t, wind_y, t));
        cyc({hd.uu, hd.vv, hd.zs, hd.ws});

        // ---- shipwave
        if (P.ships) {
            k_ship_zsvirt<<<GG, B>>>(d_zsv);
            std::map<int, double> acc;   // GRMAP2 sums per grid cell (1-based), in XBeach order
            for (auto &S : ships) {
                const int n2 = (S.nx + 1) * (S.ny + 1);
                S.shipx = lin_interp(S.tt, S.tx, t); S.shipy = lin_interp(S.tt, S.ty, t);
                S.shipz = S.flying ? lin_interp(S.tt, S.tz, t) : 0.0;
                double dir = lin_interp(S.tt, S.tdir, t), cd = cos(dir), sd = sin(dir);
                double bx0 = 1e300, bx1 = -1e300, by0 = 1e300, by1 = -1e300;
                for (int iy = 1; iy <= S.ny + 1; iy++)
                    for (int ix = 1; ix <= S.nx + 1; ix++) {
                        size_t q = (size_t)(iy - 1) * (S.nx + 1) + (ix - 1);
                        S.x[q] = S.shipx + (ix - S.nx / 2 - 1) * S.dx * cd - (iy - S.ny / 2 - 1) * S.dy * sd;
                        S.y[q] = S.shipy + (ix - S.nx / 2 - 1) * S.dx * sd + (iy - S.ny / 2 - 1) * S.dy * cd;
                        bx0 = std::min(bx0, S.x[q]); bx1 = std::max(bx1, S.x[q]); by0 = std::min(by0, S.y[q]); by1 = std::max(by1, S.y[q]);
                    }
                H2D(S.d_x2, S.x); H2D(S.d_y2, S.y);
                // MKMAP: candidates on the GPU, ipon and bilin5 on the host
                int cnt = 0;
                while (true) {
                    if (pair_cap < 16 * n2 || pair_cap < cnt) {
                        if (d_pairs) cudaFree(d_pairs);
                        pair_cap = std::max(16 * n2, 2 * cnt); d_pairs = dalloc<int>(2 * (size_t)pair_cap);
                    }
                    CK(cudaMemset(d_count, 0, sizeof(int)));
                    k_ship_cand<<<GG, B>>>(d_xz, d_yz, S.d_x2, S.d_y2, n2, bx0, bx1, by0, by1, d_pairs, pair_cap, d_count);
                    CK(cudaMemcpy(&cnt, d_count, sizeof(int), cudaMemcpyDeviceToHost));
                    if (cnt <= pair_cap) break;
                }
                std::vector<int> pr = cnt ? D2H(d_pairs, 2 * (size_t)cnt) : std::vector<int>();
                std::vector<int> win(n2, 0);
                for (int a = 0; a < cnt; a++) {
                    int L = pr[2 * a], q = pr[2 * a + 1];
                    if (L <= win[q]) continue;
                    int i1 = (L - 1) % NX1 + 1, j1 = (L - 1) / NX1 + 1;
                    size_t k1 = I(i1, j1), k2 = I(i1 + 1, j1), k3 = I(i1 + 1, j1 + 1), k4 = I(i1, j1 + 1);
                    double xp[4] = {G.xz[k1], G.xz[k2], G.xz[k3], G.xz[k4]}, yp[4] = {G.yz[k1], G.yz[k2], G.yz[k3], G.yz[k4]};
                    if (ship_ipon(xp, yp, S.x[q], S.y[q]) >= 0) win[q] = L;
                }
                std::fill(S.iref.begin(), S.iref.end(), 0); std::fill(S.w.begin(), S.w.end(), 0.0);
                for (int q = 0; q < n2; q++) {
                    int L = win[q];
                    if (!L) continue;
                    int i1 = (L - 1) % NX1 + 1, j1 = (L - 1) / NX1 + 1;
                    size_t k1 = I(i1, j1), k2 = I(i1 + 1, j1), k3 = I(i1 + 1, j1 + 1), k4 = I(i1, j1 + 1);
                    double xp[4] = {G.xz[k1], G.xz[k2], G.xz[k3], G.xz[k4]}, yp[4] = {G.yz[k1], G.yz[k2], G.yz[k3], G.yz[k4]};
                    double hb[4] = {0, 0, 0, 0};
                    if (ship_bilin5(xp, yp, S.x[q], S.y[q], hb)) for (int c4 = 0; c4 < 4; c4++) S.w[4 * q + c4] = (double)(float)hb[c4];
                    S.iref[4 * q] = (int)k1 + 1; S.iref[4 * q + 1] = (int)k2 + 1; S.iref[4 * q + 2] = (int)k3 + 1; S.iref[4 * q + 3] = (int)k4 + 1;
                }
                // GRMAP: zs + ph at the ship points (unmapped points keep their value)
                if (!d_sidx || (int)S.iref.size() > pair_cap) {
                    if (d_sidx) { cudaFree(d_sidx); cudaFree(d_sbuf); }
                    size_t m = std::max((size_t)pair_cap, S.iref.size()); d_sidx = dalloc<int>(m); d_sbuf = dalloc<double>(m);
                }
                CK(cudaMemcpy(d_sidx, S.iref.data(), S.iref.size() * sizeof(int), cudaMemcpyHostToDevice));
                k_ship_gather<<<(unsigned)((S.iref.size() + 255) / 256), 256>>>(d_zsv, d_sidx, d_sbuf, (int)S.iref.size());
                std::vector<double> fv = D2H(d_sbuf, S.iref.size());
                for (int q = 0; q < n2; q++) {
                    if (S.iref[4 * q] <= 0) continue;
                    double f2 = 0;
                    for (int c4 = 0; c4 < 4; c4++) f2 = f2 + S.w[4 * q + c4] * fv[4 * q + c4];
                    S.zs[q] = f2;
                }
                // hull pressure head
                if (S.compute_force) {
                    for (int q = 0; q < n2; q++) S.ph[q] = S.zs[q] - S.zhull[q];
                    ship_force(S, P);
                    if (S.compute_motion)   // the ship motion angles (shipchi, shipphi) are never computed in XBeach: 0
                        for (int q = 0; q < n2; q++) S.zhull[q] = S.shipz - S.zCG - S.depth[q] - (S.x[q] - S.xCG) * sin(0.0) + (S.y[q] - S.yCG) * sin(0.0);
                    for (int q = 0; q < n2; q++) { S.ph[q] = std::max(-S.zhull[q], 0.0); if (S.depth[q] == 0) S.ph[q] = 0; }
                } else
                    for (int q = 0; q < n2; q++) S.ph[q] = std::max(-S.zhull[q] - S.shipz, 0.0);
                // GRMAP2: area-weighted mapping back to the grid
                const double cz2 = S.dx * S.dy;
                for (int c4 = 0; c4 < 4; c4++)
                    for (int q = 0; q < n2; q++) {
                        int i1 = S.iref[4 * q + c4];
                        if (i1 > 0) acc[i1] = acc[i1] + S.w[4 * q + c4] * S.ph[q] * cz2 * G.dsdnzi[i1 - 1];
                    }
            }
            if (!acc.empty()) {
                std::vector<int> ix; std::vector<double> vx;
                for (auto &kv : acc) { ix.push_back(kv.first - 1); vx.push_back(kv.second); }
                int *di = dalloc<int>(ix.size()); double *dv = dalloc<double>(vx.size());
                H2D(di, ix); H2D(dv, vx);
                k_ship_scatter<<<(unsigned)((ix.size() + 255) / 256), 256>>>(di, dv, (int)ix.size());
                cudaFree(di); cudaFree(dv);
            }
            if (nstep == 0) k_ship_first<<<GG, B>>>();
        }
        // ---- vegatt
        if (P.vegetation) {
            if (P.porcanflow) { k_porcan<<<GG, B>>>((real)dt); k_porcan_uv<<<GG, B>>>(); }
            else k_momeqveg<<<GG, B>>>();
        }
        // ---- gwflow
        if (P.gwflow) {
            k_gw_a<<<GG, B>>>((real)dt, (real)std::min(4 * dt / P.Trep, 1.0));
            k_gw_hu<<<GG, B>>>(gw_initial ? 1 : 0); gw_initial = false;
            if (P.gwnonh == 0) k_gw_head_hyd<<<GG, B>>>();
            else if (ny < 3) {
                k_gw_tri_build<<<(nx + 255) / 256, 256>>>(1); k_gw_thomas<<<1, 1>>>(1); k_gw_tri_after<<<bi, 256>>>(1);
            } else {
                int itg = 0;
                if (P.gwheadmodel == 0) { k_gw_sip_build<<<GG, B>>>(); bicgstab(itg); k_gw_sip_after<<<GG, B>>>(1); }
                k_gw_head2d<<<GG, B>>>();
            }
            k_gw_vel<<<GG, B>>>(); k_gw_ksmooth<<<GG, B>>>(); k_gw_q<<<GG, B>>>(); k_gw_w<<<GG, B>>>();
            if (P.gwnonh) k_gw_infilcon<<<GG, B>>>((real)dt);
            else {
                k_gw_hyd1<<<GG, B>>>((real)dt); k_gw_hyd_bci<<<bj, 256>>>();
                if (ny > 0) k_gw_hyd_bcj<<<bi, 256>>>();
                k_gw_hyd2<<<GG, B>>>((real)dt);
                if (P.gwhorinfil) k_gw_hor<<<GG, B>>>((real)dt);
            }
            k_gw_level<<<GG, B>>>((real)dt);
            k_gw_bnd_i<<<bj, 256>>>();
            if (ny > 0) k_gw_bnd_j<<<bi, 256>>>();
            k_gw_headb<<<GG, B>>>();
        }

        // ---- flow
        if (first_flow) { k_rough<<<GG, B>>>(1, 0, 1, 0); first_flow = false; }
        for (auto &D : disch) if (D.pnt == 1) {
            int m1 = std::min(std::max(D.m1, 1), nx), n1 = std::min(std::max(D.n1, 1), std::max(ny, 1));
            k_disch_v<<<1, 32>>>(m1, n1, (real)(qdisch(D, t) * dt));
        }
        k_slopes<<<GG, B>>>();
        {   // bedroughness_update; dtold(1) of the acceleration effect is the dt of two calls back (XBeach)
            static double dto[2] = {-1, -1};
            if (dto[0] < 0) dto[0] = dto[1] = dt;
            double fa = std::min(dto[0] / (P.Trep / 20), 1.0), iom = 1.0 / (2 * M_PI / P.Trep);
            k_rough<<<GG, B>>>(0, (real)fa, (real)dto[0], (real)iom);
            if (P.friction_acceleration) { dto[0] = dto[1]; dto[1] = dt; }
        }
        k_advection<<<GG, B>>>();
        if (P.viscosity == 0) k_visc_off<<<GG, B>>>();
        else {
            if (P.smag) {
                k_smag<<<GG, B>>>();
                cyc({hd.nuh});
                if (!oned && !h.cyclic) k_nuh_bc_j<<<bi, 256>>>();
                k_nuh_bc_i<<<bj, 256>>>();
            }
            k_nuh_const_brk<<<GG, B>>>();
            k_viscu<<<GG, B>>>();
            k_viscv1<<<GG, B>>>();
            if (!oned && !h.cyclic) k_viscv_bc<<<bi, 256>>>();
            k_viscv2<<<GG, B>>>();
            k_nuh_scale<<<GG, B>>>();
        }
        k_momentum<<<GG, B>>>((real)dt);
        if (!oned && !h.cyclic) { k_lat_uv<<<bi, 256>>>((real)dt); k_lat_v_apply<<<bi, 256>>>(); }
        cyc({hd.uu, hd.vv});
        // nonh_cor(0): predictor
        k_nonhUV<<<GG, B>>>();
        k_break<<<GG, B>>>(); k_break_apply<<<GG, B>>>();
        cyc({hd.pres}, {hd.breaking, hd.nonhZ, hd.nonhU, hd.nonhV});
        if (!q3) {
            k_wm_adv<<<GG, B>>>((real)dt); k_wm_zero<<<GG, B>>>();
            cyc({hd.Wm_old, hd.ws, hd.wb, hd.pres});
            k_wm_visc_coef<<<GG, B>>>((real)dt);
            cyc({hd.Wm});
            if (P.secorder) {
                k_pred_pres<<<GG, B>>>((real)dt);
                cyc({hd.uu, hd.vv, hd.Wm});
                k_advW_flux<<<GG, B>>>(hd.Wm, hd.Wm_old); k_advW_apply<<<GG, B>>>(hd.Wm, (real)dt);
                cyc({hd.Wm});
            }
        } else {
            k_zuzv<<<GG, B>>>();
            if (!oned) {
                k_q3_coef<<<GG, B>>>((real)dt); k_q3_coef_bc<<<GG, B>>>();
                k_q3_wm_adv<<<GG, B>>>((real)dt); k_q3_wm_set<<<GG, B>>>();
                cyc({hd.Wm, hd.Wm_old, hd.pres});
                if (q3lay) { k_q3_dudv_adv<<<GG, B>>>((real)dt); cyc({hd.dU, hd.dV}); }
            } else {
                k_q1_coef<<<bi, 256>>>((real)dt);
                k_q1_adv<<<bi, 256>>>((real)dt); k_q3_wm_set<<<GG, B>>>();
            }
            if (P.secorder) {
                k_q3_pred_pres<<<GG, B>>>();
                cyc({hd.uu, hd.vv, hd.Wm, hd.dU, hd.dV});
                k_advW_flux<<<GG, B>>>(hd.Wm, hd.Wm_old); k_advW_apply<<<GG, B>>>(hd.Wm, (real)dt);
                cyc({hd.Wm});
            }
            if (q3lay) { if (!oned) k_q3_fric<<<GG, B>>>((real)dt); else k_q1_fric<<<bi, 256>>>((real)dt); cyc({hd.dU, hd.dV}); }
        }
        // flow_secondorder_advUV
        if (P.secorder) {
            k_advUV_flux<<<GG, B>>>(); k_advUV_apply<<<GG, B>>>((real)dt);
            cyc({hd.uu, hd.vv});
        }
        // nonh_cor(1): corrector
        if (!q3) { k_zuzv<<<GG, B>>>(); k_build1<<<GG, B>>>((real)dt); }
        else if (!oned) k_build3<<<GG, B>>>();
        else k_build3_1d<<<bi, 256>>>();
        cudaEventRecord(e0);
        int it = 0;
        if (oned) {
            k_pcr<<<1, 1024>>>(NX1);
        } else bicgstab(it);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); tsolve += ms * 1e-3;
        itsum += it; itmax = std::max(itmax, (long)it); nsolve++;
        cyc({hd.dp});
        if (!q3) {
            k_correct1<<<GG, B>>>();
            cyc({hd.uu, hd.vv, hd.pres});
            CK(cudaMemsetAsync(hd.skey, 0, 2 * sizeof(unsigned long long)));
            k_stale_find<<<GG, B>>>(); k_stale_val<<<1, 32>>>();
            k_wswb1<<<GG, B>>>();
            cyc({hd.ws, hd.wb});
            if (!oned && !h.cyclic) k_wswb_bc_j<<<bi, 256>>>();
            k_wswb_bc_i<<<bj, 256>>>(); k_wmold1<<<GG, B>>>();
        } else {
            k_correct3<<<GG, B>>>();
            cyc({hd.uu, hd.vv, hd.pres, hd.dU, hd.dV});
            if (!oned) k_q3_after<<<GG, B>>>(1); else k_q1_after<<<bi, 256>>>();
            cyc({hd.ws, hd.wb, hd.Wm, hd.omega, hd.Wm0});
            k_q3_wm_bc<<<bj, 256>>>();
            if (!oned && !h.cyclic) k_q3_wm_bc_j<<<bi, 256>>>();
            k_q3_roll<<<GG, B>>>();
        }
        // hu, hv, fluxes, discharges, continuity, derived fields
        k_huhv<<<GG, B>>>();
        for (auto &D : disch) if (D.pnt == 0) {
            double qn = qdisch(D, t);
            if (D.n1 == D.n2 && !oned) {   // along v points
                int m1 = std::min(std::max(D.m1, 1), nx), m2 = std::min(std::max(D.m2, 1), nx), n1 = std::min(std::max(D.n1, 1), ny);
                int isb = !h.cyclic && (n1 == 1 || n1 == ny), sh = 0;
                if (isb) { if (n1 > 1) { qn = -qn; sh = 0; } else sh = 1; }
                k_disch_h<<<1, 256>>>(1, std::max(m1, 1), std::min(m2, NX1), n1, isb, sh, (real)qn, 0);
            } else if (D.m1 == D.m2) {    // along u points
                int m1 = std::min(std::max(D.m1, 1), nx), n1 = std::min(std::max(D.n1, 1), std::max(ny, 1)), n2 = std::min(std::max(D.n2, 1), std::max(ny, 1));
                int isb = (m1 == 1 || m1 == nx), sh = 0;
                if (isb) { if (m1 > 1) { qn = -qn; sh = 0; } else sh = 1; }
                k_disch_h<<<1, 256>>>(0, n1, n2, m1, isb, sh, (real)qn, 0);
            }
        }
        if (P.rainfall && !rain_t.empty()) {
            size_t q = std::upper_bound(rain_t.begin(), rain_t.end(), t) - rain_t.begin();
            if (q == 0) k_rain<<<GG, B>>>(rain_maps[0], rain_maps[0], 0);
            else if (q >= rain_t.size()) k_rain<<<GG, B>>>(rain_maps.back(), rain_maps.back(), 0);
            else k_rain<<<GG, B>>>(rain_maps[q - 1], rain_maps[q], (real)((t - rain_t[q - 1]) / (rain_t[q] - rain_t[q - 1])));
        }
        k_continuity<<<GG, B>>>((real)dt);
        cyc({hd.zs});
        k_ve_centre<<<GG, B>>>();
        if (P.sedtrans) CK(cudaMemcpy(SH.h.hold, hd.hh, N * sizeof(real), cudaMemcpyDeviceToDevice));   // s%hold
        k_post<<<GG, B>>>();
        k_uv_centre<<<GG, B>>>();
        if (P.ndrifter > 0) k_drift<<<1, 64>>>(d_idr, d_jdr, d_tdb, d_tde, P.ndrifter, t, (real)dt);
        // ---- transus
        if (P.sedtrans) {
            if (SH.k.Trep != (real)P.Trep) { SH.k.Trep = (real)P.Trep; CK(cudaMemcpyToSymbol(cs, &SH.k, sizeof(SedConst))); }
            k_turb1<<<GG, B>>>((real)dt); k_turb2<<<GG, B>>>((real)dt);
            k_turb_bci<<<bj, 256>>>(); if (!oned) k_turb_bcj<<<bi, 256>>>();
            if (gravel) {   // Nielsen2006 / mccall_vanrijn: bed load only, transus returns (bulk = 0)
                CK(cudaMemset(SH.h.dcsdx, 0, N * sizeof(real))); CK(cudaMemset(SH.h.dcsdy, 0, N * sizeof(real)));
                if (P.form == 3) {
                    if (grav_first) { CK(cudaMemcpy(GD.ulocal, hd.uu, N * sizeof(real), cudaMemcpyDeviceToDevice));
                                      CK(cudaMemcpy(GD.ulocalold, hd.uu, N * sizeof(real), cudaMemcpyDeviceToDevice)); }
                    k_niel_state<<<GG, B>>>((real)dt);
                    for (int jg = 0; jg < P.ngd; jg++) { k_niel_q<<<GG, B>>>(jg); k_grav_sub<<<GG, B>>>(jg, 0); }
                } else for (int jg = 0; jg < P.ngd; jg++) { k_mccall_q<<<GG, B>>>(jg, P.form); k_grav_sub<<<GG, B>>>(jg, 1); }
                grav_first = false;
            } else {
                if (P.form == 11) for (int jg = 0; jg < P.ngd; jg++) { k_intra1<<<GG, B>>>(jg); k_intra2<<<GG, B>>>(jg); }
                else if (P.form <= 2) k_sedtransform<<<GG, B>>>((real)dt);
                k_transus_fac<<<GG, B>>>((real)dt);
                for (int jg = 0; jg < P.ngd; jg++) {
                    k_transus_flux<<<GG, B>>>(jg); k_transus_cc<<<GG, B>>>(jg, (real)dt);
                    k_transus_bc<<<GG, B>>>(jg); k_transus_end<<<GG, B>>>(jg);
                }
            }
        }
        // ---- bed_update
        if (P.morphology && !P.setbathy) {
            k_bed_reset<<<GG, B>>>();
            if (t >= P.morstart && t < P.morstop && P.morfac > .999) {
                if (P.struct_ == 1) for (int jg = 0; jg < P.ngd; jg++) { k_bed_sout<<<GG, B>>>(jg, (real)dt, hd.w4); k_bed_slimit<<<GG, B>>>(jg, hd.w4); }
                k_bed_level<<<GG, B>>>((real)dt);
                k_bed_zs<<<GG, B>>>((real)dt, P.gwflow);
                if (P.avalanching) {
                    double avaltime = P.fixedavaltime == 0 ? P.nTrepavaltime * P.Trep : P.avaltime;
                    double Sig = (double)0.078f * sqrt(P.Hrms * P.g * (P.Trep * P.Trep) / 2 / M_PI);
                    for (int ii = 1; ii <= (int)lround(P.morfac); ii++) {
                        k_bed_slopes<<<GG, B>>>(1);
                        CK(cudaMemset(SH.h.naval, 0, sizeof(int)));
                        k_aval_x<<<bj, 256>>>((real)dt, (real)avaltime, (real)Sig);
                        if (!oned) k_aval_y<<<bi, 256>>>((real)dt, (real)avaltime, (real)Sig);
                        int na; CK(cudaMemcpy(&na, SH.h.naval, sizeof(int), cudaMemcpyDeviceToHost));
                        if (na == 0) break;
                    }
                } else k_bed_slopes<<<GG, B>>>(0);
                if (!oned && !h.cyclic) k_bed_lat<<<bi, 256>>>();
                if (P.ngd > 1) k_bed_d50top<<<GG, B>>>(nx, std::max(ny, 1));
            }
        }
        if (P.setbathy) {   // setbathy_update (linear_interp in time between the bracketing beds)
            int jj = xb_binary_search(sb_t.data(), (int)sb_t.size(), t), ns = (int)sb_t.size();
            static int jlast = -999; static real *z0d = nullptr, *z1d = nullptr;
            if (!z0d) { z0d = R(); z1d = R(); }
            if (jj != jlast) {
                int a = jj <= 0 ? 0 : (jj >= ns ? ns - 1 : jj - 1), b = std::min(a + 1, ns - 1);
                H2D(z0d, toreal(sb_z[a])); H2D(z1d, toreal(sb_z[b])); jlast = jj;
            }
            if (ns <= 1 || jj <= 0 || jj >= ns) k_setbathy<<<GG, B>>>(z0d, z1d, t, 0, 0, 0);
            else k_setbathy<<<GG, B>>>(z0d, z1d, t, sb_t[jj - 1], sb_t[jj], 1);
        }
        nstep++;
        // output (outputext)
        out.accumulate(t, dt);
        if (out.output) {
            set_shipv(); out.write(t, 0);
            if (out.outputp && P.ndrifter > 0) {
                idr = D2H(d_idr, P.ndrifter); jdr = D2H(d_jdr, P.ndrifter);
                out.write_drifters(t, idr, jdr, tdb, tde);
            }
            if (out.outputh && P.writehotstart) write_hotstart_all(t);
        }
        if (aborted) break;
        if (nstep % 2000 == 0) {
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
            printf("t = %8.2f s  steps %ld  dt %.4f  wall %.1f s  solver it mean %.1f\n", t, nstep, dt, el, (double)itsum / std::max(nsolve, 1L));
            fflush(stdout);
        }
    }
    CK(cudaDeviceSynchronize());
    out.close();
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    double loop = std::chrono::duration<double>(std::chrono::steady_clock::now() - loop0).count();
    // wall_s: whole run; loop_s: the time loop, as XBeach reports it; unit speed from the time loop
    fprintf(flog, "version %s\nprecision %s\nnx %d\nny %d\ncells %zu\ncyclic %d\nnonhq3d %d\nsteps %ld\nwall_s %.3f\nloop_s %.3f\n"
                  "solver_s %.3f\nsolver_it_mean %.2f\nsolver_it_max %ld\nsolver_nonconv %ld\nunit_speed_s %.4e\naborted %d\n",
            XBNH_VERSION, sizeof(real) == 8 ? "fp64" : "fp32", nx, ny, N, h.cyclic, P.nonhq3d, nstep, wall, loop, tsolve,
            (double)itsum / std::max(nsolve, 1L), itmax, nonconv, loop / ((double)std::max(nstep, 1L) * N), aborted ? 1 : 0);
    fclose(flog);
    printf("done: %ld steps, wall %.2f s (time loop %.2f s, solver %.2f s), %.3e s per cell-step, solver it mean %.1f max %ld nonconv %ld\n",
           nstep, wall, loop, tsolve, loop / ((double)std::max(nstep, 1L) * N), (double)itsum / std::max(nsolve, 1L), itmax, nonconv);
    return aborted ? 2 : 0;
}
