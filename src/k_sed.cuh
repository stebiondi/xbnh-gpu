// k_sed.cuh: morphevolution.F90 transus for wavemodel = nonh (sedtrans = 1): waveturb, sedtransform
// (soulsby_vanrijn, vanthiel_vanrijn, vanrijn1993) and the advection-diffusion of the suspended concentration;
// initialize.F90 sed_init. In nonh the short-wave terms vanish (urms = 0, ua = 0, DR = 0, sws = 0) and XBeach keeps
// the bed-load fluxes only for surfbeat (Subg = Svbg = 0), so the bed-load branch of transus has no effect and is
// left out; the bed-load equilibrium concentration ceqbg is still computed (output and the struct limiter).
#pragma once

#define NGMAX 8
#define NDMAX 24

struct SedConst {
    int ngd, nd, form, sus, bed, bulk, bdslpeffmag, bdslpeffini, bdslpeffdir, lwt, fallvelred, dilatancy;
    int bermslopetransport, bermslopesus, sourcesink, oldTsmin, kmax, nz, nd_var, struct_, turbadv, morph;
    real morfac, por, rhos, delta, thetanum, cmax, facsl, z0, smax, bdslpeffdirfac, phi, tsfac, Tsmin, dtlimTs, facDc;
    real betad, beta, turbcoef, max_dzsdt, alfaD50, bermslope, bermslopefac, bermslopedepth, rheeA, pormax, kvis, Trep;
    real vonkar, deltar, rwave, umin, wetslp, dryslp, hswitch, lsgrad, frac_dz, split, merge;
    real D50[NGMAX], D90[NGMAX], D15[NGMAX], ws0[NGMAX], dster[NGMAX], shieldscrit[NGMAX], sedcal[NGMAX];
};
__constant__ SedConst cs;

struct SedDev {
    real *ccg, *ceqsg, *ceqbg, *Tsg, *Susg, *Svsg, *Subg, *Svbg, *ero, *depo_ex, *fac, *wfall;   // N * ngd
    real *pbbed;                                                                                   // N * nd * ngd
    real *dzbed;                                                                                   // N * nd
    real *z0bed, *structdepth, *D50top, *D90top, *sedero, *dzbnow, *dzbdt, *dzav, *dzbdx, *dzbdy;
    real *kturb, *rolthick, *kb, *Dc, *vmag, *ureps, *urepb, *vreps, *vrepb, *dcsdx, *dcsdy, *hold;
    real *vmagu, *vmagv, *Sturbu, *Sturbv, *ccz, *refA, *ca, *favc;
    int *naval;
};
__constant__ SedDev sd;

#define NN ((size_t)c.NX1 * c.NY1)
#define G3(a, k, jg) (a)[(size_t)(jg) * NN + (k)]                         // (i,j,jg)
#define PB(k, m, jg) sd.pbbed[((size_t)(m) + (size_t)cs.nd * (jg)) * NN + (k)]   // pbbed(i,j,m,jg), 0-based m, jg

// cell-centre Eulerian velocities as XBeach stores them (ue: current ueu; ve: lagged, see k_ve_centre)
__device__ __forceinline__ real ue_c(int i, int j) {
    if (i == c.NX1) return 0;
    return i == 1 ? d.ueu[ID(1, j)] : (real)0.5 * (d.ueu[ID(i - 1, j)] + d.ueu[ID(i, j)]);
}

// ---------------------------------------------------------------------------------------------
// host: sed_init and the constants of sedtransform's first call
// ---------------------------------------------------------------------------------------------
struct SedHost {
    SedDev h{};
    SedConst k{};
    std::vector<double> setbathy_t;
    std::vector<std::vector<double>> setbathy_z;
};

static void sed_init(const Params &P, const Grid &G, const std::vector<double> &zb, SedHost &S,
                     std::function<real *(real)> R) {
    const int NX1 = G.NX1, NY1 = G.NY1, ngd = P.ngd, nd = P.nd;
    const size_t N = (size_t)NX1 * NY1;
    if (ngd > NGMAX) fail("ngd > " + std::to_string(NGMAX) + " is not supported");
    if (nd > NDMAX) fail("nd > " + std::to_string(NDMAX) + " is not supported");
    SedConst &k = S.k;
    k.ngd = ngd; k.nd = nd; k.form = P.form; k.sus = P.sus; k.bed = P.bed; k.bulk = P.bulk; k.bdslpeffmag = P.bdslpeffmag;
    k.bdslpeffini = P.bdslpeffini; k.bdslpeffdir = P.bdslpeffdir; k.lwt = P.lwt; k.fallvelred = P.fallvelred;
    k.dilatancy = P.dilatancy; k.bermslopetransport = P.bermslopetransport; k.bermslopesus = P.bermslopesus;
    k.sourcesink = P.sourcesink; k.oldTsmin = P.oldTsmin; k.kmax = P.kmax; k.nz = P.nz; k.struct_ = P.struct_;
    k.turbadv = P.turbadv; k.morph = P.morphology;
    k.morfac = P.morfac; k.por = P.por; k.rhos = P.rhos; k.delta = (P.rhos - P.rho) / P.rho; k.thetanum = P.thetanum;
    k.cmax = P.cmax; k.facsl = P.facsl; k.z0 = P.z0; k.smax = P.smax; k.bdslpeffdirfac = P.bdslpeffdirfac;
    k.phi = P.reposeangle / 180 * M_PI; k.tsfac = P.tsfac; k.Tsmin = P.Tsmin; k.dtlimTs = P.dtlimTs; k.facDc = P.facDc;
    k.betad = P.betad; k.beta = P.beta; k.turbcoef = P.turbcoef; k.max_dzsdt = P.max_dzsdt; k.alfaD50 = P.alfaD50;
    k.bermslope = P.bermslope; k.bermslopefac = P.bermslopefac; k.bermslopedepth = P.bermslopedepth; k.rheeA = P.rheeA;
    k.pormax = P.pormax; k.vonkar = P.vonkar; k.deltar = P.deltar; k.rwave = P.rwave; k.umin = P.umin;
    k.wetslp = P.wetslp; k.dryslp = P.dryslp; k.hswitch = P.hswitch; k.lsgrad = P.lsgrad; k.frac_dz = P.frac_dz;
    k.split = P.split; k.merge = P.merge; k.Trep = P.Trep;
    // sedtransform first call: fall velocity (Van Rijn 1993), dimensionless grain size, critical Shields
    double Te = 20.0, kvis = 4.0 / (20.0 + Te) * 1e-5;
    k.kvis = kvis;
    for (int jg = 0; jg < ngd; jg++) {
        double D50 = P.D50[jg];
        double Sster = D50 / (4 * kvis) * sqrt((P.rhos / P.rho - 1) * P.g * D50);
        double cc1 = 1.06 * tanh(0.064 * Sster * exp(-7.5 / (Sster * Sster)));
        double cc2 = 0.22 * tanh(2.34 * pow(Sster, -1.18) * exp(-0.0064 * (Sster * Sster)));
        double wster = cc1 + cc2 * Sster;
        double ws0 = wster * sqrt((P.rhos / P.rho - 1.0) * P.g * D50);
        double delta = (P.rhos - P.rho) / P.rho;
        double dster = pow(delta * P.g / 1e-12, 1.0 / 3.0) * D50;
        k.D50[jg] = D50; k.D90[jg] = P.D90[jg]; k.D15[jg] = P.D15.empty() ? 0 : P.D15[jg];
        k.ws0[jg] = ws0; k.dster[jg] = dster;
        k.shieldscrit[jg] = 0.3 / (1 + 1.2 * dster) + 0.055 * (1 - exp(-0.02 * dster));
        k.sedcal[jg] = P.sedcal[jg];
    }
    // bed composition
    std::vector<double> pb((size_t)N * nd * ngd, 0.0), dz((size_t)N * nd, 0.0);
    int nd_var = P.nd_var;
    if (ngd == 1) {
        for (size_t q = 0; q < N * nd; q++) pb[q] = 1.0;
        nd_var = 2;
        for (size_t kk = 0; kk < N; kk++)
            for (int m = 0; m < nd; m++)
                dz[(size_t)m * N + kk] = m < nd_var - 1 ? std::max(P.dzg1, 10.0) : (m == nd_var - 1 ? std::max(P.dzg2, 10.0) : std::max(P.dzg3, 10.0));
    } else {
        for (int jg = 0; jg < ngd; jg++) {
            char fn[32]; snprintf(fn, sizeof fn, "gdist%d.inp", jg + 1);
            Tok t(fn);
            for (int m = 0; m < nd; m++)
                for (int j = 0; j < NY1; j++) {
                    for (int i = 0; i < NX1; i++) {
                        double v; if (!t.num(v)) fail(std::string(fn) + ": too few values");
                        pb[((size_t)m + (size_t)nd * jg) * N + (size_t)j * NX1 + i] = v;
                    }
                    t.nextline();
                }
        }
        for (int m = 0; m < nd; m++)
            for (size_t kk = 0; kk < N; kk++) {
                double s = 0; for (int jg = 0; jg < ngd; jg++) s += pb[((size_t)m + (size_t)nd * jg) * N + kk];
                if (fabs(1.0 - s) > 0.000001) {
                    for (int jg = 0; jg < ngd; jg++) {
                        double &v = pb[((size_t)m + (size_t)nd * jg) * N + kk];
                        v = s <= 2.2250738585072014e-308 ? 1.0 / ngd : v / s;
                    }
                }
            }
        for (size_t kk = 0; kk < N; kk++)
            for (int m = 0; m < nd; m++)
                dz[(size_t)m * N + kk] = m < nd_var - 1 ? P.dzg1 : (m == nd_var - 1 ? P.dzg2 : P.dzg3);
    }
    k.nd_var = nd_var;
    std::vector<double> D50top(N), D90top(N), z0bed(N), sdepth(N, 100.0);
    for (size_t kk = 0; kk < N; kk++) {
        double a = 0, b = 0;
        for (int jg = 0; jg < ngd; jg++) { a += pb[(size_t)nd * jg * N + kk] * P.D50[jg]; b += pb[(size_t)nd * jg * N + kk] * P.D90[jg]; }
        D50top[kk] = a; D90top[kk] = b;
        double sz = 0; for (int m = 0; m < nd; m++) sz += dz[(size_t)m * N + kk];
        z0bed[kk] = zb[kk] - sz;
    }
    if (P.struct_ == 1 && P.hotstart == 0) read_grid_file(P.ne_layer, NX1, NY1, sdepth, true);
    if (P.struct_ == 1 && P.hotstart == 1) read_hotstart("structdepth", P.hotstartfileno, NX1, NY1, sdepth, false);
    // device arrays
    SedDev &h = S.h;
    auto RL = [&](size_t n) { real *p; CK(cudaMalloc(&p, n * sizeof(real))); CK(cudaMemset(p, 0, n * sizeof(real))); return p; };
    for (real **a : {&h.ccg, &h.ceqsg, &h.ceqbg, &h.Tsg, &h.Susg, &h.Svsg, &h.Subg, &h.Svbg, &h.ero, &h.depo_ex, &h.fac, &h.wfall})
        *a = RL(N * ngd);
    h.pbbed = RL(N * nd * ngd); h.dzbed = RL(N * nd);
    for (real **a : {&h.z0bed, &h.structdepth, &h.D50top, &h.D90top, &h.sedero, &h.dzbnow, &h.dzbdt, &h.dzav, &h.dzbdx, &h.dzbdy,
                     &h.kturb, &h.rolthick, &h.kb, &h.Dc, &h.vmag, &h.ureps, &h.urepb, &h.vreps, &h.vrepb, &h.dcsdx, &h.dcsdy,
                     &h.hold, &h.vmagu, &h.vmagv, &h.Sturbu, &h.Sturbv, &h.refA, &h.ca, &h.favc})
        *a = R(0);
    h.ccz = RL(N * std::max(P.kmax, 1));
    CK(cudaMalloc(&h.naval, sizeof(int)));
    auto up = [&](real *dst, const std::vector<double> &v) { std::vector<real> r(v.begin(), v.end()); CK(cudaMemcpy(dst, r.data(), r.size() * sizeof(real), cudaMemcpyHostToDevice)); };
    up(h.pbbed, pb); up(h.dzbed, dz); up(h.z0bed, z0bed); up(h.structdepth, sdepth); up(h.D50top, D50top); up(h.D90top, D90top);
    // hotstart concentrations
    if (P.hotstart && (P.sus == 1 || P.bulk == 1)) {
        std::vector<double> c3; read_hotstart("ccg", P.hotstartfileno, NX1, NY1, c3, false, ngd); up(h.ccg, c3);
    }
    // initial bed slopes (sed_init)
    std::vector<double> dzx(N, 0.0), dzy(N, 0.0);
    for (int j = 1; j <= NY1; j++) {
        for (int i = 1; i <= G.nx; i++) dzx[G.I(i, j)] = (zb[G.I(i + 1, j)] - zb[G.I(i, j)]) / G.dsu[G.I(i, j)];
        dzx[G.I(NX1, j)] = dzx[G.I(G.nx, j)];
    }
    if (G.ny > 0) {
        for (int i = 1; i <= NX1; i++) {
            for (int j = 1; j <= G.ny; j++) dzy[G.I(i, j)] = (zb[G.I(i, j + 1)] - zb[G.I(i, j)]) / G.dnv[G.I(i, j)];
            dzy[G.I(i, NY1)] = dzy[G.I(i, G.ny)];
        }
    }
    up(h.dzbdx, dzx); up(h.dzbdy, dzy);
    CK(cudaMemcpyToSymbol(cs, &k, sizeof(SedConst)));
    CK(cudaMemcpyToSymbol(sd, &h, sizeof(SedDev)));
    printf("sediment: form %d, ngd %d, nd %d, D50 %.5f m, morphology %d, morfac %.2f%s\n", P.form, ngd, nd, P.D50[0], P.morphology,
           P.morfac, P.avalanching ? ", avalanching" : "");
}

// ---------------------------------------------------------------------------------------------
// waveturb (turb = wave_averaged in nonh): roller thickness from the surface velocity, turbulence advection
// ---------------------------------------------------------------------------------------------
__global__ void k_turb1(real dt) {   // rolthick; face fluxes of kturb (lagrangian or eulerian)
    IJ
    size_t k = ID(i, j);
    int nx = c.nx, ny = c.ny;
    real dcr = cs.beta * sqrt(c.g * d.hh[k]);
    real rt = sd.rolthick[k] + dt * (fmin(d.dzsdt[k], cs.max_dzsdt) - dcr);
    rt = fmax(rt, (real)0); rt = fmin(rt, d.hh[k]);
    if (d.wetz[k] == 0) rt = 0;
    sd.rolthick[k] = rt;
    if (cs.turbadv == 0) { sd.kturb[k] = 0; return; }   // (DR/rho)^(2/3), DR = 0
    const real *kt = sd.kturb;
    real ku;
    if (i <= nx) {
        real u = d.uu[k];
        if (u > 0) ku = cs.thetanum * kt[k] + (1 - cs.thetanum) * kt[ID(min(i + 1, nx), j)];
        else if (u < 0) ku = cs.thetanum * kt[ID(i + 1, j)] + (1 - cs.thetanum) * kt[ID(max(i, 2), j)];
        else ku = (real)0.5 * (kt[k] + kt[ID(i + 1, j)]);
    } else ku = kt[k];
    real kv = 0;
    if (!c.oned && j <= ny) {
        real v = d.vv[k];
        if (v > 0) kv = cs.thetanum * kt[k] + (1 - cs.thetanum) * kt[ID(i, min(j + 1, ny))];
        else if (v < 0) kv = cs.thetanum * kt[ID(i, j + 1)] + (1 - cs.thetanum) * kt[ID(i, max(j, 2))];
        else kv = 0;   // XBeach averages kturbv(i,j) and kturbv(i,j+1), both still zero
    } else kv = kt[k];   // kturbv(:,ny+1) = kturb(:,ny+1)
    if (cs.turbadv == 1) { sd.Sturbu[k] = ku * d.uu[k] * d.hu[k] * (real)d.wetu[k]; sd.Sturbv[k] = kv * d.vv[k] * d.hv[k] * (real)d.wetv[k]; }
    else { sd.Sturbu[k] = ku * d.ueu[k] * d.hu[k] * (real)d.wetu[k]; sd.Sturbv[k] = kv * d.vev[k] * d.hv[k] * (real)d.wetv[k]; }
}
__global__ void k_turb2(real dt) {   // kturb update (i = 2..nx+1, j = 2..ny+1)
    IJ
    size_t k = ID(i, j);
    if (i < 2 || (!c.oned && j < 2)) return;
    real ksource = 0;
    if (cs.lwt == 1) ksource = cs.turbcoef * c.g * sd.rolthick[k] * (cs.beta * sqrt(c.g * d.hh[k]));
    if (cs.turbadv == 0) return;
    real kt = sd.kturb[k];
    real disturb = cs.betad * pow(kt, (real)1.5);
    real div = sd.Sturbu[k] * dnu(i, j) - sd.Sturbu[ID(i - 1, j)] * dnu(i - 1, j);
    if (!c.oned) div = div + sd.Sturbv[k] * dsv(i, j) - sd.Sturbv[ID(i, j - 1)] * dsv(i, j - 1);
    kt = (sd.hold[k] * kt - dt * (div * dsdnzi(i, j) - (ksource - disturb))) / d.hh[k];
    sd.kturb[k] = fmax(kt, (real)0);
}
__global__ void k_turb_bci() {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    sd.kturb[ID(1, j)] = sd.kturb[ID(2, j)]; sd.kturb[ID(c.nx + 1, j)] = sd.kturb[ID(c.nx, j)];
}
__global__ void k_turb_bcj() {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    sd.kturb[ID(i, 1)] = sd.kturb[ID(i, 2)]; sd.kturb[ID(i, c.ny + 1)] = sd.kturb[ID(i, c.ny)];
}

// ---------------------------------------------------------------------------------------------
// sedtransform: equilibrium concentrations ceqsg, ceqbg and adaptation time Tsg (per cell, all fractions)
// ---------------------------------------------------------------------------------------------
__global__ void k_sedtransform(real dt) {
    IJ
    size_t k = ID(i, j);
    const int nx = c.nx;
    sd.kb[k] = sd.kturb[k];                                   // nonh: kb = kturb
    real ue = ue_c(i, j), ve = d.ve[k];
    real vmg = sqrt(ue * ue + ve * ve);                       // lws = 1
    real hh = d.hh[k], hloc = fmax(hh, F32(0.01));
    real urms2 = 0 * 0 + (real)1.45 * sd.kb[k];               // urms = 0 in nonh
    const real delta = cs.delta, onethird = (real)(1.0 / 3.0);
    for (int jg = 0; jg < cs.ngd; jg++) {
        real D50 = cs.D50[jg], D90 = cs.D90[jg];
        real w;
        if (cs.fallvelred == 1) {
            real ws0 = cs.ws0[jg], x = ws0 * D50 / cs.kvis;
            real alpha = (real)2.35 * (2 + (real)0.175 * pow(x, (real)0.75)) / (1 + (real)0.175 * pow(x, (real)0.75));
            w = ws0 * pow(1 - G3(sd.ccg, k, jg), alpha);
        } else w = cs.ws0[jg];
        G3(sd.wfall, k, jg) = w;
        real Ts = cs.tsfac * hloc / w;
        G3(sd.Tsg, k, jg) = cs.oldTsmin == 1 ? fmax(Ts, cs.Tsmin) : fmax(Ts, cs.dtlimTs * dt);
        real Ucr;
        if (cs.form == 0) {
            if (D50 <= (real)0.0005) Ucr = (real)0.19 * pow(D50, (real)0.1) * log10(4 * hloc / D90);
            else Ucr = (real)8.5 * pow(D50, (real)0.6) * log10(4 * hloc / D90);
        } else {
            real Ucrc, Ucrw, Trep = cs.Trep;
            if (D50 <= F32(0.0005)) {
                Ucrc = (real)0.19 * pow(D50, (real)0.1) * log10(4 * hloc / D90);
                Ucrw = (real)0.24 * pow(delta * c.g, (real)0.66) * pow(D50, (real)0.33) * pow(Trep, (real)0.33);
            } else if (D50 <= F32(0.002)) {
                Ucrc = (real)8.5 * pow(D50, (real)0.6) * log10(4 * hloc / D90);
                Ucrw = (real)0.95 * pow(delta * c.g, (real)0.57) * pow(D50, F32(0.43)) * pow(Trep, F32(0.14));
            } else {
                Ucrc = (real)1.3 * sqrt(delta * c.g * D50) * pow(hloc / D50, (real)0.5 * onethird);
                Ucrw = (real)0.95 * pow(delta * c.g, (real)0.57) * pow(D50, F32(0.43)) * pow(Trep, F32(0.14));
            }
            real B2 = vmg / fmax(vmg + sqrt(urms2), c.eps);
            Ucr = B2 * Ucrc + (1 - B2) * Ucrw;
        }
        // bed slope / dilatancy reduction (srfTotal)
        real srfRhee = 0, srfTotal = 1;
        if (cs.dilatancy == 1) {
            real vero = fmax((real)0, -1 * sd.dzbdt[k]);
            real kl = c.g / (160 * cs.kvis) * pow(cs.D15[jg], (real)2) * (pow(cs.por, (real)3) / pow(1 - cs.por, (real)2));
            srfRhee = vero / kl * (cs.pormax - cs.por) / (1 - cs.pormax) * cs.rheeA / delta;
        }
        if (cs.bdslpeffini == 0) srfTotal = 1 + srfRhee;
        else {
            real dzx = sd.dzbdx[k], dzy = sd.dzbdy[k], phi = cs.phi;
            if ((fabs(ue) > (real)0.000001 || fabs(ve) > (real)0.000001) && (fabs(dzx) > (real)0.000001 || fabs(dzy) > (real)0.000001)) {
                real alpha1 = ue < 0 ? atan(ve / ue) + c.px : atan(ve / ue);
                real alpha2 = dzy >= 0 ? -atan(dzx / dzy) + (real)1.5 * c.px : -atan(dzx / dzy) + (real)0.5 * c.px;
                real psi = alpha1 - (alpha2 - c.px);
                real beta = fabs(dzx) < (real)0.000001 ? atan(fabs(dzy)) : atan(fabs(dzx / sin(atan(dzx / dzy))));
                beta = fmin(beta, phi);
                if (cs.dilatancy == 1)
                    srfTotal = (cos(psi) * sin(beta) + sqrt((srfRhee * srfRhee + 2 * srfRhee * cos(beta) + cos(beta) * cos(beta)) * tan(phi) * tan(phi)
                                                             - sin(psi) * sin(psi) * sin(beta) * sin(beta))) / tan(phi);
                else
                    srfTotal = (cos(psi) * sin(beta) + sqrt(cos(beta) * cos(beta) * tan(phi) * tan(phi) - sin(psi) * sin(psi) * sin(beta) * sin(beta))) / tan(phi);
            }
        }
        real Ucrb = Ucr * sqrt(srfTotal);
        real Ucrs = cs.bdslpeffini == 1 ? Ucrb : Ucr * (1 + sqrt(srfRhee));
        real ceqb = 0, ceqs = 0;
        real dster = cs.dster[jg], ws0 = cs.ws0[jg];
        if (cs.form == 0) {
            real cd0 = (real)0.40 / (log(fmax(hloc, 10 * cs.z0) / cs.z0) - 1), Cd = cd0 * cd0;
            real Asb = (real)0.005 * hloc * pow(D50 / hloc / (delta * c.g * D50), (real)1.2);
            real Ass = (real)0.012 * D50 * pow(dster, (real)-0.6) / pow(delta * c.g * D50, (real)1.2);
            real term1 = vmg * vmg + F32(0.018) / Cd * 0 * urms2;   // sws = 0
            term1 = fmin(term1, cs.smax * c.g / (real)0 * D50 * delta);   // s%cf = 0: no limit (division by zero)
            term1 = sqrt(term1);
            if (i <= nx) {
                if (term1 > Ucrb && hloc > c.eps) ceqb = Asb * pow(term1 - Ucrb, (real)2.4);
                if (term1 > Ucrs && hloc > c.eps) ceqs = Ass * pow(term1 - Ucrs, (real)2.4);
            }
        } else if (cs.form == 1) {
            real Asb = (real)0.015 * hloc * pow(D50 / hloc, (real)1.2) / pow(delta * c.g * D50, (real)0.75);
            real Ass = (real)0.012 * D50 * pow(dster, (real)-0.6) / pow(delta * c.g * D50, (real)1.2);
            real term1 = vmg * vmg + (real)0.64 * 0 * urms2;      // sws = 0
            term1 = fmin(term1, cs.smax * c.g / (real)1e-10 * D50 * delta);   // max(s%cf, 1d-10), s%cf = 0
            term1 = sqrt(term1);
            if (i <= nx) {
                if (term1 > Ucrb && hloc > c.eps) ceqb = Asb * pow(term1 - Ucrb, (real)1.5);
                if (term1 > Ucrs && hloc > c.eps) ceqs = Ass * pow(term1 - Ucrs, F32(2.4));
            }
        } else {   // vanrijn1993 (Delft3D style) with ua = 0, H = 0
            real uev = sqrt(vmg * vmg), used = vmg + 0;
            if (uev > Ucrb && hloc > c.eps) {
                real Me = (used - Ucrb) * (used - Ucrb) / (delta * c.g * D50);
                real M = (used - 0) * (used - 0) / (delta * c.g * D50);
                real Asb = (real)0.006 * D50 * ws0;
                ceqb = Asb * pow(M, F32(0.5)) * pow(Me, F32(0.7));
                ceqb = ceqb / used;
            } else ceqb = 0;
            real deltar = cs.deltar, ksw = cs.rwave * deltar;
            ksw = fmax(fmin(F32(0.1), ksw), F32(0.01));
            real uorb = 0;   // pi H / Trep / sinh(k h) with H = 0 in nonh
            real A = uorb * cs.Trep / (F32(2.0) * c.px);
            if (A < (real)0.1) A = 0.1;
            real fw = fmin(exp(F32(-6.0) + F32(5.2) * pow(A / ksw, F32(-0.19))), F32(0.3));
            real muw = F32(0.6) / dster;
            real tauwav = F32(0.25) * c.rho * fw * uorb * uorb;
            real l1 = log10(F32(12.0) * hh / (F32(3.0) * 3 * D90)), l2 = log10(F32(12.0) * hh / deltar);
            real f1c = F32(0.24) * (1 / (l1 * l1)), fc = F32(0.24) * (1 / (l2 * l2));
            real muc = f1c / fc;
            real tauc = (real)0.125 * c.rho * fc * (ue * ue + ve * ve);
            real taubcw = muc * tauc + muw * tauwav;
            real taucr = cs.shieldscrit[jg] * c.g * (cs.rhos - c.rho) * D50;
            for (int ii = 0; ii < cs.kmax; ii++) sd.ccz[(size_t)ii * NN + k] = 0;
            sd.refA[k] = 0; sd.ca[k] = 0;
            if (taubcw > taucr && hloc > c.eps) {
                real refA = F32(0.5) * deltar;
                refA = fmax(F32(0.01), refA);
                refA = fmin(F32(0.2) * hh, refA);
                real delw = F32(0.072) * A * pow(A / ksw, F32(-0.25));
                real ta = fmax((real)0, (muc * tauc + muw * tauwav) / taucr - F32(1.0));
                real ca = F32(0.015) * D50 * pow(ta, F32(1.5)) / (refA * pow(dster, F32(0.3)));
                ca = fmin(ca, F32(0.05));
                real Eswm = F32(0.035) * 0 * hh / cs.Trep;      // H = 0
                real Eswb = F32(0.004) * dster * uorb * (F32(3.0) * delw);
                real uster = pow(pow(d.taubx[k] * d.taubx[k] + d.tauby[k] * d.tauby[k], F32(0.5)) / c.rho, F32(0.5));
                real Escm = F32(0.25) * cs.vonkar * uster * hh;
                // vertical profile on kmax sigma steps
                const int km = cs.kmax;
                real hprev = refA, cprev = ca, ceq_tmp = 0, sigprev = 0;
                bool zeroall = false;   // E = 0 at a step zeroes the whole profile (ceqssteps = 0)
                for (int ii = 0; ii < km; ii++) {
                    real sigs = F32(0.01);
                    real sigz = sigs * pow(1 / sigs, (real)((float)ii / (float)(km - 1)));
                    real hnext = refA + (hh - refA) * sigz;
                    real hhtmp = hprev, dhh = hnext - hhtmp;
                    real Esw;
                    if (hhtmp < 3 * delw) Esw = Eswb;
                    else if (hhtmp > F32(0.5) * hh) Esw = Eswm;
                    else Esw = Eswb + (Eswm - Eswb) * (hhtmp - 3 * delw) / (F32(0.5) * hh - 3 * delw);
                    real Esc = hhtmp > F32(0.5) * hh ? Escm : cs.vonkar * 1 * uster * hhtmp * (1 - hhtmp / hh);
                    real E = pow(Esw * Esw + Esc * Esc, F32(0.5));
                    real cnext;
                    if (E > 0) {
                        real cex = cprev - ws0 * cprev / E * dhh;
                        if (cex < 0) cex = 0;
                        real cim = cprev * (E / (dhh * (ws0 + E / dhh)));
                        cnext = F32(0.5) * cex + F32(0.5) * cim;
                        if (cnext < 0) cnext = 0;
                    } else { cnext = 0; zeroall = true; }
                    sd.ccz[(size_t)ii * NN + k] = cprev * F32(0.5) + cnext * F32(0.5);
                    ceq_tmp = ii == 0 ? sigz * (cnext * F32(0.5) + cprev * F32(0.5)) : ceq_tmp + (sigz - sigprev) * (cnext * F32(0.5) + cprev * F32(0.5));
                    sigprev = sigz; hprev = hnext; cprev = cnext;
                }
                if (zeroall) { ceq_tmp = 0; for (int ii = 0; ii < km; ii++) sd.ccz[(size_t)ii * NN + k] = 0; }
                sd.refA[k] = refA; sd.ca[k] = ca;
                if (hh < refA) { ceqs = ca; for (int ii = 0; ii < km; ii++) sd.ccz[(size_t)ii * NN + k] = ca; }
                else ceqs = ceq_tmp;
            } else ceqs = 0;
        }
        if (cs.alfaD50 > 0) {
            real f = pow(F32(0.000225) / cs.D50[jg], cs.alfaD50);
            ceqb = ceqb * f; ceqs = ceqs * f;
        }
        ceqb = fmin(ceqb / hloc, cs.cmax / 2);
        G3(sd.ceqbg, k, jg) = (1 - cs.bulk) * ceqb * cs.sedcal[jg] * (real)d.wetz[k];
        ceqs = fmin(ceqs / hloc, cs.cmax / 2);
        G3(sd.ceqsg, k, jg) = (ceqs + cs.bulk * ceqb) * cs.sedcal[jg] * (real)d.wetz[k];
    }
}

// transus: erosion limiter of the non-erodible layer (all fractions) and the horizontal diffusion coefficient
__global__ void k_transus_fac(real dt) {
    IJ
    size_t k = ID(i, j);
    for (int jg = 0; jg < cs.ngd; jg++) {
        real pb = PB(k, 0, jg);
        real exp_ero = cs.morfac * dt / (1 - cs.por) * d.hh[k] * (G3(sd.ceqsg, k, jg) * pb / G3(sd.Tsg, k, jg) + G3(sd.ceqbg, k, jg) * pb / dt);
        G3(sd.fac, k, jg) = fmin((real)1, sd.structdepth[k] * pb / fmax(NUMEPS, exp_ero));
    }
    sd.Dc[k] = cs.facDc * d.nuh[k];
    real ue = ue_c(i, j), ve = d.ve[k];
    sd.favc[k] = ue * ue + ve * ve;                           // vmag2
    sd.dcsdx[k] = 0; sd.dcsdy[k] = 0;
}
// suspended-load fluxes of fraction jg at u and v points (upwind concentration, diffusion, bed slope terms)
__global__ void k_transus_flux(int jg) {
    IJ
    size_t k = ID(i, j);
    const int nx = c.nx, ny = c.ny;
    const real *cc = sd.ccg + (size_t)jg * NN;
    // u points
    real ueu_sed = cs.nz > 1 ? (i <= nx ? (real)0.5 * (ue_c(i, j) + ue_c(i + 1, j)) : 0) : d.ueu[k];
    int iv = i <= nx ? i : nx;
    real veu_sed = (real)0.5 * (d.ve[ID(iv, j)] + d.ve[ID(iv + 1, j)]);
    real vmagu = sqrt((ueu_sed + 0) * (ueu_sed + 0) + (veu_sed + 0) * (veu_sed + 0));
    real ureps = ueu_sed + 0;
    real cu, dcx = 0;
    if (i <= nx) {
        if (ureps > 0) cu = cs.thetanum * cc[k] + (1 - cs.thetanum) * cc[ID(min(i + 1, nx), j)];
        else if (ureps < 0) cu = cs.thetanum * cc[ID(i + 1, j)] + (1 - cs.thetanum) * cc[ID(max(i, 2), j)];
        else cu = (real)0.5 * (cc[k] + cc[ID(i + 1, j)]);
        dcx = (cc[ID(i + 1, j)] - cc[k]) / dsu(i, j);
        sd.dcsdx[k] = dcx;
    } else cu = cc[k];
    sd.vmagu[k] = vmagu; sd.ureps[k] = ureps; sd.urepb[k] = ureps;
    real hu = d.hu[k];
    real Sus = cs.sus * (cu * ureps * hu - sd.Dc[k] * hu * dcx) * (real)d.wetu[k];
    // v points
    real vev_sed, uev_sed, cv, dcy = 0;
    if (!c.oned) {
        vev_sed = cs.nz > 1 ? (j <= ny ? (real)0.5 * (d.ve[k] + d.ve[ID(i, j + 1)]) : 0) : d.vev[k];
        uev_sed = j <= ny ? (real)0.5 * (ue_c(i, j) + ue_c(i, j + 1)) : 0;
    } else { uev_sed = ue_c(i, j); vev_sed = d.ve[k]; }
    real vmagv = sqrt((uev_sed + 0) * (uev_sed + 0) + (vev_sed + 0) * (vev_sed + 0));
    real vreps = vev_sed + 0;
    if (!c.oned) {
        if (j <= ny) {
            if (vreps > 0) cv = cs.thetanum * cc[k] + (1 - cs.thetanum) * cc[ID(i, min(j + 1, ny))];
            else if (vreps < 0) cv = cs.thetanum * cc[ID(i, j + 1)] + (1 - cs.thetanum) * cc[ID(i, max(j, 2))];
            else cv = (real)0.5 * (cc[k] + cc[ID(i, j + 1)]);
            dcy = (cc[ID(i, j + 1)] - cc[k]) / dnv(i, j);
            sd.dcsdy[k] = dcy;
        } else cv = cc[k];
    } else cv = cc[k];
    sd.vmagv[k] = vmagv; sd.vreps[k] = vreps; sd.vrepb[k] = vreps;
    real hv = d.hv[k];
    real Svs = cs.sus * (cv * vreps * hv - sd.Dc[k] * hv * dcy) * (real)d.wetv[k];
    // berm slope transport and bed slope effect on the suspended load
    bool berm = cs.bermslopetransport == 1 && cs.bermslopesus == 1 && hu < cs.bermslopedepth;
    if (berm) Sus = Sus - cs.sus * (cs.bermslopefac * cu * vmagu * hu * (sd.dzbdx[k] - cs.bermslope)) * (real)d.wetu[k];
    if (cs.bdslpeffmag == 1) {
        if (!berm) Sus = Sus - cs.sus * (cs.facsl * cu * vmagu * hu * sd.dzbdx[k]) * (real)d.wetu[k];
        Svs = Svs - cs.sus * (cs.facsl * cv * vmagv * hv * sd.dzbdy[k]) * (real)d.wetv[k];
    } else if (cs.bdslpeffmag == 3) {
        if ((fabs(Sus) > (real)0.000001 || fabs(Svs) > (real)0.000001) && !berm) {
            real Ssmtot = sqrt(pow(Sus, (real)2) + pow(Svs, (real)2));
            real dzbds = sd.dzbdx[k] * Sus / Ssmtot + sd.dzbdy[k] * Svs / Ssmtot;
            Sus = Sus * (1 - cs.facsl * dzbds); Svs = Svs * (1 - cs.facsl * dzbds);
        }
    }
    G3(sd.Susg, k, jg) = Sus; G3(sd.Svsg, k, jg) = Svs;
}
// concentration update of fraction jg on the inner cells (source/sink with adaptation time Tsg)
__global__ void k_transus_cc(int jg, real dt) {
    IJ
    if (!IN_ZS) return;
    size_t k = ID(i, j);
    real Ts = G3(sd.Tsg, k, jg), hh = d.hh[k];
    real ero = G3(sd.fac, k, jg) * hh * G3(sd.ceqsg, k, jg) * PB(k, 0, jg) / Ts;
    G3(sd.ero, k, jg) = ero;
    const real *Su = sd.Susg + (size_t)jg * NN, *Sv = sd.Svsg + (size_t)jg * NN;
    real div = Su[k] * dnu(i, j) - Su[ID(i - 1, j)] * dnu(i - 1, j);
    if (!c.oned) div = div + Sv[k] * dsv(i, j) - Sv[ID(i, j - 1)] * dsv(i, j - 1);
    real cc = G3(sd.ccg, k, jg);
    cc = (dt * Ts) / (dt + Ts) * (sd.hold[k] * cc / dt - (div * dsdnzi(i, j) - ero));
    cc = fmax(cc, (real)0);
    cc = fmin(cc, cs.cmax * hh);
    G3(sd.depo_ex, k, jg) = cc / Ts;
    G3(sd.ccg, k, jg) = cc / hh;
}
__global__ void k_transus_bc(int jg) {   // boundary copies (corners take the nearest inner cell), small values to zero
    IJ
    int nx = c.nx, ny = c.ny;
    size_t k = ID(i, j);
    int ii = i == 1 ? 2 : (i == nx + 1 ? nx : i), jj = c.oned ? 1 : (j == 1 ? 2 : (j == ny + 1 ? ny : j));
    if (ii != i || jj != j) {
        size_t q = ID(ii, jj);
        G3(sd.ccg, k, jg) = G3(sd.ccg, q, jg); G3(sd.ero, k, jg) = G3(sd.ero, q, jg); G3(sd.depo_ex, k, jg) = G3(sd.depo_ex, q, jg);
    }
}
__global__ void k_transus_end(int jg) {
    IJ
    size_t k = ID(i, j);
    if (G3(sd.ccg, k, jg) < (real)1e-10) G3(sd.ccg, k, jg) = 0;
    if (jg == cs.ngd - 1) sd.vmag[k] = sqrt(fmax(sd.favc[k], cs.umin));
}
