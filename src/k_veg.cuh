// k_veg.cuh: vegetation.F90 for wavemodel = nonh: veggie_init (input files), vegatt with its nonh branches,
// momeqveg (drag of the vegetation sections on the depth-averaged flow) and porcanflow (porous in-canopy model).
// The short-wave terms (swvegatt, vegnonlin, the Luhar in-canopy velocity of vegcanflo) need short-wave
// quantities that are zero in nonh (H, urms), so they add nothing and are left out.
#pragma once

// vegetation fields, host copies (layer m of a sectioned field at m*N)
struct Veg {
    int nsecmax = 1;
    std::vector<double> vegtype, nsec, isCanopy;   // integer valued
    std::vector<double> Cd, ah, bv, Nv;            // nsecmax layers
    std::vector<double> Kp, beta, cf, cm, lp;
};

static Veg veggie_init(const Params &P, const Grid &G) {
    Veg V;
    const int NX1 = G.NX1, NY1 = G.NY1;
    const size_t N = (size_t)NX1 * NY1;
    // 1) species list (count_lines: every record counts)
    std::vector<std::string> names;
    {
        std::ifstream f(P.veggiefile);
        if (!f) fail("cannot open veggiefile " + P.veggiefile);
        std::string l;
        while (std::getline(f, l)) names.push_back(trim(l));
    }
    const int nveg = (int)names.size();
    // 2) species map
    std::vector<double> map;
    read_grid_file(P.veggiemapfile, NX1, NY1, map, true);
    // 3) species properties
    struct Spec { int nsec, isCanopy; std::vector<double> ah, bv, Nv, Cd; double Kp, beta, cf, cm, por, lp; };
    std::vector<Spec> sp(nveg);
    for (int is = 0; is < nveg; is++) {
        ParFile S(names[is].c_str());
        Spec &s = sp[is];
        s.isCanopy = S.integer("isCanopy", 0);
        if (s.isCanopy == 1 && P.porcanflow != 1) fail("Canopy type vegetation can only be used together with the canopy flow model");
        s.nsec = S.integer("nsec", 1);
        if (s.isCanopy == 1 && s.nsec > 1) fail("Canopy type vegetation can only have nsec = 1");
        if (s.nsec < 1) fail(names[is] + ": nsec must be at least 1");
        s.ah = S.dblvec("ah", s.nsec, 0.1, names[is]);
        if (s.isCanopy == 0) {
            s.bv = S.dblvec("bv", s.nsec, 0.01, names[is]);
            s.Nv = S.dblvec("N", s.nsec, 100.0, names[is]);
            for (auto &x : s.Nv) x = (double)lround(x);   // integer N = nint(...)
            s.Cd = S.dblvec("Cd", s.nsec, 0.0, names[is]);
            s.Kp = std::numeric_limits<double>::max(); s.beta = 0; s.cf = 0; s.cm = 0; s.por = 1.0 - 1e-12;
        } else {
            s.bv.assign(s.nsec, 0.0); s.Nv.assign(s.nsec, 0.0); s.Cd.assign(s.nsec, 0.0);
            s.Kp = S.dbl("Kp_can", 0.0001); s.beta = S.dbl("beta_can", 15.0); s.cf = S.dbl("cf_can", 0.01);
            s.cm = S.dbl("cm_can", 1.5); s.por = S.dbl("por_can", 0.1);
        }
        s.lp = 1.0 - s.por;
        V.nsecmax = std::max(V.nsecmax, s.nsec);
        printf("vegetation species %d (%s): nsec %d%s\n", is + 1, names[is].c_str(), s.nsec, s.isCanopy ? ", canopy" : "");
    }
    const int M = V.nsecmax;
    V.vegtype.assign(N, 0); V.nsec.assign(N, 0); V.isCanopy.assign(N, 0);
    V.Cd.assign(N * M, 0); V.ah.assign(N * M, 0); V.bv.assign(N * M, 0); V.Nv.assign(N * M, 0);
    V.Kp.assign(N, std::numeric_limits<double>::max()); V.beta.assign(N, 0); V.cf.assign(N, 0); V.cm.assign(N, 0);
    V.lp.assign(N, 1e-12);
    for (size_t k = 0; k < N; k++) {
        int ind = (int)lround(map[k]);
        V.vegtype[k] = ind;
        if (ind <= 0) continue;
        if (ind > nveg) fail("veggiemapfile refers to vegetation type " + std::to_string(ind) + ", veggiefile lists " + std::to_string(nveg));
        const Spec &s = sp[ind - 1];
        V.nsec[k] = s.nsec;
        for (int m = 0; m < s.nsec; m++) {
            V.Cd[m * N + k] = s.Cd[m]; V.ah[m * N + k] = s.ah[m]; V.bv[m * N + k] = s.bv[m]; V.Nv[m * N + k] = s.Nv[m];
        }
        V.Kp[k] = s.Kp; V.beta[k] = s.beta; V.cf[k] = s.cf; V.cm[k] = s.cm; V.lp[k] = s.lp; V.isCanopy[k] = s.isCanopy;
    }
    return V;
}

// vegatt, first call: Cd < 0 asks for the bulk drag coefficient of Mendez and Losada (2004) (bulkdragcoeff). In nonh
// the wave height H is zero, so the Keulegan-Carpenter number is zero (or NaN) and the coefficient takes its
// lower-limit value at every vegetated zs point of the inner range.
static bool veg_bulkdrag(Veg &V, const Grid &G) {
    const int nx = G.nx, ny = G.ny;
    const size_t N = (size_t)G.NX1 * G.NY1;
    const double Cdterm = (double)expf(-(0.0138f * 7.0f)) / pow(7.0, 0.3);
    bool any = false;
    for (int j = (ny == 0 ? 1 : 2); j <= (ny == 0 ? 1 : ny); j++)
        for (int i = 2; i <= nx; i++) {
            size_t k = G.I(i, j);
            for (int m = 0; m < (int)V.nsec[k]; m++) if (V.Cd[m * N + k] < 0) { V.Cd[m * N + k] = Cdterm; any = true; }
        }
    return any;
}

// momeqveg (wavemodel = nonh): drag force of each vertical section below the water level on the flow,
// u-velocity and velocity magnitude at u-points
__global__ void k_momeqveg() {
    IJ
    size_t k = ID(i, j), NN = (size_t)c.NX1 * c.NY1;
    real Fu = 0, Fv = 0;
    int ns = d.nsecveg[k];
    if (ns > 0) {
        real hh = d.hh[k], ahtold = 0;
        real ul = c.veguntow ? d.ueu[k] : d.uu[k], vl = c.veguntow ? d.vev[k] : d.vv[k], vm = d.vmageu[k];
        for (int m = 0; m < ns; m++) {
            size_t km = (size_t)m * NN + k;
            real aht = d.ahveg[km] + d.zb0[k] - d.zb[k];
            real fu = 0, fv = 0;
            if (!(ahtold > hh)) {
                real a = fmax(fmin(aht, hh) - ahtold, (real)0) * (real)0.5 * d.Cdveg[km] * d.bveg[km] * d.Nveg[km];
                fu = a * (ul * vm); fv = a * (vl * vm);
            }
            ahtold = aht;
            Fu += fu; Fv += fv;
        }
    }
    d.Fvegu[k] = Fu * c.rho;
    d.Fvegv[k] = Fv * c.rho;
}

// porcanflow: in-canopy velocity (implicit in the canopy drag) and canopy force at zs points of the inner range
__global__ void k_porcan(real dt) {
    IJ
    if (!IN_ZS) return;
    size_t k = ID(i, j);
    if (d.isCanopy[k] != 1) return;
    if (d.wetz[k] != 1) { d.Fcanu[k] = 0; d.Fcanv[k] = 0; d.ucan[k] = 0; d.vcan[k] = 0; return; }
    const real mu = 1e-6;
    real hcan = d.ahveg[k];
    if (d.hh[k] < hcan) hcan = d.hh[k];
    real lp = d.lp_can[k], olp = 1 - lp;
    real A = (1 + d.cm_can[k] * lp / (1 - lp)) / dt;
    real U = (real)0.5 * (d.ueu[ID(i - 1, j)] + d.ueu[k]);          // ue(2:nx)
    real V = d.ve[k];                                                // s%ve (k_ve_centre)
    real vmag = sqrt(U * U + V * V);
    real uo = d.ucan[k], vo = d.vcan[k];
    real vvcan = sqrt((U - uo) * (U - uo) + (V - vo) * (V - vo));
    auto ph = [&](int a, int b) { return d.ph ? d.ph[ID(a, b)] : (real)0; };
    real dzx = (d.zs[ID(i + 1, j)] + ph(i + 1, j) - d.zs[ID(i - 1, j)] - ph(i - 1, j)) / (dsu(i, j) + dsu(i - 1, j));
    real dzy = c.oned ? (real)0
                      : (d.zs[ID(i, j + 1)] + ph(i, j + 1) - d.zs[ID(i, j - 1)] - ph(i, j - 1)) / (dnv(i, j) + dnv(i, j - 1));
    real den = A + mu * olp / d.Kp_can[k] + d.beta_can[k] * vmag;
    real cfh = (real)0.5 * d.cf_can[k] / hcan * vvcan;
    real un = (-(c.g * dzx) + cfh * (U - uo) + A * uo) / den;
    real vn = (-(c.g * dzy) + cfh * (V - vo) + A * vo) / den;
    d.ucan[k] = un; d.vcan[k] = vn;
    real rh = c.rho * hcan, kl = mu * (1 - lp) / d.Kp_can[k], cml = d.cm_can[k] * lp / olp;
    d.Fcanu[k] = rh * (d.beta_can[k] * vmag * un + kl * uo + cml * (un - uo) / dt);
    d.Fcanv[k] = rh * (d.beta_can[k] * vmag * vn + kl * vo + cml * (vn - vo) / dt);
}
// canopy force from zs points to u and v points
__global__ void k_porcan_uv() {
    IJ
    size_t k = ID(i, j);
    if (IN_UU) d.Fvegu[k] = (d.Fcanu[k] + d.Fcanu[ID(i + 1, j)]) / 2;
    if (c.oned) d.Fvegv[k] = d.Fcanv[k];
    else if (IN_VV) d.Fvegv[k] = (d.Fcanv[k] + d.Fcanv[ID(i, j + 1)]) / 2;
}
