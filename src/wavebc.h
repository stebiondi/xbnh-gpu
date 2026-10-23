// wavebc.h: boundaryconditions.F90 wave_bc for wavemodel = nonh (host side). Produces the offshore signal
// ui, vi, zi, wi, dui, dvi on j = 1..ny+1 each time step.
//   params      regular waves (order 1 or 2)
//   ts_nonh     boun_U.bcf (velocity_Boundary)
//   reuse       nhbcflist.bcf entries (several files, nh_reuse.bcf with the bcstarttime offset)
//   parametric, jonstable, swan, vardens: spectra -> nh_series files (wavespec.h), then as reuse
#pragma once

// boundaryconditions.F90 velocity_Boundary: scalar or vector file, linear in time between records
struct NHBC {
    Tok *tk = nullptr; bool vec = false, eof = false;
    int nvar = 0, NY1 = 0;
    int iU = -1, iV = -1, iZ = -1, iW = -1, idU = -1, idV = -1, iQ = -1;   // column index - 1 (first is t)
    double t0 = 0, t1 = 0, bcst = 0; std::vector<double> a0, a1;
    std::string fname;
    bool just_opened = false;   // velocity_Boundary: the call that opens a file interpolates (or extrapolates)
                                // between its first two records without advancing
    ~NHBC() { delete tk; }
    void open(const std::string &fn, int ny1, double bcstarttime) {
        delete tk; tk = nullptr;
        NY1 = ny1; fname = fn; bcst = bcstarttime; eof = false;
        tk = new Tok(fn);
        std::string w; tk->word(w);
        std::string lw = lower(w);
        if (lw == "vector") vec = true; else if (lw == "scalar") vec = false;
        else fail(fn + ": first line must be scalar or vector");
        double nv; tk->num(nv); nvar = (int)nv;
        // XBeach keeps the header indices of earlier files (SAVE variables never reset)
        for (int q = 0; q < nvar; q++) {
            tk->word(w);
            int col = q - 1;
            if (w == "z" || w == "Z" || w == "zs" || w == "ZS" || w == "Zs") iZ = col;
            else if (w == "t" || w == "T") { if (q != 0) fail(fn + ": t must be the first column"); }
            else if (w == "u" || w == "U") iU = col;
            else if (w == "v" || w == "V") iV = col;
            else if (w == "w" || w == "W") iW = col;
            else if (w == "dU" || w == "DU") idU = col;
            else if (w == "dV" || w == "DV") idV = col;
            else if (w == "q" || w == "Q") { iQ = col; iU = col; }
            else if (w == "dq" || w == "dQ") idU = col;
        }
        a0.assign((size_t)(nvar - 1) * NY1, 0); a1 = a0;
        if (!read_row(t0, a0)) fail(fn + ": boundary file has no data");
        t0 += bcst;
        if (!read_row(t1, a1)) { eof = true; t1 = t0; a1 = a0; } else t1 += bcst;
        just_opened = true;
    }
    bool read_row(double &t, std::vector<double> &a) {
        if (!tk->num(t)) return false;
        for (int q = 0; q < nvar - 1; q++) {
            if (vec) { for (int j = 0; j < NY1; j++) if (!tk->num(a[(size_t)q * NY1 + j])) return false; }
            else { double x; if (!tk->num(x)) return false; for (int j = 0; j < NY1; j++) a[(size_t)q * NY1 + j] = x; }
        }
        return true;
    }
    void advance(double t) {
        if (just_opened) { just_opened = false; return; }
        if (eof) return;
        while (!(t >= t0 && t < t1)) {
            t0 = t1; a0.swap(a1);
            if (!read_row(t1, a1)) { eof = true; t1 = t0; a1 = a0; break; }
            t1 += bcst;
        }
    }
    // value of column q at t (0 when the column is absent)
    void get(int q, double t, std::vector<double> &o) const {
        if (q < 0) { std::fill(o.begin(), o.end(), 0.0); return; }
        for (int j = 0; j < NY1; j++) {
            size_t k = (size_t)q * NY1 + j;
            o[j] = (eof || t1 == t0) ? a1[k] : a0[k] + (a1[k] - a0[k]) * (t - t0) / (t1 - t0);
        }
    }
};

struct WaveBCOut {
    std::vector<double> ui, vi, zi, wi, dui, dvi;
    int zmode = 0, wmode = 0;    // 1: zi = zs(2,:)*fz, wi = ws(2,:)*fz on the device
    double fz = 1;
    int isSetV = 0;
    int newwcoef = 0;            // nonh_init_wcoef call requested (start of a boundary file, params)
};

struct SpectralGen;   // wavespec.h
static int spectral_wave_bc(SpectralGen *, double t, double dt, Params &p);

struct WaveBC {
    Params *P = nullptr; const Grid *G = nullptr;
    int NY1 = 0;
    NHBC bc;
    // params (regular waves)
    std::vector<double> L, L0;
    // reuse / spectral
    int curline = 0; double bcstarttime = 0, bcendtime = 1e300; bool started = false;
    std::string nhfile;
    SpectralGen *spec = nullptr;
    void init(Params &p, const Grid &g, SpectralGen *sg = nullptr) {
        P = &p; G = &g; NY1 = g.NY1; spec = sg;
        L0.assign(NY1, p.g * p.Trep * p.Trep / 2 / M_PI); L = L0;
        // ts_nonh: the first wave_bc opens boun_U.bcf and then calls velocity_Boundary again in the same step,
        // which advances to the records around t (unlike reuse files, opened by a single call)
        if (p.wbctype == WBC_TSNONH) { bc.open("boun_U.bcf", NY1, 0); bc.just_opened = false; }
    }
    void read_list_line(int line) {
        std::ifstream f("nhbcflist.bcf");
        if (!f) fail("cannot open nhbcflist.bcf");
        double b0 = 0, b1 = 0, trep = 0, hrms = 0; std::string fn;
        for (int q = 1; q <= line; q++) if (!(f >> b0 >> b1 >> trep >> hrms >> fn)) fail("cannot read line " + std::to_string(line) + " of nhbcflist.bcf");
        bcstarttime = b0; bcendtime = b1; P->Trep = trep; P->Hrms = hrms; nhfile = fn;
    }
    // hh1, hu1: device hh(1,:), hu(1,:) (needed by params and by ts_nonh with a discharge column)
    void step(double t, double dt, const std::vector<double> &hh1, const std::vector<double> &hu1, WaveBCOut &o) {
        Params &p = *P;
        for (auto *v : {&o.ui, &o.vi, &o.zi, &o.wi, &o.dui, &o.dvi}) v->assign(NY1, 0.0);
        o.zmode = 0; o.wmode = 0; o.fz = 1; o.newwcoef = 0;
        bool q3 = p.nonhq3d == 1 && p.nhlay > 0;
        if (p.wbctype == WBC_OFF) return;
        if (p.wbctype == WBC_PARAMS) {
            double wbw = 2 * M_PI / p.Trep, arms = p.Hrms / 2, th0 = p.theta0;
            const double aphi = 1.0 / (((1.0 + sqrt(5.0)) / 2) + 1), bphi = ((1.0 + sqrt(5.0)) / 2) / (((1.0 + sqrt(5.0)) / 2) + 1);
            double xg = G->xz[0], yg = G->yz[0];
            for (int j = 0; j < NY1; j++) {
                double L1 = L[j], err = 1e300; int it = 0;
                while (err > 0.00001 && it < 150) { it++; double L2 = L0[j] * tanh(2 * M_PI * hh1[j] / L1); L1 = L1 * aphi + L2 * bphi; err = fabs(L2 - L1); }
                L[j] = L1;
                double kbw = 2 * M_PI / L[j], h = hh1[j];
                size_t k1 = G->I(1, j + 1);
                double ph = sin(th0) * (G->yz[k1] - yg) + cos(th0) * (G->xz[k1] - xg);
                if (p.order == 1) {
                    o.zi[j] = arms * sin(wbw * t - kbw * ph);
                    o.ui[j] = 1.0 / h * wbw * arms / sinh(kbw * h) * sin(wbw * t - kbw * ph) * 1.0 / kbw * sinh(kbw * h) * cos(th0 - G->alfaz[k1]);
                } else if (p.order == 2) {
                    double th = tanh(kbw * h), kx = kbw * ph - wbw * t;
                    o.zi[j] = arms * (cos(kx) + kbw * arms * (3 - th * th) / (4 * th * th * th) * cos(2 * kx));
                    o.ui[j] = (wbw / kbw) / h * o.zi[j] * cos(th0 - G->alfaz[k1]);
                }
                if (p.nonhspectrum <= 0) o.ui[j] *= p.lwave * (p.order - 1.0);
            }
            o.isSetV = 0; o.newwcoef = 1;
            return;
        }
        if (p.wbctype == WBC_TSNONH) {
            bc.advance(t);
            std::vector<double> u(NY1), du(NY1);
            bc.get(bc.iU, t, u); bc.get(bc.iV, t, o.vi); bc.get(bc.iZ, t, o.zi); bc.get(bc.iW, t, o.wi);
            bc.get(bc.idU, t, du); bc.get(bc.idV, t, o.dvi);
            if (bc.iQ >= 0) {
                for (int j = 0; j < NY1; j++) {
                    if (q3) du[j] = ((1 - 2 * p.nhlay) * u[j] + du[j]) / (2 * p.nhlay * hu1[j] * (1 - p.nhlay));
                    u[j] = u[j] / hu1[j];
                }
            }
            o.ui = u; o.dui = du;
            if (bc.iU < 0) std::fill(o.ui.begin(), o.ui.end(), 0.0);
            if (bc.iZ < 0) o.zmode = 1;
            if (bc.iW < 0) o.wmode = 1;
            o.isSetV = bc.iV >= 0;
            if (!q3) { std::fill(o.dui.begin(), o.dui.end(), 0.0); std::fill(o.dvi.begin(), o.dvi.end(), 0.0); }
            for (int j = 0; j < NY1; j++) o.ui[j] *= p.lwave * (p.order - 1.0);   // wave_bc end, nonhspectrum = 0
            return;
        }
        // reuse and the spectral types: nhbcflist.bcf
        bool startbcf = false;
        if (!started) {
            started = true; startbcf = true;
            if (p.wbctype == WBC_REUSE) curline = 1;
            else curline = spectral_wave_bc(spec, t, dt, p);
        }
        if (t >= bcendtime) {
            if (p.wbctype == WBC_REUSE) { if (t <= p.tstop - dt) curline++; }
            else curline = spectral_wave_bc(spec, t, dt, p);
            startbcf = true;
        }
        if (startbcf) {
            read_list_line(curline);
            bc.open(nhfile, NY1, nhfile == "nh_reuse.bcf" ? bcstarttime : 0);
            o.newwcoef = 1;
        }
        bc.advance(t);
        std::vector<double> u(NY1), v(NY1), du(NY1), dv(NY1);
        bc.get(bc.iU, t, u); bc.get(bc.iV, t, v); bc.get(bc.iZ, t, o.zi); bc.get(bc.iW, t, o.wi);
        bc.get(bc.idU, t, du); bc.get(bc.idV, t, dv);
        if (bc.iZ < 0) o.zmode = 1;
        if (bc.iW < 0) o.wmode = 1;
        o.isSetV = bc.iV >= 0;
        if (!q3) { std::fill(du.begin(), du.end(), 0.0); std::fill(dv.begin(), dv.end(), 0.0); }
        double tf = std::min(t / p.taper, 1.0);
        for (int j = 0; j < NY1; j++) {
            double a = G->alfau[G->I(1, j + 1)];
            o.ui[j] = (u[j] * cos(a) + v[j] * sin(a)) * tf;
            o.vi[j] = -u[j] * sin(a) + v[j] * cos(a);
            o.dui[j] = (du[j] * cos(a) + dv[j] * sin(a)) * tf;
            o.dvi[j] = dv[j];
            o.zi[j] *= tf; o.wi[j] *= tf;
        }
        o.fz = tf;
    }
};
