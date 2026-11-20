// wavespec.h: spectral wave boundary conditions for wavemodel = nonh (waveparamsnew.F90 spectral_wave_bc with
// nonhspectrum = 1; initialize.F90 spectral_wave_init). Reads JONSWAP parameter files (wbctype = parametric),
// JONSWAP tables (jonstable), SWAN 2D spectra (swan) and variance-density files (vardens), single files,
// FILELIST and LOCLIST; draws the wave-train components with XBeach's random-number generator, synthesises
// the surface elevation and velocities along the offshore boundary (on the GPU), adds the bound long waves
// (order = 2, generate_qbcf) and the second-order sum components (highcomp = 1), and writes nh_series / nh_reuse
// files and nhbcflist.bcf exactly as XBeach does, so the run reads them back through the reuse code path.
#pragma once
#include <complex>
#include <map>
#include <chrono>
static double ws_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#ifdef _OPENMP
#include <omp.h>
#endif
typedef std::complex<double> cplx;

// ---- math_tools.F90 random: Park-Miller minimal standard generator (SAVE seed)
static int xb_rng_seed = 1;
static double xb_random(int j) {
    const int a = 16807, m = 2147483647, q = 127773, r = 2836; const double rm = m;
    if (j != 0) { xb_rng_seed = abs(j) % m; if (xb_rng_seed == 0) xb_rng_seed = 1; }
    int hi = xb_rng_seed / q, lo = xb_rng_seed % q;
    int test = a * lo - r * hi;
    xb_rng_seed = test > 0 ? test : test + m;
    return xb_rng_seed / rm;
}

// ---- complex FFT of any length (inverse, unnormalised: sum_k X_k exp(+2 pi i k n / N)); Bluestein for non powers of 2
static void fft_pow2(std::vector<cplx> &a, bool inv) {
    size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; i++) { size_t bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) std::swap(a[i], a[j]); }
    thread_local std::vector<cplx> tw;
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (inv ? 1 : -1);
        tw.resize(len / 2);
        for (size_t k = 0; k < len / 2; k++) tw[k] = std::polar(1.0, ang * k);   // same value as computed per butterfly
        for (size_t i = 0; i < n; i += len)
            for (size_t k = 0; k < len / 2; k++) {
                const cplx &w = tw[k];
                cplx u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v; a[i + k + len / 2] = u - v;
            }
    }
}
static std::vector<cplx> ifft_unnorm(const std::vector<cplx> &x) {
    size_t n = x.size();
    if ((n & (n - 1)) == 0) { std::vector<cplx> a = x; fft_pow2(a, true); return a; }
    size_t m = 1; while (m < 2 * n - 1) m <<= 1;
    // chirp w and FFT(b) depend on n only: cached per thread (identical values to recomputing them)
    thread_local std::map<size_t, std::pair<std::vector<cplx>, std::vector<cplx>>> cache;
    auto it_ = cache.find(n);
    if (it_ == cache.end()) {
        std::vector<cplx> w(n), b(m, 0);
        for (size_t k = 0; k < n; k++) { double t = M_PI * (double)((k * k) % (2 * n)) / n; w[k] = std::polar(1.0, t); }   // exp(+i pi k^2/n)
        b[0] = std::conj(w[0]);
        for (size_t k = 1; k < n; k++) b[k] = b[m - k] = std::conj(w[k]);
        fft_pow2(b, false);
        it_ = cache.emplace(n, std::make_pair(std::move(w), std::move(b))).first;
    }
    const std::vector<cplx> &w = it_->second.first, &b = it_->second.second;
    std::vector<cplx> a(m, 0);
    for (size_t k = 0; k < n; k++) a[k] = x[k] * w[k];
    fft_pow2(a, false);
    for (size_t k = 0; k < m; k++) a[k] *= b[k];
    fft_pow2(a, true);
    std::vector<cplx> out(n);
    for (size_t k = 0; k < n; k++) out[k] = a[k] / (double)m * w[k];
    return out;
}

// ---- GPU synthesis of the phase-resolved time series (generate_swts), double precision
struct SwtsArgs { int K, NY1, tslen, q3; double h0, nhlay; };
__global__ void k_swts(SwtsArgs a, const double *A, const double *w, const double *kk, const double *th, const double *ph,
                       const double *tin, const double *dx, const double *dy, double *zs, double *u, double *v,
                       double *du, double *dv) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)a.NY1 * a.tslen) return;
    int j = (int)(idx % a.NY1), it = (int)(idx / a.NY1);
    double t = tin[it], z = 0, U = 0, V = 0, dU = 0, dV = 0;
    double zz = -1 * (a.h0 - a.nhlay * a.h0);
    const double *sth = th + a.K, *cth = th + 2 * a.K;   // sin(th), cos(th), filled by k_swts_trig
    double dyj = dy[j], dxj = dx[j];
    for (int ik = 0; ik < a.K; ik++) {
        double Ajk = A[(size_t)ik * a.NY1 + j];
        double sa = sin(w[ik] * t - kk[ik] * (sth[ik] * dyj + cth[ik] * dxj) + ph[ik]);
        z = z + Ajk * sa;
        if (a.q3) {
            double u1 = w[ik] * Ajk / (sinh(kk[ik] * a.h0) * kk[ik]) * (sinh(kk[ik] * (zz + a.h0))) * sa;
            u1 = u1 / (a.h0 + zz);
            double u2 = w[ik] * Ajk / (sinh(kk[ik] * a.h0) * kk[ik]) * (sinh(kk[ik] * (0 + a.h0)) - sinh(kk[ik] * (zz + a.h0))) * sa;
            u2 = u2 / (zz * -1);
            double Uk = a.nhlay * u1 + (1 - a.nhlay) * u2, dUk = u1 - u2;
            U = U + cth[ik] * Uk; V = V + sth[ik] * Uk;
            dU = dU + cth[ik] * dUk; dV = dV + sth[ik] * dUk;
        } else {
            double Uk = 1.0 / a.h0 * w[ik] * Ajk * sa * 1.0 / kk[ik];
            U = U + cth[ik] * Uk; V = V + sth[ik] * Uk;
        }
    }
    size_t o = (size_t)it * a.NY1 + j;
    zs[o] = z; u[o] = U; v[o] = V; du[o] = dU; dv[o] = dV;
}

__global__ void k_swts_trig(int K, double *th) {   // th[0..K) -> th[K..2K) = sin, th[2K..3K) = cos
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) { th[K + i] = sin(th[i]); th[2 * K + i] = cos(th[i]); }
}
// j-independent factors of k_qbcf_gn: cos/sin(-dphi3), cg3 cos(th3), cg3 sin(th3) (same expressions, same bits)
__global__ void k_qbcf_pre(size_t n, const double *dphi3, const double *cg3, const double *th3, double *pre) {
    size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;
    pre[q] = cos(-dphi3[q]); pre[n + q] = sin(-dphi3[q]);
    pre[2 * n + q] = cg3[q] * cos(th3[q]); pre[3 * n + q] = cg3[q] * sin(th3[q]);
}

// all rows j0 .. j0+nj-1 in one launch (blockIdx.y = row); per-thread arithmetic identical to k_qbcf_gn
__global__ void k_qbcf_gn_rows(int K, int NY1, int j0, int nj, double dfgen, const double *dxa, const double *dya,
                               const double *D, const double *Sq, const double *pre, const double *cg3,
                               const double *KKx, const double *KKy, double *outall) {
    int m = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jr = blockIdx.y;
    if (m > K - 1 || jr >= nj) return;
    int j = j0 + jr;
    double dyj = dya[j], dxj = dxa[j];
    const size_t n = (size_t)(K - 1) * K;
    double sr[4] = {0, 0, 0, 0}, si[4] = {0, 0, 0, 0};
    for (int k = 1; k <= K; k++) {
        size_t q = (size_t)(m - 1) + (size_t)(K - 1) * (k - 1);
        double Ef = 0;
        if (k <= K - m) Ef = 2 * D[q] * D[q] * Sq[(size_t)(k - 1) * NY1 + j] * Sq[(size_t)(k + m - 1) * NY1 + j] * dfgen;
        double Ab = sqrt(2 * Ef * dfgen) * (D[q] >= 0 ? 1.0 : -1.0);
        double er = Ab / 2 * pre[q], ei = Ab / 2 * pre[n + q];
        double ph = -(KKy[q] * dyj + KKx[q] * dxj), pr = cos(ph), pi_ = sin(ph);
        double f[4] = {pre[2 * n + q], pre[3 * n + q], cg3[q], 1.0};
        for (int iq = 0; iq < 4; iq++) {
            double ar = er * f[iq], ai = ei * f[iq];
            sr[iq] += ar * pr - ai * pi_; si[iq] += ar * pi_ + ai * pr;
        }
    }
    double *out = outall + (size_t)jr * 8 * (K - 1);
    for (int iq = 0; iq < 4; iq++) { out[(size_t)iq * 2 * (K - 1) + 2 * (m - 1)] = sr[iq]; out[(size_t)iq * 2 * (K - 1) + 2 * (m - 1) + 1] = si[iq]; }
}

// generate_qbcf: Gn(m+1) = sum_k Ftemp(m,k,iq) exp(-i (KKy dy_j + KKx dx_j)) for one offshore point j, iq = 1..4
__global__ void k_qbcf_gn(int K, int NY1, int j, double dfgen, double dyj, double dxj, const double *D, const double *Sq,
                          const double *pre, const double *cg3, const double *KKx, const double *KKy,
                          double *out) {
    const size_t n = (size_t)(K - 1) * K;
    int m = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (m > K - 1) return;
    double sr[4] = {0, 0, 0, 0}, si[4] = {0, 0, 0, 0};
    for (int k = 1; k <= K; k++) {
        size_t q = (size_t)(m - 1) + (size_t)(K - 1) * (k - 1);
        double Ef = 0;
        if (k <= K - m) Ef = 2 * D[q] * D[q] * Sq[(size_t)(k - 1) * NY1 + j] * Sq[(size_t)(k + m - 1) * NY1 + j] * dfgen;
        double Ab = sqrt(2 * Ef * dfgen) * (D[q] >= 0 ? 1.0 : -1.0);
        double er = Ab / 2 * pre[q], ei = Ab / 2 * pre[n + q];     // Abnd/2*exp(-i dphi3)
        double ph = -(KKy[q] * dyj + KKx[q] * dxj), pr = cos(ph), pi_ = sin(ph);
        double f[4] = {pre[2 * n + q], pre[3 * n + q], cg3[q], 1.0};
        for (int iq = 0; iq < 4; iq++) {
            double ar = er * f[iq], ai = ei * f[iq];
            sr[iq] += ar * pr - ai * pi_; si[iq] += ar * pi_ + ai * pr;
        }
    }
    for (int iq = 0; iq < 4; iq++) { out[(size_t)iq * 2 * (K - 1) + 2 * (m - 1)] = sr[iq]; out[(size_t)iq * 2 * (K - 1) + 2 * (m - 1) + 1] = si[iq]; }
}

struct SpecIn {
    std::vector<double> f, ang, S, Sf, Sd;   // S(i,ii) = S[i + nf*ii]
    int nf = 0, nang = 0; double df = 0, dang = 0, hm0 = -1, fp = 0, dir0 = 0, scoeff = -1, trep = 0, dirm = 0;
    double &s(int i, int ii) { return S[(size_t)i + (size_t)nf * ii]; }
    double s(int i, int ii) const { return S[(size_t)i + (size_t)nf * ii]; }
};

static const int NFINT = 801, NAINT = 401, KMIN = 200;
static const double WDMAX = 5.0;

// readkey for spectrum parameter files (key = value lines; first 10 characters for single doubles)
struct KeyFile {
    std::vector<std::pair<std::string, std::string>> kv;
    explicit KeyFile(const std::string &fn) {
        std::ifstream f(fn); if (!f) fail("cannot open " + fn);
        std::string l;
        while (std::getline(f, l)) {
            size_t pc = l.find('%'); if (pc != std::string::npos) l = l.substr(0, pc);
            size_t eq = l.find('='); if (eq == std::string::npos) continue;
            kv.push_back({lower(trim(l.substr(0, eq))), trim(l.substr(eq + 1))});
        }
    }
    const std::string *find(const std::string &k) const { for (auto &p : kv) if (p.first == lower(k)) return &p.second; return nullptr; }
    bool has(const std::string &k) const { auto v = find(k); return v && !v->empty(); }
    std::vector<double> vec(const std::string &k, int n, double def) const {
        std::vector<double> o(n, def); auto v = find(k);
        if (!v || v->empty()) return o;
        std::string s = *v; for (auto &ch : s) if (ch == ',') ch = ' ';
        std::istringstream is(s); for (int q = 0; q < n; q++) { double x; if (!(is >> x)) fail("cannot read " + k); o[q] = x; }
        return o;
    }
    double dbl(const std::string &k, double def) const {
        auto v = find(k); if (!v || v->empty()) return def;
        std::string s = v->substr(0, 10), t; for (char ch : s) if (ch != ' ') t.push_back(ch);
        for (auto &ch : t) if (ch == 'd' || ch == 'D') ch = 'e';
        return atof(t.c_str());
    }
    int integer(const std::string &k, int def) const { auto v = find(k); if (!v || v->empty()) return def; return atoi(v->c_str()); }
};

struct SpectralGen {
    Params *P = nullptr; const Grid *G = nullptr;
    const real *d_zs0 = nullptr, *d_zb = nullptr;
    int NY1 = 0;
    struct BcFile { std::string fname; int listline = 0; bool reuse = false; };
    std::vector<BcFile> bcfiles; std::vector<int> n_index_loc;
    int nspectra = 0, bccount = 0; bool reuseall = false;
    double spectrumendtime = 0, rtbc_local = 0, dtbc_local = 0, maindir_local = 0;

    // spectral_wave_init
    void init(Params &p, const Grid &g, const real *zs0, const real *zb) {
        P = &p; G = &g; d_zs0 = zs0; d_zb = zb; NY1 = g.NY1;
        std::ifstream f(p.bcfile); if (!f) fail("cannot open bcfile " + p.bcfile);
        std::string first; f >> first;
        if (first == "LOCLIST") {
            for (int q = 0; q < p.nspectrumloc; q++) {
                double xs, ys; std::string fn;
                if (!(f >> xs >> ys >> fn)) fail("LOCLIST: number of lines must equal nspectrumloc");
                double best = 1e300; int bj = 1;
                for (int j = 1; j <= NY1; j++) { size_t k = g.I(1, j); double dd = sqrt((xs - g.xz[k]) * (xs - g.xz[k]) + (ys - g.yz[k]) * (ys - g.yz[k])); if (dd < best) { best = dd; bj = j; } }
                n_index_loc.push_back(bj); bcfiles.push_back({fn, 0, false});
            }
            nspectra = p.nspectrumloc;
        } else {
            if (p.nspectrumloc != 1) fail("nspectrumloc > 1 needs a LOCLIST bcfile");
            n_index_loc = {1}; bcfiles = {{p.bcfile, 0, false}}; nspectra = 1;
        }
    }
    std::string bcname() const {
        if (reuseall) return "nh_reuse.bcf";
        char b[32]; snprintf(b, sizeof b, "nh_series%05d.bcf", bccount); return b;
    }

    // ---- spectrum readers
    void read_jonswap(const std::string &fn, int &listline, double &rtbc, double &dtbc, SpecIn &sp) {
        Params &p = *P;
        int nmodal; std::vector<double> Hm0, fp, gam, mainang, scoeff; std::vector<int> tma; double fnyq, dfj;
        if (p.wbctype != WBC_JONSTABLE) {
            KeyFile K(fn);
            nmodal = K.integer("nmodal", 1);
            Hm0 = K.vec("Hm0", nmodal, 0); if (!K.has("Hm0")) fail(fn + ": Hm0 is required");
            if (K.has("fp") && !K.has("Tp")) fp = K.vec("fp", nmodal, 0.08);
            else if (!K.has("fp") && !K.has("Tp")) fail(fn + ": Tp or fp is required");
            else { fp = K.vec("Tp", nmodal, 12.5); for (auto &x : fp) x = 1.0 / x; }
            gam = K.vec("gammajsp", nmodal, 3.3); scoeff = K.vec("s", nmodal, 10.0);
            std::vector<double> tm = K.vec("tma", nmodal, 0); for (double x : tm) tma.push_back((int)x);
            if (K.has("mainang")) mainang = K.vec("mainang", nmodal, 270);
            else if (K.has("dir0")) mainang = K.vec("dir0", nmodal, 270);
            else mainang.assign(nmodal, 270);
            fnyq = K.dbl("fnyq", std::max(0.3, 3.0 * *std::max_element(fp.begin(), fp.end())));
            dfj = K.dbl("dfj", fnyq / 200);
        } else {
            nmodal = 1;
            std::ifstream f(fn); if (!f) fail("cannot open " + fn);
            std::string line;
            for (int q = 0; q < listline; q++) if (!std::getline(f, line)) fail(fn + ": end of jonstable file");
            double h, tp, ma, g, sc;
            while (std::getline(f, line)) if (trim(line).size()) break;
            std::istringstream is(line);
            if (!(is >> h >> tp >> ma >> g >> sc >> rtbc >> dtbc)) fail(fn + ": not enough data in jonstable for the simulation time");
            if (p.morfacopt == 1) rtbc /= std::max(p.morfac, 1.0);   // morphological time
            listline++;
            Hm0 = {h}; fp = {1.0 / tp}; mainang = {ma}; gam = {g}; scoeff = {sc}; tma = {0};
            fnyq = 3.0 * fp[0]; dfj = fp[0] / 50;
        }
        sp.nf = (int)ceil((fnyq - dfj) / dfj); sp.df = dfj;
        sp.f.resize(sp.nf); for (int i = 0; i < sp.nf; i++) sp.f[i] = (i + 1) * dfj;
        sp.nang = NAINT; sp.dang = 2 * M_PI / (NAINT - 1);
        sp.ang.resize(sp.nang); for (int i = 0; i < sp.nang; i++) sp.ang[i] = i * sp.dang;
        std::vector<std::vector<double>> multi(nmodal);
        for (int ip = 0; ip < nmodal; ip++) {
            std::vector<double> y(sp.nf);
            jonswapgk(sp.f, fp[ip], gam[ip], y);
            if (tma[ip] == 1) fail("tma = 1 in the JONSWAP file is not supported");
            double sy = 0; for (double v : y) sy += v;
            double fac = pow(Hm0[ip] / (4.0 * sqrt(sy * dfj)), 2);
            for (auto &v : y) v = fac * v;
            double ma = 1.5 * M_PI - mainang[ip] * M_PI / 180;
            while (ma > 2 * M_PI || ma < 0) { if (ma > 2 * M_PI) ma -= 2 * M_PI; else if (ma < 0) ma += 2 * M_PI; }
            mainang[ip] = ma;
            std::vector<double> Dd(sp.nang), td(sp.nang);
            for (int i = 0; i < sp.nang; i++) td[i] = (sp.ang[i] - ma) / 2;
            bool again = true;
            while (again) { again = false; for (auto &x : td) { if (x > 2 * M_PI) { x -= 2 * M_PI; again = true; } else if (x < 0) { x += 2 * M_PI; again = true; } } }
            double sD = 0; int ex = 2 * (int)lround(scoeff[ip]);
            for (int i = 0; i < sp.nang; i++) { Dd[i] = pow(cos(td[i]), ex); sD += Dd[i]; }
            for (auto &x : Dd) x = x / (sD * sp.dang);
            multi[ip].resize((size_t)sp.nf * sp.nang);
            for (int i = 0; i < sp.nang; i++) for (int ii = 0; ii < sp.nf; ii++) multi[ip][(size_t)ii + (size_t)sp.nf * i] = y[ii] * Dd[i];
        }
        if (nmodal == 1) { sp.hm0 = Hm0[0]; sp.fp = fp[0]; sp.dir0 = mainang[0]; sp.scoeff = scoeff[0]; sp.S = multi[0]; }
        else {
            double sh = 0; for (double h : Hm0) sh += h * h; sp.hm0 = sqrt(sh);
            int ind = (int)(std::max_element(Hm0.begin(), Hm0.end()) - Hm0.begin());
            sp.fp = fp[ind]; sp.dir0 = mainang[ind];
            bool all1024 = true, samedir = true;
            for (int ip = 0; ip < nmodal; ip++) { if (scoeff[ip] < 1024) all1024 = false; if (mainang[ip] != mainang[ind]) samedir = false; }
            sp.scoeff = (all1024 && samedir) ? 1024 : std::min(scoeff[ind], 999.0);
            sp.S.assign((size_t)sp.nf * sp.nang, 0);
            for (int ip = 0; ip < nmodal; ip++) for (size_t k = 0; k < sp.S.size(); k++) sp.S[k] += multi[ip][k];
        }
        integrate_Sf(sp);
    }
    static void jonswapgk(const std::vector<double> &f, double fp, double gam, std::vector<double> &y) {
        // single-precision literals of XBeach: 1e-20, 0.07, 0.09, 1.25 (exact)
        const double s1 = (double)0.07f, s2 = (double)0.09f, tiny_ = (double)1e-20f;
        double ym = -1;
        for (size_t i = 0; i < f.size(); i++) {
            double xa = fabs(f[i] / fp);
            if (xa == 0) xa = tiny_;
            double sig = xa < 1.0 ? s1 : s2;
            double fac1 = pow(xa, -5), fac2 = exp(-1.25 * pow(xa, -4)), fac3 = pow(gam * 1.0, exp(-((xa - 1) * (xa - 1)) / (2 * (sig * sig))));
            y[i] = fac1 * fac2 * fac3; ym = std::max(ym, y[i]);
        }
        for (auto &v : y) v = v / ym;
    }
    static void integrate_Sf(SpecIn &sp) {
        sp.Sf.assign(sp.nf, 0);
        for (int i = 0; i < sp.nf; i++) for (int ii = 0; ii < sp.nang; ii++) {
            double w;
            if (ii == 0) w = fabs(sp.ang[1] - sp.ang[0]);
            else if (ii == sp.nang - 1) w = fabs(sp.ang[ii] - sp.ang[ii - 1]);
            else w = fabs(sp.ang[ii + 1] - sp.ang[ii - 1]) / 2;
            sp.Sf[i] = sp.Sf[i] + sp.s(i, ii) * w;
        }
    }
    void read_swan(const std::string &fn, SpecIn &sp) {
        Params &p = *P;
        std::ifstream f(fn); if (!f) fail("cannot open " + fn);
        std::string line; int sw = 0;
        auto six = [](const std::string &l) { std::string r = l.substr(0, std::min<size_t>(6, l.size())); while (r.size() < 6) r += ' '; return r; };
        while (sw == 0) { if (!std::getline(f, line)) fail(fn + ": no RFREQ/AFREQ"); std::string r = six(line); if (r == "RFREQ ") sw = 1; else if (r == "AFREQ ") sw = 2; }
        std::getline(f, line); sp.nf = atoi(trim(line).c_str());
        sp.f.resize(sp.nf); for (int i = 0; i < sp.nf; i++) { std::getline(f, line); sp.f[i] = atof(trim(line).c_str()); }
        std::getline(f, line); std::string r = six(line);
        if (r == "NDIR  ") sw = 1; else if (r == "CDIR  ") sw = 2; else fail(fn + ": SWAN directions not recognised (NDIR or CDIR)");
        std::getline(f, line); sp.nang = atoi(trim(line).c_str());
        sp.ang.resize(sp.nang); for (int i = 0; i < sp.nang; i++) { std::getline(f, line); sp.ang[i] = atof(trim(line).c_str()); }
        if (sw == 1) for (auto &a : sp.ang) a = 270.0 - a; else for (auto &a : sp.ang) a = a - p.dthetaS_XB;
        bool flipped = false;
        if (sp.ang[1] < sp.ang[0]) { std::reverse(sp.ang.begin(), sp.ang.end()); flipped = true; }
        for (auto &a : sp.ang) a = a * M_PI / 180;
        sp.dang = sp.ang[1] - sp.ang[0];
        std::getline(f, line); std::getline(f, line); std::getline(f, line); r = six(line);
        if (r == "VaDens") sw = 1; else if (r == "EnDens") sw = 2; else fail(fn + ": SWAN file must hold VaDens or EnDens");
        std::getline(f, line); std::getline(f, line); double exc = atof(trim(line).c_str());
        int found = 0;
        while (!found) { if (!std::getline(f, line)) fail(fn + ": no FACTOR"); r = six(line);
            if (r == "FACTOR") found = 1; else if (r == "ZERO  " || r == "NODATA") fail(fn + ": SWAN spectrum has no data (ZERO/NODATA)"); }
        std::getline(f, line); double factor = atof(trim(line).c_str());
        sp.S.assign((size_t)sp.nf * sp.nang, 0);
        Tok t(fn);   // re-read numerically from the current position: simpler to parse rows by line
        for (int i = 0; i < sp.nf; i++) {
            std::vector<double> row; while ((int)row.size() < sp.nang) { if (!std::getline(f, line)) fail(fn + ": short spectrum"); std::istringstream is(line); double x; while (is >> x) row.push_back(x); }
            for (int ii = 0; ii < sp.nang; ii++) sp.s(i, ii) = row[ii];
        }
        for (auto &v : sp.S) if (v == exc) v = 0;
        if (flipped) for (int i = 0; i < sp.nf; i++) for (int a = 0, b = sp.nang - 1; a < b; a++, b--) std::swap(sp.s(i, a), sp.s(i, b));
        for (auto &v : sp.S) v = v * factor;
        if (sw == 2) for (auto &v : sp.S) v = v / (p.rho * p.g);
        for (auto &v : sp.S) v = v * 180 / M_PI;
        sp.scoeff = -1; sp.hm0 = -1;
        integrate_Sf(sp);
    }
    void read_vardens(const std::string &fn, SpecIn &sp) {
        Tok t(fn); double x;
        t.num(x); sp.nf = (int)x; sp.f.resize(sp.nf); for (auto &v : sp.f) t.num(v);
        t.num(x); sp.nang = (int)x; sp.ang.resize(sp.nang); for (auto &v : sp.ang) t.num(v);
        for (auto &a : sp.ang) a = a * M_PI / 180;
        sp.dang = sp.ang[1] - sp.ang[0];
        sp.S.assign((size_t)sp.nf * sp.nang, 0);
        for (int i = 0; i < sp.nf; i++) for (int ii = 0; ii < sp.nang; ii++) { if (!t.num(x)) fail(fn + ": short spectrum"); sp.s(i, ii) = x; }
        for (auto &v : sp.S) v = v * 180 / M_PI;
        sp.Sd.assign(sp.nang, 0);
        for (int ii = 0; ii < sp.nang; ii++) for (int i = 0; i < sp.nf; i++) sp.Sd[ii] += sp.s(i, ii);
        int nnz = 0; for (double v : sp.Sd) if (v > 0) nnz++;
        if (nnz == 1) { sp.scoeff = 1024; sp.dir0 = sp.ang[std::max_element(sp.Sd.begin(), sp.Sd.end()) - sp.Sd.begin()]; }
        else sp.scoeff = -1;
        integrate_Sf(sp);
        sp.hm0 = -1;
    }
    void read_spectrum_input(BcFile &fn, double &rtbc, double &dtbc, SpecIn &sp) {
        Params &p = *P;
        std::ifstream f(fn.fname); if (!f) fail("cannot open " + fn.fname);
        std::string first; f >> first;
        bool filelist = first == "FILELIST";
        std::string readfile;
        if (filelist) {
            if (fn.listline == 0) fn.listline = 1;
            fn.reuse = false;
            f.clear(); f.seekg(0);
            std::string line;
            for (int q = 0; q < fn.listline; q++) std::getline(f, line);
            if (!(f >> rtbc >> dtbc >> readfile)) fail(fn.fname + ": FILELIST has no more entries");
            if (p.morfacopt == 1) rtbc /= std::max(p.morfac, 1.0);
            fn.listline++;
        } else {
            fn.reuse = p.wbctype != WBC_JONSTABLE;
            rtbc = p.rt; dtbc = p.dtbc; readfile = fn.fname;
        }
        sp = SpecIn();
        if (p.wbctype == WBC_PARAMETRIC || p.wbctype == WBC_JONSTABLE) read_jonswap(readfile, fn.listline, rtbc, dtbc, sp);
        else if (p.wbctype == WBC_SWAN) read_swan(readfile, sp);
        else read_vardens(readfile, sp);
    }

    // ---- interpolate_spectrum
    void interpolate_spectrum(const SpecIn &in, SpecIn &o, double fmax) {
        Params &p = *P;
        o = SpecIn(); o.nf = NFINT; o.nang = NAINT;
        o.df = fmax / (NFINT - 1); o.f.resize(NFINT); for (int i = 0; i < NFINT; i++) o.f[i] = i * o.df;
        o.dang = 2 * M_PI / (NAINT - 1); o.ang.resize(NAINT); for (int i = 0; i < NAINT; i++) o.ang[i] = i * o.dang;
        o.S.assign((size_t)NFINT * NAINT, 0);
        // interp_in_cyclic_function over angles, then linear in frequency
        std::vector<double> St((size_t)in.nf * NAINT);
        double dx = in.ang[1] - in.ang[0], xcycle = 2 * M_PI;
        int icycle = (int)lround(xcycle / dx);
        std::vector<double> xc(icycle + 1), yc(icycle + 1);
        for (int i = 0; i < in.nf; i++) {
            int n = in.nang;
            if (n > icycle + 1) fail("spectrum: more directions than one cycle");
            for (int q = 0; q < n; q++) { xc[q] = in.ang[q]; yc[q] = in.s(i, q); }
            if (n == icycle) { xc[n] = xc[n - 1] + dx; yc[n] = yc[0]; }
            else if (n < icycle) { for (int q = n; q < icycle; q++) { xc[q] = xc[q - 1] + dx; yc[q] = 0; } xc[icycle] = xc[icycle - 1] + dx; yc[icycle] = yc[0]; }
            int nc = icycle + 1;
            for (int ip = 0; ip < NAINT; ip++) {
                int il = (int)floor((o.ang[ip] - xc[0]) / dx) + 1;
                while (il < 1) il += icycle; while (il > nc) il -= icycle;
                double yl = (il > nc || il < 1) ? 0 : yc[il - 1];
                int ir = il + 1;
                while (ir < 1) ir += icycle; while (ir > nc) ir -= icycle;
                double yr = (ir > nc || ir < 1) ? 0 : yc[ir - 1];
                double fr = fmod(o.ang[ip], dx) / dx, fl = 1.0 - fr;
                St[(size_t)i + (size_t)in.nf * ip] = fl * yl + fr * yr;
            }
        }
        for (int j = 0; j < NAINT; j++) for (int i = 0; i < NFINT; i++) {
            if (o.f[i] > in.f[in.nf - 1] || o.f[i] < in.f[0]) o.s(i, j) = 0;
            else o.s(i, j) = lin_interp(in.f.data(), &St[(size_t)in.nf * j], in.nf, o.f[i]);
        }
        o.Sf.assign(NFINT, 0);
        for (int i = 0; i < NFINT; i++) { double s = 0; for (int j = 0; j < NAINT; j++) s += o.s(i, j); o.Sf[i] = s * o.dang; }
        double fmin_ = *std::min_element(in.f.begin(), in.f.end()), fmax_ = *std::max_element(in.f.begin(), in.f.end());
        for (int i = 0; i < NFINT; i++) {
            if (o.f[i] >= fmin_ && o.f[i] <= fmax_) {
                double Sfnow = lin_interp(in.f.data(), in.Sf.data(), in.nf, o.f[i]);
                if (o.Sf[i] > 0 && Sfnow > 0) { double factor = Sfnow / o.Sf[i]; o.Sf[i] = o.Sf[i] * factor; for (int j = 0; j < NAINT; j++) o.s(i, j) = o.s(i, j) * factor; }
                else if (o.Sf[i] == 0 && Sfnow >= 0) {
                    o.Sf[i] = Sfnow;
                    int jm = 0; for (int j = 1; j < NAINT; j++) if (o.s(i, j) > o.s(i, jm)) jm = j;
                    o.s(i, jm) = Sfnow / o.dang;
                } else { o.Sf[i] = 0; for (int j = 0; j < NAINT; j++) o.s(i, j) = 0; }
            }
        }
        double m0 = 0; for (double v : o.Sf) m0 += v; m0 *= o.df;
        o.hm0 = 4 * sqrt(m0);
        o.Sd.assign(NAINT, 0);
        for (int j = 0; j < NAINT; j++) { double s = 0; for (int i = 0; i < NFINT; i++) s += o.s(i, j); o.Sd[j] = s * o.df; }
        int im = (int)(std::max_element(o.Sd.begin(), o.Sd.end()) - o.Sd.begin());
        o.dir0 = 270.0 - o.ang[im] * 180 / M_PI;
        im = (int)(std::max_element(o.Sf.begin(), o.Sf.end()) - o.Sf.begin());
        o.fp = o.f[im];
        o.trep = tpDcalc(o.Sf, o.f, p.trepfac, p.Tm01switch);
        double ss = 0, sc = 0, sd = 0;
        for (int j = 0; j < NAINT; j++) { ss += sin(o.ang[j]) * o.Sd[j]; sc += cos(o.ang[j]) * o.Sd[j]; sd += o.Sd[j]; }
        o.dirm = fmod(270.0 - 180.0 / M_PI * atan2(ss / sd, sc / sd), 360.0);
    }
    static double tpDcalc(const std::vector<double> &Sf, const std::vector<double> &f, double trepfac, int sw) {
        double mx = *std::max_element(Sf.begin(), Sf.end()), a = 0, b = 0;
        for (size_t i = 0; i < Sf.size(); i++) {
            double t = Sf[i] >= trepfac * mx ? 1.0 : 0.0;
            if (sw == 1) { a += t * Sf[i]; b += t * Sf[i] * f[i]; }
            else { a += t * Sf[i] / std::max(f[i], 0.001); b += t * Sf[i]; }
        }
        return a / b;
    }

    // ---- spectral_wave_bc
    int generate(double t, double dt) {
        Params &p = *P;
        double endold = -123;
        bccount++;
        if (t > p.tstop - dt) return bccount - 1;
        if (reuseall) {
            endold = spectrumendtime; spectrumendtime += rtbc_local;
        } else {
            printf("Calculating spectral wave boundary conditions\n"); fflush(stdout);
            // calculate_average_water_depth from the current zs0, zb at i = 1
            std::vector<real> z0(NY1), zb(NY1);
            CK(cudaMemcpy2D(z0.data(), sizeof(real), d_zs0, G->NX1 * sizeof(real), sizeof(real), NY1, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy2D(zb.data(), sizeof(real), d_zb, G->NX1 * sizeof(real), sizeof(real), NY1, cudaMemcpyDeviceToHost));
            double a = 0, b = 0;
            for (int j = 1; j <= NY1; j++) { a += ((double)z0[j - 1] - zb[j - 1]) * G->dnc[G->I(1, j)]; b += G->dnc[G->I(1, j)]; }
            double h0 = a / b;
            std::vector<SpecIn> specin(nspectra), specint(nspectra);
            std::vector<double> endarr(nspectra, spectrumendtime);
            double fmax = 1, rtbc = p.rt, dtbc = p.dtbc;
            for (int il = 0; il < nspectra; il++) {
                while (t >= endarr[il]) {
                    read_spectrum_input(bcfiles[il], rtbc, dtbc, specin[il]);
                    if (il == 0) { endold = spectrumendtime; spectrumendtime += rtbc; }
                    endarr[il] += rtbc;
                }
                fmax = std::max(fmax, *std::max_element(specin[il].f.begin(), specin[il].f.end()));
            }
            for (int il = 0; il < nspectra; il++) {
                interpolate_spectrum(specin[il], specint[il], fmax);
                printf("  spectrum %d: Hm0 %.2f m, Trep %.2f s, mean dir %.2f degN\n", il + 1, specint[il].hm0, specint[il].trep, specint[il].dirm);
            }
            reuseall = true; for (auto &f : bcfiles) if (!f.reuse) reuseall = false;
            std::string nhname = bcname();
            // generate_combined_spectrum
            SpecIn cs; cs.f = specint[0].f; cs.nf = NFINT; cs.df = specint[0].df; cs.ang = specint[0].ang; cs.nang = NAINT; cs.dang = specint[0].dang;
            cs.S.assign((size_t)NFINT * NAINT, 0); cs.Sf.assign(NFINT, 0); cs.trep = 0;
            for (int il = 0; il < nspectra; il++) {
                for (size_t k = 0; k < cs.S.size(); k++) cs.S[k] = cs.S[k] + specint[il].S[k] / nspectra;
                for (int i = 0; i < NFINT; i++) cs.Sf[i] = cs.Sf[i] + specint[il].Sf[i] / nspectra;
                cs.trep = tpDcalc(cs.Sf, cs.f, p.trepfac, p.Tm01switch);
            }
            cs.Sd.assign(NAINT, 0);
            for (int j = 0; j < NAINT; j++) { double s = 0; for (int i = 0; i < NFINT; i++) s += cs.s(i, j); cs.Sd[j] = s * cs.df; }
            int il = (int)(std::max_element(cs.Sd.begin(), cs.Sd.end()) - cs.Sd.begin());
            double pS[3], pA[3];
            if (il > 0 && il < NAINT - 1) { for (int q = 0; q < 3; q++) { pS[q] = cs.Sd[il - 1 + q]; pA[q] = cs.ang[il - 1 + q]; } }
            else if (il == 0) { pS[0] = cs.Sd[NAINT - 1]; pS[1] = cs.Sd[0]; pS[2] = cs.Sd[1]; pA[0] = cs.ang[NAINT - 1] - 2 * M_PI; pA[1] = cs.ang[0]; pA[2] = cs.ang[1]; }
            else { pS[0] = cs.Sd[NAINT - 2]; pS[1] = cs.Sd[NAINT - 1]; pS[2] = cs.Sd[0]; pA[0] = cs.ang[NAINT - 2]; pA[1] = cs.ang[NAINT - 1]; pA[2] = cs.ang[0] + 2 * M_PI; }
            cs.dir0 = (pS[0] * pA[0] + pS[1] * pA[1] + pS[2] * pA[2]) / (pS[0] + pS[1] + pS[2]);
            if (cs.dir0 > 2 * M_PI) cs.dir0 -= 2 * M_PI; else if (cs.dir0 < 0) cs.dir0 += 2 * M_PI;
            double m0 = 0; for (double v : cs.Sf) m0 += v; m0 *= cs.df; cs.hm0 = 4 * sqrt(m0);
            p.Trep = cs.trep; p.Hrms = cs.hm0 / sqrt(2.0);
            printf("  overall Trep %.2f s\n", p.Trep);
            Wave w = make_wave(cs, h0, rtbc, dtbc, specin, specint);
            double t0w = ws_now();
            write_nh(w, nhname, t, dt);
            printf("  timing: write %.1f s\n", ws_now() - t0w); fflush(stdout);
            rtbc_local = w.rtbc; dtbc_local = w.dtbc; maindir_local = cs.dir0;
        }
        // generate_admin_files
        FILE *f = fopen("nhbcflist.bcf", bccount == 1 ? "w" : "a");
        if (!f) fail("cannot write nhbcflist.bcf");
        fprintf(f, "%12.3f   %12.3f %12.3f %12.3f %s\n", endold, spectrumendtime, p.Trep, p.Hrms, bcname().c_str());
        fclose(f);
        return bccount;
    }

    struct Wave {
        double h0, rtbc, dtbc, dtin, dfgen; int K, tslen, tslenbc;
        std::vector<double> fgen, thetagen, phigen, kgen, wgen, tin, taperf, Hm0interp;
        std::vector<std::vector<double>> vargen, vargenq;
        std::vector<double> A, Sfinterp, Sfinterpq;    // (NY1 x K), A[k*NY1 + j]
        std::vector<int> Findex;
        std::vector<cplx> CompFn1;                     // CompFn(1,:) (offshore point j = 1), length tslen
        std::vector<double> zs, u, v, du, dv;          // (tslen x NY1), [it*NY1 + j]
    };
    Wave make_wave(const SpecIn &cs, double h0, double rtbc, double dtbc, std::vector<SpecIn> &specin, std::vector<SpecIn> &specint) {
        Params &p = *P;
        Wave w; w.h0 = h0; w.rtbc = rtbc; w.dtbc = dtbc;
        double tc0 = ws_now();
        // generate_wavetrain_components
        double kmax = WDMAX / h0, fmax = p.g * kmax * tanh(kmax * h0) / 2 / M_PI;
        int i1 = 0, i2 = 0;
        {   // frange
            double mx = *std::max_element(cs.Sf.begin(), cs.Sf.end());
            std::vector<int> fl(NFINT, 0);
            for (int i = 0; i < NFINT; i++) { if (cs.Sf[i] > p.sprdthr * mx) fl[i] = 1; if (cs.f[i] > fmax) fl[i] = 0; }
            i1 = 0; for (int i = 0; i < NFINT; i++) if (fl[i] > fl[i1]) { i1 = i; }
            for (int i = 0; i < NFINT; i++) if (fl[i] == 1) { i1 = i; break; }
            i2 = 0; double best = -1; for (int i = 0; i < NFINT; i++) { double v = (double)(i + 1) * fl[i]; if (v > best) { best = v; i2 = i; } }
        }
        w.K = std::max((int)ceil(rtbc * (cs.f[i2] - cs.f[i1]) + 1), KMIN);
        int K = w.K;
        w.dfgen = (cs.f[i2] - cs.f[i1]) / (K - 1);
        w.fgen.resize(K); for (int i = 0; i < K; i++) w.fgen[i] = cs.f[i1] + i * w.dfgen;
        std::vector<double> rn(2 * K);
        if (p.random == 1) rn[0] = xb_random((int)(time(nullptr) & 0x7fffffff) | 1);
        else rn[0] = xb_random(100);
        for (int i = 1; i < 2 * K; i++) rn[i] = xb_random(0);
        w.phigen.resize(K); for (int i = 0; i < K; i++) w.phigen[i] = rn[i] * 2 * M_PI;
        w.thetagen.resize(K);
        std::vector<double> pdf(NAINT), cdf(NAINT);
        for (int i = 0; i < K; i++) {
            double s = 0;
            for (int ii = 0; ii < NAINT; ii++) { pdf[ii] = lin_interp(cs.f.data(), &cs.S[(size_t)NFINT * ii], NFINT, w.fgen[i]); s += pdf[ii]; }
            for (auto &x : pdf) x = x / s;
            cdf[0] = pdf[0];
            for (int ii = 1; ii < NAINT; ii++) cdf[ii] = cdf[ii - 1] + (pdf[ii] + pdf[ii - 1]) / 2;
            if (rn[K + i] >= cdf[0]) w.thetagen[i] = lin_interp(cdf.data(), cs.ang.data(), NAINT, rn[K + i]);
            else {
                double xx[2] = {0.0, cdf[0]}, yy[2] = {cs.ang[NAINT - 1] - 2 * M_PI, cs.ang[0]};
                w.thetagen[i] = lin_interp(xx, yy, 2, rn[K + i]);
                if (w.thetagen[i] < 0) w.thetagen[i] += 2 * M_PI;
            }
        }
        const double aphi = 1.0 / (((1.0 + sqrt(5.0)) / 2) + 1), bphi = ((1.0 + sqrt(5.0)) / 2) / (((1.0 + sqrt(5.0)) / 2) + 1);
        w.kgen.resize(K); w.wgen.resize(K);
        for (int i = 0; i < K; i++) {
            double L0 = p.g * pow(1 / w.fgen[i], 2) / 2 / M_PI, L1 = L0, err = 1e300; int it = 0;
            while (err > 0.00001 && it < 150) { it++; double L2 = L0 * tanh(2 * M_PI * h0 / L1); L1 = L1 * aphi + L2 * bphi; err = fabs(L2 - L1); }
            w.kgen[i] = 2 * M_PI / L1;
            w.wgen[i] = 2 * M_PI * w.fgen[i];
        }
        if (p.cyclicdiradjust == 1) {
            int jn = std::max(G->ny - 2, 1);
            double dx = G->xz[G->I(1, jn)] - G->xz[G->I(1, 1)], dy = G->yz[G->I(1, jn)] - G->yz[G->I(1, 1)];
            double Lg = sqrt(dx * dx + dy * dy);
            if (Lg > 1e-3) {
                double ag = atan2(-dx, dy);
                for (int i = 0; i < K; i++) {
                    double aw = w.thetagen[i] - ag, Lw = 2 * M_PI / w.kgen[i], Lp = Lw / sin(aw), nr = Lg / Lp;
                    int ni = (int)lround(nr);
                    ni = std::min(abs(ni), (int)floor(Lg / Lw)) * (ni >= 0 ? 1 : -1);
                    w.thetagen[i] = ni != 0 ? asin(Lw / (Lg / ni)) + ag : ag;
                }
            }
        }
        if (nspectra == 1 && specin[0].scoeff > 1000) for (auto &x : w.thetagen) x = specin[0].dir0;
        // generate_wave_time_axis
        w.dtin = dtbc;
        w.tslenbc = (int)lround(rtbc / dtbc) + 1;
        if (w.dtin > 0.1 / w.fgen[K - 1]) w.dtin = 0.1 / w.fgen[K - 1];
        w.tslen = (int)ceil(1 / w.dfgen / w.dtin) + 1;
        if (w.tslen % 2 != 0) w.tslen++;
        w.tin.resize(w.tslen); for (int i = 0; i < w.tslen; i++) w.tin[i] = i * w.dtin;
        w.taperf.assign(w.tslen, 1.0);
        int ntaper = (int)lround((2.0 * p.Trep) / w.dtin);
        for (int i = 1; i <= std::min(ntaper, w.tslen); i++) w.taperf[i - 1] = tanh(5.0 * i / ntaper);
        int indend = w.tslen;
        if (w.tin[w.tslen - 1] > rtbc) { for (int i = 0; i < w.tslen; i++) if (w.tin[i] >= rtbc) { indend = i + 1; break; } }
        for (int i = 1; i <= std::min(ntaper, indend); i++) w.taperf[indend - i] = std::min(w.taperf[indend - i], tanh(5.0 * i / ntaper));
        for (int i = indend; i <= w.tslen; i++) w.taperf[i - 1] = 0;
        // generate_wave_train_variance
        w.vargen.resize(nspectra); w.vargenq.resize(nspectra);
        for (int s = 0; s < nspectra; s++) {
            w.vargen[s].resize(K); w.vargenq[s].resize(K);
            for (int ii = 0; ii < K; ii++) w.vargen[s][ii] = lin_interp(specint[s].f.data(), specint[s].Sf.data(), NFINT, w.fgen[ii]);
            double sv = 0; for (double v : w.vargen[s]) sv += v;
            double hp = 4 * sqrt(sv * w.dfgen);
            for (auto &v : w.vargen[s]) v = pow(specint[s].hm0 / hp, 2) * v;
            for (int ii = 0; ii < K; ii++) w.vargenq[s][ii] = std::min(w.vargen[s][ii], lin_interp(specint[s].f.data(), specint[s].Sf.data(), NFINT, w.fgen[ii]));
        }
        // generate_wave_train_properties_per_offshore_point
        w.A.assign((size_t)NY1 * K, 0); w.Sfinterp = w.A; w.Sfinterpq = w.A; w.Hm0interp.assign(NY1, 0);
        int lmin = *std::min_element(n_index_loc.begin(), n_index_loc.end()), lmax = *std::max_element(n_index_loc.begin(), n_index_loc.end());
        for (int i = 1; i <= NY1; i++) {
            int ia = -1, ib = -1; bool interp_ = false;
            for (int q = 0; q < nspectra; q++) if (n_index_loc[q] == i) ia = q;
            if (ia < 0 && i < lmin) { for (int q = 0; q < nspectra; q++) if (n_index_loc[q] == lmin) ia = q; }
            else if (ia < 0 && i > lmax) { for (int q = 0; q < nspectra; q++) if (n_index_loc[q] == lmax) ia = q; }
            else if (ia < 0) {
                interp_ = true;
                int best = -1; for (int q = 0; q < nspectra; q++) { int v = n_index_loc[q] > i ? -2147483647 : n_index_loc[q]; if (ia < 0 || v > best) { best = v; ia = q; } }
                int bestb = 0; ib = -1; for (int q = 0; q < nspectra; q++) { int v = n_index_loc[q] < i ? 2147483647 : n_index_loc[q]; if (ib < 0 || v < bestb) { bestb = v; ib = q; } }
            }
            double *Sfi = &w.Sfinterp[(size_t)(i - 1)], *Sfq = &w.Sfinterpq[(size_t)(i - 1)];
            if (interp_) {
                double pos[2] = {G->ndist[G->I(1, n_index_loc[ia])], G->ndist[G->I(1, n_index_loc[ib])]}, here = G->ndist[G->I(1, i)];
                double sfnow = 0;
                for (int ii = 0; ii < K; ii++) {
                    double s1[2] = {w.vargen[ia][ii], w.vargen[ib][ii]}, s2[2] = {w.vargenq[ia][ii], w.vargenq[ib][ii]};
                    Sfi[(size_t)ii * NY1] = lin_interp(pos, s1, 2, here); Sfq[(size_t)ii * NY1] = lin_interp(pos, s2, 2, here);
                    sfnow += Sfi[(size_t)ii * NY1];
                }
                double sa = 0, sb = 0; for (int ii = 0; ii < K; ii++) { sa += w.vargen[ia][ii]; sb += w.vargen[ib][ii]; }
                double sfa[2] = {sa, sb}, sfimp = lin_interp(pos, sfa, 2, here);
                for (int ii = 0; ii < K; ii++) Sfi[(size_t)ii * NY1] = Sfi[(size_t)ii * NY1] * sfimp / sfnow;
            } else for (int ii = 0; ii < K; ii++) { Sfi[(size_t)ii * NY1] = w.vargen[ia][ii]; Sfq[(size_t)ii * NY1] = w.vargenq[ia][ii]; }
            double s = 0;
            for (int ii = 0; ii < K; ii++) { w.A[(size_t)ii * NY1 + i - 1] = sqrt(2 * Sfi[(size_t)ii * NY1] * w.dfgen); s += Sfi[(size_t)ii * NY1]; }
            w.Hm0interp[i - 1] = 4 * sqrt(s * w.dfgen);
        }
        // generate_wave_train_Fourier (only CompFn(1,:) is used afterwards, for the bound-wave phases)
        w.Findex.resize(K);
        int tempi = (int)floor(w.fgen[0] / w.dfgen);
        for (int i = 0; i < K; i++) w.Findex[i] = tempi + i + 1;
        w.CompFn1.assign(w.tslen, 0);
        for (int i = 0; i < K; i++) w.CompFn1[w.Findex[i] - 1] = w.A[(size_t)i * NY1 + 0] / 2 * std::exp(cplx(0, 1) * w.phigen[i]);
        // distribute_wave_train_directions (nonh: one bin [-pi, pi])
        for (int i = 0; i < K; i++) {
            while (w.thetagen[i] < -M_PI) w.thetagen[i] += 2 * M_PI;
            while (w.thetagen[i] > M_PI) w.thetagen[i] -= 2 * M_PI;
        }
        if (p.nspr == 1) for (auto &x : w.thetagen) x = (x >= -M_PI && x < M_PI) || x == M_PI ? 0.0 : -M_PI;
        // generate_swts on the GPU (bclwonly = 1: no phase-resolved waves, bound long waves only)
        double t0 = ws_now();
        printf("  K %d, tslen %d, boundary points %d\n", K, w.tslen, NY1);
        if (p.bclwonly == 0) synth(w);
        else { size_t M = (size_t)NY1 * w.tslen; for (auto *v : {&w.zs, &w.u, &w.v, &w.du, &w.dv}) v->assign(M, 0.0); }
        double t1 = ws_now();
        if (p.order > 1) qbcf(w);
        double t2 = ws_now();
        if (p.highcomp == 1 && p.order > 1) secondorder(w);
        printf("  timing: components %.1f s, synthesis %.1f s, bound long waves %.1f s, second order %.1f s\n", t0 - tc0, t1 - t0, t2 - t1, ws_now() - t2); fflush(stdout);
        return w;
    }
    void synth(Wave &w) {
        Params &p = *P;
        int K = w.K;
        std::vector<double> dx(NY1), dy(NY1);
        for (int j = 1; j <= NY1; j++) { dx[j - 1] = G->xz[G->I(1, j)] - G->xz[G->I(1, 1)]; dy[j - 1] = G->yz[G->I(1, j)] - G->yz[G->I(1, 1)]; }
        auto up = [](const std::vector<double> &h) { double *d_; CK(cudaMalloc(&d_, std::max<size_t>(1, h.size()) * 8)); CK(cudaMemcpy(d_, h.data(), h.size() * 8, cudaMemcpyHostToDevice)); return d_; };
        std::vector<double> th3k(3 * (size_t)K, 0.0); std::copy(w.thetagen.begin(), w.thetagen.end(), th3k.begin());
        double *dA = up(w.A), *dw = up(w.wgen), *dk = up(w.kgen), *dt_ = up(th3k), *dph = up(w.phigen), *dtin = up(w.tin), *ddx = up(dx), *ddy = up(dy);
        k_swts_trig<<<(K + 127) / 128, 128>>>(K, dt_);
        size_t M = (size_t)NY1 * w.tslen;
        double *o[5];
        for (int q = 0; q < 5; q++) CK(cudaMalloc(&o[q], M * 8));
        SwtsArgs a{K, NY1, w.tslen, (p.nonhq3d == 1 && p.nhlay > 0) ? 1 : 0, w.h0, p.nhlay};
        k_swts<<<(unsigned)((M + 127) / 128), 128>>>(a, dA, dw, dk, dt_, dph, dtin, ddx, ddy, o[0], o[1], o[2], o[3], o[4]);
        CK(cudaGetLastError());
        std::vector<double> *dst[5] = {&w.zs, &w.u, &w.v, &w.du, &w.dv};
        for (int q = 0; q < 5; q++) { dst[q]->resize(M); CK(cudaMemcpy(dst[q]->data(), o[q], M * 8, cudaMemcpyDeviceToHost)); CK(cudaFree(o[q])); }
        for (double *pp : {dA, dw, dk, dt_, dph, dtin, ddx, ddy}) CK(cudaFree(pp));
        for (int it = 0; it < w.tslen; it++) for (int j = 0; j < NY1; j++) {
            size_t k = (size_t)it * NY1 + j; double tf = w.taperf[it];
            w.u[k] *= tf; w.v[k] *= tf; w.zs[k] *= tf; w.du[k] *= tf; w.dv[k] *= tf;
        }
    }
    // generate_qbcf (nonhspectrum = 1): bound long waves added to the time series
    void qbcf(Wave &w) {
        Params &p = *P;
        int K = w.K, half = w.tslen / 2;
        auto IX = [&](int m, int k) { return (size_t)(m - 1) + (size_t)(K - 1) * (k - 1); };   // (K-1) x K
        std::vector<double> D((size_t)(K - 1) * K, 0), dth = D, KKx = D, KKy = D, dphi3 = D, k3 = D, cg3 = D, th3 = D;
        for (int m = 1; m <= K - 1; m++) {
            double deltaf = m * w.dfgen;
            for (int k = 1; k <= K - m; k++) {
                int a = k - 1, b = k + m - 1;   // components k and k+m
                size_t q = IX(m, k);
                dth[q] = fabs(w.thetagen[b] - w.thetagen[a]) + M_PI;
                KKy[q] = w.kgen[b] * sin(w.thetagen[b]) - w.kgen[a] * sin(w.thetagen[a]);
                KKx[q] = w.kgen[b] * cos(w.thetagen[b]) - w.kgen[a] * cos(w.thetagen[a]);
                k3[q] = sqrt(w.kgen[a] * w.kgen[a] + w.kgen[b] * w.kgen[b] + 2 * w.kgen[a] * w.kgen[b] * cos(dth[q]));
                cg3[q] = 2.0 * M_PI * deltaf / k3[q];
                cg3[q] = std::min(cg3[q], p.nmax * sqrt(p.g / k3[q] * tanh(k3[q] * w.h0)));
                double term1 = (-w.wgen[a]) * w.wgen[b], term2 = (-w.wgen[a]) + w.wgen[b], term2new = cg3[q] * k3[q];
                double chk1 = cosh(w.kgen[a] * w.h0), chk2 = cosh(w.kgen[b] * w.h0);
                D[q] = -p.g * w.kgen[a] * w.kgen[b] * cos(dth[q]) / 2.0 / term1 + term2 * term2 / (p.g * 2) + p.g * term2 /
                       ((p.g * k3[q] * tanh(k3[q] * w.h0) - term2new * term2new) * term1) *
                       (term2 * (term1 * term1 / p.g / p.g - w.kgen[a] * w.kgen[b] * cos(dth[q]))
                        - 0.50 * ((-w.wgen[a]) * w.kgen[b] * w.kgen[b] / (chk2 * chk2) + w.wgen[b] * w.kgen[a] * w.kgen[a] / (chk1 * chk1)));
                cplx c1 = std::conj(w.CompFn1[w.Findex[0] - 1 + m + k - 1]), c2 = std::conj(w.CompFn1[w.Findex[0] - 1 + k - 1]);
                dphi3[q] = std::imag(std::log(c1)) - std::imag(std::log(c2));
            }
            for (int k = 1; k <= K; k++) if (w.fgen[k - 1] <= deltaf) D[IX(m, k)] = 0;
            if (deltaf <= p.fcutoff) for (int k = 1; k <= K; k++) D[IX(m, k)] = 0;
        }
        for (size_t q = 0; q < th3.size(); q++) th3[q] = atan2(KKy[q], KKx[q]);
        std::vector<double> qv((size_t)NY1 * w.tslen * 4, 0);   // q(j,it,iq) -> [(iq*tslen + it)*NY1 + j]
        auto up = [](const std::vector<double> &h) { double *d_; CK(cudaMalloc(&d_, std::max<size_t>(1, h.size()) * 8)); CK(cudaMemcpy(d_, h.data(), h.size() * 8, cudaMemcpyHostToDevice)); return d_; };
        double *dD = up(D), *dS = up(p.Sfold == 1 ? w.Sfinterpq : w.Sfinterp), *dph = up(dphi3), *dcg = up(cg3), *dth3 = up(th3),
               *dkx = up(KKx), *dky = up(KKy), *dout;
        CK(cudaMalloc(&dout, (size_t)8 * 2 * std::max(1, K - 1) * 8));
        const size_t nq_ = (size_t)(K - 1) * K;
        double *dpre; CK(cudaMalloc(&dpre, std::max<size_t>(1, 4 * nq_) * 8));
        if (nq_) k_qbcf_pre<<<(unsigned)((nq_ + 255) / 256), 256>>>(nq_, dph, dcg, dth3, dpre);
        const size_t GNL = (size_t)8 * std::max(1, K - 1);
        std::vector<double> gall(GNL * NY1);
        {   // offsets of the offshore points (same expressions as before), then rows in blocks of up to 512
            std::vector<double> dxa(NY1), dya(NY1);
            for (int j = 1; j <= NY1; j++) { dxa[j - 1] = G->xz[G->I(1, j)] - G->xz[G->I(1, 1)]; dya[j - 1] = G->yz[G->I(1, j)] - G->yz[G->I(1, 1)]; }
            double *ddx = up(dxa), *ddy = up(dya), *dblk;
            const int RB = 512;
            CK(cudaMalloc(&dblk, GNL * RB * 8));
            for (int j0 = 0; j0 < NY1; j0 += RB) {
                int nj = std::min(RB, NY1 - j0);
                dim3 grid((K - 1 + 127) / 128, nj);
                k_qbcf_gn_rows<<<grid, 128>>>(K, NY1, j0, nj, w.dfgen, ddx, ddy, dD, dS, dpre, dcg, dkx, dky, dblk);
                CK(cudaGetLastError());
                CK(cudaMemcpy(&gall[GNL * j0], dblk, GNL * nj * 8, cudaMemcpyDeviceToHost));
            }
            for (double *pp : {ddx, ddy, dblk}) CK(cudaFree(pp));
        }
        #pragma omp parallel
        {
        std::vector<cplx> Gn(w.tslen, 0);
        #pragma omp for schedule(dynamic, 8)
        for (int j = 1; j <= NY1; j++) {
            const double *gn = &gall[GNL * (j - 1)];
            for (int iq = 0; iq < 4; iq++) {
                std::fill(Gn.begin(), Gn.end(), cplx(0, 0));
                for (int m = 1; m <= K - 1; m++) Gn[m] = cplx(gn[(size_t)iq * 2 * (K - 1) + 2 * (m - 1)], gn[(size_t)iq * 2 * (K - 1) + 2 * (m - 1) + 1]);
                for (int q = 1; q < half; q++) Gn[w.tslen - q] = std::conj(Gn[q]);   // Gn(half+2:tslen) = flip(conj(Gn(2:half)))
                std::vector<cplx> o = ifft_unnorm(Gn);
                for (int it = 0; it < w.tslen; it++) qv[((size_t)iq * w.tslen + it) * NY1 + j - 1] = o[it].real() * w.taperf[it];
            }
        }
        }
        for (double *pp : {dD, dS, dph, dcg, dth3, dkx, dky, dout, dpre}) CK(cudaFree(pp));
        if (p.wbcRemoveStokes == 1) {
            int nin = (int)lround(w.rtbc / w.dtin);
            for (int iq = 0; iq < 4; iq++) for (int j = 0; j < NY1; j++) {
                double s = 0; for (int it = 0; it < nin; it++) s += qv[((size_t)iq * w.tslen + it) * NY1 + j];
                double qm = s / nin; for (int it = 0; it < w.tslen; it++) qv[((size_t)iq * w.tslen + it) * NY1 + j] -= qm;
            }
        }
        if (p.wbcQvarreduce < 1.0 - 1e-10) {
            for (int iq = 0; iq < 4; iq++) for (int j = 0; j < NY1; j++) {
                double s = 0; for (int it = 0; it < w.tslen; it++) s += qv[((size_t)iq * w.tslen + it) * NY1 + j];
                double qm = s / w.tslen;
                for (int it = 0; it < w.tslen; it++) { double &x = qv[((size_t)iq * w.tslen + it) * NY1 + j]; x = p.wbcQvarreduce * (x - qm) + qm; }
            }
        }
        for (int it = 0; it < w.tslen; it++) for (int j = 0; j < NY1; j++) {
            size_t k = (size_t)it * NY1 + j;
            w.u[k] = w.u[k] + qv[((size_t)0 * w.tslen + it) * NY1 + j] / w.h0;
            w.v[k] = w.v[k] + qv[((size_t)1 * w.tslen + it) * NY1 + j] / w.h0;
            w.zs[k] = w.zs[k] + qv[((size_t)3 * w.tslen + it) * NY1 + j];
        }
    }
    // generate_secondorder (highcomp = 1): bound super-harmonics of all component pairs (ii, jj) with ii + jj = m,
    // added to the boundary series. XBeach takes the phases of CompFn(Findex(1)+ii), one bin above component ii,
    // places row m at Gn(2 Findex(1) + m - 1), blanks a whole row when k3(m,1) h0 > 5 and never adds the v flux;
    // all reproduced.
    void secondorder(Wave &w) {
        Params &p = *P;
        const int K = w.K, K2 = 2 * K, half = w.tslen / 2;
        const bool q3 = p.nonhq3d == 1 && p.nhlay > 0;
        auto IX = [&](int m, int ii) { return (size_t)(m - 1) + (size_t)K2 * (ii - 1); };   // (2K x 2K)
        const size_t NE = (size_t)K2 * K2;
        std::vector<double> D(NE, 0), dth(NE, 0), KKx(NE, 0), KKy(NE, 0), dphi3(NE, 0), k3(NE, 0), c(NE, 0), th3(NE, 0);
        auto lo = [&](int m) { return std::max(1, m - K); };
        auto hi = [&](int m) { return std::min(m - 1, K); };
        for (int m = 2; m <= K2; m++) {
            for (int ii = lo(m); ii <= hi(m); ii++) {
                int jj = m - ii, a = ii - 1, b = jj - 1;
                size_t q = IX(m, ii);
                dth[q] = fabs(w.thetagen[a] - w.thetagen[b]);
                KKy[q] = w.kgen[a] * sin(w.thetagen[a]) + w.kgen[b] * sin(w.thetagen[b]);
                KKx[q] = w.kgen[a] * cos(w.thetagen[a]) + w.kgen[b] * cos(w.thetagen[b]);
                k3[q] = sqrt(w.kgen[a] * w.kgen[a] + w.kgen[b] * w.kgen[b] + 2 * w.kgen[a] * w.kgen[b] * cos(dth[q]));
                double w3 = w.wgen[a] + w.wgen[b];
                c[q] = w3 / k3[q];
                double term1 = w.wgen[a] * w.wgen[b], term2 = w.wgen[a] + w.wgen[b];
                double chk1 = cosh(w.kgen[a] * w.h0), chk2 = cosh(w.kgen[b] * w.h0);
                D[q] = -p.g * w.kgen[a] * w.kgen[b] * cos(dth[q]) / 2.0 / term1 + term2 * term2 / (2 * p.g) + p.g * term2 /
                       ((p.g * k3[q] * tanh(k3[q] * w.h0) - term2 * term2) * term1) *
                       (term2 * (term1 * term1 / p.g / p.g - w.kgen[a] * w.kgen[b] * cos(dth[q]))
                        - 0.50 * (w.wgen[a] * w.kgen[b] * w.kgen[b] / (chk2 * chk2) + w.wgen[b] * w.kgen[a] * w.kgen[a] / (chk1 * chk1)));
            }
            if (k3[IX(m, 1)] * w.h0 > 5) for (int ii = 1; ii <= K2; ii++) D[IX(m, ii)] = 0;
            for (int ii = lo(m); ii <= hi(m); ii++) {
                int jj = m - ii;
                auto cf = [&](int idx) { return idx - 1 < (int)w.CompFn1.size() ? std::conj(w.CompFn1[idx - 1]) : cplx(0, 0); };
                dphi3[IX(m, ii)] = std::imag(std::log(cf(w.Findex[0] + ii))) + std::imag(std::log(cf(w.Findex[0] + jj)));
            }
        }
        for (size_t q = 0; q < NE; q++) th3[q] = atan2(KKy[q], KKx[q]);
        std::vector<cplx> E1(NE);
        for (size_t q = 0; q < NE; q++) E1[q] = std::exp(cplx(0, -1) * dphi3[q]);
        const int nq = q3 ? 6 : 4;
        std::vector<double> qv((size_t)nq * w.tslen * NY1, 0);
        // XBeach never resets Gn between components or rows; every entry read back is rewritten in the same
        // iteration (all rows m, then the conjugate mirror) or never written (0), so a per-thread Gn is identical
        const double z = p.nonhq3d == 1 ? -1 * (w.h0 - p.nhlay * w.h0) : 0.0;
        const std::vector<double> &Sf = p.Sfold == 1 ? w.Sfinterpq : w.Sfinterp;
        #pragma omp parallel
        {
        std::vector<cplx> Gn(w.tslen, cplx(0, 0)), E2(NE);
        std::vector<double> Ef(NE), qx1, qx2, qy1, qy2;
        #pragma omp for schedule(dynamic, 4)
        for (int j = 1; j <= NY1; j++) {
            std::fill(Ef.begin(), Ef.end(), 0.0);
            if (q3) { qx1.assign(NE, 0); qx2.assign(NE, 0); qy1.assign(NE, 0); qy2.assign(NE, 0); }
            for (int m = 2; m <= K2; m++)
                for (int ii = lo(m); ii <= hi(m); ii++) {
                    int jj = m - ii;
                    size_t q = IX(m, ii);
                    Ef[q] = D[q] * D[q] * Sf[(size_t)(ii - 1) * NY1 + (j - 1)] * Sf[(size_t)(jj - 1) * NY1 + (j - 1)] * w.dfgen;
                    if (p.nonhq3d == 1) {
                        qx1[q] = sinh(KKx[q] * (z + w.h0)) / sinh(KKx[q] * w.h0);
                        qx2[q] = (sinh(KKx[q] * w.h0) - sinh(KKx[q])) / sinh(KKx[q] * w.h0);
                        qy1[q] = sinh(KKy[q] * (z + w.h0)) / sinh(KKy[q] * w.h0);
                        qy2[q] = (sinh(KKy[q] * w.h0) - sinh(KKy[q])) / sinh(KKy[q] * w.h0);
                    }
                }
            double dxj = G->xz[G->I(1, j)] - G->xz[G->I(1, 1)], dyj = G->yz[G->I(1, j)] - G->yz[G->I(1, 1)];
            for (int m = 2; m <= K2; m++) for (int ii = lo(m); ii <= hi(m); ii++) {
                size_t q = IX(m, ii); E2[q] = std::exp(cplx(0, -1) * (KKy[q] * dyj + KKx[q] * dxj));
            }
            for (int iq = 0; iq < nq; iq++) {
                for (int m = 1; m <= K2; m++) {
                    cplx s(0, 0);   // sum over ii of row m; entries outside the pairs are exact zeros
                    for (int ii = (m >= 2 ? lo(m) : 1); ii <= (m >= 2 ? hi(m) : 0); ii++) {
                        size_t q = IX(m, ii);
                        double Ab = sqrt(2 * Ef[q] * w.dfgen) * (D[q] >= 0 ? 1.0 : -1.0);
                        cplx f = Ab / 2 * E1[q];
                        if (q3) {
                            switch (iq) {
                            case 0: f = f * c[q] * cos(th3[q]) * qx1[q]; break;
                            case 1: f = f * c[q] * cos(th3[q]) * qx2[q]; break;
                            case 2: f = f * c[q] * sin(th3[q]) * qy1[q]; break;
                            case 3: f = f * c[q] * sin(th3[q]) * qy2[q]; break;
                            case 4: f = f * c[q]; break;
                            default: break;
                            }
                        } else {
                            if (iq == 0) f = f * c[q] * cos(th3[q]);
                            else if (iq == 1) f = f * c[q] * sin(th3[q]);
                            else if (iq == 2) f = f * c[q];
                        }
                        f = f * E2[q];
                        s += f;
                    }
                    size_t g = (size_t)2 * w.Findex[0] + m - 2;   // Gn(2 Findex(1) + m - 1), 0-based
                    if (g < Gn.size()) Gn[g] = s;
                }
                for (int q = 1; q < half; q++) Gn[w.tslen - q] = std::conj(Gn[q]);   // Gn(half+2:tslen) = flip(conj(Gn(2:half)))
                std::vector<cplx> o = ifft_unnorm(Gn);
                for (int it = 0; it < w.tslen; it++) qv[((size_t)iq * w.tslen + it) * NY1 + j - 1] = o[it].real() * w.taperf[it];
            }
        }
        }
        for (int it = 0; it < w.tslen; it++) for (int j = 0; j < NY1; j++) {
            size_t k = (size_t)it * NY1 + j;
            auto Q = [&](int iq) { return qv[((size_t)iq * w.tslen + it) * NY1 + j]; };
            if (q3) {
                w.u[k] = w.u[k] + (Q(0) / z + Q(1) / (w.h0 + z)) / 2.0;
                w.du[k] = w.du[k] + (Q(0) / z - Q(1) / (w.h0 + z));
                w.zs[k] = w.zs[k] + Q(5);
            } else {
                w.u[k] = w.u[k] + Q(0) / w.h0;
                w.zs[k] = w.zs[k] + Q(3);
            }
        }
    }

    // generate_nhtimeseries_file: '(1X,E18.9)' per value
    void write_nh(const Wave &w, const std::string &fn, double t, double dt) {
        Params &p = *P;
        FILE *f = fopen(fn.c_str(), "w");
        if (!f) fail("cannot write " + fn);
        fprintf(f, "vector\n6\nt,U,V,Z,dU,dV\n");
        auto e18 = [](double v, char *b) {   // Fortran E18.9: ' 0.ddddddddd' with exponent
            if (!(fabs(v) >= 1e-300)) { snprintf(b, 24, "    0.000000000E+00"); return; }
            double a = fabs(v); int e = (int)floor(log10(a)) + 1;
            long long m = llround(a / pow(10.0, e) * 1e9);
            if (m >= 1000000000LL) { m /= 10; e++; }
            if (m < 100000000LL) { m *= 10; e--; }
            if (abs(e) < 100) snprintf(b, 24, "   %s0.%09dE%c%02d", v < 0 ? "-" : " ", (int)m, e < 0 ? '-' : '+', abs(e) % 100);
            else snprintf(b, 24, "   %s0.%09d%c%03d", v < 0 ? "-" : " ", (int)m, e < 0 ? '-' : '+', abs(e) % 1000);
        };
        auto fmt = [&](double tt, int it, std::string &line) {
            char b[24];
            line.clear(); line.reserve((size_t)19 * (5 * NY1 + 1) + 1); e18(tt, b); line += b;
            const std::vector<double> *a[5] = {&w.u, &w.v, &w.zs, &w.du, &w.dv};
            for (int q = 0; q < 5; q++) for (int j = 0; j < NY1; j++) { e18((*a[q])[(size_t)it * NY1 + j], b); line += b; }
            line += '\n';
        };
        // rows: (time, index) in file order
        std::vector<std::pair<double, int>> rows;
        rows.push_back({t - dt * p.maxdtfac, 0});
        for (int it = 0; it < w.tslen; it++) rows.push_back({w.tin[it] + t - dt, it});
        if (w.tin[w.tslen - 1] < w.rtbc + dt * p.maxdtfac) rows.push_back({w.tin[w.tslen - 1] + t + dt * p.maxdtfac, w.tslen - 1});
        const int CH = 256;
        std::vector<std::string> buf(CH);
        for (size_t r0 = 0; r0 < rows.size(); r0 += CH) {
            int nr = (int)std::min<size_t>(CH, rows.size() - r0);
            #pragma omp parallel for schedule(dynamic, 4)
            for (int r = 0; r < nr; r++) fmt(rows[r0 + r].first, rows[r0 + r].second, buf[r]);
            for (int r = 0; r < nr; r++) fwrite(buf[r].data(), 1, buf[r].size(), f);
        }
        fclose(f);
    }
};
static int spectral_wave_bc(SpectralGen *g, double t, double dt, Params &) { return g->generate(t, dt); }
