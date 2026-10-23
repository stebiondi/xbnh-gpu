// io.h: list-directed file readers, the XBeach hotstart file format, and small interpolation helpers.
#pragma once
#include <fstream>
#include <sstream>
#include <algorithm>

// buffered tokenizer for large list-directed Fortran files (separators: blanks, newlines, commas)
struct Tok {
    FILE *f; std::vector<char> b; size_t p = 0, n = 0; bool fe = false; std::string fn;
    explicit Tok(const std::string &fname) : b((1 << 22) + 1), fn(fname) {
        f = fopen(fname.c_str(), "rb");
        if (!f) fail("cannot open " + fname);
        b[0] = 0;
    }
    ~Tok() { if (f) fclose(f); }
    static bool sep(char ch) { return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t' || ch == ','; }
    void fill() {
        if (fe || n - p >= 256) return;
        memmove(b.data(), b.data() + p, n - p); n -= p; p = 0;
        size_t r = fread(b.data() + n, 1, b.size() - 1 - n, f);
        if (r == 0) fe = true;
        n += r; b[n] = 0;
    }
    bool skip() {
        while (true) {
            fill();
            while (p < n && sep(b[p])) p++;
            if (p < n) { fill(); return true; }
            if (fe) return false;
        }
    }
    bool num(double &v) {
        if (!skip()) return false;
        char *e; v = strtod(b.data() + p, &e);
        if (e == b.data() + p) {   // Fortran exponent letter d/D
            std::string w; size_t q = p; while (q < n && !sep(b[q])) w.push_back(b[q++]);
            for (auto &ch : w) if (ch == 'd' || ch == 'D') ch = 'e';
            char *e2; v = strtod(w.c_str(), &e2); if (e2 == w.c_str()) return false;
            p = q; return true;
        }
        if (*e == 'd' || *e == 'D') { std::string w(b.data() + p, e - (b.data() + p)); w += 'e';
            char *q = e + 1; while (*q && !sep(*q)) w.push_back(*q++); v = strtod(w.c_str(), nullptr); p = q - b.data(); return true; }
        p = e - b.data(); return true;
    }
    bool word(std::string &s) { if (!skip()) return false; s.clear(); while (p < n && !sep(b[p])) s.push_back(b[p++]); return true; }
    // skip to the start of the next line (list-directed read starts every READ at a new record)
    void nextline() { while (true) { fill(); while (p < n && b[p] != '\n') p++; if (p < n) { p++; return; } if (fe) return; } }
};

// read nx1*ny1 values; rowwise = true: each row is one Fortran READ (starts on a new line)
template <typename T>
static void read_grid_file(const std::string &fn, int nx1, int ny1, std::vector<T> &a, bool rowwise) {
    Tok t(fn); a.assign((size_t)nx1 * ny1, 0);
    for (int j = 0; j < ny1; j++) {
        for (int i = 0; i < nx1; i++) {
            double v; if (!t.num(v)) fail(fn + ": fewer than (nx+1)(ny+1) values");
            a[(size_t)j * nx1 + i] = (T)v;
        }
        if (rowwise) t.nextline();
    }
}

static std::vector<std::vector<double>> read_columns(const std::string &fn, int ncol) {
    std::ifstream f(fn);
    if (!f) fail("cannot open " + fn);
    std::vector<std::vector<double>> c(ncol);
    std::string line;
    while (std::getline(f, line)) {
        for (auto &ch : line) if (ch == ',' || ch == '\t') ch = ' ';
        std::istringstream is(line); std::vector<double> row; double x;
        while ((int)row.size() < ncol && (is >> x)) row.push_back(x);
        if ((int)row.size() < ncol) { if (row.empty()) continue; break; }
        for (int q = 0; q < ncol; q++) c[q].push_back(row[q]);
    }
    return c;
}

// interp.F90 binary_search + linear_interp (constant beyond the ends), same operation order as XBeach
static int xb_binary_search(const double *xx, int n, double x) {
    int jl = 0, ju = n + 1;
    bool l1 = xx[n - 1] > xx[0];
    while (ju - jl > 1) {
        int jm = (ju + jl) / 2;
        bool l2 = x > xx[jm - 1];
        if ((l1 && l2) || !(l1 || l2)) jl = jm; else ju = jm;
    }
    return jl;
}
static double lin_interp(const double *x, const double *y, int n, double xx) {
    if (n <= 0) return 0;
    if (n == 1) return y[0];
    int j = xb_binary_search(x, n, xx);
    if (j <= 0) return y[0];
    if (j >= n) return y[n - 1];
    double a = x[j], b = x[j - 1];
    double dyy = a == b ? 0.0 : (y[j] - y[j - 1]) / (a - b);
    return y[j - 1] + (xx - x[j - 1]) * dyy;
}
static double lin_interp(const std::vector<double> &x, const std::vector<double> &y, double xp) {
    if (x.empty()) return 0;
    return lin_interp(x.data(), y.data(), (int)x.size(), xp);
}

// initialize.F90 lowlevel_read_hotstart_2d: ny+1 rows of nx+1 fields '(1X,E15.8)' or '(1X,I4)'. Fixed-width
// parsing, because E15.8 drops the 'E' for three-digit exponents (e.g. 0.12345678-100).
static double fortran_real(const char *s, int w) {
    char buf[64]; int n = 0;
    for (int q = 0; q < w && s[q]; q++) if (s[q] != ' ') buf[n++] = s[q];
    buf[n] = 0;
    if (n == 0) return 0;
    char *e; double v = strtod(buf, &e);
    if (*e == '+' || *e == '-') v *= pow(10.0, atoi(e));
    return v;
}
template <typename T> static void read_hotstart(const std::string &var, int fileno, int NX1, int NY1, std::vector<T> &a,
                                                bool isint, int nlay = 1) {
    char fn[256]; snprintf(fn, sizeof fn, "hotstart_%s%06d.dat", var.c_str(), fileno);
    FILE *f = fopen(fn, "rb");
    if (!f) fail(std::string("cannot open ") + fn);
    int w = isint ? 5 : 16;
    std::vector<char> line((size_t)NX1 * w + 64);
    a.resize((size_t)NX1 * NY1 * nlay);
    for (int j = 0; j < NY1 * nlay; j++) {
        if (!fgets(line.data(), (int)line.size(), f)) { fprintf(stderr, "%s: only %d rows\n", fn, j); exit(1); }
        if ((int)strlen(line.data()) < NX1 * w) { fprintf(stderr, "%s: short row %d\n", fn, j + 1); exit(1); }
        for (int i = 0; i < NX1; i++) a[(size_t)j * NX1 + i] = (T)fortran_real(line.data() + (size_t)i * w, w);
    }
    fclose(f);
}
// ncoutput.F90 lowlevel_write_hotstart_2d: '(1X,E15.8)' (Fortran 0.dddddddd mantissa) or '(1X,I4)'
static void fortran_e15_8(double v, char *out) {   // 16 chars incl. the leading blank, NUL-terminated
    if (!(fabs(v) >= 1e-99)) { snprintf(out, 17, "  0.00000000E+00"); return; }
    double a = fabs(v); int e = (int)floor(log10(a)) + 1;
    long long m = llround(a / pow(10.0, e) * 1e8);
    if (m >= 100000000LL) { m /= 10; e++; }
    if (m < 10000000LL) { m *= 10; e--; }
    snprintf(out, 17, " %s0.%08lldE%c%02d", v < 0 ? "-" : " ", m, e < 0 ? '-' : '+', abs(e));
}
template <typename T> static void write_hotstart(const std::string &var, int fileno, int NX1, int NY1, const T *a,
                                                 bool isint, int nlay = 1) {
    char fn[256]; snprintf(fn, sizeof fn, "hotstart_%s%06d.dat", var.c_str(), fileno);
    FILE *f = fopen(fn, "w");
    if (!f) fail(std::string("cannot write ") + fn);
    std::vector<char> line((size_t)NX1 * 16 + 2);
    char cell[24];
    for (int j = 0; j < NY1 * nlay; j++) {
        size_t n = 0;
        for (int i = 0; i < NX1; i++) {
            double v = (double)a[(size_t)j * NX1 + i];
            if (isint) n += snprintf(line.data() + n, 6, " %4d", (int)v);
            else { fortran_e15_8(v, cell); memcpy(line.data() + n, cell, 16); n += 16; }
        }
        line[n++] = '\n';
        fwrite(line.data(), 1, n, f);
    }
    fclose(f);
}

// read a time list file (tsglobal, tspoints, tsmean, tshotstart): first value n, then n times
static std::vector<double> read_time_list(const std::string &fn) {
    Tok t(fn); double n; if (!t.num(n)) fail("cannot read " + fn);
    std::vector<double> v((size_t)llround(n));
    for (auto &x : v) if (!t.num(x)) fail("cannot read " + fn);
    return v;
}
