// grid.h: initialize.F90 grid_bathy and spaceparams.F90 gridprops (host, double precision).
#pragma once

struct Grid {
    int nx, ny, NX1, NY1;
    double alfa;   // radians
    std::vector<double> x, y, xz, yz, xu, yu, xv, yv, zb;
    std::vector<double> dsu, dsv, dsz, dsc, dnu, dnv, dnz, dnc, dsdnui, dsdnvi, dsdnzi, alfau, alfav, alfaz, sdist, ndist;
    bool uni = false;
    size_t I(int i, int j) const { return (size_t)(j - 1) * NX1 + (i - 1); }
};

static void read_delft3d_grid(const std::string &fn, Grid &g) {
    std::ifstream f(fn);
    if (!f) fail("cannot open xyfile " + fn);
    std::string line;
    while (std::getline(f, line)) if (!line.empty() && line[0] != '*') break;   // first non-comment line
    if (lower(line).find("cartesian") == std::string::npos) fail("Delft3D grid is not Cartesian");
    int mmax = 0, nmax = 0;
    while (std::getline(f, line)) { std::istringstream is(line); if (is >> mmax >> nmax) break; }
    std::getline(f, line);   // 0 0 0
    g.nx = mmax - 1; g.ny = nmax - 1; g.NX1 = mmax; g.NY1 = nmax;
    std::string rest((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::istringstream is(rest);
    g.x.assign((size_t)mmax * nmax, 0); g.y = g.x;
    for (int pass = 0; pass < 2; pass++) {
        auto &a = pass == 0 ? g.x : g.y;
        for (int n = 0; n < nmax; n++) {
            std::string tag; int idx; is >> tag;
            if (tag == "ETA=") is >> idx; else if (tag.rfind("ETA=", 0) == 0) { } else fail("xyfile: expected ETA=");
            for (int m = 0; m < mmax; m++) { double v; if (!(is >> v)) fail("xyfile: short record"); a[(size_t)n * mmax + m] = v; }
        }
    }
}

static void gridprops(Grid &g, double xori, double yori) {
    int nx = g.nx, ny = g.ny, NX1 = g.NX1, NY1 = g.NY1;
    size_t N = (size_t)NX1 * NY1;
    auto I = [&](int i, int j) { return g.I(i, j); };
    for (auto *a : {&g.xz, &g.yz, &g.xu, &g.yu, &g.xv, &g.yv, &g.dsu, &g.dsv, &g.dsz, &g.dsc, &g.dnu, &g.dnv, &g.dnz,
                    &g.dnc, &g.dsdnui, &g.dsdnvi, &g.dsdnzi, &g.alfau, &g.alfav, &g.alfaz, &g.sdist, &g.ndist}) a->assign(N, 0);
    double ca = cos(g.alfa), sa = sin(g.alfa);
    for (size_t k = 0; k < N; k++) { g.xz[k] = xori + g.x[k] * ca - g.y[k] * sa; g.yz[k] = yori + g.x[k] * sa + g.y[k] * ca; }
    auto &xz = g.xz, &yz = g.yz;
    for (int j = 1; j <= NY1; j++) {
        for (int i = 1; i <= nx; i++) { g.xu[I(i, j)] = .5 * (xz[I(i, j)] + xz[I(i + 1, j)]); g.yu[I(i, j)] = .5 * (yz[I(i, j)] + yz[I(i + 1, j)]); }
        g.xu[I(NX1, j)] = 1.5 * xz[I(NX1, j)] - 0.5 * xz[I(nx, j)];
        g.yu[I(NX1, j)] = 1.5 * yz[I(NX1, j)] - 0.5 * yz[I(nx, j)];
    }
    if (ny > 0) {
        for (int i = 1; i <= NX1; i++) {
            for (int j = 1; j <= ny; j++) { g.xv[I(i, j)] = .5 * (xz[I(i, j)] + xz[I(i, j + 1)]); g.yv[I(i, j)] = .5 * (yz[I(i, j)] + yz[I(i, j + 1)]); }
            g.xv[I(i, NY1)] = 1.5 * xz[I(i, NY1)] - 0.5 * xz[I(i, ny)];
            g.yv[I(i, NY1)] = 1.5 * yz[I(i, NY1)] - 0.5 * yz[I(i, ny)];
        }
    } else { g.xv = xz; g.yv = yz; }
    std::vector<double> xc(N), yc(N);
    if (ny > 0) {
        for (int j = 1; j <= ny; j++) {
            for (int i = 1; i <= nx; i++) {
                xc[I(i, j)] = .25 * (xz[I(i, j)] + xz[I(i + 1, j)] + xz[I(i, j + 1)] + xz[I(i + 1, j + 1)]);
                yc[I(i, j)] = .25 * (yz[I(i, j)] + yz[I(i + 1, j)] + yz[I(i, j + 1)] + yz[I(i + 1, j + 1)]);
            }
            xc[I(NX1, j)] = 0.5 * (g.xu[I(NX1, j)] + g.xu[I(NX1, j + 1)]);
            yc[I(NX1, j)] = 0.5 * (g.yu[I(NX1, j)] + g.yu[I(NX1, j + 1)]);
        }
        for (int i = 1; i <= nx; i++) {
            xc[I(i, NY1)] = 0.5 * (g.xv[I(i, NY1)] + g.xv[I(i + 1, NY1)]);
            yc[I(i, NY1)] = 0.5 * (g.yv[I(i, NY1)] + g.yv[I(i + 1, NY1)]);
        }
        // 0.5 is a default-real literal in XBeach here (exact)
        xc[I(NX1, NY1)] = 1.5 * g.xu[I(NX1, NY1)] - 0.5 * g.xu[I(nx, NY1)];
        yc[I(NX1, NY1)] = 1.5 * g.yu[I(NX1, NY1)] - 0.5 * g.yu[I(nx, NY1)];
    } else { xc = g.xu; yc = g.yu; }
    auto dist = [](double x1, double y1, double x2, double y2) { return pow((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1), 0.5); };
    for (int j = 1; j <= NY1; j++) {
        for (int i = 1; i <= nx; i++) g.dsu[I(i, j)] = dist(xz[I(i, j)], yz[I(i, j)], xz[I(i + 1, j)], yz[I(i + 1, j)]);
        g.dsu[I(NX1, j)] = g.dsu[I(nx, j)];
        for (int i = 2; i <= NX1; i++) g.dsz[I(i, j)] = dist(g.xu[I(i - 1, j)], g.yu[I(i - 1, j)], g.xu[I(i, j)], g.yu[I(i, j)]);
        g.dsz[I(1, j)] = g.dsz[I(2, j)];
    }
    if (ny > 0) {
        for (int j = 1; j <= NY1; j++) {
            for (int i = 2; i <= NX1; i++) g.dsv[I(i, j)] = dist(xc[I(i - 1, j)], yc[I(i - 1, j)], xc[I(i, j)], yc[I(i, j)]);
            g.dsv[I(1, j)] = g.dsv[I(2, j)];
            g.dsv[I(NX1, j)] = g.dsv[I(nx, j)];
            for (int i = 1; i <= nx; i++) g.dsc[I(i, j)] = dist(g.xv[I(i, j)], g.yv[I(i, j)], g.xv[I(i + 1, j)], g.yv[I(i + 1, j)]);
            g.dsc[I(NX1, j)] = g.dsc[I(nx, j)];
        }
        for (int j = 2; j <= NY1; j++) for (int i = 1; i <= NX1; i++) {
            g.dnu[I(i, j)] = dist(xc[I(i, j - 1)], yc[I(i, j - 1)], xc[I(i, j)], yc[I(i, j)]);
            g.dnz[I(i, j)] = dist(g.xv[I(i, j - 1)], g.yv[I(i, j - 1)], g.xv[I(i, j)], g.yv[I(i, j)]);
        }
        for (int i = 1; i <= NX1; i++) { g.dnu[I(i, 1)] = g.dnu[I(i, 2)]; g.dnz[I(i, 1)] = g.dnz[I(i, 2)]; }
        for (int j = 1; j <= ny; j++) for (int i = 1; i <= NX1; i++) {
            g.dnv[I(i, j)] = dist(xz[I(i, j)], yz[I(i, j)], xz[I(i, j + 1)], yz[I(i, j + 1)]);
            g.dnc[I(i, j)] = dist(g.xu[I(i, j)], g.yu[I(i, j)], g.xu[I(i, j + 1)], g.yu[I(i, j + 1)]);
        }
        for (int i = 1; i <= NX1; i++) { g.dnv[I(i, NY1)] = g.dnv[I(i, ny)]; g.dnc[I(i, NY1)] = g.dnc[I(i, ny)]; }
        auto quad = [](double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4) {
            return 0.5 * (fabs(x1 * y3 - x3 * y1) + fabs(x2 * y4 - x4 * y2)); };
        auto &xv = g.xv, &yv = g.yv, &xu = g.xu, &yu = g.yu;
        for (int j = 2; j <= NY1; j++) for (int i = 1; i <= nx; i++) {
            double a = quad(xv[I(i, j)] - xv[I(i, j - 1)], yv[I(i, j)] - yv[I(i, j - 1)],
                            xv[I(i + 1, j)] - xv[I(i + 1, j - 1)], yv[I(i + 1, j)] - yv[I(i + 1, j - 1)],
                            xv[I(i + 1, j - 1)] - xv[I(i, j - 1)], yv[I(i + 1, j - 1)] - yv[I(i, j - 1)],
                            xv[I(i + 1, j)] - xv[I(i, j)], yv[I(i + 1, j)] - yv[I(i, j)]);
            g.dsdnui[I(i, j)] = 1.0 / a;
        }
        for (int i = 1; i <= NX1; i++) g.dsdnui[I(i, 1)] = g.dsdnui[I(i, 2)];
        for (int j = 1; j <= NY1; j++) g.dsdnui[I(NX1, j)] = g.dsdnui[I(nx, j)];
        for (int j = 1; j <= ny; j++) for (int i = 2; i <= NX1; i++) {
            double a = quad(xu[I(i - 1, j + 1)] - xu[I(i - 1, j)], yu[I(i - 1, j + 1)] - yu[I(i - 1, j)],
                            xu[I(i, j + 1)] - xu[I(i, j)], yu[I(i, j + 1)] - yu[I(i, j)],
                            xu[I(i, j)] - xu[I(i - 1, j)], yu[I(i, j)] - yu[I(i - 1, j)],
                            xu[I(i, j + 1)] - xu[I(i - 1, j + 1)], yu[I(i, j + 1)] - yu[I(i - 1, j + 1)]);
            g.dsdnvi[I(i, j)] = 1.0 / a;
        }
        for (int i = 1; i <= NX1; i++) g.dsdnvi[I(i, NY1)] = g.dsdnvi[I(i, ny)];
        for (int j = 1; j <= NY1; j++) g.dsdnvi[I(1, j)] = g.dsdnvi[I(2, j)];
        for (int j = 2; j <= NY1; j++) for (int i = 2; i <= NX1; i++) {
            double a = quad(xc[I(i - 1, j)] - xc[I(i - 1, j - 1)], yc[I(i - 1, j)] - yc[I(i - 1, j - 1)],
                            xc[I(i, j)] - xc[I(i, j - 1)], yc[I(i, j)] - yc[I(i, j - 1)],
                            xc[I(i, j - 1)] - xc[I(i - 1, j - 1)], yc[I(i, j - 1)] - yc[I(i - 1, j - 1)],
                            xc[I(i, j)] - xc[I(i - 1, j)], yc[I(i, j)] - yc[I(i - 1, j)]);
            g.dsdnzi[I(i, j)] = 1.0 / a;
        }
        for (int i = 1; i <= NX1; i++) g.dsdnzi[I(i, 1)] = g.dsdnzi[I(i, 2)];
        for (int j = 1; j <= NY1; j++) g.dsdnzi[I(1, j)] = g.dsdnzi[I(2, j)];
    } else {
        g.dsv = g.dsz; g.dsc = g.dsu;
        std::fill(g.dnu.begin(), g.dnu.end(), 100.0); std::fill(g.dnz.begin(), g.dnz.end(), 100.0);
        std::fill(g.dnv.begin(), g.dnv.end(), 100.0); std::fill(g.dnc.begin(), g.dnc.end(), 100.0);
        for (size_t k = 0; k < N; k++) {
            g.dsdnui[k] = 1.0 / (g.dsu[k] * g.dnu[k]); g.dsdnvi[k] = 1.0 / (g.dsv[k] * g.dnv[k]);
            g.dsdnzi[k] = 1.0 / (g.dsz[k] * g.dnz[k]);
        }
    }
    for (int j = 1; j <= NY1; j++) {
        for (int i = 2; i <= nx; i++) g.alfaz[I(i, j)] = atan2(yz[I(i + 1, j)] - yz[I(i - 1, j)], xz[I(i + 1, j)] - xz[I(i - 1, j)]);
        g.alfaz[I(1, j)] = g.alfaz[I(2, j)]; g.alfaz[I(NX1, j)] = g.alfaz[I(nx, j)];
        for (int i = 1; i <= nx; i++) g.alfau[I(i, j)] = atan2(yz[I(i + 1, j)] - yz[I(i, j)], xz[I(i + 1, j)] - xz[I(i, j)]);
        g.alfau[I(NX1, j)] = g.alfau[I(nx, j)];
    }
    if (ny > 0) {
        for (int i = 1; i <= NX1; i++) {
            for (int j = 1; j <= ny; j++) g.alfav[I(i, j)] = atan2(yz[I(i, j + 1)] - yz[I(i, j)], xz[I(i, j + 1)] - xz[I(i, j)]);
            g.alfav[I(i, NY1)] = g.alfav[I(i, ny)];
        }
    } else g.alfav = g.alfaz;
    for (int j = 1; j <= NY1; j++) { g.sdist[I(1, j)] = 0; for (int i = 2; i <= NX1; i++) g.sdist[I(i, j)] = g.sdist[I(i - 1, j)] + g.dsu[I(i - 1, j)]; }
    for (int i = 1; i <= NX1; i++) { g.ndist[I(i, 1)] = 0; for (int j = 2; j <= NY1; j++) g.ndist[I(i, j)] = g.ndist[I(i, j - 1)] + g.dnv[I(i, j - 1)]; }
    // uniform-grid test: every metric equal to its first value within round-off
    g.uni = true;
    const char *names[] = {"dsu", "dsv", "dsz", "dsc", "dnu", "dnv", "dnz", "dnc", "dsdnui", "dsdnvi", "dsdnzi", "alfau", "alfav", "alfaz"};
    const std::vector<double> *arrs[] = {&g.dsu, &g.dsv, &g.dsz, &g.dsc, &g.dnu, &g.dnv, &g.dnz, &g.dnc, &g.dsdnui, &g.dsdnvi,
                                         &g.dsdnzi, &g.alfau, &g.alfav, &g.alfaz};
    for (int q = 0; q < 14 && g.uni; q++) {
        const auto &a = *arrs[q];
        double v0 = a[I(std::min(2, NX1), std::min(2, NY1))];
        for (int j = 1; j <= NY1 && g.uni; j++) for (int i = 1; i <= NX1; i++) {
            double v = a[I(i, j)];
            // XBeach's c-point (nx+1,ny+1) is not shifted alongshore, which distorts dnu, dsdnzi and dsdnvi in that
            // corner only; no flow routine reads the metrics there
            if (i == NX1 && j == NY1 && ny > 0) continue;
            if (fabs(v - v0) > 1e-10 * std::max(1.0, fabs(v0))) {
                printf("grid: non-uniform metrics (%s(%d,%d) = %.12g, %s(2,2) = %.12g)\n", names[q], i, j, v, names[q], v0);
                g.uni = false; break;
            }
        }
    }
}

// grid_bathy: read the grid and the bed, apply posdwn and the alongshore-uniform boundary rows
static void grid_bathy(const Params &p, Grid &g) {
    g.nx = p.nx; g.ny = p.ny; g.NX1 = p.nx + 1; g.NY1 = p.ny + 1;
    double alfa = p.alfa < 0 ? 360 + p.alfa : p.alfa;
    g.alfa = alfa * atan(1.0) / 45.0;
    if (p.gridform == 1) { read_delft3d_grid(p.xyfile, g); }
    size_t N = (size_t)g.NX1 * g.NY1;
    bool readbed = p.setbathy != 1 && p.hotstart != 1;
    if (p.gridform == 0 && p.vardx == 0) {
        if (readbed) read_grid_file(p.depfile, g.NX1, g.NY1, g.zb, true); else g.zb.assign(N, 0);
        g.x.assign(N, 0); g.y.assign(N, 0);
        for (int j = 1; j <= g.NY1; j++) for (int i = 1; i <= g.NX1; i++) { g.x[g.I(i, j)] = (i - 1) * p.dx; g.y[g.I(i, j)] = (j - 1) * p.dy; }
    } else if (p.gridform == 0) {
        if (readbed) read_grid_file(p.depfile, g.NX1, g.NY1, g.zb, false); else g.zb.assign(N, 0);
        read_grid_file(p.xfile, g.NX1, g.NY1, g.x, false);
        if (!p.yfile.empty()) read_grid_file(p.yfile, g.NX1, g.NY1, g.y, false); else g.y.assign(N, 0);
    } else {
        if (p.setbathy != 1) read_grid_file(p.depfile, g.NX1, g.NY1, g.zb, true); else g.zb.assign(N, 0);
    }
    double posdwn = p.posdwn * fabs(p.posdwn);   // s%posdwn*sign(s%posdwn,1.d0)
    for (auto &z : g.zb) z = -z * posdwn;
    if (g.ny > 0) for (int i = 1; i <= g.NX1; i++) { g.zb[g.I(i, 1)] = g.zb[g.I(i, 2)]; g.zb[g.I(i, g.NY1)] = g.zb[g.I(i, g.ny)]; }
    for (int j = 1; j <= g.NY1; j++) { g.zb[g.I(1, j)] = g.zb[g.I(2, j)]; g.zb[g.I(g.NX1, j)] = g.zb[g.I(g.nx, j)]; }
    gridprops(g, p.xori, p.yori);
}
