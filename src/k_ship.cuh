// k_ship.cuh: ship.F90 (ship_init, shipwave, ship_force): moving pressure fields of ships on prescribed tracks.
// Each step, per ship: the ship grid is placed on its track (linear_interp of x, y, heading); interp.F90 MKMAP finds,
// for every ship grid point, the last wet grid quadrilateral (of cell centres) that contains it and its bilinear
// weights; GRMAP interpolates zs + ph to the ship points; the hull pressure head is mapped back with GRMAP2. The
// search over grid cells runs on the GPU (candidate cell/point pairs); the point-in-polygon test (single precision,
// as ipon), the weights (bilin5) and the small ship-grid arithmetic run on the host in XBeach's order.
#pragma once

struct Ship {
    std::string name, shipgeom, shiptrack;
    double dx = 5, dy = 5, xCG = 0, yCG = 0, zCG = 0;
    int nx = 20, ny = 20, compute_force = 0, compute_motion = 0, flying = 0, read_heading = 0;
    std::vector<double> depth, zhull, ph, zs, x, y;          // (nx+1)*(ny+1), ix fastest
    std::vector<double> tt, tx, ty, tz, tdir;
    std::vector<int> iref;                                  // 4 per ship point (1-based grid index, 0 = none)
    std::vector<double> w;                                  // 4 per ship point (single precision values)
    double shipx = 0, shipy = 0, shipz = 0, Fx = 0, Fy = 0, Fz = 0, Mx = 0, My = 0, Mz = 0;
    double *d_x2 = nullptr, *d_y2 = nullptr;                // ship points on the device
};

static std::vector<Ship> ship_init(const Params &P) {
    std::vector<Ship> S;
    std::vector<std::string> names;
    {
        std::ifstream f(P.shipfile);
        if (!f) fail("cannot open shipfile " + P.shipfile);
        std::string l;
        while (std::getline(f, l)) names.push_back(trim(l));
    }
    for (auto &nm : names) {
        Ship s; s.name = nm;
        ParFile F(nm.c_str());
        s.dx = F.dbl("dx", 5.0); s.dy = F.dbl("dy", 5.0); s.nx = F.integer("nx", 20); s.ny = F.integer("ny", 20);
        s.shipgeom = F.name("shipgeom"); if (s.shipgeom.empty()) fail(nm + ": shipgeom is required");
        s.xCG = F.dbl("xCG", 0.0); s.yCG = F.dbl("yCG", 0.0); s.zCG = F.dbl("zCG", 0.0);
        s.shiptrack = F.name("shiptrack"); if (s.shiptrack.empty()) fail(nm + ": shiptrack is required");
        s.compute_force = F.integer("compute_force", 0); s.compute_motion = F.integer("compute_motion", 0);
        s.flying = F.integer("flying", 0); s.read_heading = F.integer("read_heading", 0);
        const int n2 = (s.nx + 1) * (s.ny + 1);
        read_grid_file(s.shipgeom, s.nx + 1, s.ny + 1, s.depth, true);
        s.zhull.assign(n2, 0); s.ph.assign(n2, 0); s.zs.assign(n2, 0); s.x.assign(n2, 0); s.y.assign(n2, 0);
        if (s.compute_motion == 0) for (int q = 0; q < n2; q++) s.zhull[q] = -s.depth[q];
        if (s.compute_force == 0) s.ph = s.depth;
        // track: t x y [dir] [z]
        int nt = 0;
        { std::ifstream f(s.shiptrack); if (!f) fail("cannot open shiptrack " + s.shiptrack); std::string l; while (std::getline(f, l)) nt++; }
        Tok t(s.shiptrack);
        int nv = 3 + s.read_heading + s.flying;
        for (int q = 0; q < nt; q++) {
            double v[5] = {0, 0, 0, 0, 0};
            for (int c_ = 0; c_ < nv; c_++) if (!t.num(v[c_])) fail(s.shiptrack + ": cannot read record " + std::to_string(q + 1));
            t.nextline();
            s.tt.push_back(v[0]); s.tx.push_back(v[1]); s.ty.push_back(v[2]);
            double dir = 0, z = 0;
            if (s.read_heading) { dir = M_PI / 180.0 * (270.0 - v[3]); if (s.flying) z = v[4]; }
            else if (s.flying) z = v[3];
            s.tdir.push_back(dir); s.tz.push_back(z);
        }
        if (nt < 2) fail(s.shiptrack + ": a ship track needs at least two points");
        if (s.read_heading == 0) {
            s.tdir[0] = atan2(s.ty[1] - s.ty[0], s.tx[1] - s.tx[0]);
            for (int q = 1; q < nt - 1; q++) s.tdir[q] = atan2(s.ty[q + 1] - s.ty[q - 1], s.tx[q + 1] - s.tx[q - 1]);
            s.tdir[nt - 1] = atan2(s.ty[nt - 1] - s.ty[nt - 2], s.tx[nt - 1] - s.tx[nt - 2]);
        }
        s.iref.assign(4 * n2, 0); s.w.assign(4 * n2, 0.0);
        CK(cudaMalloc(&s.d_x2, n2 * sizeof(double))); CK(cudaMalloc(&s.d_y2, n2 * sizeof(double)));
        printf("ship %s: %dx%d grid of %.2fx%.2f m, %d track points%s\n", nm.c_str(), s.nx, s.ny, s.dx, s.dy, nt,
               s.compute_force ? ", forces" : "");
        S.push_back(s);
    }
    return S;
}

// zsvirt = zs + ph; ph = 0
__global__ void k_ship_zsvirt(double *zsv) {
    IJ
    size_t k = ID(i, j);
    zsv[k] = (double)d.zs[k] + (double)d.ph[k];
    d.ph[k] = 0;
}
// MKMAP search: candidate (quadrilateral, ship point) pairs for wet quadrilaterals (all four corners wet) whose
// bounding box holds the point
__global__ void k_ship_cand(const double *xz, const double *yz, const double *x2, const double *y2, int n2, double bx0,
                            double bx1, double by0, double by1, int *pairs, int cap, int *count) {
    int i1 = blockIdx.x * blockDim.x + threadIdx.x + 1, j1 = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i1 > c.nx || j1 > c.ny) return;
    size_t k1 = ID(i1, j1), k2 = ID(i1 + 1, j1), k3 = ID(i1 + 1, j1 + 1), k4 = ID(i1, j1 + 1);
    double xp[4] = {xz[k1], xz[k2], xz[k3], xz[k4]}, yp[4] = {yz[k1], yz[k2], yz[k3], yz[k4]};
    const double xymiss = -999.0, eps = (double)0.00001f;
    for (int q = 0; q < 4; q++)
        if ((xp[q] > xymiss - eps && xp[q] < xymiss + eps) || (yp[q] > xymiss - eps && yp[q] < xymiss + eps)) return;
    double xmin = 1e10, xmax = -1e10, ymin = 1e10, ymax = -1e10;
    for (int q = 0; q < 4; q++) { xmin = fmin(xp[q], xmin); xmax = fmax(xp[q], xmax); ymin = fmin(yp[q], ymin); ymax = fmax(yp[q], ymax); }
    if (xmax < bx0 || xmin > bx1 || ymax < by0 || ymin > by1) return;
    auto wet = [&](size_t k) { int w_ = d.wetz[k]; return w_ == 1 || w_ == 3; };
    if (!(wet(k1) && wet(k2) && wet(k3) && wet(k4))) return;
    int L = i1 + (j1 - 1) * c.NX1;
    for (int q = 0; q < n2; q++) {
        if (x2[q] >= xmin && x2[q] <= xmax && y2[q] >= ymin && y2[q] <= ymax) {
            int at = atomicAdd(count, 1);
            if (at < cap) { pairs[2 * at] = L; pairs[2 * at + 1] = q; }
        }
    }
}
// GRMAP gather: the four grid values of each mapped ship point
__global__ void k_ship_gather(const double *f, const int *iref, double *out, int n) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;
    int i = iref[q];
    out[q] = f[max(i, 1) - 1];
}
// set ph at the cells reached by GRMAP2
__global__ void k_ship_scatter(const int *idx, const double *val, int n) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;
    d.ph[idx[q]] = (real)val[q];
}
__global__ void k_ship_first() {   // shipwave first call: zs = zs - ph
    IJ
    size_t k = ID(i, j);
    d.zs[k] -= d.ph[k];
}

// interp.F90 ipon in single precision: 1 inside, 0 on the boundary, -1 outside
static int ship_ipon(const double *xq, const double *yq, double xp, double yp) {
    float x[5], y[5];
    for (int q = 0; q < 4; q++) { x[q] = (float)(xq[q] - xp); y[q] = (float)(yq[q] - yp); }
    x[4] = x[0]; y[4] = y[0];
    int nunder = 0;
    for (int q = 0; q < 4; q++) {
        if ((x[q] < 0.f && x[q + 1] >= 0.f) || (x[q + 1] < 0.f && x[q] >= 0.f)) {
            if (y[q] < 0.f && y[q + 1] < 0.f) nunder++;
            else if ((y[q] <= 0.f && y[q + 1] >= 0.f) || (y[q + 1] <= 0.f && y[q] >= 0.f)) {
                volatile float a = y[q] * x[q + 1], b = x[q] * y[q + 1];
                volatile float num = a - b, den = x[q + 1] - x[q];
                float ysn = num / den;
                if (ysn < 0.f) nunder++;
                else if (ysn <= 0.f) return 0;
            }
        } else if (fabsf(x[q]) < 1.0E-8f && fabsf(x[q + 1]) < 1.0E-8f) {
            if ((y[q] <= 0.f && y[q + 1] >= 0.f) || (y[q + 1] <= 0.f && y[q] >= 0.f)) return 0;
        }
    }
    return nunder % 2 == 0 ? -1 : 1;
}
// interp.F90 bilin5: bilinear weights of (x0, y0) in the quadrilateral (xa, ya); false on failure
static bool ship_bilin5(const double *xa, const double *ya, double x, double y, double *w) {
    double x1 = xa[0], y1 = ya[0];
    double a21 = xa[1] - x1, a22 = ya[1] - y1, a31 = xa[2] - x1, a32 = ya[2] - y1, a41 = xa[3] - x1, a42 = ya[3] - y1;
    double det = a21 * a42 - a22 * a41;
    if (fabs(det) < (double)1.0e-20f) return false;
    double x3t = (a42 * a31 - a41 * a32) / det, y3t = (-a22 * a31 + a21 * a32) / det;
    double xt = (a42 * (x - x1) - a41 * (y - y1)) / det, yt = (-a22 * (x - x1) + a21 * (y - y1)) / det;
    if (x3t < 0.0 || y3t < 0.0) return false;
    double xi, eta;
    if (fabs(x3t - 1.0) < (double)1.0e-7f) {
        xi = xt;
        if (fabs(y3t - 1.0) < (double)1.0e-7f) eta = yt;
        else if (fabs(1.0 + (y3t - 1.0) * xt) < (double)1.0e-6f) return false;
        else eta = yt / (1.0 + (y3t - 1.0) * xt);
    } else if (fabs(y3t - 1.0) < (double)1.0e-6f) {
        eta = yt;
        if (fabs(1.0 + (x3t - 1.0) * yt) < (double)1.0e-6f) return false;
        xi = xt / (1.0 + (x3t - 1.0) * yt);
    } else {
        double a = y3t - 1.0, b = 1.0 + (x3t - 1.0) * yt - (y3t - 1.0) * xt, cc = -xt;
        double discr = b * b - 4.0 * a * cc;
        if (discr < (double)1.0e-6f) return false;
        xi = (-b + sqrt(discr)) / (2.0 * a);
        eta = ((y3t - 1.0) * (xi - xt) + (x3t - 1.0) * yt) / (x3t - 1.0);
    }
    w[0] = (1.0 - xi) * (1.0 - eta); w[1] = xi * (1.0 - eta); w[2] = xi * eta; w[3] = eta * (1.0 - xi);
    return true;
}

// ship_force: pressure forces and moments on the hull (per unit rho g, then scaled)
static void ship_force(Ship &s, const Params &P) {
    s.Fx = s.Fy = s.Fz = s.Mx = s.My = s.Mz = 0;
    double hdx = .5 * s.dx, hdy = .5 * s.dy;
    const int nx1 = s.nx + 1;
    auto A = [&](const std::vector<double> &a, int ix, int iy) { return a[(size_t)(iy - 1) * nx1 + (ix - 1)]; };
    for (int iy = 1; iy <= s.ny; iy++)
        for (int ix = 1; ix <= s.nx; ix++) {
            double dFx = 0.5 * (A(s.ph, ix, iy) + A(s.ph, ix + 1, iy)) * (A(s.depth, ix + 1, iy) - A(s.depth, ix, iy)) * s.dy;
            double dFy = 0.5 * (A(s.ph, ix, iy) + A(s.ph, ix, iy + 1)) * (A(s.depth, ix, iy + 1) - A(s.depth, ix, iy)) * s.dx;
            double dFz = A(s.ph, ix, iy) * s.dx * s.dy;
            s.Fx += dFx; s.Fy += dFy; s.Fz += dFz;
            s.Mx = s.Mx + ((A(s.y, ix, iy) + hdy) - s.yCG) * dFz - (.5 * (A(s.zhull, ix, iy) + A(s.zhull, ix, iy + 1)) - s.zCG) * dFy;
            s.My = s.My - ((A(s.x, ix, iy) + hdx) - s.xCG) * dFz + (.5 * (A(s.zhull, ix, iy) + A(s.zhull, ix + 1, iy)) - s.zCG) * dFx;
            s.Mz = s.Mz - (0.5 * (((iy - s.ny / 2 - 1) * s.dy - s.yCG) + ((iy + 1 - s.ny / 2 - 1) * s.dy - s.yCG))) * dFx
                 + (0.5 * (((ix - s.nx / 2 - 1) * s.dx - s.xCG) + ((ix + 1 - s.nx / 2 - 1) * s.dx - s.xCG))) * dFy;
        }
    s.Fx = s.Fx * P.rho * P.g; s.Fy = s.Fy * P.rho * P.g; s.Fz = s.Fz * P.rho * P.g;
    s.Mx = s.Mx * P.rho * P.g; s.My = s.My * P.rho * P.g; s.Mz = s.Mz * P.rho * P.g;
}
