// k_bc.cuh: boundaryconditions.F90 flow_bc (nonh: front nonh_1d, back wall/abs_1d/abs_2d, corners, laterals),
// compute_tide_zs0.F90 (boundaryinterp_tide, boundaryinterp_tide_complex, fill_tide_grid), the cyclic halo
// copies and the wind projection.
#pragma once

// cyclic halos: rows 1, 2 <- ny-2, ny-1 and ny, ny+1 <- 3, 4 (XBeach MPI overlap, period ny-3)
struct CycSet { void *p[10]; int isint[10]; int n; };
__global__ void k_cyc(CycSet s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    int ny = c.ny;
    const int dst[4] = {1, 2, ny, ny + 1}, src[4] = {ny - 2, ny - 1, 3, 4};
    for (int q = 0; q < s.n; q++)
        for (int r = 0; r < 4; r++) {
            if (s.isint[q]) { int *a = (int *)s.p[q]; a[ID(i, dst[r])] = a[ID(i, src[r])]; }
            else { real *a = (real *)s.p[q]; a[ID(i, dst[r])] = a[ID(i, src[r])]; }
        }
}

// ---------------------------------------------------------------------------------------------
// tide (compute_tide_zs0.F90)
// ---------------------------------------------------------------------------------------------
struct TideNow { real zs01, zs02, zs03, zs04; real nd1, nd2, nd3, nd4, sd1, sd2, sd3, sd4; };
// simple boundary assignments; tideloc 0/1: all edges zs01; tideloc 2 land: front zs01, back zs03;
// tideloc 2 sea: left zs01, right zs02 and dzs0dn on the laterals. Complex boundaries in k_tide_complex.
__global__ void k_tide_edges(TideNow T, int tideloc, int paulrevere, int global) {
    IJ
    size_t k = ID(i, j);
    int nx = c.nx, ny = c.ny;
    bool lat_ok = global || !c.cyclic;    // xmpi_isleft/isright (no lateral edges with cyclic MPI)
    if (tideloc <= 1) {
        if (i <= 2) d.zs0[k] = T.zs01;
        if (i >= nx) d.zs0[k] = T.zs01;
        if (lat_ok && j == 1) d.zs0[k] = T.zs01;
        if (lat_ok && ny > 0 && j == ny + 1) d.zs0[k] = T.zs01;
    } else if (tideloc == 2 && paulrevere == PR_LAND) {
        if (i <= 2) d.zs0[k] = T.zs01;
        if (i >= nx) d.zs0[k] = T.zs03;
    } else if (tideloc == 2) {
        if (lat_ok && j == 1) d.zs0[k] = T.zs01;
        if (lat_ok && ny > 0 && j == ny + 1) d.zs0[k] = T.zs02;
        real tg = ny == 0 ? (T.zs02 - T.zs01) / dnz(1, 1) : (T.zs02 - T.zs01) / (T.nd2 - T.nd1);
        if (lat_ok && j == 1) d.dzs0dn[k] = tg * (real)d.wetz[k];
        if (lat_ok && ny > 0 && j == ny + 1) d.dzs0dn[k] = tg * (real)d.wetz[k];
    }
}
// boundaryinterp_tide_complex for one boundary (id 1 front, 2 back, 3 left = row ny+1, 4 right = row 1); one block
__global__ void k_tide_complex(TideNow T, int id, const real *sdist, const real *ndist) {
    __shared__ real smax[1024]; __shared__ int sidx[1024];
    int tid = threadIdx.x, nx = c.nx, ny = c.ny;
    bool rows = id <= 2;                  // front/back: run over j
    int fixed = id == 1 ? 1 : id == 2 ? nx + 1 : id == 3 ? ny + 1 : 1;
    int n = rows ? ny + 1 : nx + 1;
    real tide1, tide2, dt1, dt2, tg1 = 0, tg2 = 0;
    if (id == 1) { tide1 = T.zs01; tide2 = T.zs02; dt1 = T.nd1; dt2 = T.nd2; }
    else if (id == 2) { tide1 = T.zs04; tide2 = T.zs03; dt1 = T.nd4; dt2 = T.nd3; }
    else if (id == 3) { tide1 = T.zs02; tide2 = T.zs03; dt1 = T.sd2; dt2 = T.sd3; }
    else { tide1 = T.zs01; tide2 = T.zs04; dt1 = T.sd1; dt2 = T.sd4; }
    if (id >= 3) {
        if (ny == 0) { tg1 = (T.zs02 - T.zs01) / dnz(1, 1); tg2 = (T.zs03 - T.zs04) / dnz(nx + 1, 1); }
        else { tg1 = (T.zs02 - T.zs01) / (T.nd2 - T.nd1); tg2 = (T.zs03 - T.zs04) / (T.nd3 - T.nd4); }
    }
    // maxval and first maxloc of zb along the boundary
    real m = -1e30; int mi = 1 << 30;
    for (int q = tid; q < n; q += blockDim.x) {
        size_t k = rows ? ID(fixed, q + 1) : ID(q + 1, fixed);
        real z = d.zb[k];
        if (z > m || (z == m && q < mi)) { m = z; mi = q; }
    }
    smax[tid] = m; sidx[tid] = mi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            real a = smax[tid], b = smax[tid + s]; int ia = sidx[tid], ib = sidx[tid + s];
            if (b > a || (b == a && ib < ia)) { smax[tid] = b; sidx[tid] = ib; }
        }
        __syncthreads();
    }
    real maxzb = smax[0]; int im = sidx[0];
    real distmax = rows ? ndist[ID(fixed, im + 1)] : sdist[ID(im + 1, fixed)];
    for (int q = tid; q < n; q += blockDim.x) {
        size_t k = rows ? ID(fixed, q + 1) : ID(q + 1, fixed);
        real dist = rows ? ndist[k] : sdist[k];
        real z0, gr = 0;
        if (tide1 <= maxzb && tide2 <= maxzb) { if (dist <= distmax) { z0 = tide1; gr = tg1; } else { z0 = tide2; gr = tg2; } }
        else if (tide1 > maxzb && tide2 > maxzb) {
            real fac = (dist - dt1) * (1 / (dt2 - dt1)); z0 = (1 - fac) * tide1 + fac * tide2; gr = (1 - fac) * tg1 + fac * tg2;
        } else if (tide1 > maxzb) {
            if (dist <= distmax) { z0 = tide1; gr = tg1; }
            else { real fac = (dist - distmax) * (1 / (dt2 - distmax)); z0 = (1 - fac) * tide1 + fac * tide2; gr = (1 - fac) * tg1 + fac * tg2; }
        } else {
            if (dist <= distmax) { real fac = (dist - dt1) * (1 / (distmax - dt1)); z0 = (1 - fac) * tide1 + fac * tide2; gr = (1 - fac) * tg1 + fac * tg2; }
            else { z0 = tide2; gr = tg1; }
        }
        d.zs0[k] = fmax(z0, d.zb[k]);
        if (!rows) d.dzs0dn[k] = gr * (real)d.wetz[k];
    }
}
__global__ void k_tide_copy_fb() {   // zs0(2,:) = zs0(1,:); zs0(nx,:) = zs0(nx+1,:)
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    d.zs0[ID(2, j)] = d.zs0[ID(1, j)];
    d.zs0[ID(c.nx, j)] = d.zs0[ID(c.NX1, j)];
}
// correctforwetz: dry points get zs0 = zb; boundary rows next to dry cells get min(zb, zb neighbour)
__global__ void k_tide_wetz() {
    IJ
    size_t k = ID(i, j);
    int nx = c.nx, ny = c.ny;
    real z = d.zs0[k];
    if (d.wetz[k] == 0) z = d.zb[k];
    if (ny > 0) {
        if (j == 1 && d.wetz[ID(i, 2)] == 0) z = fmin(d.zb[ID(i, 1)], d.zb[ID(i, 2)]);
        if (j == ny + 1 && d.wetz[ID(i, ny)] == 0) z = fmin(d.zb[ID(i, ny + 1)], d.zb[ID(i, ny)]);
    }
    if (i == 1 && d.wetz[ID(2, j)] == 0) z = fmin(d.zb[ID(1, j)], d.zb[ID(2, j)]);
    if (i == nx + 1 && d.wetz[ID(nx, j)] == 0) z = fmin(d.zb[ID(nx, j)], d.zb[ID(nx + 1, j)]);
    d.zs0[k] = z;
}
// fill_tide_grid: one block per row j = 2..ny
__global__ void k_tide_fill(const real *sdist) {
    __shared__ real smax[256]; __shared__ int sidx[256];
    int j = blockIdx.x + 2, tid = threadIdx.x, nx = c.nx;
    if (j > c.ny) return;
    real m = -1e30; int mi = 1 << 30;
    for (int q = tid; q <= nx; q += blockDim.x) {
        real z = d.zb[ID(q + 1, j)];
        if (z > m || (z == m && q < mi)) { m = z; mi = q; }
    }
    smax[tid] = m; sidx[tid] = mi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            real a = smax[tid], b = smax[tid + s]; int ia = sidx[tid], ib = sidx[tid + s];
            if (b > a || (b == a && ib < ia)) { smax[tid] = b; sidx[tid] = ib; }
        }
        __syncthreads();
    }
    real maxzb = smax[0], distmax = sdist[ID(sidx[0] + 1, j)];
    real t1 = d.zs0[ID(1, j)], t2 = d.zs0[ID(nx + 1, j)];
    real s1 = sdist[ID(1, j)], sN = sdist[ID(nx + 1, j)];
    for (int i = 3 + tid; i <= nx - 1; i += blockDim.x) {
        real sd = sdist[ID(i, j)], v;
        if (t1 <= maxzb && t2 <= maxzb) v = sd <= distmax ? t1 : t2;
        else if (t1 > maxzb && t2 > maxzb) { real fac = (sd - s1) * (1 / (sN - s1)); v = (1 - fac) * t1 + fac * t2; }
        else if (t1 > maxzb) { if (sd <= distmax) v = t1; else { real fac = (sd - distmax) * (1 / (sN - distmax)); v = (1 - fac) * t1 + fac * t2; } }
        else { if (sd <= distmax) { real fac = (sd - s1) * (1 / (distmax - s1)); v = (1 - fac) * t1 + fac * t2; } else v = t2; }
        d.zs0[ID(i, j)] = fmax(v, d.zb[ID(i, j)]);
    }
}
// tidetype = instant: shift zs with zs0 at wet points connected to the front or back boundary by water
__global__ void k_tide_instant(const real *zs0old, real dmax, real dmin) {
    __shared__ int sf[256], sl[256];
    int j = blockIdx.x + 1, tid = threadIdx.x, nx = c.nx;
    if (j > c.NY1) return;
    int fd = 1 << 30, ld = 0;   // first and last dry index in the row
    for (int q = tid; q <= nx; q += blockDim.x) if (d.wetz[ID(q + 1, j)] == 0) { fd = min(fd, q + 1); ld = max(ld, q + 1); }
    sf[tid] = fd; sl[tid] = ld;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) { sf[tid] = min(sf[tid], sf[tid + s]); sl[tid] = max(sl[tid], sl[tid + s]); } __syncthreads(); }
    fd = sf[0]; ld = sl[0];
    if (j < JZ0 || j > JZ1) return;
    for (int i = 2 + tid; i <= nx; i += blockDim.x) {
        size_t k = ID(i, j);
        if (d.wetz[k] == 1 && d.zb[k] < d.zs0[k]) {
            int conn = (i < fd ? 1 : 0) + (i > ld ? 1 : 0);
            if (conn > 0) {
                real dz = d.zs0[k] - zs0old[k];
                if (dz > 0) dz = fmin(dz, dmax); else if (dz < 0) dz = fmax(dz, dmin);
                d.zs[k] += dz;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// mean-flow relaxation at the front (side 0, i = 1) and back (side 1, i = nx); tidetype = velocity
// ---------------------------------------------------------------------------------------------
__global__ void k_bc_msum(int side) {
    __shared__ double su[1024], sv[1024], wu[1024], wv[1024];
    int tid = threadIdx.x;
    int ib = side == 0 ? 1 : c.nx;
    int ju0 = c.oned ? 1 : (c.cyclic ? 3 : 2), ju1 = c.oned ? 1 : (c.cyclic ? c.ny - 1 : c.ny);
    int jv0 = c.oned ? 1 : (c.cyclic ? 3 : 2), jv1 = c.oned ? 1 : c.ny - 1;
    double a = 0, b = 0, wa = 0, wb = 0;
    for (int j = ju0 + tid; j <= ju1; j += blockDim.x) { a += d.uu[ID(ib, j)] * dnu(ib, j); wa += dnu(ib, j); }
    for (int j = jv0 + tid; j <= jv1; j += blockDim.x) { b += d.vv[ID(ib, j)] * dnv(ib, j); wb += dnv(ib, j); }
    su[tid] = a; sv[tid] = b; wu[tid] = wa; wv[tid] = wb;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { su[tid] += su[tid + s]; sv[tid] += sv[tid + s]; wu[tid] += wu[tid + s]; wv[tid] += wv[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { d.msum[2 * side] = su[0] / wu[0]; d.msum[2 * side + 1] = sv[0] / wv[0]; }
}
__global__ void k_bc_mapply(real factime, int side, int relax) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    size_t k = ID(side == 0 ? 1 : c.nx, j);
    if (relax) {
        d.umean[k] = factime * (real)d.msum[2 * side] + (1 - factime) * d.umean[k];
        d.vmean[k] = factime * (real)d.msum[2 * side + 1] + (1 - factime) * d.vmean[k];
    } else { d.umean[k] = 0; d.vmean[k] = 0; }
}

// ---------------------------------------------------------------------------------------------
// front = nonh_1d. zmode/wmode: 0 use zi/wi, 1 use zs(2,:)*fz / ws(2,:)*fz (boundary column not set)
// ---------------------------------------------------------------------------------------------
__global__ void k_bc_front(int zmode, int wmode, real fz, int isSetV, int q3) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j < JZ0 || j > JZ1) return;
    size_t k1 = ID(1, j), k2 = ID(2, j);
    real zi = zmode ? d.zs[k2] * fz : d.zi[j - 1];
    real wi = wmode ? d.ws[k2] * fz : d.wi[j - 1];
    if (c.arc == 0) {
        d.uu[k1] = d.ui[j - 1];
        d.vv[k1] = d.vi[j - 1];
        d.zs[k1] = zi + d.zs0[k1];
        d.ws[k1] = wi;
        if (q3) d.dU[k1] = d.dui[j - 1];
    } else {
        d.uu[k1] = d.ui[j - 1] - sqrt(c.g / d.hh[k1]) * (d.zs[k2] - zi - d.zs0[k2]) + d.umean[k1];
        if (q3) d.dU[k1] = d.dui[j - 1];
        d.vv[k1] = d.vi[j - 1];
        d.zs[k1] = d.zs[k2];
        d.ws[k1] = d.ws[k2];
    }
    if (!isSetV) d.vv[k1] = d.vv[k2];
}
__global__ void k_bc_front_corners() {
    if (threadIdx.x != 0) return;
    int ny = c.ny;
    d.uu[ID(1, 1)] = d.uu[ID(1, 2)];
    d.vv[ID(1, 1)] = d.vv[ID(1, 2)];
    d.zs[ID(1, 1)] = fmax(d.zs[ID(1, 2)] - d.dzs0dn[ID(1, 1)] * dnv(1, 1), d.zb[ID(1, 1)]);
    d.uu[ID(1, ny + 1)] = d.uu[ID(1, ny)];
    d.vv[ID(1, ny + 1)] = d.vv[ID(1, ny)];
    d.zs[ID(1, ny + 1)] = fmax(d.zs[ID(1, ny)] + d.dzs0dn[ID(1, ny + 1)] * dnv(1, ny), d.zb[ID(1, ny + 1)]);
}

// ---------------------------------------------------------------------------------------------
// back boundary
// ---------------------------------------------------------------------------------------------
__global__ void k_bc_back1(real dt) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    int nx = c.nx;
    bool jz = j >= JZ0 && j <= JZ1;
    if (c.back == BACK_WALL) {
        if (jz) { d.uu[ID(nx, j)] = 0; d.zs[ID(nx + 1, j)] = d.zs[ID(nx, j)]; }
    } else if (c.back == BACK_ABS_1D) {
        if (jz) d.uu[ID(nx, j)] = sqrt(c.g / d.hh[ID(nx, j)]) * (d.zs[ID(nx, j)] - fmax(d.zb[ID(nx, j)], d.zs0[ID(nx, j)])) + d.umean[ID(nx, j)];
        d.zs[ID(nx + 1, j)] = d.zs[ID(nx, j)];
    } else {  // abs_2d: Van Dongeren (1997) weakly reflective boundary, swave = 0 (Fx = 0); ny > 2 only
        if (!jz || d.wetu[ID(nx, j)] != 1) return;
        real g = c.g;
        real ht1 = fmax(d.zs0[ID(nx, j)] - d.zb[ID(nx, j)], c.eps), ht2 = fmax(d.zs0[ID(nx + 1, j)] - d.zb[ID(nx + 1, j)], c.eps);
        auto beta1 = [&](int jj) { return d.uu[ID(nx - 1, jj)] + 2 * sqrt(g * d.hum[ID(nx - 1, jj)]); };
        real hum = d.hum[ID(nx, j)], u = d.uu[ID(nx, j)], vu = d.vu[ID(nx, j)];
        real b1 = beta1(j), b2 = u + 2 * sqrt(g * hum);
        real dvdy = (d.vu[ID(nx, j + 1)] - d.vu[ID(nx, j - 1)]) / (2 * dnu(nx, j));
        real dhdx = (ht2 - ht1) / dsu(nx, j);
        real dbetadx = (b2 - b1) / dsz(nx, j);
        real dbetady = (beta1(j + 1) - beta1(j - 1)) / (2 * dnu(nx, j));
        real bn = -(u + sqrt(g * hum)) * dbetadx - vu * dbetady - sqrt(g * hum) * dvdy
                  - d.cfu[ID(nx, j)] * sqrt(u * u + vu * vu) * u / hum + g * dhdx;
        real bnp1 = b2 + bn * dt;
        real a2 = c.theta0, ur = 0, umn = d.umean[ID(nx, j)];
        const real px = c.px;
        for (int it = 0; it < 50; it++) {
            ur = cos(a2) / (cos(a2) + 1) * (bnp1 - umn - 2 * sqrt(g * (real)0.5 * (ht1 + ht2)));
            real vert = vu - d.vmean[ID(nx, j)];
            real an = atan(vert / (ur + (real)1e-16));
            if (an > px * (real)0.5) an -= px;
            if (an <= -px * (real)0.5) an += px;
            if (fabs(an - a2) < F32(0.001)) break;
            a2 = an;
        }
        real unew = ur + umn;
        d.uu[ID(nx, j)] = unew;
        d.zs[ID(nx + 1, j)] = (real)1.5 * ((bnp1 - unew) * (bnp1 - unew) / 4 / g + (real)0.5 * (d.zb[ID(nx, j)] + d.zb[ID(nx + 1, j)]))
                            - (real)0.5 * ((b1 - d.uu[ID(nx - 1, j)]) * (b1 - d.uu[ID(nx - 1, j)]) / 4 / g
                                           + (real)0.5 * (d.zb[ID(nx - 1, j)] + d.zb[ID(nx, j)]));
    }
}
__global__ void k_bc_back2() {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j < JZ0 || j > JZ1) return;
    int nx = c.nx;
    d.vv[ID(nx + 1, j)] = d.vv[ID(nx, j)];
    d.uu[ID(nx + 1, j)] = d.uu[ID(nx, j)];
}
__global__ void k_bc_back_corners() {
    if (threadIdx.x != 0) return;
    int nx = c.nx, ny = c.ny;
    d.uu[ID(nx + 1, 1)] = d.uu[ID(nx + 1, 2)];
    d.vv[ID(nx + 1, 1)] = d.vv[ID(nx + 1, 2)];
    d.zs[ID(nx + 1, 1)] = fmax(d.zs[ID(nx + 1, 2)] - d.dzs0dn[ID(1, 1)] * dnv(1, 1), d.zb[ID(1, 1)]);           // (1,1) as in XBeach
    d.uu[ID(nx + 1, ny + 1)] = d.uu[ID(nx + 1, ny)];
    d.vv[ID(nx + 1, ny + 1)] = d.vv[ID(nx + 1, ny)];
    d.zs[ID(nx + 1, ny + 1)] = fmax(d.zs[ID(nx + 1, ny)] + d.dzs0dn[ID(1, ny + 1)] * dnv(1, ny), d.zb[ID(1, ny + 1)]);
}
__global__ void k_bc_lat() {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < 2 || i > c.nx) return;
    d.zs[ID(i, 1)] = fmax(d.zs[ID(i, 2)] - d.dzs0dn[ID(i, 1)] * dnv(i, 1), d.zb[ID(i, 1)]);
    d.zs[ID(i, c.NY1)] = fmax(d.zs[ID(i, c.ny)] + d.dzs0dn[ID(i, c.NY1)] * dnv(i, c.ny), d.zb[ID(i, c.NY1)]);
}

// wind projection on the grid (readwind / flow_bc): windsu along s, windnv along n
__global__ void k_wind(real wx, real wy) {
    IJ
    size_t k = ID(i, j);
    real au = alfau(i, j), av = alfav(i, j) - (real)0.5 * c.px;
    d.windsu[k] = wx * cos(au) + wy * sin(au);
    d.windnv[k] = wy * cos(av) - wx * sin(av);
}
