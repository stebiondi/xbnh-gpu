// k_secorder.cuh: flow_secondorder.F90 flow_secondorder_advUV and flow_secondorder_advW (serial bounds:
// advUV imin = 3, imax = nx-1, jmin = 3, jmax = ny-1; advW imin = 2, imax = nx-1, jmin = 2, jmax = ny-1; cyclic
// uses the XBeach MPI interior bounds held in c.uv_jmin/uv_jmax and c.w_jmin/w_jmax).
// XBeach keeps wrk1 between calls. In 2D every value read was written in the same call; in 1D advW skips
// masked points without resetting wrk1, so they keep the value of the previous advUV call: d.wrk1p holds it.
#pragma once

// ---- advW: vertical momentum (w at cell centres), mask nonhZ
__global__ void k_advW_flux(const real *w, const real *wo) {
    IJ
    size_t k = ID(i, j);
    if (!c.oned) {
        real f1 = 0, f2 = 0;
        int imin = 2, imax = c.nx - 1, jmin = c.w_jmin, jmax = c.w_jmax;
        if (j >= 2 && j <= c.ny && i >= 1 && i <= c.nx && d.nonhZ[k] != 0) {
            int ie = min(i + 1, imax + 1), iee = min(i + 2, imax + 1), iw = max(i - 1, 1);
            real rfac = (real)(d.nonhZ[ID(ie, j)] * d.nonhZ[ID(iee, j)] * d.nonhZ[ID(iw, j)]);
            real zsmin = 1e30, zbmax = -1e30;
            for (int ii = iw; ii <= iee; ii++) { zsmin = fmin(zsmin, d.zs[ID(ii, j)]); zbmax = fmax(zbmax, d.zb[ID(ii, j)]); }
            real qx = d.qx[k] * dnu(i, j) * rfac;
            if (zsmin - zbmax > c.eps) {
                if (qx > 0 && i > imin)
                    f1 = (real)0.5 * dsu(i, j) * minmod((w[ID(ie, j)] - wo[k]) / dsu(i, j), (w[k] - wo[ID(iw, j)]) / dsu(iw, j)) * qx;
                else if (qx < 0 && i < imax)
                    f1 = (real)-0.5 * dsu(i, j) * minmod((wo[ID(ie, j)] - w[k]) / dsu(i, j), (wo[ID(iee, j)] - w[ID(ie, j)]) / dsu(ie, j)) * qx;
            }
        }
        if (j >= 1 && j <= c.ny && i >= 2 && i <= c.nx && d.nonhZ[k] != 0) {
            int jn = min(j + 1, jmax + 1), jnn = min(j + 2, jmax + 1), js = max(j - 1, 1);
            real rfac = (real)(d.nonhZ[ID(i, jn)] * d.nonhZ[ID(i, jnn)] * d.nonhZ[ID(i, js)]);
            real zsmin = 1e30, zbmax = -1e30;
            for (int jj = js; jj <= jnn; jj++) { zsmin = fmin(zsmin, d.zs[ID(i, jj)]); zbmax = fmax(zbmax, d.zb[ID(i, jj)]); }
            real qy = d.qy[k] * dsv(i, j) * rfac;
            if (zsmin - zbmax > c.eps) {
                if (qy > 0 && j > jmin)
                    f2 = (real)0.5 * dnv(i, j) * minmod((w[ID(i, jn)] - wo[k]) / dnv(i, j), (w[k] - wo[ID(i, js)]) / dnv(i, js)) * qy;
                else if (qy < 0 && j < jmax)
                    f2 = (real)-0.5 * dnv(i, j) * minmod((wo[ID(i, jn)] - w[k]) / dnv(i, j), (wo[ID(i, jnn)] - w[ID(i, jn)]) / dnv(i, jn)) * qy;
            }
        }
        if (i == 1 || i == c.nx) f1 = 0;
        if (!c.cyclic && (j == 1 || j == c.ny)) f2 = 0;
        d.w1[k] = f1; d.w2[k] = f2;
    } else {
        if (j != 1) return;
        if (i >= 2 && i <= c.nx && d.nonhZ[k] != 0) {
            real f1 = 0;
            int ie = min(i + 1, c.nx), iee = min(i + 2, c.nx), iw = max(i - 1, 1);
            real rfac = (real)(d.nonhZ[ID(ie, 1)] * d.nonhZ[ID(iee, 1)] * d.nonhZ[ID(iw, 1)]);
            real zsmin = 1e30, zbmax = -1e30;
            for (int ii = iw; ii <= iee; ii++) { zsmin = fmin(zsmin, d.zs[ID(ii, 1)]); zbmax = fmax(zbmax, d.zb[ID(ii, 1)]); }
            real qx = d.qx[k] * rfac;
            if (zsmin - zbmax > c.eps) {
                if (qx > 0 && i > 2)
                    f1 = (real)0.5 * dsu(i, 1) * minmod((w[ID(ie, 1)] - wo[k]) / dsu(i, 1), (w[k] - wo[ID(iw, 1)]) / dsu(iw, 1)) * qx;
                else if (qx < 0 && i < c.nx - 1)
                    f1 = (real)-0.5 * dsu(i, 1) * minmod((wo[ID(ie, 1)] - w[k]) / dsu(i, 1), (wo[ID(iee, 1)] - w[ID(ie, 1)]) / dsu(ie, 1)) * qx;
            }
            d.wrk1p[k] = f1;
        }
        if (i == 1 || i == c.nx) d.wrk1p[k] = 0;
    }
}
__global__ void k_advW_apply(real *w, real dt) {
    IJ
    if (!IN_ZS) return;
    size_t k = ID(i, j);
    if (d.nonhZ[k] == 0) return;
    if (!c.oned) w[k] -= dt * dsdnzi(i, j) * (d.w1[k] + d.w2[k] - d.w1[ID(i - 1, j)] - d.w2[ID(i, j - 1)]) / d.hh[k];
    else w[k] -= dt / d.hh[k] * (d.wrk1p[k] - d.wrk1p[ID(i - 1, 1)]) / dsz(i, 1);
}

// ---- advUV
__global__ void k_advUV_flux() {
    IJ
    size_t k = ID(i, j);
    const real *uu = d.uu, *uo = d.uu_old, *vv = d.vv, *vo = d.vv_old;
    int imin = 3, imax = c.nx - 1, jmin = c.uv_jmin, jmax = c.uv_jmax;
    real a1 = 0, a2 = 0, b1 = 0, b2 = 0;
    if (i >= 2 && i <= c.nx && j >= JU0 && j <= JU1) {   // u, s-flux (wrk1), j = jmin1d..jmax1d
        int ie = min(i + 1, imax + 1), iw = max(i - 1, 1), iww = max(i - 2, 1), iee = min(i + 2, imax + 1);
        real zsmin = 1e30, zbmax = -1e30;
        for (int ii = iww; ii <= iee; ii++) { zsmin = fmin(zsmin, d.zs[ID(ii, j)]); zbmax = fmax(zbmax, d.zb[ID(ii, j)]); }
        real qx = (real)0.5 * (d.qx[k] + d.qx[ID(iw, j)]) * cos_da(alfau(i, j) - alfau(iw, j));
        if (zsmin - zbmax > c.eps) {
            if (qx > 0 && i > imin)
                a1 = (real)0.5 * dsu(iw, j) * minmod((uu[k] - uo[ID(iw, j)]) / dsz(i, j), (uu[ID(iw, j)] - uo[ID(iww, j)]) / dsz(iw, j)) * qx;
            else if (qx < 0 && i < imax + 1)
                a1 = (real)-0.5 * dsu(i, j) * minmod((uo[k] - uu[ID(iw, j)]) / dsz(i, j), (uo[ID(ie, j)] - uu[k]) / dsz(ie, j)) * qx;
        }
        if (c.oned) d.wrk1p[k] = a1;
    }
    if (!c.oned) {
        if (IN(2, c.nx - 1, 2, c.ny)) {   // u, n-flux (wrk2)
            int js = min(j + 1, jmax + 1), jss = min(j + 2, jmax + 1), jn = max(j - 1, 1);
            real zsmin = 1e30, zbmax = -1e30;
            for (int jj = jn; jj <= jss; jj++)
                for (int ii = i; ii <= i + 1; ii++) { zsmin = fmin(zsmin, d.zs[ID(ii, jj)]); zbmax = fmax(zbmax, d.zb[ID(ii, jj)]); }
            if (zsmin - zbmax > c.eps) {
                real qy = (d.qy[ID(i + 1, j)] + d.qy[k]) * (real)0.5 * cos_da(alfau(i, j) - alfau(i, j + 1));
                if (qy > 0 && j > jmin - 1)
                    a2 = (real)0.5 * dnv(i, j) * minmod((uu[ID(i, js)] - uo[k]) / dnv(i, j), (uu[k] - uo[ID(i, jn)]) / dnv(i, jn)) * qy;
                else if (qy < 0 && j < jmax)
                    a2 = (real)-0.5 * dnv(i, j) * minmod((uo[ID(i, js)] - uu[k]) / dnv(i, j), (uo[ID(i, jss)] - uu[ID(i, js)]) / dnv(i, js)) * qy;
            }
        }
        if (IN(2, c.nx, 2, c.ny)) {   // v, n-flux (wrk1)
            int js = min(j + 1, jmax + 1), jn = max(j - 1, 1), jnn = max(j - 2, 1), jss = min(j + 2, jmax + 1);
            real zsmin = 1e30, zbmax = -1e30;
            for (int jj = jnn; jj <= jss; jj++) { zsmin = fmin(zsmin, d.zs[ID(i, jj)]); zbmax = fmax(zbmax, d.zb[ID(i, jj)]); }
            real qy = (real)0.5 * (d.qy[k] + d.qy[ID(i, jn)]) * cos_da(alfav(i, j) - alfav(i, jn)) * dsz(i, j);
            if (zsmin - zbmax > c.eps) {
                if (qy > 0 && j > jmin)
                    b1 = (real)0.5 * dnv(i, jn) * minmod((vv[k] - vo[ID(i, jn)]) / dnz(i, j), (vv[ID(i, jn)] - vo[ID(i, jnn)]) / dnz(i, jn)) * qy;
                else if (qy < 0 && j < jmax + 1)
                    b1 = (real)-0.5 * dnv(i, j) * minmod((vo[k] - vv[ID(i, jn)]) / dnz(i, j), (vo[ID(i, js)] - vv[k]) / dnz(i, js)) * qy;
            }
        }
        if (IN(2, c.nx, 2, c.ny - 1)) {   // v, s-flux (wrk2)
            int ie = min(i + 1, c.nx), iee = min(i + 2, c.nx), iw = max(i - 1, 1);
            real zsmin = 1e30, zbmax = -1e30;
            for (int jj = j; jj <= j + 1; jj++)
                for (int ii = iw; ii <= iee; ii++) { zsmin = fmin(zsmin, d.zs[ID(ii, jj)]); zbmax = fmax(zbmax, d.zb[ID(ii, jj)]); }
            if (zsmin - zbmax > c.eps) {
                real qx = (real)0.5 * (d.qx[ID(i, j + 1)] + d.qx[k]) * cos_da(alfav(i, j) - alfav(i + 1, j)) * dsc(i, j);
                if (qx > c.umin && i > imin - 1)
                    b2 = (real)0.5 * dsu(i, 1) * minmod((vv[ID(ie, j)] - vo[k]) / dsu(i, 1), (vv[k] - vo[ID(iw, j)]) / dsu(iw, 1)) * qx;
                else if (qx < -c.umin && i < imax)
                    b2 = (real)-0.5 * dsu(i, 1) * minmod((vo[ID(ie, j)] - vv[k]) / dsu(ie, 1), (vo[ID(iee, j)] - vv[ID(ie, j)]) / dsu(ie, 1)) * qx;
            }
        }
        d.w1[k] = a1; d.w2[k] = a2; d.w3[k] = b1; d.w4[k] = b2;
    }
}
__global__ void k_advUV_apply(real dt) {
    IJ
    size_t k = ID(i, j);
    if (!c.oned) {
        if (IN(2, c.nx - 1, 2, c.ny)) {
            real fac = dt / d.hum[k] * dsdnui(i, j);
            real w2s = (j - 1 >= 2) ? d.w2[ID(i, j - 1)] : 0;      // wrk2(:,1) = 0
            d.uu[k] -= fac * (dnz(i + 1, j) * d.w1[ID(i + 1, j)] - dnz(i, j) * d.w1[k] + dsc(i, j) * d.w2[k] - dsc(i, j - 1) * w2s);
        }
        if (c.ny > 2 && IN(2, c.nx, 2, c.ny - 1)) {
            real fac = dt / d.hvm[k] * dsdnvi(i, j);
            real w4w = (i - 1 >= 2) ? d.w4[ID(i - 1, j)] : 0;      // wrk2(1,:) = 0
            d.vv[k] -= fac * (d.w4[k] - w4w + d.w3[ID(i, j + 1)] - d.w3[k]);
        }
    } else if (j == 1 && i >= 2 && i <= c.nx - 1) {
        d.uu[k] -= dt / d.hum[k] * ((d.wrk1p[ID(i + 1, 1)] - d.wrk1p[k]) / dsu(i, 1));
    }
}
