// k_nonh.cuh: nonh.F90. One-layer scheme (nonh_1lay_pred, nonh_1lay_cor; 2D and "superfast 1D" branches),
// reduced two-layer scheme (nonhq3d: nonh_2lay_pred_3d/_cor_3d and _pred_2dV/_cor_2dV), nonh_break
// (nhbreaker = 1) and zuzv. The pressure system is solved in solver.cuh.
#pragma once

// ---------------------------------------------------------------------------------------------
// zuzv: bed and surface levels at u/v points
// ---------------------------------------------------------------------------------------------
__global__ void k_zuzv() {
    IJ
    size_t k = ID(i, j);
    int nx = c.nx, ny = c.ny;
    real zbu = i <= nx ? fmax(d.zb[k], d.zb[ID(min(nx, i) + 1, j)]) : d.zb[k];
    d.zbu[k] = zbu;
    real zbv;
    if (ny > 2) zbv = j <= ny ? fmax(d.zb[k], d.zb[ID(i, min(ny, j) + 1)]) : d.zb[k];
    else zbv = d.zb[k];
    d.zbv[k] = zbv;
    if (i <= nx) d.zsu[k] = d.hu[k] + zbu;
    if (ny > 2) { if (j <= ny) d.zsv[k] = d.hv[k] + zbv; }
    else d.zsv[k] = d.zs[ID(i, min(2, ny + 1))];
}

// ---------------------------------------------------------------------------------------------
// nonh_init_wcoef with dispc <= 0: optimum dispersion coefficient from the current zs0 and zb (disper,
// accuracy 0.0001), in double precision. XBeach calls it at start-up, at the start of each boundary file and,
// with wbctype = params, every time step.
// ---------------------------------------------------------------------------------------------
__global__ void k_init_wcoef(double sigma) {
    IJ
    size_t k = ID(i, j);
    double g = c.g, dd = fmax((double)d.zs0[k] - (double)d.zb[k], (double)c.eps);
    double w2 = sigma * sigma, alpha = dd * w2 / g, kk = alpha * (1.0 / sqrt(tanh(alpha))) / dd;
    double Lw = 2 * M_PI / kk, Ld = g * 2 * M_PI / w2, err = fabs(g * kk * tanh(kk * dd) - w2) / w2;
    while (err > 0.0001) { Lw = Ld * tanh(2 * M_PI * dd / Lw); kk = 2 * M_PI / Lw; err = fabs(g * kk * tanh(kk * dd) - w2) / w2; }
    d.wcoef[k] = dd > 0 ? (real)(1.0 / (4.0 * g / (dd * sigma * sigma) - 4.0 / ((kk * dd) * (kk * dd)))) : (real)1;
}

// ---------------------------------------------------------------------------------------------
// nonh_break: nonhU/nonhV/nonhZ masks and the hydrostatic-front breaker (nhbreaker = 1)
// ---------------------------------------------------------------------------------------------
__global__ void k_nonhUV() {
    IJ
    size_t k = ID(i, j);
    int ie = min(c.nx, i + 1), js = min(c.ny, j + 1);
    d.nonhU[k] = (d.wetu[k] == 1 && (real)0.5 * (d.zs[k] + d.zs[ID(ie, j)]) > d.zbu[k]) ? 1 : 0;
    d.nonhV[k] = (c.ny > 2 && d.wetv[k] == 1 && (real)0.5 * (d.zs[k] + d.zs[ID(i, js)]) > d.zbv[k]) ? 1 : 0;
    if (c.xbg_brk) {   // XBeach-G (nhbreaker 2, 3): also not where the grid cannot resolve kd > kdmin
        if (!(dsu(i, 1) * c.kdmin / c.px < d.hum[k])) d.nonhU[k] = 0;
        if (!(dnv(1, j) * c.kdmin / c.px < d.hvm[k])) d.nonhV[k] = 0;
    }
}
__device__ __forceinline__ real dzdt_at(int i, int j) {
    real r = -(d.uu[ID(i, j)] * d.hu[ID(i, j)] - d.uu[ID(i - 1, j)] * d.hu[ID(i - 1, j)]) / dsz(i, j);
    if (c.ny > 1) r -= (d.vv[ID(i, j)] * d.hv[ID(i, j)] - d.vv[ID(i, j - 1)] * d.hv[ID(i, j - 1)]) / dnz(i, j);
    return r;
}
__global__ void k_break() {
    IJ
    size_t k = ID(i, j);
    if (!IN_ZS) { d.brk_new[k] = d.breaking[k]; return; }
    int js = max(j - 1, 1);
    d.nonhZ[k] = max(max(d.nonhV[k], d.nonhV[ID(i, js)]), max(d.nonhU[k], d.nonhU[ID(i - 1, j)])) > 0 ? 1 : 0;
    if (c.xbg_brk) {   // XBeach-G nonh_cor: breaking from the surface vertical velocity, lower threshold near breakers
        int jmin = c.oned ? 1 : 2, jmax = c.oned ? 1 : c.ny, near = 0;
        for (int jj = j - 1; jj <= j + 1; jj++) for (int ii = i - 1; ii <= i + 1; ii++)
            if (ii >= 2 && ii <= c.nx && jj >= jmin && jj <= jmax && (c.oned ? jj == 1 : true) && d.breaking[ID(ii, jj)] == 1) near = 1;
        real lbc = near ? c.secbrsteep : c.maxbrsteep, wcr = lbc * sqrt(c.g * d.hh[k]);
        int b = d.breaking[k];
        if (c.nhbreaker == 2) { if (b == 1) { if (d.ws[k] <= 0) b = 0; } else if (d.ws[k] >= wcr) b = 1; }
        else { if (d.wscrit) d.wscrit[k] = wcr; if (b == 1) { if (d.ws[k] <= wcr) b = 0; } else if (d.ws[k] >= wcr) b = 1; }
        d.brk_new[k] = b;
        return;
    }
    if (c.nhbreaker != 1) { d.brk_new[k] = d.breaking[k]; return; }
    real dzdt = dzdt_at(i, j);
    // XBeach updates s%breaking in place in a (j,i) sweep: west/south neighbours are already processed. A
    // neighbour that was 1 and stopped breaking (dzdt < 0) reads -1; a newly breaking one reads 2.
    int bw = d.breaking[ID(i - 1, j)];
    if (bw == 1 && i - 1 >= 2 && dzdt_at(i - 1, j) < 0) bw = -1;
    int nb = (bw == 1 || d.breaking[ID(i + 1, j)] == 1) ? 1 : 0;
    if (c.ny > 1) {
        int bs = d.breaking[ID(i, j - 1)];
        if (bs == 1 && j - 1 >= c.jlo && dzdt_at(i, j - 1) < 0) bs = -1;
        if (bs == 1 || d.breaking[ID(i, j + 1)] == 1) nb = 1;
    }
    real wmax = sqrt(c.g * d.hh[k]);
    int b = d.breaking[k];
    if (b == 0) {
        if (dzdt > c.maxbrsteep * wmax) b = 1;
        else if (dzdt > c.reformsteep * wmax && nb == 1) b = 1;
    } else if (b == 1) {
        if (dzdt < 0) b = 0;
    }
    d.brk_new[k] = b;
}
__global__ void k_break_apply() {
    IJ
    size_t k = ID(i, j);
    int b = d.brk_new[k];
    d.breaking[k] = b;
    if ((c.nhbreaker == 1 && b != 0) || (c.xbg_brk && b == 1)) { d.nonhZ[k] = 0; d.pres[k] = 0; }
}

// =============================================================================================
// One-layer predictor (nonh_1lay_pred)
// =============================================================================================
// vertical momentum advection; neighbours processed earlier in the sweep are already zeroed if non-NH
__global__ void k_wm_adv(real dt) {
    IJ
    if (!IN_ZS) return;
    size_t k = ID(i, j);
    if (d.nonhZ[k] == 1) {
        real w0 = d.Wm_old[k], h = d.hh[k];
        real ww = (i - 1 >= 2 && d.nonhZ[ID(i - 1, j)] == 0) ? 0 : d.Wm_old[ID(i - 1, j)];
        real we = d.Wm_old[ID(i + 1, j)];
        real a = d.ddxu[i - 2] * fmax(d.qx[ID(i - 1, j)], (real)0) * (w0 - ww) / h
               + d.ddxu[i - 1] * fmin(d.qx[k], (real)0) * (we - w0) / h;
        if (!c.oned) {
            real wsx = (j - 1 >= c.jlo && d.nonhZ[ID(i, j - 1)] == 0) ? 0 : d.Wm_old[ID(i, j - 1)];
            real wn = d.Wm_old[ID(i, j + 1)];
            a += d.ddyv[j - 2] * fmax(d.qy[ID(i, j - 1)], (real)0) * (w0 - wsx) / h
               + d.ddyv[j - 1] * fmin(d.qy[k], (real)0) * (wn - w0) / h;
        }
        d.Wm[k] = w0 - dt * a;
    } else d.Wm[k] = 0;
}
__global__ void k_wm_zero() {
    IJ
    if (!IN_ZS) return;
    size_t k = ID(i, j);
    if (d.nonhZ[k] == 0) { d.Wm_old[k] = 0; d.ws[k] = 0; d.wb[k] = 0; d.pres[k] = 0; }
}
// vertical viscosity (XBeach uses hu for the y-fluxes too), pressure coefficients au/av
__global__ void k_wm_visc_coef(real dt) {
    IJ
    size_t k = ID(i, j);
    if (IN_ZS) {
        real w0 = d.Wm_old[k], h = d.hh[k];
        real dwdx1 = (real)0.5 * (d.nuh[ID(i - 1, j)] + d.nuh[k]) * d.hu[ID(i - 1, j)] * (w0 - d.Wm_old[ID(i - 1, j)]) * d.ddxu[i - 2];
        real dwdx2 = (real)0.5 * (d.nuh[ID(i + 1, j)] + d.nuh[k]) * d.hu[k] * (d.Wm_old[ID(i + 1, j)] - w0) * d.ddxu[i - 1];
        if (!c.oned) {
            real dwdy1 = d.nuh[ID(i, j - 1)] * d.hu[ID(i, j - 1)] * (w0 - d.Wm_old[ID(i, j - 1)]) * d.ddyv[j - 2];
            real dwdy2 = d.nuh[k] * d.hu[k] * (d.Wm_old[ID(i, j + 1)] - w0) * d.ddyv[j - 1];
            d.Wm[k] += (1 / h) * dt * (dwdx2 - dwdx1) * d.ddxz[i - 1] * (real)(d.wetu[k] * d.wetu[ID(i - 1, j)])
                     + (1 / h) * dt * (dwdy2 - dwdy1) * d.ddyz[j - 1] * (real)(d.wetv[k] * d.wetv[ID(i, j - 1)]);
        } else d.Wm[k] += (1 / h) * dt * (dwdx2 - dwdx1) * d.ddxz[i - 1] * (real)(d.wetu[k] * d.wetu[ID(i - 1, j)]);
        if (d.nonhU[k] == 1) {
            real vol = (real)0.5 * dt / (d.hum[k] * d.dxu[i - 1]);
            d.au1[k] = -(d.zs[ID(i + 1, j)] - d.zb[k]) * vol;
            d.au0[k] = (d.zs[k] - d.zb[ID(i + 1, j)]) * vol;
        } else { d.au1[k] = 0; d.au0[k] = 0; }
        if (c.ny > 2) {
            if (d.nonhV[k] == 1) {
                real vol = (real)0.5 * dt / (d.hvm[k] * d.dyv[j - 1]);
                d.av1[k] = -(d.zs[ID(i, j + 1)] - d.zb[k]) * vol;
                d.av0[k] = (d.zs[k] - d.zb[ID(i, j + 1)]) * vol;
            } else { d.av1[k] = 0; d.av0[k] = 0; }
        }
    }
}
// secorder: explicit old pressure in u, v, w
__global__ void k_pred_pres(real dt) {
    IJ
    size_t k = ID(i, j);
    if (IN_UU) d.uu[k] += d.au1[k] * d.pres[ID(i + 1, j)] + d.au0[k] * d.pres[k];
    if (IN_VV) {
        if (!c.oned) d.vv[k] += d.av1[k] * d.pres[ID(i, j + 1)] + d.av0[k] * d.pres[k];
        else d.vv[k] += d.av1[k] * d.pres[k] + d.av0[k] * d.pres[k];
    }
    if (IN_ZS && d.nonhZ[k] == 1) d.Wm[k] += d.wcoef[k] * dt * d.pres[k] / d.hh[k];
}

// =============================================================================================
// One-layer corrector (nonh_1lay_cor): 5-diagonal (2D) or 3-diagonal (1D) pressure system
// =============================================================================================
__global__ void k_build1(real dt) {
    IJ
    size_t k = ID(i, j);
    if (!IN_ZS) return;
    real ddxz = d.ddxz[i - 1], ddyz = d.ddyz[j - 1], dxz = d.dxz[i - 1], dyz = d.dyz[j - 1];
    if (d.nonhZ[k] == 1) {
        int uC = d.nonhU[k], uW = d.nonhU[ID(i - 1, j)];
        int vC = d.nonhV[k], vS = c.oned ? d.nonhV[k] : d.nonhV[ID(i, j - 1)];
        real zbuC = d.zbu[k], zbuW = d.zbu[ID(i - 1, j)], zb = d.zb[k];
        real zbvC = d.zbv[k], zbvS = c.oned ? d.zbv[k] : d.zbv[ID(i, j - 1)];
        real dzb_e = 0, dzb_w = 0, dzb_s = 0, dzb_n = 0;
        if (uC * uW == 1) { dzb_e = (real)0.5 * (zbuC - zbuW) * ddxz; dzb_w = dzb_e; }
        else if (uC == 0) { dzb_e = 0; dzb_w = (real)0.5 * (zb - zbuW) * ddxz; }
        else              { dzb_e = (real)0.5 * (zbuC - zb) * ddxz; dzb_w = 0; }
        if (!c.oned) {
            if (vC * vS == 1) { dzb_s = (real)0.5 * (zbvC - zbvS) * ddyz; dzb_n = dzb_s; }
            else if (vC == 0) { dzb_s = 0; dzb_n = (real)0.5 * (zb - zbvS) * ddyz; }
            else              { dzb_s = (real)0.5 * (zbvC - zb) * ddyz; dzb_n = 0; }
        } else {
            if (vC == 1) { dzb_s = 0; dzb_n = 0; }
            else { dzb_s = 0; dzb_n = (real)0.5 * (zb - zbvC) * ddyz; }
        }
        // in 1D XBeach uses av(:,i,1) for the "south" terms
        size_t kS = c.oned ? k : ID(i, j - 1);
        real au0C = d.au0[k], au1W = d.au1[ID(i - 1, j)], au0W = d.au0[ID(i - 1, j)], au1C = d.au1[k];
        real av0C = d.av0[k], av1S = d.av1[kS], av0S = d.av0[kS], av1C = d.av1[k];
        real awb1 = dzb_e * au0C + dzb_w * au1W + dzb_s * av0C + dzb_n * av1S;
        real awb2 = dzb_w * au0W, awb3 = dzb_e * au1C, awb4 = dzb_n * av0S, awb5 = dzb_s * av1C;
        real uC_ = d.uu[k], uW_ = d.uu[ID(i - 1, j)], vC_ = d.vv[k], vS_ = d.vv[kS];
        real awbr = dzb_e * uC_ + dzb_w * uW_ + dzb_s * vC_ + dzb_n * vS_;
        real aws1 = d.wcoef[k] * dt * 2 / d.hh[k] - awb1;
        real aws2 = -awb2, aws3 = -awb3, aws4 = -awb4, aws5 = -awb5;
        real awsr = 2 * d.Wm[k] - awbr;
        d.awb[0][k] = awb1; d.awb[1][k] = awb2; d.awb[2][k] = awb3; d.awb[3][k] = awb4; d.awb[4][k] = awb5; d.awbr[k] = awbr;
        d.aws[0][k] = aws1; d.aws[1][k] = aws2; d.aws[2][k] = aws3; d.aws[3][k] = aws4; d.aws[4][k] = aws5; d.awsr[k] = awsr;
        real zsuC = d.zsu[k], zsuW = d.zsu[ID(i - 1, j)], zs = d.zs[k];
        real zsvC = d.zsv[k], zsvS = d.zsv[kS];
        real dzs_e = 0, dzs_w = 0, dzs_s = 0, dzs_n = 0;
        if (uC * uW == 1) { dzs_e = (real)0.5 * (zsuC - zsuW) * dyz; dzs_w = dzs_e; }
        else if (uC == 0) { dzs_e = 0; dzs_w = (real)0.5 * (zs - zsuW) * dyz; }
        else              { dzs_e = (real)0.5 * (zsuC - zs) * dyz; dzs_w = 0; }
        if (!c.oned) {
            if (vC * vS == 1) { dzs_s = (real)0.5 * (zsvC - zsvS) * dxz; dzs_n = dzs_s; }
            else if (vC == 0) { dzs_s = 0; dzs_n = (real)0.5 * (zs - zsvS) * dxz; }
            else              { dzs_s = (real)0.5 * (zsvC - zs) * dxz; dzs_n = 0; }
        } else {
            if (vC == 1) { dzs_s = 0; dzs_n = 0; }
            else { dzs_s = 0; dzs_n = (real)0.5 * (zs - zsvC) * dxz; }
        }
        real huC = d.hu[k], huW = d.hu[ID(i - 1, j)], hvC = d.hv[k], hvS = d.hv[kS];
        real A = dxz * dyz;
        // 1D: XBeach writes dzs_n*av(1,i-1,1) in mat(1) (zero there: av = 0 when ny <= 2)
        real av1Sm = c.oned ? d.av1[ID(i - 1, j)] : av1S;
        d.mat[0][k] = dyz * (huC * au0C - huW * au1W) + dxz * (hvC * av0C - hvS * av1S)
                      - dzs_e * au0C - dzs_w * au1W - dzs_s * av0C - dzs_n * av1Sm + A * aws1;
        d.mat[1][k] = -dyz * huW * au0W - dzs_w * au0W + A * aws2;
        d.mat[2][k] = dyz * huC * au1C - dzs_e * au1C + A * aws3;
        d.mat[3][k] = -dxz * hvS * av0S - dzs_n * av0S + A * aws4;
        d.mat[4][k] = dxz * hvC * av1C - dzs_s * av1C + A * aws5;
        d.rhs[k] = -dyz * (huC * uC_ - huW * uW_) - dxz * (hvC * vC_ - hvS * vS_)
                   + dzs_e * uC_ + dzs_w * uW_ + dzs_s * vC_ + dzs_n * vS_ - A * awsr;
    } else {
        d.mat[0][k] = 1; d.mat[1][k] = 0; d.mat[2][k] = 0; d.mat[3][k] = 0; d.mat[4][k] = 0; d.rhs[k] = 0;
        for (int q = 0; q < 5; q++) { d.awb[q][k] = 0; d.aws[q][k] = 0; }
        d.awbr[k] = 0; d.awsr[k] = 0;
    }
}
__global__ void k_correct1() {
    IJ
    size_t k = ID(i, j);
    real dpk = d.dp[k];
    d.pres[k] += dpk;
    if (IN_UU) d.uu[k] += d.au1[k] * d.dp[ID(i + 1, j)] + d.au0[k] * dpk;
    if (IN_VV) {
        if (!c.oned) d.vv[k] += d.av1[k] * d.dp[ID(i, j + 1)] + d.av0[k] * dpk;
        else d.vv[k] += d.av1[k] * dpk + d.av0[k] * dpk;
    }
}
// Kinematic conditions at points excluded from the NH solve. XBeach reuses two stale loop variables there:
// dzs_n = dzb_s (last value of the awb loop) and dzb_w = dzs_e (last value of the ws loop); both are
// reproduced in loop order by k_stale_find / k_stale_val.
__global__ void k_wswb1() {
    IJ
    if (!IN_ZS) return;
    size_t k = ID(i, j);
    if (d.nonhZ[k] == 1) {
        size_t kS = c.oned ? k : ID(i, j - 1), kN = c.oned ? k : ID(i, j + 1);
        real dC = d.dp[k], dW = d.dp[ID(i - 1, j)], dE = d.dp[ID(i + 1, j)], dS = d.dp[kS], dN = d.dp[kN];
        d.ws[k] = d.awsr[k] + dC * d.aws[0][k] + dW * d.aws[1][k] + dE * d.aws[2][k] + dS * d.aws[3][k] + dN * d.aws[4][k];
        d.wb[k] = d.awbr[k] + dC * d.awb[0][k] + dW * d.awb[1][k] + dE * d.awb[2][k] + dS * d.awb[3][k] + dN * d.awb[4][k];
    } else {
        real ddxz = d.ddxz[i - 1], ddyz = d.ddyz[j - 1];
        int uC = d.wetu[k], uW = d.wetu[ID(i - 1, j)];
        size_t kS = c.oned ? k : ID(i, j - 1);
        int vC = d.wetv[k], vS = d.wetv[kS];
        real zs = d.zs[k], zb = d.zb[k];
        real dzs_e = 0, dzs_w = 0, dzs_s = 0, dzs_n = 0, dzb_e = 0, dzb_w = 0, dzb_s = 0, dzb_n = 0;
        if (uC * uW == 1) { dzs_e = (real)0.5 * (d.zsu[k] - d.zsu[ID(i - 1, j)]) * ddxz; dzs_w = dzs_e;
                            dzb_e = (real)0.5 * (d.zbu[k] - d.zbu[ID(i - 1, j)]) * ddxz; dzb_w = (real)d.stale[1]; }
        else if (uC == 0) { dzs_w = (real)0.5 * (zs - d.zsu[ID(i - 1, j)]) * ddxz; dzb_w = (real)0.5 * (zb - d.zbu[ID(i - 1, j)]) * ddxz; }
        else if (uW == 0) { dzs_e = (real)0.5 * (d.zsu[k] - zs) * ddxz; dzb_e = (real)0.5 * (d.zbu[k] - zb) * ddxz; }
        if (!c.oned) {
            if (vC * vS == 1) { dzs_s = (real)0.5 * (d.zsv[k] - d.zsv[kS]) * ddyz; dzs_n = (real)d.stale[0];
                                dzb_s = (real)0.5 * (d.zbv[k] - d.zbv[kS]) * ddyz; dzb_n = dzb_s; }
            else if (vC == 0) { dzs_n = (real)0.5 * (zs - d.zsv[kS]) * ddyz; dzb_n = (real)0.5 * (zb - d.zbv[kS]) * ddyz; }
            else if (vS == 0) { dzs_s = (real)0.5 * (d.zsv[k] - zs) * ddyz; dzb_s = (real)0.5 * (d.zbv[k] - zb) * ddyz; }
        } else {
            if (vC == 1) { dzs_s = 0; dzs_n = (real)d.stale[0]; dzb_s = 0; dzb_n = 0; }
            else { dzs_n = (real)0.5 * (zs - d.zsv[k]) * ddyz; dzb_n = (real)0.5 * (zb - d.zbv[k]) * ddyz; }
        }
        real uC_ = d.uu[k], uW_ = d.uu[ID(i - 1, j)], vC_ = d.vv[k], vS_ = d.vv[kS];
        d.ws[k] = -(d.hu[k] * uC_ - d.hu[ID(i - 1, j)] * uW_) * ddxz - (d.hv[k] * vC_ - d.hv[kS] * vS_) * ddyz
                  + dzs_e * uC_ + dzs_w * uW_ + dzs_s * vC_ + dzs_n * vS_;
        d.wb[k] = dzb_e * uC_ + dzb_w * uW_ + dzb_s * vC_ + dzb_n * vS_;
    }
}
// last NH point (awb and mat loops) and last non-NH point (ws loop) in Fortran loop order (j outer, i inner)
__global__ void k_stale_find() {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    unsigned kn = 0, ko = 0;
    if (i <= c.NX1 && j <= c.NY1 && IN_ZS) {
        unsigned key = (unsigned)j * c.NX1 + i;
        if (d.nonhZ[ID(i, j)] == 1) kn = key; else ko = key;
    }
    for (int o = 16; o > 0; o >>= 1) {
        kn = max(kn, __shfl_down_sync(0xffffffff, kn, o)); ko = max(ko, __shfl_down_sync(0xffffffff, ko, o));
    }
    __shared__ unsigned sn[32], so[32];
    int t = threadIdx.y * blockDim.x + threadIdx.x, lane = t & 31, w = t >> 5, nw = (blockDim.x * blockDim.y + 31) / 32;
    if (lane == 0) { sn[w] = kn; so[w] = ko; }
    __syncthreads();
    if (t == 0) {
        for (int q = 1; q < nw; q++) { kn = max(kn, sn[q]); ko = max(ko, so[q]); }
        kn = max(kn, sn[0]); ko = max(ko, so[0]);
        if (kn) atomicMax(&d.skey[0], (unsigned long long)kn);
        if (ko) atomicMax(&d.skey[1], (unsigned long long)ko);
    }
}
__global__ void k_stale_val() {
    if (threadIdx.x) return;
    double dzb_s = 0, dzs_e = 0;
    unsigned long long kn = d.skey[0], ko = d.skey[1];
    if (kn) {   // awb loop: dzb_s at the last NH point; mat loop: dzs_e there
        int i = (int)(kn % c.NX1), j = (int)(kn / c.NX1);
        size_t k = ID(i, j);
        int uC = d.nonhU[k], uW = d.nonhU[ID(i - 1, j)];
        if (!c.oned) {
            int vC = d.nonhV[k], vS = d.nonhV[ID(i, j - 1)];
            if (vC * vS == 1) dzb_s = 0.5 * (d.zbv[k] - d.zbv[ID(i, j - 1)]) * d.ddyz[j - 1];
            else if (vC == 0) dzb_s = 0;
            else dzb_s = 0.5 * (d.zbv[k] - d.zb[k]) * d.ddyz[j - 1];
        } else dzb_s = 0;
        if (uC * uW == 1) dzs_e = 0.5 * (d.zsu[k] - d.zsu[ID(i - 1, j)]) * d.dyz[j - 1];
        else if (uC == 0) dzs_e = 0;
        else dzs_e = 0.5 * (d.zsu[k] - d.zs[k]) * d.dyz[j - 1];
    }
    if (ko) {   // ws loop: dzs_e at the last non-NH point
        int i = (int)(ko % c.NX1), j = (int)(ko / c.NX1);
        size_t k = ID(i, j);
        int uC = d.wetu[k], uW = d.wetu[ID(i - 1, j)];
        if (uC * uW == 1) dzs_e = 0.5 * (d.zsu[k] - d.zsu[ID(i - 1, j)]) * d.ddxz[i - 1];
        else if (uC == 0) dzs_e = 0;
        else if (uW == 0) dzs_e = 0.5 * (d.zsu[k] - d.zs[k]) * d.ddxz[i - 1];
        else dzs_e = 0;
    }
    d.stale[0] = dzb_s; d.stale[1] = dzs_e;
}
__global__ void k_wswb_bc_j() {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    d.ws[ID(i, 1)] = d.ws[ID(i, 2)]; d.ws[ID(i, c.NY1)] = d.ws[ID(i, c.ny)];
    d.wb[ID(i, 1)] = d.wb[ID(i, 2)]; d.wb[ID(i, c.NY1)] = d.wb[ID(i, c.ny)];
}
__global__ void k_wswb_bc_i() {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    d.ws[ID(c.NX1, j)] = d.ws[ID(c.nx, j)];
    d.wb[ID(1, j)] = d.wb[ID(2, j)]; d.wb[ID(c.NX1, j)] = d.wb[ID(c.nx, j)];
}
__global__ void k_wmold1() {
    IJ
    size_t k = ID(i, j);
    d.Wm_old[k] = (real)0.5 * (d.ws[k] + d.wb[k]);
}

// =============================================================================================
// Reduced two-layer scheme (nonhq3d = 1), 2D: nonh_2lay_pred_3d / nonh_2lay_cor_3d
// =============================================================================================
__global__ void k_q3_coef(real dt) {
    IJ
    size_t k = ID(i, j);
    int nx = c.nx, ny = c.ny;
    const real *w = d.wcoef;
    if (j >= 2 && j <= ny && i >= 1 && i <= nx) {
        if (d.nonhU[k] == 1) {
            real vol = (real)0.5 * dt / (d.hum[k] * dsu(i, j));
            real Fac = (d.zb[ID(i + 1, j)] - d.zb[k]) * vol;
            d.au1[k] = -vol * (1 + w[k]) * d.hh[ID(i + 1, j)] - Fac;
            d.au0[k] = vol * (1 + w[k]) * d.hh[k] - Fac;
        } else { d.au1[k] = 0; d.au0[k] = 0; }
        if (c.nhlay > 0) {
            if (d.nonhU[k] * (1 - d.breaking[k]) * (1 - d.breaking[ID(max(i - 1, 1), j)]) == 1) {
                real vol = (real)0.5 * dt / (d.hum[k] * dsu(i, j));
                real fac = (d.zs[ID(i + 1, j)] - d.zs[k]) * vol / (1 - (real)0.5 * w[k] - (real)0.5 * w[ID(i + 1, j)])
                         - (w[ID(i + 1, j)] - w[k]) * d.hum[k] * vol / (2 - w[k] - w[ID(i + 1, j)]);
                d.adu1[k] = -d.hh[ID(i + 1, j)] * vol + fac;
                d.adu0[k] = d.hh[k] * vol + fac;
            } else { d.adu1[k] = 0; d.adu0[k] = 0; }
        }
    }
    if (j >= 1 && j <= ny && i >= 2 && i <= nx) {
        if (d.nonhV[k] == 1) {
            real vol = (real)0.5 * dt / (d.hvm[k] * dnv(i, j));
            real Fac = (d.zb[ID(i, j + 1)] - d.zb[k]) * vol;
            d.av1[k] = -vol * (1 + w[ID(i, j + 1)]) * d.hh[ID(i, j + 1)] - Fac;
            d.av0[k] = vol * (1 + w[k]) * d.hh[k] - Fac;
        } else { d.av1[k] = 0; d.av0[k] = 0; }
        if (c.nhlay > 0) {
            if (d.nonhV[k] * (1 - d.breaking[k]) * (1 - d.breaking[ID(i, max(j - 1, 1))]) == 1) {
                real vol = (real)0.5 * dt / (d.hvm[k] * dnv(i, j));
                real fac = (d.zs[k] - d.zs[ID(i, j + 1)]) * vol / (1 - (real)0.5 * w[k] - (real)0.5 * w[ID(i, j + 1)])
                         - (w[k] - w[ID(i, j + 1)]) * d.hvm[k] * vol / (2 - w[k] - w[ID(i, j + 1)]);
                d.adv1[k] = -d.hh[ID(i, j + 1)] * vol + fac;
                d.adv0[k] = d.hh[k] * vol + fac;
            } else { d.adv1[k] = 0; d.adv0[k] = 0; }
        }
    }
    if (IN(2, nx, 2, ny)) d.aws[0][k] = d.nonhZ[k] == 1 ? dt / (d.hh[k] * (1 - w[k])) : 0;
}
__global__ void k_q3_coef_bc() {   // au(:,1,:) = au(:,nx,:) = 0; av(:,:,1) = av(:,:,ny) = 0 (serial)
    IJ
    size_t k = ID(i, j);
    if (i == 1 || i == c.nx) { d.au0[k] = 0; d.au1[k] = 0; if (d.adu0) { d.adu0[k] = 0; d.adu1[k] = 0; } }
    if (!c.cyclic && (j == 1 || j == c.ny)) { d.av0[k] = 0; d.av1[k] = 0; if (d.adv0) { d.adv0[k] = 0; d.adv1[k] = 0; } }
}
__global__ void k_q3_wm_adv(real dt) {
    IJ
    if (!IN_ZS) return;
    size_t k = ID(i, j);
    if (d.nonhZ[k] == 1) {
        real qx[2], qy[2], fx[2], fy[2];
        for (int q = 0; q < 2; q++) {
            qx[q] = d.qx[ID(i - 1 + q, j)] * dnu(i - 1 + q, j);
            qy[q] = d.qy[ID(i, j - 1 + q)] * dsv(i, j - 1 + q);
            fx[q] = qx[q] > 0 ? d.Wm_old[ID(i - 1 + q, j)] * (real)d.nonhZ[ID(i - 1 + q, j)] : d.Wm_old[ID(i + q, j)] * (real)d.nonhZ[ID(i + q, j)];
            fy[q] = qy[q] > 0 ? d.Wm_old[ID(i, j - 1 + q)] * (real)d.nonhZ[ID(i, j - 1 + q)] : d.Wm_old[ID(i, j + q)] * (real)d.nonhZ[ID(i, j + q)];
        }
        real fac = dt / d.hh[k] * dsdnzi(i, j);
        d.w1[k] = d.Wm_old[k] + fac * d.Wm_old[k] * (qx[1] + qy[1] - qx[0] - qy[0])
                - fac * (qx[1] * fx[1] + qy[1] * fy[1] - qx[0] * fx[0] - qy[0] * fy[0]);
    } else d.w1[k] = 0;
}
__global__ void k_q3_wm_set() {   // Wm = advected value; Wm_old = pres = 0 at non-NH points
    IJ
    if (!IN_ZS) return;
    size_t k = ID(i, j);
    d.Wm[k] = d.w1[k];
    if (d.nonhZ[k] == 0) { d.Wm_old[k] = 0; d.pres[k] = 0; }
}
__global__ void k_q3_dudv_adv(real dt) {
    IJ
    size_t k = ID(i, j);
    const real sig[2] = {-1, 1};
    if (IN_UU) {
        if (d.wetu[k] == 1) {
            real qx[2], qy[2], fx[2], fy[2];
            for (int q = 0; q < 2; q++) {
                qx[q] = (real)0.5 * (d.qx[ID(i - 1 + q, j)] + d.qx[ID(i + q, j)]) * dnz(i + q, j);
                qy[q] = (real)0.5 * (d.qy[ID(i + 1, j - 1 + q)] + d.qy[ID(i, j - 1 + q)]) * dsc(i, j + q);
                real dvf = (real)0.5 * (d.dV0[ID(i + q, j - 1)] + d.dV0[ID(i + q, j)]);
                real cosd = cos_da(alfau(i + q, j) - alfau(i + q - 1, j));
                real sind = sig[q] * sin_da(alfau(i + q, j) - alfaz(i + q - 1, j));
                fx[q] = qx[q] > 0 ? d.dU0[ID(i - 1 + q, j)] * cosd + dvf * sind : d.dU0[ID(i + q, j)] * cosd + dvf * sind;
                cosd = cos_da(alfau(i, j + q - 1) - alfau(i, j + q));
                sind = sig[q] * sin_da(alfau(i, j + q - 1) - alfau(i, j + q));
                dvf = (real)0.5 * (d.dV0[ID(i, j + q - 1)] + d.dV0[ID(i + 1, j + q - 1)]);
                fy[q] = qy[q] > 0 ? d.dU0[ID(i, j - 1 + q)] * cosd + dvf * sind : d.dU0[ID(i, j + q)] * cosd + dvf * sind;
            }
            real fac = dt / d.hum[k] * dsdnui(i, j);
            d.dU[k] = d.dU0[k] + fac * d.dU0[k] * (qx[1] + qy[1] - qx[0] - qy[0])
                    - fac * (qx[1] * fx[1] + qy[1] * fy[1] - qx[0] * fx[0] - qy[0] * fy[0]);
        } else d.dU[k] = 0;
    }
    if (IN_VV) {
        if (d.wetv[k] == 1) {
            real qx[2], qy[2], fx[2], fy[2];
            for (int q = 0; q < 2; q++) {
                qy[q] = (real)0.5 * (d.qy[ID(i, j - 1 + q)] + d.qy[ID(i, j + q)]) * dsz(i, j + q);
                qx[q] = (real)0.5 * (d.qx[ID(i - 1 + q, j + 1)] + d.qx[ID(i - 1 + q, j)]) * dnc(i + q, j);
                real cosd = cos_da(alfav(i + q - 1, j) - alfav(i + q, j));
                real sind = sig[q] * sin_da(alfav(i + q - 1, j) - alfav(i + q, j));
                real duf = (real)0.5 * (d.dU0[ID(i - 1 + q, j)] + d.dU0[ID(i - 1 + q, j + 1)]);
                fx[q] = qx[q] > 0 ? d.dV0[ID(i - 1 + q, j)] * cosd + duf * sind : d.dV0[ID(i + q, j)] * cosd + duf * sind;
                cosd = cos_da(alfav(i, j + q) - alfav(i, j + q - 1));
                sind = sig[q] * sin_da(alfav(i, j + q) - alfav(i, j + q - 1));
                duf = (real)0.5 * (d.dU0[ID(i - 1, j + q)] + d.dU0[ID(i, j + q)]);
                fy[q] = qy[q] > 0 ? d.dV0[ID(i, j - 1 + q)] * cosd + duf * sind : d.dV0[ID(i, j + q)] * cosd + duf * sind;
            }
            real fac = dt / d.hvm[k] * dsdnvi(i, j);
            d.dV[k] = d.dV0[k] + fac * d.dV0[k] * (qx[1] + qy[1] - qx[0] - qy[0])
                    - fac * (qx[1] * fx[1] + qy[1] * fy[1] - qx[0] * fx[0] - qy[0] * fy[0]);
        } else d.dV[k] = 0;
    }
}
__global__ void k_q3_pred_pres() {   // secorder: explicit old pressure in u, v, dU, dV, Wm
    IJ
    size_t k = ID(i, j);
    if (IN_UU) {
        d.uu[k] += d.au1[k] * d.pres[ID(i + 1, j)] + d.au0[k] * d.pres[k];
        if (c.nhlay > 0) d.dU[k] += d.adu1[k] * d.pres[ID(i + 1, j)] + d.adu0[k] * d.pres[k];
    }
    if (!c.oned && IN_VV) {
        d.vv[k] += d.av1[k] * d.pres[ID(i, j + 1)] + d.av0[k] * d.pres[k];
        if (c.nhlay > 0) d.dV[k] += d.adv1[k] * d.pres[ID(i, j + 1)] + d.adv0[k] * d.pres[k];
    }
    if (IN_ZS) d.Wm[k] += d.aws[0][k] * d.pres[k];
}
// bottom friction, vertical mixing and vertical advection of the layer velocity difference (nhlay > 0)
__global__ void k_q3_fric(real dt) {
    IJ
    size_t k = ID(i, j);
    const real *w = d.wcoef;
    real nl = c.nhlay, ve = F32(0.0001);
    if (IN_UU && d.wetu[k] == 1 && w[k] > 0) {
        real ub = d.uu_old[k] + (1 - w[k]) * d.dU[k];
        real vb = d.vu[k] + (real)0.25 * (1 - w[k]) * (d.dV0[k] + d.dV0[ID(i + 1, j)] + d.dV0[ID(i, j - 1)] + d.dV0[ID(i + 1, j - 1)]);
        real umag = sqrt(ub * ub + vb * vb);
        real fric = dt * d.cfu[k] * umag / d.hu[k];
        real stress = 2 * dt * ve / (d.hum[k] * d.hum[k] * (nl - nl * nl));
        real advec = (real)0.5 * (d.omega[k] + d.omega[ID(i + 1, j)]) * dt / d.hum[k];
        real fac = (real)0.5 * (w[k] + w[ID(i + 1, j)]);
        real rhs = d.dU[k] - d.uu_old[k] * (fric + advec);
        real Amat = 1 + (1 - fac) * fric + stress + (1 - fac) * fmax(advec, (real)0) - fac * fmin(advec, (real)0);
        d.dU[k] = rhs / Amat;
    }
    if (IN_VV && d.wetv[k] == 1 && w[k] > 0) {
        real vb = d.vv_old[k] + (1 - w[k]) * d.dV[k];
        real ub = d.uv[k] + (real)0.25 * (1 - w[k]) * (d.dU0[k] + d.dU0[ID(i, j + 1)] + d.dU0[ID(i - 1, j)] + d.dU0[ID(i - 1, j + 1)]);
        real umag = sqrt(ub * ub + vb * vb);
        real fric = dt * d.cfu[k] * umag / d.hv[k];
        real stress = 2 * dt * ve / (d.hvm[k] * d.hvm[k] * (nl - nl * nl));
        real advec = (real)0.5 * (d.omega[k] + d.omega[ID(i, j + 1)]) * dt / d.hvm[k];
        real fac = (real)0.5 * (w[k] + w[ID(i + 1, j)]);
        real rhs = d.dV[k] - d.vv_old[k] * (fric + advec);
        real Amat = 1 + (1 - fac) * fric + stress + (1 - fac) * fmax(advec, (real)0) - fac * fmin(advec, (real)0);
        d.dV[k] = rhs / Amat;
    }
}
__global__ void k_build3() {
    IJ
    size_t k = ID(i, j);
    if (!IN(2, c.nx, 2, c.ny)) return;
    if (d.nonhZ[k] == 1) {
        const real *w = d.wcoef;
        real au[2], hfU[2], hfdU[2], av[2], hfV[2], hfdV[2];
        for (int q = 0; q < 2; q++) {
            au[q] = (real)0.5 * (w[ID(i + q, j)] + w[ID(i - 1 + q, j)]);
            hfU[q] = dsdnzi(i, j) * dnu(i - 1 + q, j) * d.hu[ID(i - 1 + q, j)] * (1 + au[q]);
            hfdU[q] = dsdnzi(i, j) * dnu(i - 1 + q, j) * d.hu[ID(i - 1 + q, j)] * au[q] * (1 - au[q]);
            av[q] = (real)0.5 * (w[ID(i, j + q)] + w[ID(i, j - 1 + q)]);
            hfV[q] = dsdnzi(i, j) * dsv(i, j - 1 + q) * d.hv[ID(i, j - 1 + q)] * (1 + av[q]);
            hfdV[q] = dsdnzi(i, j) * dsv(i, j - 1 + q) * d.hv[ID(i, j - 1 + q)] * av[q] * (1 - av[q]);
        }
        size_t kW = ID(i - 1, j), kS = ID(i, j - 1);
        real dzsdx = 0, dzbdx = 0, dzsdy = 0, dzbdy = 0;
        if (d.nonhU[k] * d.nonhU[kW] == 1) {
            dzsdx = (real)0.5 * (d.zsu[k] - d.zsu[kW]) / dsz(i, j);
            dzbdx = (real)0.5 * (d.zbu[k] + au[1] * d.hu[k] - d.zbu[kW] - au[0] * d.hu[kW]) / dsz(i, j);
        }
        if (d.nonhV[k] * d.nonhV[kS] == 1) {
            dzsdy = (real)0.5 * (d.zsv[k] - d.zsv[kS]) / dnz(i, j);
            dzbdy = (real)0.5 * (d.zbv[k] + av[1] * d.hv[k] - d.zbv[kS] - av[0] * d.hv[kS]) / dnz(i, j);
        }
        real alpha = -w[k], wc = w[k];
        real au0 = d.au0[k], au1W = d.au1[kW], au0W = d.au0[kW], au1 = d.au1[k];
        real av0 = d.av0[k], av1S = d.av1[kS], av0S = d.av0[kS], av1 = d.av1[k];
        real du0 = d.adu0[k], du1W = d.adu1[kW], du0W = d.adu0[kW], du1 = d.adu1[k];
        real dv0 = d.adv0[k], dv1S = d.adv1[kS], dv0S = d.adv0[kS], dv1 = d.adv1[k];
        d.mat[0][k] = 2 * d.aws[0][k] + hfU[1] * au0 - hfU[0] * au1W + hfV[1] * av0 - hfV[0] * av1S
                    + hfdU[1] * du0 - hfdU[0] * du1W + hfdV[1] * dv0 - hfdV[0] * dv1S
                    - dzsdx * (au0 + au1W - (du0 + du1W) * wc) - dzsdy * (av0 + av1S - (dv0 + dv1S) * wc)
                    - dzbdx * (au0 + au1W + (du0 + du1W) * alpha) - dzbdy * (av0 + av1S + (dv0 + dv1S) * alpha);
        d.mat[1][k] = -hfU[0] * au0W - hfdU[0] * du0W - dzsdx * (au0W - du0W * wc) - dzbdx * (au0W + du0W * alpha);
        d.mat[3][k] = -hfV[0] * av0S - hfdV[0] * dv0S - dzsdy * (av0S - dv0S * wc) - dzbdy * (av0S + dv0S * alpha);
        d.mat[2][k] = hfU[1] * au1 + hfdU[1] * du1 - dzsdx * (au1 - du1 * wc) - dzbdx * (au1 + du1 * alpha);
        d.mat[4][k] = hfV[1] * av1 + hfdV[1] * dv1 - dzsdy * (av1 - dv1 * wc) - dzbdy * (av1 + dv1 * alpha);
        real uu = d.uu[k], uuW = d.uu[kW], vv = d.vv[k], vvS = d.vv[kS];
        real dU = d.dU[k], dUW = d.dU[kW], dV = d.dV[k], dVS = d.dV[kS];
        d.rhs[k] = -2 * d.Wm[k] - hfU[1] * uu + hfU[0] * uuW - hfdU[1] * dU + hfdU[0] * dUW
                 - hfV[1] * vv + hfV[0] * vvS - hfdV[1] * dV + hfdV[0] * dVS
                 + dzsdx * (uu + uuW - (dU + dUW) * wc) + dzsdy * (vv + vvS - (dV + dVS) * wc)
                 + dzbdx * (uu + uuW + (dU + dUW) * alpha) + dzbdy * (vv + vvS + (dV + dVS) * alpha);
    } else {
        d.mat[0][k] = 1; d.mat[1][k] = 0; d.mat[2][k] = 0; d.mat[3][k] = 0; d.mat[4][k] = 0; d.rhs[k] = 0;
    }
}
__global__ void k_correct3() {
    IJ
    size_t k = ID(i, j);
    real dpk = d.dp[k];
    d.pres[k] = c.secorder ? d.pres[k] + dpk : dpk;
    if (IN_UU) {
        d.uu[k] += d.au1[k] * d.dp[ID(i + 1, j)] + d.au0[k] * dpk;
        if (c.nhlay > 0) d.dU[k] += d.adu1[k] * d.dp[ID(i + 1, j)] + d.adu0[k] * dpk;
    }
    if (!c.oned && IN_VV) {
        d.vv[k] += d.av1[k] * d.dp[ID(i, j + 1)] + d.av0[k] * dpk;
        if (c.nhlay > 0) d.dV[k] += d.adv1[k] * d.dp[ID(i, j + 1)] + d.adv0[k] * dpk;
    }
}
__global__ void k_q3_after(int last) {   // ws, wb, Wm, omega, Wm0 (2D); last = 1: Wm_old, dU0, dV0 roll
    IJ
    size_t k = ID(i, j);
    if (!IN_ZS) return;
    const real *w = d.wcoef;
    size_t kW = ID(i - 1, j), kS = ID(i, j - 1);
    real ws = 0, wb = 0, om = 0;
    if (d.wetz[k] == 1) {
        real dzsdx = (real)0.5 * (d.zsu[k] - d.zsu[kW]) / dsz(i, j);
        real dzsdy = (real)0.5 * (d.zsv[k] - d.zsv[kS]) / dnz(i, j);
        ws = -(dnu(i, j) * d.hu[k] * d.uu[k] - dnu(i - 1, j) * d.hu[kW] * d.uu[kW]) * dsdnzi(i, j)
             - (dsv(i, j) * d.hv[k] * d.vv[k] - dsv(i, j - 1) * d.hv[kS] * d.vv[kS]) * dsdnzi(i, j)
             + dzsdx * (d.uu[k] + d.uu[kW] - (d.dU[k] + d.dU[kW]) * w[k])
             + dzsdy * (d.vv[k] + d.vv[kS] - (d.dV[k] + d.dV[kS]) * w[k]);
        real dzbdx = (real)0.5 * (d.zbu[k] - d.zbu[kW]), dzbdy = (real)0.5 * (d.zbv[k] - d.zbv[kS]);
        wb = dzbdx * (d.uu[k] + d.uu[kW] + (d.dU[k] + d.dU[kW]) * (1 - w[k])) / dsz(i, j)
           + dzbdy * (d.vv[k] + d.vv[kS] + (d.dV[k] + d.dV[kS]) * (1 - w[k])) / dnz(i, j);
        om = -(dnu(i, j) * d.hu[k] * d.dU0[k] - dnu(i - 1, j) * d.hu[kW] * d.dU0[kW]
               + dsv(i, j) * d.hv[k] * d.dV0[k] - dsv(i, j - 1) * d.hv[kS] * d.dV0[kS]) * dsdnzi(i, j);
    }
    d.ws[k] = ws; d.wb[k] = wb; d.omega[k] = om;
    real Wm = d.nonhZ[k] == 1 ? d.Wm[k] + d.aws[0][k] * d.dp[k] : 0;
    d.Wm[k] = Wm;
    d.Wm0[k] = d.wetz[k] == 1 ? Wm + (real)0.5 * (wb - ws) : 0;
}
__global__ void k_q3_wm_bc() {   // Wm(1,:)=Wm(2,:); Wm(nx+1,:)=Wm(nx,:); Wm(:,1)=Wm(:,2); Wm(:,ny+1)=Wm(:,ny)
    int t = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (t <= c.NY1) { d.Wm[ID(1, t)] = d.Wm[ID(2, t)]; d.Wm[ID(c.NX1, t)] = d.Wm[ID(c.nx, t)]; }
}
__global__ void k_q3_wm_bc_j() {
    int t = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (t <= c.NX1 && !c.oned) { d.Wm[ID(t, 1)] = d.Wm[ID(t, 2)]; d.Wm[ID(t, c.NY1)] = d.Wm[ID(t, c.ny)]; }
}
__global__ void k_q3_roll() {
    IJ
    size_t k = ID(i, j);
    d.Wm_old[k] = d.Wm[k];
    d.dU0[k] = d.dU[k];
    if (!c.oned) d.dV0[k] = d.dV[k];
}

// =============================================================================================
// Reduced two-layer scheme, 1D: nonh_2lay_pred_2dV / nonh_2lay_cor_2dV (j = 1)
// =============================================================================================
__global__ void k_q1_coef(real dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < 2 || i > c.nx) return;
    size_t k = ID(i, 1), kE = ID(i + 1, 1);
    const real *w = d.wcoef;
    if (d.nonhU[k] == 1) {
        real vol = (real)0.5 * dt / (d.hum[k] * dsu(i, 1));
        real fac = (d.zb[kE] - d.zb[k]) * vol;
        d.au1[k] = -vol * (1 + w[kE]) * d.hh[kE] - fac;
        d.au0[k] = vol * (1 + w[k]) * d.hh[k] - fac;
    } else { d.au1[k] = 0; d.au0[k] = 0; }
    if (c.nhlay > 0) {
        if (d.nonhU[k] * (1 - d.breaking[k]) * (1 - d.breaking[ID(i - 1, 1)]) == 1) {
            real vol = (real)0.5 * dt / (d.hum[k] * dsu(i, 1));
            real fac = (d.zs[kE] - d.zs[k]) * vol / (1 - (real)0.5 * w[k] - (real)0.5 * w[kE])
                     - (w[kE] - w[k]) * d.hum[k] * vol / (2 - w[k] - w[kE]);
            d.adu1[k] = -d.hh[kE] * vol + fac;
            d.adu0[k] = d.hh[k] * vol + fac;
        } else { d.adu1[k] = 0; d.adu0[k] = 0; }
    }
    d.aws[0][k] = d.nonhZ[k] == 1 ? dt / (d.hh[k] * (1 - w[k])) : 0;
    if (i == c.nx) { d.au0[k] = 0; d.au1[k] = 0; if (d.adu0) { d.adu0[k] = 0; d.adu1[k] = 0; } }   // au(:,nx,:) = 0
}
__global__ void k_q1_adv(real dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < 2 || i > c.nx) return;
    size_t k = ID(i, 1);
    if (d.nonhZ[k] == 1) {
        real qx[2], fv[2];
        for (int q = 0; q < 2; q++) {
            qx[q] = d.qx[ID(i - 1 + q, 1)];
            fv[q] = qx[q] > 0 ? d.Wm_old[ID(i - 1 + q, 1)] * (real)d.nonhZ[ID(i - 1 + q, 1)] : d.Wm_old[ID(i + q, 1)] * (real)d.nonhZ[ID(i + q, 1)];
        }
        real fac = dt / d.hh[k] / dsz(i, 1);
        d.w1[k] = d.Wm_old[k] + fac * d.Wm_old[k] * (d.qx[k] - d.qx[ID(i - 1, 1)]) - fac * (qx[1] * fv[1] - qx[0] * fv[0]);
    } else d.w1[k] = 0;
    if (c.nhlay > 0 && i <= c.nx - 1) {
        if (d.wetu[k] == 1) {
            real qx[2], fv[2];
            for (int q = 0; q < 2; q++) {
                qx[q] = (real)0.5 * (d.qx[ID(i - 1 + q, 1)] + d.qx[ID(i + q, 1)]);
                fv[q] = qx[q] > 0 ? d.dU0[ID(i - 1 + q, 1)] : d.dU0[ID(i + q, 1)];
            }
            real fac = dt / d.hum[k] / dsu(i, 1);
            d.dU[k] = d.dU0[k] + (real)0.5 * fac * d.dU0[k] * (d.qx[ID(i + 1, 1)] - d.qx[ID(i - 1, 1)]) - fac * (qx[1] * fv[1] - qx[0] * fv[0]);
        } else d.dU[k] = 0;
    }
}
__global__ void k_q1_fric(real dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < 2 || i > c.nx - 1) return;
    size_t k = ID(i, 1);
    if (d.wetu[k] != 1) return;
    const real *w = d.wcoef;
    real nl = c.nhlay, ve = F32(1.0e-4);
    real ub = d.uu_old[k] + (1 - nl) * d.dU0[k];
    real umag = fabs(ub);
    real fric = dt * d.cfu[k] * umag / d.hu[k];
    real stress = 2 * dt * ve / (d.hum[k] * d.hum[k] * (nl - nl * nl));
    real advec = (real)0.5 * (d.omega[k] + d.omega[ID(i + 1, 1)]) * dt / d.hum[k];
    real fac = (real)0.5 * (w[k] + w[ID(i + 1, 1)]);
    real rhs = d.dU[k] - d.uu_old[k] * (fric + advec);
    real Amat = 1 + (1 - fac) * fric + stress + (1 - fac) * fmax(advec, (real)0) - fac * fmin(advec, (real)0);
    d.dU[k] = rhs / Amat;
}
__global__ void k_build3_1d() {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < 2 || i > c.nx) return;
    size_t k = ID(i, 1), kW = ID(i - 1, 1);
    if (d.nonhZ[k] == 1) {
        const real *w = d.wcoef;
        real a[2], hfU[2], hfdU[2];
        for (int q = 0; q < 2; q++) {
            a[q] = (real)0.5 * (w[ID(i + q, 1)] + w[ID(i - 1 + q, 1)]);
            hfU[q] = d.hu[ID(i - 1 + q, 1)] * (1 + a[q]);
            hfdU[q] = d.hu[ID(i - 1 + q, 1)] * a[q] * (1 - a[q]);
        }
        real dzsdx = (real)0.5 * (d.zsu[k] - d.zsu[kW]);
        real dzbdx = (real)0.5 * (d.zbu[k] + a[1] * d.hu[k] - d.zbu[kW] - a[0] * d.hu[kW]);
        real alpha = -w[k], wc = w[k];
        real au0 = d.au0[k], au1W = d.au1[kW], au0W = d.au0[kW], au1 = d.au1[k];
        real du0 = d.adu0[k], du1W = d.adu1[kW], du0W = d.adu0[kW], du1 = d.adu1[k];
        d.mat[0][k] = 2 * dsz(i, 1) * d.aws[0][k] + hfU[1] * au0 - hfU[0] * au1W + hfdU[1] * du0 - hfdU[0] * du1W
                    - dzsdx * (au0 + au1W - (du0 + du1W) * wc) - dzbdx * (au0 + au1W + (du0 + du1W) * alpha);
        d.mat[1][k] = -hfU[0] * au0W - hfdU[0] * du0W - dzsdx * (au0W - du0W * wc) - dzbdx * (au0W + du0W * alpha);
        d.mat[2][k] = hfU[1] * au1 + hfdU[1] * du1 - dzsdx * (au1 - du1 * wc) - dzbdx * (au1 + du1 * alpha);
        real uu = d.uu[k], uuW = d.uu[kW], dU = d.dU[k], dUW = d.dU[kW];
        d.rhs[k] = -2 * dsz(i, 1) * d.Wm[k] - hfU[1] * uu + hfU[0] * uuW - hfdU[1] * dU + hfdU[0] * dUW
                 + dzsdx * (uu + uuW - (dU + dUW) * wc) + dzbdx * (uu + uuW + (dU + dUW) * alpha);
    } else { d.mat[0][k] = 1; d.mat[1][k] = 0; d.mat[2][k] = 0; d.rhs[k] = 0; }
    d.mat[3][k] = 0; d.mat[4][k] = 0;
}
__global__ void k_q1_after() {   // ws, wb, Wm, Wm0, omega (1D)
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < 2 || i > c.nx) return;
    size_t k = ID(i, 1), kW = ID(i - 1, 1);
    const real *w = d.wcoef;
    real ws = 0, wb = 0, om = 0;
    if (d.wetz[k] == 1) {
        real dzsdx = (real)0.5 * (d.zsu[k] - d.zsu[kW]) / dsz(i, 1);
        ws = -(d.hu[k] * d.uu[k] - d.hu[kW] * d.uu[kW]) / dsz(i, 1) + dzsdx * (d.uu[k] + d.uu[kW] - (d.dU[k] + d.dU[kW]) * w[k]);
        real dzbdx = (real)0.5 * (d.zbu[k] - d.zbu[kW]) / dsz(i, 1);
        wb = dzbdx * (d.uu[k] + d.uu[kW] + (d.dU[k] + d.dU[kW]) * (1 - w[k]));
        om = -(d.hu[k] * d.dU0[k] - d.hu[kW] * d.dU0[kW]) / dsz(i, 1);
    }
    d.ws[k] = ws; d.wb[k] = wb; d.omega[k] = om;
    real Wm = d.nonhZ[k] == 1 ? d.Wm[k] + d.aws[0][k] * d.dp[k] : 0;
    d.Wm[k] = Wm;
    if (c.nhlay > 0) d.Wm0[k] = d.nonhZ[k] == 1 ? Wm + (real)0.5 * (wb - ws) : 0;
}
