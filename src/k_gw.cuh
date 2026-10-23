// k_gw.cuh: groundwater.F90 (gw_init on the host, gw_bc, gwflow) for wavemodel = nonh: hydrostatic (gwnonh = 0)
// and non-hydrostatic (gwnonh = 1) groundwater head, laminar or turbulent Darcy flow, unconnected (swash)
// infiltration, horizontal infiltration across bed steps (gwhorinfil), plus the infiltration effect on bed friction
// (bedroughness.F90 infiltration_boundary_layer_effect). The 2D head equation of gwnonh = 1 is solved with the
// BiCGStab kernels of solver.cuh (XBeach uses SIP); the 1D/ny <= 2 one with the Thomas algorithm, as XBeach.
#pragma once

// gw_calc_local_infil: unsaturated vertical infiltration rate in one cell (implicit in the infiltration depth),
// laminar or turbulent; returns the fraction of the time step that the flow is unsaturated
__device__ void gw_local_infil(real zb, real hh, real gwlevel, real kzlocal, real dinfil, real dt,
                               real &fracdt, real &infil, real &kzupd) {
    const real visc = 1.0e-6, beta = 3.236067977499790, alfa = 2.236067977499790;
    real dtp = dt / c.por, twodtp = 2 * dtp, twodtpdi = twodtp * dinfil, fourdtph = 4 * dtp * hh;
    real dtpsq = dtp * dtp, disq = dinfil * dinfil, kz = c.kz;
    auto vimp = [&](real k) { return ((-dinfil + dtp * k) + sqrt(disq + twodtpdi * k + fourdtph * k + dtpsq * (k * k))) / twodtp; };
    real vest = vimp(kz);
    if (c.gwscheme == 1) {
        real Re = fabs(vest) * c.D50top / c.por / visc;
        if (Re > c.gwReturb) {
            real err = 1, kze = kzlocal;
            bool firstloop = true;
            while (err > (real)0.01) {
                real vupd = vimp(kze);
                if (firstloop) { err = 1; firstloop = false; }
                else err = fabs(vest - vupd) / fabs(vest);
                vest = (vest + alfa * vupd) / beta;
                Re = fabs(vest) * c.D50top / c.por / visc;
                kze = kz * sqrt(c.gwReturb / Re);
            }
            kzupd = kze;
        } else kzupd = kz;
    } else kzupd = kz;
    real dis = vest * dt / c.por;
    fracdt = fmax((real)0, fmin((zb - gwlevel) / dis, (real)1));
    infil = vest * fracdt;
    if (infil > (hh - c.eps) / dt) {
        real newinfil = fmax((hh - c.eps) / dt, (real)0);
        fracdt = fracdt * newinfil / infil;
        infil = newinfil;
    }
}

// gw_calculate_velocities_local: one velocity component (laminar or turbulent; vertical uses the head curvature)
__device__ void gw_vel_local(real &vel, real &Kupd, real headgrad, real Kin, real fracdt, bool isvert, bool isnonh,
                             real gwc, real gwh) {
    auto vvert = [&](real K) { return c.gwheadmodel == 0 ? -K * 2 * gwc * gwh : -K * gwc * sinh(gwh); };
    if (c.gwscheme == 0) {
        if (isvert && isnonh) vel = c.gwheadmodel == 0 ? -Kin * 2 * gwc * gwh * (1 - fracdt) : -Kin * gwc * sinh(gwh) * (1 - fracdt);
        else if (isvert) vel = 0;
        else vel = -Kin * headgrad;
        Kupd = Kin;
        return;
    }
    real vcr = c.gwvcr;
    auto iterate = [&](real vest, bool vert) {
        real err = 1, kest = Kin;
        while (err > (real)0.001) {
            real fac = fmin(sqrt(vcr / fmax(fabs(vest), vcr)), (real)1);
            kest = Kin * fac;
            real vupd = vert ? vvert(kest) : -kest * headgrad;
            vest = (vest + vupd) / 2;
            err = fabs(vupd - vest) / fabs(vest);
        }
        return kest;
    };
    if (isvert && isnonh) {
        real vest = vvert(Kin);
        Kupd = fabs(vest) > vcr ? iterate(vest, true) : Kin;
        vel = isnonh ? vvert(Kin) : vvert(Kupd);
        vel = vel * (1 - fracdt);
    } else if (isvert) {
        vel = 0; Kupd = Kin;
    } else {
        real vest = -Kin * headgrad;
        Kupd = fabs(vest) > vcr ? iterate(vest, false) : Kin;
        vel = isnonh ? -Kin * headgrad : -Kupd * headgrad;
    }
}

// ---- gw_bc (before flow_bc)
__global__ void k_gw_bc1() {   // gwbottom = min(gwbottom, zb)
    IJ
    size_t k = ID(i, j);
    d.gwbottom[k] = fmin(d.gwbottom[k], d.zb[k]);
}
__global__ void k_gw_bc2() {   // offshore and back boundaries (per row j)
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    int nx = c.nx;
    d.gwhead[ID(1, j)] = d.gwhead[ID(2, j)]; d.gwbottom[ID(1, j)] = d.gwbottom[ID(2, j)];
    d.gwhead[ID(nx + 1, j)] = c.gwland ? d.gwhead[ID(nx, j)] : d.gw0back[2 * (j - 1) + 1];
    d.gwbottom[ID(nx + 1, j)] = d.gwbottom[ID(nx, j)];
}
__global__ void k_gw_bc3() {   // lateral boundaries (per column i, ny > 0)
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    int ny = c.ny;
    d.gwbottom[ID(i, 1)] = d.gwbottom[ID(i, 2)]; d.gwhead[ID(i, 1)] = d.gwhead[ID(i, 2)]; d.gwlevel[ID(i, 1)] = d.gwlevel[ID(i, 2)];
    d.gwbottom[ID(i, ny + 1)] = d.gwbottom[ID(i, ny)]; d.gwhead[ID(i, ny + 1)] = d.gwhead[ID(i, ny)];
    d.gwlevel[ID(i, ny + 1)] = d.gwlevel[ID(i, ny)];
}
__global__ void k_gw_bc4() {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    size_t k1 = ID(1, j), k2 = ID(c.nx + 1, j);
    d.gwlevel[k1] = fmin(d.gwhead[k1], d.zb[k1]);
    d.gwlevel[k2] = fmin(d.gwhead[k2], d.zb[k2]);
}

// ---- gwflow, part 1 (per cell): pressure averaging, unconnected infiltration, groundwater level update,
// aquifer thickness, wet-layer ratio and top head
__global__ void k_gw_a(real dt, real factime) {
    IJ
    size_t k = ID(i, j);
    d.gww[k] = 0; d.infil[k] = 0; d.infiluncon[k] = 0; d.infilcon[k] = 0; d.infilhorgw[k] = 0; d.infilhorsw[k] = 0;
    real fracdt = 0;
    real dyn = (1 - factime) * d.dynpres[k] + factime * d.pres[k];
    d.dynpres[k] = dyn;
    real zb = d.zb[k], gwl = d.gwlevel[k];
    bool connected = d.wetz[k] == 1 && zb - gwl < c.connectcrit;
    real iu = 0;
    if (c.gwnonh) {   // gw_unconnected_infil
        real infil = 0, kzupd = d.Kzinf[k];
        if (!connected && d.wetz[k] == 1 && gwl < zb) gw_local_infil(zb, d.hh[k], gwl, d.Kzinf[k], d.dinfil[k], dt, fracdt, infil, kzupd);
        if (infil > 0) d.dinfil[k] = d.dinfil[k] + dt * infil / c.por;
        else { d.dinfil[k] = 0; kzupd = c.kz; }
        fracdt = fmin(fmax(fracdt, (real)0), (real)1);
        d.Kzinf[k] = kzupd;
        iu = infil;
        if (gwl > zb) iu = iu - (gwl - zb) / dt * c.por;
    }
    d.infiluncon[k] = iu;
    d.fracdt[k] = fracdt;
    gwl = gwl + dt * iu / c.por;
    d.gwlevel[k] = gwl;
    real zsupd = d.zs[k] - dt * iu;
    d.gwconn[k] = (zsupd - zb >= c.eps && zb - gwl < c.connectcrit) ? 1 : 0;
    d.gwheight[k] = gwl - d.gwbottom[k];
    real r = fmin(fmax((zb - gwl) / c.dwetlayer, (real)0), (real)1);
    r = 1 - (cos(r * c.px) + 1) / 2;
    d.gwratio[k] = r;
    d.gwheadtop[k] = gwl + (real)d.wetz[k] * (zsupd - gwl + dyn / c.g) * (1 - r);
}
// gw_calculate_interfaceheight (upwind aquifer thickness at u and v points)
__global__ void k_gw_hu(int initial) {
    IJ
    int nx = c.nx, ny = c.ny;
    size_t k = ID(i, j);
    {
        int ii = i == nx + 1 ? nx : i;   // gwhu(nx+1,:) = gwhu(nx,:)
        size_t a = ID(ii, j), b = ID(ii + 1, j);
        real h;
        if (initial) {
            if (d.gwhead[b] > d.gwhead[a]) h = d.gwheight[b];
            else if (d.gwhead[b] < d.gwhead[a]) h = d.gwheight[a];
            else h = (real)0.5 * (d.gwheight[b] + d.gwheight[a]);
        } else {
            if (d.gwu[a] > 0) h = fmin(d.gwheight[a], fmax(d.zb[b] - d.gwbottom[b], (real)0));
            else if (d.gwu[a] < 0) h = fmin(d.gwheight[b], fmax(d.zb[a] - d.gwbottom[a], (real)0));
            else h = (real)0.5 * (d.gwheight[b] + d.gwheight[a]);
        }
        d.gwhu[k] = fmax(h, (real)0);
    }
    if (c.oned) { d.gwhv[k] = d.gwheight[k]; return; }
    {
        int jj = j == ny + 1 ? ny : j;
        size_t a = ID(i, jj), b = ID(i, jj + 1);
        real h;
        if (initial) {
            if (d.gwhead[b] > d.gwhead[a]) h = d.gwheight[b];
            else if (d.gwhead[b] < d.gwhead[a]) h = d.gwheight[a];
            else h = (real)0.5 * (d.gwheight[b] + d.gwheight[a]);
        } else {
            if (d.gwv[a] > 0) h = fmin(d.gwheight[a], fmax(d.zb[b] - d.gwbottom[b], (real)0));
            else if (d.gwv[a] < 0) h = fmin(d.gwheight[b], fmax(d.zb[a] - d.gwbottom[a], (real)0));
            else h = (real)0.5 * (d.gwheight[b] + d.gwheight[a]);
        }
        d.gwhv[k] = fmax(h, (real)0);
    }
}
__global__ void k_gw_head_hyd() {   // hydrostatic head with time averaging
    IJ
    size_t k = ID(i, j);
    d.gwhead[k] = (d.gwhead[k] + d.gwheadtop[k]) / 2;
}

// ---- gw_solver (gwnonh = 1)
// ny < 3: tri-diagonal system on row jr for i = 2..nx (unknown k = i - 1), stored at ID(i, jr) in mat[0..2], rhs
__global__ void k_gw_tri_build(int jr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (i > c.nx) return;
    const int j = jr;
    size_t k = ID(i, j), kW = ID(i - 1, j), kE = ID(i + 1, j);
    const real twothird = (real)2.0 / 3;
    real dxum = dsu(i - 1, j), dxup = dsu(i, j), dxpc = dsz(i, j);
    real hcmx = d.gwheight[kW], hcc = d.gwheight[k], hcpx = d.gwheight[kE];
    real hum = d.gwhu[kW], hup = d.gwhu[k];
    real KxW = d.Kxupd[kW], KxC = d.Kxupd[k], KzC = d.Kzupd[k], fr = d.fracdt[k];
    const real *hbc = d.gwheadtop;
    real A1, A2, A3, b;
    if (c.gwheadmodel == 0) {
        if (d.zb[ID(i, 1)] > d.gwbottom[ID(i, 1)] + c.eps) {   // XBeach tests row 1
            A2 = -KxW * hum / dxum * twothird * (hcmx * hcmx);
            A1 = KxW * hum / dxum * twothird * (hcc * hcc) + KxC * hup / dxup * twothird * (hcc * hcc) + KzC * 2 * dxpc * hcc * (1 - fr);
            A3 = -KxC * hup / dxup * twothird * (hcpx * hcpx);
            b = KxW * hum * (hbc[k] - hbc[kW]) / dxum - KxC * hup * (hbc[kE] - hbc[k]) / dxup;
        } else { A1 = 0; A2 = 0; A3 = 0; b = 0; }
    } else {
        A2 = -KxW * hum / dxum * (cosh(hcmx) - 1 / hcmx * sinh(hcmx));
        A1 = KxW * hum / dxum * (cosh(hcc) - 1 / hcc * sinh(hcc)) + KxC * hup / dxup * (cosh(hcc) - 1 / hcc * sinh(hcc))
             + KzC * dxpc * sinh(hcc) * (1 - fr);
        A3 = -KxC * hup / dxup * (cosh(hcpx) - 1 / hcpx * sinh(hcpx));
        b = KxW * hbc[k] * hum / dxum - KxW * hbc[kW] * hum / dxum - KxC * hbc[kE] * hup / dxup + KxC * hbc[k] * hup / dxup;
    }
    d.mat[0][k] = A1; d.mat[1][k] = A2; d.mat[2][k] = A3; d.rhs[k] = b;
}
// solver_tridiag with fixshallow, one thread (the rows of a dry aquifer give zero pivots, as in XBeach)
__global__ void k_gw_thomas(int jr) {
    if (threadIdx.x || blockIdx.x) return;
#ifdef XB_SINGLE
    const real tny = 1.17549435e-38f;
#else
    const real tny = 2.2250738585072014e-308;   // tiny(0.d0)
#endif
    int n = c.nx - 1;
    auto K = [&](int q) { return ID(q + 1, jr); };   // unknown q = 1..n at i = q + 1
    real fac = d.mat[0][K(1)];
    if (fabs(fac) < tny) fac = copysign(tny, fac);
    d.dp[K(1)] = d.rhs[K(1)] / fac;
    for (int q = 2; q <= n; q++) {
        real cm = d.mat[2][K(q - 1)] / fac;
        d.w1[K(q)] = cm;
        fac = d.mat[0][K(q)] - d.mat[1][K(q)] * cm;
        if (fabs(fac) < tny) fac = copysign(tny, fac);
        d.dp[K(q)] = (d.rhs[K(q)] - d.mat[1][K(q)] * d.dp[K(q - 1)]) / fac;
    }
    for (int q = n - 1; q >= 1; q--) d.dp[K(q)] = d.dp[K(q)] - d.w1[K(q + 1)] * d.dp[K(q + 1)];
}
__global__ void k_gw_tri_after(int jr) {   // gwcurv(2:nx,jr), gwhead row jr and its copies
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    int nx = c.nx, ii = i == 1 ? 2 : (i == nx + 1 ? nx : i);
    size_t k = ID(ii, jr);
    real x = d.dp[k], h = d.gwheight[k], hd;
    if (c.gwheadmodel == 0) hd = d.gwheadtop[k] - (real)2.0 / 3 * x * (h * h);
    else hd = d.gwheadtop[k] + x / h * sinh(h) - x * cosh(h);
    if (i >= 2 && i <= nx) d.gwcurv[ID(i, jr)] = x;
    d.gwhead[ID(i, jr)] = hd;
    if (c.ny == 2) { d.gwhead[ID(i, 1)] = hd; d.gwhead[ID(i, 3)] = hd; }
}
// ny >= 3, parabolic head: 5-point system on the inner cells (SIP in XBeach)
__global__ void k_gw_sip_build() {
    IJ
    if (i < 2 || i > c.nx || j < 2 || j > c.ny) return;
    size_t k = ID(i, j), kW = ID(i - 1, j), kE = ID(i + 1, j), kS = ID(i, j - 1), kN = ID(i, j + 1);
    const real twothird = (real)2.0 / 3;
    real dxum = dsu(i - 1, j), dxup = dsu(i, j), dA = dsdnzi(i, j);   // XBeach: dA = dsdnzi
    real dyum = dnu(i - 1, j), dyup = dnu(i, j), dyvm = dnv(i, j - 1), dyvp = dnv(i, j), dxvm = dsv(i, j - 1), dxvp = dsv(i, j);
    real hcmx = d.gwheight[kW], hcpx = d.gwheight[kE], hcc = d.gwheight[k], hcmy = d.gwheight[kS], hcpy = d.gwheight[kN];
    real hum = d.gwhu[kW], hup = d.gwhu[k], hvm = d.gwhv[kS], hvp = d.gwhv[k];
    real KxW = d.Kxupd[kW], KxC = d.Kxupd[k], KyS = d.Kyupd[kS], KyC = d.Kyupd[k], KzC = d.Kzupd[k];
    real h2 = hcc * hcc;
    d.mat[0][k] = KxW * hum * dyum / dxum * twothird * h2 + KxC * hup * dyup / dxup * twothird * h2
                + KyS * hvm * dxvm / dyvm * twothird * h2 + KyC * hvp * dxvp / dyvp * twothird * h2
                + KzC * 2 * dA * hcc * (1 - d.fracdt[k]);
    d.mat[1][k] = -KxW * hum * dyum / dxum * twothird * (hcmx * hcmx);
    d.mat[2][k] = -KxC * hup * dyup / dxup * twothird * (hcpx * hcpx);
    d.mat[3][k] = -KyS * hvm * dxvm / dyvm * twothird * (hcmy * hcmy);
    d.mat[4][k] = -KyC * hvp * dxvp / dyvp * twothird * (hcpy * hcpy);
    const real *hbc = d.gwheadtop;
    d.rhs[k] = KxW * dyum * hum * (hbc[k] - hbc[kW]) / dxum - KxC * dyup * hup * (hbc[kE] - hbc[k]) / dxup
             + KyS * dxvm * hvm * (hbc[k] - hbc[kS]) / dyvm - KyC * dxvp * hvp * (hbc[kN] - hbc[k]) / dyvp;
    // a cell without aquifer (gwheight = 0) has a zero diagonal; its curvature does not enter any other row (their
    // coefficients carry its thickness squared) nor its own head, so the row is decoupled (x = 0)
    if (!(fabs(d.mat[0][k]) > (real)1e-15)) {
        d.mat[0][k] = 1; d.mat[1][k] = 0; d.mat[2][k] = 0; d.mat[3][k] = 0; d.mat[4][k] = 0; d.rhs[k] = 0;
    }
}
__global__ void k_gw_sip_after(int solved) {   // gwcurv = x with boundary copies; head from the curvature
    IJ
    int nx = c.nx, ny = c.ny;
    if (solved) {
        int ii = i == 1 ? 2 : (i == nx + 1 ? nx : i), jj = j == 1 ? 2 : (j == ny + 1 ? ny : j);
        d.gwcurv[ID(i, j)] = d.dp[ID(ii, jj)];
    }
}
__global__ void k_gw_head2d() {
    IJ
    size_t k = ID(i, j);
    real x = d.gwcurv[k], h = d.gwheight[k];
    if (c.gwheadmodel == 0) d.gwhead[k] = d.gwheadtop[k] - (real)2.0 / 3 * x * (h * h);
    else d.gwhead[k] = d.gwheadtop[k] + x / h * sinh(h) - x * cosh(h);
}

// ---- gw_calculate_velocities: head gradients, velocities and updated (turbulent) conductivities
__global__ void k_gw_vel() {
    IJ
    int nx = c.nx, ny = c.ny;
    size_t k = ID(i, j);
    real dhx = 0, dhy = 0;
    {
        int ii = i == 1 ? 2 : i;   // dheaddx(1,:) = dheaddx(2,:)
        if (ii <= nx) dhx = (d.gwhead[ID(ii + 1, j)] - d.gwhead[ID(ii, j)]) / dsu(ii, j);
    }
    if (!c.oned) {
        int jj = j == 1 ? 2 : j;
        if (jj <= ny) dhy = (d.gwhead[ID(i, jj + 1)] - d.gwhead[ID(i, jj)]) / dnv(i, jj);
    }
    real gc = d.gwcurv[k], gh = d.gwheight[k], vel, ku;
    gw_vel_local(vel, ku, dhx, c.kx, 0, false, true, gc, gh); d.gwu[k] = vel; d.KxN[k] = ku;
    gw_vel_local(vel, ku, dhy, c.ky, 0, false, true, gc, gh); d.gwv[k] = vel; d.KyN[k] = ku;
    gw_vel_local(vel, ku, 0, c.kz, d.fracdt[k], true, true, gc, gh); d.gww[k] = vel; d.KzN[k] = ku;
}
// smoothing of the conductivities over 7 cells in x (next time step and output), Kzinf output
__global__ void k_gw_ksmooth() {
    IJ
    size_t k = ID(i, j);
    int i0 = max(1, i - 3), i1 = min(i + 3, c.NX1);
    real incl = (real)(i1 - i0 + 1), sx = 0, sy = 0, sz = 0;
    for (int q = i0; q <= i1; q++) { size_t kq = ID(q, j); sx += d.KxN[kq]; sy += d.KyN[kq]; sz += d.KzN[kq]; }
    d.Kxupd[k] = sx / incl; d.Kyupd[k] = sy / incl; d.Kzupd[k] = sz / incl;
    d.gKx[k] = d.Kxupd[k]; d.gKy[k] = d.Kyupd[k]; d.gKz[k] = d.Kzupd[k]; d.gKzinf[k] = d.Kzinf[k];
}
// horizontal fluxes; cells at the aquifer bottom only take water in
__global__ void k_gw_q() {
    IJ
    int nx = c.nx, ny = c.ny;
    size_t k = ID(i, j);
    auto dry = [&](int a, int b) { size_t q = ID(a, b); return d.gwlevel[q] <= d.gwbottom[q] + c.eps; };
    real qx = d.gwu[k] * d.gwhu[k], qy = d.gwv[k] * d.gwhv[k];
    if (!c.oned) {
        bool jin = j >= 2 && j <= ny;
        if (jin && i >= 2 && i <= nx && dry(i, j)) qx = fmin(qx, (real)0);
        if (jin && i + 1 >= 2 && i + 1 <= nx && dry(i + 1, j)) qx = fmax(qx, (real)0);
        bool iin = i >= 2 && i <= nx;
        if (iin && j >= 2 && j <= ny && dry(i, j)) qy = fmin(qy, (real)0);
        if (iin && j + 1 >= 2 && j + 1 <= ny && dry(i, j + 1)) qy = fmax(qy, (real)0);
    } else {
        if (i >= 2 && i <= nx && dry(i, 1)) qx = fmin(qx, (real)0);
        if (i + 1 >= 2 && i + 1 <= nx && dry(i + 1, 1)) qx = fmax(qx, (real)0);
    }
    d.gwqx[k] = qx; d.gwqy[k] = qy;
}
// continuity: exchange velocity gww of the inner cells from the flux divergence
__global__ void k_gw_w() {
    IJ
    int nx = c.nx, ny = c.ny;
    if (i < 2 || i > nx) return;
    size_t k = ID(i, j);
    if (ny > 2) {
        if (j < 2 || j > ny) return;
        d.gww[k] = (-d.gwqx[k] * dnu(i, j) + d.gwqx[ID(i - 1, j)] * dnu(i - 1, j) - d.gwqy[k] * dsv(i, j)
                    + d.gwqy[ID(i, j - 1)] * dsv(i, j - 1)) * dsdnzi(i, j);
    } else if (ny > 0) {
        if (j > 3) return;
        d.gww[k] = (-d.gwqx[ID(i, 2)] + d.gwqx[ID(i - 1, 2)]) / dsz(i, 2);
    } else d.gww[k] = (-d.gwqx[k] + d.gwqx[ID(i - 1, 1)]) / dsz(i, 1);
}
// exchange with the surface water in connected cells (gwnonh = 1)
__global__ void k_gw_infilcon(real dt) {
    IJ
    int nx = c.nx, ny = c.ny;
    if (i < 2 || i > nx || j < min(2, ny + 1) || j > max(ny, 1)) return;
    size_t k = ID(i, j);
    real w = d.gww[k], ic = 0;
    if (w > 0) {
        if (d.gwconn[k]) {
            real dzr = fmax(d.zb[k] - d.gwlevel[k], (real)0), wdt = w * dt / c.por;
            if (wdt >= dzr) ic = -w * (1 - dzr / wdt);
        }
    } else if (w < 0) {
        if (d.gwconn[k]) {
            real dzr = fmax(d.zs[k] - d.zb[k] - c.eps, (real)0), wdt = -w * dt;
            ic = wdt >= dzr ? -w * (dzr / wdt) : -w;
        }
    }
    d.infilcon[k] = ic;
}
// hydrostatic exchange (gwnonh = 0): infiltration depth, then gw_calculate_hydrostatic_w on the inner cells
__global__ void k_gw_hyd1(real dt) {
    IJ
    int nx = c.nx, ny = c.ny;
    size_t k = ID(i, j);
    real di;
    if (d.wetz[k] == 0) di = c.dwetlayer / 3;
    else { di = fmin(d.dinfil[k], d.zb[k] - d.gwlevel[k]); di = fmax(di, c.dwetlayer / 3); }
    d.dinfil[k] = di;
    bool in = i >= 2 && i <= nx && (c.oned ? j == 1 : (j >= 2 && j <= ny));
    if (!in) return;
    real zb = d.zb[k], gwl = d.gwlevel[k], hh = d.hh[k], inf;
    if (gwl > zb) inf = c.por * (zb - gwl) / dt;
    else if (d.wetz[k] == 1) {
        real dum, w1, dum2;
        gw_local_infil(zb, hh, gwl, c.kz, di, dt, dum, w1, dum2);
        w1 = fmin(w1, (1 + hh / (c.dwetlayer / 3)) * c.kz);
        real w2 = c.por * (zb - gwl) / dt;
        w2 = fmin(w2, (hh - c.eps) / dt);
        real r = d.gwratio[k];
        real w3 = r * w1 + (1 - r) * w2;
        inf = fmin(w3, (zb - gwl) * c.por / dt);
    } else inf = 0;
    d.infilcon[k] = inf;
}
__global__ void k_gw_hyd_bci() {   // infil(1,:) = infil(2,:), infil(nx+1,:) = infil(nx,:)
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    d.infilcon[ID(1, j)] = d.infilcon[ID(2, j)];
    d.infilcon[ID(c.nx + 1, j)] = d.infilcon[ID(c.nx, j)];
}
__global__ void k_gw_hyd_bcj() {   // infil(:,1) = infil(:,2), infil(:,ny+1) = infil(:,ny)
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    d.infilcon[ID(i, 1)] = d.infilcon[ID(i, 2)];
    d.infilcon[ID(i, c.ny + 1)] = d.infilcon[ID(i, c.ny)];
}
__global__ void k_gw_hyd2(real dt) {   // limit to the available water; infiltration depth
    IJ
    size_t k = ID(i, j);
    real ic = d.infilcon[k];
    if (ic * dt > d.hh[k] - c.eps) ic = (d.hh[k] - c.eps) / dt;
    d.infilcon[k] = ic;
    d.dinfil[k] = d.dinfil[k] + ic * dt / c.por;
}
// gw_horizontal_infil_exfil: flow through the vertical faces of bed steps between surface and ground water
__device__ void gw_hor_face(int ia, int ja, int ib, int jb, bool xface, real dt, real &vel, real &sab, real &sas,
                            bool &tob) {
    // face between cells a = (ia,ja) and b = (ib,jb), b the next cell in +x or +y; tob: ground water in b
    size_t ka = ID(ia, ja), kb = ID(ib, jb);
    real dz = d.zb[kb] - d.zb[ka];
    vel = 0; sab = 0; sas = 0; tob = true;
    auto dl = [&](int a, int b) { return xface ? dsz(a, b) : dnz(a, b); };
    real Kd, dum2 = 0, dum3 = 0;   // dummies: gw_vel_local reads them only for vertical non-hydrostatic flow
    if (dz > 0 && d.wetz[ka] == 1) {
        real hsurf = d.zs[ka] + d.dynpres[ka] / c.g, hgw = d.gwhead[kb], dx = dl(ib, jb);
        gw_vel_local(vel, Kd, (hsurf - hgw) / dx, c.kx, 0, false, false, dum2, dum3);
        vel = -vel;
        sab = dz / dl(ib, jb); sas = dz / dl(ia, ja);
        if (vel > 0) vel = fmin(fmin(vel, d.hh[ka] * dsz(ia, ja) / dt / dz),
                                fmax(d.zb[kb] - d.gwlevel[kb], (real)0) * c.por * dl(ib, jb) / dt / dz);
        else vel = fmax(vel, -(d.gwlevel[kb] - d.gwbottom[kb]) * dl(ib, jb) * c.por / dt / dz);
        tob = true;
    } else if (dz < 0 && d.wetz[kb] == 1) {
        dz = -dz;
        real hsurf = d.zs[kb] + d.dynpres[kb] / c.g, hgw = d.gwhead[ka], dx = dl(ia, ja);
        gw_vel_local(vel, Kd, (hsurf - hgw) / dx, c.kx, 0, false, false, dum2, dum3);
        vel = -vel;
        sab = dz / dl(ia, ja); sas = dz / dl(ib, jb);
        if (vel > 0) vel = fmin(fmin(vel, d.hh[kb] * dl(ib, jb) / dt / dz),
                                fmax(d.zb[ka] - d.gwlevel[ka], (real)0) * c.por * dl(ia, ja) / dt / dz);
        else vel = fmax(vel, -(d.gwlevel[ka] - d.gwbottom[ka]) * dl(ia, ja) * c.por / dt / dz);
        tob = false;
    } else { vel = 0; tob = false; sab = 0; sas = 0; }
}
__global__ void k_gw_hor(real dt) {
    IJ
    int nx = c.nx, ny = c.ny;
    size_t k = ID(i, j);
    real gw = 0, sw = 0, vel, sab, sas; bool tob;
    // u-face (i-1,j): this cell is the +x neighbour
    if (i >= 2) { gw_hor_face(i - 1, j, i, j, true, dt, vel, sab, sas, tob); if (tob) gw += vel * sab; else sw += vel * sas; }
    // u-face (i,j): this cell is the -x neighbour
    if (i <= nx) { gw_hor_face(i, j, i + 1, j, true, dt, vel, sab, sas, tob); if (!tob) gw += vel * sab; else sw += vel * sas; }
    if (ny > 0) {
        if (j >= 2) { gw_hor_face(i, j - 1, i, j, false, dt, vel, sab, sas, tob); if (tob) gw += vel * sab; else sw += vel * sas; }
        if (j <= ny) { gw_hor_face(i, j, i, j + 1, false, dt, vel, sab, sas, tob); if (!tob) gw += vel * sab; else sw += vel * sas; }
    }
    d.infilhorgw[k] = gw; d.infilhorsw[k] = sw;
}
// groundwater level and total exchange
__global__ void k_gw_level(real dt) {
    IJ
    size_t k = ID(i, j);
    d.gwlevel[k] = d.gwlevel[k] + (d.gww[k] + d.infilcon[k] + d.infilhorgw[k]) * dt / c.por;
    d.infil[k] = d.infiluncon[k] + d.infilcon[k] + d.infilhorsw[k];
}
__global__ void k_gw_bnd_i() {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    real *a[5] = {d.gwlevel, d.gwhead, d.gwcurv, d.gwbottom, d.gww};
    for (auto p : a) { p[ID(1, j)] = p[ID(2, j)]; p[ID(c.nx + 1, j)] = p[ID(c.nx, j)]; }
}
__global__ void k_gw_bnd_j() {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    real *a[5] = {d.gwlevel, d.gwhead, d.gwcurv, d.gwbottom, d.gww};
    for (auto p : a) { p[ID(i, 1)] = p[ID(i, 2)]; p[ID(i, c.ny + 1)] = p[ID(i, c.ny)]; }
}
__global__ void k_gw_headb() {   // gwCalculateHeadBottom, with gwheadb(1,:) = gwheadb(2,:), gwheadb(nx+1,:) = gwheadb(nx,:)
    IJ
    int ii = i == 1 ? 2 : (i == c.nx + 1 ? c.nx : i);
    size_t q = ID(ii, j);
    real x = d.gwcurv[q], h = d.gwheight[q];
    d.gwheadb[ID(i, j)] = c.gwheadmodel == 0 ? d.gwheadtop[q] - x * (h * h) : x + d.gwheadtop[q] - x * cosh(h);
}

// bedroughness_update, friction_infiltration = 1: enhanced bed shear stress from infiltration (Conley and Inman 1994)
__device__ __forceinline__ real gw_facbl(real infilb, real Ubed, real cf) {
    real vent = -infilb / fmax(Ubed, (real)1e-6);
    vent = fmax(fmin(vent, (real)1), (real)-1);
    real blphi = (real)0.9 / 2 / cf * vent, facbl;
    if (infilb >= 0) { blphi = fmin(blphi, (real)-1e-4); facbl = blphi / (exp(blphi) - 1); facbl = fmin(facbl, (real)3); }
    else { blphi = fmax(blphi, (real)1e-4); facbl = blphi / (exp(blphi) - 1); facbl = fmax(facbl, (real)0.1); }
    return fmin(facbl, 1 / cf);
}
__device__ void gw_friction_infil(int i, int j, real cu, real cv) {
    int nx = c.nx, ny = c.ny;
    size_t k = ID(i, j);
    real tx = 0, ty = 0;
    if (d.wetu[k] == 1) {
        int ii = i == nx + 1 ? nx : i;
        real ib = d.wetu[ID(ii, j)] == 1 ? (real)0.5 * (d.infil[ID(ii, j)] + d.infil[ID(ii + 1, j)]) : 0;
        real f = gw_facbl(ib, fabs(d.vmageu[k]), cu);
        tx = (f - 1) * cu * c.rho * d.ueu[k] * d.vmageu[k];
    }
    if (d.wetv[k] == 1) {
        real ib;
        if (c.oned) ib = d.infil[k];
        else if (j == ny + 1) ib = d.infil[ID(i, ny)];
        else ib = (real)0.5 * (d.infil[k] + d.infil[ID(i, j + 1)]);
        real f = gw_facbl(ib, fabs(d.vmagev[k]), cv);
        ty = (f - 1) * cv * c.rho * d.vev[k] * d.vmagev[k];
    }
    d.taubx_add[k] = d.taubx_add[k] + tx; d.tauby_add[k] = d.tauby_add[k] + ty;
}
