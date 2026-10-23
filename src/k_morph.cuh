// k_morph.cuh: morphevolution.F90 bed_update (struct limiter, bed level change with one or more fractions and bed
// layers, water level and groundwater update, avalanching) and setbathy_update.
// Avalanching updates the bed in place in XBeach's loop order; the x sweep has no coupling between rows and the y sweep
// none between columns, so each row (column) runs in one thread, in order, and all rows (columns) in parallel.
#pragma once

// update_fractions: bed layer bookkeeping of one cell for the exchange edg (per fraction) and bed change dzb
__device__ void update_fractions(size_t k, const real *edg, real dzb, real dt) {
    const int nd = cs.nd, ngd = cs.ngd, nv = cs.nd_var - 1;   // nv: 0-based index of the variable layer
    real dz[NDMAX], pb[NDMAX][NGMAX], Sm[NDMAX][NGMAX];
    for (int m = 0; m < nd; m++) { dz[m] = sd.dzbed[(size_t)m * NN + k]; for (int g = 0; g < ngd; g++) pb[m][g] = PB(k, m, g); }
    real dzb_loc = dzb, ED = 0;
    for (int g = 0; g < ngd; g++) ED += edg[g];
    real z0b = sd.z0bed[k];
    while (fabs(dzb_loc) > 0) {
        real dzbt = fmin(dzb_loc, dz[nv]);
        dzbt = fmax(dzbt, -cs.frac_dz * dz[nv + 1]);
        real fac = dzbt / dzb_loc;
        dzb_loc = dzb_loc - dzbt;
        for (int g = 0; g < ngd; g++) {
            real A[NDMAX][3], Ap[NDMAX], b[NDMAX];
            for (int m = 0; m < nd; m++) { A[m][0] = A[m][1] = A[m][2] = 0; Ap[m] = 0; b[m] = 0; }
            for (int m = 0; m < nd; m++) Sm[m][g] = pb[m][g] * dz[m] * (1 - cs.por);
            if (cs.nd_var == 2) {
                A[0][0] = 0; A[0][1] = fmin(ED, (real)0); A[0][2] = fmax(ED, (real)0);
                A[1][0] = -fmin(ED, (real)0); A[1][1] = -fmax(ED, (real)0); A[1][2] = 0;
            } else if (cs.nd_var >= 3) {
                A[0][0] = 0; A[0][1] = fmin(ED, (real)0); A[0][2] = fmax(ED, (real)0);
                for (int m = 1; m < nv; m++) { A[m][0] = -fmin(ED, (real)0); A[m][1] = -fabs(ED); A[m][2] = fmax(ED, (real)0); }
                A[nv][0] = -fmin(ED, (real)0); A[nv][1] = -fmax(ED, (real)0); A[nv][2] = 0;
            }
            Ap[0] = A[0][1] * pb[0][g] + A[0][2] * pb[1][g];
            for (int m = 1; m <= nv; m++) Ap[m] = A[m][0] * pb[m - 1][g] + A[m][1] * pb[m][g] + A[m][2] * pb[m + 1][g];
            Ap[nv + 1] = A[nv + 1][0] * pb[nv][g] + A[nv + 1][1] * pb[nv + 1][g];
            b[0] = -edg[g];
            for (int m = 0; m <= nv + 1; m++) Sm[m][g] = Sm[m][g] + dt * fac * (Ap[m] + b[m]);
        }
        for (int m = 0; m <= nv + 1; m++) {
            real s = 0; for (int g = 0; g < ngd; g++) s += Sm[m][g];
            if (s > 0) for (int g = 0; g < ngd; g++) pb[m][g] = Sm[m][g] / s;
            dz[m] = s / (1 - cs.por);
        }
        if (dz[nv] < cs.merge * dz[nv + 1]) {
            for (int g = 0; g < ngd; g++) {
                pb[nv][g] = (dz[nv] * pb[nv][g] + dz[nv + 1] * pb[nv + 1][g]) / (dz[nv] + dz[nv + 1]);
                for (int m = nv + 1; m <= nd - 2; m++) pb[m][g] = pb[m + 1][g];
            }
            z0b = z0b - dz[nv + 1];
            dz[nv] = dz[nv + 1] + dz[nv];
        }
        if (dz[nv] > cs.split * dz[nv + 1]) {
            for (int g = 0; g < ngd; g++) for (int m = nd - 1; m >= nv + 1; m--) pb[m][g] = pb[m - 1][g];
            z0b = z0b + dz[nv + 1];
            dz[nv] = dz[nv] - dz[nv + 1];
        }
    }
    real sz = 0;
    for (int m = 0; m < nd; m++) {
        sd.dzbed[(size_t)m * NN + k] = dz[m]; sz += dz[m];
        for (int g = 0; g < ngd; g++) PB(k, m, g) = fmax((real)0, fmin(pb[m][g], (real)1));
    }
    sd.z0bed[k] = z0b;
    real zbold = d.zb[k];
    d.zb[k] = z0b + sz;
    real dzz = d.zb[k] - zbold;
    sd.dzbnow[k] = sd.dzbnow[k] + dzz;
    sd.sedero[k] = sd.sedero[k] + dzz;
    sd.structdepth[k] = fmax((real)0, sd.structdepth[k] + dzz);
}

__global__ void k_bed_reset() { IJ sd.dzbnow[ID(i, j)] = 0; }

// struct = 1: limit the outgoing fluxes of cells with a thin erodible layer (one fraction at a time)
__global__ void k_bed_sout(int jg, real dt, real *facc) {
    IJ
    int j1 = c.oned ? 1 : 2;
    size_t k = ID(i, j);
    if (i < 2 || j < j1) { facc[k] = 1; return; }
    const real *Sub = sd.Subg + (size_t)jg * NN, *Sus = sd.Susg + (size_t)jg * NN;
    const real *Svb = sd.Svbg + (size_t)jg * NN, *Svs = sd.Svsg + (size_t)jg * NN;
    real Sout = 0;
    size_t kw = ID(i - 1, j);
    if (Sub[k] > 0) Sout = Sout + Sub[k] * dnu(i, j);
    if (Sub[kw] < 0) Sout = Sout - Sub[kw] * dnu(i - 1, j);
    if (cs.sourcesink == 0) {
        if (Sus[k] > 0) Sout = Sout + Sus[k] * dnu(i, j);
        if (Sus[kw] < 0) Sout = Sout - Sus[kw] * dnu(i - 1, j);
    }
    if (!c.oned) {
        size_t ks = ID(i, j - 1);
        if (Svb[k] > 0) Sout = Sout + Svb[k] * dsv(i, j);
        if (Svb[ks] < 0) Sout = Sout - Svb[ks] * dsv(i, j - 1);
        if (cs.sourcesink == 0) {
            if (Svs[k] > 0) Sout = Sout + Svs[k] * dsv(i, j);
            if (Svs[ks] < 0) Sout = Sout - Svs[ks] * dsv(i, j - 1);
        }
    }
    real Sav = sd.structdepth[k] * PB(k, 0, jg) / cs.morfac / dt * (1 - cs.por) / dsdnzi(i, j);
    facc[k] = fmin((real)1, Sav / fmax(Sout, NUMEPS));
}
__device__ __forceinline__ void struct_face(real *S, size_t kf, bool inA, real facA, bool inB, real facB) {
    // face between cells A (upstream index) and B; ind from the original flux, as XBeach indSub/indSvb
    real s0 = S[kf], s = s0;
    real ind = (inA && s0 > 0) ? 1 : 0;
    if (inA && facA < 1) s = facA * ind * s + (1 - ind) * s;
    if (inB && facB < 1) s = facB * (1 - ind) * s + ind * s;
    S[kf] = s;
}
__global__ void k_bed_slimit(int jg, const real *facc) {
    IJ
    int j1 = c.oned ? 1 : 2, nx = c.nx, ny = c.ny;
    size_t k = ID(i, j);
    auto inc = [&](int a, int b) { return a >= 2 && a <= nx + 1 && b >= j1 && b <= ny + 1; };
    {   // u face (i,j) between cells (i,j) and (i+1,j); face nx+1 has no cell beyond
        bool a = inc(i, j), b = i <= nx && inc(i + 1, j);
        real fa = a ? facc[k] : 1, fb = b ? facc[ID(i + 1, j)] : 1;
        struct_face(sd.Subg + (size_t)jg * NN, k, a, fa, b, fb);
        if (cs.sourcesink == 0) struct_face(sd.Susg + (size_t)jg * NN, k, a, fa, b, fb);
    }
    if (!c.oned) {   // v face (i,j) between cells (i,j) and (i,j+1)
        bool a = inc(i, j), b = j <= ny && inc(i, j + 1);
        real fa = a ? facc[k] : 1, fb = b ? facc[ID(i, j + 1)] : 1;
        struct_face(sd.Svbg + (size_t)jg * NN, k, a, fa, b, fb);
        if (cs.sourcesink == 0) struct_face(sd.Svsg + (size_t)jg * NN, k, a, fa, b, fb);
    }
}

// bed level change of the inner cells from the transport gradients (or source/sink)
__global__ void k_bed_level(real dt) {
    IJ
    if (!IN_ZS) return;
    size_t k = ID(i, j);
    real dzg[NGMAX], sdz = 0;
    for (int g = 0; g < cs.ngd; g++) {
        const real *Sus = sd.Susg + (size_t)g * NN, *Svs = sd.Svsg + (size_t)g * NN, *Sub = sd.Subg + (size_t)g * NN, *Svb = sd.Svbg + (size_t)g * NN;
        size_t kw = ID(i - 1, j);
        real f = cs.morfac * dt / (1 - cs.por), q;
        if (!c.oned) {
            size_t ks = ID(i, j - 1);
            real ub = Sub[k] * dnu(i, j) - Sub[kw] * dnu(i - 1, j) + Svb[k] * dsv(i, j) - Svb[ks] * dsv(i, j - 1);
            if (cs.sourcesink == 0)
                q = (Sus[k] * dnu(i, j) - Sus[kw] * dnu(i - 1, j) + Svs[k] * dsv(i, j) - Svs[ks] * dsv(i, j - 1) + Sub[k] * dnu(i, j)
                     - Sub[kw] * dnu(i - 1, j) + Svb[k] * dsv(i, j) - Svb[ks] * dsv(i, j - 1)) * dsdnzi(i, j);
            else q = G3(sd.ero, k, g) - G3(sd.depo_ex, k, g) + ub * dsdnzi(i, j);
        } else {
            if (cs.sourcesink == 0)
                q = (Sus[k] * dnu(i, j) - Sus[kw] * dnu(i - 1, j) + Sub[k] * dnu(i, j) - Sub[kw] * dnu(i - 1, j)) * dsdnzi(i, j)
                    + (Svs[k] + Svb[k]) * cs.lsgrad;
            else q = G3(sd.ero, k, g) - G3(sd.depo_ex, k, g) + (Sub[k] * dnu(i, j) - Sub[kw] * dnu(i - 1, j)) * dsdnzi(i, j)
                     + (Svs[k] + Svb[k]) * cs.lsgrad;
        }
        dzg[g] = f * q;
        sdz += dzg[g];
    }
    if (cs.ngd == 1) {
        d.zb[k] = d.zb[k] - sdz;
        sd.dzbnow[k] = sd.dzbnow[k] - sdz;
        sd.dzbdt[k] = sd.dzbnow[k] / dt;
        sd.sedero[k] = sd.sedero[k] - sdz;
        sd.structdepth[k] = fmax((real)0, sd.structdepth[k] - sdz);
    } else {
        real edg[NGMAX];
        for (int g = 0; g < cs.ngd; g++) edg[g] = dzg[g] * (1 - cs.por) / dt;
        update_fractions(k, edg, sdz, dt);
    }
}
// water level follows the bed in wet cells (groundwater: exchange with the aquifer)
__global__ void k_bed_zs(real dt, int gw) {
    IJ
    size_t k = ID(i, j);
    real dzn = sd.dzbnow[k];
    if (gw) {
        if (d.wetz[k] == 1 && dzn > 0) {
            d.zs[k] = d.zs[k] + dzn * (1 - c.por);
            d.gwlevel[k] = d.gwlevel[k] + dzn;
            d.infil[k] = d.infil[k] + dzn * c.por / dt;
        } else if (d.wetz[k] == 1 && dzn < 0) {
            real te = fmin((d.zb[k] - d.gwlevel[k]) * c.por, (real)0);
            d.zs[k] = d.zs[k] + dzn - te;
            d.gwlevel[k] = d.gwlevel[k] + te / c.por;
            d.infil[k] = d.infil[k] + te;
        }
    } else if (IN_ZS && d.wetz[k] == 1) d.zs[k] = d.zs[k] + dzn;
}
// bed slopes (start of each avalanching iteration, or after the bed update without avalanching)
__global__ void k_bed_slopes(int aval) {
    IJ
    size_t k = ID(i, j);
    int nx = c.nx, ny = c.ny;
    if (i <= nx) sd.dzbdx[k] = (d.zb[ID(i + 1, j)] - d.zb[k]) / dsu(i, j);
    else sd.dzbdx[k] = aval ? (d.zb[ID(nx + 1, j)] - d.zb[ID(nx, j)]) / dsu(nx, j) : 0;
    if (!c.oned) {
        if (j <= ny) sd.dzbdy[k] = (d.zb[ID(i, j + 1)] - d.zb[k]) / dnv(i, j);
        else sd.dzbdy[k] = aval ? (d.zb[ID(i, ny + 1)] - d.zb[ID(i, ny)]) / dnv(i, ny) : 0;
    } else if (!aval) sd.dzbdy[k] = 0;
}
// avalanching: one thread per row j (x faces i = 2..nx in order), then one per column i (y faces j = 2..ny)
__device__ void aval_move(size_t ke, size_t kd, real dzb, real dAfac, real dt) {
    if (cs.ngd == 1) {
        real dzleft = fabs(dzb);
        d.zb[kd] = d.zb[kd] + dzleft * dAfac; d.zb[ke] = d.zb[ke] - dzleft;
        sd.dzbnow[kd] = sd.dzbnow[kd] + dzleft * dAfac; sd.dzbnow[ke] = sd.dzbnow[ke] - dzleft;
        sd.sedero[kd] = sd.sedero[kd] + dzleft * dAfac; sd.sedero[ke] = sd.sedero[ke] - dzleft;
        sd.structdepth[kd] = fmax((real)0, sd.structdepth[kd] + dzleft * dAfac);
        sd.structdepth[ke] = fmax((real)0, sd.structdepth[ke] - dzleft);
        d.zs[kd] = d.zs[kd] + dzleft * dAfac; d.zs[ke] = d.zs[ke] - dzleft;
        sd.dzav[kd] = sd.dzav[kd] + dzleft * dAfac; sd.dzav[ke] = sd.dzav[ke] - dzleft;
    } else {
        real sdz = 0; int ndz = 0;
        while (sdz < fabs(dzb)) { sdz = sdz + sd.dzbed[(size_t)ndz * NN + ke]; ndz++; }
        real dzleft = fabs(dzb), dzavt = 0;
        for (int jdz = 0; jdz < ndz; jdz++) {
            real dzt = fmin(sd.dzbed[(size_t)jdz * NN + ke], dzleft);
            dzleft = dzleft - dzt;
            real edg2[NGMAX], edg1[NGMAX], s2 = 0;
            for (int g = 0; g < cs.ngd; g++) {
                edg2[g] = cs.sedcal[g] * dzt * PB(ke, jdz, g) * (1 - cs.por) / dt;
                edg1[g] = -cs.sedcal[g] * edg2[g] * dAfac;
                s2 += edg2[g];
            }
            dzavt = dzavt + s2 * dt / (1 - cs.por);
            update_fractions(ke, edg2, dzavt, dt);
            update_fractions(kd, edg1, -dzavt * dAfac, dt);
        }
        d.zs[ke] = d.zs[ke] - dzavt; sd.dzav[ke] = sd.dzav[ke] - dzavt;
        d.zs[kd] = d.zs[kd] + dzavt * dAfac; sd.dzav[kd] = sd.dzav[kd] + dzavt * dAfac;
    }
}
__global__ void k_aval_x(real dt, real avaltime, real Sig) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    int nx = c.nx, ny = c.ny, n = 0;
    for (int i = 2; i <= nx; i++) {
        size_t k = ID(i, j);
        int ip1 = min(i + 1, nx + 1), jm1 = max(j - 1, 1);
        real dzy = c.oned ? 0 : (real)0.25 * (sd.dzbdy[k] + sd.dzbdy[ID(ip1, j)] + sd.dzbdy[ID(i, jm1)] + sd.dzbdy[ID(ip1, jm1)]);
        real dzx = sd.dzbdx[k];
        real totslp = sqrt(dzx * dzx + dzy * dzy);
        real dzmax;
        if (fmax(d.hh[k], d.hh[ID(i + 1, j)]) > cs.hswitch + c.eps) dzmax = cs.wetslp * fabs(dzx) / fmax(totslp, (real)2.2250738585072014e-308);
        else dzmax = cs.dryslp * fabs(dzx) / fmax(totslp, (real)2.2250738585072014e-308);
        // istruct = nx + 1 without swrunup: never i > istruct
        int iq = i + (int)lrint(fmax((real)0, copysign((real)1, dzx)));
        if (fabs(dzx) > dzmax + (real)1e-10 && sd.structdepth[ID(iq, j)] > c.eps) {
            n++;
            real dzb = copysign((real)1, dzx) * (fabs(dzx) - dzmax) * dsu(i, j);
            real s2 = Sig / dsu(i, j);
            dzb = dzb * dt / avaltime * fmax(s2 * s2, (real)1);
            size_t ke, kd; real dAfac;
            if (dzb >= 0) { ke = ID(i + 1, j); kd = k; dAfac = dsdnzi(i, j) / dsdnzi(i + 1, j); dzb = fmin(dzb, sd.structdepth[ID(i + 1, j)]); }
            else { ke = k; kd = ID(i + 1, j); dAfac = dsdnzi(i + 1, j) / dsdnzi(i, j); dzb = fmax(dzb, -sd.structdepth[k]); }
            aval_move(ke, kd, dzb, dAfac, dt);
        }
    }
    if (n) atomicAdd(sd.naval, n);
    (void)ny;
}
__global__ void k_aval_y(real dt, real avaltime, real Sig) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    int ny = c.ny, n = 0;
    for (int j = 2; j <= ny; j++) {
        size_t k = ID(i, j);
        int jp1 = min(j + 1, ny + 1), im1 = max(i - 1, 1);
        real dzy = sd.dzbdy[k];
        real dzx = (real)0.25 * (sd.dzbdx[k] + sd.dzbdx[ID(i, jp1)] + sd.dzbdx[ID(im1, j)] + sd.dzbdx[ID(im1, jp1)]);
        real totslp = sqrt(dzy * dzy + dzx * dzx);
        real dzmax;
        if (fmax(d.hh[k], d.hh[ID(i, j + 1)]) > cs.hswitch + c.eps) dzmax = cs.wetslp * fabs(dzy) / fmax(totslp, (real)2.2250738585072014e-308);
        else dzmax = cs.dryslp * fabs(dzy) / fmax(totslp, (real)2.2250738585072014e-308);
        int jq = j + (int)lrint(fmax((real)0, copysign((real)1, dzy)));
        if (fabs(dzy) > dzmax && sd.structdepth[ID(i, jq)] > c.eps) {
            n++;
            real dzb = copysign((real)1, dzy) * (fabs(dzy) - dzmax) * dnv(i, j);
            real s2 = Sig / dnv(i, j);
            dzb = dzb * dt / avaltime * (s2 * s2);
            size_t ke, kd; real dAfac;
            if (dzb >= 0) { ke = ID(i, j + 1); kd = k; dAfac = dsdnzi(i, j) / dsdnzi(i, j + 1); dzb = fmin(dzb, sd.structdepth[ID(i, j + 1)]); }
            else { ke = k; kd = ID(i, j + 1); dAfac = dsdnzi(i, j + 1) / dsdnzi(i, j); dzb = fmax(dzb, -sd.structdepth[k]); }
            aval_move(ke, kd, dzb, dAfac, dt);
        }
    }
    if (n) atomicAdd(sd.naval, n);
}
// lateral boundaries of the bed fields (2D)
__global__ void k_bed_lat() {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    int ny = c.ny;
    for (int s = 0; s < 2; s++) {
        size_t kt = ID(i, s == 0 ? 1 : ny + 1), kf = ID(i, s == 0 ? 2 : ny);
        d.zb[kt] = d.zb[kf]; sd.dzbdt[kt] = sd.dzbdt[kf]; sd.dzbnow[kt] = sd.dzbnow[kf]; sd.sedero[kt] = sd.sedero[kf];
        sd.structdepth[kt] = sd.structdepth[kf]; sd.z0bed[kt] = sd.z0bed[kf];
        for (int m = 0; m < cs.nd; m++) {
            sd.dzbed[(size_t)m * NN + kt] = sd.dzbed[(size_t)m * NN + kf];
            for (int g = 0; g < cs.ngd; g++) PB(kt, m, g) = PB(kf, m, g);
        }
    }
}
// ngd > 1: XBeach assigns the whole D50top/D90top arrays inside its (i,j) loop, so the last inner cell wins
__global__ void k_bed_d50top(int il, int jl) {
    IJ
    size_t k = ID(i, j), q = ID(il, jl);
    real a = 0, b = 0;
    for (int g = 0; g < cs.ngd; g++) { a += PB(q, 0, g) * cs.D50[g]; b += PB(q, 0, g) * cs.D90[g]; }
    sd.D50top[k] = a; sd.D90top[k] = b;
}

// setbathy_update: prescribed bed level series, linear in time
__global__ void k_setbathy(const real *z0, const real *z1, double t, double t0, double t1, int mode) {
    IJ
    size_t k = ID(i, j);
    real zn;
    if (mode == 0) zn = z0[k];
    else { real a = (real)t1, b = (real)t0; real dyy = a == b ? (real)0 : (z1[k] - z0[k]) / (a - b); zn = z0[k] + ((real)t - b) * dyy; }
    d.zs[k] = d.zs[k] + zn - d.zb[k];
    d.zb[k] = zn;
    real hh = fmax(d.zs[k] - d.zb[k], c.eps);
    d.hh[k] = hh;
    if (hh > c.eps) d.wetz[k] = 1;
    else { d.wetz[k] = 0; d.zs[k] = d.zb[k] + c.eps; d.hh[k] = c.eps; }
}
