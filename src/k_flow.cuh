// k_flow.cuh: wetcells.F90 compute_wetcells, timestep.F90 compute_dt, flow_timestep.F90 flow (explicit momentum,
// viscosity, continuity, derived fields), bedroughness.F90.
#pragma once

// ---------------------------------------------------------------------------------------------
// compute_wetcells
// ---------------------------------------------------------------------------------------------
__global__ void k_wetcells() {
    IJ
    size_t k = ID(i, j);
    int ie = min(c.nx, i) + 1, js = min(c.ny, j) + 1;
    real hum = fmax((real)0.5 * (d.hh[k] + d.hh[ID(ie, j)]), c.eps);
    real hvm = fmax((real)0.5 * (d.hh[k] + d.hh[ID(i, js)]), c.eps);
    d.hum[k] = hum; d.hvm[k] = hvm;
    d.wetu[k] = (d.hu[k] > c.eps + NUMEPS && hum > c.eps + NUMEPS) ? 1 : 0;
    d.wetv[k] = (d.hv[k] > c.eps + NUMEPS && hvm > c.eps + NUMEPS) ? 1 : 0;
    d.wetz[k] = (d.hh[k] > c.eps + NUMEPS) ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------
// compute_dt (time-step limit before the CFL*0.5 factor). Positive doubles order like their bit patterns.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void atomic_min_pos(unsigned long long *a, double v) {
    atomicMin(a, (unsigned long long)__double_as_longlong(v));
}
__global__ void k_dt() {
    __shared__ double sm[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    double dt = 1e300;
    const double tny = 2.2250738585072014e-308, g = c.g, nmin = (double)1e-6f;
    if (!c.oned && i >= 2 && i <= c.nx && j >= 2 && j <= c.ny) {
        size_t k = ID(i, j);
        if (d.wetu[k] == 1) {
            double mdx = dsu(i + 1, j), mdy = fmin((double)dnv(i, j), (double)dnv(i, j - 1)), cu = sqrt(g * d.hu[k]);
            dt = fmin(dt, mdx / fmax(tny, fmax(cu + fabs((double)d.uu[k]), fabs((double)d.ueu[k]))));
            dt = fmin(dt, mdy / fmax(tny, cu + fabs((double)d.vu[k])));
        }
        if (d.wetv[k] == 1) {
            double mdx = fmin((double)dsu(i, j), (double)dsu(i - 1, j)), mdy = dnv(i, j), cv = sqrt(g * d.hv[k]);
            dt = fmin(dt, mdx / fmax(tny, cv + fabs((double)d.uv[k])));
            dt = fmin(dt, mdy / fmax(tny, fmax(cv + fabs((double)d.vv[k]), fabs((double)d.vev[k]))));
        }
        if (d.wetz[k] == 1) {
            double mdx = fmin((double)dsu(i, j), (double)dsz(i, j)); mdx *= mdx;
            double mdy = fmin((double)dnv(i, j), (double)dnz(i, j)); mdy *= mdy;
            dt = fmin(dt, 0.5 * mdx * mdy / (mdx + mdy) / fmax((double)d.nuh[k], nmin));
        }
    } else if (c.oned && i >= 2 && i <= c.nx && j == 1) {
        size_t k = ID(i, 1);
        if (d.wetu[k] == 1) {
            double mdx = dsu(i, 1);
            dt = fmin(dt, mdx / fmax(tny, fmax(sqrt(g * d.hu[k]) + fabs((double)d.uu[k]), fabs((double)d.ueu[k]))));
            if (c.sed_ngd > 0) {   // u-points bed celerity
                real sb = 0, ss = 0;
                for (int g = 0; g < c.sed_ngd; g++) { sb += d.Subg[(size_t)g * c.NX1 * c.NY1 + k]; ss += d.Susg[(size_t)g * c.NX1 * c.NY1 + k]; }
                double celeru = 0.1 * fabs((double)(sb + ss)) * c.sed_morfac / (1.0 - c.por);
                dt = fmin(dt, mdx / fmax(tny, celeru));
            }
        }
        if (d.wetz[k] == 1) {
            double mdx = fmin((double)dsu(i, 1), (double)dsu(i - 1, 1));
            dt = fmin(dt, mdx / fmax(tny, sqrt(g * d.hv[k]) + fabs((double)d.uv[k])));
            mdx = fmin((double)dsu(i, 1), (double)dsz(i, 1)); mdx *= mdx;
            double mdy = c.par_dy > 0 ? (double)c.par_dy : mdx;
            dt = fmin(dt, 0.5 * mdx * mdy / (mdx + mdy) / fmax((double)d.nuh[k], nmin));
        }
    }
    sm[tid] = dt;
    __syncthreads();
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) { if (tid < s) sm[tid] = fmin(sm[tid], sm[tid + s]); __syncthreads(); }
    if (tid == 0) atomic_min_pos(d.dtmin, sm[0]);
}

// ---------------------------------------------------------------------------------------------
// bedroughness_init / bedroughness_update
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ real bfcoef(size_t k) { return d.bfc ? d.bfc[k] : c.bedfriccoef; }
// init = 1: bedroughness_init (mincf applied); init = 0: bedroughness_update (Manning and White-Colebrook recomputed
// without mincf; Chezy and cf keep their initial value), then the additional bed shear stresses (acceleration,
// turbulence, infiltration) and cfu, cfv capped at 1. fa: smoothing factor dt_old/(Trep/20), dto: the dt two
// calls back (XBeach dtold(1)), iom: 1/omega_p (Nielsen acceleration)
__device__ __forceinline__ real D90t(size_t k) { return d.D90topA ? d.D90topA[k] : c.D90top; }
__device__ __forceinline__ real D50t(size_t k) { return d.D50topA ? d.D50topA[k] : c.D50top; }
__global__ void k_rough(int init, real fa, real dto, real iom) {
    IJ
    size_t k = ID(i, j);
    int nx = c.nx, ny = c.ny;
    real cu = d.cfu[k], cv = d.cfv[k];
    if (c.bedfriction == BF_MANNING && c.dynamrough && !init) {   // dynamicroughness: n follows burial/erosion
        real sed = d.sedero ? d.sedero[k] : 0, nm = c.nmanmin, n0 = d.nman0[k], n;
        if (d.isveggie[k] && sed > 0) n = nm + fmin(fmax((c.dstem - sed) / c.dstem, (real)0), (real)1) * (n0 - nm);
        else if (d.isveggie[k] && sed < c.droot * -1) { n = nm; d.isveggie[k] = 0; }
        else n = nm + fmin(fmax((c.droot + sed) / c.droot, (real)0), (real)1) * (n0 - nm);
        d.bfc[k] = n;
    }
    real bf = bfcoef(k);
    switch (c.bedfriction) {
    case BF_CHEZY: if (init) cu = cv = c.g / (bf * bf); break;
    case BF_CF: if (init) cu = cv = bf; break;
    case BF_MANNING: {
        real n2g = c.g * bf * bf;
        if (d.wetu[k] == 1) { cu = fmin(n2g / cbrt(d.hu[k]), c.maxcf); if (init) cu = fmax(cu, c.mincf); } else cu = c.maxcf;
        if (d.wetv[k] == 1) { cv = fmin(n2g / cbrt(d.hv[k]), c.maxcf); if (init) cv = fmax(cv, c.mincf); } else cv = c.maxcf;
        break; }
    case BF_WCG: {   // roughness from the grain size of the top layer
        real kru, krv;
        if (c.sed_ngd <= 1) { kru = 3 * D90t(k); krv = kru; }
        else {
            int ii = i <= nx ? i : nx;
            kru = (init || d.wetu[ID(ii, j)] == 1) ? (real)1.5 * (D90t(ID(ii, j)) + D90t(ID(ii + 1, j))) : d.kru[ID(ii, j)];
            if (c.oned) krv = (init || d.wetv[k] == 1) ? 3 * D90t(k) : d.krv[k];
            else { int jj = j <= ny ? j : ny;
                   krv = (init || d.wetv[ID(i, jj)] == 1) ? (real)1.5 * (D90t(ID(i, jj)) + D90t(ID(i, jj + 1))) : d.krv[ID(i, jj)]; }
            if (i <= nx) d.kru[k] = kru;
            if (c.oned || j <= ny) d.krv[k] = krv;
        }
        if (d.wetu[k] == 1) { real l = 18 * log10(12 * fmax(d.hu[k], kru) / kru); cu = c.g / (l * l);
                              cu = init ? fmax(fmin(cu, c.maxcf), c.mincf) : fmin(cu, (real)0.1); } else cu = c.maxcf;
        if (d.wetv[k] == 1) { real l = 18 * log10(12 * fmax(d.hv[k], krv) / krv); cv = c.g / (l * l);
                              cv = init ? fmax(fmin(cv, c.maxcf), c.mincf) : fmin(cv, c.maxcf); } else cv = c.maxcf;
        break; }
    default: {   // white-colebrook (kru = krv = bedfriccoef)
        real kr = bf;
        if (d.wetu[k] == 1) { real l = 18 * log10(12 * fmax(d.hu[k], kr) / kr); cu = fmin(c.g / (l * l), c.maxcf); if (init) cu = fmax(cu, c.mincf); } else cu = c.maxcf;
        if (d.wetv[k] == 1) { real l = 18 * log10(12 * fmax(d.hv[k], kr) / kr); cv = fmin(c.g / (l * l), c.maxcf); if (init) cv = fmax(cv, c.mincf); } else cv = c.maxcf;
        break; }
    }
    if (!init && d.taubx_add) {
        real tx = 0, ty = 0;
        const bool vpart = !c.oned;   // XBeach skips the v part in 1D when swave = 0
        if (c.fric_acc == 1) {        // McCall: inertia of the bed layer, ci rho min(D50, h) du/dt
            if (d.wetu[k] == 1) {
                real uf = (1 - fa) * d.ueuf[k] + fa * d.ueu[k];
                real du = (uf - d.ueuold[k]) / dto;
                du = fmin(fmax(du, -100 * c.g), 100 * c.g);
                tx = tx + c.ci * c.rho * fmin(D50t(k), d.hu[k]) * du;
                d.ueuf[k] = uf;
            } else d.ueuf[k] = 0;
            d.ueuold[k] = d.ueuf[k];
            if (vpart) {
                if (d.wetv[k] == 1) {
                    real vf = (1 - fa) * d.vevf[k] + fa * d.vev[k];
                    real dv = (vf - d.vevold[k]) / dto;
                    dv = fmin(fmax(dv, -100 * c.g), 100 * c.g);
                    ty = ty + c.ci * c.rho * fmin(D50t(k), d.hv[k]) * dv;
                    d.vevf[k] = vf;
                } else d.vevf[k] = 0;
                d.vevold[k] = d.vevf[k];
            }
        } else if (c.fric_acc == 2) {   // Nielsen: phase-shifted velocity in the bed shear stress
            real du = 0, dv = 0;
            if (d.wetu[k] == 1) { real uf = (1 - fa) * d.ueuf[k] + fa * d.ueu[k]; du = (uf - d.ueuold[k]) / dto; d.ueuf[k] = uf; }
            else d.ueuf[k] = 0;
            if (vpart) {
                if (d.wetv[k] == 1) { real vf = (1 - fa) * d.vevf[k] + fa * d.vev[k]; dv = (vf - d.vevold[k]) / dto; d.vevf[k] = vf; }
                else d.vevf[k] = 0;
            }
            d.ueuold[k] = d.ueuf[k]; d.vevold[k] = d.vevf[k];
            real cph = cos(c.phiacc), sph = sin(c.phiacc);
            if (d.wetu[k] == 1) {
                real a = sph * iom * du;
                tx = tx + cu * c.rho * ((cph * cph - 1) * d.ueu[k] * d.vmageu[k] + 2 * cph * d.ueu[k] * sph * iom * du + a * a * copysign((real)1, d.ueu[k]));
            }
            if (vpart && d.wetv[k] == 1) {
                real a = sph * iom * dv;
                ty = ty + cv * c.rho * ((cph * cph - 1) * d.vev[k] * d.vmagev[k] + 2 * cph * d.vev[k] * sph * iom * dv + a * a * copysign((real)1, d.vev[k]));
            }
        }
        if (c.fric_turb && d.kbA) {   // turbulence: cf rho gamma_turb kb at the bed
            int ii = i <= nx ? i : nx;
            real kbl = d.wetu[ID(ii, j)] == 1 ? (real)0.5 * (d.kbA[ID(ii, j)] + d.kbA[ID(ii + 1, j)]) : 0;
            real uec = i == c.NX1 ? 0 : (i == 1 ? d.ueu[ID(1, j)] : (real)0.5 * (d.ueu[ID(i - 1, j)] + d.ueu[k]));
            if (d.wetu[k] == 1 && kbl > 0) tx = tx + cu * c.rho * c.gamma_turb * kbl * copysign((real)1, uec);
            real kbv;
            if (c.oned) kbv = d.kbA[k];
            else { int jj = j <= ny ? j : ny; kbv = d.wetv[ID(i, jj)] == 1 ? (real)0.5 * (d.kbA[ID(i, jj)] + d.kbA[ID(i, jj + 1)]) : 0; }
            if (d.wetv[k] == 1 && kbv > 0) ty = ty + cv * c.rho * c.gamma_turb * kbv * copysign((real)1, d.ve[k]);
        }
        d.taubx_add[k] = tx; d.tauby_add[k] = ty;
        if (c.gwfi) gw_friction_infil(i, j, cu, cv);   // adds the infiltration effect
    }
    if (!init) { cu = fmin(cu, (real)1); cv = fmin(cv, (real)1); }
    d.cfu[k] = cu; d.cfv[k] = cv;
}

// ---------------------------------------------------------------------------------------------
// flow: water level slopes
// ---------------------------------------------------------------------------------------------
__global__ void k_slopes() {
    IJ
    size_t k = ID(i, j);
    real ph = d.ph ? d.ph[k] : 0;
    if (IN_UU) d.dzsdx[k] = (d.zs[ID(i + 1, j)] + (d.ph ? d.ph[ID(i + 1, j)] : 0) - d.zs[k] - ph) / dsu(i, j);
    if (!c.oned && IN_VV) d.dzsdy[k] = (d.zs[ID(i, j + 1)] + (d.ph ? d.ph[ID(i, j + 1)] : 0) - d.zs[k] - ph) / dnv(i, j);
}

// momentum-conserving / energy-head advection (flow_timestep.F90)
__global__ void k_advection() {
    IJ
    size_t k = ID(i, j);
    real esd = c.eps_sd;
    if (IN_UU) {   // ududx
        real a = 0, qin, ui = d.uu[k];
        qin = (real)0.5 * (d.qx[k] + d.qx[ID(i - 1, j)]);
        if (qin > 0) {
            real da = alfau(i, j) - alfau(i - 1, j);
            real uin = ROTP(d.uu[ID(i - 1, j)], d.vu[ID(i - 1, j)], da);
            if ((ui - d.uu[ID(i - 1, j)]) > esd) a += (real)0.5 * (d.uu[ID(i - 1, j)] + ui) * (ui - uin) * dnz(i, j) * dsdnui(i, j);
            else a += qin / d.hum[k] * (ui - uin) * dnz(i, j) * dsdnui(i, j);
        }
        qin = (real)-0.5 * (d.qx[k] + d.qx[ID(i + 1, j)]);
        if (qin > 0) {
            real da = alfau(i, j) - alfau(i + 1, j);
            real uin = ROTP(d.uu[ID(i + 1, j)], d.vu[ID(i + 1, j)], da);
            if ((d.uu[ID(i + 1, j)] - ui) > esd) a -= (real)0.5 * (d.uu[ID(i + 1, j)] + ui) * (ui - uin) * dnz(i + 1, j) * dsdnui(i, j);
            else a += qin / d.hum[k] * (ui - uin) * dnz(i + 1, j) * dsdnui(i, j);
        }
        d.ududx[k] = a;
    }
    if (!c.oned && IN_VV) {   // vdudy (u-point term over the v-point loop bounds, as in XBeach)
        real a = 0, qin, ui = d.uu[k];
        qin = (real)0.5 * (d.qy[ID(i, j - 1)] + d.qy[ID(i + 1, j - 1)]);
        if (qin > 0) {
            real da = alfau(i, j) - alfau(i, j - 1);
            real uin = ROTP(d.uu[ID(i, j - 1)], d.vu[ID(i, j - 1)], da);
            if ((d.vv[k] - d.vv[ID(i, j - 1)]) > esd)
                a += (real)0.5 * (d.vv[ID(i, j - 1)] + d.vv[ID(i + 1, j - 1)]) * (ui - uin) * dsc(i, j - 1) * dsdnui(i, j);
            else a += qin / d.hum[k] * (ui - uin) * dsc(i, j - 1) * dsdnui(i, j);
        }
        qin = (real)-0.5 * (d.qy[k] + d.qy[ID(i + 1, j)]);
        if (qin > 0) {
            real da = alfau(i, j) - alfau(i, j + 1);
            real uin = ROTP(d.uu[ID(i, j + 1)], d.vu[ID(i, j + 1)], da);
            if ((d.vv[ID(i, j + 1)] - d.vv[k]) > esd)
                a -= (real)0.5 * (d.vv[k] + d.vv[ID(i + 1, j)]) * (ui - uin) * dsc(i, j) * dsdnui(i, j);
            else a += qin / d.hum[k] * (ui - uin) * dsc(i, j) * dsdnui(i, j);
        }
        d.vdudy[k] = a;
    }
    real vd = 0, ud = 0;
    if (!c.oned) {
        if (IN_VV) {
            real vi = d.vv[k], qin;
            qin = (real)0.5 * (d.qy[k] + d.qy[ID(i, j - 1)]);
            if (qin > 0) {
                real da = alfav(i, j) - alfav(i, j - 1);
                real vin = ROTM(d.vv[ID(i, j - 1)], d.uv[ID(i, j - 1)], da);
                if ((vi - d.vv[ID(i, j - 1)]) > esd) vd += (real)0.5 * (d.vv[ID(i, j - 1)] + vi) * (vi - vin) * dsz(i, j) * dsdnvi(i, j);
                else vd += qin / d.hvm[k] * (vi - vin) * dsz(i, j) * dsdnvi(i, j);
            }
            qin = (real)-0.5 * (d.qy[k] + d.qy[ID(i, j + 1)]);
            if (qin > 0) {
                real da = alfav(i, j) - alfav(i, j + 1);
                real vin = ROTM(d.vv[ID(i, j + 1)], d.uv[ID(i, j + 1)], da);
                if ((d.vv[ID(i, j + 1)] - vi) > esd) vd -= (real)0.5 * (d.vv[ID(i, j + 1)] + vi) * (vi - vin) * dsz(i, j + 1) * dsdnvi(i, j);
                else vd += qin / d.hvm[k] * (vi - vin) * dsz(i, j + 1) * dsdnvi(i, j);
            }
            qin = (real)0.5 * (d.qx[ID(i - 1, j)] + d.qx[ID(i - 1, j + 1)]);
            if (qin > 0) {
                real da = alfav(i, j) - alfav(i - 1, j);
                real vin = ROTM(d.vv[ID(i - 1, j)], d.uv[ID(i - 1, j)], da);
                if ((d.uu[k] - d.uu[ID(i - 1, j)]) > esd)
                    ud += (real)0.5 * (d.uu[ID(i - 1, j)] + d.uu[ID(i - 1, j + 1)]) * (vi - vin) * dnc(i - 1, j) * dsdnvi(i, j);
                else ud += qin / d.hvm[k] * (vi - vin) * dnc(i - 1, j) * dsdnvi(i, j);
            }
            qin = (real)-0.5 * (d.qx[k] + d.qx[ID(i, j + 1)]);
            if (qin > 0) {
                real da = alfav(i, j) - alfav(i + 1, j);
                real vin = ROTM(d.vv[ID(i + 1, j)], d.uv[ID(i + 1, j)], da);
                if ((d.uu[ID(i + 1, j)] - d.uu[k]) > esd)
                    ud -= (real)0.5 * (d.uu[k] + d.uu[ID(i, j + 1)]) * (vi - vin) * dnc(i, j) * dsdnvi(i, j);
                else ud += qin / d.hvm[k] * (vi - vin) * dnc(i, j) * dsdnvi(i, j);
            }
        }
        // global lateral rows of vdvdy (xmpi_isleft / xmpi_isright)
        if (!c.cyclic && j == 1 && i >= 2 && i <= c.nx) {
            real qin = (real)-0.5 * (d.qy[ID(i, 1)] + d.qy[ID(i, 2)]);
            if (qin > 0) {
                real da = alfav(i, 1) - alfav(i, 2);
                real vin = ROTM(d.vv[ID(i, 2)], d.uv[ID(i, 2)], da);
                vd += qin * (d.vv[ID(i, 1)] - vin) * dsz(i, 2) / d.hvm[k] * dsdnvi(i, 1);
            }
        }
        if (!c.cyclic && j == c.ny && i >= 2 && i <= c.nx) {
            real qin = (real)0.5 * (d.qy[ID(i, c.ny)] + d.qy[ID(i, c.ny - 1)]);
            if (qin > 0) {
                real da = alfav(i, c.ny) - alfav(i, c.ny - 1);
                real vin = ROTM(d.vv[ID(i, c.ny - 1)], d.uv[ID(i, c.ny - 1)], da);
                vd += qin * (d.vv[ID(i, c.ny)] - vin) * dsz(i, c.ny) / d.hvm[k] * dsdnvi(i, c.ny);
            }
        }
    } else if (i >= 2 && i <= c.nx) {   // 1D udvdx
        real vi = d.vv[k], qin = d.qx[ID(i - 1, 1)];
        if (qin > 0) {
            real da = alfav(i, 1) - alfav(i - 1, 1);
            real vin = ROTM(d.vv[ID(i - 1, 1)], d.uv[ID(i - 1, 1)], da);
            ud += qin * (vi - vin) * dnc(i - 1, 1) / d.hvm[k] * dsdnvi(i, 1);
        }
        qin = -d.qx[k];
        if (qin > 0) {
            real da = alfav(i, 1) - alfav(i + 1, 1);
            real vin = ROTM(d.vv[ID(i + 1, 1)], d.uv[ID(i + 1, 1)], da);
            ud += qin * (vi - vin) * dnc(i, 1) / d.hvm[k] * dsdnvi(i, 1);
        }
    }
    d.vdvdy[k] = vd;
    d.udvdx[k] = ud;
}

// visc_smagorinsky
__global__ void k_smag() {
    IJ
    size_t k = ID(i, j);
    if (!c.oned) {
        if (!IN(2, c.nx, 2, c.ny)) return;
        real dudx = (d.uu[k] - d.uu[ID(i - 1, j)]) / dsz(i, j);
        real dudy = (real)0.5 * (d.uu[ID(i, j + 1)] - d.uu[ID(i, j - 1)] + d.uu[ID(i - 1, j + 1)] - d.uu[ID(i - 1, j - 1)]) / (dnv(i, j) + dnv(i, j - 1));
        real dvdx = (real)0.5 * (d.vv[ID(i + 1, j)] - d.vv[ID(i - 1, j)] + d.vv[ID(i + 1, j - 1)] - d.vv[ID(i - 1, j - 1)]) / (dsu(i, j) + dsu(i - 1, j));
        real dvdy = (d.vv[k] - d.vv[ID(i, j - 1)]) / dnz(i, j);
        real Tau = sqrt(2 * dudx * dudx + 2 * dvdy * dvdy + (dvdx + dudy) * (dvdx + dudy));
        real l = 1 / dsdnzi(i, j);
        d.nuh[k] = c.nuh * c.nuh * l * Tau * (real)(d.wetu[k] * d.wetu[ID(i - 1, j)] * d.wetv[k] * d.wetv[ID(i, j - 1)]);
    } else {
        if (i < 2 || i > c.nx) return;
        real dudx = (d.uu[k] - d.uu[ID(i - 1, 1)]) / dsz(i, 1);
        real dvdx = (d.vv[ID(i + 1, 1)] - d.vv[ID(i - 1, 1)]) / (dsu(i, 1) + dsu(i - 1, 1));
        real Tau = sqrt(2 * dudx * dudx + dvdx * dvdx);
        real l = c.par_dy > -1 ? dsz(i, 1) * c.par_dy : dsz(i, 1) * dsz(i, 1);
        d.nuh[k] = c.nuh * c.nuh * l * Tau * (real)(d.wetu[k] * d.wetu[ID(i - 1, 1)]);
    }
}
__global__ void k_nuh_bc_j() {   // nuh(:,1) = nuh(:,2); nuh(:,ny+1) = nuh(:,ny)
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    d.nuh[ID(i, 1)] = d.nuh[ID(i, 2)];
    d.nuh[ID(i, c.NY1)] = d.nuh[ID(i, c.ny)];
}
__global__ void k_nuh_bc_i() {   // nuh(1,:) = nuh(2,:); nuh(nx+1,:) = nuh(nx,:)
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    d.nuh[ID(1, j)] = d.nuh[ID(2, j)];
    d.nuh[ID(c.NX1, j)] = d.nuh[ID(c.nx, j)];
}
__global__ void k_nuh_const_brk() {
    IJ
    size_t k = ID(i, j);
    if (!c.smag) d.nuh[k] = c.nuh;
    if (c.nhbreaker == 1 && d.breaking[k] != 0) d.nuh[k] *= c.breakviscfac;
    if (c.nhbreaker == 2 && d.breaking[k] == 1) {   // flow_timestep: velocity gradients of the viscosity terms
        real dudx = (IN_UU) ? (d.uu[k] - d.uu[ID(i - 1, j)]) / dsz(i, j) : 0;
        real dvdy = (c.ny > 2 && i >= 2 && i <= c.nx && j >= 2 && j <= c.ny - 1) ? (d.vv[k] - d.vv[ID(i, j - 1)]) / dnz(i, j) : 0;
        real l = c.nuh * c.breakvisclen * d.hh[k];
        d.nuh[k] = d.nuh[k] + l * l * sqrt(dudx * dudx + dvdy * dvdy);
    }
}
__global__ void k_viscu() {
    IJ
    if (!IN_UU) return;
    size_t k = ID(i, j);
    real sf = c.smag ? 2 : 1;
    real dudx1 = d.nuh[ID(i + 1, j)] * d.hh[ID(i + 1, j)] * (d.uu[ID(i + 1, j)] - d.uu[k]) / dsz(i + 1, j);
    real dudx2 = d.nuh[k] * d.hh[k] * (d.uu[k] - d.uu[ID(i - 1, j)]) / dsz(i, j);
    real vis = sf * ((1 / d.hum[k]) * (2 * (dudx1 - dudx2) / (dsz(i, j) + dsz(i + 1, j))));
    int jp1 = min(j + 1, c.NY1), jm1 = max(j - 1, 1);
    real nuh1 = (real)0.25 * (d.nuh[k] + d.nuh[ID(i + 1, j)] + d.nuh[ID(i + 1, jp1)] + d.nuh[ID(i, jp1)]);
    real nuh2 = (real)0.25 * (d.nuh[k] + d.nuh[ID(i + 1, j)] + d.nuh[ID(i + 1, jm1)] + d.nuh[ID(i, jm1)]);
    real dudy1 = nuh1 * (real)0.5 * (d.hvm[k] + d.hvm[ID(i + 1, j)]) * (d.uu[ID(i, jp1)] - d.uu[k]) / dnc(i, j);
    real dudy2 = nuh2 * (real)0.5 * (d.hvm[ID(i, jm1)] + d.hvm[ID(i + 1, jm1)]) * (d.uu[k] - d.uu[ID(i, jm1)]) / dnc(i, jm1);
    vis += (1 / d.hum[k]) * (2 * (dudy1 - dudy2) / (dnc(i, j) + dnc(i, jm1))) * (real)(d.wetu[ID(i, jp1)] * d.wetu[ID(i, jm1)]);
    if (c.smag) {
        real dvdx1 = nuh1 * (real)0.5 * (d.hvm[k] + d.hvm[ID(i + 1, j)]) * (d.vv[ID(i + 1, j)] - d.vv[k]) / dsc(i, j);
        real dvdx2 = nuh2 * (real)0.5 * (d.hvm[ID(i, jm1)] + d.hvm[ID(i + 1, jm1)]) * (d.vv[ID(i + 1, jm1)] - d.vv[ID(i, jm1)]) / dsc(i, jm1);
        vis += (1 / d.hum[k]) * (dvdx1 - dvdx2) / dnz(i, j) *
               (real)(d.wetv[ID(i + 1, j)] * d.wetv[k] * d.wetv[ID(i + 1, jm1)] * d.wetv[ID(i, jm1)]);
    }
    d.viscu[k] = vis;
}
__global__ void k_viscv1() {
    IJ
    size_t k = ID(i, j);
    real vis = 0;
    if (IN_VV) {
        int jp1 = min(j + 1, c.NY1), jm1 = max(j - 1, 1);
        real dvdy1 = d.nuh[ID(i, jp1)] * d.hh[ID(i, jp1)] * (d.vv[ID(i, jp1)] - d.vv[k]) / dnz(i, jp1);
        real dvdy2 = d.nuh[k] * d.hh[k] * (d.vv[k] - d.vv[ID(i, jm1)]) / dnz(i, j);
        vis = (1 / d.hvm[k]) * 2 * (dvdy1 - dvdy2) / (dnz(i, j) + dnz(i, jp1)) * (real)(d.wetv[ID(i, jp1)] * d.wetv[ID(i, jm1)]);
    }
    d.viscv[k] = vis;
}
__global__ void k_viscv_bc() {   // viscv(:,1) = viscv(:,2); viscv(:,ny) = viscv(:,ny-1)
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    d.viscv[ID(i, 1)] = d.viscv[ID(i, 2)];
    d.viscv[ID(i, c.ny)] = d.viscv[ID(i, c.ny - 1)];
}
// viscv (second part) and nuh = nuhv*nuh (XBeach keeps the scaled nuh)
__global__ void k_viscv2() {
    IJ
    size_t k = ID(i, j);
    real vis = d.viscv[k] * (c.smag ? 2 : 1);
    if (IN_VV) {
        int jp1 = min(j + 1, c.NY1);
        real nh = c.nuhv;
        real nC = nh * d.nuh[k], nE = nh * d.nuh[ID(i + 1, j)], nEN = nh * d.nuh[ID(i + 1, jp1)], nN = nh * d.nuh[ID(i, jp1)];
        real nW = nh * d.nuh[ID(i - 1, j)], nWN = nh * d.nuh[ID(i - 1, jp1)];
        real nuh1 = (real)0.25 * (nC + nE + nEN + nN);
        real nuh2 = (real)0.25 * (nC + nW + nWN + nN);
        real dvdx1 = nuh1 * (real)0.5 * (d.hum[k] + d.hum[ID(i, jp1)]) * (d.vv[ID(i + 1, j)] - d.vv[k]) / dsc(i, j);
        real dvdx2 = nuh2 * (real)0.5 * (d.hum[ID(i - 1, j)] + d.hum[ID(i - 1, jp1)]) * (d.vv[k] - d.vv[ID(i - 1, j)]) / dsc(i - 1, j);
        vis += (1 / d.hvm[k]) * (2 * (dvdx1 - dvdx2) / (dsc(i - 1, j) + dsc(i, j))) * (real)(d.wetv[ID(i + 1, j)] * d.wetv[ID(i - 1, j)]);
        if (c.smag) {
            real dudy1 = nuh1 * (real)0.5 * (d.hum[k] + d.hum[ID(i, jp1)]) * (d.uu[ID(i, jp1)] - d.uu[k]) / dnc(i, j);
            real dudy2 = nuh2 * (real)0.5 * (d.hum[ID(i - 1, j)] + d.hum[ID(i - 1, jp1)]) * (d.uu[ID(i - 1, jp1)] - d.uu[ID(i - 1, j)]) / dnc(i - 1, j);
            vis += (1 / d.hvm[k]) * (dudy1 - dudy2) / dsz(i, j) *
                   (real)(d.wetu[ID(i, jp1)] * d.wetu[k] * d.wetu[ID(i - 1, jp1)] * d.wetv[ID(i - 1, j)]);
        }
    }
    d.viscv[k] = vis;
}
__global__ void k_nuh_scale() {   // s%nuh = par%nuhv*s%nuh after viscv has used the scaled values
    IJ
    if (c.nuhv != 1) d.nuh[ID(i, j)] *= c.nuhv;
}
__global__ void k_visc_off() {
    IJ
    size_t k = ID(i, j);
    d.nuh[k] = 0; d.viscu[k] = 0;
}

// bed shear stress and explicit momentum update
__global__ void k_momentum(real dt) {
    IJ
    size_t k = ID(i, j);
    real cap = 100 * c.g * c.rho;
    real tbx = 0, tby = 0;
    if (d.wetu[k] == 1) { tbx = d.cfu[k] * c.rho * d.ueu[k] * d.vmageu[k]; if (d.taubx_add) tbx += d.taubx_add[k]; }
    if (fabs(tbx) > cap * d.hu[k]) tbx = copysign(cap * d.hu[k], tbx);
    if (d.wetv[k] == 1) { tby = d.cfv[k] * c.rho * d.vev[k] * d.vmagev[k]; if (d.tauby_add) tby += d.tauby_add[k]; }
    if (fabs(tby) > cap * d.hv[k]) tby = copysign(cap * d.hv[k], tby);
    d.taubx[k] = tbx; d.tauby[k] = tby;
    real fmax_ = c.maxfacg * c.g;
    real wsu = d.windsu ? d.windsu[k] : 0, wnv = d.windnv ? d.windnv[k] : 0;
    real wmag = sqrt(wsu * wsu + wnv * wnv);
    if (IN_UU) {
        if (d.wetu[k] == 1) {
            real dudt = d.ududx[k] + d.vdudy[k] - d.viscu[k] + c.g * d.dzsdx[k] + tbx / (c.rho * d.hu[k])
                        + (d.Fvegu ? d.Fvegu[k] / (c.rho * d.hu[k]) : 0) - c.fc * d.vu[k]
                        - c.rhoa * c.Cd * wsu * wmag / (c.rho * d.hum[k]);
            dudt = fmin(fmax(dudt, -fmax_), fmax_);
            d.uu[k] -= dt * dudt;
        } else d.uu[k] = 0;
    }
    if (IN_VV) {
        if (d.wetv[k] == 1) {
            real dvdt = d.udvdx[k] + d.vdvdy[k] - d.viscv[k] + c.g * d.dzsdy[k] + tby / (c.rho * d.hv[k])
                        + (d.Fvegv ? d.Fvegv[k] / (c.rho * d.hv[k]) : 0) + c.fc * d.uv[k]
                        - c.rhoa * c.Cd * wnv * wmag / (c.rho * d.hvm[k]);
            dvdt = fmin(fmax(dvdt, -fmax_), fmax_);
            d.vv[k] -= dt * dvdt;
        } else d.vv[k] = 0;
    }
}
// lateral boundaries after the momentum update: uu copies, then flow_lat_bc for vv (2D only)
__global__ void k_lat_uv(real dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    d.uu[ID(i, 1)] = d.uu[ID(i, 2)];
    d.uu[ID(i, c.NY1)] = d.uu[ID(i, c.ny)];
    if (i < 2 || i > c.nx) return;
    for (int side = 0; side < 2; side++) {
        int jb = side == 0 ? 1 : c.ny, jn = side == 0 ? 2 : c.ny - 1;
        int type = side == 0 ? c.right : c.left;     // XBeach passes par%right at j = 1
        size_t k = ID(i, jb);
        real vb = 0;
        real wsu = d.windsu ? d.windsu[k] : 0, wnv = d.windnv ? d.windnv[k] : 0;
        switch (type) {
        case LR_WALL: vb = 0; break;
        case LR_NEUMANN_V: vb = d.vv[ID(i, jn)]; break;
        case LR_NO_ADVEC:
            if (d.wetv[k] == 1) {
                real adv = d.udvdx[k] + d.vdvdy[k] - d.viscv[k] + c.fc * d.uv[k];
                if (d.vv[k] > 0) adv = fmax(adv, (real)0); else if (d.vv[k] < 0) adv = fmin(adv, (real)0); else adv = 0;
                vb = d.vv[k] - dt * (adv + c.g * d.dzsdy[k] + d.tauby[k] / (c.rho * d.hvm[k])
                                     - c.rhoa * c.Cd * wnv * sqrt(wsu * wsu + wnv * wnv) / (c.rho * d.hvm[k]));
            }
            break;
        case LR_NEUMANN:
            if (d.wetv[k] == 1)
                vb = d.vv[k] - dt * (d.udvdx[k] + d.vdvdy[k] - d.viscv[k] + c.g * d.dzsdy[k] + d.tauby[k] / (c.rho * d.hvm[k])
                                     + c.fc * d.uv[k] - c.rhoa * c.Cd * wnv * sqrt(wsu * wsu + wnv * wnv) / (c.rho * d.hvm[k]));
            break;
        case LR_ABS_1D:
            if (d.wetv[k] == 1) {
                size_t kn = ID(i, jn);
                vb = sqrt(c.g / d.hh[kn]) * (d.zs[kn] - fmax(d.zb[kn], d.zs0[kn])) * (real)(jb - jn);
                d.zs[k] = d.zs[kn];
            }
            break;
        }
        d.w1[ID(i, side + 1)] = vb;                 // scratch, applied in k_lat_v_apply
    }
}
__global__ void k_lat_v_apply() {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    for (int side = 0; side < 2; side++) {
        int jb = side == 0 ? 1 : c.ny, jn = side == 0 ? 2 : c.ny - 1;
        int type = side == 0 ? c.right : c.left;
        real vb;
        if (type == LR_WALL) vb = 0;
        else if (type == LR_NEUMANN_V) vb = d.vv[ID(i, jn)];
        else if (i == 1) vb = d.w1[ID(2, side + 1)] * (real)d.wetv[ID(1, jb)];
        else if (i == c.NX1) vb = d.w1[ID(c.nx, side + 1)] * (real)d.wetv[ID(c.NX1, jb)];
        else vb = d.w1[ID(i, side + 1)];
        d.vv[ID(i, jb)] = vb;
    }
}

// ---------------------------------------------------------------------------------------------
// hu/hv (upwind) + flow_secondorder_huhv, fluxes, continuity, derived fields
// ---------------------------------------------------------------------------------------------
__global__ void k_huhv() {
    IJ
    size_t k = ID(i, j);
    int nx = c.nx, ny = c.ny;
    int ie = min(nx, i) + 1, js = min(ny, j) + 1;
    real u = d.uu[k], v = d.vv[k], zbm, hu, hv;
    zbm = fmax(d.zb[k], d.zb[ID(ie, j)]);
    if (u > c.umin) hu = c.oldhu ? d.hh[k] : d.zs[k] - zbm;
    else if (u < -c.umin) hu = c.oldhu ? d.hh[ID(ie, j)] : d.zs[ID(ie, j)] - zbm;
    else hu = fmax(fmax(d.zs[k], d.zs[ID(ie, j)]) - zbm, c.eps);
    hu = fmax(hu, (real)0);
    zbm = fmax(d.zb[k], d.zb[ID(i, js)]);
    if (v > c.umin) hv = c.oldhu ? d.hh[k] : d.zs[k] - zbm;
    else if (v < -c.umin) hv = c.oldhu ? d.hh[ID(i, js)] : d.zs[ID(i, js)] - zbm;
    else hv = fmax(fmax(d.zs[k], d.zs[ID(i, js)]) - zbm, c.eps);
    hv = fmax(hv, (real)0);
    if (c.secorder) {
        if (IN_UU) {     // imin = 2, imax = nx-1 (serial)
            int imin = 2, imax = nx - 1;
            int ie2 = min(i + 1, imax + 1), iee = min(i + 2, imax + 1), iw = max(i - 1, 1);
            real zsmin = 1e30, zbmax = -1e30;
            for (int ii = iw; ii <= iee; ii++) { zsmin = fmin(zsmin, d.zs[ID(ii, j)]); zbmax = fmax(zbmax, d.zb[ID(ii, j)]); }
            if (zsmin - zbmax > c.eps) {
                if (u > c.umin && i > imin)
                    hu += (real)0.5 * dsu(i, 1) * minmod((d.zs[ID(ie2, j)] - d.zs[k]) / dsu(i, 1), (d.zs[k] - d.zs[ID(iw, j)]) / dsu(i - 1, 1));
                else if (u < -c.umin && i < imax)
                    hu -= (real)0.5 * dsu(i, 1) * minmod((d.zs[ID(iee, j)] - d.zs[ID(ie2, j)]) / dsu(i + 1, 1), (d.zs[ID(ie2, j)] - d.zs[k]) / dsu(i, 1));
            }
        }
        if (!c.oned && IN_VV) {     // jmin = 2, jmax = ny-1 (serial; cyclic as in XBeach MPI)
            int jmin = c.w_jmin, jmax = c.w_jmax;
            int jn = min(j + 1, jmax + 1), jnn = min(j + 2, jmax + 1), jw = max(j - 1, 1);
            real zsmin = 1e30, zbmax = -1e30;
            for (int jj = jw; jj <= jnn; jj++) { zsmin = fmin(zsmin, d.zs[ID(i, jj)]); zbmax = fmax(zbmax, d.zb[ID(i, jj)]); }
            if (zsmin - zbmax > c.eps) {
                if (v > c.umin && j > jmin)
                    hv += (real)0.5 * dnv(i, j) * minmod((d.zs[ID(i, jn)] - d.zs[k]) / dnv(i, j), (d.zs[k] - d.zs[ID(i, jw)]) / dnv(i, j - 1));
                else if (v < -c.umin && j < jmax)
                    hv -= (real)0.5 * dnv(i, j) * minmod((d.zs[ID(i, jnn)] - d.zs[ID(i, jn)]) / dnv(i, j + 1), (d.zs[ID(i, jn)] - d.zs[k]) / dnv(i, j));
            }
        }
    }
    d.hu[k] = hu; d.hv[k] = hv;
    d.qx[k] = u * hu; d.qy[k] = v * hv;
}
__global__ void k_continuity(real dt) {
    IJ
    if (!IN_ZS) return;
    size_t k = ID(i, j);
    real dzsdt;
    if (!c.oned)
        dzsdt = -(d.qx[k] * dnu(i, j) - d.qx[ID(i - 1, j)] * dnu(i - 1, j) + d.qy[k] * dsv(i, j) - d.qy[ID(i, j - 1)] * dsv(i, j - 1)) * dsdnzi(i, j);
    else
        dzsdt = -(d.qx[k] * dnu(i, 1) - d.qx[ID(i - 1, 1)] * dnu(i - 1, 1)) * dsdnzi(i, 1);
    if (d.infil) dzsdt -= d.infil[k];
    if (d.rainrate) dzsdt += d.rainrate[k];
    d.dzsdt[k] = dzsdt;
    d.zs[k] += dzsdt * dt;
}
// u, v at cell centres; vu, uv; Eulerian velocities (no Stokes drift in nonh); hh; maxzs/minzs
__global__ void k_post() {
    IJ
    size_t k = ID(i, j);
    int nx = c.nx, ny = c.ny;
    d.uu_old[k] = d.uu[k]; d.vv_old[k] = d.vv[k];
    real vu = 0, uv = 0;
    if (!c.oned) {
        int jj = j == 1 ? 2 : (j == ny + 1 ? ny : j);    // vu(1:nx,2:ny), then copies to j = 1, ny+1
        if (i <= nx) vu = (real)0.25 * (d.vv[ID(i, jj - 1)] + d.vv[ID(i, jj)] + d.vv[ID(i + 1, jj - 1)] + d.vv[ID(i + 1, jj)]);
        int j2 = j == ny + 1 ? ny : j;                    // uv(2:nx,1:ny), copy to ny+1
        if (i >= 2 && i <= nx) uv = (real)0.25 * (d.uu[ID(i - 1, j2)] + d.uu[ID(i, j2)] + d.uu[ID(i - 1, j2 + 1)] + d.uu[ID(i, j2 + 1)]);
    } else {
        if (i <= nx) vu = (real)0.5 * (d.vv[ID(i, 1)] + d.vv[ID(i + 1, 1)]);
        if (i >= 2 && i <= nx) uv = (real)0.5 * (d.uu[ID(i - 1, 1)] + d.uu[ID(i, 1)]);
    }
    vu *= (real)d.wetu[k];
    uv *= (real)d.wetv[k];
    d.vu[k] = vu; d.uv[k] = uv;
    d.ueu[k] = d.uu[k]; d.vev[k] = d.vv[k];
    d.vmageu[k] = sqrt(d.uu[k] * d.uu[k] + vu * vu);
    d.vmagev[k] = sqrt(uv * uv + d.vv[k] * d.vv[k]);
    real hh = fmax(d.zs[k] - d.zb[k], c.eps);
    d.hh[k] = hh;
    // maxzs/minzs of wet points (flow_timestep end); dFill marks never wet
    if (d.maxzs && d.wetz[k] == 1) {
        if (d.maxzs[k] > DFILL) { d.maxzs[k] = fmax(d.zs[k], d.maxzs[k]); d.minzs[k] = fmin(d.zs[k], d.minzs[k]); }
        else { d.maxzs[k] = d.zs[k]; d.minzs[k] = d.zs[k]; }
    }
}
// flow_timestep end: ve(:,2:ny) = mean of vev, ve(:,1) = vev(:,1) (ve(:,ny+1) never set), from the vev of the
// previous flow step, because vev is updated further down in flow_timestep
__global__ void k_ve_centre() {
    IJ
    if (c.oned) { d.ve[ID(i, 1)] = d.vev[ID(i, 1)]; return; }
    if (j == c.NY1) return;
    d.ve[ID(i, j)] = j == 1 ? d.vev[ID(i, 1)] : (real)0.5 * (d.vev[ID(i, j - 1)] + d.vev[ID(i, j)]);
}
// cell-centre velocities (flow_timestep end): u(2:nx) = mean of uu, u(1) = uu(1), u(nx+1) = u(nx);
// v(:,2:ny) = mean of vv, v(:,1) = vv(:,1), v(:,ny+1) = v(:,ny), v(nx+1,:) = v(nx,:); 1D: v = vv
__global__ void k_uv_centre() {
    IJ
    size_t k = ID(i, j);
    int nx = c.nx, ny = c.ny;
    int ii = i == nx + 1 ? nx : i;
    d.u[k] = ii == 1 ? d.uu[ID(1, j)] : (real)0.5 * (d.uu[ID(ii - 1, j)] + d.uu[ID(ii, j)]);
    if (c.oned) { d.v[k] = d.vv[k]; return; }
    int jj = j == ny + 1 ? ny : j;
    d.v[k] = jj == 1 ? d.vv[ID(ii, 1)] : (real)0.5 * (d.vv[ID(ii, jj - 1)] + d.vv[ID(ii, jj)]);
}
