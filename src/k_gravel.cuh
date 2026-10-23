// k_gravel.cuh: morphevolution.F90 Nielsen2006 and mccall_vanrijn (bed load at u points only, as XBeach), the
// gravel formulations of XBeach-G (form = nielsen2006, mccall_vanrijn, mpm, wong_parker, fl_vb, wilcock_crow,
// engelund_fredsoe, fredsoe_deigaard). With these forms transus returns after the bed load (sus = bulk = 0).
#pragma once

struct GravConst {
    int sedfricfac, phaselag, streaming, slopecorr, incldzdx, inclrelweight, gwflow, lwt;
    real Ctrans, phirad, Arms, fsed, uprushfac, backwashfac, yturb, facthr, reposerad, reposedzdx;
    real shieldscrit[NGMAX], dstar[NGMAX];
};
__constant__ GravConst gc;
struct GravDev { real *ulocal, *ulocalold, *umeanupd, *uvarupd, *dudts, *dcfl, *Arms, *qtmp; };
__constant__ GravDev gd;

__device__ __forceinline__ real sgn1(real x) { return copysign((real)1, x); }   // Fortran sign(1.d0, x)

// slope corrections of the Shields number (slopecorr = nielsen or fredsoe_deigaard), at a wet u point
__device__ __forceinline__ real slopecorr(real shields, real dzbdx, real dzf, real sgn) {
    const real rr = gc.reposerad, rd = gc.reposedzdx;
    if (gc.slopecorr == 1) {
        if (dzbdx > 0) shields = fmax(shields / cos(fmin(atan(dzf), rr)) - fmin(dzf, rd) * sgn, (real)0);
        else if (dzbdx < 0) shields = fmax(shields / cos(fmax(atan(dzf), -rr)) - fmax(dzf, -rd) * sgn, (real)0);
    } else if (gc.slopecorr == 2) {
        if (dzbdx > 0) shields = shields * cos(atan(fmin(dzf, rr))) * fmax(1 - sgn * fmin(dzf / rd, (real)1), (real)0);
        else if (dzbdx < 0) shields = shields * cos(atan(fmax(dzf, -rr))) * fmax(1 - sgn * fmax(dzf / rd, (real)-1), (real)0);
    }
    return shields;
}

// Nielsen2006: smoothed velocity, acceleration and velocity variance (once per step, before the fractions)
__global__ void k_niel_state(real dt) {
    IJ
    size_t k = ID(i, j);
    real Trep = cs.Trep;
    real f = fmin(dt / (Trep / 20), (real)1);
    real ul = (1 - f) * gd.ulocal[k] + f * d.uu[k];
    gd.ulocal[k] = ul;
    gd.dudts[k] = d.wetu[k] == 1 ? (ul - gd.ulocalold[k]) / dt : 0;
    gd.ulocalold[k] = ul;
    if (gc.lwt == 1 && gc.yturb > 0) {
        real dcfinl = exp(fmin((real)100, d.hh[k] / fmax(gc.facthr * sd.rolthick[k], (real)0.01)));
        real dcfl = fmin((real)1, 1 / (dcfinl - 1));
        gd.dcfl[k] = fmax((real)0, dcfl);
    }
    f = fmin(dt / (Trep * 5), (real)1);
    real um = (1 - f) * gd.umeanupd[k] + f * ul;
    gd.umeanupd[k] = um;
    gd.uvarupd[k] = (1 - f) * gd.uvarupd[k] + f * ((ul - um) * (ul - um));
    if (gc.sedfricfac == 2) gd.Arms[k] = sqrt((real)2) / (2 * c.px / Trep) * sqrt(gd.uvarupd[k]);
}
__global__ void k_niel_q(int jg) {
    IJ
    size_t k = ID(i, j);
    const real D50 = cs.D50[jg], delta = cs.delta, Trep = cs.Trep, omegap = 2 * c.px / Trep;
    bool wu = d.wetu[k] == 1;
    real fsed;
    switch (gc.sedfricfac) {
    case 2: fsed = wu ? exp((real)5.5 * pow((real)2.5 * D50 / gd.Arms[k], (real)0.2) - (real)6.3) : 0;
            fsed = fmin(fsed, F32(0.05)); fsed = fmax(fsed, (real)0); break;   // max(fsed, s%cf) with s%cf = 0
    case 3: fsed = wu ? exp((real)5.213 * pow((real)2.5 / gc.Arms * D50, F32(0.194)) - F32(5.977)) : 0; break;
    case 1: fsed = 0; break;                                                  // s%cf = 0
    case 4: fsed = (real)0.114 * (gc.Arms / (c.g * delta * (Trep * Trep))) * (real)0.4; break;
    default: fsed = gc.fsed;
    }
    real ustar = 0;
    if (wu) ustar = gc.phaselag == 1 ? sqrt(fsed / 2) * (cos(gc.phirad) * gd.ulocal[k] + sin(gc.phirad) / omegap * gd.dudts[k])
                                     : sqrt(fsed / 2) * gd.ulocal[k];
    if (gc.lwt == 1 && gc.yturb > 0 && wu) ustar = ustar + sqrt(fsed / 2) * (2 * sqrt(sd.kturb[k]) * gd.dcfl[k] * gc.yturb) * sgn1(ustar);
    real shields = wu ? ustar * ustar / (delta * c.g * D50) : 0;
    if (wu) shields = slopecorr(shields, sd.dzbdx[k], sd.dzbdx[k], sgn1(ustar));
    if (gc.streaming == 1 && wu) {
        real um = fabs(gd.umeanupd[k]), uv = gd.uvarupd[k], Ar = gd.Arms[k], fe;
        real ex = exp((real)5.5 * pow(fmin(170 * sqrt(fmax(shields - (real)0.05, (real)0)) * D50, d.hh[k]) / Ar, F32(0.2)) - (real)6.3);
        if (uv > 1 * um) fe = ex;
        else if (uv < (real)0.25 * um) fe = 0;
        else fe = ((uv / um) - (real)0.25) / (real)0.75 * ex;
        shields = shields + ((real)2 / 3 / c.px * fe * (Ar * Ar * Ar) * (omegap * omegap * omegap) / (c.g * d.hh[k])) / (delta * c.g * D50) * sgn1(ustar);
        shields = fmax(shields, (real)0);
    }
    real th = gc.shieldscrit[jg], q = 0;
    if (wu && shields > th) q = (gc.Ctrans * (shields - th) * sqrt(shields)) * sqrt(delta * c.g * (D50 * D50 * D50)) * sgn1(ustar);
    if (q > 0) q = fmin(q, (1 - (real)0.60) * fabs(d.qx[k]));
    else if (q < 0) q = fmax(q, (1 - (real)0.60) * (-fabs(d.qx[k])));
    gd.qtmp[k] = q;
}
// mccall_vanrijn and the other gravel formulas: bed shear stress from the flow (taubx)
__global__ void k_mccall_q(int jg, int form) {
    IJ
    size_t k = ID(i, j);
    const real D50 = cs.D50[jg], delta = cs.delta, rho = c.rho, g = c.g;
    bool wu = d.wetu[k] == 1;
    real facrw = (gc.gwflow == 1 && gc.inclrelweight == 1) ? (real)0.5 * d.infil[k] / d.gKzinf[k] : 0;
    real tb = d.taubx[k] - gc.incldzdx * d.dzsdx[k] * rho * g * D50;
    real shields = 0, sg = 0;
    if (wu) { shields = fabs(tb) / (rho * g * D50 * fmax(delta + facrw, F32(1e-6))); sg = sgn1(tb); }
    if (wu) shields = slopecorr(shields, sd.dzbdx[k], sd.dzbdx[k], sg);
    real th = gc.shieldscrit[jg], q = 0, gD3 = sqrt(delta * g * (D50 * D50 * D50));
    switch (form) {
    case 7: if (wu && shields > th) q = (8 * pow(shields - th, (real)1.5)) * gD3 * sg; break;
    case 8: if (wu && shields > th) q = ((real)3.97 * pow(shields - th, (real)1.5)) * gD3 * sg; break;
    case 9: if (wu && shields > th) q = ((real)5.7 * pow(shields - th, (real)1.5)) * gD3 * sg; break;
    case 5:
        if (wu) {
            real ph = shields / th;
            if (ph >= (real)1.35) q = 14 * pow(1 - (real)0.894 / sqrt(ph), (real)4.5) * pow(shields, (real)1.5) * sqrt(delta * g) * pow(D50, (real)1.5) * sg;
            else q = (real)0.002 * pow(ph, (real)7.5) * pow(shields, (real)1.5) * sqrt(delta * g) * pow(D50, (real)1.5) * sg;
        }
        break;
    case 6:
        if (wu && shields > th) {
            real x = (real)0.5236 * (real)0.5 / (shields - th), x2 = x * x, x4 = x2 * x2;   // gfortran x**4 = (x*x)*(x*x)
            real nEF = pow(1 + x4, (real)-0.25);
            q = 5 * nEF * (sqrt(shields) - (real)0.7 * sqrt(th)) * gD3 * sg;
        }
        break;
    case 10: if (wu && shields > th) q = (real)14.6912 * (shields - th) * (sqrt(shields) - (real)0.7 * sqrt(th)) * gD3 * sg; break;
    default:   // mccall_vanrijn (Ctrans = 0.5)
        if (wu && shields > th) q = (real)0.5 * D50 * pow(gc.dstar[jg], (real)-0.3) * sqrt(fabs(d.taubx[k]) / rho) * ((shields - th) / th) * sg;
    }
    q = q > 0 ? q * gc.uprushfac : q * gc.backwashfac;
    if (q > 0) q = fmin(q, (1 - (real)0.60) * fabs(d.qx[k]));
    else if (q < 0) q = fmax(q, (1 - (real)0.60) * (-fabs(d.qx[k])));
    gd.qtmp[k] = q;
}
// Subg: smoothing over three u points when thetanum < 1 (Subg(1,:) takes qsedu(nx+1,:), the stale loop index of
// XBeach); mccall group: weighted by the top-layer fraction of the upwind cell
__global__ void k_grav_sub(int jg, int weight) {
    IJ
    size_t k = ID(i, j);
    int nx = c.nx;
    const real *q = gd.qtmp;
    real s;
    if (cs.thetanum < 1) {
        if (i >= 2 && i <= nx) s = (1 - cs.thetanum) / 2 * (q[ID(i - 1, j)] + q[ID(i + 1, j)]) + cs.thetanum * q[k];
        else s = q[ID(nx + 1, j)];
    } else s = q[k];
    if (weight) s = s > 0 ? s * cs.sedcal[jg] * PB(k, 0, jg) : s * cs.sedcal[jg] * PB(ID(min(i + 1, nx + 1), j), 0, jg);
    G3(sd.Subg, k, jg) = s;
}

// intra_sedtr (form = intrasedtr): intrawave bed load from the instantaneous velocity (taubxsed = cf_transport rho
// u|u|) and an equilibrium suspended concentration from a Rouse profile. XBeach never sets Tsg for this form, so the
// suspended concentration of transus stays zero; only the bed load moves the bed.
struct IntraDev { real *taubxsed, *taubysed, *taubx_turb, *rouse, *ca_nonh, *c1mean; };
__constant__ IntraDev idv;
struct IntraConst { real cf_transport, nonh_za, ws_nonh, yturb, facthr; int lwt, bed, sus, slopecorr;
                    real dstar[NGMAX], taucr[NGMAX], ws[NGMAX], za[NGMAX], reposerad, reposedzd; };
__constant__ IntraConst ic_;

__global__ void k_intra1(int jg) {
    IJ
    size_t k = ID(i, j);
    const real rho = c.rho, g = c.g, D50 = cs.D50[jg], rr = ic_.reposerad, rd = ic_.reposedzd;
    const real dfac = (cs.rhos - rho) * g * D50, taucr = ic_.taucr[jg];
    real u = d.uu[k], v = d.vv[k];
    real tx = d.wetu[k] == 1 ? ic_.cf_transport * rho * u * fabs(u) : 0;
    real ty = d.wetv[k] == 1 ? ic_.cf_transport * rho * v * fabs(v) : 0;
    real tturb = 0;
    if (ic_.lwt == 1) {
        real dcfl;
        if (ic_.yturb == -1) {
            real dcfinl = exp(fmin((real)100, d.hh[k] / fmax(ic_.facthr * sd.rolthick[k], (real)0.01)));
            dcfl = fmax((real)0, fmin((real)1, 1 / (dcfinl - 1)));
        } else dcfl = ic_.yturb;
        tturb = ic_.cf_transport * (sd.kturb[k] * rho * dcfl);
        if (d.wetu[k] == 1) tx = tx + tturb * sgn1(tx);
    }
    if (jg == 0) { idv.taubxsed[k] = tx; idv.taubysed[k] = ty; idv.taubx_turb[k] = ic_.lwt == 1 ? tturb : idv.taubx_turb[k]; }
    // bed load, x
    if (ic_.bed == 1) {
        real sh = tx / dfac, sg = sgn1(sh);
        if (d.wetu[k] == 1) {
            real dz = sd.dzbdx[k];
            if (ic_.slopecorr == 1) {
                if (dz > 0) sh = fmax(sh / cos(fmin(atan(dz), rr)) - fmin(dz, rd) * sg, (real)0);
                else if (dz < 0) sh = fmax(sh / cos(fmax(atan(dz), -rr)) - fmax(dz, -rd) * sg, (real)0);
            } else if (ic_.slopecorr == 2) {
                if (dz > 0) sh = sh * cos(atan(fmin(dz, rr))) * fmax(1 - sg * fmin(dz / rd, (real)1), (real)0);
                else if (dz < 0) sh = sh * cos(atan(fmax(dz, -rr))) * fmax(1 - sg * fmax(dz / rd, (real)-1), (real)0);
            }
        }
        real tb = sh * dfac; sg = sgn1(tb);
        real Sb = (real)d.wetu[k] * sg * (real)0.5 * 1 * D50 * pow(ic_.dstar[jg], (real)-0.3) * pow(fabs(tb) / rho, (real)0.5)
                  * pow((fabs(tb) - taucr) / taucr, (real)1);
        G3(sd.ceqbg, k, jg) = Sb / (d.hh[k] * d.uu[k]);
        if (fabs(tb) <= taucr) Sb = 0;
        G3(sd.Subg, k, jg) = Sb;
    } else G3(sd.Subg, k, jg) = 0;
    // bed load, y
    if (ic_.bed == 1 && !c.oned) {
        real sh = ty / dfac, sg = sgn1(sh);
        if (d.wetv[k] == 1) {
            real dz = sd.dzbdy[k];
            if (ic_.slopecorr == 1) {   // XBeach tests dzbdx in the second branch
                if (dz > 0) sh = fmax(sh / cos(fmin(atan(dz), rr)) - fmin(dz, rd) * sg, (real)0);
                else if (sd.dzbdx[k] < 0) sh = fmax(sh / cos(fmax(atan(dz), -rr)) - fmax(dz, -rd) * sg, (real)0);
            } else if (ic_.slopecorr == 2) {
                if (dz > 0) sh = sh * cos(atan(fmin(dz, rr))) * fmax(1 - sg * fmin(dz / rd, (real)1), (real)0);
                else if (dz < 0) sh = sh * cos(atan(fmax(dz, -rr))) * fmax(1 - sg * fmax(dz / rd, (real)-1), (real)0);
            }
        }
        real tb = sh * dfac; sg = sgn1(tb);
        real Sb = (real)d.wetv[k] * sg * (real)0.5 * 1 * D50 * pow(ic_.dstar[jg], (real)-0.3) * pow(fabs(tb) / rho, (real)0.5)
                  * pow((fabs(tb) - taucr) / taucr, (real)1);
        if (fabs(tb) <= taucr) Sb = 0;
        G3(sd.Svbg, k, jg) = Sb;
    } else G3(sd.Svbg, k, jg) = 0;
    // suspended equilibrium concentration (Rouse profile)
    if (ic_.sus == 1) {
        real za = ic_.za[jg];
        real tt = sqrt(ty * ty + tx * tx);
        real tturb2 = idv.taubx_turb[k];
        real taurel = (fabs(tt) + tturb2 - taucr) / taucr;
        if (fabs(tt) <= taucr) taurel = 0;
        real ca = (real)0.015 * (1 - 0) * 1 * D50 / za * taurel * sqrt(taurel) / pow(ic_.dstar[jg], (real)0.3);
        if (ca >= (real)0.05) ca = 0.05;
        G3(idv.ca_nonh, k, jg) = ca;
        real ust = fmax(sqrt(sqrt(ty * ty + tx * tx) / rho), (real)2.2250738585072014e-308);
        real rouse = ic_.ws[jg] / ((real)0.4 * (ust + sqrt(tturb2 / rho)));
        if (fabs(rouse) >= 5) rouse = 5;
        if (fabs(rouse - 1) < (real)1e-12) rouse = 1 + (real)1e-12;
        idv.rouse[k] = rouse;
        if (IN_ZS) {
            real hh = d.hh[k];
            idv.c1mean[k] = -ca * ((za - hh * pow(hh / za, -1 * rouse)) / (-1 * rouse + 1)) / (hh - za);
        }
    }
}
__global__ void k_intra2(int jg) {   // c1mean boundaries (nearest inner cell), limit, depth-averaged ceqsg
    IJ
    if (ic_.sus != 1) return;
    int nx = c.nx, ny = c.ny;
    size_t k = ID(i, j);
    int ii = i == 1 ? 2 : (i == nx + 1 ? nx : i), jj = c.oned ? 1 : (j == 1 ? 2 : (j == ny + 1 ? ny : j));
    real hh = d.hh[k], c1 = idv.c1mean[ID(ii, jj)];
    if (c1 >= cs.cmax * hh) c1 = cs.cmax * hh;
    real za = ic_.za[jg];
    G3(sd.ceqsg, k, jg) = (za * G3(idv.ca_nonh, k, jg) + (hh - za) * c1) / hh;
}
