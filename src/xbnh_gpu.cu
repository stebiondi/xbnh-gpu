// xbnh_gpu: GPU (CUDA) port of the XBeach one-layer non-hydrostatic flow solver.
//
// Port of XBeach trunk r6167 (Deltares), routines followed line by line:
//   wetcells.F90          compute_wetcells
//   timestep.F90          compute_dt, timestep
//   boundaryconditions.F90 flow_bc (front nonh_1d with ARC=1, back wall/abs_1d/abs_2d, lateral neumann
//                         or cyclic), flow_lat_bc (neumann), wave_bc + velocity_Boundary (wbctype = ts_nonh
//                         or reuse; scalar or vector files, V optional)
//   compute_tide_zs0.F90  tide_init, tide_boundary_timestep (tideloc = 0/1, tidetype = velocity)
//   readwind.F90          stationary or time-varying wind (windfile)
//   initialize.F90        flow_init, hotstart_init_1/2 (ASCII hotstart files)
//   flow_timestep.F90     flow, visc_smagorinsky
//   bedroughness.F90      manning
//   nonh.F90              nonh_1lay_pred, nonh_break (nhbreaker = 1), nonh_1lay_cor, zuzv, nonh_init_wcoef
//   flow_secondorder.F90  flow_secondorder_advUV, flow_secondorder_advW, flow_secondorder_huhv
//   postprocess.F90       snappointstogrid (point output, outputformat = fortran)
// Scope: wavemodel = nonh, nonhq3d = 0, rectilinear uniform grid (vardx = 0, alfa = 0), ny > 4,
// no short-wave/sediment/morphology.
// Cyclic laterals (cyclic = 1): XBeach runs these only under MPI, where rows 1-2 and ny-ny+1 are halos
// filled from the opposite side (period ny-3: row 1 <- ny-2, 2 <- ny-1, ny <- 3, ny+1 <- 4). Here the
// whole domain is one block; the same halo copies are made after every update of a prognostic field,
// including uu/vv after the pressure correction, which XBeach MPI does not exchange before continuity.
// The pressure Poisson system is solved on rows 3..ny-1 with periodic coupling.
// Intentional numerical differences: the pressure solver (XBeach SIP, an ILU sweep that is inherently
// sequential; here Jacobi-preconditioned BiCGStab on the same 5-diagonal system, gpu_solver_acc) and the
// single-domain periodic halo described above.
//
// Arrays use the XBeach layout: index (i,j), i = 1..nx+1 fastest (cross-shore), j = 1..ny+1.
// Build: nvcc -O3 -std=c++20 -arch=native [-DXB_SINGLE] -o xbnh_gpu xbnh_gpu.cu  (compute capability >= 6.0)
// Run  : xbnh_gpu   (in a directory holding an XBeach params.txt and its input files)
// Output (outputformat = fortran, double): zs.dat, u.dat, v.dat ((nx+1)*(ny+1) float64 per tintg) when
// nglobalvar > 0; one <name>.dat per output point ([t, vars] float64 per tintp); xbgpu_log.txt (timings).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

#ifdef XB_SINGLE
typedef float real;
#define NUMEPS 1.1920929e-07f
#else
typedef double real;
#define NUMEPS 2.220446049250313e-16
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

enum { BACK_WALL = 0, BACK_ABS_1D = 1, BACK_ABS_2D = 2 };
enum { LR_NEUMANN = 0, LR_WALL = 1 };

struct Const {
    int nx, ny, NX1, NY1;
    real dx, dy, g, rho, eps, eps_sd, umin;
    real nman, maxcf, mincf, nuh, nuhv, breakviscfac, maxbrsteep, reformsteep, maxfacg;
    real rhoa, Cd, theta0;
    int smag, viscosity, secorder, back, left, right;
    int cyclic, isSetV, isSetZ;
    int jlo;                       // first row already updated in XBeach's (j,i) sweeps (2; 3 with MPI halos)
    int uv_jmin, uv_jmax;          // flow_secondorder_advUV jmin/jmax
    int w_jmin, w_jmax;            // flow_secondorder_advW and _huhv jmin/jmax
    int ps_j0, ps_j1;              // rows of the pressure unknowns
};
__constant__ Const c;

#define ID(i, j) ((size_t)((j) - 1) * c.NX1 + ((i) - 1))
#define IJ int i = blockIdx.x * blockDim.x + threadIdx.x + 1; \
           int j = blockIdx.y * blockDim.y + threadIdx.y + 1; \
           if (i > c.NX1 || j > c.NY1) return;
#define IN(i0, i1, j0, j1) (i >= (i0) && i <= (i1) && j >= (j0) && j <= (j1))

__device__ __forceinline__ real minmod(real d1, real d2) {
    if (d1 * d2 <= 0) return 0;
    if (d1 > 0) return fmin(d1, d2);
    if (d1 < 0) return fmax(d1, d2);
    return 0;
}

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------
struct Dev {
    real *zb, *zs, *hh, *uu, *vv, *hu, *hv, *hum, *hvm, *qx, *qy;
    int *wetu, *wetv, *wetz;
    real *dzsdx, *dzsdy, *ududx, *vdudy, *vdvdy, *udvdx, *nuh, *viscu, *viscv, *taubx, *tauby, *cfu, *cfv;
    real *vu, *uv, *ueu, *vev, *vmageu, *vmagev, *uu_old, *vv_old;
    int *breaking, *brk_new, *nonhU, *nonhV, *nonhZ;
    real *pres, *ws, *wb, *Wm, *Wm_old, *wcoef, *au0, *au1, *av0, *av1;
    real *awb[5], *awbr, *aws[5], *awsr, *mat[5], *rhs, *dp;
    real *zbu, *zbv, *zsu, *zsv, *w1, *w2, *w3, *w4;
    real *r, *rh, *p, *v, *s, *t, *y, *z;     // BiCGStab
    double *sc;                                 // solver scalars
    unsigned long long *dtmin;
    real *zs0;                                  // tide level field (compute_tide_zs0)
    real *ui, *vi, *zi;                         // offshore boundary signal, length ny+1
    real *um1, *vm1, *umN, *vmN;                // umean(1,:), vmean(1,:), umean(nx,:), vmean(nx,:)
    double *msum;                               // boundary-mean u, v: [0,1] front, [2,3] back
    unsigned long long *skey;                   // last NH / non-NH point in XBeach loop order
    double *stale;                              // stale loop variables of nonh_1lay_cor: dzb_s, dzs_e
};

// ---------------------------------------------------------------------------------------------
// wetcells.F90 : compute_wetcells
// ---------------------------------------------------------------------------------------------
__global__ void k_wetcells(Dev d) {
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
// timestep.F90 : compute_dt (ny > 2 branch). Positive reals order like their unsigned bit patterns.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void atomic_min_pos(unsigned long long *a, double v) {
    atomicMin(a, (unsigned long long)__double_as_longlong(v));
}
__global__ void k_dt(Dev d) {
    __shared__ double sm[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    double dt = 1e300;
    if (i >= 2 && i <= c.nx && j >= 2 && j <= c.ny) {
        size_t k = ID(i, j);
        const double tny = 2.2250738585072014e-308;
        double g = c.g, dx = c.dx, dy = c.dy;
        if (d.wetu[k] == 1) {
            double cu = sqrt(g * d.hu[k]);
            dt = fmin(dt, dx / fmax(tny, fmax(cu + fabs((double)d.uu[k]), fabs((double)d.ueu[k]))));
            dt = fmin(dt, dy / fmax(tny, cu + fabs((double)d.vu[k])));
        }
        if (d.wetv[k] == 1) {
            double cv = sqrt(g * d.hv[k]);
            dt = fmin(dt, dx / fmax(tny, cv + fabs((double)d.uv[k])));
            dt = fmin(dt, dy / fmax(tny, fmax(cv + fabs((double)d.vv[k]), fabs((double)d.vev[k]))));
        }
        if (d.wetz[k] == 1) {
            double mdx = dx * dx, mdy = dy * dy;
            dt = fmin(dt, 0.5 * mdx * mdy / (mdx + mdy) / fmax((double)d.nuh[k], 1e-6));
        }
    }
    sm[tid] = dt;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) sm[tid] = fmin(sm[tid], sm[tid + s]); __syncthreads(); }
    if (tid == 0) atomic_min_pos(d.dtmin, sm[0]);
}

// ---------------------------------------------------------------------------------------------
// boundaryconditions.F90 : flow_bc
// ---------------------------------------------------------------------------------------------
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

// compute_tide_zs0.F90 : boundaryinterp_tide (tideloc 0/1) with correctforwetz, called from flow_bc
__global__ void k_tide(Dev d, real zs01) {
    IJ
    size_t k = ID(i, j);
    int nx = c.nx, ny = c.ny;
    real z = d.zs0[k];
    if (i <= 2 || i >= nx) z = zs01;                                  // istop: zs0(1:2,:), isbot: zs0(nx:nx+1,:)
    if (!c.cyclic && (j == 1 || j == ny + 1)) z = zs01;               // isleft / isright
    if (d.wetz[k] == 0) z = d.zb[k];
    if (j == 1 && d.wetz[ID(i, 2)] == 0) z = fmin(d.zb[ID(i, 1)], d.zb[ID(i, 2)]);
    if (j == ny + 1 && d.wetz[ID(i, ny)] == 0) z = fmin(d.zb[ID(i, ny + 1)], d.zb[ID(i, ny)]);
    if (i == 1 && d.wetz[ID(2, j)] == 0) z = fmin(d.zb[ID(1, j)], d.zb[ID(2, j)]);
    if (i == nx + 1 && d.wetz[ID(nx, j)] == 0) z = fmin(d.zb[ID(nx, j)], d.zb[ID(nx + 1, j)]);
    d.zs0[k] = z;
}

// umean relaxation (tidetype = velocity): boundary-mean u and v (single block), then the row update.
// Serial bounds jmin_uu..jmax_uu = 2..ny, jmin_vv..jmax_vv = 2..ny-1; cyclic (MPI, no isleft/isright):
// every rank sums its rows 3..ny_local-1, i.e. the unique global rows 3..ny-1.
__global__ void k_bc_msum(Dev d, int side) {
    __shared__ double su[1024], sv[1024];
    int tid = threadIdx.x;
    int ib = side == 0 ? 1 : c.nx;
    int ju0 = c.cyclic ? 3 : 2, ju1 = c.cyclic ? c.ny - 1 : c.ny;
    int jv0 = c.cyclic ? 3 : 2, jv1 = c.ny - 1;
    double a = 0, b = 0;
    for (int j = ju0 + tid; j <= ju1; j += blockDim.x) a += d.uu[ID(ib, j)] * c.dy;
    for (int j = jv0 + tid; j <= jv1; j += blockDim.x) b += d.vv[ID(ib, j)] * c.dy;
    su[tid] = a; sv[tid] = b;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) { su[tid] += su[tid + s]; sv[tid] += sv[tid + s]; } __syncthreads(); }
    if (tid == 0) {
        d.msum[2 * side]     = su[0] / ((ju1 - ju0 + 1) * (double)c.dy);
        d.msum[2 * side + 1] = sv[0] / ((jv1 - jv0 + 1) * (double)c.dy);
    }
}
__global__ void k_bc_mapply(Dev d, real factime, int side) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    real *um = side == 0 ? d.um1 : d.umN, *vm = side == 0 ? d.vm1 : d.vmN;
    um[j - 1] = factime * (real)d.msum[2 * side] + (1 - factime) * um[j - 1];
    vm[j - 1] = factime * (real)d.msum[2 * side + 1] + (1 - factime) * vm[j - 1];
}
// front = nonh_1d, ARC = 1 (radiating boundary for short waves)
__global__ void k_bc_front(Dev d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j < 2 || j > c.ny) return;
    size_t k1 = ID(1, j), k2 = ID(2, j);
    real zi = c.isSetZ ? d.zi[j - 1] : d.zs[k2];
    d.uu[k1] = d.ui[j - 1] - sqrt(c.g / d.hh[k1]) * (d.zs[k2] - zi - d.zs0[k2]) + d.um1[j - 1];
    d.vv[k1] = c.isSetV ? d.vi[j - 1] : d.vv[k2];
    d.zs[k1] = d.zs[k2];
    d.ws[k1] = d.ws[k2];
}
__global__ void k_bc_front_corners(Dev d) {
    if (threadIdx.x != 0) return;
    d.uu[ID(1, 1)] = d.uu[ID(1, 2)];
    d.vv[ID(1, 1)] = d.vv[ID(1, 2)];
    d.zs[ID(1, 1)] = fmax(d.zs[ID(1, 2)], d.zb[ID(1, 1)]);
    d.uu[ID(1, c.ny + 1)] = d.uu[ID(1, c.ny)];
    d.vv[ID(1, c.ny + 1)] = d.vv[ID(1, c.ny)];
    d.zs[ID(1, c.ny + 1)] = fmax(d.zs[ID(1, c.ny)], d.zb[ID(1, c.ny + 1)]);
}
__global__ void k_bc_back1(Dev d, real dt) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    int nx = c.nx;
    if (c.back == BACK_WALL) {
        if (j >= 2 && j <= c.ny) { d.uu[ID(nx, j)] = 0; d.zs[ID(nx + 1, j)] = d.zs[ID(nx, j)]; }
    } else if (c.back == BACK_ABS_1D) {
        if (j >= 2 && j <= c.ny)
            d.uu[ID(nx, j)] = sqrt(c.g / d.hh[ID(nx, j)]) * (d.zs[ID(nx, j)] - fmax(d.zb[ID(nx, j)], d.zs0[ID(nx, j)])) + d.umN[j - 1];
        d.zs[ID(nx + 1, j)] = d.zs[ID(nx, j)];
    } else {  // abs_2d: Van Dongeren (1997) weakly reflective boundary, swave = 0 (Fx = 0)
        if (j < 2 || j > c.ny || d.wetu[ID(nx, j)] != 1) return;
        real g = c.g;
        real ht1 = fmax(d.zs0[ID(nx, j)] - d.zb[ID(nx, j)], c.eps), ht2 = fmax(d.zs0[ID(nx + 1, j)] - d.zb[ID(nx + 1, j)], c.eps);
        auto beta1 = [&](int jj) { return d.uu[ID(nx - 1, jj)] + 2 * sqrt(g * d.hum[ID(nx - 1, jj)]); };
        real hum = d.hum[ID(nx, j)], u = d.uu[ID(nx, j)], vu = d.vu[ID(nx, j)];
        real b1 = beta1(j), b2 = u + 2 * sqrt(g * hum);
        real dvdy = (d.vu[ID(nx, j + 1)] - d.vu[ID(nx, j - 1)]) / (2 * c.dy);
        real dhdx = (ht2 - ht1) / c.dx;
        real dbetadx = (b2 - b1) / c.dx;
        real dbetady = (beta1(j + 1) - beta1(j - 1)) / (2 * c.dy);
        real bn = -(u + sqrt(g * hum)) * dbetadx - vu * dbetady - sqrt(g * hum) * dvdy
                  - d.cfu[ID(nx, j)] * sqrt(u * u + vu * vu) * u / hum + g * dhdx;
        real bnp1 = b2 + bn * dt;
        real a2 = c.theta0, ur = 0, umn = d.umN[j - 1];
        const real px = (real)M_PI;
        for (int it = 0; it < 50; it++) {
            ur = cos(a2) / (cos(a2) + 1) * (bnp1 - umn - 2 * sqrt(g * (real)0.5 * (ht1 + ht2)));
            real vert = vu - d.vmN[j - 1];
            real an = atan(vert / (ur + (real)1e-16));
            if (an > px * (real)0.5) an -= px;
            if (an <= -px * (real)0.5) an += px;
            if (fabs(an - a2) < (real)0.001) break;
            a2 = an;
        }
        real unew = ur + umn;
        d.uu[ID(nx, j)] = unew;
        d.zs[ID(nx + 1, j)] = (real)1.5 * ((bnp1 - unew) * (bnp1 - unew) / 4 / g + (real)0.5 * (d.zb[ID(nx, j)] + d.zb[ID(nx + 1, j)]))
                            - (real)0.5 * ((b1 - d.uu[ID(nx - 1, j)]) * (b1 - d.uu[ID(nx - 1, j)]) / 4 / g
                                           + (real)0.5 * (d.zb[ID(nx - 1, j)] + d.zb[ID(nx, j)]));
    }
}
__global__ void k_bc_back2(Dev d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j < 2 || j > c.ny) return;
    int nx = c.nx;
    d.vv[ID(nx + 1, j)] = d.vv[ID(nx, j)];
    d.uu[ID(nx + 1, j)] = d.uu[ID(nx, j)];
}
__global__ void k_bc_back_corners(Dev d) {
    if (threadIdx.x != 0) return;
    int nx = c.nx, ny = c.ny;
    d.uu[ID(nx + 1, 1)] = d.uu[ID(nx + 1, 2)];
    d.vv[ID(nx + 1, 1)] = d.vv[ID(nx + 1, 2)];
    d.zs[ID(nx + 1, 1)] = fmax(d.zs[ID(nx + 1, 2)], d.zb[ID(1, 1)]);              // zb(1,1) as in XBeach
    d.uu[ID(nx + 1, ny + 1)] = d.uu[ID(nx + 1, ny)];
    d.vv[ID(nx + 1, ny + 1)] = d.vv[ID(nx + 1, ny)];
    d.zs[ID(nx + 1, ny + 1)] = fmax(d.zs[ID(nx + 1, ny)], d.zb[ID(1, ny + 1)]);   // zb(1,ny+1) as in XBeach
}
__global__ void k_bc_lat(Dev d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < 2 || i > c.nx) return;
    d.zs[ID(i, 1)] = fmax(d.zs[ID(i, 2)], d.zb[ID(i, 1)]);
    d.zs[ID(i, c.NY1)] = fmax(d.zs[ID(i, c.ny)], d.zb[ID(i, c.NY1)]);
}

// ---------------------------------------------------------------------------------------------
// flow_timestep.F90 : flow (first part, explicit momentum)
// ---------------------------------------------------------------------------------------------
__global__ void k_slopes_rough(Dev d) {
    IJ
    size_t k = ID(i, j);
    if (IN(2, c.nx - 1, 2, c.ny)) d.dzsdx[k] = (d.zs[ID(i + 1, j)] - d.zs[k]) / c.dx;
    if (IN(2, c.nx, 2, c.ny - 1)) d.dzsdy[k] = (d.zs[ID(i, j + 1)] - d.zs[k]) / c.dy;
    // bedroughness_update, manning
    real n2g = c.g * c.nman * c.nman;
    d.cfu[k] = d.wetu[k] == 1 ? fmax(fmin(n2g / cbrt(d.hu[k]), c.maxcf), c.mincf) : c.maxcf;
    d.cfv[k] = d.wetv[k] == 1 ? fmax(fmin(n2g / cbrt(d.hv[k]), c.maxcf), c.mincf) : c.maxcf;
}

// momentum-conserving advection (uniform grid: dalfa = 0, dnz*dsdnui = 1/dx, dsc*dsdnui = 1/dy)
__global__ void k_advection(Dev d) {
    IJ
    size_t k = ID(i, j);
    real esd = c.eps_sd;
    if (IN(2, c.nx - 1, 2, c.ny)) {
        real a = 0, qin;
        real ui = d.uu[k];
        qin = (real)0.5 * (d.qx[k] + d.qx[ID(i - 1, j)]);
        if (qin > 0) {
            real uw = d.uu[ID(i - 1, j)];
            if ((ui - uw) > esd) a += (real)0.5 * (uw + ui) * (ui - uw) / c.dx;
            else                 a += qin / d.hum[k] * (ui - uw) / c.dx;
        }
        qin = (real)-0.5 * (d.qx[k] + d.qx[ID(i + 1, j)]);
        if (qin > 0) {
            real ue = d.uu[ID(i + 1, j)];
            if ((ue - ui) > esd) a -= (real)0.5 * (ue + ui) * (ui - ue) / c.dx;
            else                 a += qin / d.hum[k] * (ui - ue) / c.dx;
        }
        d.ududx[k] = a;
    }
    // vdudy: XBeach loops this u-point term over the v-point bounds (jmin_vv..jmax_vv, imin_vv..imax_vv)
    if (IN(2, c.nx, 2, c.ny - 1)) {
        real a = 0, qin;
        real ui = d.uu[k];
        qin = (real)0.5 * (d.qy[ID(i, j - 1)] + d.qy[ID(i + 1, j - 1)]);
        if (qin > 0) {
            real us = d.uu[ID(i, j - 1)];
            if ((d.vv[k] - d.vv[ID(i, j - 1)]) > esd)
                a += (real)0.5 * (d.vv[ID(i, j - 1)] + d.vv[ID(i + 1, j - 1)]) * (ui - us) / c.dy;
            else a += qin / d.hum[k] * (ui - us) / c.dy;
        }
        qin = (real)-0.5 * (d.qy[k] + d.qy[ID(i + 1, j)]);
        if (qin > 0) {
            real un = d.uu[ID(i, j + 1)];
            if ((d.vv[ID(i, j + 1)] - d.vv[k]) > esd)
                a -= (real)0.5 * (d.vv[k] + d.vv[ID(i + 1, j)]) * (ui - un) / c.dy;
            else a += qin / d.hum[k] * (ui - un) / c.dy;
        }
        d.vdudy[k] = a;
    }
    // vdvdy, udvdx (zeroed outside their loops)
    real vd = 0, ud = 0;
    if (IN(2, c.nx, 2, c.ny - 1)) {
        real vi = d.vv[k], qin;
        qin = (real)0.5 * (d.qy[k] + d.qy[ID(i, j - 1)]);
        if (qin > 0) {
            real vs = d.vv[ID(i, j - 1)];
            if ((vi - vs) > esd) vd += (real)0.5 * (vs + vi) * (vi - vs) / c.dy;
            else                 vd += qin / d.hvm[k] * (vi - vs) / c.dy;
        }
        qin = (real)-0.5 * (d.qy[k] + d.qy[ID(i, j + 1)]);
        if (qin > 0) {
            real vn = d.vv[ID(i, j + 1)];
            if ((vn - vi) > esd) vd -= (real)0.5 * (vn + vi) * (vi - vn) / c.dy;
            else                 vd += qin / d.hvm[k] * (vi - vn) / c.dy;
        }
        qin = (real)0.5 * (d.qx[ID(i - 1, j)] + d.qx[ID(i - 1, j + 1)]);
        if (qin > 0) {
            real vw = d.vv[ID(i - 1, j)];
            if ((d.uu[k] - d.uu[ID(i - 1, j)]) > esd)
                ud += (real)0.5 * (d.uu[ID(i - 1, j)] + d.uu[ID(i - 1, j + 1)]) * (vi - vw) / c.dx;
            else ud += qin / d.hvm[k] * (vi - vw) / c.dx;
        }
        qin = (real)-0.5 * (d.qx[k] + d.qx[ID(i, j + 1)]);
        if (qin > 0) {
            real ve = d.vv[ID(i + 1, j)];
            if ((d.uu[ID(i + 1, j)] - d.uu[k]) > esd)
                ud -= (real)0.5 * (d.uu[k] + d.uu[ID(i, j + 1)]) * (vi - ve) / c.dx;
            else ud += qin / d.hvm[k] * (vi - ve) / c.dx;
        }
    }
    // lateral rows of vdvdy (xmpi_isleft / xmpi_isright)
    if (!c.cyclic && j == 1 && i >= 2 && i <= c.nx) {
        real qin = (real)-0.5 * (d.qy[ID(i, 1)] + d.qy[ID(i, 2)]);
        if (qin > 0) vd += qin * (d.vv[ID(i, 1)] - d.vv[ID(i, 2)]) * c.dx / d.hvm[k] / (c.dx * c.dy);
    }
    if (!c.cyclic && j == c.ny && i >= 2 && i <= c.nx) {
        real qin = (real)0.5 * (d.qy[ID(i, c.ny)] + d.qy[ID(i, c.ny - 1)]);
        if (qin > 0) vd += qin * (d.vv[ID(i, c.ny)] - d.vv[ID(i, c.ny - 1)]) * c.dx / d.hvm[k] / (c.dx * c.dy);
    }
    d.vdvdy[k] = vd;
    d.udvdx[k] = ud;
}

// visc_smagorinsky (interior), then boundary copies in k_nuh_bc
__global__ void k_smag(Dev d) {
    IJ
    if (!IN(2, c.nx, 2, c.ny)) return;
    size_t k = ID(i, j);
    real dudx = (d.uu[k] - d.uu[ID(i - 1, j)]) / c.dx;
    real dudy = (real)0.5 * (d.uu[ID(i, j + 1)] - d.uu[ID(i, j - 1)] + d.uu[ID(i - 1, j + 1)] - d.uu[ID(i - 1, j - 1)]) / (2 * c.dy);
    real dvdx = (real)0.5 * (d.vv[ID(i + 1, j)] - d.vv[ID(i - 1, j)] + d.vv[ID(i + 1, j - 1)] - d.vv[ID(i - 1, j - 1)]) / (2 * c.dx);
    real dvdy = (d.vv[k] - d.vv[ID(i, j - 1)]) / c.dy;
    real Tau = sqrt(2 * dudx * dudx + 2 * dvdy * dvdy + (dvdx + dudy) * (dvdx + dudy));
    real w = (real)(d.wetu[k] * d.wetu[ID(i - 1, j)] * d.wetv[k] * d.wetv[ID(i, j - 1)]);
    d.nuh[k] = c.nuh * c.nuh * c.dx * c.dy * Tau * w;
}
__global__ void k_nuh_bc_j(Dev d) {   // nuh(:,1) = nuh(:,2); nuh(:,ny+1) = nuh(:,ny)
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    d.nuh[ID(i, 1)] = d.nuh[ID(i, 2)];
    d.nuh[ID(i, c.NY1)] = d.nuh[ID(i, c.ny)];
}
__global__ void k_nuh_bc_i(Dev d) {   // nuh(1,:) = nuh(2,:); nuh(nx+1,:) = nuh(nx,:); then breaking factor
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    d.nuh[ID(1, j)] = d.nuh[ID(2, j)];
    d.nuh[ID(c.NX1, j)] = d.nuh[ID(c.nx, j)];
}
__global__ void k_nuh_const_brk(Dev d) {
    IJ
    size_t k = ID(i, j);
    if (!c.smag) d.nuh[k] = c.nuh;
    if (d.breaking[k] != 0) d.nuh[k] *= c.breakviscfac;      // nhbreaker = 1
}
__global__ void k_viscu(Dev d) {
    IJ
    if (!IN(2, c.nx - 1, 2, c.ny)) return;
    size_t k = ID(i, j);
    real sf = c.smag ? 2 : 1;
    real dudx1 = d.nuh[ID(i + 1, j)] * d.hh[ID(i + 1, j)] * (d.uu[ID(i + 1, j)] - d.uu[k]) / c.dx;
    real dudx2 = d.nuh[k] * d.hh[k] * (d.uu[k] - d.uu[ID(i - 1, j)]) / c.dx;
    real vis = sf * (1 / d.hum[k]) * (2 * (dudx1 - dudx2) / (2 * c.dx));
    int jp1 = min(j + 1, c.NY1), jm1 = max(j - 1, 1);
    real nuh1 = (real)0.25 * (d.nuh[k] + d.nuh[ID(i + 1, j)] + d.nuh[ID(i + 1, jp1)] + d.nuh[ID(i, jp1)]);
    real nuh2 = (real)0.25 * (d.nuh[k] + d.nuh[ID(i + 1, j)] + d.nuh[ID(i + 1, jm1)] + d.nuh[ID(i, jm1)]);
    real dudy1 = nuh1 * (real)0.5 * (d.hvm[k] + d.hvm[ID(i + 1, j)]) * (d.uu[ID(i, jp1)] - d.uu[k]) / c.dy;
    real dudy2 = nuh2 * (real)0.5 * (d.hvm[ID(i, jm1)] + d.hvm[ID(i + 1, jm1)]) * (d.uu[k] - d.uu[ID(i, jm1)]) / c.dy;
    vis += (1 / d.hum[k]) * (2 * (dudy1 - dudy2) / (2 * c.dy)) * (real)(d.wetu[ID(i, jp1)] * d.wetu[ID(i, jm1)]);
    if (c.smag) {
        real dvdx1 = nuh1 * (real)0.5 * (d.hvm[k] + d.hvm[ID(i + 1, j)]) * (d.vv[ID(i + 1, j)] - d.vv[k]) / c.dx;
        real dvdx2 = nuh2 * (real)0.5 * (d.hvm[ID(i, jm1)] + d.hvm[ID(i + 1, jm1)]) * (d.vv[ID(i + 1, jm1)] - d.vv[ID(i, jm1)]) / c.dx;
        vis += (1 / d.hum[k]) * (dvdx1 - dvdx2) / c.dy *
               (real)(d.wetv[ID(i + 1, j)] * d.wetv[k] * d.wetv[ID(i + 1, jm1)] * d.wetv[ID(i, jm1)]);
    }
    d.viscu[k] = vis;
}
__global__ void k_viscv1(Dev d) {
    IJ
    size_t k = ID(i, j);
    real vis = 0;
    if (IN(2, c.nx, 2, c.ny - 1)) {
        int jp1 = min(j + 1, c.NY1), jm1 = max(j - 1, 1);
        real dvdy1 = d.nuh[ID(i, jp1)] * d.hh[ID(i, jp1)] * (d.vv[ID(i, jp1)] - d.vv[k]) / c.dy;
        real dvdy2 = d.nuh[k] * d.hh[k] * (d.vv[k] - d.vv[ID(i, jm1)]) / c.dy;
        vis = (1 / d.hvm[k]) * 2 * (dvdy1 - dvdy2) / (2 * c.dy) * (real)(d.wetv[ID(i, jp1)] * d.wetv[ID(i, jm1)]);
    }
    d.viscv[k] = vis;
}
__global__ void k_viscv_bc(Dev d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    d.viscv[ID(i, 1)] = d.viscv[ID(i, 2)];
    d.viscv[ID(i, c.ny)] = d.viscv[ID(i, c.ny - 1)];
}
__global__ void k_viscv2(Dev d) {
    IJ
    size_t k = ID(i, j);
    real vis = d.viscv[k] * (c.smag ? 2 : 1);
    if (IN(2, c.nx, 2, c.ny - 1)) {
        int jp1 = min(j + 1, c.NY1);
        real nh = c.nuhv;
        real nuh1 = nh * (real)0.25 * (d.nuh[k] + d.nuh[ID(i + 1, j)] + d.nuh[ID(i + 1, jp1)] + d.nuh[ID(i, jp1)]);
        real nuh2 = nh * (real)0.25 * (d.nuh[k] + d.nuh[ID(i - 1, j)] + d.nuh[ID(i - 1, jp1)] + d.nuh[ID(i, jp1)]);
        real dvdx1 = nuh1 * (real)0.5 * (d.hum[k] + d.hum[ID(i, jp1)]) * (d.vv[ID(i + 1, j)] - d.vv[k]) / c.dx;
        real dvdx2 = nuh2 * (real)0.5 * (d.hum[ID(i - 1, j)] + d.hum[ID(i - 1, jp1)]) * (d.vv[k] - d.vv[ID(i - 1, j)]) / c.dx;
        vis += (1 / d.hvm[k]) * (2 * (dvdx1 - dvdx2) / (2 * c.dx)) * (real)(d.wetv[ID(i + 1, j)] * d.wetv[ID(i - 1, j)]);
        if (c.smag) {
            real dudy1 = nuh1 * (real)0.5 * (d.hum[k] + d.hum[ID(i, jp1)]) * (d.uu[ID(i, jp1)] - d.uu[k]) / c.dy;
            real dudy2 = nuh2 * (real)0.5 * (d.hum[ID(i - 1, j)] + d.hum[ID(i - 1, jp1)]) * (d.uu[ID(i - 1, jp1)] - d.uu[ID(i - 1, j)]) / c.dy;
            vis += (1 / d.hvm[k]) * (dudy1 - dudy2) / c.dx *
                   (real)(d.wetu[ID(i, jp1)] * d.wetu[k] * d.wetu[ID(i - 1, jp1)] * d.wetv[ID(i - 1, j)]);
        }
    }
    d.viscv[k] = vis;
}
// bed shear stress and explicit momentum update
__global__ void k_momentum(Dev d, real dt, real wsu, real wnv) {
    IJ
    size_t k = ID(i, j);
    real cap = 100 * c.g * c.rho;
    real tbx = d.wetu[k] == 1 ? d.cfu[k] * c.rho * d.ueu[k] * d.vmageu[k] : 0;
    if (fabs(tbx) > cap * d.hu[k]) tbx = copysign(cap * d.hu[k], tbx);
    real tby = d.wetv[k] == 1 ? d.cfv[k] * c.rho * d.vev[k] * d.vmagev[k] : 0;
    if (fabs(tby) > cap * d.hv[k]) tby = copysign(cap * d.hv[k], tby);
    d.taubx[k] = tbx; d.tauby[k] = tby;
    real wmag = sqrt(wsu * wsu + wnv * wnv);
    real fmax_ = c.maxfacg * c.g;
    if (IN(2, c.nx - 1, 2, c.ny)) {
        if (d.wetu[k] == 1) {
            real dudt = d.ududx[k] + d.vdudy[k] - d.viscu[k] + c.g * d.dzsdx[k] + tbx / (c.rho * d.hu[k])
                        - c.rhoa * c.Cd * wsu * wmag / (c.rho * d.hum[k]);
            dudt = fmin(fmax(dudt, -fmax_), fmax_);
            d.uu[k] -= dt * dudt;
        } else d.uu[k] = 0;
    }
    if (IN(2, c.nx, 2, c.ny - 1)) {
        if (d.wetv[k] == 1) {
            real dvdt = d.udvdx[k] + d.vdvdy[k] - d.viscv[k] + c.g * d.dzsdy[k] + tby / (c.rho * d.hv[k])
                        - c.rhoa * c.Cd * wnv * wmag / (c.rho * d.hvm[k]);
            dvdt = fmin(fmax(dvdt, -fmax_), fmax_);
            d.vv[k] -= dt * dvdt;
        } else d.vv[k] = 0;
    }
}
// lateral boundaries after the momentum update: uu copies and flow_lat_bc for vv
__global__ void k_lat_uv(Dev d, real dt, real wsu, real wnv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    d.uu[ID(i, 1)] = d.uu[ID(i, 2)];
    d.uu[ID(i, c.NY1)] = d.uu[ID(i, c.ny)];
    if (i < 2 || i > c.nx) return;
    real wmag = sqrt(wsu * wsu + wnv * wnv);
    for (int side = 0; side < 2; side++) {
        int jb = side == 0 ? 1 : c.ny;
        int type = side == 0 ? c.right : c.left;     // XBeach passes par%right at j = 1
        size_t k = ID(i, jb);
        real vb;
        if (type == LR_WALL) vb = 0;
        else if (d.wetv[k] == 1)
            vb = d.vv[k] - dt * (d.udvdx[k] + d.vdvdy[k] - d.viscv[k] + c.g * d.dzsdy[k] + d.tauby[k] / (c.rho * d.hvm[k])
                                 - c.rhoa * c.Cd * wnv * wmag / (c.rho * d.hvm[k]));
        else vb = 0;
        d.w1[ID(i, side + 1)] = vb;                 // scratch, applied in k_lat_v_apply
    }
}
__global__ void k_lat_v_apply(Dev d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    for (int side = 0; side < 2; side++) {
        int jb = side == 0 ? 1 : c.ny;
        int type = side == 0 ? c.right : c.left;
        real vb;
        if (type == LR_WALL) vb = 0;
        else if (i == 1) vb = d.w1[ID(2, side + 1)] * (real)d.wetv[ID(1, jb)];
        else if (i == c.NX1) vb = d.w1[ID(c.nx, side + 1)] * (real)d.wetv[ID(c.NX1, jb)];
        else vb = d.w1[ID(i, side + 1)];
        d.vv[ID(i, jb)] = vb;
    }
}

// ---------------------------------------------------------------------------------------------
// nonh.F90 : nonh_1lay_pred with nonh_break (nhbreaker = 1)
// ---------------------------------------------------------------------------------------------
__global__ void k_nonhUV(Dev d) {
    IJ
    size_t k = ID(i, j);
    int ie = min(c.nx, i + 1), js = min(c.ny, j + 1);
    d.nonhU[k] = (d.wetu[k] == 1 && (real)0.5 * (d.zs[k] + d.zs[ID(ie, j)]) > d.zbu[k]) ? 1 : 0;
    d.nonhV[k] = (d.wetv[k] == 1 && (real)0.5 * (d.zs[k] + d.zs[ID(i, js)]) > d.zbv[k]) ? 1 : 0;
}
__device__ __forceinline__ real dzdt_at(const Dev &d, int i, int j) {
    return -(d.uu[ID(i, j)] * d.hu[ID(i, j)] - d.uu[ID(i - 1, j)] * d.hu[ID(i - 1, j)]) / c.dx
           - (d.vv[ID(i, j)] * d.hv[ID(i, j)] - d.vv[ID(i, j - 1)] * d.hv[ID(i, j - 1)]) / c.dy;
}
__global__ void k_break(Dev d) {
    IJ
    size_t k = ID(i, j);
    if (!IN(2, c.nx, 2, c.ny)) { d.brk_new[k] = d.breaking[k]; return; }
    d.nonhZ[k] = max(max(d.nonhV[k], d.nonhV[ID(i, j - 1)]), max(d.nonhU[k], d.nonhU[ID(i - 1, j)])) > 0 ? 1 : 0;
    real dzdt = dzdt_at(d, i, j);
    // XBeach updates s%breaking in place in an (j,i) sweep: west/south neighbours already processed.
    // A neighbour that was 1 and stopped breaking (dzdt < 0) reads -1; a newly breaking one reads 2.
    int bw = d.breaking[ID(i - 1, j)], bs = d.breaking[ID(i, j - 1)];
    if (bw == 1 && i - 1 >= 2 && dzdt_at(d, i - 1, j) < 0) bw = -1;
    if (bs == 1 && j - 1 >= c.jlo && dzdt_at(d, i, j - 1) < 0) bs = -1;
    int nb = (bw == 1 || d.breaking[ID(i + 1, j)] == 1 || bs == 1 || d.breaking[ID(i, j + 1)] == 1) ? 1 : 0;
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
__global__ void k_break_apply(Dev d) {
    IJ
    size_t k = ID(i, j);
    int b = d.brk_new[k];
    d.breaking[k] = b;
    if (b != 0) { d.nonhZ[k] = 0; d.pres[k] = 0; }
}
// vertical momentum advection; neighbours processed earlier in the sweep are already zeroed if non-NH
__global__ void k_wm_adv(Dev d, real dt) {
    IJ
    if (!IN(2, c.nx, 2, c.ny)) return;
    size_t k = ID(i, j);
    if (d.nonhZ[k] == 1) {
        real w0 = d.Wm_old[k];
        real ww = (i - 1 >= 2 && d.nonhZ[ID(i - 1, j)] == 0) ? 0 : d.Wm_old[ID(i - 1, j)];
        real wsx = (j - 1 >= c.jlo && d.nonhZ[ID(i, j - 1)] == 0) ? 0 : d.Wm_old[ID(i, j - 1)];
        real we = d.Wm_old[ID(i + 1, j)], wn = d.Wm_old[ID(i, j + 1)];
        real h = d.hh[k];
        d.Wm[k] = w0 - dt * (fmax(d.qx[ID(i - 1, j)], (real)0) * (w0 - ww) / h / c.dx
                             + fmin(d.qx[k], (real)0) * (we - w0) / h / c.dx
                             + fmax(d.qy[ID(i, j - 1)], (real)0) * (w0 - wsx) / h / c.dy
                             + fmin(d.qy[k], (real)0) * (wn - w0) / h / c.dy);
    } else d.Wm[k] = 0;
}
__global__ void k_wm_zero(Dev d) {
    IJ
    if (!IN(2, c.nx, 2, c.ny)) return;
    size_t k = ID(i, j);
    if (d.nonhZ[k] == 0) { d.Wm_old[k] = 0; d.ws[k] = 0; d.wb[k] = 0; d.pres[k] = 0; }
}
// vertical viscosity (XBeach uses hu for the y-fluxes too), pressure coefficients au/av
__global__ void k_wm_visc_coef(Dev d, real dt) {
    IJ
    size_t k = ID(i, j);
    if (IN(2, c.nx, 2, c.ny)) {
        real w0 = d.Wm_old[k];
        real dwdx1 = (real)0.5 * (d.nuh[ID(i - 1, j)] + d.nuh[k]) * d.hu[ID(i - 1, j)] * (w0 - d.Wm_old[ID(i - 1, j)]) / c.dx;
        real dwdx2 = (real)0.5 * (d.nuh[ID(i + 1, j)] + d.nuh[k]) * d.hu[k] * (d.Wm_old[ID(i + 1, j)] - w0) / c.dx;
        real dwdy1 = d.nuh[ID(i, j - 1)] * d.hu[ID(i, j - 1)] * (w0 - d.Wm_old[ID(i, j - 1)]) / c.dy;
        real dwdy2 = d.nuh[k] * d.hu[k] * (d.Wm_old[ID(i, j + 1)] - w0) / c.dy;
        real h = d.hh[k];
        d.Wm[k] += (1 / h) * dt * (dwdx2 - dwdx1) / c.dx * (real)(d.wetu[k] * d.wetu[ID(i - 1, j)])
                 + (1 / h) * dt * (dwdy2 - dwdy1) / c.dy * (real)(d.wetv[k] * d.wetv[ID(i, j - 1)]);
        if (d.nonhU[k] == 1) {
            real vol = (real)0.5 * dt / (d.hum[k] * c.dx);
            d.au1[k] = -(d.zs[ID(i + 1, j)] - d.zb[k]) * vol;
            d.au0[k] = (d.zs[k] - d.zb[ID(i + 1, j)]) * vol;
        } else { d.au1[k] = 0; d.au0[k] = 0; }
        if (d.nonhV[k] == 1) {
            real vol = (real)0.5 * dt / (d.hvm[k] * c.dy);
            d.av1[k] = -(d.zs[ID(i, j + 1)] - d.zb[k]) * vol;
            d.av0[k] = (d.zs[k] - d.zb[ID(i, j + 1)]) * vol;
        } else { d.av1[k] = 0; d.av0[k] = 0; }
    }
}
// secorder: explicit old pressure in u, v, w
__global__ void k_pred_pres(Dev d, real dt) {
    IJ
    size_t k = ID(i, j);
    if (IN(2, c.nx - 1, 2, c.ny)) d.uu[k] += d.au1[k] * d.pres[ID(i + 1, j)] + d.au0[k] * d.pres[k];
    if (IN(2, c.nx, 2, c.ny - 1)) d.vv[k] += d.av1[k] * d.pres[ID(i, j + 1)] + d.av0[k] * d.pres[k];
    if (IN(2, c.nx, 2, c.ny) && d.nonhZ[k] == 1) d.Wm[k] += d.wcoef[k] * dt * d.pres[k] / d.hh[k];
}
// flow_secondorder_advW (serial bounds: imin = 2, imax = nx-1, jmin = 2, jmax = ny-1)
__global__ void k_advW_flux(Dev d) {
    IJ
    size_t k = ID(i, j);
    const real *w = d.Wm, *wo = d.Wm_old;
    real f1 = 0, f2 = 0;
    int imin = 2, imax = c.nx - 1, jmin = c.w_jmin, jmax = c.w_jmax;
    if (j >= 2 && j <= c.ny && i >= 1 && i <= c.nx && d.nonhZ[k] != 0) {
        int ie = min(i + 1, imax + 1), iee = min(i + 2, imax + 1), iw = max(i - 1, 1);
        real rfac = (real)(d.nonhZ[ID(ie, j)] * d.nonhZ[ID(iee, j)] * d.nonhZ[ID(iw, j)]);
        real zsmin = 1e30, zbmax = -1e30;
        for (int ii = iw; ii <= iee; ii++) { zsmin = fmin(zsmin, d.zs[ID(ii, j)]); zbmax = fmax(zbmax, d.zb[ID(ii, j)]); }
        real qx = d.qx[k] * c.dy * rfac;
        if (zsmin - zbmax > c.eps) {
            if (qx > 0 && i > imin)
                f1 = (real)0.5 * c.dx * minmod((w[ID(ie, j)] - wo[k]) / c.dx, (w[k] - wo[ID(iw, j)]) / c.dx) * qx;
            else if (qx < 0 && i < imax)
                f1 = (real)-0.5 * c.dx * minmod((wo[ID(ie, j)] - w[k]) / c.dx, (wo[ID(iee, j)] - w[ID(ie, j)]) / c.dx) * qx;
        }
    }
    if (j >= 1 && j <= c.ny && i >= 2 && i <= c.nx && d.nonhZ[k] != 0) {
        int jn = min(j + 1, jmax + 1), jnn = min(j + 2, jmax + 1), js = max(j - 1, 1);
        real rfac = (real)(d.nonhZ[ID(i, jn)] * d.nonhZ[ID(i, jnn)] * d.nonhZ[ID(i, js)]);
        real zsmin = 1e30, zbmax = -1e30;
        for (int jj = js; jj <= jnn; jj++) { zsmin = fmin(zsmin, d.zs[ID(i, jj)]); zbmax = fmax(zbmax, d.zb[ID(i, jj)]); }
        real qy = d.qy[k] * c.dx * rfac;
        if (zsmin - zbmax > c.eps) {
            if (qy > 0 && j > jmin)
                f2 = (real)0.5 * c.dy * minmod((w[ID(i, jn)] - wo[k]) / c.dy, (w[k] - wo[ID(i, js)]) / c.dy) * qy;
            else if (qy < 0 && j < jmax)
                f2 = (real)-0.5 * c.dy * minmod((wo[ID(i, jn)] - w[k]) / c.dy, (wo[ID(i, jnn)] - w[ID(i, jn)]) / c.dy) * qy;
        }
    }
    if (i == 1 || i == c.nx) f1 = 0;
    if (!c.cyclic && (j == 1 || j == c.ny)) f2 = 0;
    d.w1[k] = f1; d.w2[k] = f2;
}
__global__ void k_advW_apply(Dev d, real dt) {
    IJ
    if (!IN(2, c.nx, 2, c.ny)) return;
    size_t k = ID(i, j);
    if (d.nonhZ[k] == 0) return;
    d.Wm[k] -= dt / (c.dx * c.dy) * (d.w1[k] + d.w2[k] - d.w1[ID(i - 1, j)] - d.w2[ID(i, j - 1)]) / d.hh[k];
}

// ---------------------------------------------------------------------------------------------
// flow_secondorder.F90 : flow_secondorder_advUV (serial: imin = 3, imax = nx-1, jmin = 3, jmax = ny-1)
// ---------------------------------------------------------------------------------------------
__global__ void k_advUV_flux(Dev d) {
    IJ
    size_t k = ID(i, j);
    const real *uu = d.uu, *uo = d.uu_old, *vv = d.vv, *vo = d.vv_old;
    int imin = 3, imax = c.nx - 1, jmin = c.uv_jmin, jmax = c.uv_jmax;
    real dx = c.dx, dy = c.dy;
    real a1 = 0, a2 = 0, b1 = 0, b2 = 0;
    if (IN(2, c.nx, 2, c.ny)) {   // u, x-flux (wrk1)
        int ie = min(i + 1, imax + 1), iee = min(i + 2, imax + 1), iw = max(i - 1, 1), iww = max(i - 2, 1);
        real zsmin = 1e30, zbmax = -1e30;
        for (int ii = iww; ii <= iee; ii++) { zsmin = fmin(zsmin, d.zs[ID(ii, j)]); zbmax = fmax(zbmax, d.zb[ID(ii, j)]); }
        real qx = (real)0.5 * (d.qx[k] + d.qx[ID(iw, j)]);
        if (zsmin - zbmax > c.eps) {
            if (qx > 0 && i > imin)
                a1 = (real)0.5 * dx * minmod((uu[k] - uo[ID(iw, j)]) / dx, (uu[ID(iw, j)] - uo[ID(iww, j)]) / dx) * qx;
            else if (qx < 0 && i < imax + 1)
                a1 = (real)-0.5 * dx * minmod((uo[k] - uu[ID(iw, j)]) / dx, (uo[ID(ie, j)] - uu[k]) / dx) * qx;
        }
    }
    if (IN(2, c.nx - 1, 2, c.ny)) {   // u, y-flux (wrk2)
        int js = min(j + 1, jmax + 1), jss = min(j + 2, jmax + 1), jn = max(j - 1, 1);
        real zsmin = 1e30, zbmax = -1e30;
        for (int jj = jn; jj <= jss; jj++)
            for (int ii = i; ii <= i + 1; ii++) { zsmin = fmin(zsmin, d.zs[ID(ii, jj)]); zbmax = fmax(zbmax, d.zb[ID(ii, jj)]); }
        if (zsmin - zbmax > c.eps) {
            real qy = (d.qy[ID(i + 1, j)] + d.qy[k]) * (real)0.5;
            if (qy > 0 && j > jmin - 1)
                a2 = (real)0.5 * dy * minmod((uu[ID(i, js)] - uo[k]) / dy, (uu[k] - uo[ID(i, jn)]) / dy) * qy;
            else if (qy < 0 && j < jmax)
                a2 = (real)-0.5 * dy * minmod((uo[ID(i, js)] - uu[k]) / dy, (uo[ID(i, jss)] - uu[ID(i, js)]) / dy) * qy;
        }
    }
    if (IN(2, c.nx, 2, c.ny)) {   // v, y-flux (wrk1)
        int js = min(j + 1, jmax + 1), jn = max(j - 1, 1), jnn = max(j - 2, 1), jss = min(j + 2, jmax + 1);
        real zsmin = 1e30, zbmax = -1e30;
        for (int jj = jnn; jj <= jss; jj++) { zsmin = fmin(zsmin, d.zs[ID(i, jj)]); zbmax = fmax(zbmax, d.zb[ID(i, jj)]); }
        real qy = (real)0.5 * (d.qy[k] + d.qy[ID(i, jn)]) * dx;
        if (zsmin - zbmax > c.eps) {
            if (qy > 0 && j > jmin)
                b1 = (real)0.5 * dy * minmod((vv[k] - vo[ID(i, jn)]) / dy, (vv[ID(i, jn)] - vo[ID(i, jnn)]) / dy) * qy;
            else if (qy < 0 && j < jmax + 1)
                b1 = (real)-0.5 * dy * minmod((vo[k] - vv[ID(i, jn)]) / dy, (vo[ID(i, js)] - vv[k]) / dy) * qy;
        }
    }
    if (IN(2, c.nx, 2, c.ny - 1)) {   // v, x-flux (wrk2)
        int ie = min(i + 1, c.nx), iee = min(i + 2, c.nx), iw = max(i - 1, 1);
        real zsmin = 1e30, zbmax = -1e30;
        for (int jj = j; jj <= j + 1; jj++)
            for (int ii = iw; ii <= iee; ii++) { zsmin = fmin(zsmin, d.zs[ID(ii, jj)]); zbmax = fmax(zbmax, d.zb[ID(ii, jj)]); }
        if (zsmin - zbmax > c.eps) {
            real qx = (real)0.5 * (d.qx[ID(i, j + 1)] + d.qx[k]) * dy;
            if (qx > c.umin && i > imin - 1)
                b2 = (real)0.5 * dx * minmod((vv[ID(ie, j)] - vo[k]) / dx, (vv[k] - vo[ID(iw, j)]) / dx) * qx;
            else if (qx < -c.umin && i < imax)
                b2 = (real)-0.5 * dx * minmod((vo[ID(ie, j)] - vv[k]) / dx, (vo[ID(iee, j)] - vv[ID(ie, j)]) / dx) * qx;
        }
    }
    d.w1[k] = a1; d.w2[k] = a2; d.w3[k] = b1; d.w4[k] = b2;
}
__global__ void k_advUV_apply(Dev d, real dt) {
    IJ
    size_t k = ID(i, j);
    real dx = c.dx, dy = c.dy;
    if (IN(2, c.nx - 1, 2, c.ny)) {
        real fac = dt / d.hum[k] / (dx * dy);
        real w2s = (j - 1 >= 2) ? d.w2[ID(i, j - 1)] : 0;      // wrk2(:,1) = 0
        d.uu[k] -= fac * (dy * d.w1[ID(i + 1, j)] - dy * d.w1[k] + dx * d.w2[k] - dx * w2s);
    }
    if (IN(2, c.nx, 2, c.ny - 1)) {
        real fac = dt / d.hvm[k] / (dx * dy);
        real w4w = (i - 1 >= 2) ? d.w4[ID(i - 1, j)] : 0;      // wrk2(1,:) = 0
        d.vv[k] -= fac * (d.w4[k] - w4w + d.w3[ID(i, j + 1)] - d.w3[k]);
    }
}

// ---------------------------------------------------------------------------------------------
// nonh.F90 : nonh_1lay_cor (2D branch). Builds the 5-diagonal Poisson system.
// ---------------------------------------------------------------------------------------------
__global__ void k_zuzv(Dev d) {
    IJ
    size_t k = ID(i, j);
    if (i <= c.nx) d.zsu[k] = d.hu[k] + d.zbu[k];
    if (j <= c.ny) d.zsv[k] = d.hv[k] + d.zbv[k];
}
__global__ void k_build(Dev d, real dt) {
    IJ
    size_t k = ID(i, j);
    if (!IN(2, c.nx, 2, c.ny)) return;
    real dx = c.dx, dy = c.dy;
    if (d.nonhZ[k] == 1) {
        int uC = d.nonhU[k], uW = d.nonhU[ID(i - 1, j)], vC = d.nonhV[k], vS = d.nonhV[ID(i, j - 1)];
        real zbuC = d.zbu[k], zbuW = d.zbu[ID(i - 1, j)], zbvC = d.zbv[k], zbvS = d.zbv[ID(i, j - 1)], zb = d.zb[k];
        real dzb_e, dzb_w, dzb_s, dzb_n;
        if (uC * uW == 1) { dzb_e = (real)0.5 * (zbuC - zbuW) / dx; dzb_w = dzb_e; }
        else if (uC == 0) { dzb_e = 0; dzb_w = (real)0.5 * (zb - zbuW) / dx; }
        else              { dzb_e = (real)0.5 * (zbuC - zb) / dx; dzb_w = 0; }
        if (vC * vS == 1) { dzb_s = (real)0.5 * (zbvC - zbvS) / dy; dzb_n = dzb_s; }
        else if (vC == 0) { dzb_s = 0; dzb_n = (real)0.5 * (zb - zbvS) / dy; }
        else              { dzb_s = (real)0.5 * (zbvC - zb) / dy; dzb_n = 0; }
        real au0C = d.au0[k], au1W = d.au1[ID(i - 1, j)], au0W = d.au0[ID(i - 1, j)], au1C = d.au1[k];
        real av0C = d.av0[k], av1S = d.av1[ID(i, j - 1)], av0S = d.av0[ID(i, j - 1)], av1C = d.av1[k];
        real awb1 = dzb_e * au0C + dzb_w * au1W + dzb_s * av0C + dzb_n * av1S;
        real awb2 = dzb_w * au0W, awb3 = dzb_e * au1C, awb4 = dzb_n * av0S, awb5 = dzb_s * av1C;
        real uC_ = d.uu[k], uW_ = d.uu[ID(i - 1, j)], vC_ = d.vv[k], vS_ = d.vv[ID(i, j - 1)];
        real awbr = dzb_e * uC_ + dzb_w * uW_ + dzb_s * vC_ + dzb_n * vS_;
        real aws1 = d.wcoef[k] * dt * 2 / d.hh[k] - awb1;
        real aws2 = -awb2, aws3 = -awb3, aws4 = -awb4, aws5 = -awb5;
        real awsr = 2 * d.Wm[k] - awbr;
        d.awb[0][k] = awb1; d.awb[1][k] = awb2; d.awb[2][k] = awb3; d.awb[3][k] = awb4; d.awb[4][k] = awb5; d.awbr[k] = awbr;
        d.aws[0][k] = aws1; d.aws[1][k] = aws2; d.aws[2][k] = aws3; d.aws[3][k] = aws4; d.aws[4][k] = aws5; d.awsr[k] = awsr;
        real zsuC = d.zsu[k], zsuW = d.zsu[ID(i - 1, j)], zsvC = d.zsv[k], zsvS = d.zsv[ID(i, j - 1)], zs = d.zs[k];
        real dzs_e, dzs_w, dzs_s, dzs_n;
        if (uC * uW == 1) { dzs_e = (real)0.5 * (zsuC - zsuW) * dy; dzs_w = dzs_e; }
        else if (uC == 0) { dzs_e = 0; dzs_w = (real)0.5 * (zs - zsuW) * dy; }
        else              { dzs_e = (real)0.5 * (zsuC - zs) * dy; dzs_w = 0; }
        if (vC * vS == 1) { dzs_s = (real)0.5 * (zsvC - zsvS) * dx; dzs_n = dzs_s; }
        else if (vC == 0) { dzs_s = 0; dzs_n = (real)0.5 * (zs - zsvS) * dx; }
        else              { dzs_s = (real)0.5 * (zsvC - zs) * dx; dzs_n = 0; }
        real huC = d.hu[k], huW = d.hu[ID(i - 1, j)], hvC = d.hv[k], hvS = d.hv[ID(i, j - 1)];
        real A = dx * dy;
        d.mat[0][k] = dy * (huC * au0C - huW * au1W) + dx * (hvC * av0C - hvS * av1S)
                      - dzs_e * au0C - dzs_w * au1W - dzs_s * av0C - dzs_n * av1S + A * aws1;
        d.mat[1][k] = -dy * huW * au0W - dzs_w * au0W + A * aws2;
        d.mat[2][k] = dy * huC * au1C - dzs_e * au1C + A * aws3;
        d.mat[3][k] = -dx * hvS * av0S - dzs_n * av0S + A * aws4;
        d.mat[4][k] = dx * hvC * av1C - dzs_s * av1C + A * aws5;
        d.rhs[k] = -dy * (huC * uC_ - huW * uW_) - dx * (hvC * vC_ - hvS * vS_)
                   + dzs_e * uC_ + dzs_w * uW_ + dzs_s * vC_ + dzs_n * vS_ - A * awsr;
    } else {
        d.mat[0][k] = 1; d.mat[1][k] = 0; d.mat[2][k] = 0; d.mat[3][k] = 0; d.mat[4][k] = 0; d.rhs[k] = 0;
        for (int q = 0; q < 5; q++) { d.awb[q][k] = 0; d.aws[q][k] = 0; }
        d.awbr[k] = 0; d.awsr[k] = 0;
    }
}

// ---------------------------------------------------------------------------------------------
// Pressure solver: Jacobi-preconditioned BiCGStab on the interior (i 2..nx, j ps_j0..ps_j1; periodic in j when
// cyclic); dp = 0 elsewhere.
// sc: 0 rho, 1 rho_old, 2 alpha, 3 omega, 4 (rhat,v), 5 (t,s), 6 (t,t), 7 (r,r), 8 (b,b), 9 (rhat,r),
// 10 beta, 11 converged flag, 12 iteration count, 13 target |r|^2. Kernels return at once when converged, so the
// host launches iterations in batches and synchronises once per batch.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void block_sum2(double a, double b, double *out_a, double *out_b) {
    for (int o = 16; o > 0; o >>= 1) { a += __shfl_down_sync(0xffffffff, a, o); b += __shfl_down_sync(0xffffffff, b, o); }
    __shared__ double sa[32], sb[32];
    int lane = (threadIdx.y * blockDim.x + threadIdx.x) & 31, wid = (threadIdx.y * blockDim.x + threadIdx.x) >> 5;
    if (lane == 0) { sa[wid] = a; sb[wid] = b; }
    __syncthreads();
    int nw = (blockDim.x * blockDim.y + 31) / 32;
    if (wid == 0) {
        a = lane < nw ? sa[lane] : 0; b = lane < nw ? sb[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) { a += __shfl_down_sync(0xffffffff, a, o); b += __shfl_down_sync(0xffffffff, b, o); }
        if (lane == 0) { if (out_a) atomicAdd(out_a, a); if (out_b) atomicAdd(out_b, b); }
    }
}
#define INTERIOR (i >= 2 && i <= c.nx && j >= c.ps_j0 && j <= c.ps_j1)
__device__ __forceinline__ real Ax(const Dev &d, const real *x, int i, int j) {
    size_t k = ID(i, j);
    int js = j - 1, jn = j + 1;                   // cyclic: rows 2 and ny are the periodic images of ny-1 and 3
    if (c.cyclic) { if (j == c.ps_j0) js = c.ps_j1; if (j == c.ps_j1) jn = c.ps_j0; }
    return d.mat[0][k] * x[k] + d.mat[1][k] * x[ID(i - 1, j)] + d.mat[2][k] * x[ID(i + 1, j)]
         + d.mat[3][k] * x[ID(i, js)] + d.mat[4][k] * x[ID(i, jn)];
}
__global__ void k_s_init(Dev d) {   // x = 0, r = rhat = b, p = v = 0; (b,b), (rhat,r)
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double bb = 0;
    if (i <= c.NX1 && j <= c.NY1) {
        size_t k = ID(i, j);
        real b = INTERIOR ? d.rhs[k] : 0;
        d.dp[k] = 0; d.r[k] = b; d.rh[k] = b; d.p[k] = 0; d.v[k] = 0; d.y[k] = 0; d.z[k] = 0;
        bb = (double)b * b;
    }
    block_sum2(bb, bb, &d.sc[8], &d.sc[9]);
}
__global__ void k_s_beta(Dev d, int first) {
    if (d.sc[11] != 0) return;
    if (threadIdx.x) return;
    double *s = d.sc;
    if (s[11] != 0) return;
    s[0] = s[9];
    if (first) { s[10] = 0; } else { s[10] = (s[0] / s[1]) * (s[2] / s[3]); }   // beta
    s[4] = 0; s[5] = 0; s[6] = 0; s[7] = 0; s[9] = 0;
}
__global__ void k_s_p(Dev d) {      // p = r + beta (p - omega v); y = p / diag
    if (d.sc[11] != 0) return;
    IJ
    if (!INTERIOR) return;
    size_t k = ID(i, j);
    real beta = (real)d.sc[10], om = (real)d.sc[3];
    real p = d.r[k] + beta * (d.p[k] - om * d.v[k]);
    d.p[k] = p; d.y[k] = p / d.mat[0][k];
}
__global__ void k_s_Ay(Dev d) {     // v = A y; (rhat, v)
    if (d.sc[11] != 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double a = 0;
    if (i <= c.NX1 && j <= c.NY1 && INTERIOR) {
        size_t k = ID(i, j);
        real v = Ax(d, d.y, i, j);
        d.v[k] = v; a = (double)d.rh[k] * v;
    }
    block_sum2(a, 0, &d.sc[4], nullptr);
}
__global__ void k_s_s(Dev d) {      // alpha = rho/(rhat,v); s = r - alpha v; z = s / diag
    if (d.sc[11] != 0) return;
    IJ
    if (!INTERIOR) return;
    size_t k = ID(i, j);
    real alpha = (real)(d.sc[0] / d.sc[4]);
    real s = d.r[k] - alpha * d.v[k];
    d.s[k] = s; d.z[k] = s / d.mat[0][k];
}
__global__ void k_s_Az(Dev d) {     // t = A z; (t,s), (t,t)
    if (d.sc[11] != 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double ts = 0, tt = 0;
    if (i <= c.NX1 && j <= c.NY1 && INTERIOR) {
        size_t k = ID(i, j);
        real t = Ax(d, d.z, i, j);
        d.t[k] = t; ts = (double)t * d.s[k]; tt = (double)t * t;
    }
    block_sum2(ts, tt, &d.sc[5], &d.sc[6]);
}
__global__ void k_s_omega(Dev d) {
    if (d.sc[11] != 0) return;
    if (threadIdx.x) return;
    double *s = d.sc;
    if (s[11] != 0) return;
    s[2] = s[0] / s[4];
    s[3] = s[6] > 0 ? s[5] / s[6] : 0;
}
__global__ void k_s_x(Dev d) {      // x += alpha y + omega z; r = s - omega t; (r,r), (rhat,r)
    if (d.sc[11] != 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double rr = 0, rhr = 0;
    if (i <= c.NX1 && j <= c.NY1 && INTERIOR) {
        size_t k = ID(i, j);
        real al = (real)d.sc[2], om = (real)d.sc[3];
        d.dp[k] += al * d.y[k] + om * d.z[k];
        real r = d.s[k] - om * d.t[k];
        d.r[k] = r; rr = (double)r * r; rhr = (double)d.rh[k] * r;
    }
    block_sum2(rr, rhr, &d.sc[7], &d.sc[9]);
}
__global__ void k_s_roll(Dev d) {
    if (threadIdx.x || d.sc[11] != 0) return;
    d.sc[1] = d.sc[0]; d.sc[12] += 1;
    if (d.sc[7] < d.sc[13]) d.sc[11] = 1;
}

// ---------------------------------------------------------------------------------------------
// nonh_1lay_cor: corrections and vertical velocities
// ---------------------------------------------------------------------------------------------
__global__ void k_correct(Dev d) {
    IJ
    size_t k = ID(i, j);
    real dpk = d.dp[k];
    d.pres[k] += dpk;
    if (IN(2, c.nx - 1, 2, c.ny)) d.uu[k] += d.au1[k] * d.dp[ID(i + 1, j)] + d.au0[k] * dpk;
    if (IN(2, c.nx, 2, c.ny - 1)) d.vv[k] += d.av1[k] * d.dp[ID(i, j + 1)] + d.av0[k] * dpk;
}
__global__ void k_wswb(Dev d) {
    IJ
    if (!IN(2, c.nx, 2, c.ny)) return;
    size_t k = ID(i, j);
    real dx = c.dx, dy = c.dy;
    if (d.nonhZ[k] == 1) {
        real dC = d.dp[k], dW = d.dp[ID(i - 1, j)], dE = d.dp[ID(i + 1, j)], dS = d.dp[ID(i, j - 1)], dN = d.dp[ID(i, j + 1)];
        d.ws[k] = d.awsr[k] + dC * d.aws[0][k] + dW * d.aws[1][k] + dE * d.aws[2][k] + dS * d.aws[3][k] + dN * d.aws[4][k];
        d.wb[k] = d.awbr[k] + dC * d.awb[0][k] + dW * d.awb[1][k] + dE * d.awb[2][k] + dS * d.awb[3][k] + dN * d.awb[4][k];
    } else {
        // kinematic conditions at points excluded from the NH solve. XBeach reuses two stale loop
        // variables here: dzs_n = dzb_s (last value of the awb loop) and dzb_w = dzs_e (last value of the
        // ws loop); both are reproduced in serial loop order by k_stale_find / k_stale_val.
        int uC = d.wetu[k], uW = d.wetu[ID(i - 1, j)], vC = d.wetv[k], vS = d.wetv[ID(i, j - 1)];
        real zs = d.zs[k], zb = d.zb[k];
        real dzs_e = 0, dzs_w = 0, dzs_s = 0, dzs_n = 0, dzb_e = 0, dzb_w = 0, dzb_s = 0, dzb_n = 0;
        if (uC * uW == 1) { dzs_e = (real)0.5 * (d.zsu[k] - d.zsu[ID(i - 1, j)]) / dx; dzs_w = dzs_e;
                            dzb_e = (real)0.5 * (d.zbu[k] - d.zbu[ID(i - 1, j)]) / dx; dzb_w = (real)d.stale[1]; }
        else if (uC == 0) { dzs_w = (real)0.5 * (zs - d.zsu[ID(i - 1, j)]) / dx; dzb_w = (real)0.5 * (zb - d.zbu[ID(i - 1, j)]) / dx; }
        else if (uW == 0) { dzs_e = (real)0.5 * (d.zsu[k] - zs) / dx; dzb_e = (real)0.5 * (d.zbu[k] - zb) / dx; }
        if (vC * vS == 1) { dzs_s = (real)0.5 * (d.zsv[k] - d.zsv[ID(i, j - 1)]) / dy; dzs_n = (real)d.stale[0];
                            dzb_s = (real)0.5 * (d.zbv[k] - d.zbv[ID(i, j - 1)]) / dy; dzb_n = dzb_s; }
        else if (vC == 0) { dzs_n = (real)0.5 * (zs - d.zsv[ID(i, j - 1)]) / dy; dzb_n = (real)0.5 * (zb - d.zbv[ID(i, j - 1)]) / dy; }
        else if (vS == 0) { dzs_s = (real)0.5 * (d.zsv[k] - zs) / dy; dzb_s = (real)0.5 * (d.zbv[k] - zb) / dy; }
        real uC_ = d.uu[k], uW_ = d.uu[ID(i - 1, j)], vC_ = d.vv[k], vS_ = d.vv[ID(i, j - 1)];
        d.ws[k] = -(d.hu[k] * uC_ - d.hu[ID(i - 1, j)] * uW_) / dx - (d.hv[k] * vC_ - d.hv[ID(i, j - 1)] * vS_) / dy
                  + dzs_e * uC_ + dzs_w * uW_ + dzs_s * vC_ + dzs_n * vS_;
        d.wb[k] = dzb_e * uC_ + dzb_w * uW_ + dzb_s * vC_ + dzb_n * vS_;
    }
}
// last NH point (awb and mat loops, j = 2..ny, i = 2..nx) and last non-NH point (ws loop) in Fortran order
__global__ void k_stale_find(Dev d) {     // block-level maxima, one atomic per block and kind
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    unsigned kn = 0, ko = 0;
    if (i <= c.NX1 && j <= c.NY1 && IN(2, c.nx, 2, c.ny)) {
        unsigned key = (unsigned)j * c.NX1 + i;
        if (d.nonhZ[ID(i, j)] == 1) kn = key; else ko = key;
    }
    for (int o = 16; o > 0; o >>= 1) {         // warp maximum (shuffles: works on sm_60 and newer)
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
__global__ void k_stale_val(Dev d) {
    if (threadIdx.x) return;
    double dzb_s = 0, dzs_e = 0;
    unsigned long long kn = d.skey[0], ko = d.skey[1];
    if (kn) {                       // awb loop: dzb_s at the last NH point; mat loop: dzs_e there
        int i = (int)(kn % c.NX1), j = (int)(kn / c.NX1);
        size_t k = ID(i, j);
        int vC = d.nonhV[k], vS = d.nonhV[ID(i, j - 1)], uC = d.nonhU[k], uW = d.nonhU[ID(i - 1, j)];
        if (vC * vS == 1) dzb_s = 0.5 * (d.zbv[k] - d.zbv[ID(i, j - 1)]) / c.dy;
        else if (vC == 0) dzb_s = 0;
        else dzb_s = 0.5 * (d.zbv[k] - d.zb[k]) / c.dy;
        if (uC * uW == 1) dzs_e = 0.5 * (d.zsu[k] - d.zsu[ID(i - 1, j)]) * c.dy;
        else if (uC == 0) dzs_e = 0;
        else dzs_e = 0.5 * (d.zsu[k] - d.zs[k]) * c.dy;
    }
    if (ko) {                       // ws loop: dzs_e at the last non-NH point
        int i = (int)(ko % c.NX1), j = (int)(ko / c.NX1);
        size_t k = ID(i, j);
        int uC = d.wetu[k], uW = d.wetu[ID(i - 1, j)];
        if (uC * uW == 1) dzs_e = 0.5 * (d.zsu[k] - d.zsu[ID(i - 1, j)]) / c.dx;
        else if (uC == 0) dzs_e = 0;
        else if (uW == 0) dzs_e = 0.5 * (d.zsu[k] - d.zs[k]) / c.dx;
        else dzs_e = 0;
    }
    d.stale[0] = dzb_s; d.stale[1] = dzs_e;
}
__global__ void k_wswb_bc_j(Dev d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > c.NX1) return;
    d.ws[ID(i, 1)] = d.ws[ID(i, 2)]; d.ws[ID(i, c.NY1)] = d.ws[ID(i, c.ny)];
    d.wb[ID(i, 1)] = d.wb[ID(i, 2)]; d.wb[ID(i, c.NY1)] = d.wb[ID(i, c.ny)];
}
__global__ void k_wswb_bc_i(Dev d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > c.NY1) return;
    d.ws[ID(c.NX1, j)] = d.ws[ID(c.nx, j)];
    d.wb[ID(1, j)] = d.wb[ID(2, j)]; d.wb[ID(c.NX1, j)] = d.wb[ID(c.nx, j)];
}
__global__ void k_wmold(Dev d) {
    IJ
    size_t k = ID(i, j);
    d.Wm_old[k] = (real)0.5 * (d.ws[k] + d.wb[k]);
}

// ---------------------------------------------------------------------------------------------
// flow_timestep.F90 : hu/hv (upwind) + flow_secondorder_huhv, fluxes, continuity, derived fields
// ---------------------------------------------------------------------------------------------
__global__ void k_huhv(Dev d) {
    IJ
    size_t k = ID(i, j);
    int nx = c.nx, ny = c.ny;
    int ie = min(nx, i) + 1, js = min(ny, j) + 1;
    real u = d.uu[k], v = d.vv[k], zbm, hu, hv;
    zbm = fmax(d.zb[k], d.zb[ID(ie, j)]);
    if (u > c.umin) hu = d.zs[k] - zbm;
    else if (u < -c.umin) hu = d.zs[ID(ie, j)] - zbm;
    else hu = fmax(fmax(d.zs[k], d.zs[ID(ie, j)]) - zbm, c.eps);
    hu = fmax(hu, (real)0);
    zbm = fmax(d.zb[k], d.zb[ID(i, js)]);
    if (v > c.umin) hv = d.zs[k] - zbm;
    else if (v < -c.umin) hv = d.zs[ID(i, js)] - zbm;
    else hv = fmax(fmax(d.zs[k], d.zs[ID(i, js)]) - zbm, c.eps);
    hv = fmax(hv, (real)0);
    if (c.secorder) {
        if (IN(2, nx - 1, 2, ny)) {     // imin = 2, imax = nx-1
            int imin = 2, imax = nx - 1;
            int ie2 = min(i + 1, imax + 1), iee = min(i + 2, imax + 1), iw = max(i - 1, 1);
            real zsmin = 1e30, zbmax = -1e30;
            for (int ii = iw; ii <= iee; ii++) { zsmin = fmin(zsmin, d.zs[ID(ii, j)]); zbmax = fmax(zbmax, d.zb[ID(ii, j)]); }
            if (zsmin - zbmax > c.eps) {
                if (u > c.umin && i > imin)
                    hu += (real)0.5 * c.dx * minmod((d.zs[ID(ie2, j)] - d.zs[k]) / c.dx, (d.zs[k] - d.zs[ID(iw, j)]) / c.dx);
                else if (u < -c.umin && i < imax)
                    hu -= (real)0.5 * c.dx * minmod((d.zs[ID(iee, j)] - d.zs[ID(ie2, j)]) / c.dx, (d.zs[ID(ie2, j)] - d.zs[k]) / c.dx);
            }
        }
        if (IN(2, nx, 2, ny - 1)) {     // jmin = 2, jmax = ny-1
            int jmin = c.w_jmin, jmax = c.w_jmax;
            int jn = min(j + 1, jmax + 1), jnn = min(j + 2, jmax + 1), jw = max(j - 1, 1);
            real zsmin = 1e30, zbmax = -1e30;
            for (int jj = jw; jj <= jnn; jj++) { zsmin = fmin(zsmin, d.zs[ID(i, jj)]); zbmax = fmax(zbmax, d.zb[ID(i, jj)]); }
            if (zsmin - zbmax > c.eps) {
                if (v > c.umin && j > jmin)
                    hv += (real)0.5 * c.dy * minmod((d.zs[ID(i, jn)] - d.zs[k]) / c.dy, (d.zs[k] - d.zs[ID(i, jw)]) / c.dy);
                else if (v < -c.umin && j < jmax)
                    hv -= (real)0.5 * c.dy * minmod((d.zs[ID(i, jnn)] - d.zs[ID(i, jn)]) / c.dy, (d.zs[ID(i, jn)] - d.zs[k]) / c.dy);
            }
        }
    }
    d.hu[k] = hu; d.hv[k] = hv;
    d.qx[k] = u * hu; d.qy[k] = v * hv;
}
__global__ void k_continuity(Dev d, real dt) {
    IJ
    if (!IN(2, c.nx, 2, c.ny)) return;
    size_t k = ID(i, j);
    real dzsdt = -((d.qx[k] - d.qx[ID(i - 1, j)]) / c.dx + (d.qy[k] - d.qy[ID(i, j - 1)]) / c.dy);
    d.zs[k] += dzsdt * dt;
}
__global__ void k_post(Dev d) {
    IJ
    size_t k = ID(i, j);
    int nx = c.nx, ny = c.ny;
    d.uu_old[k] = d.uu[k]; d.vv_old[k] = d.vv[k];
    // vu: (1:nx, 2:ny), copies to j = 1, ny+1, times wetu
    int jj = j == 1 ? 2 : (j == ny + 1 ? ny : j);
    real vu = 0;
    if (i <= nx) vu = (real)0.25 * (d.vv[ID(i, jj - 1)] + d.vv[ID(i, jj)] + d.vv[ID(i + 1, jj - 1)] + d.vv[ID(i + 1, jj)]);
    vu *= (real)d.wetu[k];
    // uv: (2:nx, 1:ny), copy to ny+1, times wetv
    int j2 = j == ny + 1 ? ny : j;
    real uv = 0;
    if (i >= 2 && i <= nx) uv = (real)0.25 * (d.uu[ID(i - 1, j2)] + d.uu[ID(i, j2)] + d.uu[ID(i - 1, j2 + 1)] + d.uu[ID(i, j2 + 1)]);
    uv *= (real)d.wetv[k];
    d.vu[k] = vu; d.uv[k] = uv;
    d.ueu[k] = d.uu[k]; d.vev[k] = d.vv[k];
    d.vmageu[k] = sqrt(d.uu[k] * d.uu[k] + vu * vu);
    d.vmagev[k] = sqrt(uv * uv + d.vv[k] * d.vv[k]);
    d.hh[k] = fmax(d.zs[k] - d.zb[k], c.eps);
}
// cell-centre velocities for output (flow_timestep end)
__global__ void k_out_u(Dev d, double *o) {
    IJ
    int nx = c.nx;
    int ii = i == nx + 1 ? nx : i;
    real u = ii == 1 ? d.uu[ID(1, j)] : (real)0.5 * (d.uu[ID(ii - 1, j)] + d.uu[ID(ii, j)]);
    o[ID(i, j)] = u;
}
__global__ void k_out_v(Dev d, double *o) {
    IJ
    int ny = c.ny, nx = c.nx;
    int ii = i == nx + 1 ? nx : i;
    int jj = j == ny + 1 ? ny : j;
    real v = jj == 1 ? d.vv[ID(ii, 1)] : (real)0.5 * (d.vv[ID(ii, jj - 1)] + d.vv[ID(ii, jj)]);
    o[ID(i, j)] = v;
}
__global__ void k_copy_out_int(const int *a, double *o, size_t n) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) o[k] = a[k];
}
__global__ void k_copy_out(const real *a, double *o, size_t n) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) o[k] = a[k];
}

// point output (postprocess.F90 snappointstogrid: nearest cell centre); u, v as in flow_timestep
enum { PV_ZS = 0, PV_UU, PV_VV, PV_U, PV_V, PV_HH, PV_WS, PV_WB, PV_PRES };
__global__ void k_points(Dev d, const int *pij, int np, const int *var, int nv, int uvzero, double *o) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= np * nv) return;
    int p = q / nv, v = var[q % nv];
    int i = pij[2 * p], j = pij[2 * p + 1], nx = c.nx, ny = c.ny;
    int ii = i == nx + 1 ? nx : i;
    real val = 0;
    switch (v) {
    case PV_ZS: val = d.zs[ID(i, j)]; break;
    case PV_UU: val = d.uu[ID(i, j)]; break;
    case PV_VV: val = d.vv[ID(i, j)]; break;
    case PV_U: if (!uvzero) val = ii == 1 ? d.uu[ID(1, j)] : (real)0.5 * (d.uu[ID(ii - 1, j)] + d.uu[ID(ii, j)]); break;
    case PV_V: if (!uvzero) { int jj = j == ny + 1 ? ny : j;
                              val = jj == 1 ? d.vv[ID(ii, 1)] : (real)0.5 * (d.vv[ID(ii, jj - 1)] + d.vv[ID(ii, jj)]); } break;
    case PV_HH: val = d.hh[ID(i, j)]; break;
    case PV_WS: val = d.ws[ID(i, j)]; break;
    case PV_WB: val = d.wb[ID(i, j)]; break;
    case PV_PRES: val = d.pres[ID(i, j)]; break;
    }
    o[q] = val;
}

// ---------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------
static std::map<std::string, std::string> read_params(const char *fn) {
    std::map<std::string, std::string> m;
    std::ifstream f(fn);
    if (!f) { fprintf(stderr, "cannot open %s\n", fn); exit(1); }
    std::string line;
    while (std::getline(f, line)) {
        size_t pc = line.find('%'); if (pc != std::string::npos) line = line.substr(0, pc);
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        auto trim = [](std::string s) { size_t a = s.find_first_not_of(" \t\r"), b = s.find_last_not_of(" \t\r");
                                        return a == std::string::npos ? std::string() : s.substr(a, b - a + 1); };
        std::string k = trim(line.substr(0, eq)), v = trim(line.substr(eq + 1));
        std::transform(k.begin(), k.end(), k.begin(), ::tolower);
        m[k] = v;
    }
    return m;
}
static double getd(std::map<std::string, std::string> &m, const char *k, double def) {
    auto it = m.find(k); return it == m.end() ? def : atof(it->second.c_str());
}
static std::string gets(std::map<std::string, std::string> &m, const char *k, const char *def) {
    auto it = m.find(k); return it == m.end() ? std::string(def) : it->second;
}
// lines listed after "key = n" in params.txt (output points, point variables, global variables)
static std::vector<std::string> list_after(const char *fn, const char *key) {
    std::ifstream f(fn); std::string line; std::vector<std::string> out; int n = -1;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (n > 0) {
            if (line.find_first_not_of(" \t") == std::string::npos) continue;
            out.push_back(line); if (--n == 0) break; continue;
        }
        size_t eq = line.find('=');
        if (eq == std::string::npos || n == 0) continue;
        std::string k = line.substr(0, eq);
        k.erase(std::remove_if(k.begin(), k.end(), ::isspace), k.end());
        std::transform(k.begin(), k.end(), k.begin(), ::tolower);
        if (k == key) n = atoi(line.substr(eq + 1).c_str());
    }
    return out;
}
static double disper(double w, double dd, double g) {
    double pi2 = 2 * M_PI, w2 = w * w, alpha = dd * w2 / g;
    double k = alpha * (1.0 / sqrt(tanh(alpha))) / dd, L = pi2 / k, Ldeep = g * pi2 / w2;
    double err = fabs(g * k * tanh(k * dd) - w2) / w2;
    while (err > 0.0001) { L = Ldeep * tanh(pi2 * dd / L); k = pi2 / L; err = fabs(g * k * tanh(k * dd) - w2) / w2; }
    return k;
}
// interp.F90 linear_interp (constant beyond the ends)
static double lin_interp(const std::vector<double> &x, const std::vector<double> &y, double xp) {
    if (xp <= x.front()) return y.front();
    if (xp >= x.back()) return y.back();
    size_t k = std::upper_bound(x.begin(), x.end(), xp) - x.begin();
    double f = (xp - x[k - 1]) / (x[k] - x[k - 1]);
    return y[k - 1] + f * (y[k] - y[k - 1]);
}
static std::vector<std::vector<double>> read_columns(const std::string &fn, int ncol) {
    std::ifstream f(fn);
    if (!f) { fprintf(stderr, "cannot open %s\n", fn.c_str()); exit(1); }
    std::vector<std::vector<double>> c(ncol); std::vector<double> row(ncol);
    while (true) {
        for (int q = 0; q < ncol; q++) if (!(f >> row[q])) return c;
        for (int q = 0; q < ncol; q++) c[q].push_back(row[q]);
    }
}

// buffered tokenizer for large list-directed Fortran files (separators: blanks, newlines, commas)
struct Tok {
    FILE *f; std::vector<char> b; size_t p = 0, n = 0; bool fe = false;
    explicit Tok(const std::string &fn) : b((1 << 22) + 1) {
        f = fopen(fn.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", fn.c_str()); exit(1); }
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
    bool num(double &v) { if (!skip()) return false; char *e; v = strtod(b.data() + p, &e); p = e - b.data(); return true; }
    bool word(std::string &s) { if (!skip()) return false; s.clear(); while (p < n && !sep(b[p])) s.push_back(b[p++]); return true; }
};

// boundaryconditions.F90 velocity_Boundary: scalar or vector file, linear in time between records
struct NHBC {
    Tok *tk = nullptr; bool vec = false, eof = false, first = true;
    int nvar = 0, NY1 = 0, iU = -1, iV = -1, iZ = -1, iW = -1;
    double t0 = 0, t1 = 0; std::vector<double> a0, a1;
    void open(const std::string &fn, int ny1) {
        NY1 = ny1; tk = new Tok(fn);
        std::string w; tk->word(w);
        std::string lw = w; for (auto &ch : lw) ch = (char)tolower(ch);
        if (lw == "vector") vec = true; else if (lw == "scalar") vec = false;
        else { fprintf(stderr, "%s: first line must be scalar or vector\n", fn.c_str()); exit(1); }
        double nv; tk->num(nv); nvar = (int)nv;
        for (int q = 0; q < nvar; q++) {
            tk->word(w);
            if (w == "z" || w == "Z" || w == "zs" || w == "ZS" || w == "Zs") iZ = q - 1;
            else if (w == "t" || w == "T") { if (q != 0) { fprintf(stderr, "%s: t must be the first column\n", fn.c_str()); exit(1); } }
            else if (w == "u" || w == "U") iU = q - 1;
            else if (w == "v" || w == "V") iV = q - 1;
            else if (w == "w" || w == "W") iW = q - 1;
            else if (w == "dU" || w == "DU" || w == "dV" || w == "DV") { }   // two-layer only, read and ignored
            else { fprintf(stderr, "%s: column %s not supported\n", fn.c_str(), w.c_str()); exit(1); }
        }
        a0.assign((size_t)(nvar - 1) * NY1, 0); a1 = a0;
    }
    bool read_row(double &t, std::vector<double> &a) {
        if (!tk->num(t)) return false;
        for (int q = 0; q < nvar - 1; q++) {
            if (vec) { for (int j = 0; j < NY1; j++) if (!tk->num(a[(size_t)q * NY1 + j])) return false; }
            else { double x; if (!tk->num(x)) return false; for (int j = 0; j < NY1; j++) a[(size_t)q * NY1 + j] = x; }
        }
        return true;
    }
    // values at time t into u, v, z (length NY1); unset columns give 0
    void get(double t, std::vector<double> &u, std::vector<double> &v, std::vector<double> &z) {
        if (first) {
            first = false;
            if (!read_row(t0, a0)) { fprintf(stderr, "boundary file has no data\n"); exit(1); }
            if (!read_row(t1, a1)) { eof = true; t1 = t0; a1 = a0; }
        } else if (!eof) {
            while (!(t >= t0 && t < t1)) {
                t0 = t1; a0.swap(a1);
                if (!read_row(t1, a1)) { eof = true; t1 = t0; a1 = a0; break; }
            }
        }
        double f = (eof || t1 == t0) ? 1.0 : (t - t0) / (t1 - t0);
        auto col = [&](int q, std::vector<double> &o) {
            if (q < 0) { std::fill(o.begin(), o.end(), 0.0); return; }
            for (int j = 0; j < NY1; j++) { size_t k = (size_t)q * NY1 + j; o[j] = a0[k] + (a1[k] - a0[k]) * f; }
        };
        col(iU, u); col(iV, v); col(iZ, z);
    }
};

// initialize.F90 lowlevel_read_hotstart_2d: rows j = 1..ny+1 of (nx+1) fields '(1X,E15.8)' or '(1X,I4)'.
// Fixed-width parsing, because E15.8 drops the 'E' for three-digit exponents (e.g. 0.12345678-100).
static double fortran_real(const char *s, int w) {
    char buf[64]; int n = 0;
    for (int q = 0; q < w && s[q]; q++) if (s[q] != ' ') buf[n++] = s[q];
    buf[n] = 0;
    if (n == 0) return 0;
    char *e; double v = strtod(buf, &e);
    if (*e == '+' || *e == '-') v *= pow(10.0, atoi(e));
    return v;
}
template <typename T> static void read_hotstart(const std::string &var, int fileno, int NX1, int NY1, std::vector<T> &a, bool isint) {
    char fn[256]; snprintf(fn, sizeof fn, "hotstart_%s%06d.dat", var.c_str(), fileno);
    FILE *f = fopen(fn, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", fn); exit(1); }
    int w = isint ? 5 : 16;
    std::vector<char> line((size_t)NX1 * w + 64);
    a.resize((size_t)NX1 * NY1);
    for (int j = 0; j < NY1; j++) {
        if (!fgets(line.data(), (int)line.size(), f)) { fprintf(stderr, "%s: only %d rows\n", fn, j); exit(1); }
        if ((int)strlen(line.data()) < NX1 * w) { fprintf(stderr, "%s: short row %d\n", fn, j + 1); exit(1); }
        for (int i = 0; i < NX1; i++) a[(size_t)j * NX1 + i] = (T)fortran_real(line.data() + (size_t)i * w, w);
    }
    fclose(f);
}

template <typename T> static T *dalloc(size_t n, T val = 0) {
    T *p; CK(cudaMalloc(&p, n * sizeof(T)));
    std::vector<T> h(n, val); CK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

int main(int argc, char **argv) {
    auto P = read_params("params.txt");
    Const h{};
    h.nx = (int)getd(P, "nx", 0); h.ny = (int)getd(P, "ny", 0);
    h.NX1 = h.nx + 1; h.NY1 = h.ny + 1;
    auto fail = [](const char *msg) { fprintf(stderr, "xbnh_gpu: %s\n", msg); exit(1); };
    if (h.ny <= 4) fail("only 2D grids (ny > 4) are supported");
    if ((int)getd(P, "vardx", 0) != 0) fail("vardx = 1 not supported");
    if (getd(P, "alfa", 0) != 0) fail("alfa /= 0 not supported");
    if ((int)getd(P, "nonhq3d", 0) != 0) fail("nonhq3d = 1 not supported");
    if (gets(P, "wavemodel", "nonh") != "nonh") fail("wavemodel must be nonh");
    if ((int)getd(P, "sedtrans", 0) != 0 || (int)getd(P, "morphology", 0) != 0) fail("sedtrans/morphology not supported");
    h.dx = getd(P, "dx", 1); h.dy = getd(P, "dy", 1);
    h.g = getd(P, "g", 9.81); h.rho = getd(P, "rho", 1025);
    h.eps = getd(P, "eps", 0.005); h.eps_sd = getd(P, "eps_sd", 0.5); h.umin = getd(P, "umin", 0);
    if (gets(P, "bedfriction", "manning") != "manning") fail("only bedfriction = manning is supported");
    h.nman = getd(P, "bedfriccoef", 0.02); h.maxcf = getd(P, "maxcf", 0.04); h.mincf = getd(P, "mincf", 0);
    h.nuh = getd(P, "nuh", 0.1); h.nuhv = getd(P, "nuhv", 1); h.smag = (int)getd(P, "smag", 1);
    h.breakviscfac = getd(P, "breakviscfac", 1.5); h.maxbrsteep = getd(P, "maxbrsteep", 0.4);
    h.reformsteep = getd(P, "reformsteep", 0.25 * h.maxbrsteep);
    h.maxfacg = getd(P, "maxfacg", 10); h.rhoa = getd(P, "rhoa", 1.25); h.Cd = getd(P, "cd", 0.002);
    h.secorder = (int)getd(P, "secorder", 1); h.viscosity = 1;
    double dir0 = getd(P, "dir0", 270);
    double th0 = 1.5 * M_PI - dir0 * M_PI / 180;
    while (th0 < -M_PI) th0 += 2 * M_PI;
    while (th0 > M_PI) th0 -= 2 * M_PI;
    h.theta0 = th0;
    std::string back = gets(P, "back", "abs_2d");
    if (back == "wall") h.back = BACK_WALL; else if (back == "abs_1d") h.back = BACK_ABS_1D;
    else if (back == "abs_2d") h.back = BACK_ABS_2D;
    else fail("back must be wall, abs_1d or abs_2d");
    h.left = gets(P, "left", "neumann") == "wall" ? LR_WALL : LR_NEUMANN;
    h.right = gets(P, "right", "neumann") == "wall" ? LR_WALL : LR_NEUMANN;
    h.cyclic = (int)getd(P, "cyclic", 0);
    h.jlo = h.cyclic ? 3 : 2;
    h.uv_jmin = h.cyclic ? 2 : 3;  h.uv_jmax = h.cyclic ? h.ny : h.ny - 1;
    h.w_jmin = h.cyclic ? 1 : 2;   h.w_jmax = h.cyclic ? h.ny : h.ny - 1;
    h.ps_j0 = h.cyclic ? 3 : 2;    h.ps_j1 = h.cyclic ? h.ny - 1 : h.ny;
    std::string wbctype = gets(P, "wbctype", "ts_nonh");
    if (gets(P, "front", "nonh_1d") != "nonh_1d" || (int)getd(P, "arc", 1) != 1) fail("front must be nonh_1d with ARC = 1");
    if (wbctype != "ts_nonh" && wbctype != "reuse") fail("wbctype must be ts_nonh or reuse");
    if (gets(P, "tidetype", "velocity") != "velocity") fail("only tidetype = velocity is supported");
    double CFL = getd(P, "cfl", 0.7), tstop = getd(P, "tstop", 100), tintg = getd(P, "tintg", 1);
    double tstart = getd(P, "tstart", 0);
    double tintp = getd(P, "tintp", tintg);
    double epsi = getd(P, "epsi", -1), taper = getd(P, "taper", 100);
    double Topt = getd(P, "topt", 10), dispc = getd(P, "dispc", -1);
    double sacc = getd(P, "gpu_solver_acc", 0.001);   // BiCGStab; plateau of agreement with XBeach SIP
    int smaxit = (int)getd(P, "gpu_maxit", 200);
    int posdwn = (int)getd(P, "posdwn", 1);
    int hotstart = (int)getd(P, "hotstart", 0), hsno = (int)getd(P, "hotstartfileno", 0);
    int keepuv = (int)getd(P, "gpu_hotstart_keepuv", 1);   // 1: as the hsfix build; 0: trunk r6167 zeroes uu, vv
    int nglobal = (int)getd(P, "nglobalvar", 0);

    // tide (readtide / tide_init): tideloc 0 -> constant zs0, tideloc 1 -> one time series
    std::vector<double> tide_t, tide_z;
    std::string zs0file = gets(P, "zs0file", "");
    int tideloc = (int)getd(P, "tideloc", zs0file.empty() ? 0 : 1);
    if (tideloc == 1) {
        if (zs0file.empty()) fail("tideloc = 1 needs zs0file");
        auto cols = read_columns(zs0file, 2); tide_t = cols[0]; tide_z = cols[1];
    } else if (tideloc != 0) fail("only tideloc = 0 or 1 is supported");
    double zs0c = getd(P, "zs0", 0);
    auto zs01_at = [&](double tt) { return tideloc == 1 ? lin_interp(tide_t, tide_z, tt) : zs0c; };
    // wind (readwind): stationary windv/windth or windfile (t, speed, nautical direction)
    std::vector<double> wind_t, wind_x, wind_y;
    std::string windfile = gets(P, "windfile", "");
    if (!windfile.empty()) {
        auto cols = read_columns(windfile, 3);
        wind_t = cols[0];
        for (size_t q = 0; q < wind_t.size(); q++) {
            double th = (270.0 - cols[2][q]) * M_PI / 180.0;
            wind_x.push_back(cos(th) * cols[1][q]); wind_y.push_back(sin(th) * cols[1][q]);
        }
        if (wind_t.back() < tstop) fail("wind time series shorter than tstop");
    } else {
        double windv = getd(P, "windv", 0), windth = getd(P, "windth", 270);
        double th = (270.0 - windth) * M_PI / 180.0;
        wind_t = {tstop}; wind_x = {windv * cos(th)}; wind_y = {windv * sin(th)};
    }
    // offshore boundary file
    std::string bcfname = "boun_U.bcf";
    bool reuse = wbctype == "reuse";
    if (reuse) {
        std::ifstream f("nhbcflist.bcf"); double b0, b1, trep, hrms; std::string fn;
        if (!(f >> b0 >> b1 >> trep >> hrms >> fn)) fail("cannot read nhbcflist.bcf");
        if (b1 < tstop) fail("nhbcflist.bcf: only one boundary file covering tstop is supported");
        if (fn == "nh_reuse.bcf") fail("nh_reuse.bcf (time offset by bcstarttime) not supported");
        bcfname = fn;
    }
    NHBC bc; bc.open(bcfname, h.NY1);
    h.isSetV = bc.iV >= 0; h.isSetZ = bc.iZ >= 0;
    // output points
    auto ptl = list_after("params.txt", "npoints");
    auto pvl = list_after("params.txt", "npointvar");
    struct Pt { double x, y; std::string name; int i, j; FILE *f; };
    std::vector<Pt> pts; std::vector<int> pvar;
    for (auto &l : ptl) {
        std::istringstream is(l); Pt p{}; is >> p.x >> p.y >> p.name;
        p.i = std::min(std::max((int)lround(p.x / h.dx) + 1, 1), h.NX1);
        p.j = std::min(std::max((int)lround(p.y / h.dy) + 1, 1), h.NY1);
        pts.push_back(p);
    }
    const char *pvn[] = {"zs", "uu", "vv", "u", "v", "hh", "ws", "wb", "pres"};
    for (auto &l : pvl) {
        std::istringstream is(l); std::string v; is >> v;
        int code = -1; for (int q = 0; q < 9; q++) if (v == pvn[q]) code = q;
        if (code < 0) { fprintf(stderr, "xbnh_gpu: point variable %s not supported\n", v.c_str()); return 1; }
        pvar.push_back(code);
    }
    if (!pts.empty() && pvar.empty()) fail("npoints > 0 needs npointvar");
    CK(cudaMemcpyToSymbol(c, &h, sizeof(Const)));

    int NX1 = h.NX1, NY1 = h.NY1, nx = h.nx, ny = h.ny;
    size_t N = (size_t)NX1 * NY1;
    auto I = [&](int i, int j) { return (size_t)(j - 1) * NX1 + (i - 1); };
    auto cyc_host = [&](auto &a) {   // periodic halo rows (cyclic)
        if (!h.cyclic) return;
        const int dst[4] = {1, 2, ny, ny + 1}, src[4] = {ny - 2, ny - 1, 3, 4};
        for (int r = 0; r < 4; r++) for (int i = 1; i <= NX1; i++) a[I(i, dst[r])] = a[I(i, src[r])];
    };
    // flow_init / hotstart_init_1 / hotstart_init_2
    std::vector<real> zb(N), zs(N), uu(N, 0), vv(N, 0), ws(N, 0), wb(N, 0), umean(N, 0), vmean(N, 0);
    std::vector<int> brk(N, 0);
    if (hotstart) {
        read_hotstart("zs", hsno, NX1, NY1, zs, false);
        read_hotstart("zb", hsno, NX1, NY1, zb, false);
        if (keepuv) { read_hotstart("uu", hsno, NX1, NY1, uu, false); read_hotstart("vv", hsno, NX1, NY1, vv, false); }
        read_hotstart("breaking", hsno, NX1, NY1, brk, true);
        read_hotstart("wb", hsno, NX1, NY1, wb, false);
        read_hotstart("ws", hsno, NX1, NY1, ws, false);
        read_hotstart("umean", hsno, NX1, NY1, umean, false);
        read_hotstart("vmean", hsno, NX1, NY1, vmean, false);
        cyc_host(zs); cyc_host(uu); cyc_host(vv); cyc_host(ws); cyc_host(wb); cyc_host(brk);
    } else {
        std::ifstream f(gets(P, "depfile", "z.grd"));
        if (!f) fail("cannot open depfile");
        for (size_t k = 0; k < N; k++) { double v; f >> v; zb[k] = posdwn == 1 ? -v : v; }
    }
    double zs01_0 = zs01_at(0.0);
    std::vector<real> zs0(N, (real)zs01_0), hh(N), hu(N), hv(N), wc(N), zbu(N), zbv(N);
    if (!hotstart) for (size_t k = 0; k < N; k++) zs[k] = std::max((double)zb[k] + 2.220446049250313e-16, (double)zs0[k]);
    for (size_t k = 0; k < N; k++) hh[k] = std::max(zs[k] - zb[k], h.eps);
    for (int j = 1; j <= NY1; j++) {
        for (int i = 1; i <= nx; i++) hu[I(i, j)] = std::max(zs[I(i, j)], zs[I(i + 1, j)]) - std::max(zb[I(i, j)], zb[I(i + 1, j)]);
        hu[I(NX1, j)] = hu[I(nx, j)];
    }
    for (int i = 1; i <= NX1; i++) {
        for (int j = 1; j <= ny; j++) hv[I(i, j)] = std::max(zs[I(i, j)], zs[I(i, j + 1)]) - std::max(zb[I(i, j)], zb[I(i, j + 1)]);
        hv[I(i, NY1)] = hv[I(i, ny)];
    }
    for (int j = 1; j <= NY1; j++) {   // zuzv: static bed levels at u/v points
        for (int i = 1; i <= nx; i++) zbu[I(i, j)] = std::max(zb[I(i, j)], zb[I(std::min(nx, i) + 1, j)]);
        zbu[I(NX1, j)] = zb[I(NX1, j)];
    }
    for (int i = 1; i <= NX1; i++) {
        for (int j = 1; j <= ny; j++) zbv[I(i, j)] = std::max(zb[I(i, j)], zb[I(i, std::min(ny, j) + 1)]);
        zbv[I(i, NY1)] = zb[I(i, NY1)];
    }
    double sigma = 2 * M_PI / Topt;
    for (size_t k = 0; k < N; k++) {     // nonh_init_wcoef (zs0 from tide_init)
        if (dispc > 0) { wc[k] = dispc; continue; }
        double dd = std::max((double)zs0[k] - zb[k], (double)h.eps);
        double kk = disper(sigma, dd, h.g);
        wc[k] = 1.0 / (4.0 * h.g / (dd * sigma * sigma) - 4.0 / ((kk * dd) * (kk * dd)));
    }
    std::vector<real> um1(NY1), vm1(NY1), umN(NY1), vmN(NY1);
    for (int j = 1; j <= NY1; j++) {
        um1[j - 1] = umean[I(1, j)]; vm1[j - 1] = vmean[I(1, j)];
        umN[j - 1] = umean[I(nx, j)]; vmN[j - 1] = vmean[I(nx, j)];
    }

    Dev d{};
    auto R = [&](real v = 0) { return dalloc<real>(N, v); };
    auto Iz = [&](int v = 0) { return dalloc<int>(N, v); };
    d.zb = R(); d.zs = R(); d.hh = R(); d.uu = R(); d.vv = R(); d.hu = R(); d.hv = R(); d.hum = R(); d.hvm = R();
    d.qx = R(); d.qy = R(); d.wetu = Iz(); d.wetv = Iz(); d.wetz = Iz();
    d.dzsdx = R(); d.dzsdy = R(); d.ududx = R(); d.vdudy = R(); d.vdvdy = R(); d.udvdx = R(); d.nuh = R();
    d.viscu = R(); d.viscv = R(); d.taubx = R(); d.tauby = R(); d.cfu = R(); d.cfv = R();
    d.vu = R(); d.uv = R(); d.ueu = R(); d.vev = R(); d.vmageu = R(); d.vmagev = R(); d.uu_old = R(); d.vv_old = R();
    d.breaking = Iz(); d.brk_new = Iz(); d.nonhU = Iz(1); d.nonhV = Iz(1); d.nonhZ = Iz(1);
    d.pres = R(); d.ws = R(); d.wb = R(); d.Wm = R(); d.Wm_old = R(); d.wcoef = R();
    d.au0 = R(); d.au1 = R(); d.av0 = R(); d.av1 = R();
    for (int q = 0; q < 5; q++) { d.awb[q] = R(); d.aws[q] = R(); d.mat[q] = R(q == 0 ? 1 : 0); }
    d.awbr = R(); d.awsr = R(); d.rhs = R(); d.dp = R();
    d.zbu = R(); d.zbv = R(); d.zsu = R(); d.zsv = R(); d.w1 = R(); d.w2 = R(); d.w3 = R(); d.w4 = R();
    d.r = R(); d.rh = R(); d.p = R(); d.v = R(); d.s = R(); d.t = R(); d.y = R(); d.z = R();
    d.sc = dalloc<double>(16); d.dtmin = dalloc<unsigned long long>(1);
    d.zs0 = R(); d.ui = dalloc<real>(NY1); d.vi = dalloc<real>(NY1); d.zi = dalloc<real>(NY1);
    d.um1 = dalloc<real>(NY1); d.vm1 = dalloc<real>(NY1); d.umN = dalloc<real>(NY1); d.vmN = dalloc<real>(NY1);
    d.msum = dalloc<double>(4); d.skey = dalloc<unsigned long long>(2); d.stale = dalloc<double>(2);
    double *dout; CK(cudaMalloc(&dout, N * sizeof(double)));
    auto H2D = [&](real *dst, std::vector<real> &src) { CK(cudaMemcpy(dst, src.data(), src.size() * sizeof(real), cudaMemcpyHostToDevice)); };
    H2D(d.zb, zb); H2D(d.zs, zs); H2D(d.hh, hh); H2D(d.hu, hu); H2D(d.hv, hv); H2D(d.wcoef, wc); H2D(d.zbu, zbu); H2D(d.zbv, zbv);
    H2D(d.uu, uu); H2D(d.vv, vv); H2D(d.ws, ws); H2D(d.wb, wb); H2D(d.zs0, zs0);
    H2D(d.um1, um1); H2D(d.vm1, vm1); H2D(d.umN, umN); H2D(d.vmN, vmN);
    CK(cudaMemcpy(d.breaking, brk.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    { std::vector<real>().swap(umean); std::vector<real>().swap(vmean); std::vector<real>().swap(wc); }

    dim3 B(32, 8), G((NX1 + 31) / 32, (NY1 + 7) / 8);
    int bj = (NY1 + 255) / 256, bi = (NX1 + 255) / 256;
    auto cyc = [&](std::initializer_list<real *> fr, std::initializer_list<int *> fi = {}) {
        if (!h.cyclic) return;
        CycSet s{}; s.n = 0;
        for (auto p : fr) { s.p[s.n] = p; s.isint[s.n++] = 0; }
        for (auto p : fi) { s.p[s.n] = p; s.isint[s.n++] = 1; }
        k_cyc<<<bi, 256>>>(s);
    };

    // output
    // global output: the nglobalvar names listed in params.txt, one <name>.dat each
    std::vector<std::string> gnames;
    for (auto &l : list_after("params.txt", "nglobalvar")) { std::istringstream is(l); std::string v; is >> v; gnames.push_back(v); }
    std::vector<FILE *> gfiles;
    for (auto &v : gnames) {
        const char *ok[] = {"zs", "u", "v", "uu", "vv", "ws", "wb", "hh", "pres", "breaking", "zb", "nuh", "zs0"};
        if (std::find_if(std::begin(ok), std::end(ok), [&](const char *q) { return v == q; }) == std::end(ok)) {
            fprintf(stderr, "xbnh_gpu: global variable %s not supported\n", v.c_str()); return 1;
        }
        gfiles.push_back(fopen((v + ".dat").c_str(), "wb"));
    }
    if (nglobal != (int)gnames.size()) fail("nglobalvar does not match the listed variables");
    for (auto &p : pts) { p.f = fopen((p.name + ".dat").c_str(), "wb"); if (!p.f) fail("cannot open point output file"); }
    int np = (int)pts.size(), nv = (int)pvar.size();
    int *dpij = nullptr, *dpvar = nullptr; double *dpout = nullptr;
    if (np > 0) {
        std::vector<int> pij; for (auto &p : pts) { pij.push_back(p.i); pij.push_back(p.j); }
        dpij = dalloc<int>(2 * np); dpvar = dalloc<int>(nv); dpout = dalloc<double>((size_t)np * nv);
        CK(cudaMemcpy(dpij, pij.data(), pij.size() * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpvar, pvar.data(), nv * sizeof(int), cudaMemcpyHostToDevice));
    }
    FILE *flog = fopen("xbgpu_log.txt", "w");
    std::vector<double> hbuf(N), pbuf((size_t)np * nv + 1);
    auto output_global = [&](int uvzero) {   // uvzero: XBeach writes u = v = 0 before the first flow step
        unsigned nb = (unsigned)((N + 255) / 256);
        for (size_t q = 0; q < gnames.size(); q++) {
            const std::string &v = gnames[q];
            if (v == "u" || v == "v") {
                if (uvzero) CK(cudaMemset(dout, 0, N * 8));
                else if (v == "u") k_out_u<<<G, B>>>(d, dout); else k_out_v<<<G, B>>>(d, dout);
            } else if (v == "breaking") k_copy_out_int<<<nb, 256>>>(d.breaking, dout, N);
            else {
                real *a = v == "zs" ? d.zs : v == "uu" ? d.uu : v == "vv" ? d.vv : v == "ws" ? d.ws : v == "wb" ? d.wb
                        : v == "hh" ? d.hh : v == "pres" ? d.pres : v == "zb" ? d.zb : v == "nuh" ? d.nuh : d.zs0;
                k_copy_out<<<nb, 256>>>(a, dout, N);
            }
            CK(cudaMemcpy(hbuf.data(), dout, N * 8, cudaMemcpyDeviceToHost)); fwrite(hbuf.data(), 8, N, gfiles[q]);
        }
    };
    auto output_points = [&](double tt, int uvzero) {
        k_points<<<(np * nv + 127) / 128, 128>>>(d, dpij, np, dpvar, nv, uvzero, dpout);
        CK(cudaMemcpy(pbuf.data(), dpout, (size_t)np * nv * 8, cudaMemcpyDeviceToHost));
        for (int p = 0; p < np; p++) { fwrite(&tt, 8, 1, pts[p].f); fwrite(&pbuf[(size_t)p * nv], 8, nv, pts[p].f); }
    };
    double t = tstart;
    long nstep = 0, itsum = 0, itmax = 0, nsolve = 0, nonconv = 0;
    double tsolve = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto wall0 = std::chrono::steady_clock::now();
    double dt = 0, dt_first = 0;
    const double BIG = 1e300;
    double next_g = nglobal > 0 ? tstart : BIG, next_p = np > 0 ? tstart : BIG;
    if (nglobal > 0) { output_global(1); next_g += tintg; }
    if (np > 0) { output_points(t, 1); next_p += tintp; }
    // XBeach also steps onto the global-output times when no global variables are written
    double next_g_clock = tstart + tintg;
    auto tnext_of = [&]() { return std::min(std::min(std::min(next_g, next_p), next_g_clock), tstop); };
    double tnext = tnext_of();
    std::vector<double> bu(NY1), bv(NY1), bz(NY1);
    std::vector<real> ru(NY1), rv(NY1), rz(NY1);
    printf("xbnh_gpu %s: nx=%d ny=%d cells=%zu cyclic=%d back=%s wbctype=%s hotstart=%d points=%d\n",
           sizeof(real) == 8 ? "fp64" : "fp32", nx, ny, N, h.cyclic, back.c_str(), wbctype.c_str(), hotstart, np);
    fflush(stdout);
    while (t < tstop - 1e-9) {
        k_wetcells<<<G, B>>>(d);
        // compute_dt
        if (nstep == 0) {
            double m = 0;
            for (int j = 2; j <= ny; j++)
                for (int i = 2; i <= nx; i++) {
                    size_t k = I(i, j);
                    double v = hotstart ? sqrt(std::max((double)zs[k] - zb[k], (double)h.eps) * h.g) + sqrt((double)uu[k] * uu[k] + (double)vv[k] * vv[k])
                                        : sqrt((double)hh[k] * h.g);
                    m = std::max(m, v);
                }
            dt = CFL * std::min(h.dx, h.dy) / m;
            if (tnext > dt) { long n = (long)ceil((tnext - t) / dt); dt = (tnext - t) / n; }
        } else {
            unsigned long long big = (unsigned long long)__builtin_bit_cast(long long, 1e300);
            CK(cudaMemcpy(d.dtmin, &big, 8, cudaMemcpyHostToDevice));
            k_dt<<<G, B>>>(d);
            unsigned long long bits; CK(cudaMemcpy(&bits, d.dtmin, 8, cudaMemcpyDeviceToHost));
            dt = __builtin_bit_cast(double, (long long)bits) * CFL * 0.5;
        }
        if (nstep == 0) dt_first = dt;
        if (!(dt > 1e-4 * dt_first)) {
            fprintf(stderr, "xbnh_gpu: time step collapsed (dt = %.3e s at t = %.3f s); stopping\n", dt, t);
            fprintf(flog, "ABORTED dt collapse at t = %.3f\n", t); fclose(flog); return 2;
        }
        t += dt;
        if (t >= tnext) { dt -= (t - tnext); t = tnext; }
        // wave_bc: offshore signal at t (ts_nonh or reuse; taper only for reuse)
        bc.get(t, bu, bv, bz);
        double tf = (reuse && taper > 0) ? std::min(t / taper, 1.0) : 1.0;
        for (int j = 0; j < NY1; j++) { ru[j] = (real)(bu[j] * tf); rv[j] = (real)bv[j]; rz[j] = (real)(bz[j] * tf); }
        H2D(d.ui, ru); H2D(d.vi, rv); H2D(d.zi, rz);
        // flow_bc
        k_tide<<<G, B>>>(d, (real)zs01_at(t));
        real factime = (real)(epsi < 0 ? dt / 60.0 : epsi);
        k_bc_msum<<<1, 1024>>>(d, 0); k_bc_mapply<<<bj, 256>>>(d, factime, 0);
        k_bc_front<<<bj, 256>>>(d);
        if (!h.cyclic) k_bc_front_corners<<<1, 32>>>(d);
        k_bc_msum<<<1, 1024>>>(d, 1); k_bc_mapply<<<bj, 256>>>(d, factime, 1);
        k_bc_back1<<<bj, 256>>>(d, (real)dt);
        k_bc_back2<<<bj, 256>>>(d);
        if (!h.cyclic) { k_bc_back_corners<<<1, 32>>>(d); k_bc_lat<<<bi, 256>>>(d); }
        cyc({d.uu, d.vv, d.zs, d.ws});
        real wsu = (real)lin_interp(wind_t, wind_x, t), wnv = (real)lin_interp(wind_t, wind_y, t);
        // flow: explicit part
        k_slopes_rough<<<G, B>>>(d);
        k_advection<<<G, B>>>(d);
        if (h.smag) {
            k_smag<<<G, B>>>(d);
            cyc({d.nuh});
            if (!h.cyclic) k_nuh_bc_j<<<bi, 256>>>(d);
            k_nuh_bc_i<<<bj, 256>>>(d);
        }
        k_nuh_const_brk<<<G, B>>>(d);
        k_viscu<<<G, B>>>(d);
        k_viscv1<<<G, B>>>(d);
        if (!h.cyclic) k_viscv_bc<<<bi, 256>>>(d);
        k_viscv2<<<G, B>>>(d);
        k_momentum<<<G, B>>>(d, (real)dt, wsu, wnv);
        if (!h.cyclic) { k_lat_uv<<<bi, 256>>>(d, (real)dt, wsu, wnv); k_lat_v_apply<<<bi, 256>>>(d); }
        cyc({d.uu, d.vv});
        // nonh_cor(0): predictor
        k_nonhUV<<<G, B>>>(d);
        k_break<<<G, B>>>(d); k_break_apply<<<G, B>>>(d);
        cyc({d.pres}, {d.breaking, d.nonhZ, d.nonhU, d.nonhV});
        k_wm_adv<<<G, B>>>(d, (real)dt); k_wm_zero<<<G, B>>>(d);
        cyc({d.Wm_old, d.ws, d.wb, d.pres});
        k_wm_visc_coef<<<G, B>>>(d, (real)dt);
        cyc({d.Wm});
        if (h.secorder) {
            k_pred_pres<<<G, B>>>(d, (real)dt);
            cyc({d.uu, d.vv, d.Wm});
            k_advW_flux<<<G, B>>>(d); k_advW_apply<<<G, B>>>(d, (real)dt);
            cyc({d.Wm});
            // flow_secondorder_advUV
            k_advUV_flux<<<G, B>>>(d); k_advUV_apply<<<G, B>>>(d, (real)dt);
            cyc({d.uu, d.vv});
        }
        // nonh_cor(1): corrector
        k_zuzv<<<G, B>>>(d);
        k_build<<<G, B>>>(d, (real)dt);
        cudaEventRecord(e0);
        CK(cudaMemset(d.sc, 0, 16 * sizeof(double)));
        k_s_init<<<G, B>>>(d);
        double sch[16]; CK(cudaMemcpy(sch, d.sc, sizeof(sch), cudaMemcpyDeviceToHost));
        double bnorm = sqrt(sch[8]);
        int it = 0;
        if (bnorm > 0) {
            double tgt = sacc * sacc * sch[8];
            CK(cudaMemcpy(d.sc + 13, &tgt, 8, cudaMemcpyHostToDevice));
            const int batch = 4;
            for (it = 1; it <= smaxit; it++) {
                k_s_beta<<<1, 32>>>(d, it == 1);
                k_s_p<<<G, B>>>(d);
                k_s_Ay<<<G, B>>>(d);
                k_s_s<<<G, B>>>(d);
                k_s_Az<<<G, B>>>(d);
                k_s_omega<<<1, 32>>>(d);
                k_s_x<<<G, B>>>(d);
                k_s_roll<<<1, 32>>>(d);
                if (it % batch == 0 || it == smaxit) {
                    double f[2]; CK(cudaMemcpy(f, d.sc + 11, 16, cudaMemcpyDeviceToHost));
                    if (f[0] != 0) { it = (int)f[1]; break; }
                }
            }
            if (it > smaxit) { nonconv++; it = smaxit; }
        } else CK(cudaMemset(d.dp, 0, N * sizeof(real)));
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); tsolve += ms * 1e-3;
        itsum += it; itmax = std::max(itmax, (long)it); nsolve++;
        cyc({d.dp});
        k_correct<<<G, B>>>(d);
        cyc({d.uu, d.vv, d.pres});
        CK(cudaMemsetAsync(d.skey, 0, 2 * sizeof(unsigned long long)));
        k_stale_find<<<G, B>>>(d); k_stale_val<<<1, 32>>>(d);
        k_wswb<<<G, B>>>(d);
        cyc({d.ws, d.wb});
        if (!h.cyclic) k_wswb_bc_j<<<bi, 256>>>(d);
        k_wswb_bc_i<<<bj, 256>>>(d); k_wmold<<<G, B>>>(d);
        // hu, hv, fluxes, continuity, derived fields
        k_huhv<<<G, B>>>(d);
        k_continuity<<<G, B>>>(d, (real)dt);
        cyc({d.zs});
        k_post<<<G, B>>>(d);
        nstep++;
        if (t >= tnext - 1e-9) {
            if (t >= next_g - 1e-9) { output_global(0); next_g += tintg; }
            if (t >= next_p - 1e-9) { output_points(t, 0); next_p += tintp; }
            if (t >= next_g_clock - 1e-9) next_g_clock += tintg;
            tnext = tnext_of();
        }
        if (nstep % 2000 == 0) {
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
            printf("t = %8.2f s  steps %ld  dt %.4f  wall %.1f s  solver it mean %.1f\n", t, nstep, dt, el, (double)itsum / nsolve);
            fflush(stdout);
        }
    }
    CK(cudaDeviceSynchronize());
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    for (auto f : gfiles) fclose(f);
    for (auto &p : pts) fclose(p.f);
    fprintf(flog, "precision %s\nnx %d\nny %d\ncells %zu\ncyclic %d\nsteps %ld\nwall_s %.3f\nsolver_s %.3f\n"
                  "solver_it_mean %.2f\nsolver_it_max %ld\nsolver_nonconv %ld\nunit_speed_s %.4e\n",
            sizeof(real) == 8 ? "fp64" : "fp32", nx, ny, N, h.cyclic, nstep, wall, tsolve, (double)itsum / std::max(nsolve, 1L), itmax,
            nonconv, wall / ((double)nstep * N));
    fclose(flog);
    printf("done: %ld steps, wall %.2f s (solver %.2f s), %.3e s per cell-step, solver it mean %.1f max %ld nonconv %ld\n",
           nstep, wall, tsolve, wall / ((double)nstep * N), (double)itsum / std::max(nsolve, 1L), itmax, nonconv);
    return 0;
}
