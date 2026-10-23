// common.cuh: precision, error checks, the constant-memory parameter block (Const c), the device array set
// (Dev d), index macros and grid-metric accessors shared by all kernels.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <limits>
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

// XBeach option codes (paramsconst.F90 values where they exist)
enum { BACK_WALL = 0, BACK_ABS_1D = 1, BACK_ABS_2D = 2 };
enum { LR_NEUMANN = 0, LR_WALL = 1, LR_NO_ADVEC = 2, LR_NEUMANN_V = 3, LR_ABS_1D = 4 };
enum { BF_CHEZY = 0, BF_CF = 1, BF_WC = 2, BF_MANNING = 3, BF_WCG = 4 };
enum { TT_INSTANT = 0, TT_VELOCITY = 1, TT_HYBRID = 2 };
enum { PR_LAND = 0, PR_SEA = 1 };

struct Const {
    // grid
    int nx, ny, NX1, NY1, oned;          // oned: ny == 0 ("superfast 1D" code paths)
    int uni;                             // metrics constant over the grid (values below)
    real m_dsu, m_dsv, m_dsz, m_dsc, m_dnu, m_dnv, m_dnz, m_dnc, m_dsdnui, m_dsdnvi, m_dsdnzi;
    real m_alfau, m_alfav, m_alfaz;
    real par_dy;                         // par%dy as read (-1 when not given), used by the 1D branches
    // physics and numerics
    real g, rho, eps, eps_sd, umin, px;
    real maxfacg, nuh, nuhv, rhoa, Cd, fc, lwave;
    int smag, viscosity, secorder, oldhu;
    int bedfriction; real bedfriccoef, maxcf, mincf;
    // non-hydrostatic
    int nonhq3d, nhbreaker; real nhlay, breakviscfac, maxbrsteep, reformsteep, kdmin;
    int xbg_brk; real secbrsteep, breakvisclen;   // XBeach-G breaking (useXBeachGSettings, nhbreaker 2, 3)
    // boundaries
    int back, left, right, arc, cyclic, wbcoff, tidetype, front_nonh;
    real order, theta0;
    // cyclic index bounds (single-domain periodic halo scheme, see k_cyc)
    int jlo, uv_jmin, uv_jmax, w_jmin, w_jmax, ps_j0, ps_j1;
    // processes
    int wind, ships, vegetation, gwflow, rainfall, sedtrans;
    int veguntow;
    int fric_acc, fric_turb, dynamrough; real ci, phiacc, gamma_turb, D90top, nmanmin, droot, dstem;
    int sed_ngd; real sed_morfac;   // 1D time step: sediment celerity (0 fractions without sediment)
    // groundwater
    int gwnonh, gwscheme, gwheadmodel, gwland, gwfi;
    real kx, ky, kz, dwetlayer, por, D50top, gwReturb, gwvcr, connectcrit;
};
__constant__ Const c;

struct Dev {
    // metrics (allocated only when c.uni == 0)
    real *dsu, *dsv, *dsz, *dsc, *dnu, *dnv, *dnz, *dnc, *dsdnui, *dsdnvi, *dsdnzi, *alfau, *alfav, *alfaz;
    // row/column spacings used by the one-layer non-hydrostatic routines (nonh_init: dsu(:,1), dsz(:,1),
    // dnz(1,:), dnv(1,:)) and their inverses; always allocated
    real *dxu, *dxz, *dyz, *dyv, *ddxu, *ddxz, *ddyz, *ddyv;
    // flow state
    real *zb, *zs, *hh, *uu, *vv, *hu, *hv, *hum, *hvm, *qx, *qy;
    int *wetu, *wetv, *wetz;
    real *dzsdx, *dzsdy, *ududx, *vdudy, *vdvdy, *udvdx, *nuh, *viscu, *viscv, *taubx, *tauby, *cfu, *cfv;
    real *vu, *uv, *ueu, *vev, *vmageu, *vmagev, *uu_old, *vv_old, *u, *v, *dzsdt;
    real *ve;   // XBeach s%ve: cell-centre Eulerian v, computed before vev is updated (lags one flow step)
    real *zs0, *dzs0dn, *umean, *vmean, *bfc, *windsu, *windnv, *maxzs, *minzs;
    real *ph, *Fvegu, *Fvegv, *taubx_add, *tauby_add, *infil, *rainrate;    // optional (nullptr when off)
    // vegetation (sectioned fields: layer m at m*(nx+1)*(ny+1)); zb0 initial bed
    int *nsecveg, *isCanopy;
    real *Cdveg, *ahveg, *bveg, *Nveg, *Kp_can, *beta_can, *cf_can, *cm_can, *lp_can, *ucan, *vcan, *Fcanu, *Fcanv, *zb0;
    // groundwater (groundwater.F90 s% fields and the saved work arrays of gwflow)
    real *gwhead, *gwheadb, *gwlevel, *gwheight, *gwu, *gwv, *gwqx, *gwqy, *gww, *gwbottom, *dinfil, *gw0back, *gwcurv;
    real *gKx, *gKy, *gKz, *gKzinf, *gwhu, *gwhv, *gwheadtop, *Kzinf, *Kxupd, *Kyupd, *Kzupd, *KxN, *KyN, *KzN;
    real *fracdt, *gwratio, *dynpres, *infiluncon, *infilcon, *infilhorgw, *infilhorsw;
    int *gwconn;
    real *kru, *krv, *ueuf, *vevf, *ueuold, *vevold, *nman0, *D50topA, *D90topA, *kbA, *sedero; int *isveggie;
    real *wscrit;
    real *Susg, *Subg;   // sediment fluxes (first fraction at offset 0), for the 1D time step
    // non-hydrostatic
    int *breaking, *brk_new, *nonhU, *nonhV, *nonhZ;
    real *pres, *ws, *wb, *Wm, *Wm_old, *wcoef, *au0, *au1, *av0, *av1;
    real *awb[5], *awbr, *aws[5], *awsr, *mat[5], *rhs, *dp;
    real *zbu, *zbv, *zsu, *zsv;
    real *dU, *dV, *dU0, *dV0, *adu0, *adu1, *adv0, *adv1, *Wm0, *omega;    // nonhq3d
    // scratch and persistent work arrays (flow_secondorder wrk1/wrk2)
    real *w1, *w2, *w3, *w4, *wrk1p;
    // pressure solvers
    real *r, *rh, *p, *v_, *s, *t, *y, *z;
    double *sc;
    // offshore boundary signal (length ny+1)
    real *ui, *vi, *zi, *wi, *dui, *dvi;
    double *msum;
    unsigned long long *dtmin, *skey;
    double *stale;
};
__constant__ Dev d;

#define ID(i, j) ((size_t)((j) - 1) * c.NX1 + ((i) - 1))
#define IJ int i = blockIdx.x * blockDim.x + threadIdx.x + 1; \
           int j = blockIdx.y * blockDim.y + threadIdx.y + 1; \
           if (i > c.NX1 || j > c.NY1) return;
#define IN(i0, i1, j0, j1) (i >= (i0) && i <= (i1) && j >= (j0) && j <= (j1))

// loop bounds of XBeach ranges_init (serial): uu i 2..nx-1, vv i 2..nx, zs i 2..nx; j 2..ny (vv 2..ny-1),
// or j = 1 in 1D
#define JU0 (c.oned ? 1 : 2)
#define JU1 (c.oned ? 1 : c.ny)
#define JV0 (c.oned ? 1 : 2)
#define JV1 (c.oned ? 1 : c.ny - 1)
#define JZ0 (c.oned ? 1 : 2)
#define JZ1 (c.oned ? 1 : c.ny)
#define IN_UU IN(2, c.nx - 1, JU0, JU1)
#define IN_VV IN(2, c.nx, JV0, JV1)
#define IN_ZS IN(2, c.nx, JZ0, JZ1)

// grid metrics (spaceparams.F90 gridprops)
#define METRIC(name) \
    __device__ __forceinline__ real name(int i, int j) { return c.uni ? c.m_##name : d.name[ID(i, j)]; }
METRIC(dsu) METRIC(dsv) METRIC(dsz) METRIC(dsc) METRIC(dnu) METRIC(dnv) METRIC(dnz) METRIC(dnc)
METRIC(dsdnui) METRIC(dsdnvi) METRIC(dsdnzi) METRIC(alfau) METRIC(alfav) METRIC(alfaz)
#undef METRIC
// cos and sin of the difference of two grid angles: exactly 1 and 0 on a uniform grid (constant alfa), so the
// trigonometry and the cross-velocity loads are skipped there without changing a bit of the result.
__device__ __forceinline__ real cos_da(real da) { return c.uni ? (real)1 : cos(da); }
__device__ __forceinline__ real sin_da(real da) { return c.uni ? (real)0 : sin(da); }
#define ROTP(U, V, DA) (c.uni ? (U) : (U) * cos(DA) + (V) * sin(DA))
#define ROTM(U, V, DA) (c.uni ? (U) : (U) * cos(DA) - (V) * sin(DA))

__device__ __forceinline__ real minmod(real d1, real d2) {
    if (d1 * d2 <= 0) return 0;
    if (d1 > 0) return fmin(d1, d2);
    if (d1 < 0) return fmax(d1, d2);
    return 0;
}

// constants.F90 dFill = -dble(huge(0.0)); iFill = -huge(0)
#define DFILL ((real)-3.4028234663852886e38)
#define IFILL (-2147483647)

// Fortran single-precision literals that XBeach mixes into double expressions
#define F32(x) ((real)(double)(float)(x))

// launch helpers (set in main)
struct Launch { dim3 B, G; unsigned bi, bj; };
static Launch L;

static inline void fail(const char *msg) { fprintf(stderr, "xbnh_gpu: %s\n", msg); exit(1); }
static inline void fail(const std::string &msg) { fail(msg.c_str()); }
