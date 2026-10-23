// k_extra.cuh: discharges (boundaryconditions.F90 discharge_boundary_h/_v, initialize.F90 discharge_init),
// rainfall (rainfall.F90), drifters (drifters.F90, initialize.F90 drifter_init).
#pragma once

// ---- discharges: line discharges along a u or v row (pntdisch = 0), point sources (pntdisch = 1)
struct Discharge { int pnt, m1, n1, m2, n2; std::vector<double> q; };
__global__ void k_disch_v(int m1, int n1, real qdt) {
    if (threadIdx.x) return;
    d.zs[ID(m1, n1)] += qdt * dsdnzi(m1, n1);
}
// line discharge: n1 == n2 -> along v points (m1..m2, n1); m1 == m2 -> along u points (m1, n1..n2)
__global__ void k_disch_h(int alongv, int a0, int a1, int fixed, int isborder, int shift, real qnow, int nfix) {
    __shared__ double sA[256];
    int tid = threadIdx.x;
    real *h = alongv ? d.hv : d.hu, *vel = alongv ? d.vv : d.uu, *q = alongv ? d.qy : d.qx;
    if (isborder) {   // border segments take the water depth of the cell inside the domain
        for (int p = a0 + tid; p <= a1; p += blockDim.x) {
            size_t k = alongv ? ID(p, fixed) : ID(fixed, p);
            size_t kin = alongv ? ID(p, fixed + shift) : ID(fixed + shift, p);
            h[k] = d.hh[kin];
        }
        __syncthreads();
    }
    double a = 0;
    for (int p = a0 + tid; p <= a1; p += blockDim.x) {
        size_t k = alongv ? ID(p, fixed) : ID(fixed, p);
        a += pow((double)h[k], 1.5) * (alongv ? dsv(p, fixed) : dnu(fixed, p));
    }
    sA[tid] = a;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sA[tid] += sA[tid + s]; __syncthreads(); }
    real CONST = qnow / fmax((real)sA[0], c.eps);
    for (int p = a0 + tid; p <= a1; p += blockDim.x) {
        size_t k = alongv ? ID(p, fixed) : ID(fixed, p);
        vel[k] = CONST * sqrt(h[k]);
        q[k] = CONST * sqrt(h[k]) * h[k];
    }
    (void)nfix;
}

// ---- rainfall from a file: linear interpolation in time between two maps
__global__ void k_rain(const real *r0, const real *r1, real f) {
    IJ
    size_t k = ID(i, j);
    d.rainrate[k] = r0[k] + (r1[k] - r0[k]) * f;
}

// ---- drifters: positions in grid index space (idrift, jdrift), advected with uu, vv
__global__ void k_drift(double *idr, double *jdr, const double *tb, const double *te, int n, double t, real dt) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;
    if (!(t > tb[q] && t < te[q])) return;
    // XBeach: iu = nint(idrift - 0.d5) etc., where 0.d5 = 0, so all four indices are rounded positions
    int iu = (int)llround(idr[q]), ju = (int)llround(jdr[q]), iv = iu, jv = ju;
    if (iu >= 1 && iu <= c.nx && ju >= 1 && ju <= c.ny && iv >= 1 && iv <= c.nx && jv >= 1 && jv <= c.ny) {
        double di = d.uu[ID(iu, ju)] / dsu(iu, ju) * dt;
        double dj = d.vv[ID(iv, jv)] / dnv(iv, jv) * dt;
        idr[q] += di; jdr[q] += dj;
    }
}
