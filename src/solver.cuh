// solver.cuh: pressure-correction solvers.
// 2D: Jacobi-preconditioned BiCGStab on the 5-diagonal system of nonh_1lay_cor / nonh_2lay_cor_3d (XBeach uses
//     SIP, an incomplete-LU sweep that is inherently sequential), on i = 2..nx, j = ps_j0..ps_j1 (periodic in j
//     with cyclic laterals); dp = 0 elsewhere.
// 1D: parallel cyclic reduction of the tri-diagonal system over i = 1..nx+1 (XBeach solver_tridiag solves the
//     same system with the Thomas algorithm).
#pragma once

// ---- BiCGStab. sc: 0 rho, 1 rho_old, 2 alpha, 3 omega, 4 (rhat,v), 5 (t,s), 6 (t,t), 7 (r,r), 8 (b,b), 9 (rhat,r),
// 10 beta, 11 converged flag, 12 iteration count, 13 target |r|^2. Kernels return at once when converged, so the
// host launches iterations in batches and synchronises once per batch.
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
__device__ __forceinline__ real Ax(const real *x, int i, int j) {
    size_t k = ID(i, j);
    int js = j - 1, jn = j + 1;                   // cyclic: rows 2 and ny are the periodic images of ny-1 and 3
    if (c.cyclic) { if (j == c.ps_j0) js = c.ps_j1; if (j == c.ps_j1) jn = c.ps_j0; }
    return d.mat[0][k] * x[k] + d.mat[1][k] * x[ID(i - 1, j)] + d.mat[2][k] * x[ID(i + 1, j)]
         + d.mat[3][k] * x[ID(i, js)] + d.mat[4][k] * x[ID(i, jn)];
}
__global__ void k_s_init() {   // x = 0, r = rhat = b, p = v = 0; (b,b), (rhat,r)
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double bb = 0;
    if (i <= c.NX1 && j <= c.NY1) {
        size_t k = ID(i, j);
        real b = INTERIOR ? d.rhs[k] : 0;
        d.dp[k] = 0; d.r[k] = b; d.rh[k] = b; d.p[k] = 0; d.v_[k] = 0; d.y[k] = 0; d.z[k] = 0;
        bb = (double)b * b;
    }
    block_sum2(bb, bb, &d.sc[8], &d.sc[9]);
}
__global__ void k_s_beta(int first) {
    if (threadIdx.x) return;
    double *s = d.sc;
    if (s[11] != 0) return;
    s[0] = s[9];
    if (first) { s[10] = 0; } else { s[10] = (s[0] / s[1]) * (s[2] / s[3]); }
    s[4] = 0; s[5] = 0; s[6] = 0; s[7] = 0; s[9] = 0;
}
__global__ void k_s_p() {      // p = r + beta (p - omega v); y = p / diag
    if (d.sc[11] != 0) return;
    IJ
    if (!INTERIOR) return;
    size_t k = ID(i, j);
    real beta = (real)d.sc[10], om = (real)d.sc[3];
    real p = d.r[k] + beta * (d.p[k] - om * d.v_[k]);
    d.p[k] = p; d.y[k] = p / d.mat[0][k];
}
__global__ void k_s_Ay() {     // v = A y; (rhat, v)
    if (d.sc[11] != 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double a = 0;
    if (i <= c.NX1 && j <= c.NY1 && INTERIOR) {
        size_t k = ID(i, j);
        real v = Ax(d.y, i, j);
        d.v_[k] = v; a = (double)d.rh[k] * v;
    }
    block_sum2(a, 0, &d.sc[4], nullptr);
}
__global__ void k_s_s() {      // alpha = rho/(rhat,v); s = r - alpha v; z = s / diag
    if (d.sc[11] != 0) return;
    IJ
    if (!INTERIOR) return;
    size_t k = ID(i, j);
    real alpha = (real)(d.sc[0] / d.sc[4]);
    real s = d.r[k] - alpha * d.v_[k];
    d.s[k] = s; d.z[k] = s / d.mat[0][k];
}
__global__ void k_s_Az() {     // t = A z; (t,s), (t,t)
    if (d.sc[11] != 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double ts = 0, tt = 0;
    if (i <= c.NX1 && j <= c.NY1 && INTERIOR) {
        size_t k = ID(i, j);
        real t = Ax(d.z, i, j);
        d.t[k] = t; ts = (double)t * d.s[k]; tt = (double)t * t;
    }
    block_sum2(ts, tt, &d.sc[5], &d.sc[6]);
}
__global__ void k_s_omega() {
    if (threadIdx.x) return;
    double *s = d.sc;
    if (s[11] != 0) return;
    s[2] = s[0] / s[4];
    s[3] = s[6] > 0 ? s[5] / s[6] : 0;
}
__global__ void k_s_x() {      // x += alpha y + omega z; r = s - omega t; (r,r), (rhat,r)
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
__global__ void k_s_roll() {
    if (threadIdx.x || d.sc[11] != 0) return;
    d.sc[1] = d.sc[0]; d.sc[12] += 1;
    if (d.sc[7] < d.sc[13]) d.sc[11] = 1;
}

// ---- tri-diagonal system (1D): a = mat(2) (west), b = mat(1), cc = mat(3) (east), rhs, over i = 1..nx+1.
// One block; ping-pong buffers in r/rh/p/v_ (step in) and s/t/y/z (step out). Double precision accumulation
// is not needed: the system is diagonally dominant.
__global__ void k_pcr(int n) {
    real *A0 = d.r, *B0 = d.rh, *C0 = d.p, *D0 = d.v_, *A1 = d.s, *B1 = d.t, *C1 = d.y, *D1 = d.z;
    for (int q = threadIdx.x; q < n; q += blockDim.x) {
        size_t k = q;   // row j = 1: ID(q+1, 1) = q
        A0[q] = d.mat[1][k]; B0[q] = d.mat[0][k]; C0[q] = d.mat[2][k]; D0[q] = d.rhs[k];
        if (q == 0) A0[q] = 0;
        if (q == n - 1) C0[q] = 0;
    }
    __syncthreads();
    for (int s = 1; s < n; s <<= 1) {
        for (int q = threadIdx.x; q < n; q += blockDim.x) {
            real a = A0[q], b = B0[q], cc = C0[q], dd = D0[q];
            real an = 0, cn = 0;
            if (q - s >= 0) { real al = -a / B0[q - s]; an = al * A0[q - s]; b += al * C0[q - s]; dd += al * D0[q - s]; }
            if (q + s < n) { real ga = -cc / B0[q + s]; cn = ga * C0[q + s]; b += ga * A0[q + s]; dd += ga * D0[q + s]; }
            A1[q] = an; B1[q] = b; C1[q] = cn; D1[q] = dd;
        }
        __syncthreads();
        real *t;
        t = A0; A0 = A1; A1 = t; t = B0; B0 = B1; B1 = t; t = C0; C0 = C1; C1 = t; t = D0; D0 = D1; D1 = t;
    }
    for (int q = threadIdx.x; q < n; q += blockDim.x) d.dp[q] = D0[q] / B0[q];
}
