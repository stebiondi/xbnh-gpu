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
// fp32 products and in-block sums (<= 256 terms, relative error ~1e-5), one fp64 atomic per block
__device__ __forceinline__ void block_sum2f(float a, float b, double *out_a, double *out_b) {
    for (int o = 16; o > 0; o >>= 1) { a += __shfl_down_sync(0xffffffff, a, o); b += __shfl_down_sync(0xffffffff, b, o); }
    __shared__ float sa[32], sb[32];
    int lane = (threadIdx.y * blockDim.x + threadIdx.x) & 31, wid = (threadIdx.y * blockDim.x + threadIdx.x) >> 5;
    if (lane == 0) { sa[wid] = a; sb[wid] = b; }
    __syncthreads();
    int nw = (blockDim.x * blockDim.y + 31) / 32;
    if (wid == 0) {
        a = lane < nw ? sa[lane] : 0; b = lane < nw ? sb[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) { a += __shfl_down_sync(0xffffffff, a, o); b += __shfl_down_sync(0xffffffff, b, o); }
        if (lane == 0) { if (out_a) atomicAdd(out_a, (double)a); if (out_b) atomicAdd(out_b, (double)b); }
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
// Warm start (gpu_warm 1: x0 = previous solution, 2: linear extrapolation 2 x_{n-1} - x_{n-2}); x0 = 0 outside
// the system and at excluded (identity) rows. Then r = rhat = b - A x0; (b,b) -> sc[8], (r,r) -> sc[7] (copied to sc[9] = (rhat,r) by the host).
__global__ void k_s_guess(const real *x1, const real *x2, int mode, const int *mask) {
    IJ
    size_t k = ID(i, j);
    real g = 0;
    if (INTERIOR && (!mask || mask[k] == 1)) g = mode == 2 ? 2 * x1[k] - x2[k] : x1[k];
    d.dp[k] = g;
}
__global__ void k_s_init_warm() {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double bb = 0, rr = 0;
    if (i <= c.NX1 && j <= c.NY1) {
        size_t k = ID(i, j);
        real b = 0, r = 0;
        if (INTERIOR) { b = d.rhs[k]; r = b - Ax(d.dp, i, j); }
        d.r[k] = r; d.rh[k] = r; d.p[k] = 0; d.v_[k] = 0; d.y[k] = 0; d.z[k] = 0;
        bb = (double)b * b; rr = (double)r * r;
    }
    block_sum2(bb, rr, &d.sc[8], &d.sc[7]);
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

// ---- Chebyshev iteration (gpu_solver = 1) on the Jacobi-scaled system D^-1 A x = D^-1 b. The eigenvalues of D^-1 A
// lie in the Gershgorin discs |lambda - 1| <= g_k = sum|offdiag_k| / |diag_k|; for this system they are real in practice
// (near-symmetric operator), so [1 - g, 1 + g] with g = max g_k bounds them. One fused kernel per iteration (one SpMV,
// no inner products): x_{n+1} = x_n + a_n (x_n - x_{n-1}) + c_n D^-1 (b - A x_n), with |b - A x_n|^2 accumulated into
// rr[n] for the same stopping test as BiCGStab. Launch n returns at once when some earlier rr[m] met the target, so
// x_m stays intact in buffer m % 3. Not converged (complex eigenvalues, g >= gmax): BiCGStab from the last iterate.
__global__ void k_ch_gersh(unsigned *gmax) {   // g = max over the system rows (identity rows: 0); float bits as unsigned
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    float g = 0;
    if (i <= c.NX1 && j <= c.NY1 && INTERIOR) {
        size_t k = ID(i, j);
        float dg = fabsf((float)d.mat[0][k]);
        float o = fabsf((float)d.mat[1][k]) + fabsf((float)d.mat[2][k]) + fabsf((float)d.mat[3][k]) + fabsf((float)d.mat[4][k]);
        g = dg > 0 ? o / dg : 1e30f;
    }
    for (int q = 16; q > 0; q >>= 1) g = fmaxf(g, __shfl_down_sync(0xffffffff, g, q));
    __shared__ float sg[32];
    int lane = (threadIdx.y * blockDim.x + threadIdx.x) & 31, wid = (threadIdx.y * blockDim.x + threadIdx.x) >> 5;
    if (lane == 0) sg[wid] = g;
    __syncthreads();
    int nw = (blockDim.x * blockDim.y + 31) / 32;
    if (wid == 0) {
        g = lane < nw ? sg[lane] : 0;
        for (int q = 16; q > 0; q >>= 1) g = fmaxf(g, __shfl_down_sync(0xffffffff, g, q));
        if (lane == 0) atomicMax(gmax, __float_as_uint(g));
    }
}
__global__ void k_ch_bb(const real *x0) {      // (b,b) -> sc[8]; x0 = 0 outside the system rows
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double bb = 0;
    if (i <= c.NX1 && j <= c.NY1 && INTERIOR) { real b = d.rhs[ID(i, j)]; bb = (double)b * b; }
    block_sum2(bb, 0, &d.sc[8], nullptr);
}
__global__ void k_ch_step(int n, const real *xn, const real *xm, real *xp, real a, real cc, double *rr, double tgt) {
    if (n > 0 && rr[n - 1] < tgt) return;   // converged at an earlier iterate (skipped launches leave rr = 0)
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double r2 = 0;
    if (i <= c.NX1 && j <= c.NY1 && INTERIOR) {
        size_t k = ID(i, j);
        real x = xn[k];
        real r = d.rhs[k] - Ax(xn, i, j);
        xp[k] = x + a * (x - xm[k]) + cc * r / d.mat[0][k];
        r2 = (double)r * r;
    }
    block_sum2(r2, 0, &rr[n], nullptr);
}

// ---- Fused BiCGStab (gpu_solver = 2): the same Jacobi right-preconditioned BiCGStab in 3 kernels per iteration
// instead of 8. Neighbour values of p and s are formed on the fly (p and v ping-pong), so y = p/D, z = s/D and s are
// never stored; the scalars of iteration n are computed by every thread from the inner products of slots n-1 and n
// (S[8 n + q]: 0 (rhat,v), 1 (t,s), 2 (t,t), 3 (r,r), 4 (rhat,r); slot 0 holds (r0,r0), (rhat,r0)). A kernel returns
// at once when an earlier iterate met the target (skipped slots stay 0, so later launches skip as well).
struct FB { real *x, *r, *rh, *t, *pi, *po, *vi, *vo; double *S; double tgt; };
__device__ __forceinline__ bool fb_in(int i, int j) { return i >= 2 && i <= c.nx && j >= c.ps_j0 && j <= c.ps_j1; }
__device__ __forceinline__ void fb_nb(int i, int j, int &js, int &jn) {
    js = j - 1; jn = j + 1;
    if (c.cyclic) { if (j == c.ps_j0) js = c.ps_j1; if (j == c.ps_j1) jn = c.ps_j0; }
}
__device__ __forceinline__ void fb_scal(const double *S, int n, double &rho, double &alpha, double &omega) {
    // rho_n = (rhat, r_{n-1}); alpha_n = rho_n / (rhat, v_n); omega_n = (t,s)_n / (t,t)_n
    rho = S[8 * (n - 1) + 4];
    double q0 = S[8 * n + 0], q2 = S[8 * n + 2];
    alpha = q0 != 0 ? rho / q0 : 0;
    omega = q2 > 0 ? S[8 * n + 1] / q2 : 0;
}
__global__ void k_fb1(FB f, int n) {      // p_n = r + beta (p - omega_{n-1} v); v_n = A (p_n / D); (rhat, v_n)
    if (f.S[8 * (n - 1) + 3] < f.tgt) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    float a = 0;
    __shared__ real sh[2];   // scalars once per block (fp64 division is slow on consumer GPUs)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        real beta = 0, om = 0;
        if (n > 1) {
            double rhom, alm, omm;
            fb_scal(f.S, n - 1, rhom, alm, omm);
            double rho = f.S[8 * (n - 1) + 4];
            beta = (real)((rho / rhom) * (alm / omm)); om = (real)omm;
        }
        sh[0] = beta; sh[1] = om;
    }
    __syncthreads();
    if (i <= c.NX1 && j <= c.NY1 && INTERIOR) {
        real beta = sh[0], om = sh[1];
        int js, jn; fb_nb(i, j, js, jn);
        auto q = [&](int ii, int jj) -> real {
            if (!fb_in(ii, jj)) return 0;
            size_t kk = ID(ii, jj);
            return (f.r[kk] + beta * (f.pi[kk] - om * f.vi[kk])) / d.mat[0][kk];
        };
        size_t k = ID(i, j);
        real pc = f.r[k] + beta * (f.pi[k] - om * f.vi[k]);
        f.po[k] = pc;
        real v = pc + d.mat[1][k] * q(i - 1, j) + d.mat[2][k] * q(i + 1, j) + d.mat[3][k] * q(i, js) + d.mat[4][k] * q(i, jn);
        f.vo[k] = v; a = (float)f.rh[k] * (float)v;
    }
    block_sum2f(a, 0, &f.S[8 * n + 0], nullptr);
}
__global__ void k_fb2(FB f, int n) {      // s = r - alpha v_n; t = A (s / D); (t,s), (t,t)
    if (f.S[8 * (n - 1) + 3] < f.tgt) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    float ts = 0, tt = 0;
    __shared__ real sh[1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        double rho = f.S[8 * (n - 1) + 4], q0 = f.S[8 * n + 0];
        sh[0] = (real)(q0 != 0 ? rho / q0 : 0);
    }
    __syncthreads();
    if (i <= c.NX1 && j <= c.NY1 && INTERIOR) {
        real al = sh[0];
        int js, jn; fb_nb(i, j, js, jn);
        auto z = [&](int ii, int jj) -> real {
            if (!fb_in(ii, jj)) return 0;
            size_t kk = ID(ii, jj);
            return (f.r[kk] - al * f.vo[kk]) / d.mat[0][kk];
        };
        size_t k = ID(i, j);
        real sc_ = f.r[k] - al * f.vo[k];
        real t = sc_ + d.mat[1][k] * z(i - 1, j) + d.mat[2][k] * z(i + 1, j) + d.mat[3][k] * z(i, js) + d.mat[4][k] * z(i, jn);
        f.t[k] = t; ts = (float)t * (float)sc_; tt = (float)t * (float)t;
    }
    block_sum2f(ts, tt, &f.S[8 * n + 1], &f.S[8 * n + 2]);
}
__global__ void k_fb3(FB f, int n) {      // x += (alpha p + omega s) / D; r = s - omega t; (r,r), (rhat,r)
    if (f.S[8 * (n - 1) + 3] < f.tgt) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1, j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    float rr = 0, rhr = 0;
    __shared__ real sh[2];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        double rho, al, om; fb_scal(f.S, n, rho, al, om);
        sh[0] = (real)al; sh[1] = (real)om;
    }
    __syncthreads();
    if (i <= c.NX1 && j <= c.NY1 && INTERIOR) {
        size_t k = ID(i, j);
        real a = sh[0], o = sh[1];
        real sc_ = f.r[k] - a * f.vo[k];
        f.x[k] += (a * f.po[k] + o * sc_) / d.mat[0][k];
        real r = sc_ - o * f.t[k];
        f.r[k] = r; rr = (float)r * (float)r; rhr = (float)f.rh[k] * (float)r;
    }
    block_sum2f(rr, rhr, &f.S[8 * n + 3], &f.S[8 * n + 4]);
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
