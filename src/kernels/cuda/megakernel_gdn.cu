// src/kernels/cuda/megakernel_gdn.cu - prototype 1 of the per-layer persistent kernel (megakernel.hpp).
//
// The bodies below are the window's kernels' (native_mmvq.cu: native_quantize_q8_1_kernel, SmallTraits<Q80Block, 8>,
// native_mmvq_wave_exact_kernel; verify_kernels.cu: gdn_conv_l2_body, gdn_ab_body, gdn_step_norm_multi_kernel<true,
// true>) with blockIdx / threadIdx replaced by the virtual block and thread the persistent block lends them, and the
// conv body's conditional __syncthreads made unconditional (four 128-thread virtual blocks share one 512-thread block,
// and a block straddling the q/k heads and the v heads would otherwise diverge at the barrier; every element's
// arithmetic is unchanged).  mk_gdn_parity checks every output against the library's kernels bit for bit.
#include "strata/kernels/megakernel.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

#ifndef MK_TH
#define MK_TH 512
#endif
constexpr int TH = MK_TH;        // the persistent block
constexpr int S = 128;           // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;
constexpr int WARP = 32;
constexpr int WARPS = 4;
constexpr int Q8K = 32;
constexpr int QUANT_THREADS = 256;

struct Q81Block { half2 ds; int8_t qs[32]; };
struct Q80Block { half d; int8_t qs[32]; };
static_assert(sizeof(Q81Block) == 36 && sizeof(Q80Block) == 34);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP));
    return x;
}
__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

// ---- native_quantize_q8_1_kernel (one 256-thread virtual block over 256 elements)
__device__ __forceinline__ void quant_body(const float* __restrict__ x, Q81Block* __restrict__ y, int n_in, int vb, int vtid) {
    const int i = vb * QUANT_THREADS + vtid;
    if (i >= n_in) return;   // n_in is a multiple of 32: only whole warps return
    const float xi = x[i];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

// ---- SmallTraits<Q80Block, 8> + native_mmvq_wave_exact_kernel (one 256-thread virtual block = 8 rows, a wave each)
struct Q80Traits {
    static constexpr int DIV = 32, T = 4, KBY = 1, BPI = 2 * WARPS * WARP / 8;
    __device__ static int kqs(int tid) { return 2 * (tid % 4); }
    struct W { int v[2]; float d0; };
    __device__ static W load(const Q80Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = load_int_b2(w->qs, iqs + i);
        r.d0 = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
            sumi = STRATA_DP4A(r.v[i], u, sumi);
        }
        const float d1 = __low2float(x->ds);
        return r.d0 * d1 * float(sumi);
    }
};
template <int NCOLS>
__device__ __forceinline__ void gemv_q80_body(const Q80Block* __restrict__ w, const Q81Block* __restrict__ x,
                                              float* __restrict__ y, int n_in, int n_out, int vb, int vtid, int xs_zero = 0) {
    using F = Q80Traits;
    const int lane = vtid & 31;
    const int row = vb * 8 + vtid / 32;
    if (row >= n_out) return;
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = xs_zero ? 0 : n_in / Q8K;
    float tmp[NCOLS][WARPS] = {};
#pragma unroll
    for (int g = 0; g < WARPS; ++g) {
        const int tid = lane + g * WARP;
        const int kqs = F::kqs(tid);
        for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += F::BPI) {
            const F::W wv = F::load(w + std::size_t(row) * blocks_per_row + kbx, kqs);
            // every column's activation loads are issued before the first dot product: inside the persistent kernel the
            // compiler otherwise serializes them (one load, s_waitcnt vmcnt(0), one dot4 ... - 2x slower at NCOLS 8);
            // the arithmetic per column is unchanged (same operands, same order)
            int u[NCOLS][2];
            float d1[NCOLS];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * x_stride + kbx * F::KBY;
#pragma unroll
                for (int i = 0; i < 2; ++i) u[j][i] = reinterpret_cast<const int*>(xb->qs)[kqs + i];
                d1[j] = __low2float(xb->ds);
            }
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 2; ++i) sumi = STRATA_DP4A(wv.v[i], u[j][i], sumi);
                tmp[j][g] += wv.d0 * d1[j] * float(sumi);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
        float v = tmp[j][0];
#pragma unroll
        for (int g = 1; g < WARPS; ++g) v += tmp[j][g];
        v = warp_sum(v);
        if (lane == 0) y[std::size_t(j) * n_out + row] = v;
    }
}

// ---- gdn_conv_l2_body (one 128-thread virtual block: channel block bx of token by); `part` per virtual block
__device__ __forceinline__ void conv_body(float* part, int bx, int by, const float* hist, const float* __restrict__ qkv,
                                          const float* __restrict__ w, float* __restrict__ h, int C, int qk_heads, float eps,
                                          int vtid) {
    const int t = by;
    const int c = bx * S + vtid;
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + __expf(-sum));
    // the L2 norm of the q/k heads: every virtual block takes the barrier (the v heads discard the sum)
    float sq = y * y;
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if ((vtid & 31) == 0) part[vtid >> 5] = sq;
    __syncthreads();
    if (bx < qk_heads) {
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

// ---- gdn_ab_body (one 64-thread virtual block: rows bx * 2, bx * 2 + 1 of [alpha rows | beta rows])
template <int MAX_T>
__device__ __forceinline__ void ab_body(int bx, const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                        const uint16_t* __restrict__ wb, const float* __restrict__ dt,
                                        const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                        int n, int h_v, int vtid) {
    const int row = bx * 2 + (vtid >> 5), lane = vtid & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint4* w4 = reinterpret_cast<const uint4*>((is_beta ? wb : wa) + (size_t) r * n);
    float acc[MAX_T];
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const uint4 wv = __ldg(w4 + j);
        const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
        const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
        const float w4f = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
        const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
#pragma unroll
        for (int t = 0; t < MAX_T; ++t) {
            const float* xt = x + (size_t) t * n;
            const float4 xa = __ldg(reinterpret_cast<const float4*>(xt + j * 8));
            const float4 xb = __ldg(reinterpret_cast<const float4*>(xt + j * 8 + 4));
            float a = acc[t];
            a = fmaf(w0, xa.x, a); a = fmaf(w1, xa.y, a);
            a = fmaf(w2, xa.z, a); a = fmaf(w3, xa.w, a);
            a = fmaf(w4f, xb.x, a); a = fmaf(w5, xb.y, a);
            a = fmaf(w6, xb.z, a); a = fmaf(w7, xb.w, a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) {
        float a = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t) t * h_v + r] = 1.0f / (1.0f + __expf(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : log1pf(__expf(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

// ---- gdn_q8_1_store + gdn_step_norm_multi_kernel<ALL_OUT = true, Q = true> (one 512-thread block per v-head)
struct GdnQ81 { half2 ds; int8_t qs[32]; };
__device__ __forceinline__ void gdn_q8_1_store(GdnQ81* __restrict__ xq, size_t idx, float xi) {
#pragma clang fp contract(off)
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    xq[idx / 32].qs[idx % 32] = q;
    if (idx % 32 == 0) xq[idx / 32].ds = q8_1_ds(d, sum);
}
struct StepShared { float sk[2][S], sq[2][S], red_kv[RG][S], red_o[RG][S], wsum[S / 32]; };
__device__ __forceinline__ void step_body(StepShared& sh, int head, const float* __restrict__ state,
                                          const float* __restrict__ hbuf, int C, const float* __restrict__ gate,
                                          const float* __restrict__ beta, const float* __restrict__ z,
                                          const float* __restrict__ gamma, float eps, float* __restrict__ y, int h_k, int h_v,
                                          int T, GdnQ81* __restrict__ xq, int tid) {
    const int col = tid % S;
    const int rg = tid / S;
    const int qh = head % h_k;
    const int qk = S * h_k;
    const int value_dim = S * h_v;
    const int n = T;
    if (n > 0) {
        if (rg == 0) sh.sk[0][col] = hbuf[qk + qh * S + col];
        else if (rg == 1) sh.sq[0][col] = hbuf[qh * S + col];
    }
    const float gam = (rg == 0) ? gamma[col] : 0.0f;
    float s[RPG];
    const float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = __ldg(&base[r * row_stride]);
    for (int t = 0; t < n; ++t) {
        const int cur = t & 1, nxt = (t + 1) & 1;
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();
        if (t + 1 < n) {
            const float* ht_next = ht + C;
            if (rg == 0) sh.sk[nxt][col] = ht_next[qk + qh * S + col];
            else if (rg == 1) sh.sq[nxt][col] = ht_next[qh * S + col];
        }
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sh.sk[cur][rg * RPG + r], kv);
        sh.red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = sh.red_kv[0][col] + sh.red_kv[1][col] + sh.red_kv[2][col] + sh.red_kv[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sh.sk[cur][rg * RPG + r] * delta);
            o = fmaf(s[r], sh.sq[cur][rg * RPG + r], o);
        }
        sh.red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (sh.red_o[0][col] + sh.red_o[1][col] + sh.red_o[2][col] + sh.red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if (rg == 0 && (col & 31) == 0) sh.wsum[col >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = sh.wsum[0] + sh.wsum[1] + sh.wsum[2] + sh.wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            const float yv = oc * scale * gam * (1.0f / (1.0f + __expf(-zz)));
            y[(size_t) t * value_dim + head * S + col] = yv;
            gdn_q8_1_store(xq, (size_t) t * value_dim + head * S + col, yv);
        }
    }
}

// ---- the persistent kernel: phases of work items pulled with an atomic, a counter per phase
// counters: [0..4] items done per phase, [8..12] next item per phase
// The scheduler's atomics are issued by the WHOLE first wave (lane 0 adds 1, the others 0) and broadcast with
// readfirstlane, never under `if (threadIdx.x == 0)`: with a lane-0 branch clang 22 (ROCm 7.2) jump-threads lanes
// 1..31 from the item's end straight back to grab's barrier + LDS load, so the wave loops there with lane 0 masked
// off - and lane 0 holds the atomic they wait on (a SIMT forward-progress deadlock, even with one wave).  A wave-
// uniform branch leaves no lane masked off while its wave loops.
__device__ __forceinline__ bool wave0() { return threadIdx.x < WARP; }
__device__ __forceinline__ unsigned one0() { return threadIdx.x == 0 ? 1u : 0u; }

__device__ __forceinline__ int grab(unsigned* next, int* slot, unsigned* diag) {
    // wave 0 pulls the next item of the block
    if (wave0()) {
        const unsigned v = __builtin_amdgcn_readfirstlane(atomicAdd(next, one0()));
        atomicMax(diag, v);   // diagnostics: the largest item index handed out
        *slot = (int) v;
    }
    __syncthreads();
    const int it = __builtin_amdgcn_readfirstlane(*slot);
    __syncthreads();
    return it;
}
// (bounded: a phase that never completes sets counters[15] to the phase and the kernel goes on, so the host can read
// the counters instead of hanging - a diagnostic, not a recovery)
__device__ __forceinline__ void phase_wait(unsigned* done, unsigned items, unsigned* diag, unsigned phase) {
    if (wave0()) {
        long spins = 0;
        while (__builtin_amdgcn_readfirstlane(__hip_atomic_load(done, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) < items) {
            __builtin_amdgcn_s_sleep(1);
            if (++spins > (1l << 24)) { atomicMax(diag, phase + 1); break; }
        }
    }
    __syncthreads();
    __threadfence();
}
__device__ __forceinline__ void item_done(unsigned* done) {
    __syncthreads();
    __threadfence();
    if (wave0()) atomicAdd(done, one0());
}
// Phase k over n items.  Default: block-strided static assignment (the items are uniform), no atomic or barrier
// between items unless the body reuses shared memory (`sync_between`), one done-increment per block at the end, so
// the next phase waits for gridDim.x.  diag & 1: the dynamic grab scheduler (one atomic + barriers an item).
template <class Body>
__device__ __forceinline__ void run_phase(const MkGdnArgs& a, int k, unsigned n, int* slot, bool sync_between, Body body) {
    unsigned* done = a.counters + k;
    if (a.diag & 1) {
        unsigned* next = a.counters + 8 + k;
        for (int it; (it = grab(next, slot, a.counters + 7)) < (int) n;) {
            body(it);
            item_done(done);
        }
    } else {
        for (int it = (int) blockIdx.x; it < (int) n; it += (int) gridDim.x) {
            body(it);
            if (sync_between) __syncthreads();
        }
        item_done(done);
    }
}
__device__ __forceinline__ unsigned phase_items(const MkGdnArgs& a, unsigned n) { return (a.diag & 1) ? n : gridDim.x; }

// The last block out zeroes the counters for the next launch (stream order puts that before the next kernel), so
// a launch needs no memset node.  counters[14] counts the blocks out; [5..7] and [15] are diagnostics and are cleared too.
__device__ __forceinline__ void mk_exit(const MkGdnArgs& a) {
    __syncthreads();
    __threadfence();
    if (wave0()) {
        const unsigned out = __builtin_amdgcn_readfirstlane(atomicAdd(a.counters + 14, one0()));
        if (out + 1 == gridDim.x) {
            __threadfence();
            if (threadIdx.x < 16) __hip_atomic_store(a.counters + threadIdx.x, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

#ifndef MK_WAVES_PER_EU
#define MK_WAVES_PER_EU 0
#endif
#if MK_WAVES_PER_EU > 0
#define MK_WAVES_ATTR __attribute__((amdgpu_waves_per_eu(MK_WAVES_PER_EU, MK_WAVES_PER_EU)))
#else
#define MK_WAVES_ATTR
#endif
template <int NCOLS>
__global__ void __launch_bounds__(TH) MK_WAVES_ATTR mk_gdn_kernel(MkGdnArgs a) {
    __shared__ int slot;
    __shared__ float conv_part[4][S / 32];
    __shared__ StepShared step_sh;
    const int tid = (int) threadIdx.x;
    const int N = a.n_embd, C = a.conv_channels, ZV = a.value_dim, HK = a.h_k, HV = a.h_v;
    const int T = NCOLS;
    unsigned* done = a.counters;
    unsigned* next = a.counters + 8;
    const Q81Block* xq = (const Q81Block*) a.xq;
    const Q81Block* yq = (const Q81Block*) a.yq;
    if (wave0()) atomicAdd(a.counters + 6, one0());   // blocks that entered the kernel
    if (a.phases <= 0) { mk_exit(a); return; }
    // phase 0: the input's q8_1 image (256-thread virtual blocks, two per block)
    const unsigned n0 = (unsigned) ((T * N + (TH / QUANT_THREADS) * QUANT_THREADS - 1) / ((TH / QUANT_THREADS) * QUANT_THREADS));
    run_phase(a, 0, n0, &slot, false, [&](int it) {
        quant_body(a.x, (Q81Block*) a.xq, T * N, it * (TH / QUANT_THREADS) + tid / QUANT_THREADS, tid % QUANT_THREADS);
    });
    if (blockIdx.x == 0 && wave0()) atomicMax(a.counters + 5, 1u);
    if (a.phases <= 1) { mk_exit(a); return; }
    // phase 1: the qkv and z projections (8 rows a wave group, two groups per block) and the a/b rows (64-thread
    // virtual blocks, eight per block)
    const unsigned n_qkv = (unsigned) ((C / 8 + TH / 256 - 1) / (TH / 256)), n_z = (unsigned) ((ZV / 8 + TH / 256 - 1) / (TH / 256));
    const unsigned n_ab = (unsigned) (((2 * HV + 1) / 2 + 7) / 8);
    const unsigned n1 = n_qkv + n_z + n_ab;
    phase_wait(done + 0, phase_items(a, n0), a.counters + 15, 0);
    // diag & 4: a barrier between the GEMV items (the waves re-align on x); diag & 8: the qkv rows only
    run_phase(a, 1, (a.diag & 8) ? n_qkv : n1, &slot, (a.diag & 4) != 0, [&](int it) {
        if (it < (int) n_qkv) {
            gemv_q80_body<NCOLS>((const Q80Block*) a.w_qkv, xq, a.qkv, N, C, it * (TH / 256) + tid / 256, tid % 256, a.diag & 16);
        } else if (it < (int) (n_qkv + n_z)) {
            gemv_q80_body<NCOLS>((const Q80Block*) a.w_z, xq, a.z, N, ZV, (it - (int) n_qkv) * (TH / 256) + tid / 256, tid % 256);
        } else {
            ab_body<NCOLS>((it - (int) (n_qkv + n_z)) * 8 + tid / 64, a.x, a.w_alpha, a.w_beta, a.dt, a.ssm_a, a.gate, a.beta,
                           N, HV, tid % 64);
        }
    });
    if (blockIdx.x == 0 && wave0()) atomicMax(a.counters + 5, 2u);
    if (a.phases <= 2) { mk_exit(a); return; }
    // phase 2: the conv + L2 norm (128-thread virtual blocks: channel block x token, four per block)
    const unsigned n_conv_vb = (unsigned) ((C / S) * T), n2 = (n_conv_vb + 3) / 4;
    phase_wait(done + 1, phase_items(a, n1), a.counters + 15, 1);
    run_phase(a, 2, n2, &slot, true, [&](int it) {
        const int vb = it * 4 + tid / S;
        if (vb < (int) n_conv_vb)
            conv_body(conv_part[tid / S], vb % (C / S), vb / (C / S), a.hist, a.qkv, a.w_conv, a.h, C, 2 * HK, a.eps, tid % S);
        else
            __syncthreads();   // (the body's barrier, so the block stays uniform)
    });
    if (blockIdx.x == 0 && wave0()) atomicMax(a.counters + 5, 3u);
    if (a.phases <= 3) { mk_exit(a); return; }
    // phase 3: the recurrence + output norm + q8_1 image, one block per v-head
    phase_wait(done + 2, phase_items(a, n2), a.counters + 15, 2);
    run_phase(a, 3, (unsigned) HV, &slot, true, [&](int it) {
        step_body(step_sh, it, a.state, a.h, C, a.gate, a.beta, a.z, a.gamma, a.eps, a.y, HK, HV, T, (GdnQ81*) a.yq, tid);
    });
    if (blockIdx.x == 0 && wave0()) atomicMax(a.counters + 5, 4u);
    if (a.phases <= 4) { mk_exit(a); return; }
    // phase 4: the out projection
    const unsigned n4 = (unsigned) ((N / 8 + TH / 256 - 1) / (TH / 256));
    phase_wait(done + 3, phase_items(a, (unsigned) HV), a.counters + 15, 3);
    run_phase(a, 4, n4, &slot, (a.diag & 4) != 0, [&](int it) {
        gemv_q80_body<NCOLS>((const Q80Block*) a.w_out, yq, a.bo, ZV, N, it * (TH / 256) + tid / 256, tid % 256);
    });
    mk_exit(a);
}

int g_grid = 0;
template <int NCOLS>
void launch(const MkGdnArgs& a, cudaStream_t s) {
    if (g_grid == 0) {
        int dev = 0, cus = 0, per = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cus, cudaDevAttrMultiProcessorCount, dev);
#if defined(STRATA_USE_HIP)
        (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, mk_gdn_kernel<NCOLS>, TH, 0);
#else
        (void) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, mk_gdn_kernel<NCOLS>, TH, 0);
#endif
        g_grid = cus * (per > 0 ? per : 1);
        if (g_grid <= 0) g_grid = 64;
    }
    // (no memset: the kernel's last block zeroes the counters on the way out; mk_counters_alloc zeroes them once)
    mk_gdn_kernel<NCOLS><<<g_grid, TH, 0, s>>>(a);
}
}  // namespace

unsigned* mk_counters_alloc() {
    unsigned* p = nullptr;
    if (cudaMalloc((void**) &p, 64) != cudaSuccess) return nullptr;
    cudaMemset(p, 0, 64);
    return p;
}
int mk_gdn_grid() { return g_grid; }

void mk_gdn_dense(const MkGdnArgs& a, int n_tok, void* stream) {
    if (n_tok < 1 || n_tok > 8 || a.counters == nullptr) { std::fprintf(stderr, "mk_gdn_dense: invalid arguments\n"); std::exit(1); }
    cudaStream_t s = (cudaStream_t) stream;
    switch (n_tok) {
        case 1: launch<1>(a, s); break;
        case 2: launch<2>(a, s); break;
        case 3: launch<3>(a, s); break;
        case 4: launch<4>(a, s); break;
        case 5: launch<5>(a, s); break;
        case 6: launch<6>(a, s); break;
        case 7: launch<7>(a, s); break;
        default: launch<8>(a, s); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "mk_gdn_dense: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::kernels
