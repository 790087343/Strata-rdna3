// src/kernels/cuda/rdna_gr.cu - an RDNA3 (gfx1100) hyper-connection read: split-K down + partial-folding up.
// Same maths as fused_gr_read_multi, another summation order (opt-in, STRATA_GR_RDNA=1):
//   down: block (8-row group, stream c, half h) dots its rows with R'[c,h] * w_norm[c,h] (no rms yet) and, for row
//         group 0, writes the slice's sum of squares;  w_down . xn = sum_c rs[c] * partial[c]  (rs is per stream).
//   up:   every block folds the 8 partials into lo (and block 0 into inject_out / rs) in its prologue, then four lanes
//         per w_up row, two xor steps per token.  No counters, no inter-block order: safe on any stream.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>

#ifndef STRATA_GR_WPE
#define STRATA_GR_WPE 8
#endif
namespace strata::kernels {
namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, PR = LR + HC;   // partial rows: 320 down + 4 inject
constexpr int H = N / 2;                  // half-stream slice, 1280
constexpr int TH = 256, WV = TH / 32;     // threads, waves (wave32)
constexpr int HQ = H / 8 / 32;            // uint4 per lane per row slice: 5
constexpr int UC = 16;                    // up: columns per block (x 4 streams = 64 rows = 256 quarter-rows)
constexpr int UB = N / UC;                // 160 blocks

unsigned* g_gr_bar[64] = {};   // STRATA_GR_FUSED: per device {arrived, generation}, zeroed at load (rdna_r8_scratch_init)
struct RdnaGr { FusedGrArgs a[kFusedGrMaxT]; float* part; float* ss; unsigned* bar = nullptr; bool uppre = true; };   // bar: STRATA_GR_FUSED grid barrier   // part [T][2*HC][PR], ss [T][2*HC]

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ uint4 ntload(const uint4* p) {
    const u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p));
    return make_uint4(v.x, v.y, v.z, v.w);
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
// wave32 all-reduce with DPP and permlanex16 (no LDS round trip, unlike __shfl_xor's ds_bpermute)
__device__ __forceinline__ float dpp_f(float v, int ctrl);
#define STRATA_DPP(v, ctrl) __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), ctrl, 0xf, 0xf, false))
__device__ __forceinline__ float wsum(float v) {
    v += STRATA_DPP(v, 0xB1);    // quad_perm [1,0,3,2]
    v += STRATA_DPP(v, 0x4E);    // quad_perm [2,3,0,1]
    v += STRATA_DPP(v, 0x141);   // row_half_mirror: 8-lane sums
    v += STRATA_DPP(v, 0x140);   // row_mirror: 16-lane sums
    v += __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210, 0xfedcba98, false, false));
    return v;
}
__device__ __forceinline__ float dot8(const uint4 w, const float4 x0, const float4 x1, float acc) {
    acc = fmaf(__uint_as_float(w.x << 16), x0.x, acc); acc = fmaf(__uint_as_float(w.x & 0xffff0000u), x0.y, acc);
    acc = fmaf(__uint_as_float(w.y << 16), x0.z, acc); acc = fmaf(__uint_as_float(w.y & 0xffff0000u), x0.w, acc);
    acc = fmaf(__uint_as_float(w.z << 16), x1.x, acc); acc = fmaf(__uint_as_float(w.z & 0xffff0000u), x1.y, acc);
    acc = fmaf(__uint_as_float(w.w << 16), x1.z, acc); acc = fmaf(__uint_as_float(w.w & 0xffff0000u), x1.w, acc);
    return acc;
}

template <int T, int RW>
__global__ void __launch_bounds__(TH) __attribute__((amdgpu_waves_per_eu(STRATA_GR_WPE, 16))) rdna_gr_down(RdnaGr m) {
    __shared__ __align__(16) float xs[T][H];
    __shared__ float sq[WV][T];
    const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
    const int rg = blockIdx.x, slice = blockIdx.y, c = slice >> 1, h = slice & 1;   // slice = 2c + h
    uint4 wv4[RW][HQ];
    bool act[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int row = (rg * WV + wv) * RW + r;
        act[r] = row < LR || (row < PR && m.a[0].w_inject != nullptr);
        const uint16_t* wbase = row < LR ? m.a[0].w_down + (size_t) row * D : m.a[0].w_inject + (size_t) (row - LR) * D;
        const uint4* w4 = reinterpret_cast<const uint4*>(wbase + c * N + h * H);
        if (act[r]) {
#pragma unroll
            for (int q = 0; q < HQ; ++q) wv4[r][q] = ntload(w4 + lane + 32 * q);
        }
    }
    const int e0 = c * N + h * H;   // this slice's first element in D
    // stage R'[c,h] * w_norm for every token: one flattened (token, float4) loop, loads of several tokens in flight
#pragma unroll 4
    for (int it = t; it < T * (H / 4); it += TH) {
        const int k = it / (H / 4), i = (it - k * (H / 4)) * 4;
        const FusedGrArgs& a = m.a[k];
        float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
        if (a.apply) {
            const float gw = 2.0f * sigm(a.inj_prev[c] / (float) HC);
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + e0 + i);
            *reinterpret_cast<float4*>(&xs[k][i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
    if (rg == 0) {   // the slice's sums of squares (only row group 0 publishes them)
#pragma unroll 1
        for (int k = 0; k < T; ++k) {
            const FusedGrArgs& a = m.a[k];
            const float gw = a.apply ? 2.0f * sigm(a.inj_prev[c] / (float) HC) : 0.0f;
            float s = 0.0f;
            for (int i = t * 4; i < H; i += TH * 4) {
                float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
                if (a.apply) {
                    const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
                    r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
                }
                s = fmaf(r.w, r.w, fmaf(r.z, r.z, fmaf(r.y, r.y, fmaf(r.x, r.x, s))));   // explicit: the same in every instance
            }
            const float v = wsum(s);
            if (lane == 0) sq[wv][k] = v;
        }
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WV; ++w) s += sq[w][t];
        m.ss[t * 2 * HC + slice] = s;
    }
    // tokens in pairs: the weights stay in registers, the accumulators do not grow with T
#pragma unroll 1
    for (int k0 = 0; k0 < T; k0 += 2) {
        const bool two = k0 + 1 < T;
        float acc0[RW], acc1[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }
#pragma unroll
        for (int q = 0; q < HQ; ++q) {
            const int j = (lane + 32 * q) * 8;
            const float4 a0 = *reinterpret_cast<const float4*>(&xs[k0][j]);
            const float4 a1 = *reinterpret_cast<const float4*>(&xs[k0][j + 4]);
            float4 b0 = make_float4(0.f, 0.f, 0.f, 0.f), b1 = b0;
            if (two) { b0 = *reinterpret_cast<const float4*>(&xs[k0 + 1][j]); b1 = *reinterpret_cast<const float4*>(&xs[k0 + 1][j + 4]); }
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                acc0[r] = dot8(wv4[r][q], a0, a1, acc0[r]);
                acc1[r] = dot8(wv4[r][q], b0, b1, acc1[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int row = (rg * WV + wv) * RW + r;
            const float v0 = wsum(acc0[r]), v1 = wsum(acc1[r]);
            if (act[r] && lane == 0) m.part[((size_t) k0 * 2 * HC + slice) * PR + row] = v0;
            if (act[r] && lane == 1 && two) m.part[((size_t) (k0 + 1) * 2 * HC + slice) * PR + row] = v1;
        }
    }
}


// ---- STRATA_GR_RDNA_Q8: the same two kernels on int8 + fp16-scale weights (hc_r8 layout; inject rows stay BF16)
__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ float dot8q(const uint2 w, float d, const float4 x0, const float4 x1, float acc) {
    float s = 0.0f;
    s = fmaf((float) (int8_t) (w.x), x0.x, s);       s = fmaf((float) (int8_t) (w.x >> 8), x0.y, s);
    s = fmaf((float) (int8_t) (w.x >> 16), x0.z, s); s = fmaf((float) (int8_t) (w.x >> 24), x0.w, s);
    s = fmaf((float) (int8_t) (w.y), x1.x, s);       s = fmaf((float) (int8_t) (w.y >> 8), x1.y, s);
    s = fmaf((float) (int8_t) (w.y >> 16), x1.z, s); s = fmaf((float) (int8_t) (w.y >> 24), x1.w, s);
    return fmaf(d, s, acc);
}
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ uint2 ntload2(const uint2* p) {
    const u32x2 v = __builtin_nontemporal_load(reinterpret_cast<const u32x2*>(p));
    return make_uint2(v.x, v.y);
}
constexpr size_t r8_soff(size_t n) { return (n + 255) / 256 * 256; }

template <int T, int RW>
__global__ void __launch_bounds__(TH) __attribute__((amdgpu_waves_per_eu(STRATA_GR_WPE, 16))) rdna_gr_down_q8(RdnaGr m) {
    __shared__ __align__(16) float xs[T][H];
    __shared__ float sq[WV][T];
    const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
    const int rg = blockIdx.x, slice = blockIdx.y, c = slice >> 1, h = slice & 1;
    const int8_t* qd = m.a[0].r8_down;
    const uint16_t* sd = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(qd) + r8_soff((size_t) LR * D));
    uint2 wq[RW][HQ];
    float wd[RW][HQ];
    uint4 wb[RW][HQ];   // the BF16 inject rows
    bool act[RW], inj[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int row = (rg * WV + wv) * RW + r;
        inj[r] = row >= LR;
        act[r] = row < LR || (row < PR && m.a[0].w_inject != nullptr);
        if (!act[r]) continue;
        if (!inj[r]) {
            const uint2* w2 = reinterpret_cast<const uint2*>(qd + (size_t) row * D + c * N + h * H);
            const uint16_t* s2 = sd + (size_t) row * (D / 32) + (c * N + h * H) / 32;
#pragma unroll
            for (int q = 0; q < HQ; ++q) { wq[r][q] = ntload2(w2 + lane + 32 * q); wd[r][q] = h2f(s2[(lane + 32 * q) >> 2]); }
        } else {
            const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_inject + (size_t) (row - LR) * D + c * N + h * H);
#pragma unroll
            for (int q = 0; q < HQ; ++q) wb[r][q] = ntload(w4 + lane + 32 * q);
        }
    }
    const int e0 = c * N + h * H;   // this slice's first element in D
    // stage R'[c,h] * w_norm for every token: one flattened (token, float4) loop, loads of several tokens in flight
#pragma unroll 4
    for (int it = t; it < T * (H / 4); it += TH) {
        const int k = it / (H / 4), i = (it - k * (H / 4)) * 4;
        const FusedGrArgs& a = m.a[k];
        float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
        if (a.apply) {
            const float gw = 2.0f * sigm(a.inj_prev[c] / (float) HC);
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + e0 + i);
            *reinterpret_cast<float4*>(&xs[k][i]) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
    if (rg == 0) {   // the slice's sums of squares (only row group 0 publishes them)
#pragma unroll 1
        for (int k = 0; k < T; ++k) {
            const FusedGrArgs& a = m.a[k];
            const float gw = a.apply ? 2.0f * sigm(a.inj_prev[c] / (float) HC) : 0.0f;
            float s = 0.0f;
            for (int i = t * 4; i < H; i += TH * 4) {
                float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
                if (a.apply) {
                    const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
                    r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
                }
                s = fmaf(r.w, r.w, fmaf(r.z, r.z, fmaf(r.y, r.y, fmaf(r.x, r.x, s))));   // explicit: the same in every instance
            }
            const float v = wsum(s);
            if (lane == 0) sq[wv][k] = v;
        }
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WV; ++w) s += sq[w][t];
        m.ss[t * 2 * HC + slice] = s;
    }
#pragma unroll 1
    for (int k0 = 0; k0 < T; k0 += 2) {
        const bool two = k0 + 1 < T;
        float acc0[RW], acc1[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) { acc0[r] = 0.0f; acc1[r] = 0.0f; }
#pragma unroll
        for (int q = 0; q < HQ; ++q) {
            const int j = (lane + 32 * q) * 8;
            const float4 a0 = *reinterpret_cast<const float4*>(&xs[k0][j]);
            const float4 a1 = *reinterpret_cast<const float4*>(&xs[k0][j + 4]);
            float4 b0 = make_float4(0.f, 0.f, 0.f, 0.f), b1 = b0;
            if (two) { b0 = *reinterpret_cast<const float4*>(&xs[k0 + 1][j]); b1 = *reinterpret_cast<const float4*>(&xs[k0 + 1][j + 4]); }
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                if (!inj[r]) {
                    acc0[r] = dot8q(wq[r][q], wd[r][q], a0, a1, acc0[r]);
                    acc1[r] = dot8q(wq[r][q], wd[r][q], b0, b1, acc1[r]);
                } else {
                    acc0[r] = dot8(wb[r][q], a0, a1, acc0[r]);
                    acc1[r] = dot8(wb[r][q], b0, b1, acc1[r]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int row = (rg * WV + wv) * RW + r;
            const float v0 = wsum(acc0[r]), v1 = wsum(acc1[r]);
            if (act[r] && lane == 0) m.part[((size_t) k0 * 2 * HC + slice) * PR + row] = v0;
            if (act[r] && lane == 1 && two) m.part[((size_t) (k0 + 1) * 2 * HC + slice) * PR + row] = v1;
        }
    }
}


// ---- STRATA_GR_RDNA_A8: the Q8 read with int8 activations (per-32 scale, q8_1-style rounding) and int8 dot4
__device__ __forceinline__ float max8(float v) {   // max over each aligned group of 8 lanes (DPP row_xmask 1, 2, 4)
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x161, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x162, 0xf, 0xf, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x164, 0xf, 0xf, false)));
    return v;
}
// 4 floats of an aligned 32-group (8 lanes x 4) -> 4 int8 + the group scale (written by the group's first lane)
__device__ __forceinline__ void quant4(float4 x, uint32_t* q, float* d, bool first) {
    const float amax = max8(fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w))));
    const float dd = amax / 127.0f, id = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int a = __float2int_rn(x.x * id), b = __float2int_rn(x.y * id), c = __float2int_rn(x.z * id), e = __float2int_rn(x.w * id);
    *q = (uint32_t) (a & 0xff) | ((uint32_t) (b & 0xff) << 8) | ((uint32_t) (c & 0xff) << 16) | ((uint32_t) (e & 0xff) << 24);
    if (first) *d = dd;
}
__device__ __forceinline__ int dot8i(uint2 w, uint2 a) {
    return __builtin_amdgcn_sudot4(true, (int) w.y, true, (int) a.y, __builtin_amdgcn_sudot4(true, (int) w.x, true, (int) a.x, 0, false), false);
}

// STRATA_GR_INJ_UP=1 (!INJ): the inject rows' slice dots in their own blocks (row group LR / 16, one per slice) with
// no LDS - wave w takes (row, token) items w, w + 8, ...; each is the INJ kernel's lane chunks and dot8 order over
// R'[c,h] * w_norm (the staging expressions), then wsum. The int8 blocks then need no float copy of the rows.
template <int T>
__device__ __forceinline__ void inject_slice(const RdnaGr& m, int slice) {
    const int t = threadIdx.x, lane = t & 31, wv = t >> 5, c = slice >> 1, h = slice & 1, e0 = c * N + h * H;
    for (int it = wv; it < (PR - LR) * T; it += WV) {
        const int row = it / T, k = it - row * T;
        const FusedGrArgs& a = m.a[k];
        const uint4* w4 = reinterpret_cast<const uint4*>(a.w_inject + (size_t) row * D + e0);
        const float gw = a.apply ? 2.0f * sigm(a.inj_prev[c] / (float) HC) : 0.0f;
        uint4 wb[HQ];
#pragma unroll
        for (int q = 0; q < HQ; ++q) wb[q] = ntload(w4 + lane + 32 * q);
        float acc = 0.0f;
#pragma unroll
        for (int q = 0; q < HQ; ++q) {
            const int i = (lane + 32 * q) * 8;
            float4 v[2];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i + 4 * u);
                if (a.apply) {
                    const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i + 4 * u);
                    r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
                }
                const float4 g = *reinterpret_cast<const float4*>(a.w_norm + e0 + i + 4 * u);
                v[u] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
            }
            acc = dot8(wb[q], v[0], v[1], acc);
        }
        const float s = wsum(acc);
        if (lane == 0) m.part[((size_t) k * 2 * HC + slice) * PR + LR + row] = s;
    }
}

template <int T, int RW, bool INJ = true>
__device__ __forceinline__ void down_a8_body(const RdnaGr& m, const int bx, const int by) {
    // !INJ (STRATA_GR_INJ_UP): the 4 BF16 inject rows are rdna_gr_up's extra blocks - no float copy of the rows here
    __shared__ __align__(16) float xs[INJ ? T : 1][INJ ? H : 4];   // float rows, for the BF16 inject rows
    __shared__ __align__(16) uint32_t xq[T][H / 4];       // int8 rows
    __shared__ float xd[T][H / 32];
    __shared__ float sq[WV][T];
    const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
    const int rg = bx, slice = by, c = slice >> 1, h = slice & 1;
    if constexpr (!INJ) {
        if (rg == LR / (RW * WV)) { if (m.a[0].w_inject != nullptr) inject_slice<T>(m, slice); return; }
    }
    const int8_t* qd = m.a[0].r8_down;
    const uint16_t* sd = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(qd) + r8_soff((size_t) LR * D));
    uint2 wq[RW][HQ];
    float wd[RW][HQ];
    uint4 wb[RW][HQ];
    bool act[RW], inj[RW];
    bool any_inj = false;
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        const int row = (rg * WV + wv) * RW + r;
        inj[r] = INJ && row >= LR;
        act[r] = row < LR || (INJ && row < PR && m.a[0].w_inject != nullptr);
        any_inj |= inj[r] && act[r];
        if (!act[r]) continue;
        if (!inj[r]) {
            const uint2* w2 = reinterpret_cast<const uint2*>(qd + (size_t) row * D + c * N + h * H);
            const uint16_t* s2 = sd + (size_t) row * (D / 32) + (c * N + h * H) / 32;
#pragma unroll
            for (int q = 0; q < HQ; ++q) { wq[r][q] = ntload2(w2 + lane + 32 * q); wd[r][q] = h2f(s2[(lane + 32 * q) >> 2]); }
        } else {
            const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_inject + (size_t) (row - LR) * D + c * N + h * H);
#pragma unroll
            for (int q = 0; q < HQ; ++q) wb[r][q] = ntload(w4 + lane + 32 * q);
        }
    }
    const int e0 = c * N + h * H;   // this slice's first element in D
    // stage R'[c,h] * w_norm for every token: one flattened (token, float4) loop, loads of several tokens in flight
#pragma unroll 4
    for (int it = t; it < T * (H / 4); it += TH) {
        const int k = it / (H / 4), i = (it - k * (H / 4)) * 4;
        const FusedGrArgs& a = m.a[k];
        float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
        if (a.apply) {
            const float gw = 2.0f * sigm(a.inj_prev[c] / (float) HC);
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + e0 + i);
            const float4 v = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
            if constexpr (INJ) *reinterpret_cast<float4*>(&xs[k][i]) = v;
            quant4(v, &xq[k][i >> 2], &xd[k][i >> 5], (i & 31) == 0);
    }
    if (rg == 0) {   // the slice's sums of squares (only row group 0 publishes them)
#pragma unroll 1
        for (int k = 0; k < T; ++k) {
            const FusedGrArgs& a = m.a[k];
            const float gw = a.apply ? 2.0f * sigm(a.inj_prev[c] / (float) HC) : 0.0f;
            float s = 0.0f;
            for (int i = t * 4; i < H; i += TH * 4) {
                float4 r = *reinterpret_cast<const float4*>(a.R + e0 + i);
                if (a.apply) {
                    const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + h * H + i);
                    r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y); r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
                }
                s = fmaf(r.w, r.w, fmaf(r.z, r.z, fmaf(r.y, r.y, fmaf(r.x, r.x, s))));   // explicit: the same in every instance
            }
            const float v = wsum(s);
            if (lane == 0) sq[wv][k] = v;
        }
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float s = 0.0f;
        for (int w = 0; w < WV; ++w) s += sq[w][t];
        m.ss[t * 2 * HC + slice] = s;
    }
#pragma unroll 1
    for (int k = 0; k < T; ++k) {
        float acc[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) acc[r] = 0.0f;
#pragma unroll
        for (int q = 0; q < HQ; ++q) {
            const int ch = lane + 32 * q;                  // chunk of 8 values
            const uint2 av = *reinterpret_cast<const uint2*>(&xq[k][ch * 2]);
            const float ad = xd[k][ch >> 2];
#pragma unroll
            for (int r = 0; r < RW; ++r) {
                if (!inj[r]) acc[r] = fmaf(wd[r][q] * ad, (float) dot8i(wq[r][q], av), acc[r]);
                else if (INJ && any_inj) {
                    if constexpr (INJ) {
                    const float4 x0 = *reinterpret_cast<const float4*>(&xs[k][ch * 8]);
                    const float4 x1 = *reinterpret_cast<const float4*>(&xs[k][ch * 8 + 4]);
                    acc[r] = dot8(wb[r][q], x0, x1, acc[r]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            const int row = (rg * WV + wv) * RW + r;
            const float v = wsum(acc[r]);
            if (act[r] && lane == 0) m.part[((size_t) k * 2 * HC + slice) * PR + row] = v;
        }
    }
}

template <int T, int RW, bool INJ = true>
__global__ void __launch_bounds__(TH) __attribute__((amdgpu_waves_per_eu(STRATA_GR_WPE, 16))) rdna_gr_down_a8(RdnaGr m) {
    down_a8_body<T, RW, INJ>(m, blockIdx.x, blockIdx.y);
}

// STRATA_QFUSE: the consumer's q8_1 image of `mixed`, as fused_gr.cu's gr_q8_tail - the two blocks of a 32-column
// group count in, the last one quantizes the group (one wave per token, the same bytes as native_quantize_q8_1)
struct RdnaQ81 { half2 ds; int8_t qs[32]; };
template <int T>
__device__ __forceinline__ void q8_tail(const RdnaGr& m, int d0) {
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* c = m.a[0].q8_cnt + d0 / 32;
        last = atomicAdd(c, 1u) == 1u;
        if (last) atomicExch(c, 0u);
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (wave >= T) return;
    const int c0 = (d0 / 32) * 32;
    const float xi = *reinterpret_cast<const volatile float*>(m.a[wave].mixed + c0 + lane);
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    RdnaQ81* y = reinterpret_cast<RdnaQ81*>(m.a[wave].q8_mixed) + c0 / 32;
    y->qs[lane] = q;
    if (lane == 0) y->ds = q8_1_ds(d, sum);
}

// The up pass's per-thread inputs that do not depend on the down pass (its weight quarter-rows, R and w_norm at its
// element): rdna_gr_fused issues them BEFORE the down pass, so their DRAM latency hides under it instead of following
// the grid barrier.  The values (and so every product) are the ones up_body loaded itself.
template <int T, bool Q8>
struct UpPre {
    uint4 w[Q8 ? 1 : 10];
    uint2 w8[Q8 ? 10 : 1];
    float d8[Q8 ? 10 : 1];
    float rv[T];
    float wn;
};
template <int T, bool Q8>
__device__ __forceinline__ void up_pre(const RdnaGr& m, const int bx, UpPre<T, Q8>& P) {
    const int t = threadIdx.x;
    // four lanes per row: quarter q reads uint4 q, q+4, ..., q+36 (10 of the row's 40)
    const int rr = t >> 2, q = t & 3;          // 64 rows: stream c = rr / UC, column dd = rr % UC
    const int c = rr / UC, dd = rr - c * UC, d = bx * UC + dd, i = c * N + d;
    if constexpr (Q8) {
        const int8_t* qu = m.a[0].r8_up;
        const uint16_t* su = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(qu) + r8_soff((size_t) D * LR));
        const uint2* w2 = reinterpret_cast<const uint2*>(qu + (size_t) i * LR);
#pragma unroll
        for (int u = 0; u < 10; ++u) { P.w8[u] = ntload2(w2 + q + 4 * u); P.d8[u] = h2f(su[(size_t) i * (LR / 32) + ((q + 4 * u) >> 2)]); }
    } else {
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
#pragma unroll
        for (int u = 0; u < 10; ++u) P.w[u] = ntload(w4 + q + 4 * u);
    }
    P.wn = m.a[0].w_norm[i];
#pragma unroll
    for (int k = 0; k < T; ++k) P.rv[k] = m.a[k].R[i];
}
template <int T, bool Q8 = false, bool A8 = false>
__device__ __forceinline__ void up_main(const RdnaGr& m, const int bx, const UpPre<T, Q8>& P) {
    __shared__ __align__(16) uint32_t loq[A8 ? T : 1][LR / 4];
    __shared__ float lod[A8 ? T : 1][LR / 32];
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float rs[T][HC];
    __shared__ float g[T][HC][UC];
    const int t = threadIdx.x;
    const int rr = t >> 2, q = t & 3;
    const int c = rr / UC, dd = rr - c * UC, d = bx * UC + dd, i = c * N + d;
    const auto& w = P.w;
    const auto& w8 = P.w8;
    const auto& d8 = P.d8;
    const auto& rv = P.rv;
    const float wn = P.wn;
    if (t < T * HC) {
        const int k = t / HC, cs = t - k * HC;
        const float s = m.ss[k * 2 * HC + 2 * cs] + m.ss[k * 2 * HC + 2 * cs + 1];
        rs[k][cs] = rsqrtf(s / (float) N + m.a[k].eps);
        if (bx == 0) m.a[k].rs[cs] = rs[k][cs];
    }
    __syncthreads();
    for (int i = t; i < T * PR; i += TH) {
        const int k = i / PR, j = i - k * PR;
        const float* p = m.part + (size_t) k * 2 * HC * PR + j;
        float s = 0.0f;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) s = fmaf(rs[k][cc], p[(2 * cc) * PR] + p[(2 * cc + 1) * PR], s);
        if (j < LR) {
            const float x = s / (float) HC;
            lo[k][j] = x / (1.0f + __expf(-x));
            if (bx == 0) m.a[k].lo[j] = lo[k][j];
        } else if (bx == 0 && m.a[0].w_inject != nullptr) {
            m.a[k].inject_out[j - LR] = s;
        }
    }
    __syncthreads();
    if constexpr (A8) {   // lo -> int8 per 32 (aligned 8-lane groups of float4)
        for (int i = t; i < T * (LR / 4); i += TH) {
            const int k = i / (LR / 4), f = i - k * (LR / 4);
            const float4 v = *reinterpret_cast<const float4*>(&lo[k][4 * f]);
            quant4(v, &loq[k][f], &lod[k][f >> 3], (f & 7) == 0);
        }
        __syncthreads();
    }
    float acc[T];
#pragma unroll
    for (int k = 0; k < T; ++k) {
        float a = 0.0f;
#pragma unroll
        for (int u = 0; u < 10; ++u) {
            const int j = (q + 4 * u) * 8;
            if constexpr (A8) a = fmaf(d8[u] * lod[k][j >> 5], (float) dot8i(w8[u], *reinterpret_cast<const uint2*>(&loq[k][j >> 2])), a);
            else if constexpr (Q8) a = dot8q(w8[u], d8[u], *reinterpret_cast<const float4*>(&lo[k][j]), *reinterpret_cast<const float4*>(&lo[k][j + 4]), a);
            else a = dot8(w[u], *reinterpret_cast<const float4*>(&lo[k][j]), *reinterpret_cast<const float4*>(&lo[k][j + 4]), a);
        }
        a += __shfl_xor(a, 1);
        a += __shfl_xor(a, 2);
        acc[k] = a;
    }
    if (q == 0) {
#pragma unroll
        for (int k = 0; k < T; ++k) {
            const FusedGrArgs& a = m.a[k];
            float r = rv[k];
            if (a.apply) {
                r = fmaf(a.bo_prev[d], 2.0f * sigm(a.inj_prev[c] / (float) HC), r);
                a.R_out[i] = r;
            }
            g[k][c][dd] = r * wn * rs[k][c] * sigm(acc[k]);
        }
    }
    __syncthreads();
    if (t < T * UC) {
        const int k = t / UC, col = t - k * UC;
        float s = 0.0f;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) s += g[k][cc][col];
        m.a[k].mixed[bx * UC + col] = s / (float) HC;
    }
    if (m.a[0].q8_mixed != nullptr) q8_tail<T>(m, bx * UC);   // STRATA_QFUSE
}

template <int T, bool Q8 = false, bool A8 = false>
__device__ __forceinline__ void up_body(const RdnaGr& m, const int bx) {
    UpPre<T, Q8> P;
    up_pre<T, Q8>(m, bx, P);
    up_main<T, Q8, A8>(m, bx, P);
}

template <int T, bool Q8 = false, bool A8 = false>
__global__ void __launch_bounds__(TH) rdna_gr_up(RdnaGr m) {
    up_body<T, Q8, A8>(m, blockIdx.x);
}

// STRATA_GR_FUSED=1: the down and up passes of the A8 read (inject rows in their own blocks) in ONE launch - every
// block runs its down slice, a sense-reversing grid barrier (all 168 blocks are co-resident: 113 VGPRs, 4.4 KB LDS),
// then the first UB blocks run the up pass.  The same arithmetic per element as the two launches; one graph node less.
__device__ __forceinline__ void grid_barrier(unsigned* bar, unsigned nblocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned* gen = bar + 1;
        const unsigned g = *gen;
        __threadfence();
        if (atomicAdd(bar, 1u) == nblocks - 1) {
            atomicExch(bar, 0u);
            __threadfence();
            atomicAdd(bar + 1, 1u);
        } else {
            while (*gen == g) __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    __threadfence();   // every wave: the other blocks' partials and sums
}
template <int T>
__global__ void __launch_bounds__(TH) __attribute__((amdgpu_waves_per_eu(STRATA_GR_WPE, 16))) rdna_gr_fused(RdnaGr m) {
    constexpr int NBX = LR / (2 * WV) + 1;   // the down grid's x extent (row groups + the inject group)
    const int b = (int) blockIdx.x;
    // STRATA_GR_UPPRE (default on; =0 the earlier order): the up pass's weights are in flight during the down pass
    UpPre<T, true> P;
    if (m.uppre && b < UB) up_pre<T, true>(m, b, P);
    down_a8_body<T, 2, false>(m, b % NBX, b / NBX);
    grid_barrier(m.bar, gridDim.x);
    if (b < UB) {
        if (!m.uppre) up_pre<T, true>(m, b, P);
        up_main<T, true, true>(m, b, P);
    }
}

__global__ void inj_check(const float* a, const float* b, const float* sa, const float* sb, int T, int apply, int inj, volatile float* rec) {
    if (threadIdx.x == 0) atomicAdd((int*) rec + 62, 1);
    for (int i = threadIdx.x; i < T * 2 * HC * PR; i += blockDim.x)
        if (__float_as_uint(a[i]) != __float_as_uint(b[i])) {
            const int n = atomicAdd((int*) rec, 1);
            if (n < 8) { rec[1 + n * 6] = T; rec[2 + n * 6] = apply; rec[3 + n * 6] = inj; rec[4 + n * 6] = i; rec[5 + n * 6] = a[i]; rec[6 + n * 6] = b[i]; }
        }
    for (int i = threadIdx.x; i < T * 2 * HC; i += blockDim.x)
        if (__float_as_uint(sa[i]) != __float_as_uint(sb[i])) atomicAdd((int*) rec + 63, 1);
}
static volatile float* inj_rec() {
    static volatile float* h = nullptr;
    if (!h) {
        void* p = nullptr;
        cudaHostAlloc(&p, 64 * 4, cudaHostAllocMapped);
        std::memset(p, 0, 256);
        h = (volatile float*) p;
        std::thread([] {
            int last = 0;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                const int n = *(volatile int*) h;
                std::fprintf(stderr, "INJCHK calls %d mismatches %d ss %d\n", ((volatile int*) h)[62], n, ((volatile int*) h)[63]);
                if (n != last) {
                    last = n;
                    std::fprintf(stderr, "INJCHK mismatches %d ss %d\n", n, ((volatile int*) h)[63]);
                    for (int j = 0; j < 8 && j < n; ++j) {
                        const int i = (int) h[4 + j * 6], T = (int) h[1 + j * 6];
                        std::fprintf(stderr, "INJCHK T%d apply%d inj%d i%d k%d slice%d row%d new %.9g old %.9g\n", T, (int) h[2 + j * 6],
                                     (int) h[3 + j * 6], i, i / (2 * HC * PR), (i / PR) % (2 * HC), i % PR, h[5 + j * 6], h[6 + j * 6]);
                    }
                }
            }
        }).detach();
    }
    static void* d = nullptr;
    if (!d) cudaHostGetDevicePointer(&d, (void*) h, 0);
    return (volatile float*) d;
}
template <int T>
void launch(const RdnaGr& m, cudaStream_t st, unsigned long long* sb, int si) {
    static const int only = [] { const char* v = std::getenv("STRATA_GR_RDNA_ONLY"); return v ? std::atoi(v) : 0; }();   // bench: 1 down, 2 up
    static const int rw = [] { const char* v = std::getenv("STRATA_GR_RDNA_RW"); return v ? std::atoi(v) : 2; }();
    if (sb) gpu_stamp(sb, si, st);
    if (m.a[0].r8_down != nullptr && m.a[0].r8_up != nullptr) {   // STRATA_GR_RDNA_Q8
        static const bool a8 = [] { const char* v = std::getenv("STRATA_GR_RDNA_A8"); return v && v[0] == '1'; }();
        if (a8) {
            static const bool inj_up = [] { const char* v = std::getenv("STRATA_GR_INJ_UP"); return v && v[0] == '1'; }();
            static const bool fused = [] { const char* v = std::getenv("STRATA_GR_FUSED"); return v && v[0] == '1'; }();
            if (inj_up && fused && m.bar != nullptr && sb == nullptr) {   // one launch (the profiler's stamps need two)
                rdna_gr_fused<T><<<(LR / (2 * WV) + 1) * 2 * HC, TH, 0, st>>>(m);
                return;
            }
            if (inj_up) {   // the inject rows in their own blocks: no float copy of the rows in the int8 blocks
                static const bool chk = std::getenv("STRATA_GR_INJ_CHECK") != nullptr;
                if (chk) {   // debug: the INJ kernel into a second part buffer, compared after the new one
                    RdnaGr m2 = m; m2.part = m.part + 22000; m2.ss = m2.part + (size_t) T * 2 * HC * PR;
                    rdna_gr_down_a8<T, 2><<<dim3((PR + 2 * WV - 1) / (2 * WV), 2 * HC), TH, 0, st>>>(m2);
                    rdna_gr_down_a8<T, 2, false><<<dim3(LR / (2 * WV) + 1, 2 * HC), TH, 0, st>>>(m);
                    inj_check<<<1, 256, 0, st>>>(m.part, m2.part, m.ss, m2.ss, T, m.a[0].apply ? 1 : 0, m.a[0].w_inject != nullptr, inj_rec());
                } else
                if (only != 2) rdna_gr_down_a8<T, 2, false><<<dim3(LR / (2 * WV) + 1, 2 * HC), TH, 0, st>>>(m);
                if (sb) gpu_stamp(sb, si + 1, st);
                if (only != 1) rdna_gr_up<T, true, true><<<UB, TH, 0, st>>>(m);
                return;
            }
            if (only != 2) rdna_gr_down_a8<T, 2><<<dim3((PR + 2 * WV - 1) / (2 * WV), 2 * HC), TH, 0, st>>>(m);
            if (sb) gpu_stamp(sb, si + 1, st);
            if (only != 1) rdna_gr_up<T, true, true><<<UB, TH, 0, st>>>(m);
            return;
        }
        if (only != 2) rdna_gr_down_q8<T, 2><<<dim3((PR + 2 * WV - 1) / (2 * WV), 2 * HC), TH, 0, st>>>(m);
        if (sb) gpu_stamp(sb, si + 1, st);
        if (only != 1) rdna_gr_up<T, true><<<UB, TH, 0, st>>>(m);
        return;
    }
    if (only != 2) {
        if (rw == 1) rdna_gr_down<T, 1><<<dim3((PR + WV - 1) / WV, 2 * HC), TH, 0, st>>>(m);
        else if (rw == 4) rdna_gr_down<T, 4><<<dim3((PR + 4 * WV - 1) / (4 * WV), 2 * HC), TH, 0, st>>>(m);
        else rdna_gr_down<T, 2><<<dim3((PR + 2 * WV - 1) / (2 * WV), 2 * HC), TH, 0, st>>>(m);
    }
    if (sb) gpu_stamp(sb, si + 1, st);
    if (only != 1) rdna_gr_up<T><<<UB, TH, 0, st>>>(m);
}
}  // namespace


// ---- STRATA_GR_RDNA_Q8 with the canonical BF16 copies skipped: other readers (the prompt path, the single-token
// layer) get a BF16 image dequantized from hc_r8 into a per-device scratch, one per role (down / up), right before use
// on their own stream.  The scratch is allocated at load (never during a graph capture).
namespace {
__global__ void r8_to_bf16_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ sc, uint16_t* __restrict__ out,
                                  long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = (float) q[i] * h2f(sc[i >> 5]);
    const uint32_t u = __float_as_uint(v);
    out[i] = (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);   // round to nearest even
}
uint16_t* g_r8_scratch[64][2] = {};
}  // namespace

bool rdna_r8_scratch_init(long long elems) {
    if (std::getenv("STRATA_GR_INJ_CHECK")) (void) inj_rec();   // debug buffer: mapped at load, not during a capture
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (g_gr_bar[dev] == nullptr) {   // STRATA_GR_FUSED's barrier words
        if (cudaMalloc(&g_gr_bar[dev], 64) != cudaSuccess || cudaMemset(g_gr_bar[dev], 0, 64) != cudaSuccess) return false;
    }
    for (int r = 0; r < 2; ++r)
        if (g_r8_scratch[dev][r] == nullptr && cudaMalloc(&g_r8_scratch[dev][r], (size_t) elems * 2) != cudaSuccess) return false;
    return true;
}

const uint16_t* rdna_r8_to_bf16(const int8_t* r8, long long rows, long long cols, int role, void* stream) {
    int dev = 0;
    cudaGetDevice(&dev);
    uint16_t* out = (dev >= 0 && dev < 64) ? g_r8_scratch[dev][role & 1] : nullptr;
    if (out == nullptr) {
        std::fprintf(stderr, "rdna_r8_to_bf16: no scratch on device %d\n", dev);
        std::exit(1);
    }
    const long long n = rows * cols;
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(r8) + r8_soff((size_t) n));
    r8_to_bf16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(r8, sc, out, n);
    return out;
}

bool rdna_gr_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_GR_RDNA"); return v != nullptr && v[0] == '1'; }();
    return on;
}

// scratch: at least T * 2*HC * (PR + 1) floats (the caller's xn scratch, T * HC * N floats, is far larger)
void rdna_gr_read_multi(const FusedGrArgs* a, int n_tok, float* scratch, void* stream, unsigned long long* sb, int si) {
    RdnaGr m;
    for (int t = 0; t < n_tok; ++t) m.a[t] = a[t];
    m.part = scratch;
    m.ss = scratch + (size_t) n_tok * 2 * HC * PR;
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64) m.bar = g_gr_bar[dev];
    static const bool uppre = [] { const char* v = std::getenv("STRATA_GR_UPPRE"); return v == nullptr || v[0] != '0'; }();
    m.uppre = uppre;
    cudaStream_t st = (cudaStream_t) stream;
    switch (n_tok) {
        case 1: launch<1>(m, st, sb, si); break;
        case 2: launch<2>(m, st, sb, si); break;
        case 3: launch<3>(m, st, sb, si); break;
        case 4: launch<4>(m, st, sb, si); break;
        case 5: launch<5>(m, st, sb, si); break;
        case 6: launch<6>(m, st, sb, si); break;
        case 7: launch<7>(m, st, sb, si); break;
        default: launch<8>(m, st, sb, si); break;
    }
}
}  // namespace strata::kernels
