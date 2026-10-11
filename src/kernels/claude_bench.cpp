// claude_bench: isolated decode-kernel timings at Flash-Next production shapes (gfx1100 tuning work).
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>
namespace K = strata::kernels;
static void ck(cudaError_t e, const char* at) { if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", at, cudaGetErrorString(e)); std::exit(2); } }
static cudaStream_t st;
static void* dalloc_rand(size_t bytes, unsigned seed, bool sane_fp16 = true) {
    std::vector<uint8_t> h(bytes); std::mt19937 r(seed);
    for (size_t i = 0; i < bytes; ++i) h[i] = (uint8_t) r();
    if (sane_fp16) for (size_t i = 1; i < bytes; i += 2) h[i] &= 0x3b;   // keep fp16/bf16 high bytes finite and small
    void* p; ck(cudaMalloc(&p, bytes), "malloc"); ck(cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice), "upload"); return p;
}
// time `calls` launches rotating over `reps` replicas inside one graph; return us per call
static double timeit(int reps, int calls, const std::function<void(int)>& f) {
    for (int i = 0; i < reps; ++i) f(i);
    ck(cudaStreamSynchronize(st), "warm");
    cudaGraph_t g; cudaGraphExec_t e;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "cap");
    for (int i = 0; i < calls; ++i) f(i % reps);
    ck(cudaStreamEndCapture(st, &g), "capend"); ck(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0), "inst");
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    ck(cudaGraphLaunch(e, st), "l0"); ck(cudaStreamSynchronize(st), "s0");
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(a, st); cudaGraphLaunch(e, st); cudaEventRecord(b, st); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b); best = std::min(best, ms * 1000.0 / calls);
    }
    cudaGraphExecDestroy(e); cudaGraphDestroy(g);
    return best;
}
static void report(const char* name, double us, double bytes) {
    std::printf("%-44s %8.2f us  %7.1f GB/s\n", name, us, bytes / us / 1e3);
}
int main(int argc, char** argv) {
    const std::string which = argc > 1 ? argv[1] : "all";
    const int TT = argc > 2 ? std::atoi(argv[2]) : 3;
    ck(cudaStreamCreate(&st), "stream");
    const size_t WS = 384ull << 20;   // rotating working set > 96 MiB infinity cache
    if (which == "all" || which == "gr" || which == "grq8") {
        constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
        const size_t wb = D * 4 + 2 * (size_t) D * LR * 2 + (size_t) HC * D * 2;
        const int reps = (int) (WS / wb) + 1;
        std::vector<uint8_t*> w(reps);
        for (int i = 0; i < reps; ++i) w[i] = (uint8_t*) dalloc_rand(wb, 11 + i);
        // int8 copies (rdna layout: int8 [rows][cols] then fp16 scales at a 256-aligned offset) for the Q8 read
        const size_t nmat = (size_t) D * LR, r8b = (nmat + 255) / 256 * 256 + nmat / 32 * 2;
        std::vector<int8_t*> r8d(reps), r8u(reps);
        for (int i = 0; i < reps; ++i) {
            std::vector<uint8_t> h(r8b); std::mt19937 rr(500 + i);
            for (size_t k = 0; k < nmat; ++k) h[k] = (uint8_t) (int8_t) ((int) (rr() % 255) - 127);
            uint16_t* sc = (uint16_t*) (h.data() + (nmat + 255) / 256 * 256);
            for (size_t k = 0; k < nmat / 32; ++k) sc[k] = 0x1c00 + (rr() % 256);   // fp16 ~ 0.004
            ck(cudaMalloc(&r8d[i], r8b), "r8d"); ck(cudaMemcpy(r8d[i], h.data(), r8b, cudaMemcpyHostToDevice), "r8d up");
            ck(cudaMalloc(&r8u[i], r8b), "r8u"); ck(cudaMemcpy(r8u[i], h.data(), r8b, cudaMemcpyHostToDevice), "r8u up");
        }
        float *R = (float*) dalloc_rand(8 * D * 4, 3), *bo = (float*) dalloc_rand(8 * N * 4, 4), *inj = (float*) dalloc_rand(8 * HC * 4, 5);
        float *xn, *lo, *rs, *io, *mix;
        ck(cudaMalloc(&xn, 8 * D * 4 * 2), "xn"); ck(cudaMalloc(&lo, 8 * LR * 4), "lo"); ck(cudaMalloc(&rs, 8 * HC * 4), "rs");
        ck(cudaMalloc(&io, 8 * HC * 4), "io"); ck(cudaMalloc(&mix, 8 * N * 4), "mix");
        for (int T = 1; T <= 6; ++T) {
            if (TT > 0 && T != TT && which == "gr") continue;
            auto f = [&](int r) {
                K::FusedGrArgs a[8];
                for (int t = 0; t < T; ++t) {
                    auto& x = a[t]; uint8_t* p = w[r];
                    x.R = R + t * D; x.R_out = R + t * D; x.apply = true; x.bo_prev = bo + t * N; x.inj_prev = inj + t * HC;
                    x.w_norm = (const float*) p; x.w_down = (const uint16_t*) (p + D * 4); x.w_up = (const uint16_t*) (p + D * 4 + (size_t) D * LR * 2);
                    x.w_inject = (const uint16_t*) (p + D * 4 + 2 * (size_t) D * LR * 2);
                    x.lo = lo + t * LR; x.rs = rs + t * HC; x.inject_out = io + t * HC; x.mixed = mix + t * N;
                }
                K::fused_gr_read_multi(a, T, xn, st);
            };
            char nm[64]; std::snprintf(nm, 64, "gr_read T%d (13.2MB)", T);
            if (which != "grq8") report(nm, timeit(reps, 400, f), (double) wb);
            auto f2 = [&](int r) {
                K::FusedGrArgs a[8];
                for (int t = 0; t < T; ++t) {
                    auto& x = a[t]; uint8_t* p = w[r];
                    x.R = R + t * D; x.R_out = R + t * D; x.apply = true; x.bo_prev = bo + t * N; x.inj_prev = inj + t * HC;
                    x.w_norm = (const float*) p; x.w_down = (const uint16_t*) (p + D * 4); x.w_up = (const uint16_t*) (p + D * 4 + (size_t) D * LR * 2);
                    x.w_inject = (const uint16_t*) (p + D * 4 + 2 * (size_t) D * LR * 2);
                    x.lo = lo + t * LR; x.rs = rs + t * HC; x.inject_out = io + t * HC; x.mixed = mix + t * N;
                }
                K::rdna_gr_read_multi(a, T, xn, st);
            };
            // numeric comparison on replica 0, bo zeroed so the in-place R update is idempotent
            ck(cudaMemset(bo, 0, 8 * N * 4), "bo0");
            auto grab = [&](std::vector<float>& o) {
                ck(cudaStreamSynchronize(st), "s"); o.resize(T * (LR + HC + HC + N));
                ck(cudaMemcpy(o.data(), lo, T * LR * 4, cudaMemcpyDeviceToHost), "c1");
                ck(cudaMemcpy(o.data() + T * LR, rs, T * HC * 4, cudaMemcpyDeviceToHost), "c2");
                ck(cudaMemcpy(o.data() + T * (LR + HC), io, T * HC * 4, cudaMemcpyDeviceToHost), "c3");
                ck(cudaMemcpy(o.data() + T * (LR + 2 * HC), mix, T * N * 4, cudaMemcpyDeviceToHost), "c4");
            };
            std::vector<float> o1, o2;
            if (which != "grq8") { f(0); grab(o1); f2(0); grab(o2); } else { o1.assign(1, 0.f); o2 = o1; }
            double maxrel = 0, maxabs = 0, ref = 0;
            for (size_t i = 0; i < o1.size(); ++i) { maxabs = std::max(maxabs, (double) std::fabs(o1[i] - o2[i])); ref = std::max(ref, (double) std::fabs(o1[i])); }
            for (size_t i = 0; i < o1.size(); ++i) if (!std::isfinite(o2[i])) maxrel = 1e9;
            maxrel = std::max(maxrel, maxabs / (ref + 1e-30));
            std::snprintf(nm, 64, "rdna_gr T%d (maxabs/maxref %.2e)", T, maxrel);
            if (which != "grq8") report(nm, timeit(reps, 400, f2), (double) wb);
            auto f3 = [&](int r) {
                K::FusedGrArgs a[8];
                for (int t = 0; t < T; ++t) {
                    auto& x = a[t]; uint8_t* p = w[r];
                    x.R = R + t * D; x.R_out = R + t * D; x.apply = true; x.bo_prev = bo + t * N; x.inj_prev = inj + t * HC;
                    x.w_norm = (const float*) p; x.w_down = nullptr; x.w_up = nullptr;
                    x.w_inject = (const uint16_t*) (p + D * 4 + 2 * (size_t) D * LR * 2);
                    x.r8_down = r8d[r]; x.r8_up = r8u[r];
                    x.lo = lo + t * LR; x.rs = rs + t * HC; x.inject_out = io + t * HC; x.mixed = mix + t * N;
                }
                K::rdna_gr_read_multi(a, T, xn, st);
            };
            std::snprintf(nm, 64, "rdna_gr q8 T%d", T);
            report(nm, timeit(reps, which == "grq8" ? 60 : 400, f3), (double) r8b * 2);
        }
    }
    if (which == "all" || which == "mmvf") {
        const int shapes[][2] = {{2560, 512}, {2560, 1}, {320, 10240}, {2560, 128}};
        float* x = (float*) dalloc_rand(8 * 10240 * 4, 7, false);
        std::vector<float> hx(8 * 10240); for (auto& v : hx) v = (float) (rand() % 2001 - 1000) / 1000.0f;
        ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
        float* y; ck(cudaMalloc(&y, 8 * 10240 * 4), "y");
        for (auto& sh : shapes) {
            const size_t wb = (size_t) sh[0] * sh[1] * 2;
            const int reps = (int) (WS / wb) + 1 > 4096 ? 4096 : (int) (WS / wb) + 1;
            std::vector<void*> w(reps);
            for (int i = 0; i < reps; ++i) w[i] = dalloc_rand(wb, 300 + i);
            for (int T : {1, 3, 6}) {
                char nm[96]; std::snprintf(nm, 96, "bf16 mmvf %d->%d T%d (%.2fMB)", sh[0], sh[1], T, wb / 1e6);
                report(nm, timeit(reps, 300, [&](int r) { K::bf16_gemv_fp32_mmvf_multi(x, sh[0], (const uint16_t*) w[r], y, sh[1], sh[0], sh[1], T, st); }), (double) wb);
            }
            for (void* p : w) cudaFree(p);
        }
    }
    if (which == "all" || which == "mmvq") {
        struct S { int type; int nin, nout; const char* name; };
        const S shapes[] = {{14, 2560, 10240, "Q6_K qkv 2560->10240"}, {13, 2560, 10240, "Q5_K qkv"}, {12, 2560, 10240, "Q4_K qkv"},
                            {23, 2560, 10240, "IQ4_XS qkv"}, {14, 2560, 6144, "Q6_K gate 2560->6144"}, {14, 6144, 2560, "Q6_K out 6144->2560"},
                            {14, 2560, 12288, "Q6_K q 2560->12288"}, {14, 2560, 512, "Q6_K k 2560->512"}, {8, 2560, 10240, "Q8_0 qkv"},
                            {14, 2560, 640, "Q6_K shexp 2560->640"}, {20, 640, 2560, "IQ4_NL shexp down 640->2560"},
                            {14, 2560, 248320, "Q6_K head 2560->248320"}};
        void* xq; ck(cudaMalloc(&xq, 8 * 6144 / 32 * 36 + 4096), "xq");
        float* x = (float*) dalloc_rand(8 * 6144 * 4, 9, false);
        std::vector<float> hx(8 * 6144); for (auto& v : hx) v = (float) (rand() % 2001 - 1000) / 1000.0f;
        ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
        float* y; ck(cudaMalloc(&y, 8ull * 248320 * 4), "y");
        for (const S& s : shapes) {
            const size_t wb = K::native_mmvq_weight_bytes(s.type, s.nin, s.nout);
            const int reps = (int) (WS / wb) + 1;
            std::vector<void*> w(reps);
            for (int i = 0; i < reps; ++i) w[i] = dalloc_rand(wb, 100 + i);
            K::native_quantize_q8_1(x, xq, s.nin, 6, st);
            for (int T : {1, 2, 3, 4, 6}) {
                if (TT > 0 && T != TT && which == "mmvq") continue;
                char nm[96]; std::snprintf(nm, 96, "%s T%d (%.1fMB)", s.name, T, wb / 1e6);
                report(nm, timeit(reps, 300, [&](int r) { K::native_mmvq(s.type, w[r], xq, y, s.nin, s.nout, T, st); }), (double) wb);
            }
            if (s.type == 14 || s.type == 12 || s.type == 13 || s.type == 23) {   // the RDNA kernels: same timing loop + max relative difference vs the native kernel
                float *y1, *y2; ck(cudaMalloc(&y1, 8ull * s.nout * 4), "y1"); ck(cudaMalloc(&y2, 8ull * s.nout * 4), "y2");
                for (int T : {1, 2, 3, 4, 6}) {
                    if (TT > 0 && T != TT && which == "mmvq") continue;
                    K::native_mmvq(s.type, w[0], xq, y1, s.nin, s.nout, T, st);
                    if (s.type == 14) K::rdna_q6k_mmvq(w[0], xq, y2, s.nin, s.nout, T, st); else K::rdna_kq_mmvq(s.type, w[0], xq, y2, s.nin, s.nout, T, st);
                    std::vector<float> a((size_t) T * s.nout), b((size_t) T * s.nout);
                    ck(cudaMemcpy(a.data(), y1, a.size() * 4, cudaMemcpyDeviceToHost), "a"); ck(cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost), "b");
                    double md = 0, mr = 0; for (size_t i = 0; i < a.size(); ++i) { md = std::max(md, (double) std::fabs(a[i] - b[i])); mr = std::max(mr, (double) std::fabs(a[i])); }
                    char nm[96]; std::snprintf(nm, 96, "  rdna %s T%d (rel %.1e)", s.name, T, md / (mr + 1e-30));
                    report(nm, timeit(reps, 300, [&](int r) { if (s.type == 14) K::rdna_q6k_mmvq(w[r], xq, y, s.nin, s.nout, T, st); else K::rdna_kq_mmvq(s.type, w[r], xq, y, s.nin, s.nout, T, st); }), (double) wb);
                }
                cudaFree(y1); cudaFree(y2);
            }
            for (void* p : w) cudaFree(p);
        }
        for (int T : {1, 3, 6}) {
            char nm[64]; std::snprintf(nm, 64, "quantize_q8_1 2560 T%d", T);
            report(nm, timeit(1, 500, [&](int) { K::native_quantize_q8_1(x, xq, 2560, T, st); }), 0);
        }
    }
    return 0;
}
