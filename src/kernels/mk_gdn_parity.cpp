// src/kernels/mk_gdn_parity.cpp - the persistent GDN dense kernel (megakernel.hpp) against the window's six kernels,
// bitwise, and timed over 36 layers of distinct weights (>= 256 MB rotate past the Infinity Cache).
// Run with STRATA_RDNA_MMVQ_WAVE=3 (the window's Q8_0 kernel on gfx1100); --selftest for the parity only.
#include "strata/kernels/megakernel.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

namespace {
uint16_t bf16_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) (u >> 16); }
uint16_t f16_of(float f) {   // round-to-nearest fp16 for the Q8_0 scales (small positive values)
    uint32_t x; std::memcpy(&x, &f, 4);
    const int e = (int) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return 0;
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}
std::vector<uint8_t> q80_matrix(std::mt19937& rng, int n_in, int n_out) {
    std::vector<uint8_t> w((size_t) n_out * (size_t) (n_in / 32) * 34);
    std::uniform_real_distribution<float> ds(0.002f, 0.02f);
    std::uniform_int_distribution<int> qs(-127, 127);
    for (size_t b = 0; b < w.size() / 34; ++b) {
        const uint16_t d = f16_of(ds(rng));
        std::memcpy(w.data() + b * 34, &d, 2);
        for (int j = 0; j < 32; ++j) w[b * 34 + 2 + j] = (uint8_t) (int8_t) qs(rng);
    }
    return w;
}
template <class T> T* up(const std::vector<T>& v) { T* d = nullptr; CK(cudaMalloc((void**) &d, v.size() * sizeof(T))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <class T> std::vector<T> down(const T* d, size_t n) { std::vector<T> v(n); CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
template <class T> size_t ndiff(const std::vector<T>& a, const std::vector<T>& b) { size_t n = 0; for (size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(T)) != 0; return n; }
}  // namespace

int main(int argc, char** argv) {
    using namespace strata::kernels;
    const bool selftest = argc > 1 && std::string(argv[1]) == "--selftest";
    const int N = 2560, HK = 8, HV = 24, S = 128;   // TP's half geometry (card 0)
    const int C = 2 * HK * S + HV * S, ZV = HV * S;
    const float eps = 1e-6f;
    const int T = std::getenv("STRATA_MK_T") ? std::atoi(std::getenv("STRATA_MK_T")) : 4;   // 1..8 tokens
    const int L = std::getenv("STRATA_MK_L") ? std::atoi(std::getenv("STRATA_MK_L")) : (selftest ? 2 : 36);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> xs(-1.0f, 1.0f), small(-0.05f, 0.05f), ws(-0.2f, 0.2f);
    // the activations
    std::vector<float> x((size_t) T * N);
    for (float& v : x) v = xs(rng);
    float* d_x = up(x);
    // per-layer weights and state
    struct Layer { void *qkv, *z, *out; uint16_t *wa, *wb; float *dt, *a, *gamma, *conv, *state, *hist; };
    std::vector<Layer> lay(L);
    for (int l = 0; l < L; ++l) {
        lay[l].qkv = up(q80_matrix(rng, N, C)); lay[l].z = up(q80_matrix(rng, N, ZV)); lay[l].out = up(q80_matrix(rng, ZV, N));
        std::vector<uint16_t> wa((size_t) HV * N), wb((size_t) HV * N);
        for (auto& v : wa) v = bf16_of(ws(rng));
        for (auto& v : wb) v = bf16_of(ws(rng));
        lay[l].wa = up(wa); lay[l].wb = up(wb);
        std::vector<float> dt(HV), a(HV), gamma(S), conv((size_t) C * 4), state((size_t) S * HV * S), hist((size_t) C * 3);
        for (auto& v : dt) v = xs(rng);
        for (auto& v : a) v = -std::abs(xs(rng));
        for (auto& v : gamma) v = 1.0f + small(rng);
        for (auto& v : conv) v = ws(rng);
        for (auto& v : state) v = small(rng);
        for (auto& v : hist) v = xs(rng);
        lay[l].dt = up(dt); lay[l].a = up(a); lay[l].gamma = up(gamma); lay[l].conv = up(conv); lay[l].state = up(state); lay[l].hist = up(hist);
    }
    // scratch, two sets: the reference's and the megakernel's
    struct Buf { void *xq, *yq; float *qkv, *z, *h, *gate, *beta, *y, *bo; };
    auto mk_buf = [&]() {
        Buf b;
        CK(cudaMalloc(&b.xq, native_q8_1_bytes(N, 8))); CK(cudaMalloc(&b.yq, native_q8_1_bytes(ZV, 8)));
        CK(cudaMalloc((void**) &b.qkv, (size_t) T * C * 4)); CK(cudaMalloc((void**) &b.z, (size_t) T * ZV * 4));
        CK(cudaMalloc((void**) &b.h, (size_t) T * C * 4)); CK(cudaMalloc((void**) &b.gate, (size_t) T * HV * 4));
        CK(cudaMalloc((void**) &b.beta, (size_t) T * HV * 4)); CK(cudaMalloc((void**) &b.y, (size_t) T * ZV * 4));
        CK(cudaMalloc((void**) &b.bo, (size_t) T * N * 4));
        return b;
    };
    Buf ref = mk_buf(), mk = mk_buf();
    cudaStream_t st; CK(cudaStreamCreate(&st));
    // the counters live in coherent mapped host memory under STRATA_MK_HOSTCNT so a watchdog thread can print them
    // while the kernel runs (a hung phase is then visible)
    unsigned* counters = nullptr;
    unsigned* host_cnt = nullptr;
    std::atomic<bool> watch_stop{false};
    std::thread watch;
    if (std::getenv("STRATA_MK_HOSTCNT")) {
        CK(cudaHostAlloc((void**) &host_cnt, 64, cudaHostAllocMapped));
        std::memset(host_cnt, 0, 64);
        CK(cudaHostGetDevicePointer((void**) &counters, host_cnt, 0));
        watch = std::thread([&] {
            while (!watch_stop.load()) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                volatile unsigned* c = host_cnt;
                std::printf("[watch] done %u %u %u %u %u  next %u %u %u %u %u  b0 %u entered %u max-item %u stalled %u\n", c[0], c[1], c[2], c[3], c[4],
                            c[8], c[9], c[10], c[11], c[12], c[5], c[6], c[7], c[15]);
                std::fflush(stdout);
            }
        });
    } else {
        counters = mk_counters_alloc();
    }
    auto run_ref = [&](int l, cudaStream_t s) {
        native_quantize_q8_1(d_x, ref.xq, N, T, s);
        native_mmvq(8, lay[l].qkv, ref.xq, ref.qkv, N, C, T, s);
        native_mmvq(8, lay[l].z, ref.xq, ref.z, N, ZV, T, s);
        gdn_conv_ab_multi(lay[l].hist, ref.qkv, lay[l].conv, ref.h, C, 2 * HK, eps, T, 0, false, d_x, lay[l].wa, lay[l].wb,
                          lay[l].dt, lay[l].a, ref.gate, ref.beta, N, HV, s);
        gdn_step_norm_multi(lay[l].state, ref.h, C, ref.gate, ref.beta, ref.z, lay[l].gamma, eps, ref.y, HK, HV, T, nullptr, s, 0, ref.yq);
        native_mmvq(8, lay[l].out, ref.yq, ref.bo, ZV, N, T, s);
    };
    auto run_mk = [&](int l, cudaStream_t s) {
        MkGdnArgs a;
        a.x = d_x; a.xq = mk.xq; a.w_qkv = lay[l].qkv; a.w_z = lay[l].z; a.w_out = lay[l].out; a.qkv = mk.qkv; a.z = mk.z;
        a.hist = lay[l].hist; a.w_conv = lay[l].conv; a.h = mk.h; a.w_alpha = lay[l].wa; a.w_beta = lay[l].wb; a.dt = lay[l].dt;
        a.ssm_a = lay[l].a; a.gate = mk.gate; a.beta = mk.beta; a.state = lay[l].state; a.gamma = lay[l].gamma; a.y = mk.y; a.yq = mk.yq;
        a.bo = mk.bo; a.phases = std::getenv("STRATA_MK_PHASES") ? std::atoi(std::getenv("STRATA_MK_PHASES")) : 5; a.diag = std::getenv("STRATA_MK_DIAG") ? std::atoi(std::getenv("STRATA_MK_DIAG")) : 0; a.grid = std::getenv("STRATA_MK_GRID") ? std::atoi(std::getenv("STRATA_MK_GRID")) : 0; a.n_embd = N; a.conv_channels = C; a.value_dim = ZV; a.h_k = HK; a.h_v = HV; a.eps = eps; a.counters = counters;
        mk_gdn_dense(a, T, s);
    };
    // ---- parity
    const char* skip_env = std::getenv("STRATA_MK_SKIP");
    const std::string skip = skip_env ? skip_env : "";
    std::printf("setup done (T %d, L %d)\n", T, L); std::fflush(stdout);
    size_t bad = 0;
    for (int l = 0; l < L; ++l) {
        if (skip != "ref") { run_ref(l, st); CK(cudaStreamSynchronize(st)); }
        std::printf("layer %d: reference done\n", l); std::fflush(stdout);
        if (skip != "mk") { run_mk(l, st); CK(cudaStreamSynchronize(st)); }
        std::printf("layer %d: megakernel done\n", l); std::fflush(stdout);
        {
            unsigned c[16];
            CK(cudaMemcpy(c, counters, 64, cudaMemcpyDeviceToHost));
            if (c[15] != 0 || l == 0)
                std::printf("counters: done %u %u %u %u %u  next %u %u %u %u %u  block0-reached %u entered %u max-item %u stalled-phase %u (grid %d)\n", c[0], c[1], c[2], c[3], c[4],
                            c[8], c[9], c[10], c[11], c[12], c[5], c[6], c[7], c[15], mk_gdn_grid());
        }
        const size_t d_qkv = ndiff(down(ref.qkv, (size_t) T * C), down(mk.qkv, (size_t) T * C));
        const size_t d_z = ndiff(down(ref.z, (size_t) T * ZV), down(mk.z, (size_t) T * ZV));
        const size_t d_h = ndiff(down(ref.h, (size_t) T * C), down(mk.h, (size_t) T * C));
        const size_t d_g = ndiff(down(ref.gate, (size_t) T * HV), down(mk.gate, (size_t) T * HV));
        const size_t d_b = ndiff(down(ref.beta, (size_t) T * HV), down(mk.beta, (size_t) T * HV));
        const size_t d_y = ndiff(down(ref.y, (size_t) T * ZV), down(mk.y, (size_t) T * ZV));
        const size_t d_yq = ndiff(down((const uint8_t*) ref.yq, (size_t) T * (ZV / 32) * 36), down((const uint8_t*) mk.yq, (size_t) T * (ZV / 32) * 36));
        const size_t d_bo = ndiff(down(ref.bo, (size_t) T * N), down(mk.bo, (size_t) T * N));
        const size_t d_xq = ndiff(down((const uint8_t*) ref.xq, (size_t) T * (N / 32) * 36), down((const uint8_t*) mk.xq, (size_t) T * (N / 32) * 36));
        bad += d_qkv + d_z + d_h + d_g + d_b + d_y + d_yq + d_bo + d_xq;
        if (l < 2 || d_qkv + d_z + d_h + d_g + d_b + d_y + d_yq + d_bo + d_xq)
            std::printf("layer %d: diffs xq %zu qkv %zu z %zu h %zu gate %zu beta %zu y %zu yq %zu bo %zu\n", l, d_xq, d_qkv, d_z, d_h, d_g, d_b, d_y, d_yq, d_bo);
    }
    std::printf("parity: %s (%zu differing values over %d layers, T=%d, grid %d)\n", bad == 0 ? "BITWISE EQUAL" : "DIFFERENT", bad, L, T, mk_gdn_grid());
    if (selftest) return bad == 0 ? 0 : 1;
    // ---- timing: graphs of L layers, replayed
    auto timed = [&](const char* name, auto&& body) {
        cudaGraph_t g; cudaGraphExec_t ex;
        CK(cudaStreamBeginCapture(st, hipStreamCaptureModeRelaxed));
        for (int l = 0; l < L; ++l) body(l, st);
        CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ex, g, 0, 0, 0));
        for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(ex, st));
        CK(cudaStreamSynchronize(st));
        const int reps = 200;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) CK(cudaGraphLaunch(ex, st));
        CK(cudaStreamSynchronize(st));
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps / L;
        std::printf("%-28s %8.2f us a layer (%d layers a graph, %d replays)\n", name, us, L, reps);
        (void) cudaGraphExecDestroy(ex); (void) cudaGraphDestroy(g);
        return us;
    };
    if (std::getenv("STRATA_MK_REFSPLIT")) {   // each reference kernel in its own graph of L layers
        timed("ref: quantize q8_1", [&](int l, cudaStream_t s) { native_quantize_q8_1(d_x, ref.xq, N, T, s); });
        timed("ref: qkv gemv", [&](int l, cudaStream_t s) { native_mmvq(8, lay[l].qkv, ref.xq, ref.qkv, N, C, T, s); });
        timed("ref: z gemv", [&](int l, cudaStream_t s) { native_mmvq(8, lay[l].z, ref.xq, ref.z, N, ZV, T, s); });
        timed("ref: conv+ab", [&](int l, cudaStream_t s) {
            gdn_conv_ab_multi(lay[l].hist, ref.qkv, lay[l].conv, ref.h, C, 2 * HK, eps, T, 0, false, d_x, lay[l].wa, lay[l].wb,
                              lay[l].dt, lay[l].a, ref.gate, ref.beta, N, HV, s); });
        timed("ref: step+norm", [&](int l, cudaStream_t s) {
            gdn_step_norm_multi(lay[l].state, ref.h, C, ref.gate, ref.beta, ref.z, lay[l].gamma, eps, ref.y, HK, HV, T, nullptr, s, 0, ref.yq); });
        timed("ref: out gemv", [&](int l, cudaStream_t s) { native_mmvq(8, lay[l].out, ref.yq, ref.bo, ZV, N, T, s); });
        for (int ph = 1; ph <= 5; ++ph)
            timed((std::string("mk: phases <= ") + std::to_string(ph)).c_str(), [&](int l, cudaStream_t s) {
                MkGdnArgs a;
                a.x = d_x; a.xq = mk.xq; a.w_qkv = lay[l].qkv; a.w_z = lay[l].z; a.w_out = lay[l].out; a.qkv = mk.qkv; a.z = mk.z;
                a.hist = lay[l].hist; a.w_conv = lay[l].conv; a.h = mk.h; a.w_alpha = lay[l].wa; a.w_beta = lay[l].wb; a.dt = lay[l].dt;
                a.ssm_a = lay[l].a; a.gate = mk.gate; a.beta = mk.beta; a.state = lay[l].state; a.gamma = lay[l].gamma; a.y = mk.y; a.yq = mk.yq;
                a.bo = mk.bo; a.phases = ph; a.n_embd = N; a.conv_channels = C; a.value_dim = ZV; a.h_k = HK; a.h_v = HV; a.eps = eps; a.counters = counters;
                mk_gdn_dense(a, T, s); });
    }
    const double a = timed("six kernels (the window's)", run_ref);
    const double b = timed("persistent kernel", run_mk);
    std::printf("saving %.2f us a layer (%.1f%%); x36 GDN layers = %.2f ms a window\n", a - b, 100.0 * (a - b) / a, (a - b) * 36 / 1000.0);
    watch_stop.store(true); if (watch.joinable()) watch.join();
    return 0;
}
