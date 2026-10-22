// src/core/tp.cpp - tensor-parallel decode: the half geometry, the exchange pages, the sliced weights (tp.hpp).
#include "strata/core/tp.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"

#include <cuda_runtime.h>
#if defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>
#ifndef cudaMemcpy2D
#define cudaMemcpy2D hipMemcpy2D   // (the compat header has no 2-D copy)
#endif
#endif

#include <cstdio>
#include <cstring>
#include <exception>
#include <algorithm>
#include <set>

namespace strata::core {

int tp_shexp_mode() {
    static const int mode = [] {
        const char* v = std::getenv("STRATA_TP_SHEXP");
        if (v == nullptr) return 1;
        const std::string s(v);
        return s == "split" ? 0 : s == "card0" || s == "0" ? 2 : 1;
    }();
    return mode;
}

bool tp_geometry(const ModelGeometry& g, int rank, int nranks, ModelGeometry& out, std::string& err) {
    if (nranks < 2 || rank < 0 || rank >= nranks) { err = "tp: rank out of range"; return false; }
    if (g.ssm_k_heads % nranks || g.ssm_v_heads % nranks || g.n_head % nranks || g.n_head_kv % nranks ||
        (g.n_head / g.n_head_kv) * (g.n_head_kv / nranks) != g.n_head / nranks) {
        err = "tp: the heads do not divide by the cards (GDN " + std::to_string(g.ssm_k_heads) + "/" +
              std::to_string(g.ssm_v_heads) + ", QSA " + std::to_string(g.n_head) + "/" + std::to_string(g.n_head_kv) + ")";
        return false;
    }
    if (g.ssm_conv_channels != 2 * g.ssm_k_heads * g.ssm_state_size + g.ssm_v_heads * g.ssm_state_size ||
        g.ssm_value_dim != g.ssm_v_heads * g.ssm_state_size) {
        err = "tp: the GDN channel layout is not q|k|v by head";
        return false;
    }
    out = g;
    out.ssm_k_heads = g.ssm_k_heads / nranks;
    out.ssm_v_heads = g.ssm_v_heads / nranks;
    out.ssm_conv_channels = 2 * out.ssm_k_heads * g.ssm_state_size + out.ssm_v_heads * g.ssm_state_size;
    out.ssm_value_dim = out.ssm_v_heads * g.ssm_state_size;
    out.n_head = g.n_head / nranks;
    out.n_head_kv = g.n_head_kv / nranks;
    return true;
}

// ---- the exchange pages
namespace {
bool pinned_coherent(void** h, size_t bytes) {
#if defined(STRATA_USE_HIP)
    // coherent (fine-grained) host memory: a store from one card is seen by a kernel running on the other without
    // either stream syncing - the design's M0 finding (peer stores into VRAM are not)
    return hipHostMalloc(h, bytes, hipHostMallocMapped | hipHostMallocCoherent | hipHostMallocPortable) == hipSuccess;
#else
    return cudaHostAlloc(h, bytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess;
#endif
}
}  // namespace

TpPages::~TpPages() {
    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < 2; ++p) if (inbox_h[c][p]) cudaFreeHost(inbox_h[c][p]);
        if (flag_h[c]) cudaFreeHost(flag_h[c]);
    }
}

bool TpPages::alloc(int dev0, int dev1, int64_t n, std::string& err) {
    dev[0] = dev0;
    dev[1] = dev1;
    floats = n;
    const size_t bytes = (size_t) n * sizeof(float);
    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < 2; ++p) {
            if (!pinned_coherent((void**) &inbox_h[c][p], bytes)) { err = "tp: the exchange inbox allocation failed"; return false; }
            std::memset(inbox_h[c][p], 0, bytes);
        }
        if (!pinned_coherent((void**) &flag_h[c], 256)) { err = "tp: the exchange flag allocation failed"; return false; }
        std::memset(flag_h[c], 0, 256);
    }
    for (int d = 0; d < 2; ++d) {
        const OnDevice on(dev[d]);
        for (int c = 0; c < 2; ++c) {
            for (int p = 0; p < 2; ++p)
                if (cudaHostGetDevicePointer((void**) &inbox_d[c][p][d], inbox_h[c][p], 0) != cudaSuccess) {
                    err = "tp: the exchange inbox has no device address on CUDA" + std::to_string(dev[d]);
                    return false;
                }
            if (cudaHostGetDevicePointer((void**) &flag_d[c][d], flag_h[c], 0) != cudaSuccess) {
                err = "tp: the exchange flag has no device address on CUDA" + std::to_string(dev[d]);
                return false;
            }
        }
    }
    return true;
}

// ---- the sliced weights
TpWeights::~TpWeights() {
    for (void* p : owned_) cudaFree(p);
}

void* TpWeights::take(uint64_t bytes, std::string& err) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        err = "tp: " + std::to_string(bytes >> 20) + " MiB of VRAM for a sliced weight is not there";
        return nullptr;
    }
    owned_.push_back(p);
    bytes_ += bytes;
    return p;
}

namespace {
struct Range { uint64_t begin, count; };   // rows [begin, begin + count)
// THE GDN HEAD SPLIT.  A v-head H reads k-head H % h_k (gdn_step_norm_multi_kernel: qh = head % h_k), so card r, which
// holds k-heads [r * hk, (r + 1) * hk), must hold the v-heads whose H % h_k falls there: the h_v / h_k ranges
// j * h_k + r * hk + [0, hk), in that (local) order - local v-head h then reads local k-head h % hk, as the kernel does.
std::vector<Range> v_heads(const ModelGeometry& g, int rank, int nranks) {
    const uint64_t HK = (uint64_t) g.ssm_k_heads, hk = HK / nranks;
    std::vector<Range> v;
    for (uint64_t j = 0; j < (uint64_t) g.ssm_v_heads / HK; ++j) v.push_back({j * HK + rank * hk, hk});
    return v;
}
std::vector<Range> scaled(const std::vector<Range>& r, uint64_t per, uint64_t base = 0) {
    std::vector<Range> o;
    for (const Range& x : r) o.push_back({base + x.begin * per, x.count * per});
    return o;
}
// the GDN qkv / conv channel order: q heads, k heads, v heads - card r's slices of each
std::vector<Range> qkv_rows(const ModelGeometry& g, int rank, int nranks, uint64_t per_head) {
    const uint64_t hk = (uint64_t) g.ssm_k_heads / nranks;
    const uint64_t k0 = (uint64_t) g.ssm_k_heads * per_head, v0 = 2 * (uint64_t) g.ssm_k_heads * per_head;
    std::vector<Range> r{{rank * hk * per_head, hk * per_head}, {k0 + rank * hk * per_head, hk * per_head}};
    for (const Range& x : scaled(v_heads(g, rank, nranks), per_head, v0)) r.push_back(x);
    return r;
}
}  // namespace

bool TpWeights::build(const WeightTable& full, const ModelGeometry& g, int rank, int nranks,
                      const std::vector<std::string>& shards, std::string& err) {
    if (!owned_.empty()) { err = "tp: weights already built"; return false; }
    table_ = full;   // the map of references: the small tensors keep their (device) pointers
    const uint64_t S = (uint64_t) g.ssm_state_size, HD = (uint64_t) g.head_dim;
    const uint64_t hk = (uint64_t) g.ssm_k_heads / nranks, hv = (uint64_t) g.ssm_v_heads / nranks;
    const uint64_t nh = (uint64_t) g.n_head / nranks, nkv = (uint64_t) g.n_head_kv / nranks;
    // ---- the small tensors (pack arena, f32 / bf16 bits): row slices by head
    auto slice_rows = [&](const std::string& name, const std::vector<Range>& rows, int64_t want_ne1) -> bool {
        auto it = table_.table_.find(name);
        if (it == table_.table_.end()) { err = "tp: " + name + " is not in the table"; return false; }
        WeightRef& r = it->second;
        if (r.data == nullptr) { err = "tp: " + name + " has no bytes on this card (the arena must hold every layer)"; return false; }
        if (r.elements <= 0 || r.ne1 <= 0 || r.bytes % (uint64_t) r.elements != 0) { err = "tp: " + name + ": odd element size"; return false; }
        const uint64_t eb = r.bytes / (uint64_t) r.elements, row = (uint64_t) r.ne0 * eb;
        uint64_t total = 0;
        for (const Range& rg : rows) total += rg.count;
        if ((int64_t) total != want_ne1) { err = "tp: " + name + ": the slice does not add up"; return false; }
        void* dst = take(total * row, err);
        if (dst == nullptr) return false;
        uint64_t off = 0;
        for (const Range& rg : rows) {
            if (rg.begin + rg.count > (uint64_t) r.ne1) { err = "tp: " + name + ": slice past the end"; return false; }
            if (cudaMemcpy((uint8_t*) dst + off, (const uint8_t*) r.data + rg.begin * row, rg.count * row,
                           cudaMemcpyDeviceToDevice) != cudaSuccess) {
                err = "tp: " + name + ": the slice copy failed";
                return false;
            }
            off += rg.count * row;
        }
        r.data = dst;
        r.ne1 = want_ne1;
        r.elements = (int64_t) (total * (uint64_t) r.ne0);
        r.bytes = total * row;
        return true;
    };
    // a vector of per-head scalars: ne0 = heads (one row): slice along ne0
    auto slice_vec = [&](const std::string& name, const std::vector<Range>& rs) -> bool {
        auto it = table_.table_.find(name);
        if (it == table_.table_.end()) { err = "tp: " + name + " is not in the table"; return false; }
        WeightRef& r = it->second;
        if (r.data == nullptr) { err = "tp: " + name + " has no bytes on this card"; return false; }
        uint64_t count = 0;
        for (const Range& x : rs) count += x.count;
        if (r.elements <= 0 || r.bytes % (uint64_t) r.elements != 0) { err = "tp: " + name + ": odd vector"; return false; }
        const uint64_t eb = r.bytes / (uint64_t) r.elements;
        void* dst = take(count * eb, err);
        if (dst == nullptr) return false;
        uint64_t off = 0;
        for (const Range& x : rs) {
            if (x.begin + x.count > (uint64_t) r.elements) { err = "tp: " + name + ": slice past the end"; return false; }
            if (cudaMemcpy((uint8_t*) dst + off, (const uint8_t*) r.data + x.begin * eb, x.count * eb, cudaMemcpyDeviceToDevice) != cudaSuccess) {
                err = "tp: " + name + ": the slice copy failed";
                return false;
            }
            off += x.count * eb;
        }
        r.data = dst;
        r.ne0 = (int64_t) count;
        r.elements = (int64_t) count;
        r.bytes = count * eb;
        return true;
    };
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const std::string pfx = "blk." + std::to_string(l) + ".";
        if (!is_qsa_layer(g, l)) {
            // alpha / beta: bf16 [h_v][n_embd]; dt bias / ssm_a: [h_v]; conv1d: [channels][d_conv]
            const std::vector<Range> heads = v_heads(g, rank, nranks);
            if (!slice_rows(pfx + "ssm_alpha.weight", heads, (int64_t) hv) ||
                !slice_rows(pfx + "ssm_beta.weight", heads, (int64_t) hv) ||
                !slice_vec(pfx + "ssm_dt.bias", heads) || !slice_vec(pfx + "ssm_a", heads) ||
                !slice_rows(pfx + "ssm_conv1d.weight", qkv_rows(g, rank, nranks, S), (int64_t) (2 * hk * S + hv * S)))
                return false;
        }
        // (q_norm / k_norm / ssm_norm are per head-dim: shared; the indexer, router, hyper-connections: replicated)
    }
    // ---- the dense projections from the GGUF shards at half width
    // kind 0: GDN qkv rows; 1: rows [rank * n / nranks, ...); 2: an n_in slice [rank * n / nranks, ...); 3: the GDN
    // v-head rows (attn_gate); 4: the GDN v-head columns of n_in (ssm_out)
    struct Want { const char* suffix; int kind; };
    // (the shared expert too: gate / up by ff unit, down over the ff units - each card's partial is added in the
    // layer's second exchange)
    static const Want wants[] = {{".attn_qkv.weight", 0}, {".attn_gate.weight", 3}, {".ssm_out.weight", 4},
                                 {".attn_q.weight", 1}, {".attn_k.weight", 1}, {".attn_v.weight", 1}, {".attn_output.weight", 2},
                                 {".ffn_gate_shexp.weight", 1}, {".ffn_up_shexp.weight", 1}, {".ffn_down_shexp.weight", 2}};
    std::set<std::string> done;
    uint64_t max_in = 0;
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& t : gguf.tensors()) {
                if (t.name.rfind("blk.", 0) != 0 || t.shape.size() != 2) continue;
                int kind = -1;
                for (const Want& w : wants) if (t.name.ends_with(w.suffix)) kind = w.kind;
                if (kind < 0) continue;
                const bool shexp = t.name.find("_shexp.") != std::string::npos;
                if (shexp && !tp_shexp_split()) {
                    // whole on one card: that card loads the layers its stage's table has no bytes for (kind 5)
                    auto it0 = table_.table_.find(t.name);
                    const bool mine = (tp_shexp_mode() == 1) == (rank == 1);
                    if (!mine || (it0 != table_.table_.end() && it0->second.native_data != nullptr)) continue;
                    kind = 5;
                }
                if (!strata::kernels::native_mmvq_supported(t.type)) { err = "tp: " + t.name + ": format not native"; return false; }
                if (!done.insert(t.name).second) { err = "tp: duplicate tensor " + t.name; return false; }
                auto it = table_.table_.find(t.name);
                if (it == table_.table_.end()) { err = "tp: " + t.name + " is not in the table"; return false; }
                WeightRef& r = it->second;
                const uint64_t n_in = t.shape[0], n_out = t.shape[1];
                if (n_in != (uint64_t) r.ne0 || n_out != (uint64_t) r.ne1) { err = "tp: " + t.name + ": shape differs from the pack"; return false; }
                int be = 0, bb = 0;
                if (!strata::block_geometry(t.type, be, bb) || n_in % (uint64_t) be) { err = "tp: " + t.name + ": block geometry"; return false; }
                const uint64_t row_bytes = n_in / (uint64_t) be * (uint64_t) bb;
                const uint8_t* src = gguf.tensor_data(t);
                uint64_t new_in = n_in, new_out = n_out;
                std::vector<Range> rows, cols;   // cols: n_in element ranges (kind 2 / 4)
                if (kind == 0) {
                    rows = qkv_rows(g, rank, nranks, S);
                    new_out = 2 * hk * S + hv * S;
                } else if (kind == 1) {
                    new_out = n_out / (uint64_t) nranks;
                    if (new_out * (uint64_t) nranks != n_out) { err = "tp: " + t.name + ": rows do not divide"; return false; }
                    rows = {{rank * new_out, new_out}};
                } else if (kind == 3) {
                    rows = scaled(v_heads(g, rank, nranks), S);
                    new_out = hv * S;
                } else if (kind == 5) {   // the whole matrix
                    rows = {{0, n_out}};
                } else if (kind == 2) {
                    new_in = n_in / (uint64_t) nranks;
                    if (new_in * (uint64_t) nranks != n_in) { err = "tp: " + t.name + ": the reduction does not divide"; return false; }
                    cols = {{rank * new_in, new_in}};
                } else {
                    cols = scaled(v_heads(g, rank, nranks), S);
                    new_in = hv * S;
                }
                for (const Range& c : cols)
                    if (c.begin % (uint64_t) be || c.count % (uint64_t) be) { err = "tp: " + t.name + ": a reduction slice cuts a block"; return false; }
                const uint64_t bytes = strata::kernels::native_mmvq_weight_bytes((int) t.type, (int) new_in, (int) new_out);
                void* dst = take(bytes, err);
                if (dst == nullptr) return false;
                cudaError_t st = cudaSuccess;
                if (!cols.empty()) {   // every row's blocks of the column ranges: strided copies from the mapped file
                    const uint64_t new_row = new_in / (uint64_t) be * (uint64_t) bb;
                    uint64_t doff = 0;
                    for (const Range& c : cols) {
                        const uint64_t w = c.count / (uint64_t) be * (uint64_t) bb, so = c.begin / (uint64_t) be * (uint64_t) bb;
                        st = cudaMemcpy2D((uint8_t*) dst + doff, new_row, src + so, row_bytes, w, n_out, cudaMemcpyHostToDevice);
                        if (st != cudaSuccess) break;
                        doff += w;
                    }
                } else {
                    uint64_t off = 0;
                    for (const Range& rg : rows) {
                        if (rg.begin + rg.count > n_out) { err = "tp: " + t.name + ": slice past the end"; return false; }
                        st = cudaMemcpy((uint8_t*) dst + off, src + rg.begin * row_bytes, rg.count * row_bytes, cudaMemcpyHostToDevice);
                        if (st != cudaSuccess) break;
                        off += rg.count * row_bytes;
                    }
                    if (off != bytes) { err = "tp: " + t.name + ": the slice bytes do not add up"; return false; }
                }
                if (st != cudaSuccess) { err = "tp: " + t.name + ": upload: " + cudaGetErrorString(st); return false; }
                r.native_data = dst;
                r.native_type = (int) t.type;
                r.ne0 = (int64_t) new_in;
                r.ne1 = (int64_t) new_out;
                r.elements = (int64_t) (new_in * new_out);
                r.data = nullptr;   // the canonical form (if any) is the full matrix: never read it by mistake
                r.bytes = 0;
                if (new_in > max_in) max_in = new_in;
            }
        }
    } catch (const std::exception& e) {
        err = std::string("tp: ") + e.what();
        return false;
    }
    if (done.size() < (size_t) (g.n_gdn_layers() * 3 + g.n_qsa_layers() * 4)) {
        err = "tp: the shards hold " + std::to_string(done.size()) + " dense projections, not every layer's";
        return false;
    }
    // the shared q8_1 scratch the table's references name (the window quantizes into its own buffers; the layer path's)
    void* scratch = take(strata::kernels::native_q8_1_bytes((int) max_in, 8), err);
    if (scratch == nullptr) return false;
    for (auto& [name, r] : table_.table_)
        if (r.native_data != nullptr) r.native_q8_1 = scratch;
    (void) nh; (void) nkv; (void) HD;
    return true;
}

// ---- the state hand-over (tp.hpp)
namespace {
// A strided copy between two devices through a pinned staging buffer (no GPU-to-GPU peer access needed): `height`
// rows of `width` bytes, pitches in bytes.
struct Stager {
    uint8_t* h = nullptr;
    size_t cap = 0;
    ~Stager() { if (h) cudaFreeHost(h); }
    bool ensure(size_t n) {
        if (n <= cap) return true;
        if (h) cudaFreeHost(h);
        h = nullptr;
        cap = 0;
        if (cudaHostAlloc((void**) &h, n, cudaHostAllocDefault) != cudaSuccess) { (void) cudaGetLastError(); return false; }
        cap = n;
        return true;
    }
};
bool copy2d(Stager& st, int ddev, void* dst, size_t dpitch, int sdev, const void* src, size_t spitch, size_t width,
            size_t height, std::string& err) {
    if (height == 0 || width == 0) return true;
    if (!st.ensure(width * height)) { err = "tp: the hand-over staging buffer"; return false; }
    {
        const OnDevice on(sdev);
        if (cudaMemcpy2D(st.h, width, src, spitch, width, height, cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "tp: the hand-over read failed";
            return false;
        }
    }
    {
        const OnDevice on(ddev);
        if (cudaMemcpy2D(dst, dpitch, st.h, width, width, height, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "tp: the hand-over write failed";
            return false;
        }
    }
    return true;
}
}  // namespace

bool tp_state_move(const ModelGeometry& g, const ModelGeometry& tg, SessionState* const* stage_ss, const int* stage_dev,
                   int n_stages, const int64_t* stage_lb, SessionState* const* tp_ss, const int* tp_dev, int64_t cells,
                   bool to_tp, std::string& err) {
    using strata::kernels::QsaShapes;
    Stager stg;
    const int R = 2;
    const uint64_t S = (uint64_t) g.ssm_state_size;
    const uint64_t HV = (uint64_t) g.ssm_v_heads, hv = (uint64_t) tg.ssm_v_heads;
    const uint64_t gfull = S * HV * S + (uint64_t) g.ssm_conv_channels * (uint64_t) (g.ssm_d_conv - 1);
    const uint64_t ghalf = S * hv * S + (uint64_t) tg.ssm_conv_channels * (uint64_t) (tg.ssm_d_conv - 1);
    const QsaShapes sf = [&] { QsaShapes s = strata::kernels::qsa_real_shapes(); s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads; s.idx_dim = g.idx_key_dim; return s; }();
    auto stage_of = [&](int64_t l) { int st = 0; while (st + 1 < n_stages && l >= stage_lb[st + 1]) ++st; return st; };
    int64_t gi = 0, qi = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const int st = stage_of(l);
        SessionState& S0 = *stage_ss[st];
        const int d0 = stage_dev[st];
        if (!is_qsa_layer(g, l)) {
            float* full = S0.gdn_state + (size_t) (gi - S0.gdn_ord0) * gfull;
            for (int r = 0; r < R; ++r) {
                float* half = tp_ss[r]->gdn_state + (size_t) gi * ghalf;
                // the recurrence [S][HV][S]: this card's v-head ranges (v_heads), in local order
                uint64_t lh = 0;
                for (const Range& vr : v_heads(g, r, R)) {
                    float* f_st = full + vr.begin * S;
                    float* h_st = half + lh * S;
                    if (to_tp) { if (!copy2d(stg, tp_dev[r], h_st, hv * S * 4, d0, f_st, HV * S * 4, vr.count * S * 4, S, err)) return false; }
                    else if (!copy2d(stg, d0, f_st, HV * S * 4, tp_dev[r], h_st, hv * S * 4, vr.count * S * 4, S, err)) return false;
                    lh += vr.count;
                }
                // the conv history [C][d_conv - 1]: the q, k and v channel ranges of this card's heads
                float* fconv = full + S * HV * S;
                float* hconv = half + S * hv * S;
                const uint64_t dc = (uint64_t) (g.ssm_d_conv - 1);
                uint64_t ho = 0;
                for (const Range& cr : qkv_rows(g, r, R, S)) {
                    const size_t bytes = cr.count * dc * 4;
                    if (to_tp) { if (!copy2d(stg, tp_dev[r], hconv + ho * dc, bytes, d0, fconv + cr.begin * dc, bytes, bytes, 1, err)) return false; }
                    else if (!copy2d(stg, d0, fconv + cr.begin * dc, bytes, tp_dev[r], hconv + ho * dc, bytes, bytes, 1, err)) return false;
                    ho += cr.count;
                }
            }
            ++gi;
        } else {
            const QsaState& F = S0.qsa_states[qi];
            const uint64_t NKV = (uint64_t) g.n_head_kv, nkv = (uint64_t) tg.n_head_kv, PS = (uint64_t) sf.page_size, HD = (uint64_t) g.head_dim;
            const uint64_t pages_used = std::min<uint64_t>((uint64_t) F.n_pages, (uint64_t) ((cells + PS - 1) / PS) + 1);
            const uint64_t scale_row = HD / (uint64_t) strata::kernels::KV_Q8_GROUP * 2, q4_row = strata::kernels::kv_q4_bytes_per_head((int) HD);
            for (int r = 0; r < R; ++r) {
                const QsaState& H = tp_ss[r]->qsa_states[qi];
                if (H.n_pages != F.n_pages || H.n_slots != F.n_slots || H.kv_mode != F.kv_mode || H.kv_int8 != F.kv_int8 ||
                    H.kv_q4 != F.kv_q4 || H.kv_hybrid != F.kv_hybrid) {
                    err = "tp: the TP session's K/V differs from the stage's (layer " + std::to_string(l) + ")";
                    return false;
                }
                // [page][kv_head][PS][...]: card r's kv-heads of every page
                auto pool = [&](const void* fp, const void* hp, uint64_t per_head, uint64_t pages) -> bool {
                    if (fp == nullptr || hp == nullptr || pages == 0) return true;
                    const uint64_t fpitch = NKV * PS * per_head, hpitch = nkv * PS * per_head, w = nkv * PS * per_head;
                    const uint8_t* f = (const uint8_t*) fp + r * w;
                    if (to_tp) return copy2d(stg, tp_dev[r], (void*) hp, hpitch, d0, f, fpitch, w, pages, err);
                    return copy2d(stg, d0, (void*) f, fpitch, tp_dev[r], hp, hpitch, w, pages, err);
                };
                // the VRAM pools: every slot (streamed) or the used pages (resident)
                const uint64_t vp = F.kv_mode == 1 ? (uint64_t) F.n_slots : pages_used;
                if (!pool(F.k_pool, H.k_pool, HD * 2, vp) || !pool(F.v_pool, H.v_pool, HD * 2, vp) ||
                    !pool(F.k_q, H.k_q, HD, vp) || !pool(F.v_q, H.v_q, HD, vp) || !pool(F.k_scale, H.k_scale, scale_row, vp) ||
                    !pool(F.v_scale, H.v_scale, scale_row, vp) || !pool(F.k_q4, H.k_q4, q4_row, vp) || !pool(F.v_q4, H.v_q4, q4_row, vp))
                    return false;
                if (F.kv_mode == 1) {   // streamed: the host copy (the used pages), the page table, the slot map
                    if (!pool(F.host.k_pool, H.host.k_pool, HD * 2, pages_used) || !pool(F.host.v_pool, H.host.v_pool, HD * 2, pages_used) ||
                        !pool(F.host.k_q, H.host.k_q, HD, pages_used) || !pool(F.host.v_q, H.host.v_q, HD, pages_used) ||
                        !pool(F.host.k_scale, H.host.k_scale, scale_row, pages_used) || !pool(F.host.v_scale, H.host.v_scale, scale_row, pages_used) ||
                        !pool(F.host.k_q4, H.host.k_q4, q4_row, pages_used) || !pool(F.host.v_q4, H.host.v_q4, q4_row, pages_used))
                        return false;
                    auto words = [&](const int32_t* f, const int32_t* h, uint64_t n) -> bool {
                        if (f == nullptr || h == nullptr || n == 0) return true;
                        if (to_tp) return copy2d(stg, tp_dev[r], (void*) h, n * 4, d0, f, n * 4, n * 4, 1, err);
                        return copy2d(stg, d0, (void*) f, n * 4, tp_dev[r], h, n * 4, n * 4, 1, err);
                    };
                    if (!words(F.page_table, H.page_table, (uint64_t) F.n_pages) || !words(F.map.slot_block, H.map.slot_block, (uint64_t) F.n_slots) ||
                        !words(F.map.slot_stamp, H.map.slot_stamp, (uint64_t) F.n_slots) || !words(F.map.slot_ref, H.map.slot_ref, (uint64_t) F.n_slots) ||
                        !words(F.map.miss_block, H.map.miss_block, (uint64_t) F.n_slots) || !words(F.map.miss_slot, H.map.miss_slot, (uint64_t) F.n_slots) ||
                        !words(F.map.ctl, H.map.ctl, (uint64_t) strata::kernels::kKvCtlInts))
                        return false;
                }
                // the indexer: replicated (both cards select identically)
                const uint64_t ID = (uint64_t) g.idx_key_dim, TS = (uint64_t) (sf.idx_block - 1) * ID;
                const uint64_t prow = std::min<uint64_t>((uint64_t) F.idx_pooled_rows, (uint64_t) (cells / sf.idx_block) + 2);
                auto flat = [&](const void* f, const void* h, uint64_t bytes) -> bool {
                    if (f == nullptr || h == nullptr || bytes == 0) return true;
                    if (to_tp) return copy2d(stg, tp_dev[r], (void*) h, bytes, d0, f, bytes, bytes, 1, err);
                    return copy2d(stg, d0, (void*) f, bytes, tp_dev[r], h, bytes, bytes, 1, err);
                };
                if (to_tp || r == 0) {   // back: rank 0's copy (both are the same)
                    if (!flat(F.idx_tail, H.idx_tail, TS * 4) || !flat(F.idx_dead, H.idx_dead, ID * 4) ||
                        !flat(F.idx_pooled, H.idx_pooled, prow * ID * 4) || !flat(F.idx_block_pos, H.idx_block_pos, 4))
                        return false;
                }
            }
            ++qi;
        }
    }
    // the PLE history and token window (stage 0's session holds layer 1), replicated
    {
        SessionState& S0 = *stage_ss[0];
        const uint64_t hb = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM * 4;
        for (int r = 0; r < R; ++r) {
            if (to_tp) {
                if (!copy2d(stg, tp_dev[r], tp_ss[r]->ple_hist, hb, stage_dev[0], S0.ple_hist, hb, hb, 1, err)) return false;
                tp_ss[r]->ple_prev[0] = S0.ple_prev[0];
                tp_ss[r]->ple_prev[1] = S0.ple_prev[1];
                tp_ss[r]->ple_token = S0.ple_token;
            } else if (r == 0) {
                if (!copy2d(stg, stage_dev[0], S0.ple_hist, hb, tp_dev[0], tp_ss[0]->ple_hist, hb, hb, 1, err)) return false;
                S0.ple_prev[0] = tp_ss[0]->ple_prev[0];
                S0.ple_prev[1] = tp_ss[0]->ple_prev[1];
                S0.ple_token = tp_ss[0]->ple_token;
            }
        }
    }
    return true;
}

}  // namespace strata::core
