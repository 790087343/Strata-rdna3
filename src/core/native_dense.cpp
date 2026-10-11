#include "strata/core/native_dense.hpp"
#include "strata/kernels/fused_gr.hpp"
#include <cstdlib>
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/memory.hpp"
#include "ggml.h"   // STRATA_GR_RDNA_Q8: other hc formats (IQ3_S / IQ4_NL ...) dequantized for their int8 copies

#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <set>
#include <cmath>
#include <cstring>
#include <vector>

namespace strata::core {
namespace {
// `load` makes one cudaMalloc per projection matrix, and the driver backs each call with 2 MiB granules
// (measured on the RTX 5060: 300 buffers of 6.73 MiB asked 2019 MiB and cost 2400).  So a stage's real
// footprint is the granule-rounded sum, not the payload sum.  This rule - a granule for every matrix of a
// MiB or more, the payload for the rest - prices the iq3_s pack's 300 matrices at 2255 MiB where they
// really take 2238 (+0.8%).  It over-prices, which is the safe side.  (Pricing every matrix at a granule is
// far worse, 2378 MiB: the driver packs the sub-MiB matrices into the slack of the rounded ones.)
constexpr uint64_t kAllocGranule = 2ull << 20;
constexpr uint64_t kSmallAlloc = 1ull << 20;
uint64_t alloc_bytes(uint64_t bytes) {
    return bytes < kSmallAlloc ? bytes : (bytes + kAllocGranule - 1) / kAllocGranule * kAllocGranule;
}
int g_layer_lb = -1, g_layer_le = -1;   // set_layer_range; -1: every layer
bool in_range(const std::string& name) {
    if (g_layer_lb < 0 || name.rfind("blk.", 0) != 0) return true;
    const int l = std::atoi(name.c_str() + 4);
    return l >= g_layer_lb && l < g_layer_le;
}
// S23 experiment (STRATA_HC_Q8=1): the hyper-connection projections' Q8_0 bytes for the verify read
bool hc_q8_requested() {
    static const bool on = [] { const char* v = std::getenv("STRATA_HC_Q8"); return v != nullptr && v[0] == '1'; }();
    return on;
}
bool hc_q8_name(const std::string& name) {
    if (name == "output_hc_down.weight" || name == "output_hc_up.weight") return true;   // S25: the final mixer too
    if (name.rfind("blk.", 0) != 0) return false;
    static const char* suffixes[] = {".hc_attn_down.weight", ".hc_attn_up.weight", ".hc_attn_inject.weight",
                                     ".hc_ffn_down.weight", ".hc_ffn_up.weight", ".hc_ffn_inject.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// S25: the GGUF keeps the inject rows as F32 - quantized here to Q8_0 (ggml's reference rounding) so the Q8_0 read
// covers every hc projection
uint16_t half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {                                   // subnormal or zero
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - e);
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) (sign | h);
}
float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, x;
    if (e == 0) {
        if (m == 0) x = sign;
        else { int s = -1; do { m <<= 1; ++s; } while (!(m & 0x400u)); x = sign | ((uint32_t) (127 - 15 - s) << 23) | ((m & 0x3ffu) << 13); }
    } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
    else x = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}
std::vector<uint8_t> q8_0_of(const float* x, uint64_t n) {
    std::vector<uint8_t> out((size_t) (n / 32 * 34));
    for (uint64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = (std::max)(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t* o = out.data() + b * 34;
        const uint16_t hb = half_bits(d);
        std::memcpy(o, &hb, 2);
        for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(x[b * 32 + j] * id);
    }
    return out;
}
bool eligible(const strata::TensorInfo& tensor, bool include_ple_key) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
struct DeviceFree { void operator()(void* p) const { if (p) cudaFree(p); } };
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
    DevicePtr packed;   // STRATA_Q8_PACKED=1 copy, or null
};
}

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

void NativeDense::set_layer_range(int lb, int le) { g_layer_lb = lb; g_layer_le = le; }
// The byte walk `load` does for the layers [lb, le), without the allocations: the same filters in the same order,
// reading the GGUF headers and the canonical table only, so it needs no device.  A later stage's projections are not
// on the card when the split search prices that stage, and `load` runs one cudaMalloc per matrix (#1238).  The two
// have to stay in step.  A repeated name is skipped rather than refused (`load` refuses it).
bool NativeDense::weight_bytes_for(const std::vector<std::string>& shards, WeightTable& table, bool include_ple_key,
                                   int64_t lb, int64_t le, uint64_t& out, std::string& err) {
    out = 0;
    try {
        std::set<std::string> seen;
        uint64_t total = 0;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key)) continue;
                if (tensor.name.rfind("blk.", 0) == 0) {
                    const long l = std::strtol(tensor.name.c_str() + 4, nullptr, 10);
                    if (l < lb || l >= le) continue;
                }
                if (!seen.insert(tensor.name).second) continue;
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name;
                    return false;
                }
                const auto& ref = found->second;
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name;
                    return false;
                }
                total += alloc_bytes(strata::kernels::native_mmvq_weight_bytes(tensor.type, (int) ref.ne0, (int) ref.ne1));
            }
        }
        out = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip,
                                           std::string& err) {
    const std::string key = "blk.1.ple_key.weight";
    if (!skip.count(key)) return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, key, code_bits, err)) return false;
    if (code_bits == 0) skip.erase(key);
    return true;
}

NativeDense::~NativeDense() {
    for (const void* p : packed_keys_) strata::kernels::native_q8_0_packed_unregister(p);
    if (scratch_) cudaFree(scratch_);
    for (void* p : weights_) cudaFree(p);
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key, int64_t layer_lo, int64_t layer_hi) {
    auto outside = [&](const std::string& name) {   // a blk.<l>. tensor of another stage's layers
        if (layer_hi < 0 || name.rfind("blk.", 0) != 0) return false;
        const long l = std::strtol(name.c_str() + 4, nullptr, 10);
        return l < layer_lo || l >= layer_hi;
    };
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0, hc_q8_bytes = 0, allocated = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            const auto* arch = gguf.get("general.architecture");
            if (gguf.get("general.architecture") && (!number || number->u == 0)) {
                // splitter may copy general.architecture into every shard (Huihui abliterated re-split)
                err = strata::check_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            // Some publishers repeat only the architecture name on data shards.
            // Their split identity is checked above; any repeated model keys must
            // still agree with the fully validated first shard.
            if (have_architecture && number && number->u != 0) {
                strata::GgufFile first(shards.front());
                const auto* first_arch = first.get("general.architecture");
                if (arch && (!first_arch || arch->s != first_arch->s)) {
                    err = "native dense: additional shard architecture mismatch";
                    return false;
                }
                for (const char* key : {"qwen4exp.block_count", "qwen4exp.embedding_length",
                                       "qwen4exp.expert_count", "qwen4exp.expert_used_count",
                                       "qwen4exp.attention.head_count", "qwen4exp.attention.head_count_kv"}) {
                    const auto* value = gguf.get(key);
                    const auto* expected = first.get(key);
                    if (value && (!expected || value->u != expected->u)) {
                        err = std::string("native dense: additional shard metadata mismatch: ") + key;
                        return false;
                    }
                }
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
                // the uploads below read these: ask for them now so the reads overlap
                if (eligible(tensor, include_ple_key) && !outside(tensor.name) &&
                    strata::kernels::native_mmvq_supported(tensor.type))
                    strata::platform::advise_willneed(gguf.tensor_data(tensor), bytes);
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key) || outside(tensor.name)) continue;
                if (!in_range(tensor.name) && tensor.name.find("ple") == std::string::npos) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    tensor.type, (int) ref.ne0, (int) ref.ne1);
                void* allocation = nullptr;
                auto status = cudaMalloc(&allocation, bytes);
                DevicePtr data(allocation);
                if (status == cudaSuccess)
                    status = cudaMemcpy(data.get(), gguf.tensor_data(tensor), bytes, cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status); return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                allocated += alloc_bytes(bytes);
                // STRATA_Q8_PACKED=1: a second, packed copy of an eligible Q8_0 matrix for the decode MMVQ (the GGUF
                // copy stays: the prompt path's GEMMs read it).
                DevicePtr packed;
                if (tensor.type == 8 && tensor.name != "blk.1.ple_key.weight" &&   // the PLE has its own kernel
                    strata::kernels::native_q8_0_packed_enabled() &&
                    strata::kernels::native_q8_0_packed_eligible((int) ref.ne0, (int) ref.ne1)) {
                    std::vector<uint8_t> host(bytes);
                    strata::kernels::native_q8_0_pack_host(gguf.tensor_data(tensor), host.data(), (int) ref.ne0,
                                                           (int) ref.ne1);
                    void* packed_allocation = nullptr;
                    auto packed_status = cudaMalloc(&packed_allocation, bytes);
                    packed.reset(packed_allocation);
                    if (packed_status == cudaSuccess)
                        packed_status = cudaMemcpy(packed.get(), host.data(), bytes, cudaMemcpyHostToDevice);
                    if (packed_status != cudaSuccess) {
                        err = "native dense packed upload " + tensor.name + ": " + cudaGetErrorString(packed_status);
                        return false;
                    }
                }
                pending.push_back(Pending{&ref, (int) tensor.type, bytes, std::move(data), std::move(packed)});
            }
            if (hc_q8_requested())
                for (const auto& tensor : gguf.tensors()) {
                    // S25: the F32 inject rows hold BF16-exact values: the read takes the pack's BF16 rows unless
                    // STRATA_HC_Q8_INJECT=1 asks for a Q8_0 copy
                    static const bool q8_inject = [] { const char* v = std::getenv("STRATA_HC_Q8_INJECT"); return v && v[0] == '1'; }();
                    const bool f32_inject = q8_inject && tensor.type == 0 && tensor.name.ends_with("_inject.weight");
                    if ((tensor.type != 8 && !f32_inject) || !hc_q8_name(tensor.name) || tensor.shape.size() != 2) continue;
                    auto found = table.table_.find(tensor.name);
                    if (found == table.table_.end() || found->second.hc_q8 != nullptr) continue;
                    auto& ref = found->second;
                    if (tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1 || ref.ne0 % 32 != 0) {
                        err = "native dense (STRATA_HC_Q8): incompatible matrix " + tensor.name; return false;
                    }
                    const uint64_t bytes = (uint64_t) ref.ne0 / 32 * 34 * (uint64_t) ref.ne1;
                    void* p = nullptr;
                    std::vector<uint8_t> q8;
                    if (f32_inject) q8 = q8_0_of((const float*) gguf.tensor_data(tensor), (uint64_t) ref.ne0 * (uint64_t) ref.ne1);
                    if (cudaMalloc(&p, bytes) != cudaSuccess ||
                        cudaMemcpy(p, f32_inject ? (const void*) q8.data() : gguf.tensor_data(tensor), bytes,
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
                        err = "native dense (STRATA_HC_Q8) upload " + tensor.name; return false;
                    }
                    ref.hc_q8 = p;
                    weights_.push_back(p);
                    hc_q8_bytes += bytes;
                }
            {   // STRATA_GR_RDNA_Q8=1: int8 + fp16-scale copies of the hyper-connection down/up projections (rdna_gr.cu)
                static const bool r8 = [] { const char* v = std::getenv("STRATA_GR_RDNA_Q8"); return v && v[0] == '1'; }();
                if (r8)
                    for (const auto& tensor : gguf.tensors()) {
                        // Q8_0 and BF16 as stored; any other format ggml can dequantize (the Huihui packs' IQ3_S / IQ4_NL
                        // hc tensors) through to_float, then the BF16 path's per-32 quantization
                        const auto* tt = (tensor.type != 8 && tensor.type != 30) ? ggml_get_type_traits((ggml_type) tensor.type) : nullptr;
                        const bool other = tt != nullptr && tt->to_float != nullptr && tt->blck_size > 0;
                        if ((tensor.type != 8 && tensor.type != 30 && !other) || !hc_q8_name(tensor.name) || tensor.shape.size() != 2 ||
                            tensor.name.ends_with("_inject.weight"))
                            continue;
                        auto found = table.table_.find(tensor.name);
                        if (found == table.table_.end() || found->second.hc_r8 != nullptr) continue;
                        auto& ref = found->second;
                        const uint64_t n0 = tensor.shape[0], n1 = tensor.shape[1], n = n0 * n1;
                        if (n0 != (uint64_t) ref.ne0 || n1 != (uint64_t) ref.ne1 || n0 % 32 != 0) {
                            err = "native dense (STRATA_GR_RDNA_Q8): incompatible matrix " + tensor.name; return false;
                        }
                        const uint64_t soff = (n + 255) / 256 * 256, bytes = soff + n / 32 * 2;
                        std::vector<uint8_t> host((size_t) bytes, 0);
                        int8_t* q = (int8_t*) host.data();
                        uint16_t* sc = (uint16_t*) (host.data() + soff);
                        const uint8_t* src = (const uint8_t*) gguf.tensor_data(tensor);
                        std::vector<float> deq;
                        if (other) {
                            if (n0 % (uint64_t) tt->blck_size != 0) { err = "native dense (STRATA_GR_RDNA_Q8): block size " + tensor.name; return false; }
                            deq.resize((size_t) n);
                            const size_t row_bytes = (size_t) (n0 / (uint64_t) tt->blck_size) * tt->type_size;
                            for (uint64_t r = 0; r < n1; ++r) tt->to_float(src + r * row_bytes, deq.data() + r * n0, (int64_t) n0);
                        }
                        for (uint64_t b = 0; b < n / 32; ++b) {
                            if (tensor.type == 8) {                       // Q8_0: fp16 d + 32 int8, as stored
                                std::memcpy(&sc[b], src + b * 34, 2);
                                std::memcpy(q + b * 32, src + b * 34 + 2, 32);
                            } else {                                      // BF16: quantized per 32 on the stored fp16 scale
                                float x[32], amax = 0.0f;
                                for (int j = 0; j < 32; ++j) {
                                    if (other) x[j] = deq[(size_t) (b * 32 + j)];
                                    else {
                                        uint16_t h; std::memcpy(&h, src + (b * 32 + j) * 2, 2);
                                        const uint32_t u = (uint32_t) h << 16; std::memcpy(&x[j], &u, 4);
                                    }
                                    amax = (std::max)(amax, std::fabs(x[j]));
                                }
                                const uint16_t hb = half_bits(amax / 127.0f);
                                sc[b] = hb;
                                const float d = half_to_float(hb), id = d != 0.0f ? 1.0f / d : 0.0f;
                                for (int j = 0; j < 32; ++j) {
                                    const long v = std::lround(x[j] * id);
                                    q[b * 32 + j] = (int8_t) (v > 127 ? 127 : v < -127 ? -127 : v);
                                }
                            }
                        }
                        void* p = nullptr;
                        if (cudaMalloc(&p, bytes) != cudaSuccess || cudaMemcpy(p, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                            err = "native dense (STRATA_GR_RDNA_Q8) upload " + tensor.name; return false;
                        }
                        ref.hc_r8 = p;
                        weights_.push_back(p);
                        hc_q8_bytes += bytes;
                        if (!strata::kernels::rdna_r8_scratch_init((long long) n)) {
                            err = "native dense (STRATA_GR_RDNA_Q8): scratch allocation"; return false;
                        }
                    }
            }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        const auto status = cudaMalloc(&allocation, strata::kernels::native_q8_1_bytes(max_in));
        DevicePtr scratch(allocation);
        if (status != cudaSuccess) { err = std::string("native dense scratch: ") + cudaGetErrorString(status); return false; }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        uint64_t packed_bytes = 0;
        size_t packed_count = 0;
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            if (item.packed) {
                strata::kernels::native_q8_0_packed_register(item.data.get(), item.packed.get(), (int) item.ref->ne0,
                                                             (int) item.ref->ne1);
                packed_keys_.push_back(item.data.get());
                packed_bytes += item.bytes;
                ++packed_count;
                weights_.push_back(item.packed.release());
            }
            weights_.push_back(item.data.release());
        }
        if (packed_count)
            std::fprintf(stderr, "native dense: STRATA_Q8_PACKED=1 packed %zu Q8_0 matrices for decode (+%.1f MiB)\n",
                         packed_count, packed_bytes / 1048576.0);
        scratch_ = scratch.release();
        bytes_ = total + hc_q8_bytes;
        allocated_ = allocated;
        if (hc_q8_requested())
            std::fprintf(stderr, "strata: STRATA_HC_Q8=1: %.2f GiB of Q8_0 hyper-connection projections for the verify read\n",
                         (double) hc_q8_bytes / 1073741824.0);
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
} // namespace strata::core
