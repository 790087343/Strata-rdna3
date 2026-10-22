// include/strata/core/tp.hpp - TENSOR-PARALLEL DECODE across two cards (docs/STRATA_TP_DESIGN_20261022.txt, M1).
//
// Each card runs EVERY layer of the window on half the heads: GDN k-heads 16 -> 8 a card (v-heads 48 -> 24, the conv
// and the recurrence head-local), QSA kv-heads 2 -> 1 (with its 12 query heads), and the out-projections row-split so
// that each card holds a partial sum of the block output, which the two cards add through coherent pinned host memory
// (Verifier::TpLink, verify_kernels.hpp tp_exchange) in a fixed order - so both cards hold bitwise the same residual,
// hyper-connection read and router input.  The routed experts stay where the layer split put them (card 0 the first
// layers, card 1 the rest): the owner computes the layer's MoE and sends the block output to the other card.
//
// The verifier itself does not know it is running half a model: it is given a half ModelGeometry (tp_geometry), a
// session carved for it, and a weight table whose head-split tensors are slices (TpWeights).  The kernels take the head
// counts as parameters already.
#pragma once

#include "strata/core/layout.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// STRATA_TP_SHEXP: where the shared expert runs.  split (0): by ff unit across the cards (its weights sliced);
/// card1 (1): whole on card 1, which otherwise waits for card 0's MoE at the layer's second exchange (the default);
/// card0 (2): whole on card 0.  The whole card loads the shared-expert weights of the layers its stage does not hold.
int tp_shexp_mode();
inline bool tp_shexp_split() { return tp_shexp_mode() == 0; }

/// The geometry card `rank` of `nranks` runs: the GDN / QSA heads split by head, everything else the model's.
/// False with `err` when the heads do not divide.
bool tp_geometry(const ModelGeometry& g, int rank, int nranks, ModelGeometry& out, std::string& err);

/// The pinned pages two cards exchange through: two inboxes a card (by exchange parity) of `floats` each and one flag a
/// card, coherent and portable, with each card's device addresses of all of them (Verifier::TpLink is filled from here).
struct TpPages {
    int dev[2] = {0, 1};
    float* inbox_h[2][2] = {};        ///< [card][parity] host address
    float* inbox_d[2][2][2] = {};     ///< [card][parity][as seen from device index 0/1]
    uint32_t* flag_h[2] = {};
    uint32_t* flag_d[2][2] = {};      ///< [card][as seen from device index 0/1]
    int64_t floats = 0;
    ~TpPages();
    TpPages() = default;
    TpPages(const TpPages&) = delete;
    TpPages& operator=(const TpPages&) = delete;
    /// `floats` per inbox (kVerifyMaxT x n_embd for the block output).  Needs both devices' peer access to host memory
    /// only (no GPU-to-GPU peer access).
    bool alloc(int dev0, int dev1, int64_t floats, std::string& err);
};

/// Card `rank`'s weights for the TP window: `full` (a table on this device that holds EVERY layer's small tensors - the
/// hyper-connections, norms, routers, indexers, the GDN a/b/conv rows, as the untrimmed pack arena does) copied, with
/// the head-split tensors replaced by slices uploaded here, and the dense projections of all layers loaded from the
/// GGUF shards at half width (row slices for qkv / z / q / k / v, a reduction-dimension slice for the out-projections).
/// Everything else keeps `full`'s pointers.  Build on the card the table is for.
class TpWeights {
public:
    TpWeights() = default;
    ~TpWeights();
    TpWeights(const TpWeights&) = delete;
    TpWeights& operator=(const TpWeights&) = delete;
    bool build(const WeightTable& full, const ModelGeometry& g, int rank, int nranks,
               const std::vector<std::string>& shards, std::string& err);
    const WeightTable& table() const { return table_; }
    uint64_t bytes() const { return bytes_; }

private:
    WeightTable table_;
    std::vector<void*> owned_;
    uint64_t bytes_ = 0;
    void* take(uint64_t bytes, std::string& err);
};

/// THE STATE HAND-OVER.  The prompt is read layer-split into the stages' sessions (each holding every head of its
/// layers); the TP window reads card r's session, which holds heads [r/2, (r+1)/2) of every layer.  `to_tp` copies
/// the first `cells` positions' state over (the GDN recurrence and conv history by head, the K/V pools - the VRAM slots
/// and, when streamed, the host copy and the slot map - by kv-head, the indexer / PLE history / token window
/// replicated); `to_tp == false` copies it back (before a snapshot or the next prompt read).  `stage_of(l)` names the
/// stage session that holds layer l.  Every stream of both sides must be idle.
bool tp_state_move(const ModelGeometry& g, const ModelGeometry& tg, SessionState* const* stage_ss, const int* stage_dev,
                   int n_stages, const int64_t* stage_lb, SessionState* const* tp_ss, const int* tp_dev, int64_t cells,
                   bool to_tp, std::string& err);

}  // namespace strata::core
