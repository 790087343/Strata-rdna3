# Fast engine starts for experiments (STRATA_ARENA_SHM) and A/B inside one engine

[claude] 2026-10-24, branch `fast-start-20261024` (from c9bb9d4e, the `claude-20261024-lv` release source).
Opt-in only: with neither variable set the engine behaves exactly as before.

## 1. STRATA_ARENA_SHM: keep the expert arena in shared memory across starts

Every benchmark run restarts the engine, and most of a start is the expert arena: 46.84 GiB (qwen IQ3_S) read from
experts.bin into an anonymous mapping that dies with the process.

`STRATA_ARENA_SHM=<name>` backs the arena with a shared-memory object that outlives the engine:

- **The first start** creates `/dev/shm/strata-arena-<name>-<key>`: `fallocate` (tmpfs pages, so a full /dev/shm is an
  error now rather than a SIGBUS later), then `MADV_COLLAPSE` into 2 MiB folios. The normal load fills it, 32 sampled
  experts are compared with the files, and the header is then marked "complete".
- **Every later start with the same key** maps that object as its arena (MAP_SHARED, nothing copied), then registers it
  with the driver (`cudaHostRegister`) exactly as before and compares 32 random experts with the files. It reads nothing.
- **When /dev/shm cannot hold the object** (65.0 GB here, so huihui IQ4_XS at 65.4 GB and both Q4_K_XL packs at 77.0 GB),
  the object is a memfd held by a `sleep infinity` keeper in its own session. Its index file is
  `/dev/shm/strata-arena-<name>-<key>.keeper` (pid and fd), and it survives the harness's process-group kill.

### How the key works, and why a stale object is never used

- **What the key covers:** a 64-bit FNV hash of the pack's metadata (`manifest.json`, `index.txt`,
  `native_experts.txt`), sampled content of every source file, every source file's canonical path, size, mtime (ns),
  inode and device, the pack directory, the arena size and per-layer layout, and a loader layout version
  (`kArenaShmLayout`; bump it when the loader writes different bytes for the same files).
- **Where the key appears:** the key is part of the object's name, and the header repeats the key, the arena size and
  the state.
- **A different pack, quant or touched file** gets another name. A wrong size or key in the header, or a state that is
  not complete, is never mapped.
- **Stale objects of the same `<name>` are deleted.** These are objects with another key, or an incomplete one whose
  creator died. They are deleted when no engine holds them, so their RAM comes back. An object in use by another engine
  is kept, and this start then uses its own object.
- **A failed sampled check** makes the start load a private arena as usual. The bad object is also deleted if nothing
  else holds it.
- **Locking:** a creator holds `flock(LOCK_EX)` while it fills the object, and every engine using an object holds
  `LOCK_SH` for its whole life. A second engine that starts during the fill waits up to about 10 minutes, then attaches.
- **A failed attach or create is never fatal:** the start loads a private arena exactly as without the variable.

### Use it (prun.py)

```
prun.py CONFIG OUT ... --exe <a fast-start build> --env STRATA_ARENA_SHM=iq3
```

- Use one name per pack, for example `iq3`, `iq4` or `hq4`. Reusing a name for another pack deletes the old object when
  it is unused.
- The 1-GPU and 2-GPU layouts of the same pack share one object. The arena is the same bytes for both, so a 2-GPU run
  attaches to the object a 1-GPU run made.
- Two engines of the same pack running at once share it too, so their RAM is not counted twice.
- Done for the day: `rm /dev/shm/strata-arena-<name>-*`. For a memfd object, `kill <pid in the .keeper file>` and then
  remove the `.keeper` file. Both are only safe when no engine uses the object; a running engine keeps its mapping
  anyway.
- Knobs:
  - `STRATA_ARENA_SHM_DIR=<dir>`: another tmpfs instead of /dev/shm.
  - `STRATA_ARENA_SHM_MEMFD=1`: always use the memfd and keeper.
  - `STRATA_ARENA_SHM_THP=0`: keep 4 KiB pages (A/B only).
- Linux only.

### RAM

- **While an engine runs:** the same as before. The arena is the shm object, not a copy, so it shows under
  `Shmem`/`ShmemHugePages` instead of `AnonHugePages`.
- **Between runs:** the object keeps its 47 to 72 GB, depending on the pack, out of `MemAvailable` until it is
  deleted. That RAM is not available to other agents' engines, so delete it when the session ends.
- **Reading free RAM at startup:** a reused start sees about 47 GB less `MemAvailable` before its arena exists than a
  cold one does. The engine's arena RAM check is skipped for a shm arena (the memory is already held). The memory
  guard's start-up line reads lower, but its runtime readings are the same as with a private arena.
- **Page backing:** tmpfs does not give huge pages on fault while `shmem_enabled` is `never` (as here).
  `MADV_COLLAPSE` ignores that setting (all but `deny`). The object's folios stay 2 MiB for every later start, and the
  attach's `MADV_COLLAPSE` maps them by PMD in about 0 s. The arena therefore has the same 2 MiB backing as the
  anonymous THP arena (measured: 46.8 of 46.8 GiB).

### Measured (2026-10-24, qwen IQ3_S, results/merge-20261024/fs-*)

All times are seconds from prun's start to the server's `ready` line.

| start | 1-GPU (prod4-1gpu) | 2-GPU TP (prod4-2gpu) |
|---|---|---|
| before: private arena, lv binary | ~53 s (89.5 s with STRATA_VERIFY_ARENA's 36 s checksum) | 42-94 s (earlier logs) |
| first start with STRATA_ARENA_SHM (creates the object) | 60.1 s (+8.4 s fallocate + collapse) | |
| later starts (attach) | **7.2 s** | **14.6 s** |

Correctness:

- **Full checksum:** STRATA_VERIFY_ARENA gives 8ae1446c68dc969b for both the private load and the attached object.
- **Sampled check:** 32 sampled experts match the files on every attach.
- **qeval:** qeval2 40/40, and qeval arith 6 / fact 29 (earlier same-config runs: 6-7 / 29).

Where the 1-GPU start goes before and after:

| step | before | attach |
|---|---|---|
| server, vision encoder (CPU, no warm-up) | 0.7 | 0.7 |
| process + HIP init, dense weights, native projections (2.2 s) | 3.5 | 3.4 |
| KV pinned RAM, PLE, MTP draft layer | 1.0 | 1.0 |
| expert arena: register (fresh pages faulted in and zeroed) | ~2.4 | 0.09 |
| expert arena: read | ~44 (1.07 GiB/s buffered, the page cache half warm; 25 s when unbuffered at 1.91 GiB/s) | 0 (attach 0.0, sampled check 0.07) |
| GEMM library warm (a thread beside the arena; STRATA_PREFILL_WARM=0 skips) | 2.3, hidden by the arena | 0.1 |
| expert-cache prefill from the arena (17.7 GiB to VRAM) | 1.9 | 1.9 |
| session, MTP head, server ready | 0.7 | 0.7 |

What is left, about 7 s on one GPU, is mostly real work. These are other costs that were looked at and not changed:

- **Native projections (2.2 s, 2.6 GB from the GGUF).** A shm cache like the arena's could save about 2 s. Left alone:
  small next to the arena.
- **GEMM warm.** It runs on a thread beside the arena: 2.3 s while the arena loads, 0.1 s on an attached start.
  Nothing to gain.
- **Vision encoder (0.4 s on the CPU).** Not worth skipping, and skipping it would change what the config runs.
- **Window graph captures (~1 s).** These run in the first request, not in the start, so they already sit in prun's
  warmup.
- **cudaHostRegister.** 0.09 s on one GPU and 0.36 s on two, for 46.8 GiB of 2 MiB folios. It was never a big cost.

## 2. A/B inside one engine (per-request switches)

The idea: one engine load serves both arms of an experiment, interleaved request by request, instead of one start per
arm.

### The audit

Every STRATA_* switch the current experiments touch (STRATA_REFILL_*, STRATA_PREFILL_HELP*, STRATA_MTP_*,
STRATA_TP_NOMEMSET) is read once per process: either a `static const` lambda, or a read at load time. None is re-read
per request today. They fall into three classes:

- **(a) Host-only, read per request or per window. An atomic is enough.**
  - REFILL_PACE, RATE, DEMAND, DEFER, JOIN: **done here.**
  - REFILL_BLOCKING, PARTS, SERIAL, TRANCHES, TRANCHE_SLOTS, SDMA.
  - REFILL_OVERLAP / ORDER: only if `Prefill::init` always turns the prompt's routing counts on.
  - PREFILL_HELP_FRAC, PREFILL_HELP_RING (clamped to the allocated ring).
  - MTP_TOP2, MTP_BATCH / _RING / _F16.
  - MTP_CATCHUP_ALL and MTP_DRAFT_ALL. Their graphs are cached per round size or per mode, so the first switch pays one
    capture, and the extra graphs cost VRAM.
  - MTP_PREFILL_SYNC, with care: the other cache is prepared lazily, and that path once hung.
- **(b) Baked into a captured graph.** These are TP_NOMEMSET=1 (the memset node in the verify window), MTP_MULTICOPY and
  MTP_QFUSE (the draft round and step graphs), and MTP_SHARED_BRANCH (CUDA only).
  - No capture is ever dropped today. `Verifier::exec_[T]` / `exec_nr_[T]` and `MtpDrafter`'s `round_` / `step_` /
    `all_` / `prefill_*_exec_` are freed only in the destructors.
  - Two ways to switch them:
    - Key the caches by the switch value, as `refresh_ar` already keeps two variants. This costs VRAM, which a 2-GPU TP
      start does not have: 358 MiB free on CUDA1.
    - Add a "drop and re-capture" call (sync, destroy, null) behind a `SET` command between requests, modelled on the
      `VRAM <mib>` command.
- **(c) Start-up only: allocations, streams, threads, placement.** These are PREFILL_HELP (the helper wiring and
  `PeerPrefill` at load), PREFILL_HELPER_SOURCE, MTP_FULL_HEAD, MTP_GR_R8 (a buffer, plus kernels inside the captured
  draft graphs), MTP_HIP_PRIO / PRIORITY / SIDE_STREAM (streams), MTP_PROFILE, MTP_STAGE0, REFILL_SDMA_ENGINE (a
  per-device copier) and TP_NOMEMSET=2 (an allocation in `set_tp`). These keep a restart, which STRATA_ARENA_SHM now
  makes cheap.

### Implemented (the cheap, safe part)

- **Engine:** `include/strata/core/request_switches.hpp`. A GEN/GENI line takes `env=NAME:VALUE,NAME:VALUE`.
  - Every GEN/GENI starts from the start-up values (the environment) and then applies its own list, right after the
    key parse and before any code of the request reads a switch.
  - A request without the key runs exactly as the engine was started.
  - The engine logs `strata serve: this request's switches: ...`, and logs any name that is not switchable as ignored.
  - The five refill knobs read the request's value. The two lines that were `static const` per request are now read at
    the request's refill setup and in each window.
- **Server:** `"strata_tune": {"env": {"STRATA_REFILL_PACE": 64}}` in a request. Names and values must be of
  `[A-Za-z0-9_.-]`.
- **Caveats:**
  - BGEN batch slots do not parse the key; they keep the last GEN's values.
  - A refill that a request left in flight under `--pipeline-windows` is paced by the next request's values. Only the
    timing changes, not the result.
  - Arms in one engine share state: the adaptive tier's cache contents, the conversation cache and the card
    temperature. Interleave ABBA, avoid prefix reuse between arms (other prompts, or `"strata_checkpoint": false`), and
    still confirm a win with a restart pair.

### Effort for the rest

| step | work |
|---|---|
| the rest of class (a), about 15 switches | about 1 day: the same pattern. Each needs a check that greedy output with `env=` matches an engine started with the variable. |
| class (b), the drop-and-re-capture command | about 2-3 days: a `SET` command, `Verifier::drop_captures()`, `MtpDrafter::drop_captures()` (also fixing the unfreed `all_exec_`), a VRAM check, and a soak. The risk is the captures' VRAM on a full card. |
| class (c) | no change planned: a restart with STRATA_ARENA_SHM (7-15 s). |
