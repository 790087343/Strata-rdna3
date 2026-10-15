# Reproducing the RDNA3 numbers

Everything needed to rebuild the engine and re-measure the README's table and charts on your own RX 7900 XTX
(gfx1100) from this repository alone. Every version is pinned, and `setup.py` downloads each one.

## What was measured, on what

| | |
|---|---|
| GPUs | 1 or 2 × AMD Radeon RX 7900 XTX, 24 GB each, PCIe 4.0 x8 each (HIP devices 1 and 1,0 below) |
| CPU, RAM | AMD Ryzen 9 9950X3D2, 128 GB DDR5 |
| Disk | NVMe SSD (the models, packs and the n-gram table are read from it) |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic with its in-kernel `amdgpu` driver (no DKMS, no system ROCm) |
| Compiler | GCC 15.2.0 (any C++20 GCC should do) |
| Python | 3.13 (any 3.10+ with `venv`) |
| ROCm | TheRock wheels `rocm-sdk-*==7.10.0a20251120`, `gfx110X-dgpu` index, installed into `.venv` by `setup.py` |
| rocWMMA | headers at commit `1ab208f4` (tag `rocm-7.1.1`), fetched by `setup.py` |
| llama.cpp (ggml, gguf-py) | commit `3cf03257`, fetched by `setup.py` |
| cmake, ninja | from `PATH` when present (here cmake 4.2.3, ninja 1.13.2), else 4.4.3 / 1.13.2 from pip (`requirements.txt`) |
| Engine | this branch (`rdna3`), compiled by `./setup.sh --build` |

RAM: a 1-GPU setup keeps the experts that do not fit the card in RAM, so you need roughly the size of the pack's
`experts.bin` plus ~15 GB: IQ3_S 50 GB, UD-IQ4_XS 60 GB, UD-Q4_K_XL 77 GB. With less, the engine reads experts from
the SSD while it answers and is much slower than the table. No `ulimit -l` change and no root are needed.

## 1. System packages

```bash
sudo apt install build-essential git python3-venv     # Fedora: gcc-c++ git python3
```

Your user must be able to open `/dev/kfd` and `/dev/dri/renderD*` (the `render` and `video` groups on Ubuntu).

## 2. Get the code

```bash
git clone https://github.com/nicejunjie/Strata-rdna3.git
cd Strata-rdna3
```

## 3. Install the toolchain and build the engine (and fetch the models)

One run per model. The first one makes `.venv`, installs ROCm from AMD's wheels, fetches llama.cpp and rocWMMA, and
compiles the engine and the CPU image encoder for gfx1100 with the fork's options (`-DSTRATA_MMQ_KQUANTS=ON` and
`STRATA_ROCWMMA_INCLUDE_DIR`; 10-20 minutes). Later runs reuse all of that. Each run downloads its model's GGUF
shards from Hugging Face (checked against their published sha256), writes its pack, and fetches the MTP draft head.

> [!WARNING]
> `setup.py` (upstream behaviour) **moves** the model files of every Strata install it ran from before, listed in
> `~/.config/strata/settings.json`, into the data folder you give it. If this PC already has a Strata install whose
> models you want to keep where they are, give this one its own settings: `export XDG_CONFIG_HOME=$PWD/.config`
> before the commands below (and before every later `./setup.sh`).

```bash
D=$PWD/../Strata-data     # models/, packs/, mtp/: an absolute path
./setup.sh --setup --build --backend hip --family qwen    --model IQ3_S      --vision cpu --data-dir $D --no-start --yes
./setup.sh --setup --build --backend hip --family unsloth --model UD-IQ4_XS  --vision cpu --data-dir $D --no-start --yes
./setup.sh --setup --build --backend hip --family unsloth --model UD-Q4_K_XL --vision cpu --data-dir $D --no-start --yes
```

Already have the GGUFs? Put them, under their original names, in `$D/models/IQ3_S/`, `$D/models/unsloth-UD-IQ4_XS/`
and `$D/models/unsloth-UD-Q4_K_XL/` before running setup (it then only checks that each file is whole), and compare
them with [`SHA256SUMS`](SHA256SUMS). `--gguf-dir` also works for running the model with `./setup.sh`, but
`rdna3/run.py` expects the files in `$D/models/`. `UD-Q4_K_XL` prints a warning about AMD cards:
that warning is upstream's, written before this fork's K-quant prompt kernels; `--yes` accepts it.

## 4. One expert file per model

The measured setups read the experts from one prepacked file per model (on Linux with `O_DIRECT`, ~1.9 GiB/s from a
PCIe 4 drive) instead of out of the GGUF shards. It is a byte-for-byte copy of the GGUF's expert tensors in the order
the engine reads them, written once:

```bash
export STRATA_GGUF_PY=$PWD/third_party/llama.cpp/gguf-py
.venv/bin/python tools/iq_pack.py --gguf $D/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
    --out $D/packs/iq3_s --experts-bin
.venv/bin/python tools/iq_pack.py --gguf $D/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
    --out $D/packs/unsloth-ud-iq4_xs --compat-bf16 --experts-bin
.venv/bin/python tools/iq_pack.py --gguf $D/models/unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    --out $D/packs/unsloth-ud-q4_k_xl --compat-bf16 --experts-bin
```

(50, 60 and 77 GB. `--compat-bf16` is what `setup.py` passes for the Unsloth files: their Q8_0 hyper-connection
projections become BF16, the form the engine reads; without it `iq_pack.py` stops before writing anything.)

## 5. Run a setup

`rdna3/configs/` holds the six setups exactly as measured: the engine arguments and every environment switch.
`rdna3/run.py` fills in your paths and starts the OpenAI-compatible server (`/v1/chat/completions`):

```bash
.venv/bin/python rdna3/run.py iq4-2gpu --data $D --gpus 1,0      # iq3-1gpu iq3-2gpu iq4-1gpu iq4-2gpu q4-1gpu q4-2gpu
```

`--gpus` takes HIP device numbers. With two, the first holds layers 0-23 and the second 24-47. The README used
`1` and `1,0`; on your PC `0` / `0,1` (the default) is as good, unless one card also drives the desktop. Leave that
card out of a 1-GPU setup, or put it second. `--print-config` shows the filled-in config; the same settings are
written out in [`docs/RDNA3_SETTINGS.md`](../docs/RDNA3_SETTINGS.md).

## 6. Measure

```bash
.venv/bin/python rdna3/bench.py iq4-2gpu --data $D --gpus 1,0                        # 1K ... 256K, ~20 min
.venv/bin/python rdna3/bench.py q4-1gpu  --data $D --gpus 1 --lengths 8192,32768     # a quick check
```

`bench.py` starts the setup's server itself (stop any other one first: both GPUs must be idle) and measures the
README's way. Prompts are EXACTLY 1K, 2K, ... 128K and 261,632 tokens, built with the model's own tokenizer and chat
template. Three secret codes sit at the start, middle and end, followed by a request for a long answer. Each length
runs 3 times with 256 output tokens, temperature 0, seed 42, no prompt cache and thinking off. Every request must
report exactly n prompt tokens (none cached) and 256 completion tokens, and must return the three codes. Speeds are
the engine's own timings; a cell is the median of the three. It prints a table against
[`reference.json`](reference.json), the README run of 2026-10-13. Results go to `rdna3/results/<setup>/`.

What to expect: correctness checks pass every time. Speeds within a few percent of the reference on the same
hardware, with decode varying more than prefill (it depends on how many MTP drafts are accepted, and on the
cards' clocks and temperature). A 1K-8K first cell can read lower while the cards warm up.

## Checked from scratch

On 2026-10-14, a fresh anonymous clone of this repository (commit `7b30f1bb`) went through steps 1-6 exactly as
written above, with system Python 3.14. The fresh install was kept apart from an existing Strata install with
`XDG_CONFIG_HOME`. The results:

- the model files matched `SHA256SUMS`;
- the three `experts.bin`, the packs' `dense.bin`, `index.txt` and `conversions.json`, and the MTP head came out
  byte-identical to the ones the README was measured with;
- all 162 benchmark requests (6 setups x 9 lengths x 3) passed every check;
- prefill was within 0.97-1.03x of `reference.json` in every cell;
- decode was within 0.95-1.09x of `reference.json` in every cell but one. The exception, `iq3-1gpu` at 4K, read
  1.26x: the reference cell there is low (86 tok/s against 95-108 for its neighbours).

