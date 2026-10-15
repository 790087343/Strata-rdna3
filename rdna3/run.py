#!/usr/bin/env python3
"""Start the OpenAI-compatible server with one of the six RDNA3 setups the README measured (rdna3/configs/).

    .venv/bin/python rdna3/run.py iq4-2gpu --data ../Strata-data                # serve on 127.0.0.1:8080
    .venv/bin/python rdna3/run.py iq3-1gpu --data ../Strata-data --gpus 1       # HIP device 1
    .venv/bin/python rdna3/run.py q4-2gpu --data ../Strata-data --print-config  # the filled-in config only

--data is setup.py's data folder (models/, packs/, mtp/).  The engine is the one `./setup.sh --build` compiled
(engine/strata, engine/strata-vision; ROCm's library folders come from its engine/BUILD.json).  See rdna3/README.md.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUPS = [f"{q}-{g}gpu" for q in ("iq3", "iq4", "q4") for g in (1, 2)]
PACKS = {"iq3": ("iq3_s", "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"),   # setup.py's folder names
         "iq4": ("unsloth-ud-iq4_xs", "unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"),
         "q4": ("unsloth-ud-q4_k_xl", "unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")}


def fill(value, subst):
    if isinstance(value, str):
        for key, new in subst.items():
            value = value.replace(key, new)
        return value
    if isinstance(value, list):
        return [fill(v, subst) for v in value]
    if isinstance(value, dict):
        return {k: fill(v, subst) for k, v in value.items()}
    return value


def config(setup, data, gpus=None, vision=True):
    """The filled-in server config of one setup; exits with what is missing."""
    data = Path(data).resolve()
    cfg = json.loads((ROOT / "rdna3" / "configs" / f"{setup}.json").read_text())
    cfg = fill(cfg, {"@STRATA@": str(ROOT), "@DATA@": str(data)})
    count = int(setup.split("-")[1][0])
    cfg["gpu"] = [int(x) for x in gpus.split(",")] if gpus else list(range(count))
    if len(cfg["gpu"]) != count:
        sys.exit(f"{setup} needs {count} GPU(s); --gpus gave {cfg['gpu']}")
    build = ROOT / "engine" / "BUILD.json"
    if not build.is_file() or json.loads(build.read_text()).get("backend") != "hip":
        sys.exit("no HIP engine: run ./setup.sh --build first (rdna3/README.md, step 3)")
    cfg["lib_dirs"] = json.loads(build.read_text()).get("lib_dirs", [])
    if not vision:
        cfg.pop("vision", None)
        cfg["args"] = [a for a in cfg["args"] if a != "--vision"]
    pack = data / "packs" / PACKS[setup.split("-")[0]][0]
    shard = data / "models" / PACKS[setup.split("-")[0]][1]
    need = {"engine": Path(cfg["exe"]), "GGUF shard 1": shard, "pack index": pack / "index.txt",
            "tokenizer": Path(cfg["tokenizer"]) / "vocab.json", "MTP head": data / "mtp" / "rt" / "experts.bin"}
    if vision:
        need |= {"vision encoder": Path(cfg["vision"]["exe"]), "mmproj": Path(cfg["vision"]["mmproj"])}
    missing = [f"{what}: {path}" for what, path in need.items() if not path.is_file()]
    if missing:
        sys.exit("missing (rdna3/README.md, step 4):\n  " + "\n  ".join(missing))
    if not (pack / "experts.bin").is_file():
        # the measured setups read the experts from one prepacked file (rdna3/README.md, step 5)
        sys.exit(f"{pack / 'experts.bin'} is missing; write it once with\n"
                 f"  STRATA_GGUF_PY={ROOT / 'third_party/llama.cpp/gguf-py'} .venv/bin/python tools/iq_pack.py "
                 f"--gguf {shard} --out {pack} --experts-bin")
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("setup", choices=SETUPS)
    ap.add_argument("--data", required=True, help="setup.py's data folder (models/, packs/, mtp/)")
    ap.add_argument("--gpus", help="HIP device numbers, first = layers 0-23 (default 0 or 0,1; the README used 1 and 1,0)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-vision", action="store_true", help="no image encoder (saves its RAM and start time)")
    ap.add_argument("--log", help="engine log file")
    ap.add_argument("--print-config", action="store_true")
    a = ap.parse_args()
    cfg = config(a.setup, a.data, a.gpus, not a.no_vision)
    if a.log:
        cfg["log"] = str(Path(a.log).resolve())
    if a.print_config:
        print(json.dumps(cfg, indent=1))
        return 0
    path = ROOT / "rdna3" / "state" / f"{a.setup}-{a.port}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1) + "\n")
    os.chdir(ROOT)
    argv = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(path),
            "--host", a.host, "--port", str(a.port)]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    sys.exit(main())
