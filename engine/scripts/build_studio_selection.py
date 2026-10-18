"""Explicit offline editor build; runtime operations never call this script."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def compile_worker(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    args = parser.parse_args(argv)
    os.environ["TORCH_CUDA_ARCH_LIST"] = args.architecture
    import torch
    from torch.utils.cpp_extension import load
    source = args.source_root
    key = hashlib.sha256((torch.__version__ + sys.version).encode())
    for name in ("selection.cpp", "selection.cu"):
        key.update((source / name).read_bytes())
    output = args.output / "studio-selection" / key.hexdigest()[:20]
    output.mkdir(parents=True, exist_ok=False)
    load(name="studio_selection_cuda",
         sources=[str(source / "selection.cpp"), str(source / "selection.cu")],
         build_directory=str(output), extra_cflags=["/O2"],
         extra_cuda_cflags=["-O3"], verbose=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("Offline compilation initialized CUDA")
    proof = {"output": str(output), "architecture": args.architecture,
             "cuda_initialized": False, "source_sha256": {
                 name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                 for name in ("selection.cpp", "selection.cu")}}
    (output / "build-proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        compile_worker(argv)
        return 0
    # -I deliberately omits the script directory; add this reviewed helper only.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_native import main as build_main
    return build_main(["studio-selection", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
