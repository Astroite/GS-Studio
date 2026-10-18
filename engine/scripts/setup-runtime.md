# Engine provisioning sources

The current GS Studio release target is one locked Python environment with
Torch 2.8.0+cu129 and matching torchvision/native extensions, shared by separate
preprocessing, training and editing worker processes. Use the
[Windows package guide](../../docs/development/WINDOWS-RELEASE.md) and
[contribution guide](../../CONTRIBUTING.md) for current setup boundaries.

The retained setup-runtime.ps1, setup-runtime-requirements.lock.txt,
setup-build-*.cmd and runtime recorders are engine provisioning/build sources.
They preserve native build recipes and artifact provenance; some scripts retain
developer-specific defaults. Inspect their explicit inputs before reuse. The
engine lock alone is not the complete current application runtime lock: the
packager requires the exact prepared shared environment's python_lock_file.
Do not install into an accepted package or mix environments to satisfy imports.

Native compiler sources remain in engine/native and engine/editor_native.
Current GPU binary validation covers architecture 8.9 only. Other architectures
need a matching rebuild and actual CUDA validation; packaging must never JIT a
missing extension. Preserve module licenses, upstream notices, build proof and
source/binary hashes.

quality_v2 and observed_v5 numerical baselines are frozen; observed_v6 remains
experimental. Missing optional extensions fail explicitly instead of silently
changing algorithms. Current tests retain CPU reference calculations and
explicit GPU gates. Read the test before running it: the complete engine suite
can initialize CUDA. Raw experiment/RE records are external; public methods and
aggregate results are in the [benchmark notes](../../docs/development/BENCHMARKS.md).
