# Engine provisioning sources

The current GS Studio release target is one locked Python environment with
Torch 2.8.0+cu129 and matching torchvision/native extensions, shared by separate
preprocessing, training and editing worker processes. Use the
[Windows package guide](../../docs/development/WINDOWS-RELEASE.md) and
[contribution guide](../../CONTRIBUTING.md) for current setup boundaries.

The repository-level `scripts/runtime/bootstrap_windows.py` owns the complete
application bootstrap. The former engine-only `setup-runtime.ps1` now refuses
execution and points to that recipe; it cannot install into a guessed data root.
The engine lock alone is not the complete current application runtime lock: the
packager requires the exact prepared shared environment's `python_lock_file`.
Do not install into an accepted package or mix environments to satisfy imports.

## Explicit native builds

`build_native.py` is the common offline entrypoint for gsplat, fused-ssim, SPZ,
the four independent Splat3 extensions and the editor selection helper. Invoke
it with an existing build Python, an explicit MSVC `vcvarsall.bat`, CUDA 12.9,
pinned source trees and a **new output directory outside the repository**:

```powershell
$BuildPython = (Resolve-Path '..\build-env\python.exe').Path
$Job = Join-Path (Resolve-Path '..\build-jobs').Path 'render-001'
& $BuildPython -I -B -X utf8 .\engine\scripts\build_native.py splat3-render --python $BuildPython --output $Job --plan
# A real build additionally receives --vcvars <path> and --cuda-home <path>.
# Remove --plan only after inspecting the plan and toolchain inputs.
```

`--plan`/`--dry-run` validates the source paths, snapshots their hashes and emits
JSON without importing Torch, running a compiler or creating the output.
It records expected external revisions; only a real build verifies the Git
checkout or complete archive receipt before invoking the compiler.
Local source paths resolve from the script, independent of the current working
directory. A real build copies source files into its new job before compiling;
it never installs packages, downloads dependencies, edits the input checkout or
changes global environment settings. Failed jobs retain their log and proof.
The default architecture is 8.9, jobs default to two and MSVC toolset defaults to
14.44; all are explicit overrides rather than GPU discovery.

| Module argument | Source inputs | Output relative to the new job |
| --- | --- | --- |
| `gsplat` | `--gsplat-source`, including pinned GLM | `wheels/*.whl` |
| `fused-ssim` | `--fused-ssim-source` | `wheels/*.whl` |
| `spz` | `--spz-source`, `--zlib-source`, `--zstd-source` | `wheels/*.whl` |
| `splat3-stats`, `splat3-render`, `splat3-tiles` | Script-relative `engine/native` or `--source-root` | `wheels/*.whl` |
| `splat3-projection` | Script-relative source and `--gsplat-source` for GLM | `wheels/*.whl` |
| `studio-selection` | Script-relative `engine/editor_native` or `--source-root` | `extensions/studio-selection/<ABI-source-key>/*.pyd` |

External Git checkouts must match the fixed commits in the helper and have clean
working trees. Verified archive inputs instead require `--source-receipts` with
each component's revision and complete relative-path/SHA256 map, using the same
`source_files()` snapshot function. The gsplat receipt also binds GLM's revision.
SPZ uses local zlib/zstd sources with CMake fetching disabled and produces a
normal 1.1.0 CPython 3.10 Windows x64 wheel. Each wheel's name, version and ABI
metadata are checked before the build is reported successful.

The seven `setup-build-*.cmd` wrappers call this same helper. Set only the current
process's `GSSTUDIO_BUILD_PYTHON` to launch them, and pass the same explicit CLI
options; `LP_PYTHON` remains an interpreter-only alias. They do not infer a Python
installation layout or Visual Studio edition. `build_studio_selection.ps1`
accepts `-PythonPath`, `-Output`, `-VcVars`, `-CudaHome`, `-Architecture` and
`-PlanOnly`, without loading a session or editing environment variables.

Run the CPU-only entrypoint contracts with:

```powershell
python -I -B -X utf8 -m unittest discover -s tests/build_scripts -v
```

These fixture checks cover relative paths, spaces/Unicode, archive identity,
fresh-output protection, wrapper failure codes and wheel metadata. They do not
establish a successful clean Windows compile or CUDA correctness.

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
