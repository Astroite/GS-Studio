# Contributing

Use native Windows for product development. Start with the [source structure](docs/development/STRUCTURE.md) and [maintenance guide](docs/development/MAINTENANCE.md). Keep user media, Runs, environments, SDKs, credentials, caches and raw benchmark/RE evidence outside public source. Preserve existing local changes.

## Source-only Git publication

Before committing or pushing, install the repository-local guards with your existing Python 3.10+ executable:

```powershell
.\scripts\install-source-guards.ps1 -PythonPath (Get-Command python).Source
python -I -B scripts/check_source_payload.py staged
python -I -B scripts/check_source_payload.py history --all
python -I -B -m unittest discover -s tests/source_guard -v
```

The installer backs up the previous local hooks setting and existing hook files inside the private Git directory. It changes only this repository's `core.hooksPath` and `gsstudio.guardPython`; it does not change global settings or remove local data. Commits inspect the complete staged Git snapshot, and pushes inspect every ancestor of each pushed branch/tag. Hooks never invoke Git LFS. Media, models, checkpoints, runtime binaries, raster images, LFS pointers under any filename and blobs larger than 5 MiB are rejected. Local data directories are rejected in staged/current tip snapshots. Existing small synthetic binary fixtures and the application icon have exact path/hash exceptions; future raster source assets require the same explicit review. See [source publication policy and recovery](docs/development/SOURCE-PUBLISHING.md) for the historical metadata boundary and limitations. Do not bypass the hooks or push private checkpoint refs.

## Development runtime

Build a new environment with the [public-input Windows bootstrap](docs/development/CLEAN-WINDOWS-BUILD.md): `plan`, `prepare`, `build`, then `verify`, using explicit workspace, Python/toolchain sources and GPU architecture. It installs the frozen 52-package dependency set and builds native extensions from pinned public sources and this repository. An existing matching environment can still be used; the older `scripts/provision_runtime.py` copies a frozen core and adds eight offline wheels, which does not establish a build from public inputs.

Set `GSSTUDIO_RUNTIME` to your own runtime JSON; `python` and `preprocess_python` must resolve to the same Python 3.10.22 executable with Torch 2.8.0+cu129, torchvision 0.23.0+cu129 and matching native extensions. Source roots and tool/cache paths are explicit configuration. Do not install into the directory package or change global Python/PATH to develop here. Runtime startup never invokes the build scripts or downloads dependencies.

From the repository root, after configuring your runtime:

```powershell
. .\scripts\session.ps1
$env:PYTHONPATH = (Join-Path $PWD 'src') + ';' + (Join-Path $PWD 'engine')
Invoke-Gsstudio --help
& $GsstudioPython -m pytest tests/unit tests/contract -ra
if ($LASTEXITCODE -ne 0) { throw 'pipeline CPU tests failed' }
```

Those suites may require development dependencies absent from a stripped release runtime. The retired GS Studio Qt frontend and its Qt-only tests are externally archived; do not install the old `studio` extra or PySide/PyQt for the current desktop. Retained `pipeline/training/native.py` and `pipeline/editor/runtime.py` are source-only legacy support for useful validation, not the release/runtime entry path. COLMAP 4.2.1 retains its legitimate linked Qt DLLs. Native compiler sources remain under engine/native and engine/editor_native. Building GPU extensions requires the matching CUDA/MSVC toolchain and separate validation on the target architecture; the application must not JIT them at startup.

## Frontend and Rust

Use Node.js 22 and the installed Rust 1.95.0 Windows MSVC toolchain, WebView2 and Tauri prerequisites. Select Rust for the current process, install the frontend exactly from its lock, then run its tests and build:

```powershell
$env:RUSTUP_TOOLCHAIN = '1.95.0'
Push-Location desktop
try {
    npm ci --ignore-scripts
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
    npm test
    if ($LASTEXITCODE -ne 0) { throw 'frontend tests failed' }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw 'frontend build failed' }
    cargo test --locked --manifest-path src-tauri/Cargo.toml -- --skip actual_worker_bootstrap_consumes_module_paths_before_argparse
    if ($LASTEXITCODE -ne 0) { throw 'Rust tests failed' }
} finally { Pop-Location }
```

These commands do not install the application/runtime. The skipped Rust startup regression requires a configured full private runtime; run it separately during local acceptance. Build and package instructions are in [Windows release](docs/development/WINDOWS-RELEASE.md). Dependency updates must include the corresponding lock and license review.

## Review and validation

Run `python .github/check_source.py` and `git diff --check` for documentation/structure changes. Build and packaging script changes also require `python -I -B -m unittest discover -s tests/build_scripts -v`; these tests use only the standard library and do not install or import the application runtime. The hosted [source workflow](.github/workflows/source-checks.yml) also runs the build contracts, a small pure-Python atomic JSON contract suite, locked frontend tests/build and Rust unit tests. It uses read-only repository permissions, no self-hosted runner and no release/publishing secrets. Action usage follows the upstream [checkout](https://github.com/actions/checkout), [setup-python](https://github.com/actions/setup-python) and [setup-node](https://github.com/actions/setup-node) documentation.

The CPU CI dependency set is intentionally narrow and does not install the app or Torch runtime. A separate, manually dispatched [clean Windows build](.github/workflows/clean-windows-build.yml) builds the exact public runtime inputs and the desktop executable on fresh hosted Windows VMs, without uploading binaries. It has no NVIDIA GPU; its compilation/import checks do not establish numerical, real-media, quality, final package or full desktop acceptance. Check the actual run for each source commit and never label pending checks passed. Run broader tests appropriate to the change in a configured development environment; record actual commands, failures, skips, source/lock hashes and output paths. GPU and release gates are scheduled locally and recorded separately. The [candidate verifier](scripts/release/verify_candidate.py) adds a read-only package-integrity check; it does not prove compiled origin, GPU behavior or redistribution permission.

Keep quality_v2/v5 math frozen. Preserve input identity, trustworthy split/QA gates, shared GPU/Run locks and failure evidence. Tests for current numerical functions, compiler sources and minimal synthetic fixtures must not be removed as historical clutter. Old Run compatibility is intentionally absent. Submit a focused change explaining the trigger, resulting behavior and actual validation; no commit, push or publication is implied by local editing.

Own contributions use Apache-2.0; preserve upstream copyright, LICENSE and NOTICE files for third-party changes. See [third-party notices](THIRD_PARTY_NOTICES.md).
