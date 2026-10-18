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

## Existing development runtime

Use an already provisioned private Python environment with Torch 2.8.0+cu129, matching torchvision/native extensions and the pipeline dependencies. Set `GSSTUDIO_RUNTIME` to your own runtime JSON; `python` and `preprocess_python` must resolve to the same executable. Source roots and tool/cache paths are explicit configuration. Do not install into the directory package or change global Python/PATH to develop here.

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

Use Node.js 22 and Rust with the Windows MSVC toolchain, WebView2 and Tauri prerequisites. Install the frontend exactly from its lock, then run its tests and build:

```powershell
Push-Location desktop
try {
    npm ci
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

Run `python .github/check_source.py` and `git diff --check` for documentation/structure changes. The hosted [workflow](.github/workflows/source-checks.yml) also runs a small pure-Python atomic JSON contract suite, locked frontend tests/build and Rust unit tests. It uses read-only repository permissions, no self-hosted runner and no release/publishing secrets. Action usage follows the upstream [checkout](https://github.com/actions/checkout), [setup-python](https://github.com/actions/setup-python) and [setup-node](https://github.com/actions/setup-node) documentation.

The CPU CI dependency set is intentionally narrow and does not install the app or Torch runtime. It does not establish numerical, CUDA, real-media, quality, offline-package or full desktop acceptance. Run broader tests appropriate to the change in a configured development environment; record actual commands, failures, skips, source/lock hashes and output paths. Never label pending checks passed. GPU and release gates are scheduled locally and recorded separately.

Keep quality_v2/v5 math frozen. Preserve input identity, trustworthy split/QA gates, shared GPU/Run locks and failure evidence. Tests for current numerical functions, compiler sources and minimal synthetic fixtures must not be removed as historical clutter. Old Run compatibility is intentionally absent. Submit a focused change explaining the trigger, resulting behavior and actual validation; no commit, push or publication is implied by local editing.

Own contributions use Apache-2.0; preserve upstream copyright, LICENSE and NOTICE files for third-party changes. See [third-party notices](THIRD_PARTY_NOTICES.md).
