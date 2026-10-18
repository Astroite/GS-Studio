# GS Studio desktop

Windows React/Tauri workstation. CLI and desktop use the same pipeline and runtime configuration. The runtime's two interpreter aliases point to one locked Python environment in new v0.3 packages. Existing private environments are provisioning inputs only; do not install into or alter them.

Build v0.3.0-rc.1 from the repository root with the locked dependencies:

~~~powershell
# Configure GSSTUDIO_RUNTIME to your existing shared runtime before full worker tests.
Push-Location desktop
try {
    npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw 'frontend dependency installation failed' }
    npm.cmd test
    if ($LASTEXITCODE -ne 0) { throw 'frontend tests failed' }
    npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'frontend build failed' }
    cargo test --locked --offline --manifest-path src-tauri/Cargo.toml
    if ($LASTEXITCODE -ne 0) { throw 'Rust tests failed' }
    cargo build --release --locked --offline --manifest-path src-tauri/Cargo.toml
    if ($LASTEXITCODE -ne 0) { throw 'Rust build failed' }
} finally { Pop-Location }
~~~

The release EXE is src-tauri/target/release/gs-studio.exe. The directory packager supplies runtime modules, tools and weights; a bare EXE or NSIS bundle alone is incomplete. npm run dev provides a browser preview without an engine. Tests use temporary projects and CPU-only probes.

Prepare a **new** environment from the frozen core plus required pipeline dependencies, preserving native component versions and numerical sources. Supply its exact frozen requirements file as package.python_lock_file. Neither packaging nor the build commands provision or upgrade Python runtime dependencies. The minimum runtime includes torchvision for masking, the pipeline's Pydantic/Typer/YAML/OpenCV dependencies, the frozen scientific dependencies, gsplat.csrc, fused_ssim_cuda, local_splat3_{stats,render,tiles,projection} and spz.spz. Preserve their distribution metadata and licenses.

Prepare a staging manifest using scripts/runtime-package.example.json. Paths resolve relative to that JSON. Both runtime root fields must name the same root and both interpreter aliases the same executable. Supply explicit FFmpeg and the existing official COLMAP 4.2.1 roots, verified offline Torch weights (including Mask R-CNN and AlexNet checkpoints), redistribution notices and the frozen requirements file. model_cache_roots has one source matching torch_home, with destination torch. Do not use the accepted application or its runtime as output.

Build selection offline against the final Python/Torch/source identity using engine/scripts/build_studio_selection.ps1. Set torch_extensions_dir to the staged engine/extensions directory, keep release=false while provisioning, and retain the resulting build-proof.json and source files. The packaging inspector recomputes the key without initializing CUDA or compiling. Packaged selection loads from engine/extensions/studio-selection/<key>, independent of TORCH_EXTENSIONS_DIR or a mutable user cache. The final runtime is marked release=true automatically.

Inspect and then package into a **new, nonexistent directory**:

~~~powershell
# Run from the repository root.
pwsh -NoProfile -File desktop/scripts/package-gsstudio.ps1 -RuntimeConfig '<stage>/runtime/runtime.json' -Output '<new-public-directory>' -Profile public -PlanOnly
pwsh -NoProfile -File desktop/scripts/package-gsstudio.ps1 -RuntimeConfig '<stage>/runtime/runtime.json' -Output '<new-public-directory>' -Profile public -RuntimeReady
# Explicit local distribution: also requires mediasdk_runtime_root and media_helper.
pwsh -NoProfile -File desktop/scripts/package-gsstudio.ps1 -RuntimeConfig '<stage>/runtime/runtime.json' -Output '<new-local-directory>' -Profile local -RuntimeReady
pwsh -NoProfile -File desktop/scripts/test-package-plan.ps1
~~~

RuntimeReady records explicitly verified provisioning readiness. All profiles copy runtime/python once, preserve python/preprocess_python aliases, put models in models/torch (TORCH_HOME), and retain FFmpeg, COLMAP and notices. COLMAP 4.2.1 and its legitimate linked Qt DLLs are copied without modifying their bytes; do not rebuild COLMAP or delete those dependencies. The retired GS Studio Python PySide/PyQt frontend is excluded separately. COLMAP build test executables/headers are pruned. The local profile copies only the SDK's bin tree, including helper and models, once under tools/insta360/bin. helper_root is obsolete and unused. Public packages omit commercial SDK/helper binaries and keep nullable media_helper, mediasdk_root and realityscan fields for external configuration. RealityScan is never copied or found implicitly on PATH. Optional export XML is data and may be bundled. Extra redistributable_files are restricted to notices/<filename> so this option cannot bypass the public profile.

scripts/runtime-prune-policy.json is the explicit exclusion policy. package-exclusions.json records excluded path, bytes, reason, rule, logical source and SHA-256, including present commercial input omissions in public builds. In-package reports use logical component/relative paths; complete absolute provenance is written to the sibling <output>.audit directory. That directory also retains original local direct_url metadata and unmodified copies of provenance text sanitized for public distribution. Vendor source files are untouched. package-inputs.json binds the policy, requirements lock and selection proof; package-inventory.json hashes every copied file. Build libraries/symbols, specified headers, test/example trees and development hooks are pruned. Whole non-runtime cmake/pytest/scikit_build_core/nanobind/python-devtools/pip/wheel payloads and compiler executables are excluded, while their metadata/licenses remain for audit. setuptools and tomli stay because cpp_extension imports them; ninja Python wrappers/metadata stay for gsplat, without ninja executables. Runtime DLL/PYD files, resources, package metadata/licenses and torch.utils.cpp_extension.py remain; retired Python Qt/gsdb/editable/trainer artifacts are excluded. Copying refuses reparse points, source/output overlap, duplicate destinations and existing output. Failed partial outputs and their audit remain for inspection; retry into a new directory.

All launchers resolve runtime JSON in this order: GSSTUDIO_RUNTIME, compatibility LOCAL_POSTSHOT_RUNTIME, then runtime/runtime.json beside the application. Public releases have no machine-specific or repository fallback. Data defaults use the ${user_data_root} token, resolved to LOCALAPPDATA/GSStudio. Projects, mutable cache, logs and the ONE cross-process GPU lock default to projects, cache, logs and cache/gpu.lock beneath that root. Changing GSSTUDIO_DATA_ROOT changes projects only, preserving the shared lock.

User overrides live outside the package at LOCALAPPDATA/GSStudio/runtime.override.json or the file selected by GSSTUDIO_RUNTIME_OVERRIDE. Each file's relative paths resolve against its own directory. Explicit path environment variables have highest precedence; saved schema-1 settings.json data_root (GSSTUDIO_SETTINGS_DIR supported) precedes the runtime data_root. Explicit local runtime configuration can retain existing project and shared-lock locations. To configure commercial dependencies without modifying the public package:

~~~json
{
  "media_helper": "F:/SDK/Insta360/bin/gsstudio-media-helper.exe",
  "mediasdk_root": "F:/SDK/Insta360",
  "realityscan": "C:/Program Files/Epic Games/RealityScan_2.2/RealityScan.exe",
  "realityscan_export_params": "F:/GSStudio/colmap-export-params.xml"
}
~~~

GSSTUDIO_MEDIA_HELPER, INSTA360_MEDIA_SDK_ROOT, GSSTUDIO_REALITYSCAN_CLI and GSSTUDIO_REALITYSCAN_EXPORT_PARAMS override these fields. TORCH_HOME and cache/lock overrides also retain explicit environment precedence across Rust, CLI, pipeline and engine children. The supervisor preserves those explicit configuration variables while isolating Python paths and owning descendants in a Windows JobObject.

Release entry points check required native binary files before imports; gsplat's backend gets a load check before its ImportError-to-JIT branch. Runtime JIT and Torch Hub downloads fail with a package repair diagnostic. Cached offline weights still load. No numerical kernels/training formulas or local_postshot module names change. Release validation must independently verify the final provisioned package, model hashes, imports, relocation and real GPU pipeline; synthetic CPU package tests do not establish GPU acceptance. The pruned release package remains pending actual validation.

The frontend retains mask/QA review, immutable training identity, checkpoint controls and paired previews. GS Studio does not approve a Run or start training during packaging or tests.


Provisioning sources are [scripts/provision_runtime.py](../scripts/provision_runtime.py) and [scripts/runtime/](../scripts/runtime/) at the repository root: core-constraints.txt freezes the 44 baseline distributions; runtime-requirements.lock.txt contains 41 runtime/import distributions, build-test-requirements.lock.txt the 11 build/test distributions, and unified-requirements.lock.txt includes both (52). pipeline-additions.lock.txt and pipeline-wheels.json bind exactly eight added wheels to the recorded SHA-256 installation receipt. Packaging also accepts an equivalent existing flat 52-package lock.

Verify an existing provisioned environment without modifying it, from the repository root:

~~~powershell
$RuntimePython = (Resolve-Path '<existing-runtime>/python/python.exe').Path
& $RuntimePython -I -B .\scripts\provision_runtime.py verify --python $RuntimePython --kind unified
if ($LASTEXITCODE -ne 0) { throw 'runtime verification failed' }
~~~

For future reproduction, use plan with --core-root <frozen-core44>, --output <new-python-directory>, and --wheelhouse <eight-receipted-wheels>. The provision subcommand accepts the same arguments, clones the frozen core into a nonexistent directory, verifies copied native binary hashes, and installs only those eight hash-locked wheels with --no-index/--no-deps/--require-hashes, then checks the exact 52 versions and pip check. It refuses existing environments/evidence, reparse points and source/output overlap. It never rebuilds native extensions or changes the source core. Source CI does not execute provisioning or create the application package.
