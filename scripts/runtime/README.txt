Frozen Windows Python 3.10.22 dependencies for GS Studio 0.3.0rc1.

- core-constraints.txt: the parent's exact 44-package core version baseline.
- pipeline-additions.lock.txt: the eight additional wheels with SHA-256 hashes
  from dependency-install.json; no native core requirement is reinstalled.
- pipeline-wheels.json: exact offline wheel filenames and hashes.
- runtime-requirements.lock.txt: 41 runtime/import distributions.
- build-test-requirements.lock.txt: 11 build/test/installer distributions.
- unified-requirements.lock.txt: the combined 52-package development lock.

The runtime partition retains setuptools and tomli because the frozen
torch.utils.cpp_extension imports them, and ninja's Python wrapper/metadata
because gsplat declares it. Package pruning removes compiler executables and
development payloads while retaining licenses and distribution metadata.
The partition does not authorize a different Torch, Pillow, OpenCV, native
component, Python or COLMAP version.

Run scripts/provision_runtime.py verify against an existing environment without
installing anything. For future reproduction, plan/provision require an explicit
frozen core root, a nonexistent output directory and an offline wheelhouse
containing the eight exact receipted wheels. Provision copies the original core,
checks copied native binary hashes, installs only the eight wheels offline with
hashes and no dependency resolution, verifies all 52 exact versions and runs pip
check. Partial output is retained on failure. Source runtimes are never modified.
No CUDA module is imported by the provisioning helper.

This lock reproduces the supplied core plus the receipted additions; it does not
rebuild private native wheels from upstream sources. See desktop/README.md for
packaging and the sibling provenance audit.
