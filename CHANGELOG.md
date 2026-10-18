# Changelog

## v0.3.0-rc.1 — source baseline (2026-10-18)

The [GS-Studio repository](https://github.com/Astroite/GS-Studio) starts with a reviewed source snapshot and a new root commit. This baseline is **source only, with no binary attachments**; previous Git ancestry, LFS objects, user media, trained models and runtime payloads are excluded. Directory-package instructions describe locally built or private packages.

- Unify the Tauri + React + Spark desktop and CLI around the same Capture/Run/QA/training operations.
- Use one locked Python 3.10.22 / Torch 2.8/cu129 environment with separate preprocessing, training and editing workers. Retain the frozen 44-distribution core and eight hash-locked additions, with offline provisioning into a new environment and a combined 52-distribution development lock.
- Make the public package exclude the commercial Insta360 SDK and dependent helper binaries; users configure their licensed installation separately. RealityScan remains separately installed.
- Prepare own code under Apache-2.0 while retaining third-party licenses and packaging notices; this does not relicense dependencies, including GPL `plyfile`.
- Replace superseded operating/build documentation, archive retired tests/prototypes/experiments and raw evidence with SHA256 provenance, and add hosted source/CPU/frontend/Rust CI.
- Retire the Qt frontend after confirming no external production importers, archive Qt-only tests, and retain useful non-Qt worker and numerical validation. Keep COLMAP 4.2.1 and its legitimate Qt DLL dependencies intact.
- Add repository-local source guards and CI checks for accidental media, model, checkpoint, runtime and LFS-pointer uploads. Keep exact path/hash exceptions for the small synthetic fixtures required by source tests.
- Organize current user, maintainer, architecture, distribution and historical evidence documents, with one current workflow and separate historical acceptance records.

quality_v2 and observed_v5 numerical behavior stays frozen; observed_v6 is experimental. Panorama quality remains low in representative scenes, SPZ is lossy, and GPU architecture 8.9 is the only validated architecture. Old Run/checkpoint formats and `.psht` are unsupported.

The [migration record](docs/development/PIPELINE-MIGRATION.md), [runtime slimming record](docs/development/RUNTIME-SLIMMING.md) and [historical benchmark summary](docs/development/BENCHMARKS.md) retain earlier package identities, test counts and numerical results. These are historical evidence, not newly executed acceptance of the fresh source snapshot. Source CI does not replace CUDA, real-media, full-budget training or desktop acceptance.

Binary publication remains blocked by unresolved [third-party distribution evidence](docs/distribution/third-party-sources.json). Source publication and runtime/package acceptance do not establish redistribution rights for bundled dependencies.
