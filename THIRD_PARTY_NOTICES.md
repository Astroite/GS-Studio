# Third-party notices

The root [Apache-2.0 license](LICENSE) applies to original GS Studio code.
It does not replace third-party copyright, license, attribution or binary
redistribution conditions. Preserve component licenses when modifying or
distributing those components.

## Source included in this repository

| Component | Source and retained terms |
| --- | --- |
| Modified gsplat rasterizer | [native renderer LICENSE](engine/native/splat3_render/LICENSE), [NOTICE](engine/native/splat3_render/NOTICE) and provenance in that directory; Apache-2.0 upstream code and identified local modifications |
| Modified gsplat projection | [projection LICENSE](engine/native/splat3_projection/LICENSE), [NOTICE](engine/native/splat3_projection/NOTICE), [GLM license](engine/native/splat3_projection/GLM-LICENSE.txt) and local provenance |
| Independent tile support / CUDA CUB use | [tile LICENSE](engine/native/splat3_tiles/LICENSE), [NOTICE](engine/native/splat3_tiles/NOTICE) and [CUB attribution text](engine/reference/THIRD_PARTY_NOTICES.md) |
| Vendored nerfstudio/COLMAP utilities | [provenance](src/gsstudio/_vendor/PROVENANCE.md), [Apache license](src/gsstudio/_vendor/LICENSE-APACHE-2.0.txt) and [COLMAP BSD terms](src/gsstudio/_vendor/LICENSE-COLMAP-BSD.txt); preserve upstream file notices |

The independently authored native statistics and editor helpers are project
code; their compiler/library dependencies retain their own terms. No original
commercial Postshot executable, DLL, shader, resource or plugin is included or
required by the implementation.

## Dependencies supplied with a directory package

Python/PSF, PyTorch/torchvision, NVIDIA CUDA runtime, gsplat, fused-ssim,
SPZ, COLMAP and its dependencies, FFmpeg, Tauri, React, Three.js, Spark and
other transitive packages have separate terms. Exact versions come from the
Python runtime lock/provenance, `desktop/package-lock.json` and
`desktop/src-tauri/Cargo.lock`, not from this summary.

The actual FFmpeg build determines its license and source obligations; do not
infer an LGPL-only build from the program name. Preserve its build information,
full license and any required source offer/access information. Apply the same
component-specific review to native tools, model weights and CUDA libraries.

`desktop/scripts/generate-notices.mjs` collects installed frontend dependency
licenses into `desktop/frontend-notices.txt`. The current packager copies that
file and declared `package.notice_files`; the copied runtime retains dependency
metadata/licenses. Supply native module LICENSE/NOTICE texts as explicit notice
inputs when not already carried by their installed distributions. A package
review must verify the resulting notices
against the actual payload; this source notice does not certify a binary's
redistribution eligibility. See [Windows packaging](docs/development/WINDOWS-RELEASE.md).

## External licensed software

Insta360 MediaSDK, its headers, DLLs, libraries and models are external licensed
inputs. Public packages exclude the SDK and helper binaries dependent on it.
Users obtain their own licensed SDK and build/configure the project's bridge.
RealityScan is installed separately and remains subject to its own terms.
The project's Apache license grants no rights to those products or user media.
