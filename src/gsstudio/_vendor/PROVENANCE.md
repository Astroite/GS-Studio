# Minimal Nerfstudio / COLMAP excerpts

The accepted source is the **installed, locally patched Nerfstudio 1.1.5**,
identified relative to its distribution as `nerfstudio==1.1.5:nerfstudio/`.
Its distribution records upstream commit
`758ea1918e082aa44776009d8e755c2f3a88d2ee` from
<https://github.com/nerfstudio-project/nerfstudio>.
The installed bytes, rather than a fresh upstream checkout, are authoritative.
`provenance.json` records their SHA256, retained symbols and line ranges,
normalized excerpt hashes, vendored file hashes and license hashes.

| Local module | Original module | License |
| --- | --- | --- |
| `nerfstudio/colmap_parsing_utils.py` | `data/utils/colmap_parsing_utils.py` | COLMAP BSD-3-Clause |
| `nerfstudio/equirect_utils.py` | `process_data/equirect_utils.py` | Apache-2.0 |
| `nerfstudio/colmap_utils.py` | `process_data/colmap_utils.py` | Apache-2.0 |
| `nerfstudio/camera_model.py` | `process_data/process_data_utils.py`, `CameraModel` only | Apache-2.0 |

The parser itself attributes COLMAP's
[read_write_model.py at 1a4d0bad2e90aa65ce997c9d1779518eaed998d5](https://github.com/colmap/colmap/blob/1a4d0bad2e90aa65ce997c9d1779518eaed998d5/scripts/python/read_write_model.py).
Its entire attribution, copyright, BSD conditions, disclaimer and author header
are retained in the module and `LICENSE-COLMAP-BSD.txt`. The Apache headers are
retained in their modules; the installed distribution's complete license is
copied without modification to `LICENSE-APACHE-2.0.txt`.

## Changes on 2026-10-17

Only the function, class and data closure called by GS Studio is retained.
Function bodies and signatures are verbatim, including quaternion arithmetic,
all eleven camera model IDs, namedtuple fields, distortion handling, unsupported
model errors, camera/world axis conversions, JSON fields and ASCII PLY output.
The baseline's existing `grid_sample(...).clamp(0.0, 1.0)` and its default
`align_corners` behavior are preserved. No projection correction is introduced.

Internal imports point to local sibling modules. Unused commands, downloads,
depth-generation tools, image processing drivers and heavyweight imports are
omitted. `CONSOLE = Console(width=120)` is constructed directly, exactly as in
the baseline's `utils/rich_utils.py`; that source hash is also recorded.
There is no top-level `nerfstudio` compatibility package.

The retained closure uses only Python's standard library, NumPy, Torch and Rich.
The surrounding GS Studio pipeline continues to use OpenCV and Pillow.
Nerfstudio and its transitive process-data dependencies are unnecessary here.
Package builders must include this directory's Markdown, JSON and license text
alongside the Python modules (the setuptools package-data entry is owned by the
parent migration task).

## Independent CPU baseline

`tests/contract/nerfstudio_baseline.py` is an explicit fixture generator, never
imported by the parity tests. It imports the original installed modules in the
historical baseline Python distribution and writes synthetic COLMAP text/binary models,
projection inputs/results, parser results, accepted/rejected camera metadata,
poses, transforms JSON (mask/depth/rename variants) and PLY outputs. It does not
import this vendor package. Run it with `-B` and `CUDA_VISIBLE_DEVICES=-1`.

The original baseline evidence remains in the external migration archive.
Its distribution-independent, small synthetic fixture snapshot is maintained under
`tests/fixtures/nerfstudio_vendor`, with generator/source/output hashes and
Python/Torch/NumPy identities in `baseline.json`. Tests compare imports from
`gsstudio._vendor.nerfstudio` against those independent, frozen results. They
also audit retained excerpt hashes and block all top-level Nerfstudio imports
in a fresh CPU process. Production images, models and masks are never inputs.
