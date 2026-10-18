# GSSTUDIO MediaSDK helper

The Insta360 SDK remains an external, licensed installation. Do not copy its
headers, DLLs, import libraries, models, license files, or built helper binaries
into this repository. Keep all vendor material under `INSTA360_MEDIA_SDK_ROOT`;
the repository's `build/` and `vendor/` paths are ignored as an additional guard.

The checked-in C++ bridge implements capabilities, metadata probing and frame
export against MediaSDK and InsMetaDataSDK. It must be built against the locally
installed SDK with `build.ps1`; the native workflow has exercised X6 decoding.
Support for a particular source is determined by the helper capabilities and
probe response, not by a model name in this document. Validate a new SDK/camera
combination with `tests/manual/test-mediasdk-helper.ps1` before production use.
`tests/manual/fake-helper.py` is only the CI seam; it never parses INSV. Python
helpers are rejected by default; tests deliberately set
`GSSTUDIO_ALLOW_FAKE_MEDIA_HELPER=1`. Test scripts live under `tests/`; test
byproducts never enter this repository.

The user workflow is documented in [the current manual](../../docs/user/CURRENT-WORKFLOW.md).
The JSON examples below illustrate the protocol and are not a complete camera
support list. Build outputs and locally copied runtime DLLs remain ignored by Git.

## Invocation and common rules

The Windows x64 executable is invoked once per request:

```text
gsstudio-media-helper.exe <request-json> <response-json>
```

Both files are UTF-8 JSON. Protocol version 1 has exactly three operations:
`capabilities`, `probe`, and `export_frames`. Source files, requested outputs,
and both protocol files must be absolute drive-letter Windows paths. Reject UNC
paths and anything backed by WSL ext4. Frame indices are zero-based source-frame
indices; their timestamps are the source PTS convention `index / fps`.

Every response contains `schema_version`, `ok`, `helper_version`, `sdk_version`,
and `supported_camera_models`. On failure, write the response before exiting
non-zero:

```json
{
  "schema_version": 1,
  "ok": false,
  "helper_version": "0.2.0",
  "sdk_version": "approved-sdk-version",
  "supported_camera_models": ["Insta360 X5"],
  "error": {"code": "unsupported_camera", "message": "camera is not enabled"}
}
```

## `capabilities`

Request:

```json
{"schema_version": 1, "operation": "capabilities"}
```

Successful response:

```json
{
  "schema_version": 1,
  "ok": true,
  "helper_version": "0.2.0",
  "sdk_version": "approved-sdk-version",
  "supported_camera_models": ["Insta360 X5"]
}
```

## `probe`

`source_files` contains exactly the one or two INSV files recorded by the
capture. Never infer a `_10_` partner.

```json
{
  "schema_version": 1,
  "operation": "probe",
  "source_files": [
    "D:\\Media\\VID_001_00.insv",
    "D:\\Media\\VID_001_10.insv"
  ]
}
```

The successful response adds a `media` object. The five numeric fields below
are required; `codec` is optional.

```json
{
  "schema_version": 1,
  "ok": true,
  "helper_version": "0.2.0",
  "sdk_version": "approved-sdk-version",
  "supported_camera_models": ["Insta360 X5"],
  "media": {
    "width": 7680,
    "height": 3840,
    "fps": 29.97,
    "frame_count": 8991,
    "duration_seconds": 300.0,
    "codec": "insv",
    "camera_model": "Insta360 X5"
  }
}
```

## `export_frames`

Only export the listed candidates. `output_file` is the final filename within
GSSTUDIO's `.building` directory, not a filename template.

```json
{
  "schema_version": 1,
  "operation": "export_frames",
  "source_files": ["D:\\Media\\VID_001_00.insv"],
  "frames": [
    {
      "source_frame_index": 16,
      "output_file": "C:\\Work\\projects\\locations\\site\\scenes\\scene\\inputs\\prepared\\capture\\hash.building\\frames\\frame_000001.jpg"
    }
  ],
  "output": {
    "width": 7680,
    "height": 3840,
    "format": "jpeg",
    "jpeg_quality": 95
  },
  "stitching": {
    "mode": "dynamic",
    "flowstate": true,
    "direction_lock": false,
    "ai_denoise": false,
    "sharpening": false,
    "color_processing": false
  }
}
```

Use MediaSDK `SetExportFrameSequence` together with `SetImageSequenceInfo`.
Exporting every frame and discarding most of them is not accepted. If the SDK
chooses temporary sequence names, map each result back to its requested
`output_file` before returning. Every result must be a decodable, exact-size 2:1
JPEG. Return only after files are closed and durable, with
`"exported_count"` equal to the number of requested frames:

```json
{
  "schema_version": 1,
  "ok": true,
  "helper_version": "0.2.0",
  "sdk_version": "approved-sdk-version",
  "supported_camera_models": ["Insta360 X5"],
  "exported_count": 1
}
```

GSSTUDIO owns directory-level resume, per-frame decode/hash checks, `dataset.json`,
and the atomic `.building` rename. The helper must not modify source files or
emit unrequested frames.
