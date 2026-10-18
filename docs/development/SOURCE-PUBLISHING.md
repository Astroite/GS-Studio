# Source-only publication

Git is used for source, documentation, dependency locks, licenses and reviewed small synthetic fixtures. User video/photo collections, captures, Runs, reconstructions, trained models, checkpoints, environments, SDK payloads, credentials and caches belong outside the repository. Git LFS is not a media publishing channel for this project. Binary Release attachments remain a separately authorized distribution step.

## Local checks

Install the tracked guards once in each clone using an existing Python 3.10+ executable:

```powershell
.\scripts\install-source-guards.ps1 -PythonPath (Get-Command python).Source
python -I -B scripts/check_source_payload.py staged
python -I -B scripts/check_source_payload.py history --all
python -I -B -m unittest discover -s tests/source_guard -v
```

The installer saves previous repository-local settings, the effective hooks setting and hashes/copies of existing hook files under `.git/source-guard-backups/`. It leaves those original files in place and sets repository-local `core.hooksPath=.githooks` and `gsstudio.guardPython`. No global Python/PATH, global Git setting, media, model or LFS object is changed. Inspect the private `settings.json` receipt to restore previous local values if needed; an absent previous local value means unset that local key. Previous hooks are preserved for review, not automatically chained into source-only pushes.

The pre-commit check reads every stage-0 index blob. A clean working file cannot hide a staged LFS pointer, and an unstaged change cannot replace safe staged content. Unmerged entries, symlinks and submodules fail in the index and commit trees. Files ignored by `.gitignore` are still inspected if force-added.

The pre-push check reads Git's ref-update input, permits publishing only `refs/heads/*` and `refs/tags/*`, and checks all reachable ancestor commit trees. It checks every path/blob combination, including a blob reused under a different filename. It does not subtract ancestors already advertised by the server, so merging contaminated old history is rejected. Ref deletions upload no objects and are permitted. Annotated tags must resolve to commits. Private refs such as `refs/codex/checkpoints/*` must never be published. The hooks do not call `git-lfs pre-push`.

All modes reject video/model/checkpoint/binary and raster image suffixes without regard to case, oversized blobs above 5 MiB, LFS pointer content regardless of filename, and recognized renamed executable/archive/video/PLY signatures. Staged and current published tip trees additionally reject local data/runtime/output directory names and environment files. Older ancestors may retain small legacy metadata under former data paths; this exception does not permit media, LFS pointers, binaries or oversized blobs. [The reviewed asset manifest](../../scripts/source-fixtures.json) allows only named synthetic upstream compatibility fixtures and the application icon with exact SHA256 values, including recorded LF/CRLF variants of a three-point ASCII PLY text fixture. A changed or renamed binary does not inherit that exception. Future raster source assets require an explicit reviewed path/hash entry; small size alone does not exempt a photo. SVG and code-native assets remain source. Video/model suffixes, LFS pointers and the size ceiling cannot be exempted by an asset entry. The scanner is not a general content classification or secret scanning service.

History checks require complete history. Shallow repositories and grafts fail closed, and replace objects are ignored. The check reads Git objects with ordinary Git plumbing, does not perform LFS fetches, does not import the application and needs no runtime, model, GPU or network access. Missing Python, a missing scanner, a Git error or malformed input stops the operation.

To audit an isolated bare mirror without changing a working checkout:

```powershell
python -I -B scripts/check_source_payload.py --repo <mirror-path> history --all
```

## Hosted check and limits

The [source workflow](../../.github/workflows/source-checks.yml) fetches complete history with LFS disabled, checks HEAD plus fetched branch/tag history and runs isolated stdlib guard regressions. CI does not install hooks in contributors' clones. Local hooks can be bypassed by a caller, and hosted CI runs after Git objects reach the server; neither is a server-side deletion mechanism. Keep branch/release review in addition to these checks and never use `--no-verify` to publish data.

Deleting a file from the current tree is insufficient to remove it from old commits or Git LFS storage. A data exposure requires a separately backed-up, reviewed history rewrite, replacement of affected remote refs, and verification of provider-side retained objects/caches/LFS storage. GitHub-side retention or removal must be confirmed through supported provider mechanisms; a clean Git clone alone does not prove permanent server deletion. Preserve local originals and current source during that process. Other clones must synchronize with rewritten history without merging the old ancestors back.
