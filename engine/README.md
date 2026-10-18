# GS Studio independent engine

`local_postshot` is GS Studio's internal Python engine package. The current
Tauri/React/Spark desktop and CLI share the application layer in `src/gsstudio`.
Preprocessing and training run in separate worker processes using one locked
Torch 2.8/cu129 Python environment. No original commercial Postshot binary,
resource, shader or plugin is a runtime dependency.

See the [engine contract](../docs/architecture/ENGINE-CONTRACT.md),
[source structure](../docs/development/STRUCTURE.md) and
[maintenance guide](../docs/development/MAINTENANCE.md).
Native compiler sources, useful synthetic utilities and current numerical tests
remain here. The quality_v2 and observed_v5 numerical baselines stay frozen;
observed_v6 is experimental. Actual GPU validation covers architecture 8.9 only.

Raw benchmark/RE evidence and one-off experiment scripts are archived outside the
public repository. Public aggregate results and their limits are described in
[benchmark methodology](../docs/development/BENCHMARKS.md), with original file
hashes in the [retirement manifest](../docs/archive/source-retirement-20261017.json).
See [third-party notices](../THIRD_PARTY_NOTICES.md) for the licenses that apply
to native derivatives and distributed dependencies.
