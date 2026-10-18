#!/bin/sh
set -eu
repo=$(git rev-parse --show-toplevel) || exit 2
scanner="$repo/scripts/check_source_payload.py"
if [ ! -f "$scanner" ]; then
    echo 'Source-only guard missing; commit/push stopped.' >&2
    exit 2
fi
if guard_python=$(git config --local --get gsstudio.guardPython); then
    if [ -z "$guard_python" ]; then
        echo 'Configured source guard Python is empty; stopped.' >&2
        exit 2
    fi
    exec "$guard_python" -I -B "$scanner" "$@"
fi
for guard_python in python3 python; do
    if command -v "$guard_python" >/dev/null 2>&1 &&
       "$guard_python" -I -B -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
        exec "$guard_python" -I -B "$scanner" "$@"
    fi
done
if command -v py >/dev/null 2>&1 &&
   py -3 -I -B -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
    exec py -3 -I -B "$scanner" "$@"
fi
echo 'Python 3.10+ is required for the source-only guard; commit/push stopped.' >&2
exit 2
