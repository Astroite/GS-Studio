# Retired provisioning entrypoint. Never download/install into a guessed root.
# The complete application recipe is maintained in the repository-level builder.
$ErrorActionPreference = 'Stop'
$gsBootstrap = Join-Path $PSScriptRoot '../../scripts/runtime/bootstrap_windows.py'
Write-Error -ErrorAction Continue @"
This engine-only installer is retired. No environment was changed.
Use an explicit new workspace and the current unified Windows recipe:
  python -I -B -X utf8 "$gsBootstrap" plan --workspace <NEW_ABSOLUTE_WORKSPACE> --vcvars <VCVARSALL.BAT> --cuda-home <CUDA_12.9>
Then follow its prepare/build/verify stages. Existing runtimes and user data are not bootstrap inputs.
"@
exit 2
