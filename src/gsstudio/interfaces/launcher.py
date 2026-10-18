"""Activation-ready native console wrapper; never imports the retired Qt UI."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess


def main(argv=None):
    parser = argparse.ArgumentParser(description='Open the accepted native GS Studio package')
    parser.add_argument('--app-root', type=Path,
                        default=Path(os.environ.get('GSSTUDIO_APP_ROOT') or Path(__file__).resolve().parents[3]))
    args = parser.parse_args(argv)
    root = args.app_root.resolve()
    executable, config = root/'GS Studio.exe', root/'runtime'/'runtime.json'
    if not executable.is_file() or not config.is_file():
        parser.error('GS Studio package is missing; set GSSTUDIO_APP_ROOT or --app-root to a complete directory package')
    env = os.environ.copy()
    env.setdefault('GSSTUDIO_RUNTIME', str(config))
    subprocess.Popen([str(executable)], cwd=root, env=env, close_fds=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
