"""Execute an unchanged setup.py with bounded, source-relative object names.

Setuptools derives object paths from extension source names. Absolute source
names can repeat the complete workspace beneath build/temp on Windows. This
offline wrapper changes only ext.sources, after checking every source belongs
to the explicit staged source directory; flags and include paths stay intact.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import stat
import sys


def plain(path):
    path = Path(os.path.abspath(path))
    for item in (path, *path.parents):
        if item.is_symlink() or (item.exists() and getattr(item.lstat(), "st_file_attributes", 0)
                                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError(f"Source symlink/reparse point rejected: {item}")
    return path


def normalize_extensions(extensions, source_root):
    root = plain(source_root)
    for extension in extensions:
        relative_sources = []
        for value in extension.sources:
            path = plain(Path(value) if Path(value).is_absolute() else root / value)
            if not path.is_relative_to(root):
                raise ValueError(f"Extension source escapes staged directory: {value}")
            if not path.is_file():
                raise ValueError(f"Extension source missing: {value}")
            relative_sources.append(path.relative_to(root).as_posix())
        extension.sources = relative_sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("setup_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    root = plain(args.source_root)
    setup_path = root / "setup.py"
    if not setup_path.is_file():
        raise ValueError(f"Staged setup.py missing: {setup_path}")
    arguments = args.setup_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    import setuptools
    original_setup = setuptools.setup
    original_argv = sys.argv
    original_cwd = Path.cwd()

    def relative_setup(*positional, **keywords):
        normalize_extensions(keywords.get("ext_modules", []), root)
        return original_setup(*positional, **keywords)

    try:
        setuptools.setup = relative_setup
        sys.argv = [str(setup_path), *arguments]
        os.chdir(root)
        runpy.run_path(str(setup_path), run_name="__main__")
    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv
        setuptools.setup = original_setup
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
