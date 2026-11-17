"""Check that distribution archives preserve the bundled runtime data.

Run from the repository root with Python 3.11 or later::

    uv build --python 3.12
    python tests/check_package_data.py dist/*.tar.gz dist/*.whl

The default ``uv build`` builds a source distribution and then builds a wheel
from it. This checker uses only the standard library and does not extract or
install the archives. Run it against freshly built artifacts after data changes.
"""

import argparse
import sys
import tarfile
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

RESOURCE_PATTERNS = (
    "biomni/know_how/*.md",
    "biomni/know_how/resource/*",
    "biomni/tool/protocols/*/*.txt",
)


def source_resources() -> dict[str, bytes]:
    """Read expected files from the checkout containing this script."""
    root = Path(__file__).resolve().parents[1]
    resources = {}
    for pattern in RESOURCE_PATTERNS:
        paths = sorted(path for path in root.glob(pattern) if path.is_file())
        if not paths:
            raise ValueError(f"No source files match {pattern}")
        resources.update((path.relative_to(root).as_posix(), path.read_bytes()) for path in paths)
    return resources


def archive_resources(path: Path, expected: dict[str, bytes]) -> dict[str, bytes]:
    """Read matching members directly from each archive."""
    if path.suffix == ".whl":
        with ZipFile(path) as archive:
            names = set(archive.namelist())
            return {name: archive.read(name) for name in expected if name in names}

    if path.name.endswith(".tar.gz"):
        resources = {}
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                # An sdist has a top-level project-version directory.
                name = PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
                if member.isfile() and name in expected:
                    with archive.extractfile(member) as stream:
                        resources[name] = stream.read()
        return resources

    raise ValueError(f"Unsupported artifact: {path} (expected .whl or .tar.gz)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path, help="Wheel or source distribution to check")
    args = parser.parse_args()
    try:
        expected = source_resources()
    except (OSError, ValueError) as error:
        parser.error(str(error))

    failed = False
    for path in args.artifacts:
        try:
            actual = archive_resources(path, expected)
        except (OSError, ValueError, BadZipFile, tarfile.TarError) as error:
            print(f"{path}: {error}", file=sys.stderr)
            failed = True
            continue
        errors = [
            f"{name}: {'missing' if name not in actual else 'content differs'}"
            for name, content in expected.items()
            if actual.get(name) != content
        ]
        if errors:
            failed = True
            for error in errors:
                print(f"{path}: {error}", file=sys.stderr)
        else:
            print(f"{path}: OK ({len(expected)} bundled data files)")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
