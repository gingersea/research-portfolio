"""Restore losslessly archived results; no model downloads or inference."""
from pathlib import Path, PurePosixPath
import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(name):
    return json.loads((ROOT / "provenance" / name).read_text())


def target_for(relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative:
        raise ValueError(f"Unsafe member path: {relative}")
    path = ROOT.joinpath(*p.parts)
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path escapes checkout: {relative}")
    return path


def restore_stream(stream, target, expected):
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f"Refusing to overwrite different file: {target}")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        temp = Path(temporary.name)
        try:
            shutil.copyfileobj(stream, temporary, 4 * 1024 * 1024)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        if digest(temp) != expected:
            raise ValueError(f"Restored hash mismatch: {target}")
        # A hard link is exclusive: an existing destination is never replaced.
        target.hardlink_to(temp)
    finally:
        temp.unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, help="Directory of downloaded release ZIP assets")
    args = parser.parse_args()
    count = 0
    for item in load("compressed-results.json"):
        source = target_for(item["stored_path"])
        if digest(source) != item["stored_sha256"]:
            raise ValueError(f"Compressed hash mismatch: {source}")
        with gzip.open(source, "rb") as stream:
            count += restore_stream(stream, target_for(item["original_path"]), item["source_sha256"])
    if args.assets:
        files = load("release-files.json")
        for asset in load("release-assets.json"):
            archive_path = args.assets / asset["name"]
            if digest(archive_path) != asset["sha256"]:
                raise ValueError(f"Archive hash mismatch: {archive_path}")
            expected = {f["path"]: f["sha256"] for f in files if f["release_asset"] == asset["name"]}
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != set(expected):
                    raise ValueError(f"Unexpected/duplicate members: {archive_path}")
                for name in names:
                    with archive.open(name) as stream:
                        count += restore_stream(stream, target_for(name), expected[name])
    print(f"Restored {count} files; existing identical files were preserved.")


if __name__ == "__main__":
    main()
