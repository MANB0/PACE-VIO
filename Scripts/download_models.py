#!/usr/bin/env python3
"""Download and verify the only pretrained model used by PACE-VIO."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_FrontendCov.pth"
EXPECTED_SHA256 = "bec6edd7e195bab863132f1e9659cdd26e6eaeae7cfd24a626828de294cf5b3a"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Model/MACVO_FrontendCov.pth")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.is_file() and not args.force:
        current = digest(output)
        if current == EXPECTED_SHA256:
            print(f"Model already verified: {output}")
            return 0
        raise RuntimeError(
            f"Existing model has SHA-256 {current}, expected {EXPECTED_SHA256}. "
            "Use --force to replace it."
        )

    temporary = output.with_suffix(output.suffix + ".download")
    print(f"Downloading {URL}")
    try:
        with urllib.request.urlopen(URL) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        current = digest(temporary)
        if current != EXPECTED_SHA256:
            raise RuntimeError(f"Downloaded model SHA-256 {current}, expected {EXPECTED_SHA256}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Model verified: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
