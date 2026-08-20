#!/usr/bin/env python3

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Validate ASF license headers on tracked text files."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ASF_HEADER = """
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements. See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to You under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
EXCLUDED_DIRECTORIES = {
    ".git",
    ".idea",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "target",
}

EXEMPT_FILES = {
    "Cargo.lock",
    # Generated dependency license reports.
    "DEPENDENCIES.rust.tsv",
    "cli/DEPENDENCIES.rust.tsv",
    "core/DEPENDENCIES.rust.tsv",
    "ffi/DEPENDENCIES.rust.tsv",
    "jni/DEPENDENCIES.rust.tsv",
    "LICENSE",
    "NOTICE",
    "core/LICENSE",
    "core/NOTICE",
}


def is_generated_legal_file(file_name: str) -> bool:
    if file_name in {
        "java/src/main/binary-resources/META-INF/LICENSE",
        "java/src/main/binary-resources/META-INF/NOTICE",
    }:
        return True
    if file_name.startswith(
        "java/src/main/binary-resources/META-INF/licenses/"
    ) and file_name.endswith("/THIRD-PARTY-LICENSES.html"):
        return True
    if file_name.startswith("python/licenses/") and file_name.rsplit("/", 1)[-1] in {
        "LICENSE",
        "NOTICE",
        "THIRD-PARTY-LICENSES.html",
    }:
        return True
    return False


def repo_root() -> Path:
    """Return the source root both in a Git checkout and an extracted archive."""

    return Path(__file__).resolve().parent.parent


def is_git_root(root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
    )
    return (
        result.returncode == 0
        and Path(result.stdout.strip()).resolve() == root.resolve()
    )


def tracked_files(root: Path) -> list[str]:
    if is_git_root(root):
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.splitlines()

    # ASF source archives intentionally do not include .git. Fall back to all
    # source-tree files while excluding only well-known local build outputs.
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not any(
            part in EXCLUDED_DIRECTORIES
            for part in path.relative_to(root).parts
        )
    )


def is_text_file(path: Path) -> bool:
    return b"\0" not in path.read_bytes()


def normalized_license_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def has_asf_header(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = normalized_license_text("\n".join(lines[:80]))
    return normalized_license_text(ASF_HEADER) in header


def missing_headers(root: Path) -> list[str]:
    missing = []

    for file_name in tracked_files(root):
        if file_name in EXEMPT_FILES or is_generated_legal_file(file_name):
            continue

        path = root / file_name
        if not path.is_file() or not is_text_file(path):
            continue

        if not has_asf_header(path):
            missing.append(file_name)

    return missing


def main() -> int:
    root = repo_root()
    missing = missing_headers(root)

    if missing:
        print("Files missing ASF license headers:", file=sys.stderr)
        for file_name in missing:
            print(f"  {file_name}", file=sys.stderr)
        return 1

    print("All tracked text files have ASF license headers or are explicitly exempt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
