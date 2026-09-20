#!/usr/bin/env python3

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Verify legal files and native payloads in assembled release artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys
import zipfile


MAX_LEGAL_FILE_SIZE = 2 * 1024 * 1024
TARGETS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
)
JAVA_NATIVE_BY_TARGET = {
    "x86_64-unknown-linux-gnu": "native/linux/x86_64/libpaimon_mosaic_jni.so",
    "aarch64-unknown-linux-gnu": "native/linux/aarch64/libpaimon_mosaic_jni.so",
    "aarch64-apple-darwin": "native/macos/aarch64/libpaimon_mosaic_jni.dylib",
    "x86_64-pc-windows-msvc": "native/windows/x86_64/paimon_mosaic_jni.dll",
}
PYTHON_NATIVE_BY_TARGET = {
    "x86_64-unknown-linux-gnu": "mosaic/libpaimon_mosaic_ffi.so",
    "aarch64-unknown-linux-gnu": "mosaic/libpaimon_mosaic_ffi.so",
    "aarch64-apple-darwin": "mosaic/libpaimon_mosaic_ffi.dylib",
    "x86_64-pc-windows-msvc": "mosaic/paimon_mosaic_ffi.dll",
}


def archive_entries(archive: zipfile.ZipFile) -> Counter[str]:
    return Counter(info.filename for info in archive.infolist())


def require_single(entries: Counter[str], artifact: Path, name: str) -> None:
    count = entries[name]
    if count != 1:
        raise ValueError(
            f"{artifact}: expected exactly one {name!r} entry, found {count}"
        )


def read_text(
    archive: zipfile.ZipFile,
    entries: Counter[str],
    artifact: Path,
    name: str,
) -> str:
    require_single(entries, artifact, name)
    info = archive.getinfo(name)
    if info.file_size > MAX_LEGAL_FILE_SIZE:
        raise ValueError(
            f"{artifact}: {name} is {info.file_size} bytes, "
            f"limit is {MAX_LEGAL_FILE_SIZE}"
        )
    try:
        return archive.read(info).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{artifact}: {name} is not UTF-8") from error


def verify_java_jar(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        entries = archive_entries(archive)
        license_text = read_text(archive, entries, path, "META-INF/LICENSE")
        notice_text = read_text(archive, entries, path, "META-INF/NOTICE")
        require_single(entries, path, "META-INF/DEPENDENCIES")
        if "Apache Paimon Mosaic" not in notice_text:
            raise ValueError(f"{path}: META-INF/NOTICE does not identify the project")

        for target in TARGETS:
            native = JAVA_NATIVE_BY_TARGET[target]
            report = f"META-INF/licenses/{target}/THIRD-PARTY-LICENSES.html"
            require_single(entries, path, native)
            report_text = read_text(archive, entries, path, report)
            if report not in license_text:
                raise ValueError(f"{path}: META-INF/LICENSE does not reference {report}")
            if target not in report_text:
                raise ValueError(f"{path}: {report} does not identify target {target}")


def python_wheel_tags(path: Path) -> tuple[str, str, str]:
    if not path.name.endswith(".whl"):
        raise ValueError(f"{path}: Python artifact is not a wheel")
    parts = path.name[:-4].rsplit("-", 3)
    if len(parts) != 4:
        raise ValueError(f"{path}: invalid wheel filename")
    _, python_tag, abi_tag, platform_tag = parts
    if python_tag != "py3" or abi_tag != "none":
        raise ValueError(
            f"{path}: expected py3-none wheel tags, "
            f"found {python_tag}-{abi_tag}"
        )
    return python_tag, abi_tag, platform_tag


def python_wheel_target(path: Path) -> str:
    _, _, platform_tag = python_wheel_tags(path)
    name = platform_tag.lower()
    patterns = (
        (r"(?:manylinux.*|linux)_x86_64$", "x86_64-unknown-linux-gnu"),
        (r"(?:manylinux.*|linux)_aarch64$", "aarch64-unknown-linux-gnu"),
        (r"macosx.*_arm64$", "aarch64-apple-darwin"),
        (r"win_amd64$", "x86_64-pc-windows-msvc"),
    )
    matches = [target for pattern, target in patterns if re.search(pattern, name)]
    if len(matches) != 1:
        raise ValueError(f"{path}: unsupported or ambiguous wheel platform tag")
    return matches[0]


def verify_python_wheel(path: Path) -> str:
    python_tag, abi_tag, platform_tag = python_wheel_tags(path)
    target = python_wheel_target(path)
    with zipfile.ZipFile(path) as archive:
        entries = archive_entries(archive)
        native = PYTHON_NATIVE_BY_TARGET[target]
        require_single(entries, path, native)
        license_text = read_text(archive, entries, path, "mosaic/LICENSE")
        notice_text = read_text(archive, entries, path, "mosaic/NOTICE")
        report_text = read_text(
            archive,
            entries,
            path,
            "mosaic/THIRD-PARTY-LICENSES.html",
        )
        if "THIRD-PARTY-LICENSES.html" not in license_text:
            raise ValueError(f"{path}: mosaic/LICENSE does not reference the report")
        if "Apache Paimon Mosaic" not in notice_text:
            raise ValueError(f"{path}: mosaic/NOTICE does not identify the project")
        if target not in report_text:
            raise ValueError(
                f"{path}: third-party report does not identify target {target}"
            )
        wheel_metadata = [
            name
            for name in entries
            if name.endswith(".dist-info/WHEEL")
        ]
        if len(wheel_metadata) != 1:
            raise ValueError(
                f"{path}: expected exactly one .dist-info/WHEEL entry, "
                f"found {len(wheel_metadata)}"
            )
        metadata_text = read_text(
            archive,
            entries,
            path,
            wheel_metadata[0],
        )
        tags = {
            line.removeprefix("Tag:").strip()
            for line in metadata_text.splitlines()
            if line.startswith("Tag:")
        }
        expected_tag = f"{python_tag}-{abi_tag}-{platform_tag}"
        if tags != {expected_tag}:
            raise ValueError(
                f"{path}: WHEEL tags are {sorted(tags)}, "
                f"expected only {expected_tag!r}"
            )
    return target


def verify_python_target_set(targets: list[str]) -> None:
    counts = Counter(targets)
    expected = Counter({target: 1 for target in TARGETS})
    if counts != expected:
        raise ValueError(
            "Python wheel target set is incomplete or duplicated: "
            f"found {dict(sorted(counts.items()))}, "
            f"expected {dict(expected)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("java", "python"))
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument(
        "--require-all-python-targets",
        action="store_true",
        help="require exactly one Python wheel for each supported release target",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        for artifact in args.artifacts:
            if not artifact.is_file():
                raise ValueError(f"release artifact does not exist: {artifact}")

        if args.kind == "java":
            if args.require_all_python_targets:
                raise ValueError(
                    "--require-all-python-targets is valid only for Python wheels"
                )
            for artifact in args.artifacts:
                verify_java_jar(artifact)
        else:
            targets = [verify_python_wheel(artifact) for artifact in args.artifacts]
            if args.require_all_python_targets:
                verify_python_target_set(targets)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(error, file=sys.stderr)
        return 1

    print(f"verified {len(args.artifacts)} {args.kind} release artifact(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
