#!/usr/bin/env python3

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

"""Verify main and classifier JAR licensing matches their bundled content."""

from __future__ import annotations

import argparse
import posixpath
import stat
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from zipfile import ZipFile, ZipInfo

from native_binary import verify_native_target


TARGETS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
)
NATIVE_ENTRIES = {
    "native/linux/x86_64/libpaimon_mosaic_jni.so": "x86_64-unknown-linux-gnu",
    "native/linux/aarch64/libpaimon_mosaic_jni.so": "aarch64-unknown-linux-gnu",
    "native/macos/aarch64/libpaimon_mosaic_jni.dylib": "aarch64-apple-darwin",
    "native/windows/x86_64/paimon_mosaic_jni.dll": "x86_64-pc-windows-msvc",
}
NESTED_LICENSE_MARKERS = (
    "For Zstandard software",
    "Apache Arrow",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def validated_entries(archive: ZipFile) -> dict[str, ZipInfo]:
    entries: dict[str, ZipInfo] = {}
    normalized_names: dict[str, str] = {}
    for info in archive.infolist():
        name = info.filename
        if "\\" in name:
            raise ValueError(f"archive entry uses a backslash: {name!r}")
        if PurePosixPath(name).is_absolute() or PureWindowsPath(name).is_absolute():
            raise ValueError(f"archive entry uses an absolute path: {name!r}")
        if ".." in name.split("/"):
            raise ValueError(f"archive entry uses a '..' path component: {name!r}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"archive entry is a symbolic link: {name!r}")
        if name in entries:
            raise ValueError(f"archive contains duplicate raw entry name: {name!r}")

        normalized_name = posixpath.normpath(name)
        previous_name = normalized_names.get(normalized_name)
        if previous_name is not None:
            raise ValueError(
                "archive contains duplicate normalized entry names: "
                f"{previous_name!r} and {name!r}"
            )

        entries[name] = info
        normalized_names[normalized_name] = name
    return entries


def verify_main_jar(path: Path, root: Path, require_all_natives: bool) -> None:
    binary_resources = root / "java/src/main/binary-resources/META-INF"
    with ZipFile(path) as archive:
        entries = validated_entries(archive)
        required = {"META-INF/LICENSE", "META-INF/NOTICE"}
        required.update(
            f"META-INF/licenses/{target}/THIRD-PARTY-LICENSES.html"
            for target in TARGETS
        )
        missing = sorted(required - entries.keys())
        if missing:
            raise ValueError(f"missing legal files: {missing}")
        if "META-INF/DEPENDENCIES.rust.tsv" in entries:
            raise ValueError(
                "main JAR contains the cross-target repository dependency inventory"
            )

        expected_license = (binary_resources / "LICENSE").read_bytes()
        if archive.read(entries["META-INF/LICENSE"]) != expected_license:
            raise ValueError("main JAR LICENSE is not the binary-specific LICENSE")
        license_text = expected_license.decode("utf-8")
        expected_notice = (binary_resources / "NOTICE").read_bytes()
        if archive.read(entries["META-INF/NOTICE"]) != expected_notice:
            raise ValueError("main JAR NOTICE is not the binary-specific NOTICE")
        if b"Apache Arrow" not in expected_notice:
            raise ValueError("main JAR NOTICE omits the bundled Apache Arrow notice")

        for target in TARGETS:
            report_path = f"META-INF/licenses/{target}/THIRD-PARTY-LICENSES.html"
            if report_path not in license_text:
                raise ValueError(f"LICENSE does not point to {report_path}")
            expected_report = (
                binary_resources
                / "licenses"
                / target
                / "THIRD-PARTY-LICENSES.html"
            ).read_bytes()
            actual_report = archive.read(entries[report_path])
            if actual_report != expected_report:
                raise ValueError(f"{report_path} differs from its generated source")

            report_text = actual_report.decode("utf-8")
            if target not in report_text:
                raise ValueError(f"{report_path} does not identify its target")
            for marker in NESTED_LICENSE_MARKERS:
                if marker not in report_text:
                    raise ValueError(f"{report_path} is missing {marker!r}")

        packaged_natives = {
            name
            for name in entries
            if name.startswith("native/") and not name.endswith("/")
        }
        unexpected_natives = packaged_natives - set(NATIVE_ENTRIES)
        if unexpected_natives:
            raise ValueError(f"unexpected native entries: {sorted(unexpected_natives)}")
        if require_all_natives and packaged_natives != set(NATIVE_ENTRIES):
            raise ValueError(
                "release JAR native entries differ from the four declared targets: "
                + repr(sorted(packaged_natives))
            )
        for native_entry in packaged_natives:
            verify_native_target(
                archive.read(entries[native_entry]),
                NATIVE_ENTRIES[native_entry],
                native_entry,
            )

    print(f"verified main JAR: {path}")


def verify_classifier(path: Path, root: Path | None = None) -> None:
    if root is None:
        root = repository_root()
    with ZipFile(path) as archive:
        entries = validated_entries(archive)
        for required in ("META-INF/LICENSE", "META-INF/NOTICE"):
            if required not in entries:
                raise ValueError(f"missing {required}")

        expected_license = (root / "LICENSE").read_bytes()
        if archive.read(entries["META-INF/LICENSE"]) != expected_license:
            raise ValueError("classifier LICENSE differs from repository root LICENSE")
        expected_notice = (root / "NOTICE").read_bytes()
        if archive.read(entries["META-INF/NOTICE"]) != expected_notice:
            raise ValueError("classifier NOTICE differs from repository root NOTICE")

        forbidden = sorted(
            name
            for name in entries
            if name.startswith("native/")
            or name == "META-INF/DEPENDENCIES.rust.tsv"
            or name.endswith("/THIRD-PARTY-LICENSES.html")
        )
        if forbidden:
            raise ValueError(f"classifier contains binary-only files: {forbidden}")

    print(f"verified classifier JAR: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", required=True, type=Path)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--javadoc", required=True, type=Path)
    parser.add_argument("--require-all-natives", action="store_true")
    args = parser.parse_args()
    root = repository_root()

    try:
        verify_main_jar(args.main, root, args.require_all_natives)
        verify_classifier(args.sources, root)
        verify_classifier(args.javadoc, root)
    except (KeyError, OSError, ValueError) as error:
        print(f"Java artifact verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
