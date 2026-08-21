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
NATIVE_SUFFIXES = (".so", ".dylib", ".dll")
JAVA_CLASS_MAGIC = b"\xca\xfe\xba\xbe"
MACHO_MAGICS = {
    JAVA_CLASS_MAGIC,
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
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
        name = info.orig_filename
        if not name or "\x00" in name or name != info.filename:
            raise ValueError(f"invalid archive entry path: {name!r}")
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


def read_java_u2(data: bytes, offset: int) -> tuple[int, int] | None:
    if offset > len(data) - 2:
        return None
    return int.from_bytes(data[offset : offset + 2], "big"), offset + 2


def skip_java_attributes(
    data: bytes, offset: int, count: int
) -> int | None:
    for _ in range(count):
        if offset > len(data) - 6:
            return None
        length = int.from_bytes(data[offset + 2 : offset + 6], "big")
        offset += 6
        if length > len(data) - offset:
            return None
        offset += length
    return offset


def skip_java_members(data: bytes, offset: int) -> int | None:
    result = read_java_u2(data, offset)
    if result is None:
        return None
    count, offset = result
    for _ in range(count):
        if offset > len(data) - 8:
            return None
        attributes_count = int.from_bytes(data[offset + 6 : offset + 8], "big")
        offset = skip_java_attributes(data, offset + 8, attributes_count)
        if offset is None:
            return None
    return offset


def is_java_class(data: bytes) -> bool:
    """Distinguish a complete Java class from Mach-O's shared CAFEBABE magic."""
    if len(data) < 10 or not data.startswith(JAVA_CLASS_MAGIC):
        return False
    major_version = int.from_bytes(data[6:8], "big")
    constant_pool_count = int.from_bytes(data[8:10], "big")
    if not 45 <= major_version <= 100 or constant_pool_count == 0:
        return False

    offset = 10
    index = 1
    while index < constant_pool_count:
        if offset >= len(data):
            return False
        tag = data[offset]
        offset += 1
        if tag == 1:
            result = read_java_u2(data, offset)
            if result is None:
                return False
            length, offset = result
            if length > len(data) - offset:
                return False
            offset += length
        elif tag in (3, 4, 9, 10, 11, 12, 17, 18):
            offset += 4
        elif tag in (5, 6):
            offset += 8
            index += 1
        elif tag in (7, 8, 16, 19, 20):
            offset += 2
        elif tag == 15:
            offset += 3
        else:
            return False
        if offset > len(data):
            return False
        index += 1

    if offset > len(data) - 8:
        return False
    interfaces_count = int.from_bytes(data[offset + 6 : offset + 8], "big")
    offset += 8
    if interfaces_count > (len(data) - offset) // 2:
        return False
    offset += interfaces_count * 2

    offset = skip_java_members(data, offset)
    if offset is None:
        return False
    offset = skip_java_members(data, offset)
    if offset is None:
        return False
    result = read_java_u2(data, offset)
    if result is None:
        return False
    attributes_count, offset = result
    offset = skip_java_attributes(data, offset, attributes_count)
    return offset == len(data)


def native_binary_magic(source, size: int, name: str) -> str | None:
    """Return the native executable format, excluding valid Java class files."""
    header = source.read(min(size, 64))
    if header.startswith(b"\x7fELF"):
        return "ELF"
    if (
        header.startswith(JAVA_CLASS_MAGIC)
        and name.lower().endswith(".class")
    ):
        source.seek(0)
        if is_java_class(source.read()):
            return None
    if header[:4] in MACHO_MAGICS:
        return "Mach-O"
    if header.startswith(b"MZ") and len(header) >= 64:
        pe_offset = int.from_bytes(header[0x3C:0x40], "little")
        if pe_offset <= size - 4:
            source.seek(pe_offset)
            if source.read(4) == b"PE\0\0":
                return "PE"
    return None


def native_archive_entries(
    archive: ZipFile, entries: dict[str, ZipInfo]
) -> set[str]:
    native_entries = set()
    for name, info in entries.items():
        if info.is_dir():
            continue
        with archive.open(info) as source:
            magic = native_binary_magic(source, info.file_size, name)
        if name.lower().endswith(NATIVE_SUFFIXES) or magic is not None:
            native_entries.add(name)
    return native_entries


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

        packaged_natives = native_archive_entries(archive, entries)
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
        forbidden.extend(
            sorted(native_archive_entries(archive, entries) - set(forbidden))
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
