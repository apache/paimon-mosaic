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

"""Verify native wheels and their artifact-exact legal metadata."""

from __future__ import annotations

import argparse
import base64
import csv
import email
import hashlib
import hmac
import io
import posixpath
import re
import stat
import sys
from pathlib import Path
from zipfile import ZipFile

from native_binary import TARGET_ARCHITECTURE, verify_native_target


MAX_ARCHIVE_ENTRY_SIZE = 256 * 1024 * 1024
NATIVE_LIBRARY = {
    "x86_64-unknown-linux-gnu": "mosaic/libpaimon_mosaic_ffi.so",
    "aarch64-unknown-linux-gnu": "mosaic/libpaimon_mosaic_ffi.so",
    "aarch64-apple-darwin": "mosaic/libpaimon_mosaic_ffi.dylib",
    "x86_64-pc-windows-msvc": "mosaic/paimon_mosaic_ffi.dll",
}

EXPECTED_WHEEL_TAG = {
    "x86_64-unknown-linux-gnu": "py3-none-manylinux_2_28_x86_64",
    "aarch64-unknown-linux-gnu": "py3-none-manylinux_2_28_aarch64",
    "aarch64-apple-darwin": "py3-none-macosx_11_0_arm64",
    "x86_64-pc-windows-msvc": "py3-none-win_amd64",
}

NESTED_LICENSE_MARKERS = (
    "For Zstandard software",
    "Apache Arrow",
)

WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:/")
MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}


def _validate_target_matrix() -> None:
    if not (
        set(NATIVE_LIBRARY)
        == set(EXPECTED_WHEEL_TAG)
        == set(TARGET_ARCHITECTURE)
    ):
        raise RuntimeError("Python wheel native target matrices are inconsistent")


_validate_target_matrix()


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def expand_tag(python_tag: str, abi_tag: str, platform_tag: str) -> set[str]:
    components = (python_tag, abi_tag, platform_tag)
    if any(
        not component
        or any(not part for part in component.split("."))
        or any(character.isspace() for character in component)
        for component in components
    ):
        raise ValueError(f"invalid wheel tag components: {components}")
    return {
        f"{python}-{abi}-{platform}".lower()
        for python in python_tag.split(".")
        for abi in abi_tag.split(".")
        for platform in platform_tag.split(".")
    }


def parse_wheel_filename(name: str) -> tuple[str, str, set[str]]:
    if not name.endswith(".whl"):
        raise ValueError(f"not a wheel filename: {name}")
    parts = name[:-4].split("-")
    if len(parts) == 5:
        distribution, version, python_tag, abi_tag, platform_tag = parts
    elif len(parts) == 6:
        distribution, version, build_tag, python_tag, abi_tag, platform_tag = parts
        if not re.fullmatch(r"[0-9][A-Za-z0-9_]*", build_tag):
            raise ValueError(f"invalid wheel build tag: {build_tag}")
    else:
        raise ValueError(f"invalid wheel filename: {name}")
    if (
        not distribution
        or not version
        or any(character.isspace() for character in distribution + version)
    ):
        raise ValueError(f"invalid wheel filename: {name}")
    return (
        distribution,
        version,
        expand_tag(python_tag, abi_tag, platform_tag),
    )


def parse_wheel_metadata_tags(tags: list[str]) -> set[str]:
    parsed = set()
    for tag in tags:
        parts = tag.split("-")
        if len(parts) != 3:
            raise ValueError(f"invalid WHEEL Tag field: {tag!r}")
        expanded = expand_tag(*parts)
        if parsed.intersection(expanded):
            raise ValueError(f"duplicate WHEEL Tag field: {tag!r}")
        parsed.update(expanded)
    return parsed


def target_from_wheel_tags(tags: set[str]) -> str:
    if any(tag.rsplit("-", 1)[-1].startswith("musllinux_") for tag in tags):
        raise ValueError("musllinux wheels do not match the supported GNU targets")
    matching_targets = [
        target
        for target, expected_tag in EXPECTED_WHEEL_TAG.items()
        if tags == {expected_tag}
    ]
    if len(matching_targets) != 1:
        raise ValueError(
            f"unsupported wheel tags: {sorted(tags)}; expected exactly one of "
            f"{sorted(EXPECTED_WHEEL_TAG.values())}"
        )
    return matching_targets[0]


def target_from_wheel_name(name: str) -> str:
    _, _, tags = parse_wheel_filename(name)
    return target_from_wheel_tags(tags)


def normalized_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_dist_info_name(dist_info: str) -> tuple[str, str]:
    if "/" in dist_info or not dist_info.endswith(".dist-info"):
        raise ValueError(f"invalid .dist-info directory: {dist_info}")
    stem = dist_info[: -len(".dist-info")]
    try:
        distribution, version = stem.rsplit("-", 1)
    except ValueError as error:
        raise ValueError(f"invalid .dist-info directory: {dist_info}") from error
    if not distribution or not version:
        raise ValueError(f"invalid .dist-info directory: {dist_info}")
    return distribution, version


def validate_archive_paths(archive: ZipFile) -> set[str]:
    names = set()
    normalized_names = set()
    for info in archive.infolist():
        raw_name = info.orig_filename
        if not raw_name or "\x00" in raw_name or raw_name != info.filename:
            raise ValueError(f"invalid wheel entry path: {raw_name!r}")
        if "\\" in raw_name:
            raise ValueError(f"wheel entry uses a backslash: {raw_name!r}")
        if raw_name.startswith("/") or WINDOWS_ABSOLUTE_PATH.match(raw_name):
            raise ValueError(f"wheel entry uses an absolute path: {raw_name!r}")
        if ".." in raw_name.split("/"):
            raise ValueError(f"wheel entry uses '..': {raw_name!r}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"wheel entry is a symbolic link: {raw_name!r}")
        if info.file_size > MAX_ARCHIVE_ENTRY_SIZE:
            raise ValueError(
                f"wheel entry {raw_name!r} exceeds the size limit of "
                f"{MAX_ARCHIVE_ENTRY_SIZE} bytes: {info.file_size} bytes"
            )

        normalized_name = posixpath.normpath(raw_name)
        if normalized_name in ("", ".", "/"):
            raise ValueError(f"invalid wheel entry path: {raw_name!r}")
        if raw_name in names:
            raise ValueError(f"duplicate wheel entry path: {raw_name!r}")
        if normalized_name in normalized_names:
            raise ValueError(
                f"duplicate normalized wheel entry path: {normalized_name!r}"
            )
        names.add(raw_name)
        normalized_names.add(normalized_name)
    return names


def native_binary_magic(source, size: int) -> str | None:
    header = source.read(min(size, 64))
    if header.startswith(b"\x7fELF"):
        return "ELF"
    if header[:4] in MACHO_MAGICS:
        return "Mach-O"
    if header.startswith(b"MZ") and len(header) >= 64:
        pe_offset = int.from_bytes(header[0x3C:0x40], "little")
        if pe_offset <= size - 4:
            source.seek(pe_offset)
            if source.read(4) == b"PE\0\0":
                return "PE"
    return None


def require_single_header(message: email.message.Message, name: str) -> str:
    values = message.get_all(name, [])
    if len(values) != 1:
        raise ValueError(f"expected one {name} field, found {values}")
    return values[0]


def verify_record(archive: ZipFile, file_names: set[str], record_path: str) -> None:
    try:
        record_text = archive.read(record_path).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{record_path} is not valid UTF-8") from error

    rows = {}
    reader = csv.reader(io.StringIO(record_text, newline=""))
    for line_number, row in enumerate(reader, 1):
        if len(row) != 3:
            raise ValueError(
                f"{record_path}:{line_number} must contain exactly three fields"
            )
        entry_path, hash_field, size_field = row
        if entry_path in rows:
            raise ValueError(f"{record_path} lists {entry_path!r} more than once")
        rows[entry_path] = (hash_field, size_field)

    listed = set(rows)
    unlisted = sorted(file_names - listed)
    nonexistent = sorted(listed - file_names)
    if unlisted:
        raise ValueError(f"{record_path} omits wheel entries: {unlisted}")
    if nonexistent:
        raise ValueError(f"{record_path} lists missing wheel entries: {nonexistent}")

    if rows.get(record_path) != ("", ""):
        raise ValueError(
            f"{record_path} must list itself with a blank hash and size"
        )

    for entry_path in sorted(file_names - {record_path}):
        hash_field, size_field = rows[entry_path]
        if not hash_field or not size_field:
            raise ValueError(
                f"{record_path} omits the hash or size for {entry_path}"
            )
        try:
            algorithm, expected_hash = hash_field.split("=", 1)
        except ValueError as error:
            raise ValueError(
                f"{record_path} has an invalid hash for {entry_path}: {hash_field!r}"
            ) from error
        try:
            digest = hashlib.new(algorithm)
        except ValueError as error:
            raise ValueError(
                f"{record_path} uses unknown hash algorithm {algorithm!r} "
                f"for {entry_path}"
            ) from error
        if digest.digest_size < hashlib.sha256().digest_size:
            raise ValueError(
                f"{record_path} uses weak hash algorithm {algorithm!r} "
                f"for {entry_path}"
            )
        try:
            expected_size = int(size_field)
        except ValueError as error:
            raise ValueError(
                f"{record_path} has an invalid size for {entry_path}: {size_field!r}"
            ) from error
        if expected_size < 0:
            raise ValueError(
                f"{record_path} has a negative size for {entry_path}: {size_field!r}"
            )

        actual_size = 0
        with archive.open(entry_path) as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                actual_size += len(chunk)
        actual_hash = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()
        if not hmac.compare_digest(actual_hash, expected_hash):
            raise ValueError(f"{record_path} hash mismatch for {entry_path}")
        if actual_size != expected_size:
            raise ValueError(
                f"{record_path} size mismatch for {entry_path}: "
                f"found {actual_size}, expected {expected_size}"
            )


def require_equal(actual: bytes, expected_path: Path, archive_path: str) -> None:
    expected = expected_path.read_bytes()
    if actual != expected:
        raise ValueError(
            f"{archive_path} does not match {expected_path.as_posix()}"
        )


def verify_wheel(wheel: Path, root: Path) -> str:
    filename_distribution, filename_version, filename_tags = parse_wheel_filename(
        wheel.name
    )
    target = target_from_wheel_tags(filename_tags)
    legal_source = root / "python/licenses" / target
    expected_license_files = [
        f"licenses/{target}/LICENSE",
        f"licenses/{target}/NOTICE",
        f"licenses/{target}/THIRD-PARTY-LICENSES.html",
    ]

    with ZipFile(wheel) as archive:
        names = validate_archive_paths(archive)
        file_names = {info.filename for info in archive.infolist() if not info.is_dir()}
        dist_info_directories = sorted(
            {
                name.split("/", 1)[0]
                for name in names
                if "/" in name and name.split("/", 1)[0].endswith(".dist-info")
            }
        )
        if len(dist_info_directories) != 1:
            raise ValueError(
                f"expected one .dist-info directory, found {dist_info_directories}"
            )
        dist_info = dist_info_directories[0]
        dist_info_distribution, dist_info_version = parse_dist_info_name(dist_info)
        if dist_info_distribution != filename_distribution:
            raise ValueError(
                "wheel filename distribution "
                f"{filename_distribution!r} does not match {dist_info!r}"
            )
        if dist_info_version != filename_version:
            raise ValueError(
                f"wheel filename version {filename_version!r} "
                f"does not match {dist_info!r}"
            )

        metadata_path = f"{dist_info}/METADATA"
        wheel_metadata_path = f"{dist_info}/WHEEL"
        record_path = f"{dist_info}/RECORD"
        for required_metadata in (metadata_path, wheel_metadata_path, record_path):
            if required_metadata not in names:
                raise ValueError(f"missing {required_metadata}")
        verify_record(archive, file_names, record_path)

        package_legal = {
            "mosaic/LICENSE": legal_source / "LICENSE",
            "mosaic/NOTICE": legal_source / "NOTICE",
            "mosaic/THIRD-PARTY-LICENSES.html": (
                legal_source / "THIRD-PARTY-LICENSES.html"
            ),
        }
        standard_legal = {
            f"{dist_info}/licenses/{relative}": legal_source / Path(relative).name
            for relative in expected_license_files
        }
        required = set(package_legal) | set(standard_legal)
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"missing legal files: {missing}")
        if "mosaic/DEPENDENCIES.rust.tsv" in names:
            raise ValueError(
                "wheel contains the cross-target repository dependency inventory"
            )

        for archive_path, expected_path in {**package_legal, **standard_legal}.items():
            require_equal(archive.read(archive_path), expected_path, archive_path)

        native_entries = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            with archive.open(info) as source:
                magic = native_binary_magic(source, info.file_size)
            if (
                name.startswith("mosaic/")
                and name.endswith((".so", ".dylib", ".dll"))
            ) or magic is not None:
                native_entries.append(name)
        native_entries.sort()
        if native_entries != [NATIVE_LIBRARY[target]]:
            raise ValueError(f"unexpected native libraries: {native_entries}")
        verify_native_target(
            archive.read(native_entries[0]),
            target,
            native_entries[0],
            symbol_family="FFI",
        )

        legal_target_prefix = f"{dist_info}/licenses/licenses/"
        packaged_targets = {
            name[len(legal_target_prefix) :].split("/", 1)[0]
            for name in names
            if name.startswith(legal_target_prefix) and not name.endswith("/")
        }
        if packaged_targets != {target}:
            raise ValueError(
                f"wheel legal metadata covers {sorted(packaged_targets)}, expected {target}"
            )

        metadata = email.message_from_bytes(archive.read(metadata_path))
        metadata_name = require_single_header(metadata, "Name")
        if normalized_distribution(metadata_name) != normalized_distribution(
            filename_distribution
        ):
            raise ValueError(
                f"METADATA Name {metadata_name!r} does not match "
                f"wheel distribution {filename_distribution!r}"
            )
        metadata_version = require_single_header(metadata, "Version")
        if metadata_version != filename_version:
            raise ValueError(
                f"METADATA Version {metadata_version!r} does not match "
                f"wheel version {filename_version!r}"
            )
        if metadata.get("Metadata-Version") != "2.4":
            raise ValueError(
                f"unexpected Metadata-Version: {metadata.get('Metadata-Version')}"
            )
        if metadata.get("License-Expression") != "Apache-2.0":
            raise ValueError(
                f"unexpected License-Expression: {metadata.get('License-Expression')}"
            )
        if metadata.get_all("License-File", []) != expected_license_files:
            raise ValueError(
                "unexpected License-File fields: "
                + repr(metadata.get_all("License-File", []))
            )

        wheel_metadata = email.message_from_bytes(archive.read(wheel_metadata_path))
        if wheel_metadata.get("Root-Is-Purelib", "").lower() != "false":
            raise ValueError(
                "wheel metadata must declare Root-Is-Purelib: false"
            )
        wheel_tags = wheel_metadata.get_all("Tag", [])
        parsed_wheel_tags = parse_wheel_metadata_tags(wheel_tags)
        if parsed_wheel_tags != filename_tags:
            raise ValueError(
                f"WHEEL tags {sorted(parsed_wheel_tags)} do not match "
                f"filename tags {sorted(filename_tags)}"
            )

        license_text = archive.read("mosaic/LICENSE").decode("utf-8")
        if "THIRD-PARTY-LICENSES.html" not in license_text:
            raise ValueError("LICENSE does not point to the third-party report")
        notice_text = archive.read("mosaic/NOTICE").decode("utf-8")
        if "Apache Arrow" not in notice_text:
            raise ValueError("NOTICE omits the bundled Apache Arrow notice")

        report_text = archive.read(
            "mosaic/THIRD-PARTY-LICENSES.html"
        ).decode("utf-8")
        if target not in report_text:
            raise ValueError(f"third-party report does not identify target {target}")
        for marker in NESTED_LICENSE_MARKERS:
            if marker not in report_text:
                raise ValueError(f"third-party report is missing {marker!r}")

        if any(name.startswith("tests/") for name in names):
            raise ValueError("wheel unexpectedly contains tests/")

    print(f"verified {wheel.name}: {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument(
        "--require-all-targets",
        action="store_true",
        help="require exactly one wheel for every declared release target",
    )
    args = parser.parse_args()
    root = repository_root()

    failed = False
    targets = []
    for wheel in args.wheels:
        try:
            targets.append(verify_wheel(wheel, root))
        except (KeyError, OSError, ValueError) as error:
            failed = True
            print(f"{wheel}: {error}", file=sys.stderr)
    if args.require_all_targets:
        expected = set(NATIVE_LIBRARY)
        actual = set(targets)
        if actual != expected or len(targets) != len(expected):
            failed = True
            print(
                "wheel target set differs from the four release targets: "
                f"found {targets}, expected {sorted(expected)}",
                file=sys.stderr,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
