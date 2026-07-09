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

import base64
import csv
import hashlib
import io
import stat
import sys
import warnings
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import verify_python_wheels as verifier  # noqa: E402


SUPPORTED_WHEELS = (
    ("x86_64-unknown-linux-gnu", "manylinux_2_28_x86_64"),
    ("aarch64-unknown-linux-gnu", "manylinux_2_28_aarch64"),
    ("aarch64-apple-darwin", "macosx_11_0_arm64"),
    ("x86_64-pc-windows-msvc", "win_amd64"),
)


def write_zip(path, entries):
    with ZipFile(path, "w") as archive:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for entry, content in entries:
                archive.writestr(entry, content)


def record_bytes(contents, record_path, mutate_record=None):
    rows = []
    for path, content in contents.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
        rows.append([path, f"sha256={digest.rstrip(b'=').decode()}", str(len(content))])
    rows.append([record_path, "", ""])
    if mutate_record is not None:
        mutate_record(rows)

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().encode()


def build_wheel(
    tmp_path,
    target="aarch64-unknown-linux-gnu",
    platform_tag="linux_aarch64",
    filename_distribution="paimon_mosaic",
    filename_version="0.3.0",
    dist_info_distribution=None,
    dist_info_version=None,
    metadata_name="paimon-mosaic",
    metadata_version="0.3.0",
    wheel_tags=None,
    mutate_record=None,
    unrecorded_entries=None,
    directory_entries=None,
):
    dist_info_distribution = dist_info_distribution or filename_distribution
    dist_info_version = dist_info_version or filename_version
    wheel_tags = wheel_tags or [f"py3-none-{platform_tag}"]
    dist_info = f"{dist_info_distribution}-{dist_info_version}.dist-info"
    record_path = f"{dist_info}/RECORD"

    license_text = b"Apache License\nTHIRD-PARTY-LICENSES.html\n"
    notice_text = b"Apache Arrow\n"
    report_text = f"{target}\nFor Zstandard software\nApache Arrow\n".encode()
    legal_files = {
        "LICENSE": license_text,
        "NOTICE": notice_text,
        "THIRD-PARTY-LICENSES.html": report_text,
    }
    expected_license_fields = "".join(
        f"License-File: licenses/{target}/{name}\n"
        for name in ("LICENSE", "NOTICE", "THIRD-PARTY-LICENSES.html")
    )
    metadata = (
        "Metadata-Version: 2.4\n"
        f"Name: {metadata_name}\n"
        f"Version: {metadata_version}\n"
        "License-Expression: Apache-2.0\n"
        f"{expected_license_fields}"
        "\n"
    ).encode()
    wheel_metadata = (
        "Wheel-Version: 1.0\n"
        "Root-Is-Purelib: false\n"
        + "".join(f"Tag: {tag}\n" for tag in wheel_tags)
        + "\n"
    ).encode()

    contents = {
        "mosaic/LICENSE": license_text,
        "mosaic/NOTICE": notice_text,
        "mosaic/THIRD-PARTY-LICENSES.html": report_text,
        verifier.NATIVE_LIBRARY[target]: b"native-library",
        f"{dist_info}/METADATA": metadata,
        f"{dist_info}/WHEEL": wheel_metadata,
    }
    for name, content in legal_files.items():
        contents[f"{dist_info}/licenses/licenses/{target}/{name}"] = content
    contents[record_path] = record_bytes(contents, record_path, mutate_record)

    wheel = (
        tmp_path
        / (
            f"{filename_distribution}-{filename_version}-"
            f"py3-none-{platform_tag}.whl"
        )
    )
    entries = list(contents.items())
    entries.extend((entry, b"") for entry in (directory_entries or ()))
    entries.extend((unrecorded_entries or {}).items())
    write_zip(wheel, entries)

    root = tmp_path / "root"
    legal_root = root / "python/licenses" / target
    legal_root.mkdir(parents=True)
    for name, content in legal_files.items():
        (legal_root / name).write_bytes(content)
    return wheel, root


@pytest.mark.parametrize("target,platform_tag", SUPPORTED_WHEELS)
def test_verify_wheel_accepts_supported_targets(
    tmp_path, monkeypatch, target, platform_tag
):
    wheel, root = build_wheel(tmp_path, target=target, platform_tag=platform_tag)
    monkeypatch.setattr(verifier, "verify_native_target", lambda *args: None)

    assert verifier.verify_wheel(wheel, root) == target


def test_verify_wheel_accepts_unrecorded_directory_entries(tmp_path, monkeypatch):
    wheel, root = build_wheel(
        tmp_path,
        directory_entries=(
            "mosaic/",
            "paimon_mosaic-0.3.0.dist-info/",
            "paimon_mosaic-0.3.0.dist-info/licenses/",
        ),
    )
    monkeypatch.setattr(verifier, "verify_native_target", lambda *args: None)

    assert verifier.verify_wheel(wheel, root) == "aarch64-unknown-linux-gnu"


@pytest.mark.parametrize(
    "entry",
    (
        "/mosaic/file.py",
        "C:/mosaic/file.py",
        "mosaic\\file.py",
        "mosaic/../file.py",
    ),
)
def test_validate_archive_paths_rejects_unsafe_paths(tmp_path, entry):
    wheel = tmp_path / "unsafe.whl"
    write_zip(wheel, [(entry, b"content")])

    with ZipFile(wheel) as archive, pytest.raises(ValueError):
        verifier.validate_archive_paths(archive)


def test_validate_archive_paths_rejects_symlink(tmp_path):
    wheel = tmp_path / "symlink.whl"
    link = ZipInfo("mosaic/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    write_zip(wheel, [(link, b"mosaic/file.py")])

    with ZipFile(wheel) as archive, pytest.raises(ValueError, match="symbolic link"):
        verifier.validate_archive_paths(archive)


def test_validate_archive_paths_rejects_duplicate_raw_name(tmp_path):
    wheel = tmp_path / "duplicate.whl"
    write_zip(wheel, [("mosaic/file.py", b"one"), ("mosaic/file.py", b"two")])

    with ZipFile(wheel) as archive, pytest.raises(ValueError, match="duplicate wheel"):
        verifier.validate_archive_paths(archive)


def test_validate_archive_paths_rejects_duplicate_normalized_name(tmp_path):
    wheel = tmp_path / "duplicate-normalized.whl"
    write_zip(
        wheel,
        [("mosaic/file.py", b"one"), ("mosaic/./file.py", b"two")],
    )

    with ZipFile(wheel) as archive, pytest.raises(
        ValueError, match="duplicate normalized"
    ):
        verifier.validate_archive_paths(archive)


@pytest.mark.parametrize(
    "options,error",
    (
        (
            {"dist_info_distribution": "other"},
            "filename distribution",
        ),
        (
            {"dist_info_version": "0.4.0"},
            "filename version",
        ),
        (
            {"metadata_name": "other"},
            "METADATA Name",
        ),
        (
            {"metadata_version": "0.4.0"},
            "METADATA Version",
        ),
        (
            {"wheel_tags": ["cp39-none-linux_aarch64"]},
            "WHEEL tags",
        ),
    ),
)
def test_verify_wheel_rejects_identity_mismatches(
    tmp_path, monkeypatch, options, error
):
    wheel, root = build_wheel(tmp_path, **options)
    monkeypatch.setattr(verifier, "verify_native_target", lambda *args: None)

    with pytest.raises(ValueError, match=error):
        verifier.verify_wheel(wheel, root)


def test_verify_wheel_rejects_musllinux_for_gnu_target(tmp_path):
    wheel = (
        tmp_path / "paimon_mosaic-0.3.0-py3-none-musllinux_1_2_aarch64.whl"
    )

    with pytest.raises(ValueError, match="musllinux"):
        verifier.verify_wheel(wheel, tmp_path)


def find_record_row(rows, suffix):
    return next(row for row in rows if row[0].endswith(suffix))


@pytest.mark.parametrize(
    "mutate_record,error",
    (
        (
            lambda rows: find_record_row(rows, "/METADATA").__setitem__(
                1, "sha256=invalid"
            ),
            "hash mismatch",
        ),
        (
            lambda rows: find_record_row(rows, "/METADATA").__setitem__(2, "1"),
            "size mismatch",
        ),
        (
            lambda rows: find_record_row(rows, "/METADATA").__setitem__(1, ""),
            "omits the hash or size",
        ),
        (
            lambda rows: rows.pop(0),
            "omits wheel entries",
        ),
        (
            lambda rows: rows.append(["missing.py", "sha256=invalid", "1"]),
            "lists missing wheel entries",
        ),
        (
            lambda rows: find_record_row(rows, "/RECORD").__setitem__(
                slice(1, 3), ["sha256=invalid", "1"]
            ),
            "blank hash and size",
        ),
    ),
)
def test_verify_wheel_rejects_invalid_record(
    tmp_path, monkeypatch, mutate_record, error
):
    wheel, root = build_wheel(tmp_path, mutate_record=mutate_record)
    monkeypatch.setattr(verifier, "verify_native_target", lambda *args: None)

    with pytest.raises(ValueError, match=error):
        verifier.verify_wheel(wheel, root)


def test_verify_wheel_rejects_unrecorded_archive_entry(tmp_path, monkeypatch):
    wheel, root = build_wheel(
        tmp_path,
        unrecorded_entries={"mosaic/unlisted.py": b"unlisted"},
    )
    monkeypatch.setattr(verifier, "verify_native_target", lambda *args: None)

    with pytest.raises(ValueError, match="omits wheel entries"):
        verifier.verify_wheel(wheel, root)
