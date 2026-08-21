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
import importlib.util
import io
import shutil
import stat
import sys
import warnings
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest


TESTS = Path(__file__).resolve().parent
TOOLS = TESTS.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTS))

import verify_python_wheels as verifier  # noqa: E402
import native_binary  # noqa: E402
from native_binary_fixtures import FFI_SYMBOLS, build_elf  # noqa: E402


PE_SIDECAR = bytearray(132)
PE_SIDECAR[:2] = b"MZ"
PE_SIDECAR[0x3C:0x40] = (0x80).to_bytes(4, "little")
PE_SIDECAR[0x80:0x84] = b"PE\0\0"


SUPPORTED_WHEELS = (
    (
        "x86_64-unknown-linux-gnu",
        "manylinux_2_28_x86_64",
        "mosaic/libpaimon_mosaic_ffi.so",
    ),
    (
        "aarch64-unknown-linux-gnu",
        "manylinux_2_28_aarch64",
        "mosaic/libpaimon_mosaic_ffi.so",
    ),
    (
        "aarch64-apple-darwin",
        "macosx_11_0_arm64",
        "mosaic/libpaimon_mosaic_ffi.dylib",
    ),
    (
        "x86_64-pc-windows-msvc",
        "win_amd64",
        "mosaic/paimon_mosaic_ffi.dll",
    ),
)
EXPECTED_NATIVE_LIBRARY = {
    target: native_path for target, _platform_tag, native_path in SUPPORTED_WHEELS
}


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
    platform_tag="manylinux_2_28_aarch64",
    python_tag="py3",
    abi_tag="none",
    filename_distribution="paimon_mosaic",
    filename_version="0.3.0",
    dist_info_distribution=None,
    dist_info_version=None,
    metadata_name="paimon-mosaic",
    metadata_version="0.3.0",
    wheel_tags=None,
    mutate_record=None,
    extra_entries=None,
    unrecorded_entries=None,
    directory_entries=None,
    native_bytes=b"native-library",
    native_path=None,
):
    dist_info_distribution = dist_info_distribution or filename_distribution
    dist_info_version = dist_info_version or filename_version
    wheel_tags = wheel_tags or [f"{python_tag}-{abi_tag}-{platform_tag}"]
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

    native_path = native_path or EXPECTED_NATIVE_LIBRARY[target]
    contents = {
        "mosaic/LICENSE": license_text,
        "mosaic/NOTICE": notice_text,
        "mosaic/THIRD-PARTY-LICENSES.html": report_text,
        native_path: native_bytes,
        f"{dist_info}/METADATA": metadata,
        f"{dist_info}/WHEEL": wheel_metadata,
    }
    for name, content in legal_files.items():
        contents[f"{dist_info}/licenses/licenses/{target}/{name}"] = content
    contents.update(extra_entries or {})
    contents[record_path] = record_bytes(contents, record_path, mutate_record)

    wheel = (
        tmp_path
        / (
            f"{filename_distribution}-{filename_version}-"
            f"{python_tag}-{abi_tag}-{platform_tag}.whl"
        )
    )
    entries = list(contents.items())
    entries.extend((entry, b"") for entry in (directory_entries or ()))
    entries.extend((unrecorded_entries or {}).items())
    write_zip(wheel, entries)

    root = tmp_path / "root"
    legal_root = root / "python/licenses" / target
    legal_root.mkdir(parents=True, exist_ok=True)
    for name, content in legal_files.items():
        (legal_root / name).write_bytes(content)
    return wheel, root


@pytest.mark.parametrize("target,platform_tag,native_path", SUPPORTED_WHEELS)
def test_verify_wheel_accepts_supported_targets(
    tmp_path, monkeypatch, target, platform_tag, native_path
):
    native_bytes = f"native fixture for {target}".encode()
    wheel, root = build_wheel(
        tmp_path,
        target=target,
        platform_tag=platform_tag,
        native_bytes=native_bytes,
        native_path=native_path,
    )
    native_calls = []

    def capture_native_call(*args, **kwargs):
        native_calls.append((args, kwargs))

    monkeypatch.setattr(
        verifier,
        "verify_native_target",
        capture_native_call,
    )

    assert verifier.verify_wheel(wheel, root) == target
    assert native_calls == [
        (
            (native_bytes, target, native_path),
            {"symbol_family": "FFI"},
        )
    ]


@pytest.mark.parametrize(
    "native_bytes,error",
    (
        (
            build_elf(machine=62, symbols=FFI_SYMBOLS),
            "architectures.*expected only aarch64",
        ),
        (
            build_elf(machine=183, symbols={"unrelated_export"}),
            "missing expected Mosaic FFI exports",
        ),
    ),
)
def test_verify_wheel_rejects_invalid_native_at_canonical_path(
    tmp_path, native_bytes, error
):
    target = "aarch64-unknown-linux-gnu"
    wheel, root = build_wheel(
        tmp_path,
        target=target,
        platform_tag="manylinux_2_28_aarch64",
        native_bytes=native_bytes,
        native_path="mosaic/libpaimon_mosaic_ffi.so",
    )

    with pytest.raises(ValueError, match=error):
        verifier.verify_wheel(wheel, root)


@pytest.mark.parametrize(
    "target,python_tag,abi_tag,platform_tag",
    (
        (
            "aarch64-unknown-linux-gnu",
            "cp39",
            "cp39",
            "manylinux_2_28_aarch64",
        ),
        (
            "x86_64-unknown-linux-gnu",
            "cp312",
            "abi3",
            "manylinux_2_28_x86_64",
        ),
        (
            "aarch64-apple-darwin",
            "py2",
            "none",
            "macosx_11_0_arm64",
        ),
        (
            "aarch64-unknown-linux-gnu",
            "py3",
            "none",
            "linux_aarch64",
        ),
        (
            "x86_64-unknown-linux-gnu",
            "py3",
            "none",
            "manylinux_2_17_x86_64",
        ),
        (
            "aarch64-apple-darwin",
            "py3",
            "none",
            "macosx_10_9_arm64",
        ),
    ),
)
def test_verify_wheel_rejects_non_release_tags(
    tmp_path,
    monkeypatch,
    target,
    python_tag,
    abi_tag,
    platform_tag,
):
    wheel, root = build_wheel(
        tmp_path,
        target=target,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
    )
    monkeypatch.setattr(verifier, "verify_native_target", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="unsupported wheel tags"):
        verifier.verify_wheel(wheel, root)


def test_verify_wheel_accepts_unrecorded_directory_entries(tmp_path, monkeypatch):
    wheel, root = build_wheel(
        tmp_path,
        directory_entries=(
            "mosaic/",
            "paimon_mosaic-0.3.0.dist-info/",
            "paimon_mosaic-0.3.0.dist-info/licenses/",
        ),
    )
    monkeypatch.setattr(verifier, "verify_native_target", lambda *args, **kwargs: None)

    assert verifier.verify_wheel(wheel, root) == "aarch64-unknown-linux-gnu"


def test_main_requires_exactly_one_wheel_per_release_target(
    tmp_path, monkeypatch
):
    wheels = []
    root = None
    for target, platform_tag, _native_path in SUPPORTED_WHEELS:
        wheel, root = build_wheel(
            tmp_path,
            target=target,
            platform_tag=platform_tag,
        )
        wheels.append(wheel)

    monkeypatch.setattr(verifier, "repository_root", lambda: root)
    monkeypatch.setattr(verifier, "verify_native_target", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_python_wheels.py", "--require-all-targets", *map(str, wheels)],
    )
    assert verifier.main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_python_wheels.py", "--require-all-targets", *map(str, wheels[:-1])],
    )
    assert verifier.main() == 1

    duplicate = tmp_path / wheels[0].name.replace("-py3-", "-1-py3-")
    shutil.copy2(wheels[0], duplicate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_python_wheels.py",
            "--require-all-targets",
            *map(str, [*wheels, duplicate]),
        ],
    )
    assert verifier.main() == 1


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


def test_verify_wheel_rejects_oversized_entry_before_archive_read(
    tmp_path, monkeypatch
):
    wheel, root = build_wheel(
        tmp_path,
        extra_entries={"mosaic/oversized.bin": b"x" * 4097},
    )
    monkeypatch.setattr(
        verifier,
        "MAX_ARCHIVE_ENTRY_SIZE",
        4096,
        raising=False,
    )

    def fail_unbounded_read(*_args, **_kwargs):
        raise AssertionError("archive.read must not be called")

    monkeypatch.setattr(ZipFile, "read", fail_unbounded_read)
    with pytest.raises(ValueError, match=r"oversized\.bin.*size limit"):
        verifier.verify_wheel(wheel, root)


def test_target_matrix_guard_rejects_drift(monkeypatch):
    monkeypatch.setitem(
        verifier.EXPECTED_WHEEL_TAG,
        "unsupported-target",
        "py3-none-unsupported",
    )

    with pytest.raises(RuntimeError, match="target matrices"):
        verifier._validate_target_matrix()


def test_target_matrix_is_validated_during_import(monkeypatch):
    monkeypatch.setattr(
        native_binary,
        "TARGET_ARCHITECTURE",
        {"unsupported-target": ("ELF", "x86_64")},
    )
    spec = importlib.util.spec_from_file_location(
        "verify_python_wheels_matrix_probe",
        TOOLS / "verify_python_wheels.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(RuntimeError, match="target matrices"):
        spec.loader.exec_module(module)


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
    monkeypatch.setattr(verifier, "verify_native_target", lambda *args, **kwargs: None)

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
        (
            lambda rows: find_record_row(rows, "/METADATA").__setitem__(
                1, "sha256"
            ),
            "invalid hash",
        ),
        (
            lambda rows: find_record_row(rows, "/METADATA").__setitem__(
                1, "unsupported=ignored"
            ),
            "unknown hash algorithm",
        ),
        (
            lambda rows: find_record_row(rows, "/METADATA").__setitem__(
                1, "sha1=ignored"
            ),
            "weak hash algorithm",
        ),
    ),
)
def test_verify_wheel_rejects_invalid_record(
    tmp_path, monkeypatch, mutate_record, error
):
    wheel, root = build_wheel(tmp_path, mutate_record=mutate_record)
    monkeypatch.setattr(verifier, "verify_native_target", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match=error):
        verifier.verify_wheel(wheel, root)


def test_verify_wheel_requires_artifact_exact_legal_files(
    tmp_path, monkeypatch
):
    wheel, root = build_wheel(tmp_path)
    expected_notice = (
        root
        / "python/licenses/aarch64-unknown-linux-gnu/NOTICE"
    )
    expected_notice.write_bytes(b"different expected notice\n")
    monkeypatch.setattr(
        verifier,
        "verify_native_target",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="does not match"):
        verifier.verify_wheel(wheel, root)


def test_verify_wheel_rejects_unrecorded_archive_entry(tmp_path, monkeypatch):
    wheel, root = build_wheel(
        tmp_path,
        unrecorded_entries={"mosaic/unlisted.py": b"unlisted"},
    )
    monkeypatch.setattr(verifier, "verify_native_target", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="omits wheel entries"):
        verifier.verify_wheel(wheel, root)


@pytest.mark.parametrize(
    "entry,magic",
    (
        ("paimon_mosaic.libs/libzstd.so.1", b"\x7fELF"),
        ("paimon_mosaic.libs/helper.bin", bytes(PE_SIDECAR)),
        ("paimon_mosaic.libs/helper.data", b"\xcf\xfa\xed\xfe"),
    ),
)
def test_verify_wheel_rejects_sidecar_native_by_magic(
    tmp_path, monkeypatch, entry, magic
):
    wheel, root = build_wheel(
        tmp_path,
        extra_entries={entry: magic + b"sidecar native"},
    )
    monkeypatch.setattr(verifier, "verify_native_target", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="unexpected native libraries"):
        verifier.verify_wheel(wheel, root)


def test_verify_wheel_does_not_treat_plain_mz_resource_as_pe(
    tmp_path, monkeypatch
):
    not_pe = bytearray(132)
    not_pe[:2] = b"MZ"
    not_pe[0x3C:0x40] = (0x80).to_bytes(4, "little")
    wheel, root = build_wheel(
        tmp_path,
        extra_entries={"mosaic/data/resource.bin": bytes(not_pe)},
    )
    monkeypatch.setattr(verifier, "verify_native_target", lambda *args, **kwargs: None)

    assert verifier.verify_wheel(wheel, root) == "aarch64-unknown-linux-gnu"
