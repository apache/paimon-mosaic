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

from __future__ import annotations

from pathlib import Path
import sys
import zipfile

import pytest


TOOLS_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIRECTORY))

import verify_release_artifacts as verifier  # noqa: E402


def write_archive(path: Path, files: dict[str, str | bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def java_files() -> dict[str, str | bytes]:
    reports = {
        f"META-INF/licenses/{target}/THIRD-PARTY-LICENSES.html": target
        for target in verifier.TARGETS
    }
    return {
        "META-INF/LICENSE": "\n".join(reports) + "\n",
        "META-INF/NOTICE": "Apache Paimon Mosaic\n",
        "META-INF/DEPENDENCIES": "Mosaic\n",
        **{
            native: b"native"
            for native in verifier.JAVA_NATIVE_BY_TARGET.values()
        },
        **reports,
    }


def python_files(target: str) -> dict[str, str | bytes]:
    platform_tags = {
        "x86_64-unknown-linux-gnu": "manylinux_2_28_x86_64",
        "aarch64-unknown-linux-gnu": "manylinux_2_28_aarch64",
        "aarch64-apple-darwin": "macosx_11_0_arm64",
        "x86_64-pc-windows-msvc": "win_amd64",
    }
    return {
        verifier.PYTHON_NATIVE_BY_TARGET[target]: b"native",
        "mosaic/LICENSE": "THIRD-PARTY-LICENSES.html\n",
        "mosaic/NOTICE": "Apache Paimon Mosaic\n",
        "mosaic/THIRD-PARTY-LICENSES.html": target,
        "paimon_mosaic-0.3.0.dist-info/WHEEL": (
            "Wheel-Version: 1.0\n"
            f"Tag: py3-none-{platform_tags[target]}\n"
        ),
    }


def wheel_name(tmp_path: Path, target: str) -> Path:
    suffixes = {
        "x86_64-unknown-linux-gnu": "manylinux_2_28_x86_64",
        "aarch64-unknown-linux-gnu": "manylinux_2_28_aarch64",
        "aarch64-apple-darwin": "macosx_11_0_arm64",
        "x86_64-pc-windows-msvc": "win_amd64",
    }
    return tmp_path / f"paimon_mosaic-0.3.0-py3-none-{suffixes[target]}.whl"


def test_java_verifier_accepts_complete_multi_platform_jar(tmp_path: Path) -> None:
    jar = tmp_path / "mosaic-0.3.0.jar"
    write_archive(jar, java_files())

    verifier.verify_java_jar(jar)


def test_java_verifier_rejects_missing_third_party_report(tmp_path: Path) -> None:
    jar = tmp_path / "mosaic-0.3.0.jar"
    files = java_files()
    files.pop(
        "META-INF/licenses/aarch64-unknown-linux-gnu/"
        "THIRD-PARTY-LICENSES.html"
    )
    write_archive(jar, files)

    with pytest.raises(ValueError, match="expected exactly one"):
        verifier.verify_java_jar(jar)


@pytest.mark.parametrize("target", verifier.TARGETS)
def test_python_verifier_accepts_platform_wheel(
    tmp_path: Path, target: str
) -> None:
    wheel = wheel_name(tmp_path, target)
    write_archive(wheel, python_files(target))

    assert verifier.verify_python_wheel(wheel) == target


def test_python_verifier_rejects_native_wheel_without_legal_files(
    tmp_path: Path,
) -> None:
    target = "aarch64-unknown-linux-gnu"
    wheel = wheel_name(tmp_path, target)
    write_archive(
        wheel,
        {verifier.PYTHON_NATIVE_BY_TARGET[target]: b"native"},
    )

    with pytest.raises(ValueError, match="mosaic/LICENSE"):
        verifier.verify_python_wheel(wheel)


def test_python_verifier_rejects_narrow_cpython_abi_tag(tmp_path: Path) -> None:
    target = "aarch64-unknown-linux-gnu"
    wheel = tmp_path / (
        "paimon_mosaic-0.3.0-cp39-cp39-manylinux_2_28_aarch64.whl"
    )
    write_archive(wheel, python_files(target))

    with pytest.raises(ValueError, match="expected py3-none"):
        verifier.verify_python_wheel(wheel)


def test_python_verifier_rejects_mismatched_internal_tag(tmp_path: Path) -> None:
    target = "aarch64-unknown-linux-gnu"
    wheel = wheel_name(tmp_path, target)
    files = python_files(target)
    files["paimon_mosaic-0.3.0.dist-info/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Tag: cp39-cp39-manylinux_2_28_aarch64\n"
    )
    write_archive(wheel, files)

    with pytest.raises(ValueError, match="WHEEL tags"):
        verifier.verify_python_wheel(wheel)


def test_python_target_set_rejects_missing_platform() -> None:
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        verifier.verify_python_target_set(
            [
                "x86_64-unknown-linux-gnu",
                "aarch64-unknown-linux-gnu",
                "aarch64-apple-darwin",
            ]
        )
