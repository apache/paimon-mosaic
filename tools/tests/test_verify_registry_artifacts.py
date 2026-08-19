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

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import verify_registry_artifacts as verifier  # noqa: E402


PROJECT = "paimon-mosaic"
VERSION = "0.3.0"


def create_artifact(tmp_path: Path, filename: str, content: bytes):
    path = tmp_path / filename
    path.write_bytes(content)
    return verifier.local_artifact(path, path.suffix)


def pypi_release(*artifacts, extra_files=()):
    urls = [
        {
            "filename": artifact.filename,
            "packagetype": "bdist_wheel",
            "digests": {"sha256": artifact.sha256},
        }
        for artifact in artifacts
    ]
    urls.extend(extra_files)
    return {
        "info": {"name": PROJECT, "version": VERSION},
        "urls": urls,
    }


def crates_release(crate, *, checksum=None):
    return {
        "version": {
            "crate": "paimon-mosaic-core",
            "num": VERSION,
            "checksum": checksum or crate.sha256,
        }
    }


def run_main(monkeypatch, *args):
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_registry_artifacts.py", *map(str, args)],
    )
    return verifier.main()


def test_pypi_matching_subset_stages_only_missing_wheels(tmp_path):
    first = create_artifact(
        tmp_path,
        "paimon_mosaic-0.3.0-py3-none-manylinux_2_28_x86_64.whl",
        b"x86 wheel",
    )
    second = create_artifact(
        tmp_path,
        "paimon_mosaic-0.3.0-py3-none-manylinux_2_28_aarch64.whl",
        b"arm wheel",
    )

    missing = verifier.validate_pypi_release(
        pypi_release(first),
        PROJECT,
        VERSION,
        [first, second],
    )
    upload = tmp_path / "upload"
    verifier.stage_missing_wheels(missing, upload)

    assert missing == [second]
    assert [path.name for path in upload.iterdir()] == [second.filename]
    assert (upload / second.filename).read_bytes() == b"arm wheel"


def test_pypi_absent_release_stages_every_wheel(tmp_path):
    wheel = create_artifact(
        tmp_path,
        "paimon_mosaic-0.3.0-py3-none-win_amd64.whl",
        b"windows wheel",
    )

    assert verifier.validate_pypi_release(
        None, PROJECT, VERSION, [wheel]
    ) == [wheel]


def test_pypi_complete_matching_release_stages_nothing(tmp_path):
    wheel = create_artifact(
        tmp_path,
        "paimon_mosaic-0.3.0-py3-none-macosx_11_0_arm64.whl",
        b"macOS wheel",
    )
    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / "stale.whl").write_bytes(b"stale")

    missing = verifier.validate_pypi_release(
        pypi_release(wheel),
        PROJECT,
        VERSION,
        [wheel],
    )
    verifier.stage_missing_wheels(missing, upload)

    assert missing == []
    assert list(upload.iterdir()) == []


def test_pypi_rejects_same_filename_with_different_sha256(tmp_path):
    wheel = create_artifact(
        tmp_path,
        "paimon_mosaic-0.3.0-py3-none-manylinux_2_28_x86_64.whl",
        b"expected",
    )
    release = pypi_release(wheel)
    release["urls"][0]["digests"]["sha256"] = hashlib.sha256(b"other").hexdigest()

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verifier.validate_pypi_release(release, PROJECT, VERSION, [wheel])


def test_pypi_rejects_unexpected_release_file(tmp_path):
    wheel = create_artifact(
        tmp_path,
        "paimon_mosaic-0.3.0-py3-none-win_amd64.whl",
        b"wheel",
    )
    extra = {
        "filename": "paimon_mosaic-0.3.0.tar.gz",
        "packagetype": "sdist",
        "digests": {"sha256": hashlib.sha256(b"sdist").hexdigest()},
    }

    with pytest.raises(ValueError, match="unexpected PyPI release file type"):
        verifier.validate_pypi_release(
            pypi_release(wheel, extra_files=[extra]),
            PROJECT,
            VERSION,
            [wheel],
        )


def test_pypi_rejects_unexpected_wheel_filename(tmp_path):
    wheel = create_artifact(
        tmp_path,
        "paimon_mosaic-0.3.0-py3-none-win_amd64.whl",
        b"wheel",
    )
    extra = {
        "filename": "paimon_mosaic-0.3.0-py3-none-linux_x86_64.whl",
        "packagetype": "bdist_wheel",
        "digests": {"sha256": hashlib.sha256(b"extra").hexdigest()},
    }

    with pytest.raises(ValueError, match="unexpected files already exist"):
        verifier.validate_pypi_release(
            pypi_release(wheel, extra_files=[extra]),
            PROJECT,
            VERSION,
            [wheel],
        )


def test_pypi_rejects_yanked_matching_wheel(tmp_path):
    wheel = create_artifact(
        tmp_path,
        "paimon_mosaic-0.3.0-py3-none-win_amd64.whl",
        b"wheel",
    )
    release = pypi_release(wheel)
    release["urls"][0]["yanked"] = True

    with pytest.raises(ValueError, match="is yanked"):
        verifier.validate_pypi_release(release, PROJECT, VERSION, [wheel])


def test_crates_io_absent_version_requires_publication(tmp_path):
    crate = create_artifact(
        tmp_path,
        "paimon-mosaic-core-0.3.0.crate",
        b"crate",
    )

    assert verifier.validate_crates_io_version(
        None, "paimon-mosaic-core", VERSION, crate
    )


def test_crates_io_matching_checksum_skips_publication(tmp_path):
    crate = create_artifact(
        tmp_path,
        "paimon-mosaic-core-0.3.0.crate",
        b"crate",
    )

    assert not verifier.validate_crates_io_version(
        crates_release(crate),
        "paimon-mosaic-core",
        VERSION,
        crate,
    )


def test_crates_io_rejects_checksum_mismatch(tmp_path):
    crate = create_artifact(
        tmp_path,
        "paimon-mosaic-core-0.3.0.crate",
        b"crate",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verifier.validate_crates_io_version(
            crates_release(crate, checksum=hashlib.sha256(b"other").hexdigest()),
            "paimon-mosaic-core",
            VERSION,
            crate,
        )


def test_crates_io_rejects_yanked_matching_crate(tmp_path):
    crate = create_artifact(
        tmp_path,
        "paimon-mosaic-core-0.3.0.crate",
        b"crate",
    )
    release = crates_release(crate)
    release["version"]["yanked"] = True

    with pytest.raises(ValueError, match="is yanked"):
        verifier.validate_crates_io_version(
            release,
            "paimon-mosaic-core",
            VERSION,
            crate,
        )


def test_github_output_records_publish_decision(tmp_path):
    output = tmp_path / "github-output"

    verifier.write_github_output(output, True, 2)
    verifier.write_github_output(output, False, 0)

    assert output.read_text(encoding="utf-8").splitlines() == [
        "publish=true",
        "missing_count=2",
        "publish=false",
        "missing_count=0",
    ]


def test_pypi_cli_records_partial_and_complete_publish_decisions(
    tmp_path, monkeypatch
):
    first = create_artifact(
        tmp_path,
        "paimon_mosaic-0.3.0-py3-none-manylinux_2_28_x86_64.whl",
        b"x86 wheel",
    )
    second = create_artifact(
        tmp_path,
        "paimon_mosaic-0.3.0-py3-none-manylinux_2_28_aarch64.whl",
        b"arm wheel",
    )
    release_json = tmp_path / "release.json"
    github_output = tmp_path / "github-output"
    upload = tmp_path / "upload"

    release_json.write_text(
        json.dumps(pypi_release(first)),
        encoding="utf-8",
    )
    assert (
        run_main(
            monkeypatch,
            "pypi",
            "--project",
            PROJECT,
            "--version",
            VERSION,
            "--wheel",
            first.path,
            second.path,
            "--upload-directory",
            upload,
            "--release-json",
            release_json,
            "--github-output",
            github_output,
        )
        == 0
    )
    assert github_output.read_text(encoding="utf-8").splitlines() == [
        "publish=true",
        "missing_count=1",
    ]
    assert [path.name for path in upload.iterdir()] == [second.filename]

    github_output.unlink()
    release_json.write_text(
        json.dumps(pypi_release(first, second)),
        encoding="utf-8",
    )
    assert (
        run_main(
            monkeypatch,
            "pypi",
            "--project",
            PROJECT,
            "--version",
            VERSION,
            "--wheel",
            first.path,
            second.path,
            "--upload-directory",
            upload,
            "--release-json",
            release_json,
            "--github-output",
            github_output,
        )
        == 0
    )
    assert github_output.read_text(encoding="utf-8").splitlines() == [
        "publish=false",
        "missing_count=0",
    ]
    assert list(upload.iterdir()) == []


def test_crates_io_cli_records_absent_and_matching_publish_decisions(
    tmp_path, monkeypatch
):
    crate = create_artifact(
        tmp_path,
        "paimon-mosaic-core-0.3.0.crate",
        b"crate",
    )
    release_json = tmp_path / "release.json"
    github_output = tmp_path / "github-output"

    assert (
        run_main(
            monkeypatch,
            "crates-io",
            "--crate-name",
            "paimon-mosaic-core",
            "--version",
            VERSION,
            "--artifact",
            crate.path,
            "--version-not-found",
            "--github-output",
            github_output,
        )
        == 0
    )
    assert github_output.read_text(encoding="utf-8").splitlines() == [
        "publish=true",
        "missing_count=1",
    ]

    github_output.unlink()
    release_json.write_text(
        json.dumps(crates_release(crate)),
        encoding="utf-8",
    )
    assert (
        run_main(
            monkeypatch,
            "crates-io",
            "--crate-name",
            "paimon-mosaic-core",
            "--version",
            VERSION,
            "--artifact",
            crate.path,
            "--version-json",
            release_json,
            "--github-output",
            github_output,
        )
        == 0
    )
    assert github_output.read_text(encoding="utf-8").splitlines() == [
        "publish=false",
        "missing_count=0",
    ]
