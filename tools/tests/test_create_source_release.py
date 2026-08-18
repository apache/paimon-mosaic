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

import gzip
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = REPO_ROOT / "tools" / "create_source_release.sh"
SOURCE_VERIFIER = REPO_ROOT / "tools" / "verify_source_archive.py"
VERSION = "0.3.0"


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        input=input_bytes,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def initialize_release_repo(tmp_path: Path) -> tuple[Path, dict[str, str], str]:
    repo = tmp_path / "repo"
    tools = repo / "tools"
    fake_bin = tmp_path / "bin"
    tools.mkdir(parents=True)
    fake_bin.mkdir()

    shutil.copy2(SOURCE_SCRIPT, tools / SOURCE_SCRIPT.name)
    shutil.copy2(SOURCE_VERIFIER, tools / SOURCE_VERIFIER.name)
    for verifier in (
        "verify_release_versions.py",
        "dependencies.py",
        "generate_license_reports.py",
    ):
        write(tools / verifier, "#!/usr/bin/env python3\n")

    for required_file in (
        "Cargo.lock",
        "LICENSE",
        "NOTICE",
        "core/LICENSE",
        "core/NOTICE",
        "DEPENDENCIES.rust.tsv",
    ):
        write(repo / required_file, f"{required_file}\n")
    write(repo / ".gitignore", "tools/release/\n")

    write(
        fake_bin / "cargo",
        "#!/usr/bin/env bash\nset -euo pipefail\nexit 0\n",
    )
    write(
        fake_bin / "gpg",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ " $* " == *" --detach-sig "* ]]; then
  archive="${@: -1}"
  printf 'test signature\\n' > "${archive}.asc"
fi
""",
    )
    for executable in fake_bin.iterdir():
        executable.chmod(0o755)

    run(["git", "init", "-q"], cwd=repo)
    run(["git", "config", "user.name", "Release Test"], cwd=repo)
    run(["git", "config", "user.email", "release-test@example.invalid"], cwd=repo)
    run(["git", "add", "."], cwd=repo)
    commit_env = os.environ.copy()
    commit_env.update(
        {
            "GIT_AUTHOR_DATE": "2026-08-01T12:34:56Z",
            "GIT_COMMITTER_DATE": "2026-08-01T12:34:56Z",
        }
    )
    run(["git", "commit", "-q", "-m", "release fixture"], cwd=repo, env=commit_env)
    commit = (
        run(["git", "rev-parse", "HEAD"], cwd=repo)
        .stdout.decode("ascii")
        .strip()
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["RELEASE_VERSION"] = VERSION
    return repo, env, commit


def test_requires_release_version_without_nounset_trace() -> None:
    env = os.environ.copy()
    env.pop("RELEASE_VERSION", None)
    result = subprocess.run(
        ["bash", SOURCE_SCRIPT.name],
        cwd=SOURCE_SCRIPT.parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    output = (result.stdout + result.stderr).decode("utf-8")
    assert result.returncode != 0
    assert "RELEASE_VERSION is unset" in output
    assert "unbound variable" not in output


def test_source_archive_is_commit_bound_and_reproducible(tmp_path: Path) -> None:
    repo, env, commit = initialize_release_repo(tmp_path)
    script = repo / "tools" / SOURCE_SCRIPT.name
    archive = (
        repo
        / "tools"
        / "release"
        / f"apache-paimon-mosaic-{VERSION}-src.tgz"
    )

    run(["bash", script.name], cwd=script.parent, env=env)
    first_digest = hashlib.sha512(archive.read_bytes()).hexdigest()
    assert archive.stat().st_mode & 0o777 == 0o644
    tar_bytes = gzip.decompress(archive.read_bytes())
    embedded_commit = (
        run(
            ["git", "get-tar-commit-id"],
            cwd=repo,
            input_bytes=tar_bytes,
        )
        .stdout.decode("ascii")
        .strip()
    )

    run(["bash", script.name], cwd=script.parent, env=env)
    second_digest = hashlib.sha512(archive.read_bytes()).hexdigest()

    assert embedded_commit == commit
    assert first_digest == second_digest


def test_same_length_mutation_keeps_pax_commit_but_fails_tree_verification(
    tmp_path: Path,
) -> None:
    repo, env, commit = initialize_release_repo(tmp_path)
    script = repo / "tools" / SOURCE_SCRIPT.name
    archive = (
        repo
        / "tools"
        / "release"
        / f"apache-paimon-mosaic-{VERSION}-src.tgz"
    )
    prefix = f"paimon-mosaic-{VERSION}/"

    run(["bash", script.name], cwd=script.parent, env=env)
    tar_bytes = bytearray(gzip.decompress(archive.read_bytes()))
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as source:
        member = source.getmember(f"{prefix}NOTICE")
        original = source.extractfile(member).read()
    replacement = bytes([original[0] ^ 1]) + original[1:]
    assert len(replacement) == len(original)
    tar_bytes[member.offset_data : member.offset_data + member.size] = replacement
    archive.write_bytes(gzip.compress(bytes(tar_bytes), mtime=0))

    embedded_commit = (
        run(
            ["git", "get-tar-commit-id"],
            cwd=repo,
            input_bytes=gzip.decompress(archive.read_bytes()),
        )
        .stdout.decode("ascii")
        .strip()
    )
    assert embedded_commit == commit

    result = subprocess.run(
        [
            sys.executable,
            "tools/verify_source_archive.py",
            "verify",
            "--repository",
            ".",
            "--commit",
            commit,
            "--prefix",
            prefix,
            "--archive",
            str(archive),
        ],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert b"file content differs" in result.stderr
