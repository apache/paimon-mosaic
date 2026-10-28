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

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = ROOT / "tools/create_source_release.sh"
SOURCE_VERIFIER = ROOT / "tools/verify_source_archive.py"
VERSION_VERIFIER = ROOT / "tools/verify_release_versions.py"
VERSION = "0.3.0"
RC_TAG = f"v{VERSION}-rc1"


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture(scope="session")
def signing_home(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    if shutil.which("gpg") is None:
        pytest.skip("gpg is required for source release tests")
    home = tmp_path_factory.mktemp("gnupg")
    home.chmod(0o700)
    env = os.environ.copy()
    env["GNUPGHOME"] = str(home)
    run(
        [
            "gpg",
            "--batch",
            "--passphrase",
            "",
            "--quick-gen-key",
            "Release Test <release-test@example.invalid>",
            "rsa2048",
            "sign",
            "0",
        ],
        cwd=home,
        env=env,
    )
    listing = run(
        ["gpg", "--batch", "--with-colons", "--list-secret-keys"],
        cwd=home,
        env=env,
    ).stdout.decode()
    fingerprint = next(
        line.split(":")[9]
        for line in listing.splitlines()
        if line.startswith("fpr:")
    )
    return home, fingerprint


def initialize_release_repo(
    tmp_path: Path,
    signing_home: tuple[Path, str],
    *,
    python_version: str = VERSION,
    source_script: str | None = None,
) -> tuple[Path, dict[str, str], str]:
    repo = tmp_path / "repo"
    tools = repo / "tools"
    tools.mkdir(parents=True)
    shutil.copy2(SOURCE_SCRIPT, tools / SOURCE_SCRIPT.name)
    if source_script is not None:
        write(tools / SOURCE_SCRIPT.name, source_script)
    shutil.copy2(SOURCE_VERIFIER, tools / SOURCE_VERIFIER.name)
    shutil.copy2(VERSION_VERIFIER, tools / VERSION_VERIFIER.name)
    shutil.copy2(ROOT / "tools/release_checks.py", tools / "release_checks.py")

    write(
        repo / "Cargo.toml",
        '[workspace]\nmembers = ["core"]\nresolver = "2"\n',
    )
    write(
        repo / "core/Cargo.toml",
        f'[package]\nname = "paimon-mosaic-core"\nversion = "{VERSION}"\n',
    )
    write(
        repo / "Cargo.lock",
        "version = 4\n\n"
        "[[package]]\n"
        'name = "paimon-mosaic-core"\n'
        f'version = "{VERSION}"\n',
    )
    write(
        repo / "java/pom.xml",
        '<?xml version="1.0"?>\n'
        '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
        "  <modelVersion>4.0.0</modelVersion>\n"
        "  <groupId>org.apache.paimon</groupId>\n"
        "  <artifactId>mosaic</artifactId>\n"
        f"  <version>{VERSION}</version>\n"
        "</project>\n",
    )
    write(
        repo / "python/pyproject.toml",
        '[project]\nname = "paimon-mosaic"\n'
        f'version = "{python_version}"\n',
    )
    write(repo / "README.md", "source release fixture\n")
    write(repo / ".gitignore", "tools/release/\ntools/.source-release.*\n")

    run(["git", "init", "-q"], cwd=repo)
    run(["git", "config", "user.name", "Release Test"], cwd=repo)
    run(
        ["git", "config", "user.email", "release-test@example.invalid"],
        cwd=repo,
    )
    home, fingerprint = signing_home
    run(["git", "config", "user.signingkey", fingerprint], cwd=repo)
    run(["git", "add", "."], cwd=repo)
    run(["git", "commit", "-q", "-m", "release fixture"], cwd=repo)
    commit = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.decode().strip()

    env = os.environ.copy()
    env.update(
        {
            "GNUPGHOME": str(home),
            "RELEASE_VERSION": VERSION,
            "RC_TAG": RC_TAG,
        }
    )
    run(
        [
            "git",
            "tag",
            "-s",
            "-u",
            fingerprint,
            "-m",
            f"Release candidate {RC_TAG}",
            RC_TAG,
        ],
        cwd=repo,
        env=env,
    )
    return repo, env, commit


def run_script(
    repo: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["bash", SOURCE_SCRIPT.name],
        cwd=repo / "tools",
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def output(result: subprocess.CompletedProcess[bytes]) -> str:
    return (result.stdout + result.stderr).decode("utf-8", errors="replace")


def artifact_paths(repo: Path) -> tuple[Path, Path, Path]:
    archive = (
        repo
        / "tools/release"
        / f"apache-paimon-mosaic-{VERSION}-src.tgz"
    )
    return archive, Path(f"{archive}.asc"), Path(f"{archive}.sha512")


def test_creates_verified_artifacts_from_signed_head_tag(
    tmp_path: Path,
    signing_home: tuple[Path, str],
) -> None:
    repo, env, git_commit = initialize_release_repo(tmp_path, signing_home)

    result = run_script(repo, env)

    assert result.returncode == 0, output(result)
    archive, signature, checksum = artifact_paths(repo)
    assert archive.is_file()
    assert signature.is_file()
    assert checksum.is_file()
    assert hashlib.sha512(archive.read_bytes()).hexdigest() in checksum.read_text(
        encoding="utf-8"
    )
    run(["gpg", "--verify", str(signature), str(archive)], cwd=repo, env=env)
    run(
        [
            sys.executable,
            str(SOURCE_VERIFIER),
            "verify",
            "--repository",
            str(repo),
            "--commit",
            git_commit,
            "--prefix",
            f"paimon-mosaic-{VERSION}/",
            "--archive",
            str(archive),
        ],
        cwd=repo,
        env=env,
    )


def test_missing_tomllib_fails_before_creating_any_archive(tmp_path, signing_home):
    repo, env, _ = initialize_release_repo(tmp_path, signing_home)
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python3"
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)
    env["PATH"] = str(binary) + os.pathsep + env["PATH"]
    result = run_script(repo, env)
    assert result.returncode != 0
    assert "python3 with tomllib" in output(result)
    assert not (repo / "tools/release").exists()
    assert not list((repo / "tools").glob(".source-release.*"))


@pytest.mark.parametrize("missing", ("RELEASE_VERSION", "RC_TAG"))
def test_requires_release_version_and_rc_tag(
    tmp_path: Path,
    signing_home: tuple[Path, str],
    missing: str,
) -> None:
    repo, env, _ = initialize_release_repo(tmp_path, signing_home)
    env.pop(missing)

    result = run_script(repo, env)

    assert result.returncode != 0
    assert f"{missing} is unset" in output(result)


def test_rejects_rc_tag_that_does_not_point_to_head(
    tmp_path: Path,
    signing_home: tuple[Path, str],
) -> None:
    repo, env, _ = initialize_release_repo(tmp_path, signing_home)
    write(repo / "README.md", "new head\n")
    run(["git", "add", "README.md"], cwd=repo)
    run(["git", "commit", "-q", "-m", "move head"], cwd=repo)

    result = run_script(repo, env)

    assert result.returncode != 0
    assert "does not resolve to current HEAD" in output(result)


def test_rejects_dirty_worktree(
    tmp_path: Path,
    signing_home: tuple[Path, str],
) -> None:
    repo, env, _ = initialize_release_repo(tmp_path, signing_home)
    write(repo / "README.md", "dirty\n")

    result = run_script(repo, env)

    assert result.returncode != 0
    assert "clean Git worktree" in output(result)


def test_rejects_dirty_intended_worktree_with_foreign_git_environment(
    tmp_path: Path,
    signing_home: tuple[Path, str],
) -> None:
    repo, env, _ = initialize_release_repo(
        tmp_path / "intended",
        signing_home,
    )
    foreign_repo, _, _ = initialize_release_repo(
        tmp_path / "foreign",
        signing_home,
    )
    write(repo / "README.md", "dirty intended worktree\n")
    env.update(
        {
            "GIT_DIR": str(foreign_repo / ".git"),
            "GIT_WORK_TREE": str(foreign_repo),
        }
    )

    result = run_script(repo, env)

    assert result.returncode != 0
    assert "clean Git worktree" in output(result)


def test_existing_artifact_is_not_overwritten(
    tmp_path: Path,
    signing_home: tuple[Path, str],
) -> None:
    repo, env, _ = initialize_release_repo(tmp_path, signing_home)
    archive, signature, checksum = artifact_paths(repo)
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"existing artifact")

    result = run_script(repo, env)

    assert result.returncode != 0
    assert "already exists" in output(result)
    assert archive.read_bytes() == b"existing artifact"
    assert not signature.exists()
    assert not checksum.exists()


def test_release_version_must_match_rc_tag(
    tmp_path: Path,
    signing_home: tuple[Path, str],
) -> None:
    repo, env, _ = initialize_release_repo(tmp_path, signing_home)
    env["RELEASE_VERSION"] = "9.9.9"

    result = run_script(repo, env)

    assert result.returncode != 0
    assert "does not match RELEASE_VERSION" in output(result)


def test_signed_tag_with_cross_component_version_mismatch_is_rejected(
    tmp_path: Path,
    signing_home: tuple[Path, str],
) -> None:
    repo, env, _ = initialize_release_repo(tmp_path, signing_home)
    write(
        repo / "python/pyproject.toml",
        '[project]\nname = "paimon-mosaic"\nversion = "0.3.1"\n',
    )
    run(["git", "add", "python/pyproject.toml"], cwd=repo)
    run(["git", "commit", "-q", "-m", "mismatched Python version"], cwd=repo)

    env["RC_TAG"] = f"v{VERSION}-rc2"
    _, fingerprint = signing_home
    run(
        [
            "git",
            "tag",
            "-s",
            "-u",
            fingerprint,
            "-m",
            f"Release candidate {env['RC_TAG']}",
            env["RC_TAG"],
        ],
        cwd=repo,
        env=env,
    )

    result = run_script(repo, env)

    assert result.returncode != 0
    assert "python/pyproject.toml" in output(result)
    assert "expected '0.3.0'" in output(result)
    assert not (repo / "tools/release").exists()


def test_unsigned_tag_with_pgp_armor_text_is_rejected(
    tmp_path: Path,
    signing_home: tuple[Path, str],
) -> None:
    repo, env, _ = initialize_release_repo(tmp_path, signing_home)
    env["RC_TAG"] = f"v{VERSION}-rc2"
    run(
        [
            "git",
            "tag",
            "-a",
            "-m",
            "unsigned release candidate\n\n"
            "-----BEGIN PGP SIGNATURE-----\n"
            "not a signature\n"
            "-----END PGP SIGNATURE-----",
            env["RC_TAG"],
        ],
        cwd=repo,
    )

    result = run_script(repo, env)

    assert result.returncode != 0
    assert "not a locally verifiable GPG-signed tag" in output(result)


@pytest.mark.parametrize("injection", ("count", "parameters", "local"))
def test_injected_gpg_program_cannot_approve_forged_tag(
    tmp_path: Path,
    signing_home: tuple[Path, str],
    injection: str,
) -> None:
    repo, env, commit = initialize_release_repo(tmp_path, signing_home)
    forged_tag_object = subprocess.run(
        ["git", "hash-object", "-t", "tag", "-w", "--stdin"],
        cwd=repo,
        check=True,
        input=(
            f"object {commit}\n"
            "type commit\n"
            f"tag {RC_TAG}\n"
            "tagger Release Test <release-test@example.invalid> 0 +0000\n\n"
            "forged release tag\n\n"
            "-----BEGIN PGP SIGNATURE-----\n"
            "invalid\n"
            "-----END PGP SIGNATURE-----\n"
        ).encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.decode().strip()
    run(
        ["git", "update-ref", f"refs/tags/{RC_TAG}", forged_tag_object],
        cwd=repo,
    )
    fake_gpg = tmp_path / "fake-gpg"
    write(
        fake_gpg,
        "#!/bin/sh\n"
        "printf '%s\\n' \\\n"
        "  '[GNUPG:] NEWSIG' \\\n"
        "  '[GNUPG:] GOODSIG DEADBEEF Release Test' \\\n"
        "  '[GNUPG:] VALIDSIG 0123456789ABCDEF0123456789ABCDEF01234567 "
        "2026-08-28 0 4 0 1 10 00 "
        "0123456789ABCDEF0123456789ABCDEF01234567'\n"
        "exit 0\n",
    )
    fake_gpg.chmod(0o755)
    if injection == "count":
        env.update(
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "gpg.program",
                "GIT_CONFIG_VALUE_0": str(fake_gpg),
            }
        )
    elif injection == "parameters":
        env["GIT_CONFIG_PARAMETERS"] = (
            f"'gpg.program'='{fake_gpg}'"
        )
    else:
        run(
            ["git", "config", "gpg.program", str(fake_gpg)],
            cwd=repo,
        )

    result = run_script(repo, env)

    assert result.returncode != 0
    assert "not a locally verifiable GPG-signed tag" in output(result)
    assert not (repo / "tools/release").exists()


def test_rejects_alias_for_different_signed_tag_name(
    tmp_path: Path,
    signing_home: tuple[Path, str],
) -> None:
    repo, env, _ = initialize_release_repo(tmp_path, signing_home)
    tag_object = run(
        ["git", "rev-parse", f"{RC_TAG}^{{tag}}"],
        cwd=repo,
    ).stdout.decode().strip()
    alias = f"v{VERSION}-rc2"
    run(
        ["git", "update-ref", f"refs/tags/{alias}", tag_object],
        cwd=repo,
    )
    env["RC_TAG"] = alias

    result = run_script(repo, env)

    assert result.returncode != 0
    assert f"signed tag object names {RC_TAG}, not {alias}" in output(result)


def test_verifies_the_frozen_tag_object(
    tmp_path: Path,
    signing_home: tuple[Path, str],
) -> None:
    repo, env, commit = initialize_release_repo(tmp_path, signing_home)
    signed_tag_object = run(
        ["git", "rev-parse", f"{RC_TAG}^{{tag}}"],
        cwd=repo,
    ).stdout.decode().strip()
    forged_tag_object = subprocess.run(
        ["git", "hash-object", "-t", "tag", "-w", "--stdin"],
        cwd=repo,
        check=True,
        input=(
            f"object {commit}\n"
            "type commit\n"
            f"tag {RC_TAG}\n"
            "tagger Release Test <release-test@example.invalid> 0 +0000\n\n"
            "forged release tag\n\n"
            "-----BEGIN PGP SIGNATURE-----\n"
            "invalid\n"
            "-----END PGP SIGNATURE-----\n"
        ).encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.decode().strip()
    run(
        ["git", "update-ref", f"refs/tags/{RC_TAG}", forged_tag_object],
        cwd=repo,
    )

    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "git"
    write(
        wrapper,
        "#!/bin/sh\n"
        'for argument in "$@"; do\n'
        '  if [ "$argument" = "verify-tag" ] && [ ! -e "$MOVE_MARKER" ]; then\n'
        '    : > "$MOVE_MARKER"\n'
        '    "$REAL_GIT" -C "$RELEASE_REPO" update-ref '
        '"refs/tags/$RC_TAG" "$SIGNED_TAG_OBJECT"\n'
        "    break\n"
        "  fi\n"
        "done\n"
        'exec "$REAL_GIT" "$@"\n',
    )
    wrapper.chmod(0o755)
    move_marker = tmp_path / "tag-moved-before-verification"
    env.update(
        {
            "MOVE_MARKER": str(move_marker),
            "PATH": f"{wrapper_dir}{os.pathsep}{env['PATH']}",
            "REAL_GIT": shutil.which("git") or "git",
            "RELEASE_REPO": str(repo),
            "SIGNED_TAG_OBJECT": signed_tag_object,
        }
    )

    result = run_script(repo, env)

    assert move_marker.is_file()
    assert result.returncode != 0
    assert "not a locally verifiable GPG-signed tag" in output(result)
    assert not (repo / "tools/release").exists()


def test_tag_ref_move_after_verification_does_not_change_archive(
    tmp_path: Path,
    signing_home: tuple[Path, str],
) -> None:
    repo, env, signed_commit = initialize_release_repo(tmp_path, signing_home)
    write(repo / "README.md", "alternate commit\n")
    run(["git", "add", "README.md"], cwd=repo)
    run(["git", "commit", "-q", "-m", "alternate commit"], cwd=repo)
    alternate_commit = run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
    ).stdout.decode().strip()
    run(["git", "reset", "--hard", signed_commit], cwd=repo)

    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "python3"
    write(
        wrapper,
        "#!/bin/sh\n"
        'if [ "$1" = "tools/verify_source_archive.py" ] &&\n'
        '  [ "$2" = "create" ] && [ ! -e "$MOVE_MARKER" ]; then\n'
        '  : > "$MOVE_MARKER"\n'
        '  git -C "$RELEASE_REPO" update-ref "refs/tags/$RC_TAG" '
        '"$ALTERNATE_COMMIT"\n'
        "fi\n"
        'exec "$REAL_PYTHON" "$@"\n',
    )
    wrapper.chmod(0o755)
    move_marker = tmp_path / "tag-moved"
    env.update(
        {
            "ALTERNATE_COMMIT": alternate_commit,
            "MOVE_MARKER": str(move_marker),
            "PATH": f"{wrapper_dir}{os.pathsep}{env['PATH']}",
            "REAL_PYTHON": sys.executable,
            "RELEASE_REPO": str(repo),
        }
    )

    result = run_script(repo, env)

    assert result.returncode == 0, output(result)
    assert move_marker.is_file()
    archive, _, _ = artifact_paths(repo)
    with tarfile.open(archive, "r:gz") as source:
        assert source.pax_headers["comment"] == signed_commit
        readme = source.extractfile(f"paimon-mosaic-{VERSION}/README.md")
        assert readme is not None
        assert readme.read() == b"source release fixture\n"


@pytest.mark.parametrize(
    ("platform", "checksum_command"),
    (("Linux", "sha512sum"), ("Darwin", "shasum")),
)
def test_checksum_failure_is_not_swallowed_without_errexit(
    tmp_path: Path,
    signing_home: tuple[Path, str],
    platform: str,
    checksum_command: str,
) -> None:
    production_script = SOURCE_SCRIPT.read_text(encoding="utf-8")
    assert "set -o errexit\n" in production_script
    source_script = production_script.replace(
        "set -o errexit\n",
        "",
        1,
    )
    repo, env, _ = initialize_release_repo(
        tmp_path,
        signing_home,
        source_script=source_script,
    )
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    checksum_marker = tmp_path / "checksum-failed"
    wrapper = wrapper_dir / checksum_command
    write(
        wrapper,
        "#!/bin/sh\n"
        'if [ ! -e "$CHECKSUM_MARKER" ]; then\n'
        '  : > "$CHECKSUM_MARKER"\n'
        "  exit 42\n"
        "fi\n"
        "exit 0\n",
    )
    wrapper.chmod(0o755)
    uname = wrapper_dir / "uname"
    write(uname, '#!/bin/sh\nprintf "%s\\n" "$TEST_UNAME"\n')
    uname.chmod(0o755)
    env["PATH"] = f"{wrapper_dir}{os.pathsep}{env['PATH']}"
    env["CHECKSUM_MARKER"] = str(checksum_marker)
    env["TEST_UNAME"] = platform

    result = run_script(repo, env)

    assert checksum_marker.is_file()
    assert result.returncode != 0
    assert not (repo / "tools/release").exists()


def test_version_failure_is_not_swallowed_without_errexit(
    tmp_path: Path,
    signing_home: tuple[Path, str],
) -> None:
    production_script = SOURCE_SCRIPT.read_text(encoding="utf-8")
    assert "set -o errexit\n" in production_script
    source_script = production_script.replace(
        "set -o errexit\n",
        "",
        1,
    )
    repo, env, _ = initialize_release_repo(
        tmp_path,
        signing_home,
        python_version="0.3.1",
        source_script=source_script,
    )

    result = run_script(repo, env)

    assert result.returncode != 0
    assert "python/pyproject.toml" in output(result)
    assert not (repo / "tools/release").exists()
