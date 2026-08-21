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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import validate_release_tag as validator


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def generate_key(tmp_path: Path, identity: str) -> tuple[Path, str, Path]:
    home = tmp_path / identity.replace(" ", "-")
    home.mkdir(mode=0o700)
    env = os.environ.copy()
    env["GNUPGHOME"] = str(home)
    run(
        [
            "gpg",
            "--batch",
            "--pinentry-mode",
            "loopback",
            "--passphrase",
            "",
            "--quick-generate-key",
            f"{identity} <{identity.replace(' ', '.')}@example.test>",
            "ed25519",
            "sign",
            "0",
        ],
        cwd=tmp_path,
        env=env,
    )
    listing = run(
        ["gpg", "--batch", "--with-colons", "--list-secret-keys"],
        cwd=tmp_path,
        env=env,
    )
    fingerprint = next(
        line.split(":")[9] for line in listing.splitlines() if line.startswith("fpr:")
    )
    keys = tmp_path / f"{identity.replace(' ', '-')}.keys"
    exported = run(
        ["gpg", "--batch", "--armor", "--export", fingerprint],
        cwd=tmp_path,
        env=env,
    )
    keys.write_text(exported + "\n", encoding="utf-8")
    return home, fingerprint, keys


def short_temporary_root(platform_name: str = os.name) -> Path:
    if platform_name == "nt":
        return Path(tempfile.gettempdir()).resolve()
    return Path("/tmp").resolve()


@pytest.fixture(scope="module")
def signing_keys():
    temporary_root = short_temporary_root()
    with tempfile.TemporaryDirectory(
        prefix="pm-gpg-", dir=temporary_root
    ) as directory:
        root = Path(directory).resolve()
        trusted = generate_key(root, "Trusted Release")
        untrusted = generate_key(root, "Untrusted Release")
        yield trusted, untrusted


def test_signing_key_homes_use_a_short_canonical_temporary_root(signing_keys):
    temporary_root = short_temporary_root()

    for home, _, _ in signing_keys:
        assert home == home.resolve()
        assert home.parent.parent == temporary_root

    assert short_temporary_root("posix") == Path("/tmp").resolve()
    assert short_temporary_root("nt") == Path(tempfile.gettempdir()).resolve()


def repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run(["git", "init", "-q"], cwd=repo)
    run(["git", "config", "user.name", "Release Test"], cwd=repo)
    run(["git", "config", "user.email", "release@example.test"], cwd=repo)
    commit(repo, "first")
    return repo


def commit(repo: Path, contents: str) -> str:
    (repo / "payload").write_text(contents, encoding="utf-8")
    run(["git", "add", "payload"], cwd=repo)
    run(["git", "commit", "-q", "-m", contents], cwd=repo)
    return run(["git", "rev-parse", "HEAD"], cwd=repo)


def sign_tag(repo: Path, tag: str, home: Path, fingerprint: str) -> None:
    env = os.environ.copy()
    env["GNUPGHOME"] = str(home)
    run(
        [
            "git",
            "-c",
            "gpg.program=gpg",
            "-c",
            f"user.signingkey={fingerprint}",
            "tag",
            "-s",
            "-m",
            tag,
            tag,
        ],
        cwd=repo,
        env=env,
    )


@pytest.mark.parametrize(
    "tag",
    [
        "1.2.3",
        "v01.2.3",
        "v1.02.3",
        "v1.2.03",
        "v1.2.3-rc0",
        "v1.2.3-rc01",
        "v1.2.3-RC1",
        "v1.2.3-extra",
    ],
)
def test_parse_release_tag_rejects_noncanonical_names(tag):
    with pytest.raises(validator.TagValidationError, match="not a canonical"):
        validator.parse_release_tag(tag)


def test_signed_rc_and_final_on_same_commit_are_accepted(tmp_path, signing_keys):
    (home, fingerprint, keys), _ = signing_keys
    repo = repository(tmp_path)
    sign_tag(repo, "v1.2.3-rc1", home, fingerprint)
    sign_tag(repo, "v1.2.3", home, fingerprint)

    rc = validator.validate_release_tag(repo, "v1.2.3-rc1", keys)
    final = validator.validate_release_tag(repo, "v1.2.3", keys)

    assert rc.matching_rc is None
    assert final.commit == rc.commit
    assert final.matching_rc == "v1.2.3-rc1"


def test_main_returns_success_and_failure_status(
    tmp_path, signing_keys, monkeypatch, capsys
):
    (home, fingerprint, keys), _ = signing_keys
    repo = repository(tmp_path)
    tag = "v1.2.4-rc1"
    sign_tag(repo, tag, home, fingerprint)
    arguments = [
        "validate_release_tag.py",
        tag,
        "--keys-file",
        str(keys),
        "--repository",
        str(repo),
    ]

    monkeypatch.setattr(sys, "argv", arguments)
    assert validator.main() == 0
    capsys.readouterr()

    commit(repo, "after tag")
    monkeypatch.setattr(sys, "argv", arguments)
    assert validator.main() == 1
    captured = capsys.readouterr()
    assert "release tag validation failed" in captured.err


def test_invalid_extra_rc_does_not_hide_a_valid_matching_rc(tmp_path, signing_keys):
    (home, fingerprint, keys), _ = signing_keys
    repo = repository(tmp_path)
    sign_tag(repo, "v1.3.0-rc1", home, fingerprint)
    run(["git", "config", "tag.gpgSign", "true"], cwd=repo)
    env = os.environ.copy()
    env["GIT_EDITOR"] = "false"
    run(
        ["git", "-c", "tag.gpgSign=false", "tag", "v1.3.0-rc2"],
        cwd=repo,
        env=env,
    )
    sign_tag(repo, "v1.3.0", home, fingerprint)

    final = validator.validate_release_tag(repo, "v1.3.0", keys)

    assert final.matching_rc == "v1.3.0-rc1"


def test_final_tag_requires_matching_rc_signed_by_supplied_keys(
    tmp_path, signing_keys
):
    (trusted_home, trusted_fingerprint, trusted_keys), (
        untrusted_home,
        untrusted_fingerprint,
        _,
    ) = signing_keys
    repo = repository(tmp_path)
    sign_tag(
        repo,
        "v1.4.0-rc1",
        untrusted_home,
        untrusted_fingerprint,
    )
    sign_tag(repo, "v1.4.0", trusted_home, trusted_fingerprint)

    with pytest.raises(
        validator.TagValidationError,
        match="no matching RC tag.*valid ASF Paimon signature",
    ):
        validator.validate_release_tag(
            repo,
            "v1.4.0",
            trusted_keys,
        )


def test_final_tag_rejects_rc_on_a_different_commit(tmp_path, signing_keys):
    (home, fingerprint, keys), _ = signing_keys
    repo = repository(tmp_path)
    sign_tag(repo, "v2.0.0-rc1", home, fingerprint)
    commit(repo, "final changed")
    sign_tag(repo, "v2.0.0", home, fingerprint)

    with pytest.raises(validator.TagValidationError, match="same commit"):
        validator.validate_release_tag(repo, "v2.0.0", keys)


def test_tag_validation_rejects_git_replacement_refs(tmp_path, signing_keys):
    (home, fingerprint, keys), _ = signing_keys
    repo = repository(tmp_path)
    sign_tag(repo, "v2.1.0-rc1", home, fingerprint)
    commit(repo, "replacement")
    sign_tag(repo, "v2.1.0-rc2", home, fingerprint)
    first_tag = run(["git", "rev-parse", "refs/tags/v2.1.0-rc1"], cwd=repo)
    second_tag = run(["git", "rev-parse", "refs/tags/v2.1.0-rc2"], cwd=repo)
    run(["git", "replace", first_tag, second_tag], cwd=repo)

    with pytest.raises(validator.TagValidationError, match="replacement refs"):
        validator.validate_release_tag(repo, "v2.1.0-rc1", keys)


def test_signature_must_match_a_key_in_supplied_keys(tmp_path, signing_keys):
    (home, fingerprint, _), (_, _, unrelated_keys) = signing_keys
    repo = repository(tmp_path)
    sign_tag(repo, "v3.0.0-rc1", home, fingerprint)

    with pytest.raises(validator.TagValidationError, match="supplied ASF Paimon KEYS"):
        validator.validate_release_tag(repo, "v3.0.0-rc1", unrelated_keys)


def test_lightweight_tag_is_rejected(tmp_path, signing_keys):
    (_, _, keys), _ = signing_keys
    repo = repository(tmp_path)
    run(["git", "config", "tag.gpgSign", "true"], cwd=repo)
    env = os.environ.copy()
    env["GIT_EDITOR"] = "false"
    run(
        ["git", "-c", "tag.gpgSign=false", "tag", "v4.0.0-rc1"],
        cwd=repo,
        env=env,
    )

    with pytest.raises(validator.TagValidationError, match="annotated signed"):
        validator.validate_release_tag(repo, "v4.0.0-rc1", keys)
