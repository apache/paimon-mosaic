#!/usr/bin/env python3
#
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

"""Validate signed RC/final release tags against the Apache Paimon KEYS file."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


NUMBER = r"(?:0|[1-9][0-9]*)"
FINAL_TAG = re.compile(rf"^v({NUMBER})\.({NUMBER})\.({NUMBER})$")
RC_TAG = re.compile(rf"^v({NUMBER})\.({NUMBER})\.({NUMBER})-rc([1-9][0-9]*)$")
VALID_SIGNATURE = re.compile(r"^\[GNUPG:\] VALIDSIG ([0-9A-F]+)\b", re.MULTILINE)


class TagValidationError(RuntimeError):
    """Raised when a release tag does not satisfy the release policy."""


@dataclass(frozen=True)
class ReleaseTag:
    name: str
    version: tuple[int, int, int]
    rc: int | None

    @property
    def final_name(self) -> str:
        major, minor, patch = self.version
        return f"v{major}.{minor}.{patch}"


@dataclass(frozen=True)
class VerifiedTag:
    name: str
    commit: str
    signer_fingerprint: str
    matching_rc: str | None = None


def parse_release_tag(name: str) -> ReleaseTag:
    match = FINAL_TAG.fullmatch(name)
    if match:
        return ReleaseTag(name, tuple(map(int, match.groups())), None)

    match = RC_TAG.fullmatch(name)
    if match:
        major, minor, patch, rc = map(int, match.groups())
        return ReleaseTag(name, (major, minor, patch), rc)

    raise TagValidationError(
        f"{name!r} is not a canonical vX.Y.Z or vX.Y.Z-rcN release tag"
    )


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise TagValidationError(f"{' '.join(command)} failed: {detail}")
    return result


def git_environment(
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    result = os.environ.copy() if environment is None else environment.copy()
    result["GIT_NO_REPLACE_OBJECTS"] = "1"
    return result


def git(repo: Path, *arguments: str) -> str:
    return run(
        ["git", *arguments], cwd=repo, env=git_environment()
    ).stdout.strip()


def reject_git_replacement_refs(repo: Path) -> None:
    replacements = git(
        repo,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace",
    )
    if replacements:
        raise TagValidationError(
            "repository contains Git replacement refs that could change object "
            f"identity:\n{replacements}"
        )


def inspect_annotated_tag(repo: Path, tag: str) -> str:
    ref = f"refs/tags/{tag}"
    object_type = git(repo, "cat-file", "-t", ref)
    if object_type != "tag":
        raise TagValidationError(f"{tag} must be an annotated signed tag")

    contents = git(repo, "cat-file", "-p", ref)
    header_text, separator, body = contents.partition("\n\n")
    if not separator:
        raise TagValidationError(f"{tag} has a malformed annotated tag object")

    headers: dict[str, list[str]] = {}
    for line in header_text.splitlines():
        key, separator, value = line.partition(" ")
        if not separator:
            raise TagValidationError(f"{tag} has a malformed tag header: {line!r}")
        headers.setdefault(key, []).append(value)

    def single_header(name: str) -> str:
        values = headers.get(name, [])
        if len(values) != 1:
            raise TagValidationError(
                f"{tag} must contain exactly one {name!r} header, found {values}"
            )
        return values[0]

    if single_header("type") != "commit":
        raise TagValidationError(f"{tag} must point directly to a commit")
    if single_header("tag") != tag:
        raise TagValidationError(
            f"{tag} tag object names {single_header('tag')!r} instead"
        )
    if body.count("-----BEGIN PGP SIGNATURE-----") != 1:
        raise TagValidationError(f"{tag} must contain one OpenPGP signature")

    target = single_header("object")
    if git(repo, "cat-file", "-t", target) != "commit":
        raise TagValidationError(f"{tag} target {target} is not a commit")
    resolved = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if resolved != target:
        raise TagValidationError(
            f"{tag} resolves to {resolved}, but its direct target is {target}"
        )
    return resolved


def import_keys(repo: Path, keys_file: Path, gpg_home: Path) -> dict[str, str]:
    if not keys_file.is_file() or keys_file.stat().st_size == 0:
        raise TagValidationError(f"KEYS file is missing or empty: {keys_file}")

    gpg_home.chmod(0o700)
    env = git_environment()
    env["GNUPGHOME"] = str(gpg_home)
    run(
        ["gpg", "--batch", "--no-tty", "--import", str(keys_file.resolve())],
        cwd=repo,
        env=env,
    )
    keys = run(
        ["gpg", "--batch", "--with-colons", "--list-keys"],
        cwd=repo,
        env=env,
    ).stdout
    if not any(line.startswith("pub:") for line in keys.splitlines()):
        raise TagValidationError(f"no public keys were imported from {keys_file}")
    return env


def verify_signature(repo: Path, tag: str, env: dict[str, str]) -> str:
    result = run(
        [
            "git",
            "-c",
            "gpg.format=openpgp",
            "-c",
            "gpg.openpgp.program=gpg",
            "verify-tag",
            "--raw",
            tag,
        ],
        cwd=repo,
        env=git_environment(env),
        check=False,
    )
    status = "\n".join(part for part in (result.stdout, result.stderr) if part)
    match = VALID_SIGNATURE.search(status)
    if result.returncode != 0 or not match:
        detail = status.strip() or "no OpenPGP verification status"
        raise TagValidationError(
            f"{tag} is not signed by a key in the supplied ASF Paimon KEYS: {detail}"
        )
    return match.group(1)


def release_candidate_tags(repo: Path, final: ReleaseTag) -> list[ReleaseTag]:
    refs = git(
        repo,
        "for-each-ref",
        "--format=%(refname:strip=2)",
        f"refs/tags/{final.final_name}-rc*",
    ).splitlines()
    candidates = []
    for ref in refs:
        try:
            parsed = parse_release_tag(ref)
        except TagValidationError:
            continue
        if parsed.version == final.version and parsed.rc is not None:
            candidates.append(parsed)
    return sorted(candidates, key=lambda candidate: candidate.rc or 0, reverse=True)


def validate_release_tag(
    repo: Path,
    tag_name: str,
    keys_file: Path,
    expected_commit: str = "HEAD",
) -> VerifiedTag:
    repo = repo.resolve()
    reject_git_replacement_refs(repo)
    tag = parse_release_tag(tag_name)
    commit = inspect_annotated_tag(repo, tag.name)
    expected = git(repo, "rev-parse", "--verify", f"{expected_commit}^{{commit}}")
    if commit != expected:
        raise TagValidationError(
            f"{tag.name} points to {commit}, but the checked release commit is {expected}"
        )

    with tempfile.TemporaryDirectory(prefix="paimon-release-gpg-") as directory:
        env = import_keys(repo, keys_file, Path(directory))
        signer = verify_signature(repo, tag.name, env)

        if tag.rc is not None:
            return VerifiedTag(tag.name, commit, signer)

        matching = []
        for candidate in release_candidate_tags(repo, tag):
            try:
                candidate_commit = inspect_annotated_tag(repo, candidate.name)
            except TagValidationError:
                continue
            if candidate_commit == commit:
                matching.append(candidate)
        if not matching:
            raise TagValidationError(
                f"{tag.name} must point to the same commit as at least one "
                f"{tag.final_name}-rcN tag"
            )

        failures = []
        for candidate in matching:
            try:
                verify_signature(repo, candidate.name, env)
                return VerifiedTag(tag.name, commit, signer, candidate.name)
            except TagValidationError as error:
                failures.append(str(error))
        raise TagValidationError(
            f"no matching RC tag for {tag.name} has a valid ASF Paimon signature: "
            + "; ".join(failures)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="canonical vX.Y.Z or vX.Y.Z-rcN tag")
    parser.add_argument(
        "--keys-file",
        type=Path,
        required=True,
        help="downloaded Apache Paimon KEYS file",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="Git repository to validate (default: current directory)",
    )
    parser.add_argument(
        "--expected-commit",
        default="HEAD",
        help="commit-ish that the tag must point to (default: HEAD)",
    )
    arguments = parser.parse_args()

    try:
        verified = validate_release_tag(
            arguments.repository,
            arguments.tag,
            arguments.keys_file,
            arguments.expected_commit,
        )
    except TagValidationError as error:
        print(f"release tag validation failed: {error}", file=sys.stderr)
        return 1

    message = (
        f"Verified {verified.name} -> {verified.commit} "
        f"with ASF Paimon signer {verified.signer_fingerprint}"
    )
    if verified.matching_rc:
        message += f"; matching signed RC: {verified.matching_rc}"
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
