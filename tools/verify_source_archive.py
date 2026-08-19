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

"""Create or verify a source archive against an exact Git commit tree."""

from __future__ import annotations

import argparse
import gzip
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path


SOURCE_PATHSPECS = (
    ".",
    ":(exclude).gitignore",
    ":(exclude).gitattributes",
    ":(exclude).asf.yaml",
    ":(exclude).github",
    ":(exclude)deploysettings.xml",
    ":(exclude)target",
    ":(exclude).idea",
    ":(exclude)*.iml",
    ":(exclude).DS_Store",
)

WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class ArchiveEntry:
    path: str
    kind: str
    mode: int
    link_target: str | None
    content: bytes | None


def resolve_commit(repository: Path, commit: str) -> str:
    reject_git_replacement_refs(repository)
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "rev-parse",
            "--verify",
            f"{commit}^{{commit}}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=git_environment(),
    )
    return result.stdout.strip()


def git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["GIT_ATTR_NOSYSTEM"] = "1"
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return environment


def reject_git_replacement_refs(repository: Path) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "for-each-ref",
            "--format=%(refname)",
            "refs/replace",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=git_environment(),
    )
    replacements = result.stdout.strip()
    if replacements:
        raise ValueError(
            "repository contains Git replacement refs that could change object "
            f"identity:\n{replacements}"
        )


def reject_repository_archive_attributes(repository: Path) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "rev-parse",
            "--git-path",
            "info/attributes",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=git_environment(),
    )
    attributes = Path(result.stdout.strip())
    if not attributes.is_absolute():
        attributes = repository / attributes
    if attributes.exists() and attributes.read_bytes().strip():
        raise ValueError(
            "repository-local Git attributes could change the source archive: "
            f"{attributes}"
        )


def validated_prefix(prefix: str) -> str:
    if (
        not prefix
        or not prefix.endswith("/")
        or "\x00" in prefix
        or "\\" in prefix
        or prefix.startswith("/")
        or WINDOWS_DRIVE_PATH.match(prefix)
    ):
        raise ValueError(f"invalid archive prefix: {prefix!r}")
    root = prefix[:-1]
    if (
        not root
        or "." in root.split("/")
        or ".." in root.split("/")
        or posixpath.normpath(root) != root
    ):
        raise ValueError(f"invalid archive prefix: {prefix!r}")
    return root


def validated_member_path(name: str, prefix_root: str, kind: str) -> str:
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or WINDOWS_DRIVE_PATH.match(name)
    ):
        raise ValueError(f"unsafe archive entry path: {name!r}")
    components = name.split("/")
    if "." in components or ".." in components or "" in components:
        raise ValueError(f"unsafe archive entry path: {name!r}")
    normalized = posixpath.normpath(name)
    if normalized != name:
        raise ValueError(f"non-canonical archive entry path: {name!r}")
    if normalized == prefix_root:
        if kind != "directory":
            raise ValueError(
                f"archive prefix root must be a directory: {name!r}"
            )
    elif not normalized.startswith(prefix_root + "/"):
        raise ValueError(
            f"archive entry {name!r} is outside prefix {prefix_root + '/'!r}"
        )
    return normalized


def validated_link_target(path: str, target: str, prefix_root: str) -> str:
    if (
        not target
        or "\x00" in target
        or "\\" in target
        or target.startswith("/")
        or WINDOWS_DRIVE_PATH.match(target)
    ):
        raise ValueError(f"unsafe symbolic-link target for {path!r}: {target!r}")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), target))
    if resolved != prefix_root and not resolved.startswith(prefix_root + "/"):
        raise ValueError(
            f"symbolic link {path!r} escapes the archive prefix: {target!r}"
        )
    return target


def archive_entries(archive: tarfile.TarFile, prefix: str) -> dict[str, ArchiveEntry]:
    prefix_root = validated_prefix(prefix)
    entries: dict[str, ArchiveEntry] = {}
    normalized_names: dict[str, str] = {}
    for member in archive.getmembers():
        if member.isfile():
            kind = "file"
        elif member.isdir():
            kind = "directory"
        elif member.issym():
            kind = "symlink"
        else:
            raise ValueError(
                f"unsupported archive entry type for {member.name!r}: "
                f"{member.type!r}"
            )

        raw_name = member.name
        path = validated_member_path(raw_name, prefix_root, kind)
        if raw_name in entries:
            raise ValueError(f"duplicate archive entry path: {raw_name!r}")
        previous = normalized_names.get(path)
        if previous is not None:
            raise ValueError(
                f"duplicate normalized archive entry paths: "
                f"{previous!r} and {raw_name!r}"
            )

        content = None
        link_target = None
        if kind == "file":
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read archive file entry: {raw_name!r}")
            content = extracted.read()
            if len(content) != member.size:
                raise ValueError(
                    f"archive file size mismatch for {raw_name!r}: "
                    f"read {len(content)}, header declares {member.size}"
                )
        elif kind == "symlink":
            link_target = validated_link_target(path, member.linkname, prefix_root)

        entries[raw_name] = ArchiveEntry(
            path=path,
            kind=kind,
            mode=member.mode & 0o7777,
            link_target=link_target,
            content=content,
        )
        normalized_names[path] = raw_name
    return entries


def read_archive(path: Path, prefix: str) -> dict[str, ArchiveEntry]:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            return archive_entries(archive, prefix)
    except (tarfile.TarError, EOFError) as error:
        raise ValueError(f"cannot read source archive {path}: {error}") from error


def read_source_archive(
    path: Path, prefix: str
) -> tuple[dict[str, ArchiveEntry], str | None]:
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            embedded_commit = archive.pax_headers.get("comment")
            return archive_entries(archive, prefix), embedded_commit
    except (tarfile.TarError, EOFError) as error:
        raise ValueError(
            f"cannot read gzip source archive {path}: {error}"
        ) from error


def write_git_archive(
    repository: Path, commit: str, prefix: str, output: Path
) -> None:
    reject_repository_archive_attributes(repository)
    command = [
        "git",
        "-C",
        str(repository),
        "-c",
        "tar.umask=0002",
        "-c",
        "core.attributesFile=/dev/null",
        "archive",
        "--format=tar",
        f"--prefix={prefix}",
        commit,
        "--",
        *SOURCE_PATHSPECS,
    ]
    with output.open("wb") as destination:
        subprocess.run(
            command,
            check=True,
            stdout=destination,
            stderr=subprocess.PIPE,
            env=git_environment(),
        )


def expected_entries(
    repository: Path, commit: str, prefix: str
) -> dict[str, ArchiveEntry]:
    with tempfile.TemporaryDirectory(prefix="paimon-source-archive-") as directory:
        archive_path = Path(directory) / "expected.tar"
        write_git_archive(repository, commit, prefix, archive_path)
        try:
            with tarfile.open(archive_path, mode="r:") as archive:
                return archive_entries(archive, prefix)
        except (tarfile.TarError, EOFError) as error:
            raise ValueError(
                f"cannot read expected Git archive {archive_path}: {error}"
            ) from error


def compare_entries(
    actual: dict[str, ArchiveEntry],
    expected: dict[str, ArchiveEntry],
) -> None:
    actual_paths = set(actual)
    expected_paths = set(expected)
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    if missing:
        raise ValueError(f"source archive is missing entries: {missing}")
    if unexpected:
        raise ValueError(f"source archive has unexpected entries: {unexpected}")

    for path in sorted(expected_paths):
        actual_entry = actual[path]
        expected_entry = expected[path]
        if actual_entry.path != expected_entry.path:
            raise ValueError(f"normalized path differs for {path!r}")
        if actual_entry.kind != expected_entry.kind:
            raise ValueError(
                f"entry type differs for {path!r}: "
                f"found {actual_entry.kind}, expected {expected_entry.kind}"
            )
        if actual_entry.mode != expected_entry.mode:
            raise ValueError(
                f"entry mode differs for {path!r}: "
                f"found {actual_entry.mode:#o}, expected {expected_entry.mode:#o}"
            )
        if actual_entry.link_target != expected_entry.link_target:
            raise ValueError(
                f"symbolic-link target differs for {path!r}: "
                f"found {actual_entry.link_target!r}, "
                f"expected {expected_entry.link_target!r}"
            )
        if actual_entry.content != expected_entry.content:
            raise ValueError(f"file content differs for {path!r}")


def verify_archive(
    archive: Path,
    repository: Path,
    commit: str,
    prefix: str,
) -> str:
    validated_prefix(prefix)
    resolved_commit = resolve_commit(repository, commit)
    actual, embedded_commit = read_source_archive(archive, prefix)
    if embedded_commit != resolved_commit:
        raise ValueError(
            f"embedded Git commit differs: found {embedded_commit!r}, "
            f"expected {resolved_commit!r}"
        )
    expected = expected_entries(repository, resolved_commit, prefix)
    compare_entries(actual, expected)
    return resolved_commit


def create_archive(
    output: Path,
    repository: Path,
    commit: str,
    prefix: str,
) -> str:
    validated_prefix(prefix)
    resolved_commit = resolve_commit(repository, commit)
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_output: Path | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix="paimon-source-create-"
        ) as directory:
            raw_tar = Path(directory) / "source.tar"
            write_git_archive(repository, resolved_commit, prefix, raw_tar)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output.parent,
                prefix=output.name + ".",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_output = Path(destination.name)
                with raw_tar.open("rb") as source, gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=destination,
                    mtime=0,
                ) as compressed:
                    shutil.copyfileobj(source, compressed)

        assert temporary_output is not None
        verify_archive(
            temporary_output,
            repository,
            resolved_commit,
            prefix,
        )
        os.chmod(temporary_output, 0o644)
        os.replace(temporary_output, output)
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
    return resolved_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repository", type=Path, default=Path("."))
        command.add_argument("--commit", required=True)
        command.add_argument("--prefix", required=True)

    create = subparsers.add_parser("create")
    add_common_arguments(create)
    create.add_argument("--output", required=True, type=Path)

    verify = subparsers.add_parser("verify")
    add_common_arguments(verify)
    verify.add_argument("--archive", required=True, type=Path)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "create":
            commit = create_archive(
                args.output,
                args.repository,
                args.commit,
                args.prefix,
            )
            print(f"created {args.output} from Git commit {commit}")
        else:
            commit = verify_archive(
                args.archive,
                args.repository,
                args.commit,
                args.prefix,
            )
            print(f"verified {args.archive} against Git commit {commit}")
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            detail = (
                error.stderr.decode(errors="replace")
                if isinstance(error.stderr, bytes)
                else error.stderr
            )
        else:
            detail = str(error)
        print(f"source archive verification failed: {detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
