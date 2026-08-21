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

import copy
import gzip
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import verify_source_archive as verifier  # noqa: E402


PREFIX = "paimon-mosaic-0.3.0/"


def run(command, cwd, env=None):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def initialize_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("source contents\n", encoding="utf-8")
    (repo / "script.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (repo / "script.sh").chmod(0o755)
    (repo / ".gitignore").write_text("ignored\n", encoding="utf-8")
    (repo / ".github").mkdir()
    (repo / ".github/workflow.yml").write_text("ignored\n", encoding="utf-8")
    os.symlink("README.md", repo / "README-link")

    run(["git", "init", "-q"], cwd=repo)
    run(["git", "config", "user.name", "Archive Test"], cwd=repo)
    run(["git", "config", "user.email", "archive-test@example.invalid"], cwd=repo)
    run(["git", "add", "."], cwd=repo)
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_DATE": "2026-08-01T12:34:56Z",
            "GIT_COMMITTER_DATE": "2026-08-01T12:34:56Z",
        }
    )
    run(["git", "commit", "-q", "-m", "fixture"], cwd=repo, env=env)
    commit = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.decode().strip()
    return repo, commit


def write_tar(path, members):
    with tarfile.open(path, "w") as archive:
        for info, content in members:
            archive.addfile(info, io.BytesIO(content) if content is not None else None)


def regular_file(name, content=b"content", mode=0o664):
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.mode = mode
    info.size = len(content)
    return info, content


def repack_source_archive(path, *, drop=None, extra=None):
    with tarfile.open(path, "r:gz") as source:
        pax_headers = dict(source.pax_headers)
        members = []
        for member in source.getmembers():
            if member.name == drop:
                continue
            content = source.extractfile(member).read() if member.isfile() else None
            members.append((copy.copy(member), content))
    if extra is not None:
        members.append(regular_file(extra))

    raw_tar = io.BytesIO()
    with tarfile.open(
        fileobj=raw_tar,
        mode="w",
        format=tarfile.PAX_FORMAT,
        pax_headers=pax_headers,
    ) as destination:
        for member, content in members:
            destination.addfile(
                member,
                io.BytesIO(content) if content is not None else None,
            )
    with path.open("wb") as destination:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=destination, mtime=0
        ) as compressed:
            compressed.write(raw_tar.getvalue())


def test_create_and_verify_exact_git_tree(tmp_path):
    repo, commit = initialize_repo(tmp_path)
    archive = tmp_path / "source.tgz"

    assert verifier.create_archive(archive, repo, commit, PREFIX) == commit
    assert archive.stat().st_mode & 0o777 == 0o644
    assert verifier.verify_archive(archive, repo, commit, PREFIX) == commit

    with tarfile.open(archive, "r:gz") as source:
        names = {member.name for member in source.getmembers()}
    assert f"{PREFIX}README.md" in names
    assert f"{PREFIX}README-link" in names
    assert f"{PREFIX}script.sh" in names
    assert f"{PREFIX}.gitignore" not in names
    assert not any(name.startswith(f"{PREFIX}.github") for name in names)


def test_archive_verification_rejects_same_tree_from_different_commit(tmp_path):
    repo, first_commit = initialize_repo(tmp_path)
    archive = tmp_path / "source.tgz"
    verifier.create_archive(archive, repo, first_commit, PREFIX)

    run(["git", "commit", "-q", "--allow-empty", "-m", "empty"], cwd=repo)
    second_commit = (
        run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.decode().strip()
    )
    assert first_commit != second_commit

    with pytest.raises(ValueError, match="embedded Git commit differs"):
        verifier.verify_archive(archive, repo, second_commit, PREFIX)


def test_archive_verification_rejects_trailing_gzip_data(tmp_path):
    repo, commit = initialize_repo(tmp_path)
    archive = tmp_path / "source.tgz"
    verifier.create_archive(archive, repo, commit, PREFIX)
    with archive.open("ab") as output:
        output.write(b"trailing data")

    with pytest.raises(ValueError, match="trailing data"):
        verifier.verify_archive(archive, repo, commit, PREFIX)


def test_archive_verification_rejects_oversized_gzip_stream(
    tmp_path, monkeypatch
):
    archive = tmp_path / "oversized.tgz"
    archive.write_bytes(gzip.compress(b"x" * 2048))
    max_lengths = []
    decompressobj = verifier.zlib.decompressobj

    class RecordingDecompressor:
        def __init__(self, *args, **kwargs):
            self.delegate = decompressobj(*args, **kwargs)

        def decompress(self, data, max_length=0):
            max_lengths.append(max_length)
            return self.delegate.decompress(data, max_length)

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    monkeypatch.setattr(verifier.zlib, "decompressobj", RecordingDecompressor)
    monkeypatch.setattr(
        verifier,
        "MAX_SOURCE_TAR_SIZE",
        1024,
        raising=False,
    )

    with pytest.raises(ValueError, match="uncompressed size limit"):
        verifier.read_source_archive(archive, PREFIX)
    assert max_lengths == [1025]


def test_archive_verification_rejects_second_tar_segment(tmp_path):
    repo, commit = initialize_repo(tmp_path)
    archive = tmp_path / "source.tgz"
    verifier.create_archive(archive, repo, commit, PREFIX)

    extra_tar = tmp_path / "extra.tar"
    write_tar(
        extra_tar,
        [regular_file(f"{PREFIX}UNVERIFIED.txt", b"unverified")],
    )
    combined = gzip.decompress(archive.read_bytes()) + extra_tar.read_bytes()
    with archive.open("wb") as destination:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=destination, mtime=0
        ) as output:
            output.write(combined)

    with pytest.raises(ValueError, match="trailing tar data"):
        verifier.verify_archive(archive, repo, commit, PREFIX)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (("missing", "missing entries"), ("unexpected", "unexpected entries")),
)
def test_archive_verification_rejects_tree_entry_set_changes(
    tmp_path, mutation, error
):
    repo, commit = initialize_repo(tmp_path)
    archive = tmp_path / "source.tgz"
    verifier.create_archive(archive, repo, commit, PREFIX)

    if mutation == "missing":
        repack_source_archive(archive, drop=f"{PREFIX}README.md")
    else:
        repack_source_archive(archive, extra=f"{PREFIX}UNEXPECTED.txt")

    with pytest.raises(ValueError, match=error):
        verifier.verify_archive(archive, repo, commit, PREFIX)


def test_archive_creation_rejects_git_replacement_refs(tmp_path):
    repo, first_commit = initialize_repo(tmp_path)
    (repo / "README.md").write_text("replacement contents\n", encoding="utf-8")
    run(["git", "add", "README.md"], cwd=repo)
    run(["git", "commit", "-q", "-m", "replacement"], cwd=repo)
    second_commit = (
        run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.decode().strip()
    )
    run(["git", "replace", first_commit, second_commit], cwd=repo)

    with pytest.raises(ValueError, match="replacement refs"):
        verifier.create_archive(
            tmp_path / "source.tgz", repo, first_commit, PREFIX
        )


def test_archive_creation_uses_fixed_tar_umask(tmp_path):
    repo, commit = initialize_repo(tmp_path)
    archive = tmp_path / "source.tgz"
    run(["git", "config", "tar.umask", "0077"], cwd=repo)

    verifier.create_archive(archive, repo, commit, PREFIX)
    run(["git", "config", "tar.umask", "0000"], cwd=repo)
    assert verifier.verify_archive(archive, repo, commit, PREFIX) == commit

    with tarfile.open(archive, "r:gz") as source:
        assert source.getmember(f"{PREFIX}README.md").mode == 0o664
        assert source.getmember(f"{PREFIX}script.sh").mode == 0o775


def test_archive_creation_ignores_autocrlf(tmp_path):
    repo, commit = initialize_repo(tmp_path)
    archive = tmp_path / "source.tgz"
    run(["git", "config", "core.autocrlf", "true"], cwd=repo)

    verifier.create_archive(archive, repo, commit, PREFIX)

    with tarfile.open(archive, "r:gz") as source:
        assert source.extractfile(f"{PREFIX}README.md").read() == b"source contents\n"


def test_archive_creation_rejects_repository_local_attributes(tmp_path):
    repo, commit = initialize_repo(tmp_path)
    archive = tmp_path / "source.tgz"
    attributes = Path(
        run(
            ["git", "rev-parse", "--git-path", "info/attributes"],
            cwd=repo,
        )
        .stdout.decode()
        .strip()
    )
    if not attributes.is_absolute():
        attributes = repo / attributes
    attributes.parent.mkdir(parents=True, exist_ok=True)
    attributes.write_text("README.md export-ignore\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="repository-local Git attributes"
    ):
        verifier.create_archive(archive, repo, commit, PREFIX)


def test_archive_verification_rejects_unsafe_entry_path(tmp_path):
    archive = tmp_path / "unsafe.tar"
    write_tar(archive, [regular_file("../escape")])

    with pytest.raises(ValueError, match="unsafe archive entry path"):
        verifier.read_archive(archive, PREFIX)


def test_archive_verification_rejects_duplicate_entry(tmp_path):
    archive = tmp_path / "duplicate.tar"
    path = f"{PREFIX}README.md"
    write_tar(
        archive,
        [
            regular_file(path, b"first"),
            regular_file(path, b"second"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate archive entry"):
        verifier.read_archive(archive, PREFIX)


def test_archive_verification_rejects_escaping_symlink(tmp_path):
    archive = tmp_path / "symlink.tar"
    info = tarfile.TarInfo(f"{PREFIX}link")
    info.type = tarfile.SYMTYPE
    info.mode = 0o777
    info.linkname = "../../outside"
    write_tar(archive, [(info, None)])

    with pytest.raises(ValueError, match="escapes the archive prefix"):
        verifier.read_archive(archive, PREFIX)


@pytest.mark.parametrize(
    "target",
    ("C:outside", "C:../outside", "C:/outside"),
)
def test_archive_verification_rejects_windows_drive_symlink(tmp_path, target):
    archive = tmp_path / "symlink.tar"
    info = tarfile.TarInfo(f"{PREFIX}link")
    info.type = tarfile.SYMTYPE
    info.mode = 0o777
    info.linkname = target
    write_tar(archive, [(info, None)])

    with pytest.raises(ValueError, match="unsafe symbolic-link target"):
        verifier.read_archive(archive, PREFIX)


@pytest.mark.parametrize("prefix", ("C:release/", "C:/release/"))
def test_archive_verification_rejects_windows_drive_prefix(prefix):
    with pytest.raises(ValueError, match="invalid archive prefix"):
        verifier.validated_prefix(prefix)


def test_compare_entries_checks_type_mode_link_and_content():
    expected = {
        "path": verifier.ArchiveEntry("path", "file", 0o664, None, b"same")
    }

    for changed, error in (
        (
            verifier.ArchiveEntry("path", "directory", 0o664, None, None),
            "entry type differs",
        ),
        (
            verifier.ArchiveEntry("path", "file", 0o755, None, b"same"),
            "entry mode differs",
        ),
        (
            verifier.ArchiveEntry("path", "file", 0o664, "other", b"same"),
            "symbolic-link target differs",
        ),
        (
            verifier.ArchiveEntry("path", "file", 0o664, None, b"diff"),
            "file content differs",
        ),
    ):
        with pytest.raises(ValueError, match=error):
            verifier.compare_entries({"path": changed}, expected)
