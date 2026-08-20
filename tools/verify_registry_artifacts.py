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

"""Verify resumable PyPI and crates.io publication against local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class LocalArtifact:
    path: Path
    filename: str
    sha256: str


def normalized_project_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def local_artifact(path: Path, expected_suffix: str) -> LocalArtifact:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"local artifact does not exist: {path}") from error
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"local artifact is not a regular file: {path}")
    if (
        "\\" in path.name
        or path.name != str(PurePosixPath(path.name))
        or not path.name.endswith(expected_suffix)
    ):
        raise ValueError(
            f"local artifact has an invalid {expected_suffix} filename: {path.name!r}"
        )
    return LocalArtifact(path, path.name, sha256_file(path))


def load_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return parsed


def validate_digest(value: object, description: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{description} is not a SHA-256 digest: {value!r}")
    return value.lower()


def validate_pypi_release(
    release: dict[str, Any] | None,
    project: str,
    version: str,
    wheels: list[LocalArtifact],
) -> list[LocalArtifact]:
    if not wheels:
        raise ValueError("at least one local wheel is required")

    local_by_name: dict[str, LocalArtifact] = {}
    for wheel in wheels:
        if wheel.filename in local_by_name:
            raise ValueError(f"duplicate local wheel filename: {wheel.filename}")
        local_by_name[wheel.filename] = wheel

    if release is None:
        return sorted(wheels, key=lambda artifact: artifact.filename)

    info = release.get("info")
    if not isinstance(info, dict):
        raise ValueError("PyPI release JSON is missing the info object")
    remote_name = info.get("name")
    if not isinstance(remote_name, str) or normalized_project_name(
        remote_name
    ) != normalized_project_name(project):
        raise ValueError(
            f"PyPI release project {remote_name!r} does not match {project!r}"
        )
    remote_version = info.get("version")
    if remote_version != version:
        raise ValueError(
            f"PyPI release version {remote_version!r} does not match {version!r}"
        )

    urls = release.get("urls")
    if not isinstance(urls, list):
        raise ValueError("PyPI release JSON is missing the urls list")

    remote_by_name: dict[str, str] = {}
    for index, item in enumerate(urls):
        if not isinstance(item, dict):
            raise ValueError(f"PyPI release file #{index} is not an object")
        filename = item.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or filename != PurePosixPath(filename).name
            or "\\" in filename
        ):
            raise ValueError(
                f"PyPI release file #{index} has an invalid filename: {filename!r}"
            )
        if filename in remote_by_name:
            raise ValueError(f"PyPI release lists {filename!r} more than once")
        if item.get("packagetype") != "bdist_wheel":
            raise ValueError(
                f"unexpected PyPI release file type for {filename!r}: "
                f"{item.get('packagetype')!r}"
            )
        yanked = item.get("yanked", False)
        if not isinstance(yanked, bool):
            raise ValueError(
                f"PyPI release yanked flag for {filename!r} is not boolean: "
                f"{yanked!r}"
            )
        if yanked:
            raise ValueError(
                f"PyPI release file {filename!r} is yanked; unyank it before "
                "resuming publication"
            )
        digests = item.get("digests")
        if not isinstance(digests, dict):
            raise ValueError(f"PyPI release file {filename!r} has no digests object")
        remote_by_name[filename] = validate_digest(
            digests.get("sha256"),
            f"PyPI release SHA-256 for {filename!r}",
        )

    unexpected = sorted(set(remote_by_name) - set(local_by_name))
    if unexpected:
        raise ValueError(f"unexpected files already exist on PyPI: {unexpected}")

    for filename, remote_digest in remote_by_name.items():
        local_digest = local_by_name[filename].sha256
        if not hmac.compare_digest(remote_digest, local_digest):
            raise ValueError(
                f"PyPI SHA-256 mismatch for {filename}: "
                f"registry has {remote_digest}, local artifact has {local_digest}"
            )

    return [
        local_by_name[filename]
        for filename in sorted(set(local_by_name) - set(remote_by_name))
    ]


def stage_missing_wheels(
    wheels: list[LocalArtifact],
    upload_directory: Path,
    *,
    source_wheels: list[LocalArtifact] | None = None,
) -> None:
    upload_path = (
        upload_directory
        if upload_directory.is_absolute()
        else Path.cwd() / upload_directory
    )
    for component in (upload_path, *upload_path.parents):
        if component.is_symlink():
            raise ValueError(
                "upload directory path must not contain a symbolic link: "
                f"{component}"
            )

    output = upload_directory.resolve()
    for wheel in source_wheels if source_wheels is not None else wheels:
        artifact = wheel.path.resolve()
        if output == artifact or output in artifact.parents:
            raise ValueError(
                f"upload directory {upload_directory} contains local artifact "
                f"{wheel.path}"
            )

    if upload_directory.exists():
        if not upload_directory.is_dir():
            raise ValueError(
                f"upload directory is not a directory: {upload_directory}"
            )
        shutil.rmtree(upload_directory)
    upload_directory.mkdir(parents=True)
    for wheel in wheels:
        shutil.copy2(wheel.path, upload_directory / wheel.filename)


def validate_crates_io_version(
    release: dict[str, Any] | None,
    crate_name: str,
    version: str,
    crate: LocalArtifact,
) -> bool:
    expected_filename = f"{crate_name}-{version}.crate"
    if crate.filename != expected_filename:
        raise ValueError(
            f"local crate filename {crate.filename!r} does not match "
            f"{expected_filename!r}"
        )
    if release is None:
        return True

    remote = release.get("version")
    if not isinstance(remote, dict):
        raise ValueError("crates.io version JSON is missing the version object")
    if remote.get("crate") != crate_name:
        raise ValueError(
            f"crates.io crate {remote.get('crate')!r} does not match {crate_name!r}"
        )
    if remote.get("num") != version:
        raise ValueError(
            f"crates.io version {remote.get('num')!r} does not match {version!r}"
        )
    remote_digest = validate_digest(
        remote.get("checksum"),
        f"crates.io checksum for {crate.filename!r}",
    )
    yanked = remote.get("yanked", False)
    if not isinstance(yanked, bool):
        raise ValueError(
            f"crates.io yanked flag for {crate.filename!r} is not boolean: "
            f"{yanked!r}"
        )
    if yanked:
        raise ValueError(
            f"crates.io {crate_name} {version} is yanked; unyank it before "
            "resuming publication"
        )
    if not hmac.compare_digest(remote_digest, crate.sha256):
        raise ValueError(
            f"crates.io SHA-256 mismatch for {crate.filename}: "
            f"registry has {remote_digest}, local artifact has {crate.sha256}"
        )
    return False


def write_github_output(path: Path | None, publish: bool, missing_count: int) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(f"publish={'true' if publish else 'false'}\n")
        output.write(f"missing_count={missing_count}\n")


def response_json(
    json_path: Path | None, explicitly_not_found: bool, description: str
) -> dict[str, Any] | None:
    if explicitly_not_found:
        return None
    if json_path is None:
        raise ValueError(f"{description} response was not provided")
    return load_json(json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="registry", required=True)

    pypi = subparsers.add_parser("pypi")
    pypi.add_argument("--project", required=True)
    pypi.add_argument("--version", required=True)
    pypi.add_argument("--wheel", nargs="+", required=True, type=Path)
    pypi.add_argument("--upload-directory", type=Path)
    pypi_response = pypi.add_mutually_exclusive_group(required=True)
    pypi_response.add_argument("--release-json", type=Path)
    pypi_response.add_argument("--release-not-found", action="store_true")
    pypi.add_argument("--github-output", type=Path)

    crates = subparsers.add_parser("crates-io")
    crates.add_argument("--crate-name", required=True)
    crates.add_argument("--version", required=True)
    crates.add_argument("--artifact", required=True, type=Path)
    crates_response = crates.add_mutually_exclusive_group(required=True)
    crates_response.add_argument("--version-json", type=Path)
    crates_response.add_argument("--version-not-found", action="store_true")
    crates.add_argument("--github-output", type=Path)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.registry == "pypi":
            wheels = [local_artifact(path, ".whl") for path in args.wheel]
            release = response_json(
                args.release_json,
                args.release_not_found,
                "PyPI release",
            )
            missing = validate_pypi_release(
                release,
                args.project,
                args.version,
                wheels,
            )
            if args.upload_directory is not None:
                stage_missing_wheels(
                    missing,
                    args.upload_directory,
                    source_wheels=wheels,
                )
            write_github_output(args.github_output, bool(missing), len(missing))
            if missing:
                print(
                    f"PyPI {args.project} {args.version} is missing "
                    f"{len(missing)} verified wheel(s): "
                    + ", ".join(wheel.filename for wheel in missing)
                )
            else:
                print(
                    f"PyPI {args.project} {args.version} already contains "
                    f"all {len(wheels)} verified wheel(s)"
                )
        else:
            crate = local_artifact(args.artifact, ".crate")
            release = response_json(
                args.version_json,
                args.version_not_found,
                "crates.io version",
            )
            publish = validate_crates_io_version(
                release,
                args.crate_name,
                args.version,
                crate,
            )
            write_github_output(args.github_output, publish, int(publish))
            if publish:
                print(
                    f"crates.io {args.crate_name} {args.version} is not published"
                )
            else:
                print(
                    f"crates.io {args.crate_name} {args.version} already matches "
                    f"{crate.filename}"
                )
    except (OSError, ValueError) as error:
        print(f"registry artifact verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
