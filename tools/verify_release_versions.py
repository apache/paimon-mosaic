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

"""Verify every published component uses the intended release version."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
DEPENDENCY_GROUPS = ("dependencies", "dev-dependencies", "build-dependencies")
RUST_PACKAGES = {
    "core/Cargo.toml": "paimon-mosaic-core",
    "ffi/Cargo.toml": "paimon-mosaic-ffi",
    "jni/Cargo.toml": "paimon-mosaic-jni",
    "cli/Cargo.toml": "paimon-mosaic-cli",
}


@dataclass(frozen=True)
class WorkspacePackage:
    name: str
    version: str
    manifest: Path


@dataclass(frozen=True)
class PathDependency:
    manifest: Path
    group: tuple[str, ...]
    alias: str
    requirement: str | None
    target: WorkspacePackage


@dataclass(frozen=True)
class TomlAssignment:
    table: tuple[str, ...]
    key: tuple[str, ...]
    start: int
    end: int
    equals: int


def load_toml(relative_path: str, root: Path = ROOT) -> dict:
    with (root / relative_path).open("rb") as file:
        return tomllib.load(file)


def java_version(root: Path = ROOT) -> str:
    pom = ET.parse(root / "java/pom.xml").getroot()
    namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
    version = pom.find("m:version", namespace)
    if version is None or not version.text:
        raise ValueError("java/pom.xml has no direct project version")
    return version.text.strip()


def cargo_metadata(root: Path) -> dict:
    result = subprocess.run(
        [
            "cargo",
            "metadata",
            "--format-version",
            "1",
            "--no-deps",
            "--offline",
            "--manifest-path",
            str(root / "Cargo.toml"),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"cargo metadata failed: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"cargo metadata returned invalid JSON: {error}") from error


def workspace_packages(root: Path) -> dict[Path, WorkspacePackage]:
    metadata = cargo_metadata(root)
    member_ids = set(metadata["workspace_members"])
    packages = {}
    for package in metadata["packages"]:
        if package["id"] not in member_ids:
            continue
        manifest = Path(package["manifest_path"]).resolve()
        packages[manifest] = WorkspacePackage(
            name=package["name"],
            version=package["version"],
            manifest=manifest,
        )
    if not packages:
        raise ValueError("Cargo workspace has no member packages")
    return packages


def dependency_groups(data: dict) -> list[tuple[tuple[str, ...], dict]]:
    groups = []
    for group_name in DEPENDENCY_GROUPS:
        group = data.get(group_name)
        if isinstance(group, dict):
            groups.append(((group_name,), group))

    workspace = data.get("workspace")
    if isinstance(workspace, dict):
        group = workspace.get("dependencies")
        if isinstance(group, dict):
            groups.append((("workspace", "dependencies"), group))

    targets = data.get("target")
    if isinstance(targets, dict):
        for target_name, target in targets.items():
            if not isinstance(target, dict):
                continue
            for group_name in DEPENDENCY_GROUPS:
                group = target.get(group_name)
                if isinstance(group, dict):
                    groups.append((("target", target_name, group_name), group))
    return groups


def path_dependencies(
    root: Path, packages: dict[Path, WorkspacePackage]
) -> list[PathDependency]:
    manifests = {root.resolve() / "Cargo.toml", *packages.keys()}
    dependencies = []
    for manifest in sorted(manifests):
        with manifest.open("rb") as file:
            data = tomllib.load(file)
        for group_path, group in dependency_groups(data):
            for alias, specification in group.items():
                if not isinstance(specification, dict) or "path" not in specification:
                    continue
                dependency_path = specification["path"]
                if not isinstance(dependency_path, str):
                    raise ValueError(
                        f"{manifest}: path dependency {alias} has a non-string path"
                    )
                target_manifest = (
                    manifest.parent / dependency_path / "Cargo.toml"
                ).resolve()
                target = packages.get(target_manifest)
                if target is None:
                    continue
                requirement = specification.get("version")
                if requirement is not None and not isinstance(requirement, str):
                    raise ValueError(
                        f"{manifest}: path dependency {alias} has a non-string version"
                    )
                dependencies.append(
                    PathDependency(
                        manifest=manifest,
                        group=group_path,
                        alias=alias,
                        requirement=requirement,
                        target=target,
                    )
                )
    return dependencies


def cargo_requirement_accepts(requirement: str, version: str) -> bool:
    """Ask Cargo itself whether a version satisfies one of its requirements."""
    with tempfile.TemporaryDirectory(prefix="paimon-cargo-version-check-") as directory:
        root = Path(directory)
        target = root / "target-package"
        consumer = root / "consumer"
        (target / "src").mkdir(parents=True)
        (consumer / "src").mkdir(parents=True)
        (target / "src/lib.rs").write_text("", encoding="utf-8")
        (consumer / "src/lib.rs").write_text("", encoding="utf-8")
        (target / "Cargo.toml").write_text(
            "[package]\n"
            'name = "path-version-target"\n'
            f"version = {json.dumps(version)}\n"
            'edition = "2021"\n',
            encoding="utf-8",
        )
        (consumer / "Cargo.toml").write_text(
            "[package]\n"
            'name = "path-version-consumer"\n'
            'version = "0.0.0"\n'
            'edition = "2021"\n'
            "\n"
            "[workspace]\n"
            "\n"
            "[dependencies]\n"
            "target = { "
            'package = "path-version-target", '
            'path = "../target-package", '
            f"version = {json.dumps(requirement)}"
            " }\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                "cargo",
                "metadata",
                "--format-version",
                "1",
                "--offline",
                "--manifest-path",
                str(consumer / "Cargo.toml"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    return result.returncode == 0


def path_dependency_failures(root: Path) -> list[str]:
    root = root.resolve()
    packages = workspace_packages(root)
    failures = []
    compatibility = {}
    for dependency in path_dependencies(root, packages):
        if dependency.requirement is None:
            continue
        key = (dependency.requirement, dependency.target.version)
        if key not in compatibility:
            compatibility[key] = cargo_requirement_accepts(*key)
        if compatibility[key]:
            continue
        manifest = dependency.manifest.relative_to(root)
        target_manifest = dependency.target.manifest.relative_to(root)
        dependency_name = dependency.alias
        if dependency.alias != dependency.target.name:
            dependency_name += f" (package {dependency.target.name})"
        failures.append(
            f"{manifest}: path dependency {dependency_name} requires "
            f"{dependency.requirement}, which does not accept "
            f"{dependency.target.version} from {target_manifest}"
        )
    return failures


def probe_path(value: object, marker: str, prefix: tuple[str, ...] = ()) -> tuple[str, ...]:
    if isinstance(value, dict):
        for key, nested in value.items():
            found = probe_path(nested, marker, (*prefix, key))
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = probe_path(nested, marker, prefix)
            if found:
                return found
    elif value == marker:
        return prefix
    return ()


def parse_key_path(source: str) -> tuple[str, ...]:
    marker = "__paimon_mosaic_version_probe__"
    parsed = tomllib.loads(f"{source} = {json.dumps(marker)}\n")
    path = probe_path(parsed, marker)
    if not path:
        raise ValueError(f"could not parse TOML key: {source!r}")
    return path


def parse_table_path(source: str) -> tuple[str, ...]:
    marker = "__paimon_mosaic_table_probe__"
    parsed = tomllib.loads(f"{source}\n__probe__ = {json.dumps(marker)}\n")
    path = probe_path(parsed, marker)
    if not path or path[-1] != "__probe__":
        raise ValueError(f"could not parse TOML table: {source!r}")
    return path[:-1]


def assignment_equals(source: str) -> int:
    quote = None
    escaped = False
    for index, character in enumerate(source):
        if quote == '"':
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quote = None
        elif quote == "'":
            if character == "'":
                quote = None
        elif character in "\"'":
            quote = character
        elif character == "=":
            return index
    raise ValueError(f"could not find TOML assignment operator in {source!r}")


def toml_assignments(source: str) -> list[TomlAssignment]:
    lines = source.splitlines(keepends=True)
    offsets = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    table = ()
    assignments = []
    index = 0
    while index < len(lines):
        stripped = lines[index].lstrip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        if stripped.startswith("["):
            table = parse_table_path(stripped)
            index += 1
            continue

        start = offsets[index]
        end_index = index + 1
        while True:
            statement = "".join(lines[index:end_index])
            try:
                tomllib.loads(statement)
                break
            except tomllib.TOMLDecodeError:
                if end_index >= len(lines):
                    raise
                end_index += 1
        equals = assignment_equals(statement)
        key = parse_key_path(statement[:equals].strip())
        assignments.append(
            TomlAssignment(
                table=table,
                key=key,
                start=start,
                end=start + len(statement),
                equals=start + equals,
            )
        )
        index = end_index
    return assignments


def quoted_value_span(source: str, value_start: int, value_end: int) -> tuple[int, int]:
    index = value_start
    while index < value_end and source[index].isspace():
        index += 1
    if index >= value_end or source[index] not in "\"'":
        raise ValueError("version value must be a basic or literal TOML string")
    quote = source[index]
    content_start = index + 1
    index = content_start
    escaped = False
    while index < value_end:
        character = source[index]
        if quote == '"' and escaped:
            escaped = False
        elif quote == '"' and character == "\\":
            escaped = True
        elif character == quote:
            return content_start, index
        index += 1
    raise ValueError("unterminated TOML version string")


def inline_table_items(
    source: str, value_start: int, value_end: int
) -> list[tuple[int, int]]:
    start = value_start
    while start < value_end and source[start].isspace():
        start += 1
    if start >= value_end or source[start] != "{":
        raise ValueError("path dependency must use a TOML inline table or table")

    items = []
    item_start = start + 1
    braces = 1
    brackets = 0
    quote = None
    escaped = False
    index = item_start
    while index < value_end:
        character = source[index]
        if quote == '"':
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quote = None
        elif quote == "'":
            if character == "'":
                quote = None
        elif character in "\"'":
            quote = character
        elif character == "{":
            braces += 1
        elif character == "}":
            braces -= 1
            if braces == 0:
                if source[item_start:index].strip():
                    items.append((item_start, index))
                return items
        elif character == "[":
            brackets += 1
        elif character == "]":
            brackets -= 1
        elif character == "," and braces == 1 and brackets == 0:
            items.append((item_start, index))
            item_start = index + 1
        index += 1
    raise ValueError("unterminated TOML inline table")


def inline_version_span(
    source: str, value_start: int, value_end: int
) -> tuple[int, int]:
    for item_start, item_end in inline_table_items(source, value_start, value_end):
        item = source[item_start:item_end]
        equals = assignment_equals(item)
        if parse_key_path(item[:equals].strip()) != ("version",):
            continue
        return quoted_value_span(source, item_start + equals + 1, item_end)
    raise ValueError("versioned path dependency has no source version field")


def replacement_spans(
    source: str,
    manifest: Path,
    package: WorkspacePackage | None,
    dependencies: list[PathDependency],
    new_version: str,
) -> list[tuple[int, int, str]]:
    assignments = toml_assignments(source)
    replacements = []

    if package is not None:
        matches = [
            assignment
            for assignment in assignments
            if assignment.table == ("package",) and assignment.key == ("version",)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{manifest}: expected one [package] version assignment, "
                f"found {len(matches)}"
            )
        assignment = matches[0]
        start, end = quoted_value_span(source, assignment.equals + 1, assignment.end)
        replacements.append((start, end, new_version))

    for dependency in dependencies:
        span = None
        for assignment in assignments:
            if (
                assignment.table == dependency.group
                and assignment.key == (dependency.alias,)
            ):
                span = inline_version_span(
                    source, assignment.equals + 1, assignment.end
                )
            elif (
                assignment.table == dependency.group
                and assignment.key == (dependency.alias, "version")
            ):
                span = quoted_value_span(
                    source, assignment.equals + 1, assignment.end
                )
            elif (
                assignment.table == (*dependency.group, dependency.alias)
                and assignment.key == ("version",)
            ):
                span = quoted_value_span(
                    source, assignment.equals + 1, assignment.end
                )
            else:
                continue
            break
        if span is None:
            raise ValueError(
                f"{manifest}: could not locate version for path dependency "
                f"{dependency.alias}"
            )
        replacements.append((*span, new_version))
    return replacements


def update_cargo_versions(root: Path, old_version: str, new_version: str) -> list[Path]:
    root = root.resolve()
    packages = workspace_packages(root)
    for package in packages.values():
        if package.version not in {old_version, new_version}:
            manifest = package.manifest.relative_to(root)
            raise ValueError(
                f"{manifest}: expected package version {old_version} or "
                f"{new_version}, found {package.version}"
            )

    dependencies = [
        dependency
        for dependency in path_dependencies(root, packages)
        if dependency.requirement is not None
    ]
    dependencies_by_manifest = {}
    for dependency in dependencies:
        dependencies_by_manifest.setdefault(dependency.manifest, []).append(dependency)

    manifests = set(packages)
    manifests.update(dependencies_by_manifest)
    updates = {}
    for manifest in sorted(manifests):
        source = manifest.read_text(encoding="utf-8")
        spans = replacement_spans(
            source,
            manifest,
            packages.get(manifest),
            dependencies_by_manifest.get(manifest, []),
            new_version,
        )
        for start, end, replacement in sorted(spans, reverse=True):
            source = source[:start] + replacement + source[end:]
        tomllib.loads(source)
        updates[manifest] = source

    for manifest, source in updates.items():
        manifest.write_text(source, encoding="utf-8")

    updated_packages = workspace_packages(root)
    for package in updated_packages.values():
        if package.version != new_version:
            manifest = package.manifest.relative_to(root)
            raise ValueError(
                f"{manifest}: package version was not updated to {new_version}"
            )
    for dependency in path_dependencies(root, updated_packages):
        if dependency.requirement is not None and dependency.requirement != new_version:
            manifest = dependency.manifest.relative_to(root)
            raise ValueError(
                f"{manifest}: path dependency {dependency.alias} was not updated "
                f"to {new_version}"
            )
    return sorted(updates)


def verify(expected: str, java_snapshot: bool, root: Path = ROOT) -> list[str]:
    failures = []
    expected_java = f"{expected}-SNAPSHOT" if java_snapshot else expected

    actual_java = java_version(root)
    if actual_java != expected_java:
        failures.append(
            f"java/pom.xml: expected {expected_java}, found {actual_java}"
        )

    expected_lock_versions = {}
    for manifest, package_name in RUST_PACKAGES.items():
        actual = load_toml(manifest, root)["package"]["version"]
        if actual != expected:
            failures.append(f"{manifest}: expected {expected}, found {actual}")
        expected_lock_versions[package_name] = expected

    python_version = load_toml("python/pyproject.toml", root)["project"]["version"]
    if python_version != expected:
        failures.append(
            f"python/pyproject.toml: expected {expected}, found {python_version}"
        )

    lock_packages = load_toml("Cargo.lock", root).get("package", [])
    lock_versions = {
        package["name"]: package["version"]
        for package in lock_packages
        if package["name"] in expected_lock_versions
    }
    for package_name, expected_version in expected_lock_versions.items():
        actual = lock_versions.get(package_name)
        if actual != expected_version:
            failures.append(
                f"Cargo.lock {package_name}: expected {expected_version}, found {actual}"
            )

    failures.extend(path_dependency_failures(root))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        nargs="?",
        help="base release version, for example 0.3.0",
    )
    parser.add_argument(
        "--java-snapshot",
        action="store_true",
        help="require java/pom.xml to use VERSION-SNAPSHOT",
    )
    parser.add_argument(
        "--update-cargo",
        nargs=2,
        metavar=("OLD_VERSION", "NEW_VERSION"),
        help=(
            "update workspace package versions and versioned workspace path "
            "dependencies"
        ),
    )
    args = parser.parse_args()

    if args.update_cargo:
        if args.version is not None or args.java_snapshot:
            parser.error("--update-cargo cannot be combined with version verification")
        old_version, new_version = args.update_cargo
        if not SEMVER.fullmatch(old_version) or not SEMVER.fullmatch(new_version):
            parser.error("Cargo versions must be three-part numeric release versions")
        try:
            updated = update_cargo_versions(ROOT, old_version, new_version)
        except (
            KeyError,
            OSError,
            subprocess.SubprocessError,
            tomllib.TOMLDecodeError,
            ValueError,
        ) as error:
            print(f"Cargo version update failed: {error}", file=sys.stderr)
            return 1
        for manifest in updated:
            print(f"updated {manifest.relative_to(ROOT)}")
        return 0

    if args.version is None:
        parser.error("version is required unless --update-cargo is used")
    if not SEMVER.fullmatch(args.version):
        parser.error("version must be a three-part numeric release version")

    try:
        failures = verify(args.version, args.java_snapshot)
    except (
        KeyError,
        OSError,
        subprocess.SubprocessError,
        ET.ParseError,
        tomllib.TOMLDecodeError,
        ValueError,
    ) as error:
        print(f"release version verification failed: {error}", file=sys.stderr)
        return 1

    if failures:
        print("Release component versions do not match:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    java_suffix = "-SNAPSHOT" if args.java_snapshot else ""
    print(
        f"All release component versions match {args.version} "
        f"(Java {args.version}{java_suffix})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
