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

"""Generate artifact-exact third-party license reports for native binaries."""

from __future__ import annotations

import argparse
import difflib
import html
from html.parser import HTMLParser
import json
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path


CARGO_ABOUT_VERSION = "0.9.1"
HTML_TEMPLATE = "about.html.template"
TARGETS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
)


@dataclass(frozen=True)
class Report:
    manifest: str
    package: str
    target: str
    output: str


@dataclass(frozen=True)
class RuntimeInventory:
    packages: dict[str, dict]
    features: dict[str, frozenset[str]]


@dataclass(frozen=True)
class UsedBy:
    label: str
    url: str
    package_key: str | None = None


@dataclass(frozen=True)
class LicenseRecord:
    name: str
    spdx_id: str
    text: str
    used_by: tuple[UsedBy, ...]


@dataclass(frozen=True)
class BundledComponent:
    crate: str
    license_path: str
    component: str
    component_url: str
    license_name: str
    spdx_id: str
    forbidden_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class ThirdPartyNotice:
    packages: tuple[tuple[str, str, str, str], ...]
    text: str


@dataclass(frozen=True)
class RustUnicodeAttribution:
    version: str
    license_text: str


BUNDLED_COMPONENTS = (
    BundledComponent(
        crate="zstd-sys",
        license_path="zstd/LICENSE",
        component="vendored Zstandard C sources",
        component_url="https://github.com/facebook/zstd",
        license_name="BSD 3-Clause License",
        spdx_id="BSD-3-Clause",
        # The legacy decoder links additional BSD-2-Clause source files. Keep
        # it disabled unless those separate notices are added to this report.
        forbidden_features=("legacy",),
    ),
)


class CopyrightLibraryParser(HTMLParser):
    """Extract text tokens from COPYRIGHT-library.html without regex parsing."""

    def __init__(self) -> None:
        super().__init__()
        self.tokens: list[str] = []

    def handle_data(self, data: str) -> None:
        if text := data.strip():
            self.tokens.append(text)


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def report_specs() -> list[Report]:
    reports = []
    for target in TARGETS:
        reports.append(
            Report(
                manifest="jni/Cargo.toml",
                package="paimon-mosaic-jni",
                target=target,
                output=(
                    "java/src/main/binary-resources/META-INF/licenses/"
                    f"{target}/THIRD-PARTY-LICENSES.html"
                ),
            )
        )
        reports.append(
            Report(
                manifest="ffi/Cargo.toml",
                package="paimon-mosaic-ffi",
                target=target,
                output=f"python/licenses/{target}/THIRD-PARTY-LICENSES.html",
            )
        )
    return reports


def verify_cargo_about(root: Path) -> None:
    output = subprocess.check_output(
        ["cargo", "about", "--version"], cwd=root, text=True
    ).strip()
    actual = output.rsplit(" ", 1)[-1]
    if actual != CARGO_ABOUT_VERSION:
        raise RuntimeError(
            f"cargo-about {CARGO_ABOUT_VERSION} is required, found {output!r}"
        )


def cargo_metadata(root: Path, report: Report) -> dict:
    output = subprocess.check_output(
        [
            "cargo",
            "metadata",
            "--frozen",
            "--format-version",
            "1",
            "--manifest-path",
            report.manifest,
            "--filter-platform",
            report.target,
        ],
        cwd=root,
        text=True,
    )
    try:
        return json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("cargo metadata returned invalid JSON") from error


def cargo_about_data(root: Path, report: Report, output: Path) -> dict:
    subprocess.run(
        [
            "cargo",
            "about",
            "generate",
            "--frozen",
            "--fail",
            "--config",
            str(root / "about.toml"),
            "--manifest-path",
            report.manifest,
            "--target",
            report.target,
            "--format",
            "json",
            "--output-file",
            str(output),
        ],
        cwd=root,
        check=True,
    )
    try:
        return json.loads(output.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("cargo-about returned invalid JSON") from error


def cargo_tree_output(root: Path, report: Report) -> str:
    return subprocess.check_output(
        [
            "cargo",
            "tree",
            "--frozen",
            "--manifest-path",
            report.manifest,
            "--package",
            report.package,
            "--target",
            report.target,
            "--edges",
            "normal,no-proc-macro",
            "--prefix",
            "none",
            "--no-dedupe",
            "--format",
            "{p}|{f}",
        ],
        cwd=root,
        text=True,
    )


def runtime_inventory_from_tree(metadata: dict, tree_output: str) -> RuntimeInventory:
    resolve = metadata.get("resolve")
    packages = metadata.get("packages")
    if not isinstance(resolve, dict) or not isinstance(packages, list):
        raise RuntimeError("cargo metadata is missing packages or resolve data")
    nodes = resolve.get("nodes")
    if not isinstance(nodes, list):
        raise RuntimeError("cargo metadata is missing resolved nodes")

    resolved_ids = {node.get("id") for node in nodes}
    by_name_version: dict[tuple[str, str], list[dict]] = {}
    for package in packages:
        if package.get("id") not in resolved_ids:
            continue
        key = (str(package.get("name")), str(package.get("version")))
        by_name_version.setdefault(key, []).append(package)

    selected: dict[str, dict] = {}
    features: dict[str, set[str]] = {}
    for line_number, line in enumerate(tree_output.splitlines(), start=1):
        if not line:
            continue
        package_text, separator, feature_text = line.partition("|")
        if not separator:
            raise RuntimeError(
                f"cargo tree line {line_number} has no feature separator: {line!r}"
            )
        fields = package_text.split(maxsplit=2)
        if len(fields) < 2 or not fields[1].startswith("v"):
            raise RuntimeError(
                f"cargo tree line {line_number} has no package version: {line!r}"
            )
        key = (fields[0], fields[1][1:])
        matches = by_name_version.get(key, [])
        if not matches:
            raise RuntimeError(
                "cargo tree runtime package is missing from target metadata: "
                f"{key[0]} {key[1]}"
            )
        if len(matches) != 1:
            sources = sorted(
                f"{package.get('id')} source={package.get('source')!r}"
                for package in matches
            )
            raise RuntimeError(
                "ambiguous cargo tree runtime package with the same name/version: "
                f"{key[0]} {key[1]} ({'; '.join(sources)})"
            )

        package = matches[0]
        package_id = package["id"]
        selected[package_id] = package
        enabled = features.setdefault(package_id, set())
        enabled.update(
            feature for feature in feature_text.split(",") if feature
        )

    if not selected:
        raise RuntimeError("cargo tree returned no runtime packages")
    return RuntimeInventory(
        packages=selected,
        features={
            package_id: frozenset(enabled)
            for package_id, enabled in features.items()
        },
    )


def runtime_inventory(
    root: Path, report: Report, metadata: dict
) -> RuntimeInventory:
    inventory = runtime_inventory_from_tree(
        metadata, cargo_tree_output(root, report)
    )
    roots = [
        package
        for package in inventory.packages.values()
        if package["name"] == report.package
    ]
    if len(roots) != 1:
        raise RuntimeError(
            f"expected runtime root {report.package}, found {len(roots)}"
        )
    return inventory


def package_url(package: dict) -> str:
    repository = package.get("repository") or package.get("homepage")
    if repository:
        return str(repository)
    if package.get("source") is not None:
        return f"https://crates.io/crates/{package['name']}"
    return ""


def stable_package_key(package: dict) -> str:
    return f"{package['name']}@{package['version']}"


def package_used_by(package: dict) -> UsedBy:
    return UsedBy(
        label=f"{package['name']} {package['version']}",
        url=package_url(package),
        package_key=stable_package_key(package),
    )


def validate_package_identity(actual: dict, expected: dict) -> None:
    for field in ("id", "name", "version", "source"):
        if actual.get(field) != expected.get(field):
            raise RuntimeError(
                "cargo-about package identity does not match cargo metadata: "
                f"{field}={actual.get(field)!r}, expected {expected.get(field)!r}"
            )


def filtered_cargo_licenses(
    about_data: dict, inventory: RuntimeInventory
) -> tuple[LicenseRecord, ...]:
    crates = about_data.get("crates")
    licenses = about_data.get("licenses")
    if not isinstance(crates, list) or not isinstance(licenses, list):
        raise RuntimeError("cargo-about JSON is missing crates or licenses")

    about_packages: dict[str, dict] = {}
    for item in crates:
        package = item.get("package") if isinstance(item, dict) else None
        if not isinstance(package, dict) or not isinstance(package.get("id"), str):
            raise RuntimeError("cargo-about JSON contains an invalid crate")
        package_id = package["id"]
        if package_id in about_packages:
            raise RuntimeError(
                f"cargo-about JSON contains duplicate package {package_id}"
            )
        about_packages[package_id] = package

    missing = sorted(set(inventory.packages) - set(about_packages))
    if missing:
        raise RuntimeError(
            "cargo-about JSON is missing runtime packages: " + ", ".join(missing)
        )
    for package_id, package in inventory.packages.items():
        validate_package_identity(about_packages[package_id], package)

    records = []
    covered: set[str] = set()
    for license_item in licenses:
        if not isinstance(license_item, dict):
            raise RuntimeError("cargo-about JSON contains an invalid license")
        usages = license_item.get("used_by")
        if not isinstance(usages, list):
            raise RuntimeError(
                "cargo-about JSON contains a license without used_by"
            )
        filtered = []
        for usage in usages:
            package = usage.get("crate") if isinstance(usage, dict) else None
            if not isinstance(package, dict) or not isinstance(
                package.get("id"), str
            ):
                raise RuntimeError(
                    "cargo-about JSON contains an invalid license package"
                )
            package_id = package["id"]
            if package_id not in inventory.packages:
                continue
            validate_package_identity(package, inventory.packages[package_id])
            filtered.append(package_used_by(package))
            covered.add(package_id)

        if not filtered:
            continue
        for field in ("name", "id", "text"):
            if not isinstance(license_item.get(field), str):
                raise RuntimeError(
                    f"cargo-about license is missing string field {field}"
                )
        records.append(
            LicenseRecord(
                name=license_item["name"],
                spdx_id=license_item["id"],
                text=license_item["text"],
                used_by=tuple(
                    sorted(
                        filtered,
                        key=lambda item: (
                            item.label,
                            item.package_key or "",
                        ),
                    )
                ),
            )
        )

    uncovered = sorted(set(inventory.packages) - covered)
    if uncovered:
        raise RuntimeError(
            "cargo-about licenses are missing runtime packages: "
            + ", ".join(uncovered)
        )
    return tuple(records)


def runtime_package_by_name(
    inventory: RuntimeInventory, crate_name: str
) -> dict:
    matches = [
        package
        for package in inventory.packages.values()
        if package["name"] == crate_name
    ]
    if len(matches) != 1:
        versions = sorted(
            f"{package['version']} ({package['id']})" for package in matches
        )
        raise RuntimeError(
            f"expected exactly one runtime {crate_name} package, found {versions}"
        )
    return matches[0]


def verify_component_features(
    inventory: RuntimeInventory, package: dict, component: BundledComponent
) -> None:
    enabled = inventory.features.get(package["id"])
    if enabled is None:
        raise RuntimeError(
            f"target-specific cargo tree features are missing for {component.crate}"
        )
    forbidden = sorted(enabled.intersection(component.forbidden_features))
    if forbidden:
        raise RuntimeError(
            f"{component.crate} enables separately licensed features {forbidden}; "
            "disable them or add their bundled source licenses"
        )


def bundled_component_licenses(
    inventory: RuntimeInventory,
) -> tuple[LicenseRecord, ...]:
    records = []
    for component in BUNDLED_COMPONENTS:
        package = runtime_package_by_name(inventory, component.crate)
        verify_component_features(inventory, package, component)
        crate_root = Path(package["manifest_path"]).parent
        license_file = crate_root / component.license_path
        if not license_file.is_file():
            raise RuntimeError(f"bundled license file is missing: {license_file}")
        records.append(
            LicenseRecord(
                name=component.license_name,
                spdx_id=component.spdx_id,
                text=license_file.read_text(encoding="utf-8"),
                used_by=(
                    UsedBy(
                        label=(
                            f"{component.component}, bundled by "
                            f"{component.crate} {package['version']}"
                        ),
                        url=component.component_url,
                    ),
                ),
            )
        )
    return tuple(records)


def field_values(record: tuple[str, ...], field: str) -> list[str]:
    values = []
    for index, token in enumerate(record):
        if token != field:
            continue
        if index + 1 >= len(record):
            raise RuntimeError(
                f"COPYRIGHT-library.html has no value after {field}"
            )
        values.append(record[index + 1])
    return values


def copyright_record(
    tokens: list[str], file_or_directory: str
) -> tuple[str, ...]:
    starts = [
        index
        for index, token in enumerate(tokens[:-1])
        if token == "File/Directory:" and tokens[index + 1] == file_or_directory
    ]
    if len(starts) != 1:
        raise RuntimeError(
            "COPYRIGHT-library.html must contain exactly one mapping for "
            f"{file_or_directory}, found {len(starts)}"
        )
    start = starts[0]
    end = next(
        (
            index
            for index in range(start + 2, len(tokens))
            if tokens[index] == "File/Directory:"
        ),
        len(tokens),
    )
    return tuple(tokens[start:end])


def rust_unicode_attribution(root: Path) -> RustUnicodeAttribution:
    toolchain_data = tomllib.loads(
        (root / "rust-toolchain.toml").read_text(encoding="utf-8")
    )
    try:
        pinned_version = str(toolchain_data["toolchain"]["channel"])
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "rust-toolchain.toml is missing toolchain.channel"
        ) from error

    version_output = subprocess.check_output(
        ["rustc", "--version", "--verbose"], cwd=root, text=True
    )
    releases = [
        line.partition(":")[2].strip()
        for line in version_output.splitlines()
        if line.startswith("release:")
    ]
    if releases != [pinned_version]:
        raise RuntimeError(
            f"rustc version does not match rust-toolchain.toml: "
            f"pinned {pinned_version}, found {releases}"
        )

    sysroot = Path(
        subprocess.check_output(
            ["rustc", "--print", "sysroot"], cwd=root, text=True
        ).strip()
    )
    copyright_path = sysroot / "share/doc/rust/COPYRIGHT-library.html"
    license_path = sysroot / "share/doc/rust/licenses/Unicode-3.0.txt"
    if not copyright_path.is_file():
        raise RuntimeError(
            f"pinned rustc COPYRIGHT-library.html is missing: {copyright_path}"
        )
    if not license_path.is_file():
        raise RuntimeError(
            f"pinned rustc Unicode-3.0 license is missing: {license_path}"
        )

    parser = CopyrightLibraryParser()
    parser.feed(copyright_path.read_text(encoding="utf-8"))
    unicode_path = "library/core/src/unicode/unicode_data.rs"
    record = copyright_record(parser.tokens, unicode_path)
    if field_values(record, "License:") != ["Unicode-3.0"]:
        raise RuntimeError(
            f"COPYRIGHT-library.html does not map {unicode_path} to Unicode-3.0"
        )
    copyrights = field_values(record, "Copyright:")
    if not copyrights:
        raise RuntimeError(
            f"COPYRIGHT-library.html has no copyright for {unicode_path}"
        )

    license_text = license_path.read_text(encoding="utf-8")
    missing_copyrights = [
        copyright for copyright in copyrights if copyright not in license_text
    ]
    if missing_copyrights:
        raise RuntimeError(
            "pinned Unicode-3.0 license does not contain COPYRIGHT-library "
            "attribution: "
            + ", ".join(missing_copyrights)
        )
    return RustUnicodeAttribution(
        version=pinned_version, license_text=license_text
    )


def rust_unicode_license(
    attribution: RustUnicodeAttribution,
) -> LicenseRecord:
    return LicenseRecord(
        name="Unicode License v3",
        spdx_id="Unicode-3.0",
        text=attribution.license_text,
        used_by=(
            UsedBy(
                label=(
                    f"Rust standard library {attribution.version} "
                    "core Unicode data"
                ),
                url="https://github.com/rust-lang/rust",
            ),
        ),
    )


def third_party_notices(
    inventory: RuntimeInventory,
) -> tuple[ThirdPartyNotice, ...]:
    grouped: dict[str, list[tuple[str, str, str, str]]] = {}
    for package in sorted(
        inventory.packages.values(),
        key=lambda item: (item["name"], item["version"], item["id"]),
    ):
        if package.get("source") is None:
            continue
        crate_root = Path(package["manifest_path"]).parent
        notice_paths = sorted(
            {
                path
                for pattern in ("NOTICE", "NOTICE.*", "NOTICE-*")
                for path in crate_root.glob(pattern)
                if path.is_file()
            }
        )
        repository = package_url(package)
        for notice_path in notice_paths:
            notice_text = notice_path.read_text(encoding="utf-8").rstrip() + "\n"
            grouped.setdefault(notice_text, []).append(
                (
                    package["name"],
                    package["version"],
                    repository,
                    stable_package_key(package),
                )
            )

    return tuple(
        ThirdPartyNotice(packages=tuple(sorted(packages)), text=text)
        for text, packages in sorted(
            grouped.items(),
            key=lambda item: (
                item[1][0][0],
                item[1][0][1],
                item[0],
            ),
        )
    )


def license_overview(
    licenses: tuple[LicenseRecord, ...],
) -> tuple[tuple[str, str, int, str], ...]:
    grouped: dict[str, tuple[str, int, str]] = {}
    anchors: dict[str, int] = {}
    for license_record in licenses:
        count = anchors.get(license_record.spdx_id, 0) + 1
        anchors[license_record.spdx_id] = count
        anchor = license_record.spdx_id
        if count > 1:
            anchor = f"{anchor}-{count}"

        existing = grouped.get(license_record.spdx_id)
        used_by_count = len(license_record.used_by)
        if existing is None:
            grouped[license_record.spdx_id] = (
                license_record.name,
                used_by_count,
                anchor,
            )
        else:
            name, previous_count, first_anchor = existing
            if name != license_record.name:
                raise RuntimeError(
                    f"license {license_record.spdx_id} has inconsistent names "
                    f"{name!r} and {license_record.name!r}"
                )
            grouped[license_record.spdx_id] = (
                name,
                previous_count + used_by_count,
                first_anchor,
            )
    return tuple(
        (spdx_id, name, count, anchor)
        for spdx_id, (name, count, anchor) in grouped.items()
    )


def render_overview(licenses: tuple[LicenseRecord, ...]) -> str:
    lines = []
    for _, name, count, anchor in license_overview(licenses):
        lines.append(
            "            <li>"
            f'<a href="#{html.escape(anchor, quote=True)}">'
            f"{html.escape(name)}</a> ({count})</li>"
        )
    return "\n".join(lines)


def render_licenses(licenses: tuple[LicenseRecord, ...]) -> str:
    lines = []
    anchor_counts: dict[str, int] = {}
    for license_record in licenses:
        count = anchor_counts.get(license_record.spdx_id, 0) + 1
        anchor_counts[license_record.spdx_id] = count
        anchor = license_record.spdx_id
        if count > 1:
            anchor = f"{anchor}-{count}"

        used_by_lines = []
        for usage in license_record.used_by:
            attribute = ""
            if usage.package_key is not None:
                attribute = (
                    ' data-package="'
                    + html.escape(usage.package_key, quote=True)
                    + '"'
                )
            escaped_label = html.escape(usage.label)
            if usage.url:
                rendered_usage = (
                    f'<a href="{html.escape(usage.url, quote=True)}">'
                    f"{escaped_label}</a>"
                )
            else:
                rendered_usage = escaped_label
            used_by_lines.append(
                f"                    <li{attribute}>{rendered_usage}</li>"
            )

        lines.extend(
            [
                '            <li class="license">',
                f'                <h3 id="{html.escape(anchor, quote=True)}">'
                f"{html.escape(license_record.name)}</h3>",
                "                <h4>Used by:</h4>",
                '                <ul class="license-used-by">',
                *used_by_lines,
                "                </ul>",
                '                <pre class="license-text">'
                f"{html.escape(license_record.text)}</pre>",
                "            </li>",
            ]
        )
    return "\n".join(lines)


def render_third_party_notices(
    notices: tuple[ThirdPartyNotice, ...],
) -> str:
    if not notices:
        return ""

    lines = [
        "        <h2>Required third-party notices and attributions:</h2>",
        '        <ul class="licenses-list third-party-notices">',
    ]
    for index, notice in enumerate(notices, start=1):
        lines.extend(
            [
                '            <li class="license third-party-notice">',
                f'                <h3 id="third-party-notice-{index}">'
                "Required notice or attribution</h3>",
                "                <h4>Provided by:</h4>",
                '                <ul class="license-used-by">',
            ]
        )
        for name, version, repository, package_key in notice.packages:
            label = html.escape(f"{name} {version}")
            if repository:
                rendered = (
                    f'<a href="{html.escape(repository, quote=True)}">'
                    f"{label}</a>"
                )
            else:
                rendered = label
            lines.append(
                "                    "
                f'<li data-package="{html.escape(package_key, quote=True)}">'
                f"{rendered}</li>"
            )
        lines.extend(
            [
                "                </ul>",
                '                <pre class="license-text">'
                f"{html.escape(notice.text)}</pre>",
                "            </li>",
            ]
        )
    lines.append("        </ul>")
    return "\n".join(lines)


def render_template(template: str, replacements: dict[str, str]) -> str:
    result = template
    for placeholder, value in replacements.items():
        if result.count(placeholder) != 1:
            raise RuntimeError(
                f"HTML template must contain {placeholder} exactly once"
            )
        result = result.replace(placeholder, value)
    remaining = [
        placeholder
        for placeholder in (
            "@@TARGET@@",
            "@@ROOT_CRATE@@",
            "@@OVERVIEW@@",
            "@@LICENSES@@",
            "@@THIRD_PARTY_NOTICES@@",
        )
        if placeholder in result
    ]
    if remaining:
        raise RuntimeError(
            "HTML template has unreplaced placeholders: " + ", ".join(remaining)
        )
    return "\n".join(line.rstrip() for line in result.rstrip().splitlines()) + "\n"


def complete_report(
    template: str,
    report: Report,
    licenses: tuple[LicenseRecord, ...],
    notices: tuple[ThirdPartyNotice, ...],
) -> str:
    return render_template(
        template,
        {
            "@@TARGET@@": html.escape(report.target),
            "@@ROOT_CRATE@@": html.escape(report.package),
            "@@OVERVIEW@@": render_overview(licenses),
            "@@LICENSES@@": render_licenses(licenses),
            "@@THIRD_PARTY_NOTICES@@": render_third_party_notices(notices),
        },
    )


def binary_license(apache_license: str, heading: str, details: list[str]) -> str:
    appendix = [
        "",
        "=" * 79,
        "BUNDLED THIRD-PARTY COMPONENTS",
        "=" * 79,
        "",
        heading,
        "The component inventory, copyright notices, and complete license texts",
        "are provided in:",
        "",
    ]
    appendix.extend(f"    {detail}" for detail in details)
    return apache_license.rstrip() + "\n" + "\n".join(appendix) + "\n"


def binary_notice(
    project_notice: str, notices: tuple[ThirdPartyNotice, ...]
) -> str:
    unique_texts = dict.fromkeys(notice.text for notice in notices)
    if not unique_texts:
        return project_notice.rstrip() + "\n"

    appendix = [
        "",
        "=" * 79,
        "THIRD-PARTY NOTICES",
        "=" * 79,
        "",
    ]
    for index, notice_text in enumerate(unique_texts, start=1):
        if index > 1:
            appendix.extend(["", "-" * 79, ""])
        appendix.extend(notice_text.rstrip().splitlines())
    return project_notice.rstrip() + "\n" + "\n".join(appendix) + "\n"


def generated_files(root: Path) -> dict[Path, str]:
    verify_cargo_about(root)
    template = (root / HTML_TEMPLATE).read_text(encoding="utf-8")
    unicode_attribution = rust_unicode_attribution(root)
    result = {}
    notices_by_report = {}
    with tempfile.TemporaryDirectory(prefix="paimon-license-reports-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, report in enumerate(report_specs()):
            metadata = cargo_metadata(root, report)
            inventory = runtime_inventory(root, report, metadata)
            about_data = cargo_about_data(
                root, report, temp_root / f"report-{index}.json"
            )
            licenses = (
                filtered_cargo_licenses(about_data, inventory)
                + bundled_component_licenses(inventory)
                + (rust_unicode_license(unicode_attribution),)
            )
            notices = third_party_notices(inventory)
            notices_by_report[(report.manifest, report.target)] = notices
            result[root / report.output] = complete_report(
                template, report, licenses, notices
            )

    apache_license = (root / "LICENSE").read_text(encoding="utf-8")
    project_notice = (root / "NOTICE").read_text(encoding="utf-8")

    java_report_paths = [
        f"META-INF/licenses/{target}/THIRD-PARTY-LICENSES.html"
        for target in TARGETS
    ]
    java_license = binary_license(
        apache_license,
        "This binary JAR bundles Rust native libraries for four release targets.",
        java_report_paths,
    )
    result[
        root / "java/src/main/binary-resources/META-INF/LICENSE"
    ] = java_license
    java_notices = tuple(
        notice
        for target in TARGETS
        for notice in notices_by_report[("jni/Cargo.toml", target)]
    )
    result[
        root / "java/src/main/binary-resources/META-INF/NOTICE"
    ] = binary_notice(project_notice, java_notices)

    for target in TARGETS:
        license_dir = root / "python/licenses" / target
        result[license_dir / "LICENSE"] = binary_license(
            apache_license,
            f"This binary wheel bundles the Rust native library for {target}.",
            ["THIRD-PARTY-LICENSES.html"],
        )
        result[license_dir / "NOTICE"] = binary_notice(
            project_notice,
            notices_by_report[("ffi/Cargo.toml", target)],
        )

    return result


def managed_generated_files(root: Path) -> set[Path]:
    files = {
        path
        for path in (
            root / "java/src/main/binary-resources/META-INF/LICENSE",
            root / "java/src/main/binary-resources/META-INF/NOTICE",
        )
        if path.is_file()
    }
    files.update(
        path
        for path in (
            root / "java/src/main/binary-resources/META-INF/licenses"
        ).glob("*/THIRD-PARTY-LICENSES.html")
        if path.is_file()
    )
    files.update(
        path
        for path in (root / "python/licenses").glob("*/*")
        if path.is_file()
        and path.name in {"LICENSE", "NOTICE", "THIRD-PARTY-LICENSES.html"}
    )
    return files


def check_files(files: dict[Path, str], root: Path) -> int:
    failed = False
    for path, expected in files.items():
        if not path.is_file():
            print(f"missing generated license file: {path.relative_to(root)}")
            failed = True
            continue
        actual = path.read_text(encoding="utf-8")
        if actual == expected:
            continue
        failed = True
        print(f"stale generated license file: {path.relative_to(root)}")
        diff = difflib.unified_diff(
            actual.splitlines(),
            expected.splitlines(),
            fromfile=str(path.relative_to(root)),
            tofile=f"generated/{path.relative_to(root)}",
            lineterm="",
        )
        for line in list(diff)[:200]:
            print(line)

    obsolete = managed_generated_files(root) - set(files)
    for path in sorted(obsolete):
        print(f"obsolete generated license file: {path.relative_to(root)}")
        failed = True

    return 1 if failed else 0


def write_files(files: dict[Path, str], root: Path) -> None:
    obsolete = managed_generated_files(root) - set(files)
    for path in sorted(obsolete):
        path.unlink()
        print(f"removed obsolete {path.relative_to(root)}")

    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"generated {path.relative_to(root)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if checked-in reports differ from reproducible output",
    )
    args = parser.parse_args()

    root = repository_root()
    try:
        files = generated_files(root)
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"failed to generate license reports: {error}", file=sys.stderr)
        return 1

    if args.check:
        return check_files(files, root)
    write_files(files, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
