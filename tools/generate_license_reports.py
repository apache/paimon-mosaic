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
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


CARGO_ABOUT_VERSION = "0.9.1"
TARGETS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
)


@dataclass(frozen=True)
class Report:
    manifest: str
    target: str
    output: str


@dataclass(frozen=True)
class BundledComponent:
    crate: str
    license_path: str
    component: str
    component_url: str
    license_name: str
    anchor: str
    forbidden_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class ThirdPartyNotice:
    packages: tuple[tuple[str, str, str], ...]
    text: str


BUNDLED_COMPONENTS = (
    BundledComponent(
        crate="zstd-sys",
        license_path="zstd/LICENSE",
        component="vendored Zstandard C sources",
        component_url="https://github.com/facebook/zstd",
        license_name="BSD 3-Clause License",
        anchor="bundled-zstandard-bsd-3-clause",
        # The legacy decoder links additional BSD-2-Clause source files. Keep
        # it disabled unless those separate notices are added to this report.
        forbidden_features=("legacy",),
    ),
)


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def report_specs() -> list[Report]:
    reports = []
    for target in TARGETS:
        reports.append(
            Report(
                manifest="jni/Cargo.toml",
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
            "--locked",
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
    return json.loads(output)


def generate_base_report(root: Path, report: Report, output: Path) -> str:
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
            "--output-file",
            str(output),
            str(root / "about.hbs"),
        ],
        cwd=root,
        check=True,
    )
    return output.read_text(encoding="utf-8")


def package_by_name(metadata: dict, crate_name: str) -> dict:
    resolved = {node["id"] for node in metadata["resolve"]["nodes"]}
    matches = [
        package
        for package in metadata["packages"]
        if package["id"] in resolved and package["name"] == crate_name
    ]
    if len(matches) != 1:
        versions = [package["version"] for package in matches]
        raise RuntimeError(
            f"expected exactly one resolved {crate_name} package, found {versions}"
        )
    return matches[0]


def verify_component_features(
    metadata: dict, package: dict, component: BundledComponent
) -> None:
    node = next(
        (node for node in metadata["resolve"]["nodes"] if node["id"] == package["id"]),
        None,
    )
    if node is None:
        raise RuntimeError(f"resolved node is missing for {component.crate}")
    enabled = set(node.get("features", []))
    forbidden = sorted(enabled.intersection(component.forbidden_features))
    if forbidden:
        raise RuntimeError(
            f"{component.crate} enables separately licensed features {forbidden}; "
            "disable them or add their bundled source licenses"
        )


def package_marker(package: dict) -> str:
    return f">{package['name']} {package['version']}</a>"


def included_third_party_packages(base_report: str, metadata: dict) -> list[dict]:
    return sorted(
        (
            package
            for package in metadata["packages"]
            if package.get("source") is not None
            and package_marker(package) in base_report
        ),
        key=lambda package: (package["name"], package["version"]),
    )


def third_party_notices(
    base_report: str, metadata: dict
) -> tuple[ThirdPartyNotice, ...]:
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for package in included_third_party_packages(base_report, metadata):
        crate_root = Path(package["manifest_path"]).parent
        notice_paths = sorted(
            {
                path
                for pattern in ("NOTICE", "NOTICE.*", "NOTICE-*")
                for path in crate_root.glob(pattern)
                if path.is_file()
            }
        )
        repository = package.get("repository") or (
            f"https://crates.io/crates/{package['name']}"
        )
        for notice_path in notice_paths:
            notice_text = notice_path.read_text(encoding="utf-8").rstrip() + "\n"
            grouped.setdefault(notice_text, []).append(
                (package["name"], package["version"], repository)
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


def bundled_component_html(base_report: str, metadata: dict) -> str:
    items = []
    for component in BUNDLED_COMPONENTS:
        package = package_by_name(metadata, component.crate)
        verify_component_features(metadata, package, component)
        marker = f">{component.crate} {package['version']}</a>"
        if marker not in base_report:
            raise RuntimeError(
                f"cargo-about report omitted resolved crate {component.crate} "
                f"{package['version']}"
            )

        crate_root = Path(package["manifest_path"]).parent
        license_file = crate_root / component.license_path
        if not license_file.is_file():
            raise RuntimeError(f"bundled license file is missing: {license_file}")
        license_text = license_file.read_text(encoding="utf-8")
        repository = package.get("repository") or (
            f"https://crates.io/crates/{component.crate}"
        )

        items.append(
            "\n".join(
                [
                    '            <li class="license bundled-subcomponent">',
                    f'                <h3 id="{html.escape(component.anchor)}">'
                    f"{html.escape(component.license_name)}</h3>",
                    "                <h4>Bundled component:</h4>",
                    '                <ul class="license-used-by">',
                    "                    <li>",
                    f'                        <a href="{html.escape(component.component_url, quote=True)}">'
                    f"{html.escape(component.component)}</a>, bundled by",
                    f'                        <a href="{html.escape(repository, quote=True)}">'
                    f"{html.escape(component.crate)} {html.escape(package['version'])}</a>",
                    "                    </li>",
                    "                </ul>",
                    f'                <pre class="license-text">{html.escape(license_text)}</pre>',
                    "            </li>",
                ]
            )
        )

    return "\n".join(
        [
            "",
            "        <h2>Licenses for source components bundled inside crates:</h2>",
            "        <p>",
            "            The following components are compiled into the native library but",
            "            have licenses in nested crate source directories, so they require",
            "            explicit entries in addition to the crate-level licenses above.",
            "        </p>",
            '        <ul class="licenses-list bundled-subcomponents">',
            *items,
            "        </ul>",
        ]
    )


def third_party_notices_html(notices: tuple[ThirdPartyNotice, ...]) -> str:
    if not notices:
        return ""

    items = []
    for index, notice in enumerate(notices, start=1):
        used_by = []
        for name, version, repository in notice.packages:
            used_by.extend(
                [
                    "                    <li>",
                    f'                        <a href="{html.escape(repository, quote=True)}">'
                    f"{html.escape(name)} {html.escape(version)}</a>",
                    "                    </li>",
                ]
            )
        items.append(
            "\n".join(
                [
                    '            <li class="license third-party-notice">',
                    f'                <h3 id="third-party-notice-{index}">'
                    "Required notice or attribution</h3>",
                    "                <h4>Provided by:</h4>",
                    '                <ul class="license-used-by">',
                    *used_by,
                    "                </ul>",
                    f'                <pre class="license-text">{html.escape(notice.text)}</pre>',
                    "            </li>",
                ]
            )
        )

    return "\n".join(
        [
            "",
            "        <h2>Required third-party notices and attributions:</h2>",
            '        <ul class="licenses-list third-party-notices">',
            *items,
            "        </ul>",
        ]
    )


def complete_report(
    base_report: str,
    report: Report,
    metadata: dict,
    notices: tuple[ThirdPartyNotice, ...],
) -> str:
    description = (
        "\n        <p><strong>Rust target:</strong> "
        f"<code>{html.escape(report.target)}</code></p>"
        "\n        <p><strong>Root crate:</strong> "
        f"<code>{html.escape(Path(report.manifest).parent.name)}</code></p>"
    )
    first_paragraph_end = base_report.find("</p>")
    if first_paragraph_end == -1:
        raise RuntimeError("about.hbs output has no introductory paragraph")
    first_paragraph_end += len("</p>")
    result = (
        base_report[:first_paragraph_end]
        + description
        + base_report[first_paragraph_end:]
    )

    closing_main = result.rfind("    </main>")
    if closing_main == -1:
        raise RuntimeError("about.hbs output has no closing main element")
    result = (
        result[:closing_main]
        + bundled_component_html(base_report, metadata)
        + third_party_notices_html(notices)
        + "\n"
        + result[closing_main:]
    )
    # Some upstream license files contain insignificant trailing spaces. Keep
    # generated reports friendly to git's whitespace checks without changing
    # any license wording.
    return "\n".join(line.rstrip() for line in result.rstrip().splitlines()) + "\n"


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
    result = {}
    notices_by_report = {}
    with tempfile.TemporaryDirectory(prefix="paimon-license-reports-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, report in enumerate(report_specs()):
            base = generate_base_report(
                root, report, temp_root / f"report-{index}.html"
            )
            metadata = cargo_metadata(root, report)
            notices = third_party_notices(base, metadata)
            notices_by_report[(report.manifest, report.target)] = notices
            result[root / report.output] = complete_report(
                base, report, metadata, notices
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
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"failed to generate license reports: {error}", file=sys.stderr)
        return 1

    if args.check:
        return check_files(files, root)
    write_files(files, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
