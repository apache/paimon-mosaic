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

from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys

import pytest


TOOLS_DIRECTORY = Path(__file__).resolve().parent.parent
ROOT = TOOLS_DIRECTORY.parent
sys.path.insert(0, str(TOOLS_DIRECTORY))

import generate_license_reports as generator  # noqa: E402


class PackageKeyParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.package_keys: set[str] = set()

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag != "li":
            return
        package_key = dict(attrs).get("data-package")
        if package_key is not None:
            self.package_keys.add(package_key)


def package(
    name: str,
    version: str,
    package_id: str,
    *,
    source: str | None = "registry+https://example.invalid/index",
    manifest_path: str = "/tmp/crate/Cargo.toml",
) -> dict:
    return {
        "name": name,
        "version": version,
        "id": package_id,
        "source": source,
        "manifest_path": manifest_path,
        "repository": f"https://example.invalid/{name}",
    }


def metadata(*packages: dict) -> dict:
    return {
        "packages": list(packages),
        "resolve": {
            "nodes": [{"id": package_item["id"]} for package_item in packages]
        },
    }


def inventory(*packages: dict) -> generator.RuntimeInventory:
    return generator.RuntimeInventory(
        packages={package_item["id"]: package_item for package_item in packages},
        features={package_item["id"]: frozenset() for package_item in packages},
    )


def test_check_rejects_obsolete_target_report(
    tmp_path: Path, capsys
) -> None:
    expected = (
        tmp_path
        / "python/licenses/aarch64-unknown-linux-gnu"
        / "THIRD-PARTY-LICENSES.html"
    )
    expected.parent.mkdir(parents=True)
    expected.write_text("current\n", encoding="utf-8")
    obsolete = (
        tmp_path
        / "python/licenses/obsolete-target"
        / "THIRD-PARTY-LICENSES.html"
    )
    obsolete.parent.mkdir(parents=True)
    obsolete.write_text("obsolete\n", encoding="utf-8")

    assert generator.check_files({expected: "current\n"}, tmp_path) == 1
    assert "obsolete generated license file" in capsys.readouterr().out


def test_generate_removes_obsolete_target_report(tmp_path: Path) -> None:
    expected = (
        tmp_path
        / "java/src/main/binary-resources/META-INF/licenses/current"
        / "THIRD-PARTY-LICENSES.html"
    )
    obsolete = (
        tmp_path
        / "java/src/main/binary-resources/META-INF/licenses/obsolete"
        / "THIRD-PARTY-LICENSES.html"
    )
    obsolete.parent.mkdir(parents=True)
    obsolete.write_text("obsolete\n", encoding="utf-8")

    generator.write_files({expected: "current\n"}, tmp_path)

    assert expected.read_text(encoding="utf-8") == "current\n"
    assert not obsolete.exists()


def test_cargo_tree_uses_package_target_and_runtime_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_check_output(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return "runtime v1.0.0|std\n"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    report = generator.Report(
        manifest="ffi/Cargo.toml",
        package="paimon-mosaic-ffi",
        target="aarch64-unknown-linux-gnu",
        output="unused",
    )

    assert generator.cargo_tree_output(ROOT, report) == "runtime v1.0.0|std\n"
    command = captured["command"]
    assert command[command.index("--package") + 1] == "paimon-mosaic-ffi"
    assert command[command.index("--target") + 1] == "aarch64-unknown-linux-gnu"
    assert command[command.index("--edges") + 1] == "normal,no-proc-macro"
    assert "--no-dedupe" in command
    assert captured["kwargs"] == {"cwd": ROOT, "text": True}


def test_cargo_about_generates_structured_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}
    output = tmp_path / "about.json"

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output.write_text('{"crates": [], "licenses": []}', encoding="utf-8")

    monkeypatch.setattr(subprocess, "run", fake_run)
    report = generator.Report(
        manifest="ffi/Cargo.toml",
        package="paimon-mosaic-ffi",
        target="aarch64-unknown-linux-gnu",
        output="unused",
    )

    assert generator.cargo_about_data(ROOT, report, output) == {
        "crates": [],
        "licenses": [],
    }
    command = captured["command"]
    assert command[command.index("--format") + 1] == "json"
    assert command[command.index("--output-file") + 1] == str(output)
    assert not any(argument.endswith(".hbs") for argument in command)
    assert captured["kwargs"] == {"cwd": ROOT, "check": True}


def test_runtime_inventory_fails_closed_on_ambiguous_sources() -> None:
    registry = package(
        "duplicate",
        "1.0.0",
        "registry+https://example.invalid/index#duplicate@1.0.0",
    )
    git = package(
        "duplicate",
        "1.0.0",
        "git+https://example.invalid/duplicate#duplicate@1.0.0",
        source="git+https://example.invalid/duplicate",
    )

    with pytest.raises(RuntimeError, match="ambiguous cargo tree runtime package"):
        generator.runtime_inventory_from_tree(
            metadata(registry, git), "duplicate v1.0.0|\n"
        )


def test_filtered_cargo_licenses_fails_on_missing_runtime_package() -> None:
    runtime = package("runtime", "1.0.0", "runtime@1.0.0")
    with pytest.raises(RuntimeError, match="missing runtime packages"):
        generator.filtered_cargo_licenses(
            {"crates": [], "licenses": []}, inventory(runtime)
        )


def test_filtered_cargo_licenses_fails_on_unlicensed_runtime_package() -> None:
    runtime = package("runtime", "1.0.0", "runtime@1.0.0")
    with pytest.raises(RuntimeError, match="licenses are missing runtime packages"):
        generator.filtered_cargo_licenses(
            {
                "crates": [{"package": runtime, "license": []}],
                "licenses": [],
            },
            inventory(runtime),
        )


def test_filtered_cargo_licenses_recomputes_used_by_and_overview() -> None:
    runtime = package("runtime", "1.0.0", "runtime@1.0.0")
    host_only = package("host-only", "2.0.0", "host-only@2.0.0")
    about_data = {
        "crates": [
            {"package": runtime, "license": []},
            {"package": host_only, "license": []},
        ],
        "licenses": [
            {
                "name": "Apache License 2.0",
                "id": "Apache-2.0",
                "text": "apache\n",
                "used_by": [
                    {"crate": runtime, "path": None},
                    {"crate": host_only, "path": None},
                ],
            }
        ],
    }

    records = generator.filtered_cargo_licenses(
        about_data, inventory(runtime)
    )

    assert records == (
        generator.LicenseRecord(
            name="Apache License 2.0",
            spdx_id="Apache-2.0",
            text="apache\n",
            used_by=(
                generator.UsedBy(
                    label="runtime 1.0.0",
                    url="https://example.invalid/runtime",
                    package_key="runtime@1.0.0",
                ),
            ),
        ),
    )
    assert generator.license_overview(records) == (
        ("Apache-2.0", "Apache License 2.0", 1, "Apache-2.0"),
    )


def test_zstd_feature_check_uses_target_specific_tree_features() -> None:
    zstd = package("zstd-sys", "2.0.0", "zstd-sys@2.0.0")
    component = generator.BUNDLED_COMPONENTS[0]
    target_inventory = generator.RuntimeInventory(
        packages={zstd["id"]: zstd},
        features={zstd["id"]: frozenset({"std"})},
    )

    generator.verify_component_features(target_inventory, zstd, component)

    legacy_inventory = generator.RuntimeInventory(
        packages={zstd["id"]: zstd},
        features={zstd["id"]: frozenset({"legacy", "std"})},
    )
    with pytest.raises(RuntimeError, match="separately licensed features"):
        generator.verify_component_features(
            legacy_inventory, zstd, component
        )


def test_third_party_notices_are_derived_from_runtime_inventory(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "Cargo.toml").write_text("", encoding="utf-8")
    (runtime_root / "NOTICE").write_text("runtime notice\n", encoding="utf-8")
    runtime = package(
        "runtime",
        "1.0.0",
        "runtime@1.0.0",
        manifest_path=str(runtime_root / "Cargo.toml"),
    )

    notices = generator.third_party_notices(inventory(runtime))

    assert notices == (
        generator.ThirdPartyNotice(
            packages=(
                (
                    "runtime",
                    "1.0.0",
                    "https://example.invalid/runtime",
                    "runtime@1.0.0",
                ),
            ),
            text="runtime notice\n",
        ),
    )


def test_static_html_renderer_escapes_all_dynamic_fields() -> None:
    report = generator.Report(
        manifest="ffi/Cargo.toml",
        package="<root>",
        target="<target>",
        output="unused",
    )
    licenses = (
        generator.LicenseRecord(
            name="<script>",
            spdx_id='MIT"',
            text="<license>&",
            used_by=(
                generator.UsedBy(
                    label="<package>",
                    url='https://example.invalid/?q="',
                    package_key='package"',
                ),
            ),
        ),
    )
    template = (
        "@@TARGET@@\n@@ROOT_CRATE@@\n@@OVERVIEW@@\n"
        "@@LICENSES@@\n@@THIRD_PARTY_NOTICES@@\n"
    )

    rendered = generator.complete_report(template, report, licenses, ())

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;target&gt;" in rendered
    assert "&lt;root&gt;" in rendered
    assert "&lt;license&gt;&amp;" in rendered
    assert 'data-package="package&quot;"' in rendered
    assert 'href="https://example.invalid/?q=&quot;"' in rendered


def test_pinned_rust_unicode_attribution_comes_from_sysroot() -> None:
    attribution = generator.rust_unicode_attribution(ROOT)
    record = generator.rust_unicode_license(attribution)

    assert record.spdx_id == "Unicode-3.0"
    assert "UNICODE LICENSE V3" in record.text
    assert record.used_by[0].label == (
        f"Rust standard library {attribution.version} core Unicode data"
    )


@pytest.mark.parametrize("report", generator.report_specs())
def test_checked_in_report_package_inventory_matches_runtime_tree(
    report: generator.Report,
) -> None:
    metadata_data = generator.cargo_metadata(ROOT, report)
    runtime = generator.runtime_inventory(ROOT, report, metadata_data)
    parser = PackageKeyParser()
    parser.feed((ROOT / report.output).read_text(encoding="utf-8"))

    assert parser.package_keys == {
        generator.stable_package_key(package_item)
        for package_item in runtime.packages.values()
    }
    content = (ROOT / report.output).read_text(encoding="utf-8")
    unicode_attribution = generator.rust_unicode_attribution(ROOT)
    assert (
        f"Rust standard library {unicode_attribution.version} core Unicode data"
        in content
    )
