#!/usr/bin/env python3

#
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
#

"""Verify legal files and native resources in binary convenience artifacts."""

import argparse
from datetime import datetime, timezone
import glob
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LEGAL_FILES = ("LICENSE", "NOTICE", "LICENSE-binary")
WHEEL_LEGAL_SOURCES = {
    "NOTICE": "java/src/main/binary-resources/META-INF/NOTICE",
    "LICENSE-binary": "LICENSE-binary-ffi",
}
JAR_LEGAL_SOURCES = {
    "NOTICE": "java/src/main/binary-resources/META-INF/NOTICE",
    "LICENSE-binary": "LICENSE-binary",
}
JAR_NATIVE_FILES = (
    "native/linux/x86_64/libpaimon_mosaic_jni.so",
    "native/linux/aarch64/libpaimon_mosaic_jni.so",
    "native/macos/aarch64/libpaimon_mosaic_jni.dylib",
    "native/windows/x86_64/paimon_mosaic_jni.dll",
)
SUPPORTED_TARGETS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
)


def resolve_artifacts(patterns):
    artifacts = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"artifact pattern matched no files: {pattern}")
        artifacts.extend(Path(match) for match in matches)
    return artifacts


def require_entries(archive, artifact, entries):
    names = set(archive.namelist())
    missing = [entry for entry in entries if entry not in names]
    if missing:
        raise ValueError(f"{artifact} is missing entries: {', '.join(missing)}")


def require_canonical_content(archive, artifact, archive_prefix, sources):
    for name, source in sources.items():
        entry = f"{archive_prefix}/{name}"
        expected = (ROOT / source).read_bytes()
        actual = archive.read(entry)
        if actual != expected:
            raise ValueError(f"{artifact}!/{entry} does not match repository {source}")


def binary_license_content(inventory):
    return (ROOT / "LICENSE").read_bytes().rstrip() + b"\n\n" + (
        ROOT / inventory
    ).read_bytes()


def require_binary_license(archive, artifact, entry, inventory):
    if archive.read(entry) != binary_license_content(inventory):
        raise ValueError(
            f"{artifact}!/{entry} must contain the Apache license and the "
            f"complete third-party license inventory from {inventory}"
        )


def verify_wheel(artifact):
    with zipfile.ZipFile(artifact) as archive:
        names = archive.namelist()
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError(
                f"{artifact} must contain exactly one .dist-info/METADATA entry"
            )
        dist_info = metadata[0][: -len("METADATA")]
        dist_legal = [f"{dist_info}licenses/{name}" for name in LEGAL_FILES]
        package_legal = [f"mosaic/{name}" for name in LEGAL_FILES]
        require_entries(archive, artifact, dist_legal + package_legal)
        for entry in ("mosaic/LICENSE", f"{dist_info}licenses/LICENSE"):
            require_binary_license(archive, artifact, entry, "LICENSE-binary-ffi")
        require_canonical_content(
            archive, artifact, "mosaic", WHEEL_LEGAL_SOURCES
        )
        require_canonical_content(
            archive,
            artifact,
            f"{dist_info}licenses".rstrip("/"),
            WHEEL_LEGAL_SOURCES,
        )

        native_names = [
            name
            for name in names
            if name.startswith("mosaic/")
            and name.endswith((".so", ".dylib", ".dll"))
        ]
        if len(native_names) != 1:
            raise ValueError(
                f"{artifact} must contain exactly one platform native library, found "
                f"{len(native_names)}"
            )

    print(f"Verified Python wheel: {artifact}")


def verify_jar(artifact, native_files):
    with zipfile.ZipFile(artifact) as archive:
        required = (
            "META-INF/LICENSE",
            "META-INF/NOTICE",
            "META-INF/LICENSE-binary",
            "META-INF/DEPENDENCIES",
            "org/apache/paimon/mosaic/NativeLib.class",
        ) + tuple(native_files)
        require_entries(archive, artifact, required)
        require_binary_license(archive, artifact, "META-INF/LICENSE", "LICENSE-binary")
        require_canonical_content(
            archive,
            artifact,
            "META-INF",
            JAR_LEGAL_SOURCES,
        )

    print(f"Verified Java JAR: {artifact}")


def verify_java_auxiliary(artifact, javadoc=False):
    pom = ET.parse(ROOT / "java/pom.xml")
    ns = {"m": "http://maven.apache.org/POM/4.0.0"}
    timestamp = pom.findtext("m:properties/m:project.build.outputTimestamp", namespaces=ns)
    if timestamp is None:
        raise ValueError("Java release POM must define project.build.outputTimestamp")
    if timestamp.isdigit():
        end = datetime.fromtimestamp(int(timestamp), timezone.utc).year
    else:
        end = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc).year
    start = pom.findtext("m:inceptionYear", str(end), namespaces=ns)
    years = start if start == str(end) else f"{start}-{end}"
    with zipfile.ZipFile(artifact) as archive:
        require_entries(archive, artifact, ("META-INF/LICENSE", "META-INF/NOTICE"))
        if archive.read("META-INF/LICENSE").strip() != (ROOT / "LICENSE").read_bytes().strip():
            raise ValueError(f"{artifact} must contain the canonical Apache LICENSE")
        notice = archive.read("META-INF/NOTICE").decode("utf-8")
        if f"Copyright {years} The Apache Software Foundation" not in notice:
            raise ValueError(f"{artifact} NOTICE must use UTC copyright years {years}")
        names = archive.namelist()
        if any(name.startswith("native/") or name == "META-INF/LICENSE-binary" for name in names):
            raise ValueError(f"{artifact} contains binary-only resources")
        if javadoc:
            require_entries(archive, artifact, ("META-INF/MANIFEST.MF",))
            manifest = archive.read("META-INF/MANIFEST.MF").decode("utf-8")
            if not any(
                line.strip() in ("Build-Jdk-Spec: 1.8", "Build-Jdk-Spec: 8")
                for line in manifest.splitlines()
            ):
                raise ValueError(f"{artifact} must be generated with JDK 8")
            scripts = {name for name in names if name.endswith((".js", ".css"))}
            if scripts != {"script.js", "stylesheet.css"} or any(
                name.startswith(("legal/", "resources/")) for name in names
            ):
                raise ValueError(f"{artifact} contains unexpected Javadoc assets requiring license review")
    print(f"Verified Java {'Javadoc' if javadoc else 'sources'} JAR: {artifact}")


def cargo_metadata(target):
    return json.loads(
        subprocess.check_output(
            [
                "cargo",
                "metadata",
                "--format-version",
                "1",
                "--locked",
                "--filter-platform",
                target,
            ],
            cwd=ROOT,
            text=True,
        )
    )


def resolved_runtime_dependencies(metadata, root_name):
    packages = {package["id"]: package for package in metadata["packages"]}
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    roots = [
        package["id"]
        for package in metadata["packages"]
        if package["source"] is None and package["name"] == root_name
    ]
    if len(roots) != 1:
        raise ValueError(f"expected one workspace package named {root_name}")

    resolved = set()
    pending = roots
    while pending:
        package_id = pending.pop()
        if package_id in resolved:
            continue
        resolved.add(package_id)
        for dependency in nodes[package_id]["deps"]:
            if any(
                kind["kind"] in (None, "normal")
                for kind in dependency["dep_kinds"]
            ):
                pending.append(dependency["pkg"])

    return [
        packages[package_id]
        for package_id in resolved
        if packages[package_id]["source"] is not None
    ]


def verify_source_legal_inventory():
    java_license = ROOT / "java/src/main/binary-resources/META-INF/LICENSE"
    if java_license.read_bytes() != binary_license_content("LICENSE-binary"):
        raise ValueError(
            "Java binary LICENSE is stale: regenerate it from LICENSE and "
            "LICENSE-binary with tools/generate_license_reports.py"
        )
    metadata_by_target = [cargo_metadata(target) for target in SUPPORTED_TARGETS]
    for root_name, license_name in (
        ("paimon-mosaic-ffi", "LICENSE-binary-ffi"),
        ("paimon-mosaic-jni", "LICENSE-binary"),
    ):
        dependencies = {}
        for metadata in metadata_by_target:
            for package in resolved_runtime_dependencies(metadata, root_name):
                marker = f"{package['name']} {package['version']}"
                dependencies[marker] = package.get("license") or ""

        binary_license = (ROOT / license_name).read_text(encoding="utf-8")
        missing = []
        for marker, expression in sorted(dependencies.items()):
            needs_additional_text = (
                "Apache-2.0" not in expression or " AND " in expression
            )
            if needs_additional_text and marker not in binary_license:
                missing.append(f"{marker} ({expression})")

        for marker in (
            "Rust standard library 1.97.1",
            "zstd-sys 2.0.16+zstd.1.5.7",
        ):
            if marker not in binary_license:
                missing.append(marker)

        if missing:
            raise ValueError(
                f"{license_name} does not account for {root_name} runtime "
                f"dependencies: {', '.join(missing)}"
            )

    for name in (
        "LICENSE",
        "NOTICE",
        "LICENSE-binary",
        "LICENSE-binary-ffi",
        "ffi/DEPENDENCIES.rust.tsv",
        "jni/DEPENDENCIES.rust.tsv",
        "java/src/main/binary-resources/META-INF/NOTICE",
    ):
        if not (ROOT / name).is_file():
            raise ValueError(f"repository legal file is missing: {name}")

    print("Verified binary legal inventory against native Rust dependencies")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", nargs="+", default=[], help="Wheel path or glob")
    parser.add_argument("--jar", nargs="+", default=[], help="JAR path or glob")
    parser.add_argument("--sources-jar", nargs="+", default=[], help="Java sources JAR path or glob")
    parser.add_argument("--javadoc-jar", nargs="+", default=[], help="JDK 8 Javadoc JAR path or glob")
    parser.add_argument(
        "--jar-native",
        action="append",
        choices=JAR_NATIVE_FILES,
        help=(
            "Native entry required in a JAR. Repeat for a partial-platform CI build; "
            "the default requires every release platform."
        ),
    )
    parser.add_argument(
        "--source",
        action="store_true",
        help="Verify LICENSE-binary against native Rust runtime dependencies",
    )
    args = parser.parse_args()

    if not any((args.wheel, args.jar, args.source, args.sources_jar, args.javadoc_jar)):
        parser.error("at least one artifact path or --source is required")

    try:
        if args.source:
            verify_source_legal_inventory()
        for artifact in resolve_artifacts(args.wheel):
            verify_wheel(artifact)
        for artifact in resolve_artifacts(args.jar):
            verify_jar(artifact, args.jar_native or JAR_NATIVE_FILES)
        for artifact in resolve_artifacts(args.sources_jar):
            verify_java_auxiliary(artifact)
        for artifact in resolve_artifacts(args.javadoc_jar):
            verify_java_auxiliary(artifact, javadoc=True)
    except (
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
        ET.ParseError,
        zipfile.BadZipFile,
    ) as error:
        print(f"Binary artifact verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
