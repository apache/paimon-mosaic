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

from pathlib import Path
import sys
import zipfile

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_binary_artifact as verifier  # noqa: E402


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    for name, content in {
        "LICENSE": b"Apache license\n",
        "LICENSE-binary": b"JNI third-party licenses\n",
        "LICENSE-binary-ffi": b"FFI third-party licenses\n",
        "java/src/main/binary-resources/META-INF/NOTICE": b"Binary notices\n",
        "java/pom.xml": (
            b'<project xmlns="http://maven.apache.org/POM/4.0.0">'
            b'<inceptionYear>2026</inceptionYear><properties>'
            b'<project.build.outputTimestamp>2026-01-01T00:00:00Z'
            b'</project.build.outputTimestamp></properties></project>'
        ),
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(verifier, "ROOT", root)
    return root


def jar_entries():
    return {
        "META-INF/LICENSE": b"Apache license\n\nJNI third-party licenses\n",
        "META-INF/LICENSE-binary": b"JNI third-party licenses\n",
        "META-INF/NOTICE": b"Binary notices\n",
        "META-INF/DEPENDENCIES": b"Dependencies\n",
        "org/apache/paimon/mosaic/NativeLib.class": b"class",
        **{name: b"native" for name in verifier.JAR_NATIVE_FILES},
    }


def wheel_entries():
    return {
        "mosaic-0.3.0.dist-info/METADATA": b"Name: paimon-mosaic\nVersion: 0.3.0\n",
        "mosaic/libpaimon_mosaic_ffi.so": b"native",
        **{
            f"{prefix}/{name}": content
            for prefix in ("mosaic", "mosaic-0.3.0.dist-info/licenses")
            for name, content in {
                "LICENSE": b"Apache license\n\nFFI third-party licenses\n",
                "LICENSE-binary": b"FFI third-party licenses\n",
                "NOTICE": b"Binary notices\n",
            }.items()
        },
    }


def verify_archive(tmp_path, kind, entries):
    artifact = tmp_path / f"artifact.{kind}"
    with zipfile.ZipFile(artifact, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    if kind == "jar":
        verifier.verify_jar(artifact, verifier.JAR_NATIVE_FILES)
    else:
        verifier.verify_wheel(artifact)


@pytest.mark.parametrize("kind,entries", [("jar", jar_entries()), ("whl", wheel_entries())])
def test_accepts_complete_binary_license(repository, tmp_path, kind, entries):
    verify_archive(tmp_path, kind, entries)


@pytest.mark.parametrize(
    "kind,entry",
    [
        ("jar", "META-INF/LICENSE"),
        ("whl", "mosaic/LICENSE"),
        ("whl", "mosaic-0.3.0.dist-info/licenses/LICENSE"),
    ],
)
@pytest.mark.parametrize(
    "license_content",
    [
        b"Apache license\n",
        b"Apache license\n\nSee missing/THIRD-PARTY-LICENSES.html\n",
        b"Apache license\n\nOutdated third-party inventory\n",
    ],
)
def test_rejects_incomplete_primary_license(
    repository, tmp_path, kind, entry, license_content
):
    entries = jar_entries() if kind == "jar" else wheel_entries()
    entries[entry] = license_content
    with pytest.raises(ValueError, match="complete third-party license inventory"):
        verify_archive(tmp_path, kind, entries)


@pytest.mark.parametrize(
    "kind,entry",
    [("jar", "META-INF/LICENSE-binary"), ("whl", "mosaic/LICENSE-binary")],
)
def test_still_rejects_missing_inventory(repository, tmp_path, kind, entry):
    entries = jar_entries() if kind == "jar" else wheel_entries()
    del entries[entry]
    with pytest.raises(ValueError, match="missing entries"):
        verify_archive(tmp_path, kind, entries)


def test_rejects_stale_java_license_before_resolving_dependencies(repository):
    path = repository / "java/src/main/binary-resources/META-INF/LICENSE"
    path.write_bytes(b"Apache license\n")
    with pytest.raises(ValueError, match="Java binary LICENSE is stale"):
        verifier.verify_source_legal_inventory()


def auxiliary_entries():
    return {
        "META-INF/LICENSE": b"Apache license\n",
        "META-INF/NOTICE": b"Mosaic\nCopyright 2026 The Apache Software Foundation\n",
        "META-INF/MANIFEST.MF": b"Manifest-Version: 1.0\r\nBuild-Jdk-Spec: 1.8\r\n",
        "script.js": b"JDK8 script",
        "stylesheet.css": b"JDK8 stylesheet",
    }


@pytest.mark.parametrize("javadoc", [False, True])
def test_accepts_java_auxiliary_license_and_utc_notice(repository, tmp_path, javadoc):
    artifact = tmp_path / "auxiliary.jar"
    with zipfile.ZipFile(artifact, "w") as z:
        for name, data in auxiliary_entries().items():
            z.writestr(name, data)
    verifier.verify_java_auxiliary(artifact, javadoc=javadoc)


@pytest.mark.parametrize("entry,data,message", [
    ("META-INF/NOTICE", b"Copyright 2025 The Apache Software Foundation", "UTC copyright"),
    ("META-INF/LICENSE", b"incomplete", "canonical Apache LICENSE"),
    ("META-INF/MANIFEST.MF", b"Build-Jdk-Spec: 11", "generated with JDK 8"),
    ("resources/jquery.js", b"extra third party library", "license review"),
    ("native/linux/x86_64/library.so", b"binary", "binary-only"),
])
def test_rejects_invalid_java_auxiliary_artifact(repository, tmp_path, entry, data, message):
    entries = auxiliary_entries()
    entries[entry] = data
    artifact = tmp_path / "bad.jar"
    with zipfile.ZipFile(artifact, "w") as z:
        for name, value in entries.items():
            z.writestr(name, value)
    with pytest.raises(ValueError, match=message):
        verifier.verify_java_auxiliary(artifact, javadoc=True)
