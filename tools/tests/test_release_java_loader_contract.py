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

import ctypes
import os
import platform
import re
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/release-java.yml"
RUST_JNI = ROOT / "jni/src/lib.rs"
JAVA_NATIVE_LIB = (
    ROOT / "java/src/main/java/org/apache/paimon/mosaic/NativeLib.java"
)
JNI_PREFIX = "Java_org_apache_paimon_mosaic_NativeLib_"
EXPECTED_JNI_EXPORT_COUNT = 24
RESOLUTION_LOOP = (
    "native_library = ctypes.CDLL(library)\n"
    "for symbol in jni_exports:\n"
    "    getattr(native_library, symbol)"
)


def rust_jni_exports(source: str) -> tuple[str, ...]:
    exports = tuple(
        re.findall(
            rf'(?m)^pub\s+extern\s+"system"\s+fn\s+'
            rf"({re.escape(JNI_PREFIX)}[A-Za-z0-9_]+)",
            source,
        )
    )
    assert len(exports) == len(set(exports)), "duplicate Rust JNI exports"
    return exports


def rust_generic_jni_exports(source: str) -> tuple[str, ...]:
    return tuple(
        re.findall(
            rf'(?m)^pub\s+extern\s+"system"\s+fn\s+'
            rf"({re.escape(JNI_PREFIX)}[A-Za-z0-9_]+)\s*<",
            source,
        )
    )


def java_native_methods(source: str) -> tuple[str, ...]:
    methods = tuple(
        re.findall(
            r"\bstatic\s+native\s+[A-Za-z0-9_\[\]<>.?]+\s+"
            r"(native[A-Za-z0-9_]+)\s*\(",
            source,
        )
    )
    assert len(methods) == len(set(methods)), "duplicate Java native methods"
    return methods


def loader_python_script() -> str:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step_name = "      - name: Smoke-load JNI library on target runner"
    step_start = workflow.index(step_name)
    step_end = workflow.find("\n      - ", step_start + len(step_name))
    step = workflow[step_start:] if step_end == -1 else workflow[step_start:step_end]

    run_marker = "        run: |\n"
    run_start = step.index(run_marker) + len(run_marker)
    run = textwrap.dedent(step[run_start:])

    heredoc_marker = "python - <<'PY'\n"
    if heredoc_marker in run:
        script_start = run.index(heredoc_marker) + len(heredoc_marker)
        script_end = run.index("\nPY", script_start)
        return run[script_start:script_end]

    python_c_marker = "python -c '\n"
    script_start = run.index(python_c_marker) + len(python_c_marker)
    script_end = run.index("\n'", script_start)
    return run[script_start:script_end]


def run_loader_script(
    script: str,
    rust_source: str,
    expected_exports: tuple[str, ...],
    monkeypatch,
) -> tuple[list[str], list[str]]:
    loaded_libraries: list[str] = []
    resolved_exports: list[str] = []
    expected_export_set = set(expected_exports)

    class FakeLibrary:
        def __getattr__(self, name: str):
            if name not in expected_export_set:
                raise AttributeError(name)
            resolved_exports.append(name)
            return object()

    def fake_cdll(path: str):
        loaded_libraries.append(path)
        return FakeLibrary()

    original_read_text = Path.read_text

    def fake_read_text(path: Path, *args, **kwargs) -> str:
        if path.as_posix().endswith("jni/src/lib.rs"):
            return rust_source
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.setenv("EXPECTED_ARCH", "x86_64")
    monkeypatch.setenv(
        "NATIVE_LIBRARY", "target/test/release/libpaimon_mosaic_jni.so"
    )
    monkeypatch.chdir(ROOT)

    exec(compile(script, "release-java-loader-smoke.py", "exec"), {})
    return loaded_libraries, resolved_exports


def assert_loader_resolves_all_exports(
    script: str,
    rust_source: str,
    expected_exports: tuple[str, ...],
    monkeypatch,
) -> None:
    loaded_libraries, resolved_exports = run_loader_script(
        script, rust_source, expected_exports, monkeypatch
    )
    assert loaded_libraries == [
        os.path.abspath("target/test/release/libpaimon_mosaic_jni.so")
    ]
    assert len(resolved_exports) == len(expected_exports), (
        f"resolved {len(resolved_exports)}/{len(expected_exports)} JNI exports"
    )
    assert set(resolved_exports) == set(expected_exports)


def source_contract() -> tuple[str, tuple[str, ...]]:
    rust_source = RUST_JNI.read_text(encoding="utf-8")
    java_source = JAVA_NATIVE_LIB.read_text(encoding="utf-8")
    exports = rust_jni_exports(rust_source)
    java_methods = java_native_methods(java_source)

    assert len(exports) == EXPECTED_JNI_EXPORT_COUNT
    assert len(java_methods) == EXPECTED_JNI_EXPORT_COUNT
    generic_exports = rust_generic_jni_exports(rust_source)
    assert len(generic_exports) == 6
    assert set(generic_exports) <= set(exports)
    assert {export.removeprefix(JNI_PREFIX) for export in exports} == set(
        java_methods
    )
    return rust_source, exports


def replace_once(contents: str, original: str, replacement: str) -> str:
    assert contents.count(original) == 1
    return contents.replace(original, replacement, 1)


def test_rust_jni_exports_match_java_native_declarations_24_for_24():
    source_contract()


def test_target_runner_loader_resolves_all_java_native_exports(monkeypatch):
    rust_source, exports = source_contract()

    probe_target = f"{JNI_PREFIX}nativeWriterRowGroupStatMins"
    assert probe_target in rust_generic_jni_exports(rust_source)
    probe_export = f"{JNI_PREFIX}nativeContractProbe"
    probe_source = rust_source.replace(probe_target, probe_export, 1)
    probe_exports = tuple(
        probe_export if export == probe_target else export for export in exports
    )

    script = loader_python_script()
    assert "native_binary" not in script
    assert_loader_resolves_all_exports(
        script, probe_source, probe_exports, monkeypatch
    )


def test_loader_contract_rejects_skipping_one_non_sentinel_export(monkeypatch):
    rust_source, exports = source_contract()
    script = loader_python_script()
    mutated = replace_once(
        script,
        "for symbol in jni_exports:",
        "for symbol in jni_exports[:7] + jni_exports[8:]:",
    )

    with pytest.raises(AssertionError, match=r"resolved 23/24 JNI exports"):
        assert_loader_resolves_all_exports(
            mutated, rust_source, exports, monkeypatch
        )


def test_loader_contract_rejects_load_without_symbol_resolution(monkeypatch):
    rust_source, exports = source_contract()
    script = loader_python_script()
    mutated = replace_once(script, RESOLUTION_LOOP, "ctypes.CDLL(library)")

    with pytest.raises(AssertionError, match=r"resolved 0/24 JNI exports"):
        assert_loader_resolves_all_exports(
            mutated, rust_source, exports, monkeypatch
        )
