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

from __future__ import annotations

import importlib.util
import stat
import sys
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from unittest import mock
from zipfile import ZipFile, ZipInfo


TOOLS_DIRECTORY = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = TOOLS_DIRECTORY.parent
sys.path.insert(0, str(TOOLS_DIRECTORY))

import verify_java_jars  # noqa: E402
import native_binary  # noqa: E402


EXPECTED_TARGETS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
)
EXPECTED_NATIVE_ENTRIES = {
    "native/linux/x86_64/libpaimon_mosaic_jni.so": "x86_64-unknown-linux-gnu",
    "native/linux/aarch64/libpaimon_mosaic_jni.so": "aarch64-unknown-linux-gnu",
    "native/macos/aarch64/libpaimon_mosaic_jni.dylib": "aarch64-apple-darwin",
    "native/windows/x86_64/paimon_mosaic_jni.dll": "x86_64-pc-windows-msvc",
}


class VerifyJavaJarsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "LICENSE").write_bytes(b"repository license\n")
        (self.root / "NOTICE").write_bytes(b"repository notice\n")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_jar(
        self, name: str, entries: list[tuple[str | ZipInfo, bytes]]
    ) -> Path:
        path = self.root / name
        with ZipFile(path, "w") as archive:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for entry, contents in entries:
                    archive.writestr(entry, contents)
        return path

    def replace_jar_entry_name(
        self, path: Path, original: str, replacement: str
    ) -> None:
        original_bytes = original.encode()
        replacement_bytes = replacement.encode()
        self.assertEqual(len(original_bytes), len(replacement_bytes))
        contents = path.read_bytes()
        self.assertGreaterEqual(contents.count(original_bytes), 2)
        path.write_bytes(contents.replace(original_bytes, replacement_bytes))

    def verify_classifier(self, path: Path) -> None:
        with redirect_stdout(StringIO()):
            verify_java_jars.verify_classifier(path, self.root)

    def classifier_entries(
        self, *extra_entries: tuple[str | ZipInfo, bytes]
    ) -> list[tuple[str | ZipInfo, bytes]]:
        return [
            ("META-INF/LICENSE", (self.root / "LICENSE").read_bytes()),
            ("META-INF/NOTICE", (self.root / "NOTICE").read_bytes()),
            *extra_entries,
        ]

    def prepare_main_jar_fixture(
        self, native_entries: Iterable[str]
    ) -> list[tuple[str, bytes]]:
        binary_resources = self.root / "java/src/main/binary-resources/META-INF"
        report_paths = [
            f"META-INF/licenses/{target}/THIRD-PARTY-LICENSES.html"
            for target in EXPECTED_TARGETS
        ]
        license_contents = "\n".join(report_paths).encode()
        (binary_resources / "LICENSE").parent.mkdir(parents=True, exist_ok=True)
        (binary_resources / "LICENSE").write_bytes(license_contents)
        (binary_resources / "NOTICE").write_bytes(b"Apache Arrow\n")

        entries = [
            ("META-INF/LICENSE", license_contents),
            ("META-INF/NOTICE", b"Apache Arrow\n"),
        ]
        for target, report_path in zip(EXPECTED_TARGETS, report_paths):
            report_contents = (
                f"{target}\nFor Zstandard software\nApache Arrow\n".encode()
            )
            report_source = binary_resources / report_path.removeprefix("META-INF/")
            report_source.parent.mkdir(parents=True, exist_ok=True)
            report_source.write_bytes(report_contents)
            entries.append((report_path, report_contents))
        entries.extend(
            (native_entry, native_entry.encode()) for native_entry in native_entries
        )
        return entries

    def test_rejects_unsafe_entry_paths(self) -> None:
        symlink = ZipInfo("link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        cases = {
            "absolute": "/escape",
            "windows_absolute": "C:/escape",
            "backslash": "dir\\file",
            "dot_dot": "dir/../escape",
            "symlink": symlink,
        }

        for case, entry in cases.items():
            with self.subTest(case=case):
                path = self.write_jar(
                    f"{case}.jar",
                    self.classifier_entries((entry, b"contents")),
                )
                with self.assertRaises(ValueError):
                    self.verify_classifier(path)

    def test_rejects_duplicate_raw_entry_names(self) -> None:
        path = self.write_jar(
            "duplicate-raw.jar",
            self.classifier_entries(
                ("duplicate", b"first"),
                ("duplicate", b"second"),
            ),
        )

        with self.assertRaisesRegex(ValueError, "duplicate raw entry name"):
            self.verify_classifier(path)

    def test_rejects_duplicate_normalized_entry_names(self) -> None:
        path = self.write_jar(
            "duplicate-normalized.jar",
            self.classifier_entries(
                ("path/file", b"first"),
                ("path/./file", b"second"),
            ),
        )

        with self.assertRaisesRegex(ValueError, "duplicate normalized entry names"):
            self.verify_classifier(path)

    def test_rejects_oversized_entry_before_archive_read(self) -> None:
        path = self.write_jar(
            "oversized-entry.jar",
            self.classifier_entries(("payload/oversized.bin", b"x" * 4097)),
        )

        with mock.patch.object(
            verify_java_jars,
            "MAX_ARCHIVE_ENTRY_SIZE",
            4096,
            create=True,
        ):
            with mock.patch.object(
                ZipFile,
                "read",
                side_effect=AssertionError("archive.read must not be called"),
            ):
                with self.assertRaisesRegex(
                    ValueError, r"oversized\.bin.*size limit"
                ):
                    self.verify_classifier(path)

    def test_rejects_nul_truncated_entry_name(self) -> None:
        native_entry = next(iter(EXPECTED_NATIVE_ENTRIES))
        placeholder = native_entry + "Xhidden.txt"
        malicious = native_entry + "\0hidden.txt"
        jar_entries = self.prepare_main_jar_fixture(
            entry for entry in EXPECTED_NATIVE_ENTRIES if entry != native_entry
        )
        path = self.write_jar(
            "nul-truncated.jar",
            [*jar_entries, (placeholder, native_entry.encode())],
        )
        self.replace_jar_entry_name(path, placeholder, malicious)

        with mock.patch.object(verify_java_jars, "verify_native_target"):
            with self.assertRaisesRegex(ValueError, "invalid archive entry path"):
                with redirect_stdout(StringIO()):
                    verify_java_jars.verify_main_jar(path, self.root, True)

    def test_classifier_legal_files_must_byte_match_repository_root(self) -> None:
        valid = self.write_jar("valid.jar", self.classifier_entries())
        self.verify_classifier(valid)

        wrong_license = self.write_jar(
            "wrong-license.jar",
            [
                ("META-INF/LICENSE", b"Apache License but not the repository file\n"),
                ("META-INF/NOTICE", (self.root / "NOTICE").read_bytes()),
            ],
        )
        with self.assertRaisesRegex(ValueError, "root LICENSE"):
            self.verify_classifier(wrong_license)

        wrong_notice = self.write_jar(
            "wrong-notice.jar",
            [
                ("META-INF/LICENSE", (self.root / "LICENSE").read_bytes()),
                ("META-INF/NOTICE", b"not the repository notice\n"),
            ],
        )
        with self.assertRaisesRegex(ValueError, "root NOTICE"):
            self.verify_classifier(wrong_notice)

    def test_classifier_does_not_treat_java_class_magic_as_macho(self) -> None:
        java_class = (
            b"\xca\xfe\xba\xbe"
            b"\x00\x00"
            b"\x00\x34"
            b"\x00\x05"
            b"\x01\x00\x07Example"
            b"\x07\x00\x01"
            b"\x01\x00\x10java/lang/Object"
            b"\x07\x00\x03"
            b"\x00\x21"
            b"\x00\x02"
            b"\x00\x04"
            b"\x00\x00"
            b"\x00\x00"
            b"\x00\x00"
            b"\x00\x00"
        )
        path = self.write_jar(
            "java-class.jar",
            self.classifier_entries(
                ("org/apache/paimon/mosaic/Example.class", java_class)
            ),
        )

        self.verify_classifier(path)

        malformed = self.write_jar(
            "malformed-java-class.jar",
            self.classifier_entries(
                (
                    "org/apache/paimon/mosaic/Malformed.class",
                    b"\xca\xfe\xba\xbe\x00\x00\x00\x34\x00\x01",
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "binary-only"):
            self.verify_classifier(malformed)

    def test_oversized_java_class_is_not_fully_read(self) -> None:
        class RecordingSource(BytesIO):
            def __init__(self, data: bytes):
                super().__init__(data)
                self.read_sizes = []

            def read(self, size: int = -1) -> bytes:
                self.read_sizes.append(size)
                if size < 0:
                    raise AssertionError("oversized class was read without a bound")
                return super().read(size)

        source = RecordingSource(
            verify_java_jars.JAVA_CLASS_MAGIC + b"\x00" * 60
        )
        with mock.patch.object(
            verify_java_jars,
            "MAX_JAVA_CLASS_SIZE",
            64,
            create=True,
        ):
            self.assertEqual(
                verify_java_jars.native_binary_magic(
                    source,
                    65,
                    "org/apache/paimon/mosaic/Huge.class",
                ),
                "Mach-O",
            )
        self.assertEqual(source.read_sizes, [64])

    def test_target_matrix_guard_rejects_drift(self) -> None:
        with mock.patch.dict(
            verify_java_jars.NATIVE_ENTRIES,
            {"native/unsupported/libmosaic.so": "unsupported-target"},
        ):
            with self.assertRaisesRegex(RuntimeError, "target matrices"):
                verify_java_jars._validate_target_matrix()

    def test_target_matrix_is_validated_during_import(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "verify_java_jars_matrix_probe",
            TOOLS_DIRECTORY / "verify_java_jars.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.object(
            native_binary,
            "TARGET_ARCHITECTURE",
            {"unsupported-target": ("ELF", "x86_64")},
        ):
            with self.assertRaisesRegex(RuntimeError, "target matrices"):
                spec.loader.exec_module(module)

    def test_main_jar_keeps_target_report_and_native_validation(self) -> None:
        self.assertEqual(EXPECTED_TARGETS, verify_java_jars.TARGETS)
        self.assertEqual(EXPECTED_NATIVE_ENTRIES, verify_java_jars.NATIVE_ENTRIES)
        jar_entries = self.prepare_main_jar_fixture(EXPECTED_NATIVE_ENTRIES)
        path = self.write_jar("main.jar", jar_entries)

        with mock.patch.object(
            verify_java_jars, "verify_native_target"
        ) as verify_native:
            with redirect_stdout(StringIO()):
                verify_java_jars.verify_main_jar(path, self.root, True)
            verify_native.assert_has_calls(
                [
                    mock.call(
                        native_entry.encode(),
                        native_target,
                        native_entry,
                        symbol_family="JNI",
                    )
                    for native_entry, native_target in (
                        EXPECTED_NATIVE_ENTRIES.items()
                    )
                ],
                any_order=True,
            )
            self.assertEqual(
                len(EXPECTED_NATIVE_ENTRIES), verify_native.call_count
            )

            injected = self.write_jar(
                "main-with-undeclared-native.jar",
                [
                    *jar_entries,
                    ("payload/undeclared-native.bin", b"\x7fELF" + b"\0" * 64),
                ],
            )
            with self.assertRaisesRegex(ValueError, "unexpected native"):
                with redirect_stdout(StringIO()):
                    verify_java_jars.verify_main_jar(injected, self.root, True)

    def test_main_jar_propagates_native_verification_failure(self) -> None:
        path = self.write_jar(
            "invalid-native.jar",
            self.prepare_main_jar_fixture(EXPECTED_NATIVE_ENTRIES),
        )

        with self.assertRaisesRegex(ValueError, "unrecognized native binary"):
            with redirect_stdout(StringIO()):
                verify_java_jars.verify_main_jar(path, self.root, True)

    def test_main_jar_rejects_invalid_legal_content(self) -> None:
        first_target = EXPECTED_TARGETS[0]
        first_report = (
            f"META-INF/licenses/{first_target}/THIRD-PARTY-LICENSES.html"
        )

        def replace(entries, name, contents):
            return [
                (entry, contents if entry == name else data)
                for entry, data in entries
            ]

        def remove(entries, name):
            return [(entry, data) for entry, data in entries if entry != name]

        def missing_license(entries, _resources):
            return remove(entries, "META-INF/LICENSE")

        def cross_target_inventory(entries, _resources):
            return [*entries, ("META-INF/DEPENDENCIES.rust.tsv", b"inventory")]

        def wrong_license(entries, _resources):
            return replace(entries, "META-INF/LICENSE", b"wrong license\n")

        def wrong_notice(entries, _resources):
            return replace(entries, "META-INF/NOTICE", b"wrong notice\n")

        def notice_without_arrow(entries, resources):
            contents = b"project notice\n"
            (resources / "NOTICE").write_bytes(contents)
            return replace(entries, "META-INF/NOTICE", contents)

        def license_without_report_link(entries, resources):
            license_path = resources / "LICENSE"
            contents = license_path.read_bytes().replace(first_report.encode(), b"")
            license_path.write_bytes(contents)
            return replace(entries, "META-INF/LICENSE", contents)

        def wrong_report(entries, _resources):
            return replace(entries, first_report, b"wrong report\n")

        def report_without_target(entries, resources):
            contents = b"For Zstandard software\nApache Arrow\n"
            report_source = resources / first_report.removeprefix("META-INF/")
            report_source.write_bytes(contents)
            return replace(entries, first_report, contents)

        def report_without_zstandard(entries, resources):
            contents = f"{first_target}\nApache Arrow\n".encode()
            report_source = resources / first_report.removeprefix("META-INF/")
            report_source.write_bytes(contents)
            return replace(entries, first_report, contents)

        def report_without_arrow(entries, resources):
            contents = f"{first_target}\nFor Zstandard software\n".encode()
            report_source = resources / first_report.removeprefix("META-INF/")
            report_source.write_bytes(contents)
            return replace(entries, first_report, contents)

        cases = (
            ("missing_license", "missing legal files", missing_license),
            (
                "cross_target_inventory",
                "cross-target repository dependency inventory",
                cross_target_inventory,
            ),
            ("wrong_license", "binary-specific LICENSE", wrong_license),
            ("wrong_notice", "binary-specific NOTICE", wrong_notice),
            (
                "notice_without_arrow",
                "omits the bundled Apache Arrow",
                notice_without_arrow,
            ),
            (
                "license_without_report_link",
                "LICENSE does not point",
                license_without_report_link,
            ),
            ("wrong_report", "differs from its generated source", wrong_report),
            (
                "report_without_target",
                "does not identify its target",
                report_without_target,
            ),
            (
                "report_without_zstandard",
                "missing 'For Zstandard software'",
                report_without_zstandard,
            ),
            (
                "report_without_arrow",
                "missing 'Apache Arrow'",
                report_without_arrow,
            ),
        )

        for case, error, mutate in cases:
            with self.subTest(case=case):
                entries = self.prepare_main_jar_fixture(EXPECTED_NATIVE_ENTRIES)
                resources = self.root / "java/src/main/binary-resources/META-INF"
                path = self.write_jar(
                    f"{case}.jar",
                    mutate(entries, resources),
                )
                with mock.patch.object(
                    verify_java_jars, "verify_native_target"
                ):
                    with self.assertRaisesRegex(ValueError, error):
                        with redirect_stdout(StringIO()):
                            verify_java_jars.verify_main_jar(
                                path, self.root, True
                            )

    def test_release_main_jar_requires_all_declared_natives(self) -> None:
        for index, omitted_native in enumerate(EXPECTED_NATIVE_ENTRIES):
            with self.subTest(omitted_native=omitted_native):
                jar_entries = self.prepare_main_jar_fixture(
                    [
                        native_entry
                        for native_entry in EXPECTED_NATIVE_ENTRIES
                        if native_entry != omitted_native
                    ]
                )
                path = self.write_jar(f"missing-native-{index}.jar", jar_entries)

                with mock.patch.object(verify_java_jars, "verify_native_target"):
                    with redirect_stdout(StringIO()):
                        verify_java_jars.verify_main_jar(path, self.root, False)
                    with self.assertRaisesRegex(
                        ValueError, "differ from the four declared targets"
                    ):
                        with redirect_stdout(StringIO()):
                            verify_java_jars.verify_main_jar(path, self.root, True)

    def test_main_returns_success_and_checks_javadoc_classifier(self) -> None:
        main_jar = self.write_jar(
            "main.jar",
            self.prepare_main_jar_fixture(EXPECTED_NATIVE_ENTRIES),
        )
        sources = self.write_jar("sources.jar", self.classifier_entries())
        javadoc = self.write_jar("javadoc.jar", self.classifier_entries())
        arguments = [
            "verify_java_jars.py",
            "--main",
            str(main_jar),
            "--sources",
            str(sources),
            "--javadoc",
            str(javadoc),
            "--require-all-natives",
        ]

        with mock.patch.object(
            verify_java_jars, "repository_root", return_value=self.root
        ):
            with mock.patch.object(
                verify_java_jars, "verify_native_target"
            ):
                with mock.patch.object(sys, "argv", arguments):
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        self.assertEqual(verify_java_jars.main(), 0)

                broken_javadoc = self.write_jar(
                    "broken-javadoc.jar",
                    [("META-INF/LICENSE", (self.root / "LICENSE").read_bytes())],
                )
                broken_arguments = [
                    *arguments[:-3],
                    "--javadoc",
                    str(broken_javadoc),
                    "--require-all-natives",
                ]
                with mock.patch.object(sys, "argv", broken_arguments):
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        self.assertEqual(verify_java_jars.main(), 1)

    def test_main_fails_closed_on_non_zip_artifact(self) -> None:
        # A truncated / non-zip JAR raises zipfile.BadZipFile. It must be caught
        # at the CLI boundary and reported as a failure, not escape as an
        # uncaught traceback.
        broken = self.root / "broken.jar"
        broken.write_bytes(b"not a zip file")
        classifier = self.write_jar("sources.jar", self.classifier_entries())
        arguments = [
            "verify_java_jars.py",
            "--main",
            str(broken),
            "--sources",
            str(classifier),
            "--javadoc",
            str(classifier),
        ]
        with mock.patch.object(
            verify_java_jars, "repository_root", return_value=self.root
        ):
            with mock.patch.object(sys, "argv", arguments):
                stderr = StringIO()
                with redirect_stdout(StringIO()), redirect_stderr(stderr):
                    self.assertEqual(verify_java_jars.main(), 1)
                self.assertIn(
                    "Java artifact verification failed", stderr.getvalue()
                )

    def test_classifier_rejects_binary_only_entries_by_name(self) -> None:
        # These entries carry plain-text content with no native magic, so only
        # the name-based forbidden rule can reject them — including a
        # THIRD-PARTY-LICENSES.html report placed at the archive root.
        for entry in (
            "native/notes.txt",
            "META-INF/DEPENDENCIES.rust.tsv",
            "THIRD-PARTY-LICENSES.html",
            "META-INF/licenses/x86_64-unknown-linux-gnu/THIRD-PARTY-LICENSES.html",
        ):
            with self.subTest(entry=entry):
                path = self.write_jar(
                    "classifier-by-name.jar",
                    self.classifier_entries((entry, b"not a binary\n")),
                )
                with self.assertRaisesRegex(ValueError, "binary-only files"):
                    self.verify_classifier(path)

    def test_release_profile_verifies_jars_before_gpg_signing(self) -> None:
        namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
        root = ET.parse(REPOSITORY_ROOT / "java/pom.xml").getroot()
        release_profile = next(
            profile
            for profile in root.findall("m:profiles/m:profile", namespace)
            if profile.findtext("m:id", namespaces=namespace) == "release"
        )
        plugins = release_profile.findall("m:build/m:plugins/m:plugin", namespace)
        artifact_ids = [
            plugin.findtext("m:artifactId", namespaces=namespace)
            for plugin in plugins
        ]

        self.assertLess(
            artifact_ids.index("exec-maven-plugin"),
            artifact_ids.index("maven-gpg-plugin"),
        )
        exec_plugin = plugins[artifact_ids.index("exec-maven-plugin")]
        verifier_executions = [
            execution
            for execution in exec_plugin.findall(
                "m:executions/m:execution", namespace
            )
            if execution.findtext("m:id", namespaces=namespace)
            == "verify-release-jars"
        ]
        self.assertEqual(len(verifier_executions), 1)
        execution = verifier_executions[0]
        self.assertEqual(
            execution.findtext("m:phase", namespaces=namespace),
            "verify",
        )
        self.assertEqual(
            [
                goal.text
                for goal in execution.findall("m:goals/m:goal", namespace)
            ],
            ["exec"],
        )
        self.assertEqual(
            execution.findtext(
                "m:configuration/m:executable", namespaces=namespace
            ),
            "python3",
        )
        skip = execution.findtext(
            "m:configuration/m:skip", namespaces=namespace
        )
        self.assertIn(
            skip.strip().lower() if skip is not None else None,
            (None, "false"),
        )
        arguments = [
            argument.text
            for argument in execution.findall(
                "m:configuration/m:arguments/m:argument", namespace
            )
        ]
        self.assertIn("tools/verify_java_jars.py", arguments)
        self.assertIn("--require-all-natives", arguments)


if __name__ == "__main__":
    unittest.main()
