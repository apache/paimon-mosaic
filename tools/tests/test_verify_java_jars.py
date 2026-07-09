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

import stat
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock
from zipfile import ZipFile, ZipInfo


TOOLS_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIRECTORY))

import verify_java_jars  # noqa: E402


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

    def test_main_jar_keeps_target_report_and_native_validation(self) -> None:
        binary_resources = self.root / "java/src/main/binary-resources/META-INF"
        report_paths = [
            f"META-INF/licenses/{target}/THIRD-PARTY-LICENSES.html"
            for target in verify_java_jars.TARGETS
        ]
        license_contents = "\n".join(report_paths).encode()
        (binary_resources / "LICENSE").parent.mkdir(parents=True)
        (binary_resources / "LICENSE").write_bytes(license_contents)
        (binary_resources / "NOTICE").write_bytes(b"Apache Arrow\n")

        jar_entries = [
            ("META-INF/LICENSE", license_contents),
            ("META-INF/NOTICE", b"Apache Arrow\n"),
        ]
        for target, report_path in zip(verify_java_jars.TARGETS, report_paths):
            report_contents = (
                f"{target}\nFor Zstandard software\nApache Arrow\n".encode()
            )
            report_source = binary_resources / report_path.removeprefix("META-INF/")
            report_source.parent.mkdir(parents=True, exist_ok=True)
            report_source.write_bytes(report_contents)
            jar_entries.append((report_path, report_contents))

        for native_entry in verify_java_jars.NATIVE_ENTRIES:
            jar_entries.append((native_entry, native_entry.encode()))
        path = self.write_jar("main.jar", jar_entries)

        with mock.patch.object(
            verify_java_jars, "verify_native_target"
        ) as verify_native:
            with redirect_stdout(StringIO()):
                verify_java_jars.verify_main_jar(path, self.root, True)

        verify_native.assert_has_calls(
            [
                mock.call(native_entry.encode(), native_target, native_entry)
                for native_entry, native_target in (
                    verify_java_jars.NATIVE_ENTRIES.items()
                )
            ],
            any_order=True,
        )
        self.assertEqual(len(verify_java_jars.NATIVE_ENTRIES), verify_native.call_count)


if __name__ == "__main__":
    unittest.main()
