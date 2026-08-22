#!/usr/bin/env python3

# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock


TOOLS_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIRECTORY))

import verify_release_versions  # noqa: E402


PACKAGE_NAMES = {
    "core": "paimon-mosaic-core",
    "ffi": "paimon-mosaic-ffi",
    "jni": "paimon-mosaic-jni",
    "cli": "paimon-mosaic-cli",
}


class VerifyReleaseVersionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_workspace(
        self,
        package_version: str,
        cli_requirement: str,
        ffi_requirement: str,
        jni_requirement: str,
    ) -> None:
        (self.root / "Cargo.toml").write_text(
            "[workspace]\n"
            'members = ["core", "ffi", "jni", "cli"]\n'
            'resolver = "2"\n',
            encoding="utf-8",
        )
        for directory, package_name in PACKAGE_NAMES.items():
            package = self.root / directory
            (package / "src").mkdir(parents=True)
            (package / "src/lib.rs").write_text("", encoding="utf-8")
            manifest = (
                "[package]\n"
                f'name = "{package_name}"\n'
                f'version = "{package_version}"\n'
                'edition = "2021"\n'
            )
            if directory == "cli":
                manifest += (
                    "\n[dependencies]\n"
                    "# Preserve dependency comments and unrelated formatting.\n"
                    "paimon-mosaic-core = { "
                    'path = "../core", '
                    f'version = "{cli_requirement}"'
                    " }\n"
                )
            elif directory == "ffi":
                manifest += (
                    "\n[dependencies.mosaic-core]\n"
                    'package = "paimon-mosaic-core"\n'
                    'path = "../core"\n'
                    f'version = "{ffi_requirement}"\n'
                )
            elif directory == "jni":
                manifest += (
                    "\n[target.'cfg(unix)'.build-dependencies]\n"
                    "mosaic-core = { "
                    'package = "paimon-mosaic-core", '
                    'path = "../core", '
                    f'version = "{jni_requirement}"'
                    " }\n"
                )
            (package / "Cargo.toml").write_text(manifest, encoding="utf-8")

    def write_release_components(
        self, version: str, java_snapshot: bool = True
    ) -> None:
        (self.root / "java").mkdir()
        java_version = f"{version}-SNAPSHOT" if java_snapshot else version
        (self.root / "java/pom.xml").write_text(
            '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
            f"  <version>{java_version}</version>\n"
            "</project>\n",
            encoding="utf-8",
        )
        (self.root / "python").mkdir()
        (self.root / "python/pyproject.toml").write_text(
            "[project]\n" f'version = "{version}"\n',
            encoding="utf-8",
        )
        lockfile = ["version = 4", ""]
        for package_name in PACKAGE_NAMES.values():
            lockfile.extend(
                [
                    "[[package]]",
                    f'name = "{package_name}"',
                    f'version = "{version}"',
                    "",
                ]
            )
        (self.root / "Cargo.lock").write_text(
            "\n".join(lockfile), encoding="utf-8"
        )

    def load_manifest(self, directory: str) -> dict:
        with (self.root / directory / "Cargo.toml").open("rb") as file:
            return tomllib.load(file)

    def test_updates_packages_and_all_versioned_workspace_path_dependencies(
        self,
    ) -> None:
        self.write_workspace(
            "0.3.0",
            "0.3.0",
            "^0.3.0",
            ">=0.3.0, <0.4.0",
        )

        updated = verify_release_versions.update_cargo_versions(
            self.root, "0.3.0", "0.4.0"
        )

        self.assertEqual(
            {path.relative_to(self.root).as_posix() for path in updated},
            {
                "cli/Cargo.toml",
                "core/Cargo.toml",
                "ffi/Cargo.toml",
                "jni/Cargo.toml",
            },
        )
        for directory in PACKAGE_NAMES:
            self.assertEqual(
                self.load_manifest(directory)["package"]["version"], "0.4.0"
            )
        self.assertEqual(
            self.load_manifest("cli")["dependencies"]["paimon-mosaic-core"][
                "version"
            ],
            "0.4.0",
        )
        self.assertEqual(
            self.load_manifest("ffi")["dependencies"]["mosaic-core"]["version"],
            "0.4.0",
        )
        self.assertEqual(
            self.load_manifest("jni")["target"]["cfg(unix)"][
                "build-dependencies"
            ]["mosaic-core"]["version"],
            "0.4.0",
        )
        self.assertIn(
            "# Preserve dependency comments",
            (self.root / "cli/Cargo.toml").read_text(encoding="utf-8"),
        )
        self.assertEqual(
            verify_release_versions.path_dependency_failures(self.root), []
        )
        subprocess.run(
            [
                "cargo",
                "update",
                "--manifest-path",
                str(self.root / "Cargo.toml"),
                "--workspace",
                "--offline",
            ],
            check=True,
            text=True,
            capture_output=True,
        )

    def test_update_rejects_a_path_dependency_outside_the_retry_window(
        self,
    ) -> None:
        # An earlier partial run or a manual edit left ffi pinned to an unrelated
        # version. Rewriting it silently would still pass the post-write check,
        # which only compares against the new version.
        self.write_workspace("0.3.0", "0.3.0", "0.9.9", "0.3.0")

        with self.assertRaisesRegex(
            ValueError,
            r"ffi/Cargo\.toml: path dependency mosaic-core requires 0\.9\.9",
        ):
            verify_release_versions.update_cargo_versions(
                self.root, "0.3.0", "0.4.0"
            )

        self.assertEqual(
            self.load_manifest("ffi")["dependencies"]["mosaic-core"]["version"],
            "0.9.9",
        )

    def test_update_repairs_a_partially_completed_bump_and_is_idempotent(
        self,
    ) -> None:
        self.write_workspace("0.4.0", "0.3.0", "0.4.0", "^0.4.0")

        verify_release_versions.update_cargo_versions(
            self.root, "0.3.0", "0.4.0"
        )
        first = {
            path: path.read_text(encoding="utf-8")
            for path in self.root.glob("*/Cargo.toml")
        }
        verify_release_versions.update_cargo_versions(
            self.root, "0.3.0", "0.4.0"
        )
        second = {
            path: path.read_text(encoding="utf-8")
            for path in self.root.glob("*/Cargo.toml")
        }

        self.assertEqual(first, second)
        self.assertEqual(
            self.load_manifest("cli")["dependencies"]["paimon-mosaic-core"][
                "version"
            ],
            "0.4.0",
        )

    def test_update_cargo_versions_accepts_a_symlinked_root(self) -> None:
        canonical_root = self.root / "canonical"
        canonical_root.mkdir()
        root = self.root / "workspace"
        root.symlink_to(canonical_root, target_is_directory=True)
        self.root = canonical_root
        self.write_workspace("0.3.0", "0.3.0", "0.3.0", "0.3.0")
        core = self.root / "core/Cargo.toml"
        core.write_text(
            core.read_text(encoding="utf-8").replace(
                'version = "0.3.0"', 'version = "0.2.0"', 1
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            r"core/Cargo\.toml: expected package version 0\.3\.0 or 0\.4\.0",
        ):
            verify_release_versions.update_cargo_versions(
                root, "0.3.0", "0.4.0"
            )

    def test_verifier_rejects_stale_path_dependency_requirement(self) -> None:
        self.write_workspace("0.4.0", "0.3.0", "0.4.0", "^0.4.0")
        self.write_release_components("0.4.0")

        failures = verify_release_versions.verify(
            "0.4.0", java_snapshot=True, root=self.root
        )

        self.assertEqual(len(failures), 1)
        self.assertIn("cli/Cargo.toml", failures[0])
        self.assertIn("requires 0.3.0", failures[0])
        self.assertIn("does not accept 0.4.0", failures[0])

    def test_path_dependency_failures_accepts_a_symlinked_root(self) -> None:
        canonical_root = self.root / "canonical"
        canonical_root.mkdir()
        root = self.root / "workspace"
        root.symlink_to(canonical_root, target_is_directory=True)
        self.root = canonical_root
        self.write_workspace("0.4.0", "0.3.0", "0.4.0", "^0.4.0")

        failures = verify_release_versions.path_dependency_failures(root)

        self.assertEqual(len(failures), 1)
        self.assertIn("cli/Cargo.toml", failures[0])
        self.assertIn("core/Cargo.toml", failures[0])

    def test_verifier_reports_each_stale_release_component(self) -> None:
        version = "0.4.0"
        stale = "0.3.0"
        self.write_workspace(version, version, version, version)
        self.write_release_components(version)

        cases = (
            (
                "java",
                self.root / "java/pom.xml",
                f"<version>{version}-SNAPSHOT</version>",
                f"<version>{stale}-SNAPSHOT</version>",
                f"java/pom.xml: expected {version}-SNAPSHOT, "
                f"found {stale}-SNAPSHOT",
            ),
            (
                "rust",
                self.root / "core/Cargo.toml",
                f'version = "{version}"',
                f'version = "{stale}"',
                f"core/Cargo.toml: expected {version}, found {stale}",
            ),
            (
                "python",
                self.root / "python/pyproject.toml",
                f'version = "{version}"',
                f'version = "{stale}"',
                f"python/pyproject.toml: expected {version}, found {stale}",
            ),
            (
                "lock",
                self.root / "Cargo.lock",
                f'name = "paimon-mosaic-core"\nversion = "{version}"',
                f'name = "paimon-mosaic-core"\nversion = "{stale}"',
                f"Cargo.lock paimon-mosaic-core: expected {version}, found {stale}",
            ),
        )

        with mock.patch.object(
            verify_release_versions,
            "path_dependency_failures",
            return_value=[],
        ):
            self.assertEqual(
                verify_release_versions.verify(
                    version, java_snapshot=True, root=self.root
                ),
                [],
            )
            for name, path, current, replacement, expected in cases:
                with self.subTest(component=name):
                    original = path.read_text(encoding="utf-8")
                    self.assertIn(current, original)
                    path.write_text(
                        original.replace(current, replacement, 1),
                        encoding="utf-8",
                    )
                    try:
                        self.assertEqual(
                            verify_release_versions.verify(
                                version, java_snapshot=True, root=self.root
                            ),
                            [expected],
                        )
                    finally:
                        path.write_text(original, encoding="utf-8")

            pom = self.root / "java/pom.xml"
            pom.write_text(
                pom.read_text(encoding="utf-8").replace(
                    f"{version}-SNAPSHOT", version
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                verify_release_versions.verify(
                    version, java_snapshot=False, root=self.root
                ),
                [],
            )

    def test_verifier_accumulates_multiple_stale_release_components(self) -> None:
        version = "0.4.0"
        stale = "0.3.0"
        self.write_workspace(version, version, version, version)
        self.write_release_components(version)
        pom = self.root / "java/pom.xml"
        pom.write_text(
            pom.read_text(encoding="utf-8").replace(
                f"{version}-SNAPSHOT", f"{stale}-SNAPSHOT"
            ),
            encoding="utf-8",
        )
        pyproject = self.root / "python/pyproject.toml"
        pyproject.write_text(
            pyproject.read_text(encoding="utf-8").replace(version, stale),
            encoding="utf-8",
        )

        with mock.patch.object(
            verify_release_versions,
            "path_dependency_failures",
            return_value=[],
        ):
            self.assertEqual(
                verify_release_versions.verify(
                    version, java_snapshot=True, root=self.root
                ),
                [
                    f"java/pom.xml: expected {version}-SNAPSHOT, "
                    f"found {stale}-SNAPSHOT",
                    f"python/pyproject.toml: expected {version}, found {stale}",
                ],
            )

    def test_main_returns_success_and_failure_status(self) -> None:
        version = "0.4.0"
        with mock.patch.object(
            verify_release_versions, "verify", return_value=[]
        ) as verify:
            with mock.patch.object(
                sys,
                "argv",
                ["verify_release_versions.py", version, "--java-snapshot"],
            ):
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(verify_release_versions.main(), 0)
            verify.assert_called_once_with(version, True)

        with mock.patch.object(
            verify_release_versions,
            "verify",
            return_value=["python/pyproject.toml is stale"],
        ) as verify:
            with mock.patch.object(
                sys,
                "argv",
                ["verify_release_versions.py", version, "--java-snapshot"],
            ):
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(verify_release_versions.main(), 1)
            verify.assert_called_once_with(version, True)

    def test_main_routes_cargo_updates_and_returns_failure_status(self) -> None:
        arguments = [
            "verify_release_versions.py",
            "--update-cargo",
            "0.3.0",
            "0.4.0",
        ]
        updated_manifest = verify_release_versions.ROOT / "core/Cargo.toml"
        with mock.patch.object(
            verify_release_versions,
            "update_cargo_versions",
            return_value=[updated_manifest],
        ) as update:
            with mock.patch.object(sys, "argv", arguments):
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(verify_release_versions.main(), 0)
            update.assert_called_once_with(
                verify_release_versions.ROOT, "0.3.0", "0.4.0"
            )

        with mock.patch.object(
            verify_release_versions,
            "update_cargo_versions",
            side_effect=ValueError("stale manifest"),
        ) as update:
            with mock.patch.object(sys, "argv", arguments):
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(verify_release_versions.main(), 1)
            update.assert_called_once_with(
                verify_release_versions.ROOT, "0.3.0", "0.4.0"
            )

    def test_uses_cargo_version_requirement_semantics(self) -> None:
        self.assertTrue(
            verify_release_versions.cargo_requirement_accepts(
                ">=0.3.0, <0.5.0", "0.4.0"
            )
        )
        self.assertFalse(
            verify_release_versions.cargo_requirement_accepts("0.3.0", "0.4.0")
        )


if __name__ == "__main__":
    unittest.main()
