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

from pathlib import Path
import sys


TOOLS_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIRECTORY))

import generate_license_reports as generator  # noqa: E402


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
