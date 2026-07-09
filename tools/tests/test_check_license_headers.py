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

import check_license_headers as checker  # noqa: E402


HEADER = """# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.
"""


def test_checks_extracted_source_tree_without_git(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/good.py").write_text(HEADER, encoding="utf-8")
    (tmp_path / "src/missing.py").write_text("print('missing')\n", encoding="utf-8")
    (tmp_path / "target").mkdir()
    (tmp_path / "target/generated.rs").write_text("generated\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("license\n", encoding="utf-8")

    assert checker.tracked_files(tmp_path) == [
        "LICENSE",
        "src/good.py",
        "src/missing.py",
    ]
    assert checker.missing_headers(tmp_path) == ["src/missing.py"]


def test_repo_root_is_bound_to_the_script_location() -> None:
    assert checker.repo_root() == TOOLS_DIRECTORY.parent.resolve()
