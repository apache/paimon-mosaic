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

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))

import bump_pom_version as bumper  # noqa: E402


NAMESPACE = 'xmlns="http://maven.apache.org/POM/4.0.0"'


def write_pom(tmp_path: Path, body: str) -> Path:
    pom = tmp_path / "pom.xml"
    pom.write_text(f"<project {NAMESPACE}>\n{body}\n</project>\n", encoding="utf-8")
    return pom


def test_dependency_sharing_the_project_version_is_left_alone(tmp_path):
    pom = write_pom(
        tmp_path,
        """    <artifactId>mosaic</artifactId>
    <version>0.3.0</version>
    <dependencies>
        <dependency>
            <artifactId>unrelated</artifactId>
            <version>0.3.0</version>
        </dependency>
    </dependencies>""",
    )

    bumper.bump_pom_version(pom, "0.3.0", "0.4.0-SNAPSHOT")

    assert "<version>0.4.0-SNAPSHOT</version>" in pom.read_text(encoding="utf-8")
    assert pom.read_text(encoding="utf-8").count("<version>0.3.0</version>") == 1


def test_plugin_sharing_the_project_version_is_left_alone(tmp_path):
    pom = write_pom(
        tmp_path,
        """    <version>0.3.0</version>
    <build>
        <plugins>
            <plugin>
                <artifactId>some-plugin</artifactId>
                <version>0.3.0</version>
            </plugin>
        </plugins>
    </build>""",
    )

    bumper.bump_pom_version(pom, "0.3.0", "0.4.0")

    text = pom.read_text(encoding="utf-8")
    assert "<version>0.4.0</version>" in text
    assert text.count("<version>0.3.0</version>") == 1


def test_apache_parent_version_is_not_bumped(tmp_path):
    pom = write_pom(
        tmp_path,
        """    <parent>
        <artifactId>apache</artifactId>
        <version>23</version>
    </parent>
    <version>0.3.0-SNAPSHOT</version>""",
    )

    bumper.bump_pom_version(pom, "0.3.0-SNAPSHOT", "0.3.0")

    text = pom.read_text(encoding="utf-8")
    assert "<version>23</version>" in text
    assert "<version>0.3.0</version>" in text


def test_module_parent_version_is_bumped_with_the_project(tmp_path):
    pom = write_pom(
        tmp_path,
        """    <parent>
        <artifactId>mosaic</artifactId>
        <version>0.3.0-SNAPSHOT</version>
    </parent>
    <artifactId>mosaic-child</artifactId>""",
    )

    bumper.bump_pom_version(pom, "0.3.0-SNAPSHOT", "0.3.0")

    assert "<version>0.3.0</version>" in pom.read_text(encoding="utf-8")


def test_surrounding_formatting_is_preserved(tmp_path):
    pom = write_pom(
        tmp_path,
        """    <!-- keep this comment -->
    <version>0.3.0</version>
    <properties>
        <arrow.version>18.0.0</arrow.version>
    </properties>""",
    )
    before = pom.read_text(encoding="utf-8")

    bumper.bump_pom_version(pom, "0.3.0", "0.4.0")

    after = pom.read_text(encoding="utf-8")
    assert after == before.replace(
        "<version>0.3.0</version>", "<version>0.4.0</version>"
    )


def test_missing_old_version_fails_closed(tmp_path):
    pom = write_pom(tmp_path, "    <version>9.9.9</version>")

    with pytest.raises(bumper.PomBumpError, match="no project or parent version"):
        bumper.bump_pom_version(pom, "0.3.0", "0.4.0")


def test_malformed_pom_fails_closed(tmp_path):
    pom = tmp_path / "pom.xml"
    pom.write_text("<project><version>0.3.0</version>", encoding="utf-8")

    with pytest.raises(bumper.PomBumpError, match="not well-formed"):
        bumper.bump_pom_version(pom, "0.3.0", "0.4.0")
