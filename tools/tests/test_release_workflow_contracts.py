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

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def workflow(name: str) -> str:
    return (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")


def job(workflow_text: str, name: str, next_name: str | None) -> str:
    start = workflow_text.index(f"  {name}:")
    if next_name is None:
        return workflow_text[start:]
    end = workflow_text.index(f"  {next_name}:", start)
    return workflow_text[start:end]


def test_manual_release_dispatch_is_build_only():
    release = workflow("release.yml")

    rc_publish = job(release, "python-rc-publish", "final-publication-preflight")
    final_preflight = job(
        release, "final-publication-preflight", "rust-final-publish"
    )
    assert "github.event_name == 'push'" in rc_publish
    assert "github.event_name == 'push'" in final_preflight
    assert "if: github.event_name == 'workflow_dispatch'" in release

    python_publish = workflow("release-python-publish.yml")
    publish_job = job(python_publish, "publish", None)
    assert "github.event_name == 'push'" in publish_job


def test_testpypi_publication_stages_only_missing_verified_wheels():
    python_publish = workflow("release-python-publish.yml")

    assert "id: test_registry" in python_publish
    assert "--upload-directory dist-testpypi" in python_publish
    assert "steps.test_registry.outputs.publish == 'true'" in python_publish
    assert "packages-dir: dist-testpypi" in python_publish
    assert "Require an unused TestPyPI RC version" not in python_publish
