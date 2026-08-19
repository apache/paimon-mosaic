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


def step(workflow_text: str, name: str) -> str:
    start = workflow_text.index(f"      - name: {name}")
    end = workflow_text.find("\n      - ", start + 1)
    return workflow_text[start:] if end == -1 else workflow_text[start:end]


def curl_commands(workflow_text: str) -> list[str]:
    lines = workflow_text.splitlines()
    commands = []
    for index, line in enumerate(lines):
        if not line.strip().startswith("curl"):
            continue
        if line.strip() != "curl \\":
            raise AssertionError(f"unsupported curl command form: {line}")
        command = [line]
        for continuation in lines[index + 1 :]:
            command.append(continuation)
            if not continuation.rstrip().endswith("\\"):
                break
        commands.append("\n".join(command))
    return commands


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


def test_registry_secrets_are_scoped_to_publish_workflows():
    release = workflow("release.yml")
    final_preflight = job(
        release, "final-publication-preflight", "rust-final-publish"
    )
    rust_verify = job(release, "rust", "java")
    python_wheels = job(release, "python-wheels", "python-rc-publish")
    rc_publish = job(release, "python-rc-publish", "final-publication-preflight")
    rust_publish = job(release, "rust-final-publish", "python-final-publish")
    python_publish = job(release, "python-final-publish", None)

    assert "CARGO_REGISTRY_TOKEN" not in final_preflight
    assert "PYPI_API_TOKEN" not in final_preflight
    assert "TEST_PYPI_API_TOKEN" not in job(
        release, "tag-validation", "preflight"
    )
    assert "secrets: inherit" not in release
    assert "secrets:" not in rust_verify
    assert "secrets:" not in python_wheels
    registry_secrets = {
        "TEST_PYPI_API_TOKEN": "secrets.TEST_PYPI_API_TOKEN",
        "CARGO_REGISTRY_TOKEN": "secrets.CARGO_REGISTRY_TOKEN",
        "PYPI_API_TOKEN": "secrets.PYPI_API_TOKEN",
    }
    publish_jobs = {
        "TEST_PYPI_API_TOKEN": rc_publish,
        "CARGO_REGISTRY_TOKEN": rust_publish,
        "PYPI_API_TOKEN": python_publish,
    }
    for expected_name, publish_job in publish_jobs.items():
        expected_value = registry_secrets[expected_name]
        assert f"{expected_name}: ${{{{ {expected_value} }}}}" in publish_job
        for other_name, other_value in registry_secrets.items():
            if other_name != expected_name:
                assert other_value not in publish_job

    rust_workflow = workflow("release-rust.yml")
    rust_publish_step = step(rust_workflow, "Publish paimon-mosaic-core to crates.io")
    assert rust_workflow.count("secrets.CARGO_REGISTRY_TOKEN") == 1
    assert "CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_REGISTRY_TOKEN }}" in (
        rust_publish_step
    )
    assert "secrets.CARGO_REGISTRY_TOKEN" not in rust_workflow.replace(
        rust_publish_step, ""
    )

    python_workflow = workflow("release-python-publish.yml")
    test_registry = python_workflow[
        python_workflow.index("      - name: Verify TestPyPI RC artifact state") :
        python_workflow.index("      - name: Verify final PyPI artifact state")
    ]
    assert "TEST_PYPI_API_TOKEN" not in test_registry
    test_publish_step = step(python_workflow, "Publish to TestPyPI")
    final_publish_step = step(python_workflow, "Publish to PyPI")
    assert python_workflow.count("secrets.TEST_PYPI_API_TOKEN") == 1
    assert python_workflow.count("secrets.PYPI_API_TOKEN") == 1
    assert "password: ${{ secrets.TEST_PYPI_API_TOKEN }}" in test_publish_step
    assert "password: ${{ secrets.PYPI_API_TOKEN }}" in final_publish_step
    non_publish_steps = python_workflow.replace(test_publish_step, "").replace(
        final_publish_step, ""
    )
    assert "secrets.TEST_PYPI_API_TOKEN" not in non_publish_steps
    assert "secrets.PYPI_API_TOKEN" not in non_publish_steps


def test_final_publication_preflight_binds_source_to_final_tag_commit():
    release = workflow("release.yml")
    final_preflight = job(
        release, "final-publication-preflight", "rust-final-publish"
    )

    assert "python3 tools/verify_source_archive.py verify \\" in final_preflight
    assert "--repository . \\" in final_preflight
    assert '--commit "$GITHUB_SHA" \\' in final_preflight
    assert '--prefix "paimon-mosaic-${version}/" \\' in final_preflight
    assert '--archive "$release_dir/$archive"' in final_preflight


def test_release_network_calls_have_bounded_timeouts():
    release = workflow("release.yml")
    assert "timeout-minutes: 30" in job(release, "tag-validation", "preflight")
    assert "timeout-minutes: 30" in job(
        release, "final-publication-preflight", "rust-final-publish"
    )

    rust = workflow("release-rust.yml")
    assert "timeout-minutes: 30" in job(rust, "verify", None)

    python_publish = workflow("release-python-publish.yml")
    assert "timeout-minutes: 30" in job(python_publish, "publish", None)

    python_wheels = workflow("release-python.yml")
    assert "timeout-minutes: 60" in job(
        python_wheels, "wheels-linux", "wheels-macos"
    )

    ci = workflow("ci.yml")
    assert "timeout-minutes: 30" in job(ci, "cpp-test", "java-test")

    workflows_with_curl = {}
    for path in sorted((ROOT / ".github/workflows").glob("*.y*ml")):
        contents = path.read_text(encoding="utf-8")
        commands = curl_commands(contents)
        if commands:
            workflows_with_curl[path.name] = commands

    assert set(workflows_with_curl) == {
        "ci.yml",
        "release-python-publish.yml",
        "release-python.yml",
        "release-rust.yml",
        "release.yml",
    }
    for commands in workflows_with_curl.values():
        for command in commands:
            assert "--connect-timeout 10 \\" in command, command
            assert "--max-time 300 \\" in command, command
            assert "--retry 3 \\" in command, command
            assert "--retry-connrefused \\" in command, command


def test_crates_publish_does_not_rebuild_with_registry_credentials():
    rust_workflow = workflow("release-rust.yml")
    publish_step = rust_workflow[
        rust_workflow.index(
            "      - name: Publish paimon-mosaic-core to crates.io"
        ) :
    ]

    assert "cargo publish" in publish_step
    assert "--no-verify" in publish_step


def test_snapshot_publication_cannot_run_branch_controlled_code_with_secrets():
    snapshot = workflow("publish_snapshot.yml")
    publish_job = job(snapshot, "publish-snapshot", None)

    assert "workflow_dispatch:" not in snapshot
    assert "repository_dispatch:" in snapshot
    assert "types: [publish-snapshot]" in snapshot
    assert "github.ref == 'refs/heads/main'" in publish_job
    assert "permissions:\n  contents: read" in snapshot
    assert "persist-credentials: false" in publish_job
    assert "github.run_id" not in snapshot
    assert "cancel-in-progress: false" in snapshot


def test_local_java_staging_script_runs_on_linux_and_macos():
    ci = workflow("ci.yml")
    staging_job = job(ci, "java-staging-script", "rust-test")

    assert "ubuntu-latest" in staging_job
    assert "macos-latest" in staging_job
    assert "/bin/bash -n tools/deploy_java_staging.sh" in staging_job
    assert "/bin/bash tools/tests/deploy_java_staging_test.sh" in staging_job


def test_release_guide_uses_fail_closed_java_staging_script():
    guide = (ROOT / "docs/creating-a-release.html").read_text(encoding="utf-8")
    section = guide[
        guide.index("<h3>Sign and Stage Java Artifacts Locally</h3>") :
        guide.index("<h3>Create Source Release Artifacts</h3>")
    ]

    assert "./tools/deploy_java_staging.sh" in section
    assert "--dry-run" in section
    assert "gh run view" not in section
    assert "mvn clean deploy" not in section
    tools_readme = (ROOT / "tools/README.md").read_text(encoding="utf-8")
    assert "deploy_java_staging.sh" in tools_readme
    assert "java-release-native-inputs" in tools_readme


def test_release_builds_use_the_exact_pinned_rust_toolchain():
    toolchain = (ROOT / "rust-toolchain.toml").read_text(encoding="utf-8")
    assert 'channel = "1.97.1"' in toolchain
    assert 'profile = "minimal"' in toolchain

    for name in (
        "ci.yml",
        "publish_snapshot.yml",
        "release-java.yml",
        "release-python.yml",
        "release-rust.yml",
        "release.yml",
    ):
        contents = workflow(name)
        assert "rustup update stable" not in contents
        assert "rustup default stable" not in contents

    python_release = workflow("release-python.yml")
    assert "--default-toolchain none" in python_release
