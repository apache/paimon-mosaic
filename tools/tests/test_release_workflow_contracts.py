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

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def workflow(name: str) -> str:
    return (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")


def jobs(workflow_text: str) -> dict[str, str]:
    jobs_start = workflow_text.index("\njobs:\n") + len("\njobs:\n")
    jobs_text = workflow_text[jobs_start:]
    matches = list(re.finditer(r"^  ([A-Za-z0-9_-]+):\s*$", jobs_text, re.MULTILINE))
    return {
        match.group(1): jobs_text[
            match.start() : matches[index + 1].start()
            if index + 1 < len(matches)
            else len(jobs_text)
        ]
        for index, match in enumerate(matches)
    }


def job(workflow_text: str, name: str, next_name: str | None) -> str:
    del next_name
    return jobs(workflow_text)[name]


def step(workflow_text: str, name: str) -> str:
    start = workflow_text.index(f"      - name: {name}")
    end = workflow_text.find("\n      - ", start + 1)
    return workflow_text[start:] if end == -1 else workflow_text[start:end]


def field(block: str, name: str, indent: int) -> str | None:
    matches = re.findall(
        rf"^{' ' * indent}{re.escape(name)}:\s*(.*?)\s*$",
        block,
        re.MULTILINE,
    )
    assert len(matches) <= 1, f"duplicate {name!r} fields"
    return matches[0] if matches else None


def sequence_field(block: str, name: str, indent: int) -> tuple[str, ...]:
    lines = block.splitlines()
    prefix = f"{' ' * indent}{name}:"
    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        if value:
            if value.startswith("["):
                assert value.endswith("]"), f"invalid inline sequence for {name}"
                value = value[1:-1]
                return tuple(
                    item.strip().strip("'\"")
                    for item in value.split(",")
                    if item.strip()
                )
            return (value.strip("'\""),)

        item_prefix = f"{' ' * (indent + 2)}- "
        items = []
        for continuation in lines[index + 1 :]:
            if not continuation.startswith(item_prefix):
                break
            items.append(continuation[len(item_prefix) :].strip().strip("'\""))
        return tuple(items)
    return ()


def condition_terms(condition: str | None) -> set[str]:
    assert condition is not None
    condition = condition.strip()
    if condition.startswith("${{") and condition.endswith("}}"):
        condition = condition[3:-2].strip()
    return {term.strip() for term in condition.split("&&")}


def dependency_ancestors(workflow_jobs: dict[str, str], name: str) -> set[str]:
    pending = list(sequence_field(workflow_jobs[name], "needs", 4))
    ancestors = set()
    while pending:
        dependency = pending.pop()
        if dependency in ancestors:
            continue
        ancestors.add(dependency)
        if dependency in workflow_jobs:
            pending.extend(
                sequence_field(workflow_jobs[dependency], "needs", 4)
            )
    return ancestors


def assert_final_publish_jobs_require_preflight(release: str) -> None:
    release_jobs = jobs(release)
    final_preflight = release_jobs["final-publication-preflight"]
    continue_on_error = field(final_preflight, "continue-on-error", 4)
    assert continue_on_error is None or continue_on_error.lower() == "false"
    assert sequence_field(
        release_jobs["rust-final-publish"], "needs", 4
    ) == ("final-publication-preflight",)
    for final_job in ("rust-final-publish", "python-final-publish"):
        assert field(release_jobs[final_job], "if", 4) is None
        assert "final-publication-preflight" in dependency_ancestors(
            release_jobs, final_job
        )


def assert_rust_leaf_publish_gate(rust: str) -> None:
    publish_step = step(
        jobs(rust)["verify"], "Publish paimon-mosaic-core to crates.io"
    )
    allowed_terms = {
        "inputs.publish == true",
        "github.event_name == 'push'",
        "github.repository == 'apache/paimon-mosaic'",
        "startsWith(github.ref, 'refs/tags/')",
        "!contains(github.ref_name, '-')",
        "steps.registry.outputs.publish == 'true'",
    }
    assert condition_terms(field(publish_step, "if", 8)) == allowed_terms


def assert_promoted_source_archive_gate(release: str) -> None:
    final_preflight = jobs(release)["final-publication-preflight"]
    source_archive_step = step(
        final_preflight, "Require promoted and valid ASF source release"
    )
    assert field(source_archive_step, "if", 8) is None
    continue_on_error = field(source_archive_step, "continue-on-error", 8)
    assert continue_on_error is None or continue_on_error.lower() == "false"
    assert_final_publish_jobs_require_preflight(release)


def replace_block(contents: str, original: str, replacement: str) -> str:
    assert contents.count(original) == 1
    return contents.replace(original, replacement, 1)


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
    release_jobs = jobs(release)

    assert "github.event_name == 'push'" in condition_terms(
        field(release_jobs["python-rc-publish"], "if", 4)
    )
    assert "github.event_name == 'push'" in condition_terms(
        field(release_jobs["final-publication-preflight"], "if", 4)
    )
    manual_dispatch_step = step(
        release_jobs["tag-validation"], "Confirm manual dispatch is build-only"
    )
    assert condition_terms(field(manual_dispatch_step, "if", 8)) == {
        "github.event_name == 'workflow_dispatch'"
    }
    assert_final_publish_jobs_require_preflight(release)

    python_publish = workflow("release-python-publish.yml")
    assert "github.event_name == 'push'" in condition_terms(
        field(jobs(python_publish)["publish"], "if", 4)
    )

    rust = workflow("release-rust.yml")
    assert_rust_leaf_publish_gate(rust)


def test_manual_release_contract_rejects_publication_bypass_mutations():
    release = workflow("release.yml")
    rust_publish = jobs(release)["rust-final-publish"]
    mutated_rust_publish = re.sub(
        r"^    needs:.*$",
        "    needs: [rust]",
        rust_publish,
        count=1,
        flags=re.MULTILINE,
    )
    assert mutated_rust_publish != rust_publish
    with pytest.raises(AssertionError):
        assert_final_publish_jobs_require_preflight(
            replace_block(release, rust_publish, mutated_rust_publish)
        )

    final_preflight = jobs(release)["final-publication-preflight"]
    mutated_final_preflight = final_preflight.replace(
        "\n", "\n    continue-on-error: true\n", 1
    )
    assert mutated_final_preflight != final_preflight
    with pytest.raises(AssertionError):
        assert_final_publish_jobs_require_preflight(
            replace_block(release, final_preflight, mutated_final_preflight)
        )

    python_publish = jobs(release)["python-final-publish"]
    mutated_python_publish = python_publish.replace(
        "\n", "\n    if: always()\n", 1
    )
    assert mutated_python_publish != python_publish
    with pytest.raises(AssertionError):
        assert_final_publish_jobs_require_preflight(
            replace_block(release, python_publish, mutated_python_publish)
        )

    rust = workflow("release-rust.yml")
    publish_step = step(
        jobs(rust)["verify"], "Publish paimon-mosaic-core to crates.io"
    )
    mutated_publish_step = publish_step.replace(
        "github.event_name == 'push' && ", "", 1
    )
    assert mutated_publish_step != publish_step
    with pytest.raises(AssertionError):
        assert_rust_leaf_publish_gate(
            replace_block(rust, publish_step, mutated_publish_step)
        )

    contradictory_publish_step = publish_step.replace(
        "steps.registry.outputs.publish == 'true'",
        "steps.registry.outputs.publish == 'true' "
        "&& github.event_name == 'workflow_dispatch'",
        1,
    )
    assert contradictory_publish_step != publish_step
    with pytest.raises(AssertionError):
        assert_rust_leaf_publish_gate(
            replace_block(rust, publish_step, contradictory_publish_step)
        )


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
    final_preflight = jobs(release)["final-publication-preflight"]
    source_archive_step = step(
        final_preflight, "Require promoted and valid ASF source release"
    )

    assert "python3 tools/verify_source_archive.py verify \\" in source_archive_step
    assert "--repository . \\" in source_archive_step
    assert '--commit "$GITHUB_SHA" \\' in source_archive_step
    assert '--prefix "paimon-mosaic-${version}/" \\' in source_archive_step
    assert '--archive "$release_dir/$archive"' in source_archive_step
    assert_promoted_source_archive_gate(release)


def test_promoted_source_archive_gate_rejects_skippable_mutations():
    release = workflow("release.yml")
    final_preflight = jobs(release)["final-publication-preflight"]
    source_archive_step = step(
        final_preflight, "Require promoted and valid ASF source release"
    )

    for bypass in ("        continue-on-error: true\n", "        if: false\n"):
        mutated_step = source_archive_step.replace(
            "\n", f"\n{bypass}", 1
        )
        with pytest.raises(AssertionError):
            assert_promoted_source_archive_gate(
                replace_block(release, source_archive_step, mutated_step)
            )


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
