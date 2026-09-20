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

import copy
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
RELEASE_WORKFLOW = ROOT / ".github/workflows/release.yml"
RELEASE_PREFLIGHT_WORKFLOW = ROOT / ".github/workflows/release-preflight.yml"
GATE_WORKFLOW = ROOT / ".github/workflows/release-vote-gate.yml"
RUST_RELEASE_WORKFLOW = ROOT / ".github/workflows/release-rust.yml"
JAVA_RELEASE_WORKFLOW = ROOT / ".github/workflows/release-java.yml"
PYTHON_WHEELS_WORKFLOW = ROOT / ".github/workflows/release-python.yml"
PYTHON_PUBLISH_WORKFLOW = ROOT / ".github/workflows/release-python-publish.yml"
RELEASE_DOCUMENTATION = ROOT / "docs/creating-a-release.html"
CREDENTIALED_RELEASE_WORKFLOWS = (
    (RUST_RELEASE_WORKFLOW, "publish"),
    (PYTHON_PUBLISH_WORKFLOW, "publish"),
)
TAG_CONDITION = "startsWith(github.ref, 'refs/tags/')"
RUST_PUBLISH_CONDITION = (
    "github.event_name != 'workflow_dispatch' && "
    "github.repository == 'apache/paimon-mosaic' && "
    "startsWith(github.ref, 'refs/tags/') && "
    "!contains(github.ref_name, '-')"
)
JAVA_PACKAGE_CONDITION = (
    "github.event_name != 'workflow_dispatch' && "
    "github.repository == 'apache/paimon-mosaic' && "
    "startsWith(github.ref, 'refs/tags/') && "
    "contains(github.ref_name, '-rc')"
)
PYTHON_TEST_PUBLISH_CONDITION = (
    "github.event_name != 'workflow_dispatch' && "
    "contains(github.ref_name, '-rc')"
)
PYTHON_PUBLISH_CONDITION = (
    "github.event_name != 'workflow_dispatch' && "
    "!contains(github.ref_name, '-')"
)
REQUIRED_GATE_PATHS = {
    ".gitattributes",
    ".github/workflows/**",
    "docs/creating-a-release.html",
    "docs/verifying-a-release-candidate.html",
    "java/pom.xml",
    "java/src/test/java/org/apache/paimon/mosaic/MosaicNativeLoaderSmokeTest.java",
    "python/pyproject.toml",
    "python/setup.py",
    "tools/create_source_release.sh",
    "tools/deploy_java_staging.sh",
    "tools/update_branch_version.sh",
    "tools/verify_release_artifacts.py",
    "tools/verify_release_versions.py",
    "tools/verify_source_archive.py",
    "tools/tests/test_create_source_release.py",
    "tools/tests/deploy_java_staging_test.sh",
    "tools/tests/test_python_setup.py",
    "tools/tests/test_release_vote_workflow.py",
    "tools/tests/test_update_branch_version.py",
    "tools/tests/test_verify_release_artifacts.py",
    "tools/tests/test_verify_release_versions.py",
    "tools/tests/test_verify_source_archive.py",
}
RELEASE_WORKFLOW_BY_JOB = {
    "rust": "./.github/workflows/release-rust.yml",
    "java": "./.github/workflows/release-java.yml",
    "python-wheels": "./.github/workflows/release-python.yml",
    "python-publish": "./.github/workflows/release-python-publish.yml",
}
IMPORT_VERIFICATION_KEYS_COMMAND = """set -euo pipefail
keys_file="${RUNNER_TEMP}/KEYS"
curl --fail --location --proto '=https' --tlsv1.2 \\
  --output "${keys_file}" \\
  https://downloads.apache.org/paimon/KEYS
gpg --batch --import "${keys_file}"
"""
VERIFY_RELEASE_COMMAND = (
    'python3 tools/verify_release_versions.py "${TAG_NAME}" '
    "--verify-signature"
)
VERIFY_SOURCE_ARCHIVE_COMMAND = """set -euo pipefail
release_version="${GITHUB_REF_NAME#v}"
release_version="${release_version%-rc*}"
archive="${RUNNER_TEMP}/apache-paimon-mosaic-${release_version}-src.tgz"
prefix="paimon-mosaic-${release_version}/"
commit="$(git rev-parse --verify 'HEAD^{commit}')"
python3 tools/verify_source_archive.py create \\
  --repository "${GITHUB_WORKSPACE}" \\
  --commit "${commit}" \\
  --prefix "${prefix}" \\
  --output "${archive}"
python3 tools/verify_source_archive.py verify \\
  --repository "${GITHUB_WORKSPACE}" \\
  --commit "${commit}" \\
  --prefix "${prefix}" \\
  --archive "${archive}"
"""
GATE_TEST_COMMAND = """python -m pytest -q \\
  tools/tests/test_create_source_release.py \\
  tools/tests/test_python_setup.py \\
  tools/tests/test_release_vote_workflow.py \\
  tools/tests/test_update_branch_version.py \\
  tools/tests/test_verify_release_artifacts.py \\
  tools/tests/test_verify_release_versions.py \\
  tools/tests/test_verify_source_archive.py
"""
GATE_SOURCE_TREE_COMMAND = """set -euo pipefail
output_dir="$(mktemp -d "${RUNNER_TEMP}/source-tree.XXXXXX")"
archive="${output_dir}/source.tgz"
commit="$(git rev-parse --verify 'HEAD^{commit}')"
prefix="paimon-mosaic-source-tree/"
python3 tools/verify_source_archive.py create \\
  --repository "${GITHUB_WORKSPACE}" \\
  --commit "${commit}" \\
  --prefix "${prefix}" \\
  --output "${archive}"
python3 tools/verify_source_archive.py verify \\
  --repository "${GITHUB_WORKSPACE}" \\
  --commit "${commit}" \\
  --prefix "${prefix}" \\
  --archive "${archive}"
"""
GATE_STATIC_COMMAND = """set -euo pipefail
python -m compileall -q \\
  tools/verify_release_artifacts.py \\
  tools/verify_release_versions.py \\
  tools/verify_source_archive.py \\
  tools/tests/test_create_source_release.py \\
  tools/tests/test_python_setup.py \\
  tools/tests/test_release_vote_workflow.py \\
  tools/tests/test_update_branch_version.py \\
  tools/tests/test_verify_release_artifacts.py \\
  tools/tests/test_verify_release_versions.py \\
  tools/tests/test_verify_source_archive.py
bash -n tools/create_source_release.sh
bash -n tools/deploy_java_staging.sh
bash -n tools/update_branch_version.sh
bash -n tools/tests/deploy_java_staging_test.sh
if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
  comparison_ref="origin/${GITHUB_BASE_REF}"
else
  comparison_ref="HEAD^"
fi
git diff --check "$(git merge-base HEAD "${comparison_ref}")" HEAD
"""


def load_workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def needs(job: dict) -> set[str]:
    value = job.get("needs", [])
    return {value} if isinstance(value, str) else set(value)


def release_version_step(workflow: dict) -> dict:
    steps = workflow["jobs"]["release-preflight"]["steps"]
    version_steps = [
        step
        for step in steps
        if "verify_release_versions.py" in step.get("run", "")
    ]
    assert len(version_steps) == 1
    return version_steps[0]


def named_step(workflow: dict, name: str) -> dict:
    steps = workflow["jobs"]["release-preflight"]["steps"]
    matches = [step for step in steps if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def gate_step(workflow: dict, name: str) -> dict:
    steps = workflow["jobs"]["release-vote-gate"]["steps"]
    matches = [step for step in steps if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def job_step(workflow: dict, job_name: str, name: str) -> dict:
    steps = workflow["jobs"][job_name]["steps"]
    matches = [step for step in steps if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def assert_gate_contract(workflow: dict) -> None:
    triggers = workflow["on"]
    assert "workflow_dispatch" in triggers
    assert set(triggers["pull_request"]["branches"]) == {"main", "release-*"}
    assert set(triggers["push"]["branches"]) == {"main", "release-*"}
    for event in ("pull_request", "push"):
        assert REQUIRED_GATE_PATHS <= set(triggers[event]["paths"])

    job = workflow["jobs"]["release-vote-gate"]
    assert "if" not in job
    assert "continue-on-error" not in job

    install_step = gate_step(workflow, "Install test dependencies")
    test_step = gate_step(workflow, "Run release vote tests")
    staging_step = gate_step(workflow, "Test local Java staging")
    source_tree_step = gate_step(workflow, "Verify current source tree")
    static_step = gate_step(workflow, "Run static checks")
    for step in (
        install_step,
        test_step,
        staging_step,
        source_tree_step,
        static_step,
    ):
        assert "if" not in step
        assert "continue-on-error" not in step

    assert install_step["run"] == (
        "python -m pip install pytest PyYAML setuptools wheel"
    )
    assert test_step["run"] == GATE_TEST_COMMAND
    assert staging_step["shell"] == "bash"
    assert staging_step["run"] == "bash tools/tests/deploy_java_staging_test.sh"
    assert source_tree_step["shell"] == "bash"
    assert source_tree_step["run"] == GATE_SOURCE_TREE_COMMAND
    assert static_step["shell"] == "bash"
    assert static_step["run"] == GATE_STATIC_COMMAND


def assert_release_contract(workflow: dict) -> None:
    assert workflow["concurrency"]["cancel-in-progress"] == "false"

    jobs = workflow["jobs"]
    preflight = jobs["release-preflight"]
    assert preflight["uses"] == "./.github/workflows/release-preflight.yml"
    assert "runs-on" not in preflight
    assert "continue-on-error" not in preflight

    for job_name in ("rust", "python-wheels", "python-publish"):
        assert jobs[job_name].get("secrets") == "inherit"
    assert "secrets" not in jobs["java"]

    for job_name in ("rust", "java", "python-wheels"):
        release_job = jobs[job_name]
        assert "release-preflight" in needs(release_job)
        assert "if" not in release_job
        assert "continue-on-error" not in release_job

    publish_job = jobs["python-publish"]
    assert needs(publish_job) == {"rust", "java", "python-wheels"}
    assert publish_job["if"] == TAG_CONDITION
    assert "continue-on-error" not in publish_job

    for job_name, reusable_workflow in RELEASE_WORKFLOW_BY_JOB.items():
        assert jobs[job_name]["uses"] == reusable_workflow

    source = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert "always()" not in source


def assert_release_preflight_contract(workflow: dict) -> None:
    assert "workflow_call" in workflow["on"]
    assert workflow["permissions"]["contents"] == "read"

    preflight = workflow["jobs"]["release-preflight"]
    assert "runs-on" in preflight
    assert "uses" not in preflight
    assert "continue-on-error" not in preflight
    assert "continue-on-error" not in release_version_step(workflow)

    checkout = next(
        step
        for step in preflight["steps"]
        if step.get("uses") == "actions/checkout@v6"
    )
    checkout_options = checkout.get("with", {})
    assert checkout_options.get("fetch-depth") == "0"
    assert "ref" not in checkout_options

    import_step = named_step(workflow, "Import release verification keys")
    assert import_step["if"] == TAG_CONDITION
    assert "env" not in import_step
    assert "continue-on-error" not in import_step
    assert import_step["run"] == IMPORT_VERIFICATION_KEYS_COMMAND

    version_step = release_version_step(workflow)
    assert version_step["if"] == TAG_CONDITION
    assert version_step["env"] == {
        "TAG_NAME": "${{ github.ref_name }}",
    }
    assert version_step["run"] == VERIFY_RELEASE_COMMAND

    source_step = named_step(workflow, "Verify source archive")
    assert source_step["if"] == TAG_CONDITION
    assert "continue-on-error" not in source_step
    assert source_step["run"] == VERIFY_SOURCE_ARCHIVE_COMMAND


def assert_leaf_release_contract(
    workflow: dict,
    required_job: str,
) -> None:
    assert "workflow_call" in workflow["on"]
    assert workflow["jobs"]["release-preflight"]["uses"] == (
        "./.github/workflows/release-preflight.yml"
    )
    for job_name, job in workflow["jobs"].items():
        if job_name != "release-preflight":
            assert "release-preflight" in needs(job)
    assert required_job in workflow["jobs"]


def test_release_has_signed_exact_tag_preflight() -> None:
    release = load_workflow(RELEASE_WORKFLOW)
    preflight = load_workflow(RELEASE_PREFLIGHT_WORKFLOW)
    assert_release_contract(release)
    assert_release_preflight_contract(preflight)

    triggers = release["on"]
    assert "workflow_dispatch" in triggers
    version_step = release_version_step(preflight)
    assert version_step["if"] == TAG_CONDITION
    assert "github.ref_name" in version_step["env"]["TAG_NAME"]
    assert "github.ref_name" not in version_step["run"]


def test_release_preflight_is_reusable_and_called_by_orchestrator() -> None:
    preflight = load_workflow(RELEASE_PREFLIGHT_WORKFLOW)
    release = load_workflow(RELEASE_WORKFLOW)

    assert "workflow_call" in preflight["on"]
    assert release["jobs"]["release-preflight"]["uses"] == (
        "./.github/workflows/release-preflight.yml"
    )


def test_release_preflight_does_not_use_private_key_secret() -> None:
    assert "GPG_SECRET_KEY" not in RELEASE_PREFLIGHT_WORKFLOW.read_text(
        encoding="utf-8"
    )


def test_release_tag_context_is_not_interpolated_into_shell() -> None:
    workflow = load_workflow(RELEASE_PREFLIGHT_WORKFLOW)
    version_step = release_version_step(workflow)

    assert version_step["env"] == {
        "TAG_NAME": "${{ github.ref_name }}",
    }
    assert version_step["run"] == VERIFY_RELEASE_COMMAND
    assert "${{ github.ref_name }}" not in version_step["run"]


def test_manual_rust_dispatch_cannot_publish() -> None:
    workflow = load_workflow(RUST_RELEASE_WORKFLOW)
    publish_steps = [
        step
        for step in workflow["jobs"]["publish"]["steps"]
        if step.get("name") == "Publish paimon-mosaic-core to crates.io"
    ]

    assert len(publish_steps) == 1
    assert publish_steps[0]["if"] == RUST_PUBLISH_CONDITION


def test_manual_java_dispatch_cannot_package_java() -> None:
    workflow = load_workflow(JAVA_RELEASE_WORKFLOW)
    package_job = workflow["jobs"]["package-java"]

    assert package_job["if"] == JAVA_PACKAGE_CONDITION


def test_manual_release_dispatch_cannot_publish_python() -> None:
    workflow = load_workflow(PYTHON_PUBLISH_WORKFLOW)
    steps = workflow["jobs"]["publish"]["steps"]
    test_publish = [
        step for step in steps if step.get("name") == "Publish to TestPyPI"
    ]
    final_publish = [
        step for step in steps if step.get("name") == "Publish to PyPI"
    ]

    assert len(test_publish) == 1
    assert len(final_publish) == 1
    assert test_publish[0]["if"] == PYTHON_TEST_PUBLISH_CONDITION
    assert final_publish[0]["if"] == PYTHON_PUBLISH_CONDITION


@pytest.mark.parametrize(
    ("workflow_path", "credentialed_job"),
    CREDENTIALED_RELEASE_WORKFLOWS,
)
def test_credentialed_release_jobs_require_reusable_preflight(
    workflow_path: Path,
    credentialed_job: str,
) -> None:
    workflow = load_workflow(workflow_path)

    assert_leaf_release_contract(workflow, credentialed_job)


def test_java_release_packages_and_smokes_unsigned_artifact() -> None:
    workflow = load_workflow(JAVA_RELEASE_WORKFLOW)
    assert workflow["jobs"]["release-preflight"]["uses"] == (
        "./.github/workflows/release-preflight.yml"
    )
    assert needs(workflow["jobs"]["build-native"]) == {"release-preflight"}

    package_job = workflow["jobs"]["package-java"]
    assert package_job["if"] == JAVA_PACKAGE_CONDITION
    assert needs(package_job) == {"release-preflight", "build-native"}
    package_step = job_step(workflow, "package-java", "Package Java artifacts")
    assert package_step["working-directory"] == "java"
    assert package_step["run"] == (
        "mvn clean verify -Prelease -Dgpg.skip=true -DskipTests"
    )

    verify_step = job_step(
        workflow,
        "package-java",
        "Verify multi-platform Java package",
    )
    assert "META-INF/DEPENDENCIES" in verify_step["run"]
    assert "THIRD-PARTY-LICENSES.html" in verify_step["run"]
    assert "verify_release_artifacts.py java" in verify_step["run"]
    assert "org/apache/paimon/mosaic/NativeLib.class" in verify_step["run"]
    assert "org.apache.paimon.mosaic.MosaicNativeLoaderSmokeTest" in (
        verify_step["run"]
    )

    upload_step = job_step(workflow, "package-java", "Upload Java package")
    assert upload_step["uses"] == "actions/upload-artifact@v5"
    assert upload_step["with"] == {
        "name": "java-package",
        "path": "java/target/*.jar",
        "if-no-files-found": "error",
    }

    smoke_job = workflow["jobs"]["smoke-java-package"]
    assert smoke_job["if"] == JAVA_PACKAGE_CONDITION
    assert needs(smoke_job) == {"release-preflight", "package-java"}
    matrix = smoke_job["strategy"]["matrix"]["include"]
    assert {
        (entry["os"], entry["java_version"])
        for entry in matrix
    } == {
        ("ubuntu-latest", "8"),
        ("ubuntu-24.04-arm", "8"),
        ("macos-latest", "17"),
        ("windows-latest", "8"),
    }
    smoke_step = job_step(
        workflow,
        "smoke-java-package",
        "Load the bundled JNI library from the final JAR",
    )
    assert "MosaicNativeLoaderSmokeTest.java" in smoke_step["run"]
    assert "javac -cp \"$jar_file\"" in smoke_step["run"]


def test_python_release_verifies_each_wheel_and_complete_publish_set() -> None:
    wheels = load_workflow(PYTHON_WHEELS_WORKFLOW)
    publish = load_workflow(PYTHON_PUBLISH_WORKFLOW)

    for job_name in ("wheels-linux", "wheels-macos", "wheels-windows"):
        verify_step = job_step(
            wheels,
            job_name,
            "Verify wheel legal payload",
        )
        assert "verify_release_artifacts.py python" in verify_step["run"]
        upload_step = next(
            step
            for step in wheels["jobs"][job_name]["steps"]
            if step.get("uses") == "actions/upload-artifact@v5"
        )
        assert upload_step["with"]["if-no-files-found"] == "error"

    checkout_steps = [
        step
        for step in publish["jobs"]["publish"]["steps"]
        if step.get("uses") == "actions/checkout@v6"
    ]
    assert len(checkout_steps) == 1
    verify_step = job_step(publish, "publish", "Verify wheel versions")
    assert "--require-all-python-targets" in verify_step["run"]
    assert "verify_release_artifacts.py" in verify_step["run"]


def test_java_release_never_receives_signing_or_nexus_credentials() -> None:
    release = load_workflow(RELEASE_WORKFLOW)
    assert "secrets" not in release["jobs"]["java"]

    source = JAVA_RELEASE_WORKFLOW.read_text(encoding="utf-8")
    for forbidden in (
        "secrets.",
        "GPG_SECRET_KEY",
        "GPG_PASSPHRASE",
        "NEXUS_STAGE_DEPLOYER_USER",
        "NEXUS_STAGE_DEPLOYER_PW",
    ):
        assert forbidden not in source


@pytest.mark.parametrize("job_name", ("rust", "java", "python-wheels"))
def test_contract_rejects_deleted_preflight_need(job_name: str) -> None:
    workflow = copy.deepcopy(load_workflow(RELEASE_WORKFLOW))
    workflow["jobs"][job_name].pop("needs")

    with pytest.raises(AssertionError):
        assert_release_contract(workflow)


@pytest.mark.parametrize("condition", ("${{ failure() }}", "${{ !cancelled() }}"))
def test_contract_rejects_release_job_status_override(condition: str) -> None:
    workflow = copy.deepcopy(load_workflow(RELEASE_WORKFLOW))
    workflow["jobs"]["rust"]["if"] = condition

    with pytest.raises(AssertionError):
        assert_release_contract(workflow)


@pytest.mark.parametrize(
    "mutation",
    ("status-override", "missing-dependency", "continue-on-error"),
)
def test_contract_rejects_python_publish_bypass(mutation: str) -> None:
    workflow = copy.deepcopy(load_workflow(RELEASE_WORKFLOW))
    publish_job = workflow["jobs"]["python-publish"]
    if mutation == "status-override":
        publish_job["if"] = "${{ failure() }}"
    elif mutation == "missing-dependency":
        publish_job["needs"] = ["rust"]
    elif mutation == "continue-on-error":
        publish_job["continue-on-error"] = "true"
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(AssertionError):
        assert_release_contract(workflow)


@pytest.mark.parametrize("job_name", RELEASE_WORKFLOW_BY_JOB)
def test_contract_rejects_wrong_reusable_workflow(job_name: str) -> None:
    workflow = copy.deepcopy(load_workflow(RELEASE_WORKFLOW))
    workflow["jobs"][job_name]["uses"] = "./.github/workflows/wrong.yml"

    with pytest.raises(AssertionError):
        assert_release_contract(workflow)


@pytest.mark.parametrize(
    "job_name",
    ("rust", "python-wheels", "python-publish"),
)
def test_contract_rejects_missing_secret_inheritance(job_name: str) -> None:
    workflow = copy.deepcopy(load_workflow(RELEASE_WORKFLOW))
    workflow["jobs"][job_name].pop("secrets")

    with pytest.raises(AssertionError):
        assert_release_contract(workflow)


def test_contract_rejects_java_secret_inheritance() -> None:
    workflow = copy.deepcopy(load_workflow(RELEASE_WORKFLOW))
    workflow["jobs"]["java"]["secrets"] = "inherit"

    with pytest.raises(AssertionError):
        assert_release_contract(workflow)


@pytest.mark.parametrize(
    ("workflow_path", "credentialed_job"),
    CREDENTIALED_RELEASE_WORKFLOWS,
)
@pytest.mark.parametrize(
    "mutation",
    ("missing-workflow-call", "missing-dependency", "unguarded-extra-job"),
)
def test_credentialed_release_contract_rejects_preflight_bypass(
    workflow_path: Path,
    credentialed_job: str,
    mutation: str,
) -> None:
    workflow = copy.deepcopy(load_workflow(workflow_path))
    if mutation == "missing-workflow-call":
        workflow["on"].pop("workflow_call")
    elif mutation == "missing-dependency":
        job = workflow["jobs"][credentialed_job]
        job["needs"] = [
            dependency
            for dependency in job.get("needs", [])
            if dependency != "release-preflight"
        ]
    elif mutation == "unguarded-extra-job":
        workflow["jobs"]["alternate-publish"] = {
            "runs-on": "ubuntu-latest",
            "steps": [{"run": "echo publish"}],
        }
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(AssertionError):
        assert_leaf_release_contract(workflow, credentialed_job)


def test_contract_rejects_masked_release_verifier_failure() -> None:
    workflow = copy.deepcopy(load_workflow(RELEASE_PREFLIGHT_WORKFLOW))
    release_version_step(workflow)["run"] += " || true"

    with pytest.raises(AssertionError):
        assert_release_preflight_contract(workflow)


@pytest.mark.parametrize(
    "mutation",
    ("missing-step", "masked-command", "step-continue", "wrong-commit"),
)
def test_contract_rejects_source_archive_preflight_mutations(
    mutation: str,
) -> None:
    workflow = copy.deepcopy(load_workflow(RELEASE_PREFLIGHT_WORKFLOW))
    if mutation == "missing-step":
        steps = workflow["jobs"]["release-preflight"]["steps"]
        steps.remove(named_step(workflow, "Verify source archive"))
    elif mutation == "masked-command":
        named_step(workflow, "Verify source archive")["run"] += " || true"
    elif mutation == "step-continue":
        named_step(workflow, "Verify source archive")[
            "continue-on-error"
        ] = "true"
    elif mutation == "wrong-commit":
        source_step = named_step(workflow, "Verify source archive")
        source_step["run"] = source_step["run"].replace(
            '--commit "${commit}"',
            "--commit HEAD^",
        )
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(AssertionError):
        assert_release_preflight_contract(workflow)


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong-checkout-ref",
        "missing-signature-verification",
        "masked-key-import",
        "key-import-continue",
        "private-key-secret",
    ),
)
def test_contract_rejects_signed_tag_preflight_mutations(mutation: str) -> None:
    workflow = copy.deepcopy(load_workflow(RELEASE_PREFLIGHT_WORKFLOW))
    if mutation == "wrong-checkout-ref":
        checkout = next(
            step
            for step in workflow["jobs"]["release-preflight"]["steps"]
            if step.get("uses") == "actions/checkout@v6"
        )
        checkout["with"]["ref"] = "main"
    elif mutation == "missing-signature-verification":
        release_version_step(workflow)["run"] = (
            'python3 tools/verify_release_versions.py "${TAG_NAME}"'
        )
    elif mutation == "masked-key-import":
        named_step(workflow, "Import release verification keys")[
            "run"
        ] += " || true"
    elif mutation == "key-import-continue":
        named_step(workflow, "Import release verification keys")[
            "continue-on-error"
        ] = "true"
    elif mutation == "private-key-secret":
        import_step = named_step(workflow, "Import release verification keys")
        import_step["env"] = {
            "GPG_SECRET_KEY": "${{ secrets.GPG_SECRET_KEY }}"
        }
        import_step["run"] = (
            "printf '%s' \"${GPG_SECRET_KEY}\" | gpg --batch --import"
        )
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(AssertionError):
        assert_release_preflight_contract(workflow)


@pytest.mark.parametrize(
    "mutation",
    ("cancel-in-progress", "preflight-continue", "version-step-continue"),
)
def test_contract_rejects_non_blocking_preflight_mutations(mutation: str) -> None:
    if mutation == "cancel-in-progress":
        workflow = copy.deepcopy(load_workflow(RELEASE_WORKFLOW))
        workflow["concurrency"]["cancel-in-progress"] = "true"
        contract = assert_release_contract
    elif mutation == "preflight-continue":
        workflow = copy.deepcopy(load_workflow(RELEASE_WORKFLOW))
        workflow["jobs"]["release-preflight"]["continue-on-error"] = "true"
        contract = assert_release_contract
    elif mutation == "version-step-continue":
        workflow = copy.deepcopy(load_workflow(RELEASE_PREFLIGHT_WORKFLOW))
        release_version_step(workflow)["continue-on-error"] = "true"
        contract = assert_release_preflight_contract
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(AssertionError):
        contract(workflow)


def test_source_release_documentation_passes_rc_tag_explicitly() -> None:
    source = RELEASE_DOCUMENTATION.read_text(encoding="utf-8")
    invocation = (
        "RELEASE_VERSION=${RELEASE_VERSION} "
        "RC_TAG=${RC_TAG} ./create_source_release.sh"
    )
    assert invocation in source

    assert "Cargo path dependency constraints and Cargo.lock" in source
    assert "Do not store Java signing material or Nexus credentials" in source


def test_release_documentation_describes_local_java_staging() -> None:
    source = RELEASE_DOCUMENTATION.read_text(encoding="utf-8")
    assert "--run-id ${RELEASE_RUN_ID} --dry-run" in source
    assert "--run-id ${RELEASE_RUN_ID}</code></pre>" in source
    assert "locally rebuilds and signs the Maven artifacts" in source
    assert "not directly uploaded to Nexus" in source


def test_release_documentation_describes_source_archive_preflight() -> None:
    source = RELEASE_DOCUMENTATION.read_text(encoding="utf-8")
    assert "temporary source archive" in source


def test_release_documentation_states_binding_vote_threshold() -> None:
    source = RELEASE_DOCUMENTATION.read_text(encoding="utf-8")
    assert "at least three binding <code>+1</code> votes" in source
    assert "ASF majority approval rule" in source


def test_release_documentation_preserves_previous_rc_output_before_retry() -> None:
    source = RELEASE_DOCUMENTATION.read_text(encoding="utf-8")
    retry_section = source.partition('<h2 id="fix-any-issues">')[2].partition(
        '<h2 id="finalize-the-release">'
    )[0]
    preserve_command = (
        'mv tools/release "../paimon-mosaic-${RC_TAG}-artifacts"'
    )

    assert preserve_command in retry_section
    assert retry_section.index(preserve_command) < retry_section.index(
        "Increment <code>RC_NUM</code>"
    )


def test_release_vote_gate_matches_blocking_contract() -> None:
    workflow = load_workflow(GATE_WORKFLOW)
    assert_gate_contract(workflow)


@pytest.mark.parametrize("event", ("pull_request", "push"))
def test_release_vote_gate_covers_all_workflows_and_archive_attributes(
    event: str,
) -> None:
    workflow = load_workflow(GATE_WORKFLOW)
    paths = set(workflow["on"][event]["paths"])

    assert ".github/workflows/**" in paths
    assert ".gitattributes" in paths


def test_release_vote_gate_verifies_current_source_tree() -> None:
    workflow = load_workflow(GATE_WORKFLOW)
    step = gate_step(workflow, "Verify current source tree")

    assert "if" not in step
    assert "continue-on-error" not in step
    assert step["shell"] == "bash"
    assert step["run"] == GATE_SOURCE_TREE_COMMAND


@pytest.mark.parametrize("event", ("pull_request", "push"))
@pytest.mark.parametrize("required_path", (".github/workflows/**", ".gitattributes"))
def test_gate_contract_rejects_missing_release_input_path(
    event: str,
    required_path: str,
) -> None:
    workflow = copy.deepcopy(load_workflow(GATE_WORKFLOW))
    workflow["on"][event]["paths"].remove(required_path)

    with pytest.raises(AssertionError):
        assert_gate_contract(workflow)


@pytest.mark.parametrize("event", ("pull_request", "push"))
def test_gate_contract_rejects_missing_release_branch(event: str) -> None:
    workflow = copy.deepcopy(load_workflow(GATE_WORKFLOW))
    workflow["on"][event]["branches"].remove("release-*")

    with pytest.raises(AssertionError):
        assert_gate_contract(workflow)


@pytest.mark.parametrize(
    "mutation",
    ("masked-command", "step-continue", "step-condition", "job-continue"),
)
def test_gate_contract_rejects_non_blocking_test_mutations(mutation: str) -> None:
    workflow = copy.deepcopy(load_workflow(GATE_WORKFLOW))
    if mutation == "masked-command":
        gate_step(workflow, "Run release vote tests")["run"] += " || true"
    elif mutation == "step-continue":
        gate_step(workflow, "Run release vote tests")[
            "continue-on-error"
        ] = "true"
    elif mutation == "step-condition":
        gate_step(workflow, "Run release vote tests")["if"] = "failure()"
    elif mutation == "job-continue":
        workflow["jobs"]["release-vote-gate"]["continue-on-error"] = "true"
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(AssertionError):
        assert_gate_contract(workflow)


@pytest.mark.parametrize(
    "mutation",
    ("masked-command", "step-continue", "step-condition", "wrong-commit"),
)
def test_gate_contract_rejects_source_tree_check_mutations(
    mutation: str,
) -> None:
    workflow = copy.deepcopy(load_workflow(GATE_WORKFLOW))
    source_step = gate_step(workflow, "Verify current source tree")
    if mutation == "masked-command":
        source_step["run"] += " || true"
    elif mutation == "step-continue":
        source_step["continue-on-error"] = "true"
    elif mutation == "step-condition":
        source_step["if"] = "failure()"
    elif mutation == "wrong-commit":
        source_step["run"] = source_step["run"].replace(
            '--commit "${commit}"',
            "--commit HEAD^",
        )
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(AssertionError):
        assert_gate_contract(workflow)


@pytest.mark.parametrize(
    "mutation",
    ("masked-command", "step-continue", "step-condition"),
)
def test_gate_contract_rejects_non_blocking_static_checks(mutation: str) -> None:
    workflow = copy.deepcopy(load_workflow(GATE_WORKFLOW))
    static_step = gate_step(workflow, "Run static checks")
    if mutation == "masked-command":
        static_step["run"] += " || true"
    elif mutation == "step-continue":
        static_step["continue-on-error"] = "true"
    elif mutation == "step-condition":
        static_step["if"] = "failure()"
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(AssertionError):
        assert_gate_contract(workflow)
