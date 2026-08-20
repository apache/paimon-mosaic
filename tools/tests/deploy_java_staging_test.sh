#!/usr/bin/env bash

# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -o errexit
set -o nounset
set -o pipefail

TEST_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS_DIR=$(cd "$TEST_SCRIPT_DIR/.." && pwd)
TEST_TMPDIR=$(cd "${TMPDIR:-/tmp}" && pwd -P)
TEST_ROOT=$(mktemp -d "$TEST_TMPDIR/paimon-mosaic-staging-test.XXXXXX")
REAL_PYTHON=$(command -v python3)
TEST_COUNT=0

cleanup() {
  case "$TEST_ROOT" in
    "$TEST_TMPDIR"/paimon-mosaic-staging-test.*)
      rm -rf -- "$TEST_ROOT"
      ;;
    *)
      echo "Refusing to remove unexpected test path: $TEST_ROOT" >&2
      ;;
  esac
}
trap cleanup EXIT

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

assert_contains() {
  local file=$1
  local pattern=$2
  if ! grep -Fq -- "$pattern" "$file"; then
    echo "Expected '$pattern' in $file" >&2
    sed -n '1,200p' "$file" >&2
    fail "missing expected output"
  fi
}

assert_not_contains() {
  local file=$1
  local pattern=$2
  if [[ -f "$file" ]] && grep -Fq -- "$pattern" "$file"; then
    echo "Did not expect '$pattern' in $file" >&2
    sed -n '1,200p' "$file" >&2
    fail "unexpected output"
  fi
}

assert_maven_not_invoked() {
  if [[ -s "$MAVEN_LOG" ]]; then
    sed -n '1,200p' "$MAVEN_LOG" >&2
    fail "Maven must not be invoked"
  fi
}

assert_maven_arg() {
  local expected=$1
  local args
  args=$(sed -n 's/^args=//p' "$MAVEN_LOG")
  if ! tr ' ' '\n' <<< "$args" | grep -Fxq -- "$expected"; then
    echo "Expected exact Maven argument '$expected' in $MAVEN_LOG" >&2
    sed -n '1,200p' "$MAVEN_LOG" >&2
    fail "missing exact Maven argument"
  fi
}

assert_artifact_download_not_invoked() {
  if [[ -s "$ARTIFACT_DOWNLOAD_LOG" ]]; then
    sed -n '1,200p' "$ARTIFACT_DOWNLOAD_LOG" >&2
    fail "release artifacts must not be downloaded"
  fi
}

update_artifact_digest() {
  ARTIFACT_DIGEST=$(
    "$REAL_PYTHON" - "$ARTIFACT_ZIP" <<'PY'
import hashlib
import sys

print("sha256:" + hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
  )
}

new_fixture() {
  FIXTURE_DIR=$(mktemp -d "$TEST_ROOT/fixture.XXXXXX")
  OUTPUT_LOG="$TEST_ROOT/output.$TEST_COUNT.log"
  MAVEN_LOG="$TEST_ROOT/maven.$TEST_COUNT.log"
  TAG_VALIDATION_LOG="$TEST_ROOT/tag-validation.$TEST_COUNT.log"
  ARTIFACT_DOWNLOAD_LOG="$TEST_ROOT/artifact-download.$TEST_COUNT.log"
  CURL_LOG="$TEST_ROOT/curl.$TEST_COUNT.log"
  CURL_ATTEMPT_LOG="$TEST_ROOT/curl-attempt.$TEST_COUNT.log"
  ARTIFACT_ZIP="$TEST_ROOT/artifact.$TEST_COUNT.zip"
  TEMP_ROOT="$TEST_ROOT/tmp.$TEST_COUNT"
  mkdir -p \
    "$FIXTURE_DIR/fake-bin" \
    "$FIXTURE_DIR/java" \
    "$FIXTURE_DIR/tools" \
    "$TEMP_ROOT"

  "$REAL_PYTHON" - "$ARTIFACT_ZIP" <<'PY'
import sys
import zipfile

files = (
    "linux/aarch64/libpaimon_mosaic_jni.so",
    "linux/x86_64/libpaimon_mosaic_jni.so",
    "macos/aarch64/libpaimon_mosaic_jni.dylib",
    "windows/x86_64/paimon_mosaic_jni.dll",
)
with zipfile.ZipFile(sys.argv[1], "w", compression=zipfile.ZIP_STORED) as archive:
    for name in files:
        info = zipfile.ZipInfo(name, date_time=(2026, 8, 20, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        archive.writestr(info, b"")
PY
  update_artifact_digest

  cp "$TOOLS_DIR/deploy_java_staging.sh" "$FIXTURE_DIR/tools/"
  cp "$TOOLS_DIR/native_binary.py" "$FIXTURE_DIR/tools/"
  cp "$TOOLS_DIR/validate_release_tag.py" "$FIXTURE_DIR/tools/"
  cp "$TOOLS_DIR/verify_java_jars.py" "$FIXTURE_DIR/tools/"
  chmod +x "$FIXTURE_DIR/tools/deploy_java_staging.sh"

  cat > "$FIXTURE_DIR/java/pom.xml" <<'EOF'
<project>
  <parent><version>23</version></parent>
  <version>0.3.0</version>
</project>
EOF

  cat > "$FIXTURE_DIR/fake-bin/gh" <<'EOF'
#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

[[ "${GH_HOST:-}" == "github.com" ]]
expected_repository=${FAKE_EXPECTED_REPOSITORY:-apache/paimon-mosaic}
expected_run_id=${FAKE_EXPECTED_RUN_ID:-42}

if [[ "$1" == "api" && "$2" == */actions/runs/* ]]; then
  if [[ "$2" == */artifacts?* ]]; then
    printf 'args=%s\n' "$*" >> "${ARTIFACT_DOWNLOAD_LOG:-/dev/null}"
    "$REAL_PYTHON" - <<'PY'
import json
import os
import sys

count = int(os.environ.get("FAKE_ARTIFACT_COUNT", "1"))
run_id = int(os.environ.get("FAKE_EXPECTED_RUN_ID", "42"))
head_sha = os.environ.get("FAKE_RUN_SHA") or os.popen(
    f"git -C {os.environ['FAKE_REPO']} rev-parse "
    f"{os.environ['FAKE_RUN_REF']}^{{commit}}"
).read().strip()
artifacts = [
    {
        "id": 9001 + index,
        "name": "java-release-native-inputs",
        "expired": False,
        "digest": os.environ["FAKE_ARTIFACT_DIGEST"],
        "created_at": f"2026-08-20T00:00:0{index}Z",
        "workflow_run": {"id": run_id, "head_sha": head_sha},
    }
    for index in range(count)
]
json.dump([{"total_count": count, "artifacts": artifacts}], sys.stdout)
PY
    exit 0
  fi
  [[ "$2" == "repos/$expected_repository/actions/runs/$expected_run_id" ]]
  printf 'status=completed\nconclusion=success\nhead_sha=%s\nhead_branch=%s\nworkflow_name=Release\nworkflow_path=%s\nevent=push\n' \
    "${FAKE_RUN_SHA:-$(git -C "$FAKE_REPO" rev-parse "${FAKE_RUN_REF}^{commit}")}" \
    "$FAKE_RUN_REF" \
    "${FAKE_WORKFLOW_PATH:-.github/workflows/release.yml}"
  exit 0
fi

if [[ "$1" == "api" && "$2" == */actions/artifacts/*/zip ]]; then
  printf 'args=%s\n' "$*" >> "${ARTIFACT_DOWNLOAD_LOG:-/dev/null}"
  [[ "$2" == "repos/$expected_repository/actions/artifacts/9001/zip" ]]
  cat "$FAKE_ARTIFACT_ZIP"
  exit 0
fi

exit 2
EOF

  cat > "$FIXTURE_DIR/fake-bin/mvn" <<'EOF'
#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

if [[ "${MAVEN_SKIP_RC:-}" != 1 && -n "${FAKE_MAVEN_RC:-}" ]]; then
  # Model the standard Maven launcher, which sources mavenrc files unless
  # MAVEN_SKIP_RC is set before the launcher starts.
  source "$FAKE_MAVEN_RC"
fi
if [[ "${MAVEN_BASEDIR:-}" != "$PWD" && -n "${FAKE_MAVEN_CONFIG:-}" ]]; then
  MAVEN_ARGS=$(cat "$FAKE_MAVEN_CONFIG")
fi

{
  printf 'pwd=%s\n' "$PWD"
  printf 'args=%s\n' "$*"
  printf 'maven-opts=%s\n' "${MAVEN_OPTS:-}"
  printf 'maven-args=%s\n' "${MAVEN_ARGS:-}"
  printf 'java-tool-options=%s\n' "${JAVA_TOOL_OPTIONS:-}"
  printf 'jdk-java-options=%s\n' "${JDK_JAVA_OPTIONS:-}"
  printf 'underscore-java-options=%s\n' "${_JAVA_OPTIONS:-}"
  printf 'maven-skip-rc=%s\n' "${MAVEN_SKIP_RC:-}"
  printf 'maven-basedir=%s\n' "${MAVEN_BASEDIR:-}"
  printf 'maven-debug-opts=%s\n' "${MAVEN_DEBUG_OPTS:-}"
  printf 'maven-config=%s\n' "${MAVEN_CONFIG:-}"
  sed -n 's#.*<version>\([^<]*\)</version>.*#pom-version=\1#p' pom.xml | tail -n1
} >> "$FAKE_MVN_LOG"

if [[ " $* " == *" deploy "* ]]; then
  mkdir -p target
  for artifact in \
    mosaic-0.3.0.jar \
    mosaic-0.3.0-sources.jar \
    mosaic-0.3.0-javadoc.jar \
    mosaic-0.3.0.pom; do
    : > "target/$artifact"
    : > "target/$artifact.asc"
  done
  printf 'created-signed-artifact-pairs=4\n' >> "$FAKE_MVN_LOG"
fi

if [[ "${FAKE_MAVEN_EXIT_CODE:-0}" -ne 0 ]]; then
  exit "$FAKE_MAVEN_EXIT_CODE"
fi
EOF

  cat > "$FIXTURE_DIR/fake-bin/curl" <<'EOF'
#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

printf 'args=%s\n' "$*" >> "$FAKE_CURL_LOG"

retry=
retry_connrefused=false
connect_timeout=
max_time=
output=
proto=
url=
tlsv12=false
location=false
fail_on_http=false
silent=false
show_error=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --retry)
      retry=$2
      shift 2
      ;;
    --retry-connrefused)
      retry_connrefused=true
      shift
      ;;
    --connect-timeout)
      connect_timeout=$2
      shift 2
      ;;
    --max-time)
      max_time=$2
      shift 2
      ;;
    --output)
      output=$2
      shift 2
      ;;
    --proto)
      proto=$2
      shift 2
      ;;
    --tlsv1.2)
      tlsv12=true
      shift
      ;;
    --location)
      location=true
      shift
      ;;
    --fail)
      fail_on_http=true
      shift
      ;;
    --silent)
      silent=true
      shift
      ;;
    --show-error)
      show_error=true
      shift
      ;;
    https://*)
      url=$1
      shift
      ;;
    *)
      echo "unexpected curl argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$retry" != 3 ||
      "$retry_connrefused" != true ||
      "$connect_timeout" != 10 ||
      "$max_time" != 300 ||
      -z "$output" ||
      "$proto" != "=https" ||
      "$tlsv12" != true ||
      "$location" != true ||
      "$fail_on_http" != true ||
      "$silent" != true ||
      "$show_error" != true ||
      "$url" != "https://downloads.apache.org/paimon/KEYS" ]]; then
  echo "curl invocation does not match the bounded ASF KEYS contract" >&2
  exit 2
fi

max_attempts=$((retry + 1))
failures_before_success=${FAKE_CURL_FAILURES_BEFORE_SUCCESS:-0}
for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  printf 'attempt=%s\n' "$attempt" >> "$FAKE_CURL_ATTEMPT_LOG"
  if ((attempt > failures_before_success)); then
    printf 'fake KEYS\n' > "$output"
    exit 0
  fi
done

exit 22
EOF

  cat > "$FIXTURE_DIR/fake-bin/python3" <<'EOF'
#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

if [[ $# -gt 0 && "$1" == */validate_release_tag.py ]]; then
  printf 'args=%s\n' "$*" >> "$TAG_VALIDATION_LOG"
  if [[ "${FAKE_TAG_VALIDATION_RESULT:-success}" != success ]]; then
    echo "fake release tag validation failed" >&2
    exit 1
  fi
  exit 0
fi

script=$(cat)
if grep -q "ARTIFACT_SELECTION\|ARTIFACT_EXTRACTION" <<< "$script"; then
  printf '%s\n' "$script" | "$REAL_PYTHON" "$@"
  exit 0
fi
if grep -q "xml.etree.ElementTree" <<< "$script"; then
  printf '0.3.0\n'
fi
EOF

  cat > "$FIXTURE_DIR/fake-bin/gpg" <<'EOF'
#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

fingerprint=${FAKE_GPG_FINGERPRINT:-0123456789ABCDEF0123456789ABCDEF01234567}
if [[ " $* " == *" --verify "* ]]; then
  printf '[GNUPG:] VALIDSIG %s 0 0 0 0 0 0 0 00 %s\n' \
    "${FAKE_SIGNATURE_FINGERPRINT:-$fingerprint}" \
    "${FAKE_SIGNATURE_FINGERPRINT:-$fingerprint}"
elif [[ " $* " == *" --import "* ]]; then
  printf 'pub:-:255:22:0000000000000000:0:0::::::\n'
  printf 'fpr:::::::::%s:\n' "${FAKE_KEYS_FINGERPRINT:-$fingerprint}"
else
  printf 'sec:-:255:22:0000000000000000:0:0::::::\n'
  printf 'fpr:::::::::%s:\n' "$fingerprint"
fi
EOF

  cat > "$FIXTURE_DIR/fake-bin/file" <<'EOF'
#!/bin/sh
echo "external file command must not be used" >&2
exit 99
EOF

  chmod +x \
    "$FIXTURE_DIR/fake-bin/curl" \
    "$FIXTURE_DIR/fake-bin/file" \
    "$FIXTURE_DIR/fake-bin/gh" \
    "$FIXTURE_DIR/fake-bin/gpg" \
    "$FIXTURE_DIR/fake-bin/mvn" \
    "$FIXTURE_DIR/fake-bin/python3"

  git -C "$FIXTURE_DIR" init -q
  git -C "$FIXTURE_DIR" config user.name "Release Script Test"
  git -C "$FIXTURE_DIR" config user.email "release-script-test@example.invalid"
  git -C "$FIXTURE_DIR" add .
  git -C "$FIXTURE_DIR" commit -q -m fixture
  git -C "$FIXTURE_DIR" tag v0.3.0-rc1
}

run_script() {
  (
    cd "$FIXTURE_DIR"
    PATH="$FIXTURE_DIR/fake-bin:$(dirname "$BASH"):$PATH" \
      MVN="$FIXTURE_DIR/fake-bin/mvn" \
      PYTHON="$FIXTURE_DIR/fake-bin/python3" \
      GPG="$FIXTURE_DIR/fake-bin/gpg" \
      CURL="$FIXTURE_DIR/fake-bin/curl" \
      FAKE_MVN_LOG="$MAVEN_LOG" \
      FAKE_MAVEN_EXIT_CODE="${FAKE_MAVEN_EXIT_CODE:-0}" \
      FAKE_REPO="$FIXTURE_DIR" \
      FAKE_RUN_REF="${FAKE_RUN_REF:-v0.3.0-rc1}" \
      FAKE_RUN_SHA="${FAKE_RUN_SHA:-}" \
      FAKE_EXPECTED_REPOSITORY="${FAKE_EXPECTED_REPOSITORY:-apache/paimon-mosaic}" \
      FAKE_EXPECTED_RUN_ID="${FAKE_EXPECTED_RUN_ID:-42}" \
      FAKE_ARTIFACT_COUNT="${FAKE_ARTIFACT_COUNT:-1}" \
      FAKE_ARTIFACT_DIGEST="${FAKE_ARTIFACT_DIGEST:-$ARTIFACT_DIGEST}" \
      FAKE_ARTIFACT_ZIP="$ARTIFACT_ZIP" \
      FAKE_MAVEN_RC="${FAKE_MAVEN_RC:-}" \
      FAKE_MAVEN_CONFIG="${FAKE_MAVEN_CONFIG:-}" \
      FAKE_WORKFLOW_PATH="${FAKE_WORKFLOW_PATH:-}" \
      FAKE_TAG_VALIDATION_RESULT="${FAKE_TAG_VALIDATION_RESULT:-}" \
      FAKE_GPG_FINGERPRINT="${FAKE_GPG_FINGERPRINT:-}" \
      FAKE_KEYS_FINGERPRINT="${FAKE_KEYS_FINGERPRINT:-}" \
      FAKE_SIGNATURE_FINGERPRINT="${FAKE_SIGNATURE_FINGERPRINT:-}" \
      FAKE_CURL_FAILURES_BEFORE_SUCCESS="${FAKE_CURL_FAILURES_BEFORE_SUCCESS:-0}" \
      FAKE_CURL_LOG="$CURL_LOG" \
      FAKE_CURL_ATTEMPT_LOG="$CURL_ATTEMPT_LOG" \
      REAL_PYTHON="$REAL_PYTHON" \
      ARTIFACT_DOWNLOAD_LOG="$ARTIFACT_DOWNLOAD_LOG" \
      TAG_VALIDATION_LOG="$TAG_VALIDATION_LOG" \
      GH_HOST=enterprise.example.invalid \
      TMPDIR="$TEMP_ROOT" \
      "$BASH" ./tools/deploy_java_staging.sh \
        --release-version 0.3.0 \
        --rc 1 \
        --run-id "${RUN_ID_UNDER_TEST:-42}" \
        "$@"
  )
}

test_dry_run_builds_exact_tag_in_isolated_directory() {
  new_fixture
  run_script --dry-run > "$OUTPUT_LOG" 2>&1

  assert_contains "$MAVEN_LOG" \
    "args=clean verify -Prelease -Dexec.skip=false -Dgpg.skip=true"
  assert_contains "$MAVEN_LOG" "-DskipTests"
  assert_contains "$MAVEN_LOG" "pom-version=0.3.0"
  assert_not_contains "$MAVEN_LOG" " deploy"
  assert_not_contains "$TAG_VALIDATION_LOG" "validate_release_tag.py"
  if grep -Fq "pwd=$FIXTURE_DIR/java" "$MAVEN_LOG"; then
    fail "Maven used the caller's worktree instead of an isolated tag archive"
  fi
}

test_run_tests_omits_skip_flag() {
  new_fixture
  run_script --dry-run --run-tests > "$OUTPUT_LOG" 2>&1

  assert_contains "$MAVEN_LOG" \
    "args=clean verify -Prelease -Dexec.skip=false -Dgpg.skip=true"
  assert_not_contains "$MAVEN_LOG" "-DskipTests"
}

test_missing_option_value_never_deploys() {
  new_fixture
  if run_script --staging-description --dry-run > "$OUTPUT_LOG" 2>&1; then
    fail "missing staging description value was accepted"
  fi
  assert_contains "$OUTPUT_LOG" "requires a value that is not another option"
  assert_maven_not_invoked
}

test_workflow_run_sha_must_match_tag() {
  new_fixture
  if FAKE_RUN_SHA=0000000000000000000000000000000000000000 \
    run_script --dry-run > "$OUTPUT_LOG" 2>&1; then
    fail "workflow run from another commit was accepted"
  fi
  assert_contains "$OUTPUT_LOG" "does not match v0.3.0-rc1"
  assert_maven_not_invoked
}

test_artifact_download_uses_validated_run_and_repository() {
  new_fixture
  FAKE_EXPECTED_REPOSITORY=example/fork \
    FAKE_EXPECTED_RUN_ID=314159 \
    RUN_ID_UNDER_TEST=314159 \
    run_script --repo example/fork --dry-run > "$OUTPUT_LOG" 2>&1

  assert_contains "$ARTIFACT_DOWNLOAD_LOG" \
    "args=api repos/example/fork/actions/artifacts/9001/zip"
  assert_not_contains "$ARTIFACT_DOWNLOAD_LOG" "run download"
}

test_duplicate_release_artifacts_are_rejected() {
  new_fixture
  if FAKE_ARTIFACT_COUNT=2 run_script --dry-run > "$OUTPUT_LOG" 2>&1; then
    fail "multiple same-name release artifacts were accepted"
  fi
  assert_contains "$OUTPUT_LOG" "multiple unexpired java-release-native-inputs artifacts"
  assert_contains "$OUTPUT_LOG" "id=9001 created_at=2026-08-20T00:00:00Z"
  assert_contains "$OUTPUT_LOG" "id=9002 created_at=2026-08-20T00:00:01Z"
  assert_maven_not_invoked
}

test_release_artifact_digest_must_match_metadata() {
  new_fixture
  wrong_digest="sha256:$(printf '1%.0s' {1..64})"
  if FAKE_ARTIFACT_DIGEST="$wrong_digest" \
    run_script --dry-run > "$OUTPUT_LOG" 2>&1; then
    fail "release artifact with the wrong digest was accepted"
  fi
  assert_contains "$OUTPUT_LOG" "release artifact digest mismatch"
  assert_not_contains "$OUTPUT_LOG" "Traceback"
  assert_maven_not_invoked
}

test_release_artifact_rejects_unsafe_zip_path() {
  new_fixture
  "$REAL_PYTHON" - "$ARTIFACT_ZIP" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1], "w") as archive:
    archive.writestr("../escape", b"unsafe")
PY
  update_artifact_digest

  if run_script --dry-run > "$OUTPUT_LOG" 2>&1; then
    fail "release artifact with a traversal path was accepted"
  fi
  assert_contains "$OUTPUT_LOG" "unsafe release artifact path"
  assert_maven_not_invoked
}

test_release_artifact_rejects_symlink_entry() {
  new_fixture
  "$REAL_PYTHON" - "$ARTIFACT_ZIP" <<'PY'
import stat
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1], "w") as archive:
    info = zipfile.ZipInfo("linux/x86_64/libpaimon_mosaic_jni.so")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive.writestr(info, b"../../outside")
PY
  update_artifact_digest

  if run_script --dry-run > "$OUTPUT_LOG" 2>&1; then
    fail "release artifact with a symlink entry was accepted"
  fi
  assert_contains "$OUTPUT_LOG" "unsupported release artifact entry type"
  assert_maven_not_invoked
}

test_release_artifact_rejects_duplicate_zip_path() {
  new_fixture
  "$REAL_PYTHON" - "$ARTIFACT_ZIP" <<'PY' 2>/dev/null
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1], "w") as archive:
    archive.writestr("linux/x86_64/libpaimon_mosaic_jni.so", b"first")
    archive.writestr("linux/x86_64/libpaimon_mosaic_jni.so", b"second")
PY
  update_artifact_digest

  if run_script --dry-run > "$OUTPUT_LOG" 2>&1; then
    fail "release artifact with a duplicate path was accepted"
  fi
  assert_contains "$OUTPUT_LOG" "duplicate release artifact path"
  assert_maven_not_invoked
}

test_release_artifact_rejects_extra_regular_payload() {
  new_fixture
  "$REAL_PYTHON" - "$ARTIFACT_ZIP" <<'PY'
import sys
import zipfile

files = (
    "linux/aarch64/libpaimon_mosaic_jni.so",
    "linux/x86_64/libpaimon_mosaic_jni.so",
    "macos/aarch64/libpaimon_mosaic_jni.dylib",
    "windows/x86_64/paimon_mosaic_jni.dll",
    "README.txt",
)
with zipfile.ZipFile(sys.argv[1], "w", compression=zipfile.ZIP_STORED) as archive:
    for name in files:
        info = zipfile.ZipInfo(name, date_time=(2026, 8, 20, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        archive.writestr(info, b"regular payload")
PY
  update_artifact_digest

  if run_script --dry-run > "$OUTPUT_LOG" 2>&1; then
    fail "release artifact with an extra regular payload was accepted"
  fi
  assert_contains "$OUTPUT_LOG" \
    "Downloaded Java native inputs differ from the four expected files."
  assert_contains "$OUTPUT_LOG" "README.txt"
  assert_maven_not_invoked
}

test_real_deploy_requires_official_repository_run() {
  new_fixture
  if run_script --repo example/fork > "$OUTPUT_LOG" 2>&1; then
    fail "real deployment accepted a fork workflow run"
  fi
  assert_contains "$OUTPUT_LOG" "official apache/paimon-mosaic repository"
  assert_maven_not_invoked
}

test_git_index_flags_are_rejected() {
  new_fixture
  git -C "$FIXTURE_DIR" update-index --assume-unchanged java/pom.xml
  cat > "$FIXTURE_DIR/java/pom.xml" <<'EOF'
<project>
  <parent><version>23</version></parent>
  <version>9.9.9</version>
</project>
EOF

  if run_script --dry-run > "$OUTPUT_LOG" 2>&1; then
    fail "assume-unchanged package input was accepted"
  fi
  assert_contains "$OUTPUT_LOG" "index flags"
  assert_maven_not_invoked
}

test_dirty_caller_worktree_is_rejected() {
  new_fixture
  printf '\n# local change\n' >> "$FIXTURE_DIR/tools/deploy_java_staging.sh"

  if run_script --dry-run > "$OUTPUT_LOG" 2>&1; then
    fail "dirty caller worktree was accepted"
  fi
  assert_contains "$OUTPUT_LOG" "worktree must be completely clean"
  assert_maven_not_invoked
}

test_git_replacement_refs_are_rejected() {
  new_fixture
  first_blob=$(printf 'first\n' | git -C "$FIXTURE_DIR" hash-object -w --stdin)
  second_blob=$(printf 'second\n' | git -C "$FIXTURE_DIR" hash-object -w --stdin)
  git -C "$FIXTURE_DIR" replace "$first_blob" "$second_blob"

  if run_script --dry-run > "$OUTPUT_LOG" 2>&1; then
    fail "Git replacement refs were accepted"
  fi
  assert_contains "$OUTPUT_LOG" "replacement refs"
  assert_maven_not_invoked
}

test_repository_local_archive_attributes_are_rejected() {
  new_fixture
  mkdir -p "$FIXTURE_DIR/.git/info"
  printf 'java/pom.xml export-ignore\n' > "$FIXTURE_DIR/.git/info/attributes"

  if run_script --dry-run > "$OUTPUT_LOG" 2>&1; then
    fail "repository-local archive attributes were accepted"
  fi
  assert_contains "$OUTPUT_LOG" "repository-local Git attributes"
  assert_maven_not_invoked
}

test_invalid_native_files_fail_without_external_file_command() {
  new_fixture
  (
    cd "$FIXTURE_DIR"
    PATH="$FIXTURE_DIR/fake-bin:$PATH" \
      MVN="$FIXTURE_DIR/fake-bin/mvn" \
      PYTHON="$REAL_PYTHON" \
      FAKE_MVN_LOG="$MAVEN_LOG" \
      FAKE_REPO="$FIXTURE_DIR" \
      FAKE_RUN_REF=v0.3.0-rc1 \
      FAKE_ARTIFACT_DIGEST="$ARTIFACT_DIGEST" \
      FAKE_ARTIFACT_ZIP="$ARTIFACT_ZIP" \
      REAL_PYTHON="$REAL_PYTHON" \
      TMPDIR="$TEMP_ROOT" \
      "$BASH" ./tools/deploy_java_staging.sh \
        --release-version 0.3.0 \
        --rc 1 \
        --run-id 42 \
        --dry-run
  ) > "$OUTPUT_LOG" 2>&1 &&
    fail "invalid native files were accepted"

  assert_contains "$OUTPUT_LOG" "unrecognized native binary format"
  assert_maven_not_invoked
}

test_real_deploy_uses_one_verified_maven_lifecycle() {
  new_fixture
  settings="$TEST_ROOT/settings.$TEST_COUNT.xml"
  keys="$TEST_ROOT/keys.$TEST_COUNT"
  maven_rc="$TEST_ROOT/mavenrc.$TEST_COUNT"
  maven_config="$TEST_ROOT/maven.config.$TEST_COUNT"
  cat > "$settings" <<'EOF'
<settings>
  <profiles>
    <profile>
      <id>hostile-staging-defaults</id>
      <properties>
        <stagingRepositoryId>orgapachepaimon-from-settings</stagingRepositoryId>
        <stagingProfileId>deadbeef-from-settings</stagingProfileId>
        <keepStagingRepositoryOnFailure>true</keepStagingRepositoryOnFailure>
        <keepStagingRepositoryOnCloseRuleFailure>true</keepStagingRepositoryOnCloseRuleFailure>
      </properties>
    </profile>
  </profiles>
  <activeProfiles>
    <activeProfile>hostile-staging-defaults</activeProfile>
  </activeProfiles>
</settings>
EOF
  printf 'fake KEYS\n' > "$keys"
  cat > "$maven_rc" <<'EOF'
export MAVEN_OPTS='-DstagingRepositoryId=orgapachepaimon-from-rc'
export JAVA_TOOL_OPTIONS='-DkeepStagingRepositoryOnFailure=true'
EOF
  printf '%s\n' '-f=/tmp/hostile-pom.xml' > "$maven_config"
  hostile_maven_opts='-Xmx1g -DstagingRepositoryId=orgapachepaimon-4242 -DkeepStagingRepositoryOnFailure=true -Dexec.skip=true -Dgpg.skip=true'
  hostile_java_tool_options='-Xms256m -DkeepStagingRepositoryOnCloseRuleFailure=true -DskipNexusStagingDeployMojo=true'
  hostile_jdk_java_options='-DstagingProfileId=deadbeef -DskipStaging=true'
  hostile_underscore_java_options='-DskipStagingRepositoryClose=true -Dmaven.wagon.http.ssl.insecure=true'

  MAVEN_OPTS="$hostile_maven_opts" \
    JAVA_TOOL_OPTIONS="$hostile_java_tool_options" \
    JDK_JAVA_OPTIONS="$hostile_jdk_java_options" \
    _JAVA_OPTIONS="$hostile_underscore_java_options" \
    MAVEN_BASEDIR="$TEST_ROOT/hostile-basedir" \
    MAVEN_DEBUG_OPTS='-DstagingRepositoryId=from-debug-opts' \
    MAVEN_CONFIG='-f=/tmp/hostile-config-pom.xml' \
    FAKE_MAVEN_RC="$maven_rc" \
    FAKE_MAVEN_CONFIG="$maven_config" \
    run_script \
      --maven-settings "$settings" \
      --gpg-keyname 0123456789ABCDEF0123456789ABCDEF01234567 \
      --keys-file "$keys" > "$OUTPUT_LOG" 2>&1

  assert_contains "$MAVEN_LOG" \
    "args=-s $settings clean deploy -Prelease"
  assert_contains "$MAVEN_LOG" "-Dexec.skip=false"
  assert_contains "$MAVEN_LOG" "-Dgpg.skip=false"
  for property in \
    skipLocalStaging \
    skipNexusStagingDeployMojo \
    skipRemoteStaging \
    skipStaging \
    skipStagingRepositoryClose \
    maven.wagon.http.ssl.allowall \
    maven.wagon.http.ssl.insecure; do
    assert_contains "$MAVEN_LOG" "-D${property}=false"
  done
  # Maven command-line -D user properties take precedence over properties from
  # active profiles in settings.xml, so these argv assertions prove the
  # hostile settings defaults cannot select or retain a staging repository.
  assert_maven_arg "-DstagingRepositoryId="
  assert_maven_arg "-DstagingProfileId="
  assert_maven_arg "-DkeepStagingRepositoryOnFailure=false"
  assert_maven_arg \
    "-DkeepStagingRepositoryOnCloseRuleFailure=false"
  assert_contains "$MAVEN_LOG" \
    "-Dgpg.keyname=0123456789ABCDEF0123456789ABCDEF01234567!"
  assert_contains "$MAVEN_LOG" "maven-opts="
  assert_not_contains "$MAVEN_LOG" "maven-opts=$hostile_maven_opts"
  assert_contains "$MAVEN_LOG" "java-tool-options="
  assert_not_contains "$MAVEN_LOG" \
    "java-tool-options=$hostile_java_tool_options"
  assert_contains "$MAVEN_LOG" "jdk-java-options="
  assert_not_contains "$MAVEN_LOG" \
    "jdk-java-options=$hostile_jdk_java_options"
  assert_contains "$MAVEN_LOG" "underscore-java-options="
  assert_not_contains "$MAVEN_LOG" \
    "underscore-java-options=$hostile_underscore_java_options"
  assert_contains "$MAVEN_LOG" "maven-skip-rc=1"
  maven_pwd=$(sed -n 's/^pwd=//p' "$MAVEN_LOG")
  maven_basedir=$(sed -n 's/^maven-basedir=//p' "$MAVEN_LOG")
  if [[ "$maven_basedir" != "$maven_pwd" ]]; then
    fail "Maven base directory was not pinned to the isolated Java project"
  fi
  assert_contains "$MAVEN_LOG" "maven-debug-opts="
  assert_not_contains "$MAVEN_LOG" "from-debug-opts"
  assert_contains "$MAVEN_LOG" "maven-config="
  assert_not_contains "$MAVEN_LOG" "hostile-config-pom"
  assert_contains "$MAVEN_LOG" "maven-args="
  assert_contains "$TAG_VALIDATION_LOG" \
    "$FIXTURE_DIR/tools/validate_release_tag.py v0.3.0-rc1 --keys-file $keys --repository $FIXTURE_DIR --expected-commit $(git -C "$FIXTURE_DIR" rev-parse HEAD)"
  if [[ $(grep -c '^pwd=' "$MAVEN_LOG") -ne 1 ]]; then
    fail "real deploy should invoke Maven exactly once"
  fi
}

test_real_deploy_preserves_maven_partial_failure() {
  new_fixture
  settings="$TEST_ROOT/settings.$TEST_COUNT.xml"
  keys="$TEST_ROOT/keys.$TEST_COUNT"
  printf '<settings/>\n' > "$settings"
  printf 'fake KEYS\n' > "$keys"

  set +o errexit
  (
    export FAKE_MAVEN_EXIT_CODE=42
    run_script \
      --maven-settings "$settings" \
      --gpg-keyname 0123456789ABCDEF0123456789ABCDEF01234567 \
      --keys-file "$keys"
  ) > "$OUTPUT_LOG" 2>&1
  status=$?
  set -o errexit

  if [[ "$status" -ne 42 ]]; then
    fail "Maven exit 42 was not preserved; got $status"
  fi
  assert_contains "$MAVEN_LOG" "args=-s $settings clean deploy -Prelease"
  assert_contains "$MAVEN_LOG" "created-signed-artifact-pairs=4"
  assert_not_contains "$OUTPUT_LOG" "Java staging deploy finished."
}

test_default_asf_keys_download_retries_then_succeeds() {
  new_fixture
  settings="$TEST_ROOT/settings.$TEST_COUNT.xml"
  printf '<settings/>\n' > "$settings"

  (
    export FAKE_CURL_FAILURES_BEFORE_SUCCESS=2
    run_script \
      --maven-settings "$settings" \
      --gpg-keyname 0123456789ABCDEF0123456789ABCDEF01234567
  ) > "$OUTPUT_LOG" 2>&1

  assert_contains "$CURL_LOG" \
    "args=--proto =https --tlsv1.2 --location --fail --silent --show-error --retry 3 --retry-connrefused --connect-timeout 10 --max-time 300"
  if [[ $(grep -c '^attempt=' "$CURL_ATTEMPT_LOG") -ne 3 ]]; then
    fail "default ASF KEYS download did not succeed on the third attempt"
  fi
  assert_contains "$CURL_ATTEMPT_LOG" "attempt=1"
  assert_contains "$CURL_ATTEMPT_LOG" "attempt=2"
  assert_contains "$CURL_ATTEMPT_LOG" "attempt=3"
  assert_contains "$OUTPUT_LOG" "Java staging deploy finished."
}

test_default_asf_keys_download_permanent_failure_stops_release() {
  new_fixture
  settings="$TEST_ROOT/settings.$TEST_COUNT.xml"
  printf '<settings/>\n' > "$settings"

  set +o errexit
  (
    export FAKE_CURL_FAILURES_BEFORE_SUCCESS=100
    run_script \
      --maven-settings "$settings" \
      --gpg-keyname 0123456789ABCDEF0123456789ABCDEF01234567
  ) > "$OUTPUT_LOG" 2>&1
  status=$?
  set -o errexit

  if [[ "$status" -ne 22 ]]; then
    fail "curl exit 22 was not preserved; got $status"
  fi

  if [[ $(grep -c '^attempt=' "$CURL_ATTEMPT_LOG") -ne 4 ]]; then
    fail "curl retry bound did not stop after one attempt plus three retries"
  fi
  assert_artifact_download_not_invoked
  assert_maven_not_invoked
  assert_not_contains "$OUTPUT_LOG" "Java staging deploy finished."
}

test_real_deploy_requires_signed_release_tag() {
  new_fixture
  settings="$TEST_ROOT/settings.$TEST_COUNT.xml"
  keys="$TEST_ROOT/keys.$TEST_COUNT"
  printf '<settings/>\n' > "$settings"
  printf 'fake KEYS\n' > "$keys"

  if FAKE_TAG_VALIDATION_RESULT=failure run_script \
    --maven-settings "$settings" \
    --gpg-keyname 0123456789ABCDEF0123456789ABCDEF01234567 \
    --keys-file "$keys" > "$OUTPUT_LOG" 2>&1; then
    fail "real deployment accepted a tag that failed release validation"
  fi

  assert_contains "$OUTPUT_LOG" "fake release tag validation failed"
  assert_contains "$TAG_VALIDATION_LOG" \
    "validate_release_tag.py v0.3.0-rc1"
  assert_artifact_download_not_invoked
  assert_maven_not_invoked
}

test_run_must_use_canonical_release_workflow() {
  new_fixture

  if FAKE_WORKFLOW_PATH=.github/workflows/not-release.yml \
    run_script --dry-run > "$OUTPUT_LOG" 2>&1; then
    fail "non-canonical Release workflow was accepted"
  fi

  assert_contains "$OUTPUT_LOG" "canonical Release workflow"
  assert_maven_not_invoked
}

test_real_deploy_requires_full_signing_fingerprint() {
  new_fixture
  if run_script --gpg-keyname ABCDEF > "$OUTPUT_LOG" 2>&1; then
    fail "real deployment accepted a short signing key id"
  fi
  assert_contains "$OUTPUT_LOG" "full 40- or 64-hex OpenPGP fingerprint"
  assert_maven_not_invoked
}

test_real_deploy_rejects_unexpected_signature_key() {
  new_fixture
  keys="$TEST_ROOT/keys.$TEST_COUNT"
  printf 'fake KEYS\n' > "$keys"

  if FAKE_SIGNATURE_FINGERPRINT=89ABCDEF0123456789ABCDEF0123456789ABCDEF \
    run_script \
      --gpg-keyname 0123456789ABCDEF0123456789ABCDEF01234567 \
      --keys-file "$keys" > "$OUTPUT_LOG" 2>&1; then
    fail "real deployment accepted artifacts signed by another key"
  fi
  assert_contains "$OUTPUT_LOG" "Unexpected signer"
}

test_real_deploy_requires_signing_key_in_asf_keys() {
  new_fixture
  keys="$TEST_ROOT/keys.$TEST_COUNT"
  printf 'fake KEYS\n' > "$keys"

  if FAKE_KEYS_FINGERPRINT=89ABCDEF0123456789ABCDEF0123456789ABCDEF \
    run_script \
      --gpg-keyname 0123456789ABCDEF0123456789ABCDEF01234567 \
      --keys-file "$keys" > "$OUTPUT_LOG" 2>&1; then
    fail "real deployment accepted a signing key absent from ASF KEYS"
  fi
  assert_contains "$OUTPUT_LOG" "is not present in the ASF Paimon KEYS file"
  assert_maven_not_invoked
}

run_test() {
  local name=$1
  "$name"
  TEST_COUNT=$((TEST_COUNT + 1))
  echo "PASS: $name"
}

run_test test_dry_run_builds_exact_tag_in_isolated_directory
run_test test_run_tests_omits_skip_flag
run_test test_missing_option_value_never_deploys
run_test test_workflow_run_sha_must_match_tag
run_test test_artifact_download_uses_validated_run_and_repository
run_test test_duplicate_release_artifacts_are_rejected
run_test test_release_artifact_digest_must_match_metadata
run_test test_release_artifact_rejects_unsafe_zip_path
run_test test_release_artifact_rejects_symlink_entry
run_test test_release_artifact_rejects_duplicate_zip_path
run_test test_release_artifact_rejects_extra_regular_payload
run_test test_real_deploy_requires_official_repository_run
run_test test_git_index_flags_are_rejected
run_test test_dirty_caller_worktree_is_rejected
run_test test_git_replacement_refs_are_rejected
run_test test_repository_local_archive_attributes_are_rejected
run_test test_invalid_native_files_fail_without_external_file_command
run_test test_real_deploy_uses_one_verified_maven_lifecycle
run_test test_real_deploy_preserves_maven_partial_failure
run_test test_default_asf_keys_download_retries_then_succeeds
run_test test_default_asf_keys_download_permanent_failure_stops_release
run_test test_real_deploy_requires_signed_release_tag
run_test test_run_must_use_canonical_release_workflow
run_test test_real_deploy_requires_full_signing_fingerprint
run_test test_real_deploy_rejects_unexpected_signature_key
run_test test_real_deploy_requires_signing_key_in_asf_keys

echo "All $TEST_COUNT deploy_java_staging tests passed with Bash $BASH_VERSION."
