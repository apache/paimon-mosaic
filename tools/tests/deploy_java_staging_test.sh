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

new_fixture() {
  FIXTURE_DIR=$(mktemp -d "$TEST_ROOT/fixture.XXXXXX")
  OUTPUT_LOG="$TEST_ROOT/output.$TEST_COUNT.log"
  MAVEN_LOG="$TEST_ROOT/maven.$TEST_COUNT.log"
  TEMP_ROOT="$TEST_ROOT/tmp.$TEST_COUNT"
  mkdir -p \
    "$FIXTURE_DIR/fake-bin" \
    "$FIXTURE_DIR/java" \
    "$FIXTURE_DIR/tools" \
    "$TEMP_ROOT"

  cp "$TOOLS_DIR/deploy_java_staging.sh" "$FIXTURE_DIR/tools/"
  cp "$TOOLS_DIR/native_binary.py" "$FIXTURE_DIR/tools/"
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

if [[ "$1 $2" == "run view" ]]; then
  printf 'status=completed\nconclusion=success\nhead_sha=%s\nhead_branch=%s\nworkflow_name=Release\nevent=push\n' \
    "${FAKE_RUN_SHA:-$(git -C "$FAKE_REPO" rev-parse "${FAKE_RUN_REF}^{commit}")}" \
    "$FAKE_RUN_REF"
  exit 0
fi

if [[ "$1 $2" == "run download" ]]; then
  destination=
  artifact=
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --name) artifact=$2; shift 2 ;;
      --dir) destination=$2; shift 2 ;;
      *) shift ;;
    esac
  done
  [[ "$artifact" == "java-release-native-inputs" ]]
  mkdir -p \
    "$destination/linux/x86_64" \
    "$destination/linux/aarch64" \
    "$destination/macos/aarch64" \
    "$destination/windows/x86_64"
  : > "$destination/linux/x86_64/libpaimon_mosaic_jni.so"
  : > "$destination/linux/aarch64/libpaimon_mosaic_jni.so"
  : > "$destination/macos/aarch64/libpaimon_mosaic_jni.dylib"
  : > "$destination/windows/x86_64/paimon_mosaic_jni.dll"
  exit 0
fi

exit 2
EOF

  cat > "$FIXTURE_DIR/fake-bin/mvn" <<'EOF'
#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

{
  printf 'pwd=%s\n' "$PWD"
  printf 'args=%s\n' "$*"
  printf 'maven-opts=%s\n' "${MAVEN_OPTS:-}"
  printf 'maven-args=%s\n' "${MAVEN_ARGS:-}"
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
fi
EOF

  cat > "$FIXTURE_DIR/fake-bin/python3" <<'EOF'
#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

script=$(cat)
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
      FAKE_MVN_LOG="$MAVEN_LOG" \
      FAKE_REPO="$FIXTURE_DIR" \
      FAKE_RUN_REF="${FAKE_RUN_REF:-v0.3.0-rc1}" \
      FAKE_RUN_SHA="${FAKE_RUN_SHA:-}" \
      FAKE_GPG_FINGERPRINT="${FAKE_GPG_FINGERPRINT:-}" \
      FAKE_KEYS_FINGERPRINT="${FAKE_KEYS_FINGERPRINT:-}" \
      FAKE_SIGNATURE_FINGERPRINT="${FAKE_SIGNATURE_FINGERPRINT:-}" \
      GH_HOST=enterprise.example.invalid \
      TMPDIR="$TEMP_ROOT" \
      "$BASH" ./tools/deploy_java_staging.sh \
        --release-version 0.3.0 \
        --rc 1 \
        --run-id 42 \
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
  printf '<settings/>\n' > "$settings"
  printf 'fake KEYS\n' > "$keys"

  MAVEN_OPTS='-Dexec.skip=true -Dgpg.skip=true' run_script \
    --maven-settings "$settings" \
    --gpg-keyname 0123456789ABCDEF0123456789ABCDEF01234567 \
    --keys-file "$keys" > "$OUTPUT_LOG" 2>&1

  assert_contains "$MAVEN_LOG" \
    "args=-s $settings clean deploy -Prelease"
  assert_contains "$MAVEN_LOG" "-Dexec.skip=false"
  assert_contains "$MAVEN_LOG" "-Dgpg.skip=false"
  assert_contains "$MAVEN_LOG" \
    "-Dgpg.keyname=0123456789ABCDEF0123456789ABCDEF01234567!"
  assert_contains "$MAVEN_LOG" \
    "maven-opts=-Dexec.skip=true -Dgpg.skip=true"
  assert_contains "$MAVEN_LOG" "maven-args="
  if [[ $(grep -c '^pwd=' "$MAVEN_LOG") -ne 1 ]]; then
    fail "real deploy should invoke Maven exactly once"
  fi
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
run_test test_real_deploy_requires_official_repository_run
run_test test_git_index_flags_are_rejected
run_test test_dirty_caller_worktree_is_rejected
run_test test_git_replacement_refs_are_rejected
run_test test_repository_local_archive_attributes_are_rejected
run_test test_invalid_native_files_fail_without_external_file_command
run_test test_real_deploy_uses_one_verified_maven_lifecycle
run_test test_real_deploy_requires_full_signing_fingerprint
run_test test_real_deploy_rejects_unexpected_signature_key
run_test test_real_deploy_requires_signing_key_in_asf_keys

echo "All $TEST_COUNT deploy_java_staging tests passed with Bash $BASH_VERSION."
