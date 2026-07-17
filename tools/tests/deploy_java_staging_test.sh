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
TEST_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/paimon-mosaic-staging-test.XXXXXX")
TEST_COUNT=0

cleanup() {
  case "$TEST_ROOT" in
    "${TMPDIR:-/tmp}"/paimon-mosaic-staging-test.*)
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
    sed -n '1,160p' "$file" >&2
    fail "missing expected output"
  fi
}

assert_not_contains() {
  local file=$1
  local pattern=$2
  if [[ -f "$file" ]] && grep -Fq -- "$pattern" "$file"; then
    echo "Did not expect '$pattern' in $file" >&2
    sed -n '1,160p' "$file" >&2
    fail "unexpected output"
  fi
}

assert_maven_not_invoked() {
  local fixture=$1
  if [[ -s "$fixture/maven.log" ]]; then
    sed -n '1,160p' "$fixture/maven.log" >&2
    fail "Maven must not be invoked"
  fi
}

new_fixture() {
  FIXTURE_DIR=$(mktemp -d "$TEST_ROOT/fixture.XXXXXX")
  export FIXTURE_DIR

  mkdir -p \
    "$FIXTURE_DIR/fake-bin" \
    "$FIXTURE_DIR/java/src/main/resources" \
    "$FIXTURE_DIR/tools"

  cp "$TOOLS_DIR/deploy_java_staging.sh" "$FIXTURE_DIR/tools/"
  cp "$TOOLS_DIR/validate_java_staging_artifacts.sh" "$FIXTURE_DIR/tools/"
  chmod +x \
    "$FIXTURE_DIR/tools/deploy_java_staging.sh" \
    "$FIXTURE_DIR/tools/validate_java_staging_artifacts.sh"

  cat > "$FIXTURE_DIR/java/pom.xml" <<'EOF'
<project>
  <parent><version>23</version></parent>
  <version>0.3.0</version>
</project>
EOF

  cat > "$FIXTURE_DIR/.gitignore" <<'EOF'
java/target/
java/src/main/resources/native/
*.class
EOF

  cat > "$FIXTURE_DIR/fake-bin/gh" <<'EOF'
#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

if [[ "$1 $2" == "run view" ]]; then
  printf 'completed\nsuccess\n%s\n%s\nRelease\npush\n' \
    "$(git rev-parse HEAD)" "${FAKE_RUN_REF:-v0.3.0-rc1}"
  exit 0
fi

if [[ "$1 $2" == "run download" ]]; then
  artifact=
  destination=
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --name) artifact=$2; shift 2 ;;
      --dir) destination=$2; shift 2 ;;
      *) shift ;;
    esac
  done
  mkdir -p "$destination"
  case "$artifact" in
    native-linux-x86_64|native-linux-aarch64) file=libpaimon_mosaic_jni.so ;;
    native-macos-aarch64) file=libpaimon_mosaic_jni.dylib ;;
    native-windows-x86_64) file=paimon_mosaic_jni.dll ;;
    *) exit 2 ;;
  esac
  : > "$destination/$file"
  exit 0
fi

exit 2
EOF

  cat > "$FIXTURE_DIR/fake-bin/mvn" <<'EOF'
#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

printf '%s\n' "$*" >> "$FAKE_MVN_LOG"
validation_script=
for argument in "$@"; do
  case "$argument" in
    -DstagingValidationScript=*) validation_script=${argument#*=} ;;
  esac
done

mkdir -p target
: > target/mosaic-0.3.0.jar
: > target/mosaic-0.3.0-sources.jar
: > target/mosaic-0.3.0-javadoc.jar

if [[ -z "$validation_script" ]]; then
  echo "Missing stagingValidationScript Maven property" >&2
  exit 2
fi
"$validation_script" "$PWD/target" 0.3.0
EOF

  cat > "$FIXTURE_DIR/fake-bin/jar" <<'EOF'
#!/usr/bin/env bash
set -o errexit
set -o nounset

if [[ "${1-}" != "tf" ]]; then
  exit 2
fi
cat <<'ENTRIES'
native/linux/x86_64/libpaimon_mosaic_jni.so
native/linux/aarch64/libpaimon_mosaic_jni.so
native/macos/aarch64/libpaimon_mosaic_jni.dylib
native/windows/x86_64/paimon_mosaic_jni.dll
ENTRIES
EOF

  chmod +x "$FIXTURE_DIR/fake-bin/gh" "$FIXTURE_DIR/fake-bin/mvn" "$FIXTURE_DIR/fake-bin/jar"
  : > "$FIXTURE_DIR/DEPENDENCIES.rust.tsv"

  git -C "$FIXTURE_DIR" init -q
  git -C "$FIXTURE_DIR" config user.name "Release Script Test"
  git -C "$FIXTURE_DIR" config user.email "release-script-test@example.invalid"
  git -C "$FIXTURE_DIR" add .
  git -C "$FIXTURE_DIR" commit -q -m fixture
  git -C "$FIXTURE_DIR" tag v0.3.0-rc1
}

run_script() {
  local run_ref=${FAKE_RUN_REF:-v0.3.0-rc1}
  (
    cd "$FIXTURE_DIR"
    PATH="$FIXTURE_DIR/fake-bin:$(dirname "$BASH"):$PATH" \
      MVN="$FIXTURE_DIR/fake-bin/mvn" \
      FAKE_MVN_LOG="$FIXTURE_DIR/maven.log" \
      FAKE_RUN_REF="$run_ref" \
      "$BASH" ./tools/deploy_java_staging.sh \
        --release-version 0.3.0 \
        --rc 1 \
        --run-id 42 \
        --skip-native-file-check \
        "$@"
  )
}

test_missing_option_value_never_deploys() {
  new_fixture
  if run_script --staging-description --dry-run > "$FIXTURE_DIR/output.log" 2>&1; then
    fail "missing staging description value was accepted"
  fi
  assert_contains "$FIXTURE_DIR/output.log" "requires a value that is not another option"
  assert_maven_not_invoked "$FIXTURE_DIR"
}

test_ignored_package_input_is_rejected() {
  new_fixture
  printf 'unexpected bytecode\n' > "$FIXTURE_DIR/java/src/main/resources/Unexpected.class"
  if run_script --dry-run > "$FIXTURE_DIR/output.log" 2>&1; then
    fail "ignored Java package input was accepted"
  fi
  assert_contains "$FIXTURE_DIR/output.log" "java/src/main/resources/Unexpected.class"
  assert_maven_not_invoked "$FIXTURE_DIR"
}

test_workflow_run_must_match_tag() {
  new_fixture
  if FAKE_RUN_REF=v0.3.0-rc2 run_script --dry-run > "$FIXTURE_DIR/output.log" 2>&1; then
    fail "workflow run from another tag was accepted"
  fi
  assert_contains "$FIXTURE_DIR/output.log" "expected 'v0.3.0-rc1'"
  assert_maven_not_invoked "$FIXTURE_DIR"
}

test_rc_and_tag_must_match() {
  new_fixture
  if run_script --rc 2 --tag v0.3.0-rc1 --dry-run > "$FIXTURE_DIR/output.log" 2>&1; then
    fail "mismatched RC number and tag were accepted"
  fi
  assert_contains "$FIXTURE_DIR/output.log" "expected: v0.3.0-rc2"
  assert_maven_not_invoked "$FIXTURE_DIR"
}

test_relative_maven_settings_becomes_absolute() {
  new_fixture
  printf '<settings/>\n' > "$FIXTURE_DIR/deploysettings.xml"
  run_script --maven-settings deploysettings.xml --dry-run > "$FIXTURE_DIR/output.log" 2>&1
  assert_contains "$FIXTURE_DIR/maven.log" "-s $FIXTURE_DIR/deploysettings.xml"
}

test_dry_run_never_invokes_deploy() {
  new_fixture
  run_script --dry-run > "$FIXTURE_DIR/output.log" 2>&1
  assert_contains "$FIXTURE_DIR/maven.log" "clean verify"
  assert_not_contains "$FIXTURE_DIR/maven.log" " deploy"
  if [[ $(wc -l < "$FIXTURE_DIR/maven.log") -ne 1 ]]; then
    fail "dry-run should invoke Maven exactly once"
  fi
}

test_real_deploy_uses_one_validated_lifecycle() {
  new_fixture
  run_script > "$FIXTURE_DIR/output.log" 2>&1
  assert_contains "$FIXTURE_DIR/maven.log" "clean deploy"
  assert_contains "$FIXTURE_DIR/maven.log" "-DstagingValidationScript="
  if [[ $(wc -l < "$FIXTURE_DIR/maven.log") -ne 1 ]]; then
    fail "real deploy should invoke Maven exactly once"
  fi
}

run_test() {
  local name=$1
  "$name"
  TEST_COUNT=$((TEST_COUNT + 1))
  echo "PASS: $name"
}

run_test test_missing_option_value_never_deploys
run_test test_ignored_package_input_is_rejected
run_test test_workflow_run_must_match_tag
run_test test_rc_and_tag_must_match
run_test test_relative_maven_settings_becomes_absolute
run_test test_dry_run_never_invokes_deploy
run_test test_real_deploy_uses_one_validated_lifecycle

echo "All $TEST_COUNT deploy_java_staging tests passed with Bash $BASH_VERSION."
