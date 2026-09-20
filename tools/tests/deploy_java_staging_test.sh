#!/usr/bin/env bash

#
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
#

set -o errexit
set -o nounset
set -o pipefail

SOURCE_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TEST_ROOT=$(mktemp -d)
trap 'rm -rf "$TEST_ROOT"' EXIT
TESTS=0
FIXTURES=0

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_contains() {
  local file=$1
  local text=$2
  grep -F -- "$text" "$file" >/dev/null || fail "$file does not contain: $text"
}

assert_not_contains() {
  local file=$1
  local text=$2
  if grep -F -- "$text" "$file" >/dev/null; then
    fail "$file unexpectedly contains: $text"
  fi
}

new_fixture() {
  FIXTURE="$TEST_ROOT/fixture-$FIXTURES"
  FIXTURES=$((FIXTURES + 1))
  MOCK_BIN="$FIXTURE/mock-bin"
  MOCK_LOG="$FIXTURE/mock.log"
  mkdir -p "$FIXTURE/tools" "$FIXTURE/java/src/test/java/org/apache/paimon/mosaic" "$MOCK_BIN"
  cp "$SOURCE_REPO/tools/deploy_java_staging.sh" "$FIXTURE/tools/"
  cp "$SOURCE_REPO/tools/verify_release_artifacts.py" "$FIXTURE/tools/"
  cp "$SOURCE_REPO/java/src/test/java/org/apache/paimon/mosaic/MosaicNativeLoaderSmokeTest.java" \
    "$FIXTURE/java/src/test/java/org/apache/paimon/mosaic/"
  cat > "$FIXTURE/java/pom.xml" <<'POM'
<project>
  <parent><version>23</version></parent>
  <artifactId>mosaic</artifactId>
  <version>1.2.3</version>
</project>
POM

  cat > "$MOCK_BIN/gh" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
printf 'gh %s\n' "$*" >> "$MOCK_LOG"
if [[ "$1 $2" == "run view" ]]; then
  printf 'completed\nsuccess\n%s\n%s\nRelease\npush\n' "$MOCK_RUN_SHA" "$MOCK_TAG"
  exit 0
fi
if [[ "$1 $2" != "run download" ]]; then
  exit 2
fi
name=
dir=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name=$2; shift 2 ;;
    --dir) dir=$2; shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p "$dir"
case "$name" in
  native-linux-x86_64) file=libpaimon_mosaic_jni.so ;;
  native-linux-aarch64) file=libpaimon_mosaic_jni.so ;;
  native-macos-aarch64) file=libpaimon_mosaic_jni.dylib ;;
  native-windows-x86_64) file=paimon_mosaic_jni.dll ;;
  java-package)
    printf jar > "$dir/mosaic-1.2.3.jar"
    printf sources > "$dir/mosaic-1.2.3-sources.jar"
    if [[ "${OMIT_CI_JAVADOC:-0}" != 1 ]]; then
      printf javadoc > "$dir/mosaic-1.2.3-javadoc.jar"
    fi
    exit 0
    ;;
  *) exit 3 ;;
esac
printf native > "$dir/$file"
MOCK

  cat > "$MOCK_BIN/mvn" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
printf 'mvn %s\n' "$*" >> "$MOCK_LOG"
mkdir -p target/test-classes/org/apache/paimon/mosaic
printf class > target/test-classes/org/apache/paimon/mosaic/MosaicNativeLoaderSmokeTest.class
if [[ "${OMIT_LOCAL_MAIN:-0}" != 1 ]]; then
  printf jar > target/mosaic-1.2.3.jar
fi
printf sources > target/mosaic-1.2.3-sources.jar
printf javadoc > target/mosaic-1.2.3-javadoc.jar
MOCK

  cat > "$MOCK_BIN/jar" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
file=${2:?}
if [[ "$file" == *-sources.jar ]]; then
  printf '%s\n' org/apache/paimon/mosaic/NativeLib.java
  exit 0
fi
cat <<'ENTRIES'
org/apache/paimon/mosaic/NativeLib.class
native/linux/x86_64/libpaimon_mosaic_jni.so
native/linux/aarch64/libpaimon_mosaic_jni.so
native/macos/aarch64/libpaimon_mosaic_jni.dylib
native/windows/x86_64/paimon_mosaic_jni.dll
META-INF/LICENSE
META-INF/NOTICE
META-INF/DEPENDENCIES
META-INF/licenses/x86_64-unknown-linux-gnu/THIRD-PARTY-LICENSES.html
META-INF/licenses/aarch64-unknown-linux-gnu/THIRD-PARTY-LICENSES.html
META-INF/licenses/aarch64-apple-darwin/THIRD-PARTY-LICENSES.html
META-INF/licenses/x86_64-pc-windows-msvc/THIRD-PARTY-LICENSES.html
ENTRIES
if [[ "${OMIT_NATIVE_ENTRY:-0}" == 1 ]]; then
  exit 0
fi
MOCK
  # Rewrite the mock when an entry must be omitted; doing it here keeps the
  # normal listing easy to audit.
  cat > "$MOCK_BIN/java" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
printf 'java %s\n' "$*" >> "$MOCK_LOG"
MOCK
  cat > "$MOCK_BIN/python3" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
printf 'python3 %s\n' "$*" >> "$MOCK_LOG"
case "${1:-}" in
  */tools/verify_release_artifacts.py)
    exit 0
    ;;
  *)
    exec /usr/bin/python3 "$@"
    ;;
esac
MOCK
  cat > "$MOCK_BIN/file" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  *linux-x86_64*) echo "$1: ELF 64-bit LSB shared object, x86-64" ;;
  *linux-aarch64*) echo "$1: ELF 64-bit LSB shared object, ARM aarch64" ;;
  *macos-aarch64*) echo "$1: Mach-O 64-bit dynamically linked shared library arm64" ;;
  *windows-x86_64*) echo "$1: PE32+ executable (DLL) (console) x86-64" ;;
  *) exit 2 ;;
esac
MOCK
  chmod +x "$MOCK_BIN"/* "$FIXTURE/tools/deploy_java_staging.sh"

  git -C "$FIXTURE" init -q
  git -C "$FIXTURE" config user.name "Java Staging Test"
  git -C "$FIXTURE" config user.email "staging-test@example.com"
  git -C "$FIXTURE" add .
  git -C "$FIXTURE" commit -q -m initial
  git -C "$FIXTURE" tag v1.2.3-rc1
  HEAD_SHA=$(git -C "$FIXTURE" rev-parse HEAD)
  : > "$MOCK_LOG"
}

omit_native_from_mock_jar() {
  python3 - "$MOCK_BIN/jar" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
text = text.replace("native/windows/x86_64/paimon_mosaic_jni.dll\n", "")
path.write_text(text)
PY
}

run_stage() {
  env \
    PATH="$MOCK_BIN:$PATH" \
    MOCK_LOG="$MOCK_LOG" \
    MOCK_RUN_SHA="${MOCK_RUN_SHA:-$HEAD_SHA}" \
    MOCK_TAG=v1.2.3-rc1 \
    OMIT_CI_JAVADOC="${OMIT_CI_JAVADOC:-0}" \
    OMIT_LOCAL_MAIN="${OMIT_LOCAL_MAIN:-0}" \
    "$FIXTURE/tools/deploy_java_staging.sh" \
      --release-version 1.2.3 \
      --rc 1 \
      --run-id 12345 \
      --skip-native-file-check \
      "$@"
}

pass() {
  TESTS=$((TESTS + 1))
  echo "ok $TESTS - $1"
}

new_fixture
run_stage --dry-run > "$FIXTURE/output" 2>&1
for artifact in native-linux-x86_64 native-linux-aarch64 native-macos-aarch64 native-windows-x86_64 java-package; do
  assert_contains "$MOCK_LOG" "--name $artifact"
done
assert_contains "$MOCK_LOG" "mvn clean verify -Prelease -Dgpg.skip=true -DskipTests"
assert_not_contains "$MOCK_LOG" "mvn deploy"
[[ $(grep -c '^java ' "$MOCK_LOG") -eq 2 ]] || fail "dry-run must smoke local and CI JARs"
[[ $(grep -c '^python3 .*verify_release_artifacts.py java ' "$MOCK_LOG") -eq 2 ]] ||
  fail "dry-run must content-verify local and CI JARs"
pass "successful dry-run validates all five artifacts and both JARs"

new_fixture
settings="$FIXTURE/settings.xml"
printf '<settings/>\n' > "$settings"
run_stage --maven-settings "$settings" --staging-description "Mosaic RC staging" > "$FIXTURE/output" 2>&1
verify_line=$(grep -n '^mvn .*clean verify ' "$MOCK_LOG" | cut -d: -f1)
deploy_line=$(grep -n '^mvn .*deploy ' "$MOCK_LOG" | cut -d: -f1)
[[ -n "$verify_line" && -n "$deploy_line" && "$verify_line" -lt "$deploy_line" ]] || fail "verify must run before deploy"
assert_contains "$MOCK_LOG" "-s $settings clean verify"
assert_contains "$MOCK_LOG" "-s $settings deploy -Prelease -DstagingDescription=Mosaic RC staging"
pass "successful real run verifies before deploy and forwards Maven options"

new_fixture
MOCK_RUN_SHA=0000000000000000000000000000000000000000
if run_stage --dry-run > "$FIXTURE/output" 2>&1; then
  fail "wrong run SHA should fail"
fi
assert_not_contains "$MOCK_LOG" "mvn "
unset MOCK_RUN_SHA
new_fixture
printf '\n<!-- dirty -->\n' >> "$FIXTURE/java/pom.xml"
if run_stage --dry-run > "$FIXTURE/output" 2>&1; then
  fail "dirty Java input should fail"
fi
assert_not_contains "$MOCK_LOG" "mvn "
new_fixture
printf '\n# dirty\n' >> "$FIXTURE/tools/deploy_java_staging.sh"
if run_stage --dry-run > "$FIXTURE/output" 2>&1; then
  fail "dirty deploy script should fail"
fi
assert_not_contains "$MOCK_LOG" "mvn "
pass "wrong run SHA and dirty Java/script inputs fail before Maven"

new_fixture
OMIT_CI_JAVADOC=1
if run_stage > "$FIXTURE/output" 2>&1; then
  fail "missing CI Javadoc JAR should fail"
fi
assert_not_contains "$MOCK_LOG" "mvn deploy"
unset OMIT_CI_JAVADOC
new_fixture
omit_native_from_mock_jar
if run_stage > "$FIXTURE/output" 2>&1; then
  fail "missing native JAR entry should fail"
fi
assert_not_contains "$MOCK_LOG" "mvn deploy"
pass "missing required JAR or native entry blocks deploy"

echo "PASS: $TESTS focused Java staging tests"
