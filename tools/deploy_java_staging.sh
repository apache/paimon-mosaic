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

MVN=${MVN:-mvn}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

RELEASE_VERSION=
RC_NUMBER=
TAG=
NATIVE_DIR=
RUN_ID=
DRY_RUN=false
SKIP_TESTS=true
CLEANUP_NATIVE_RESOURCES=true
MAVEN_SETTINGS=
STAGING_DESCRIPTION=
CHECK_NATIVE_FILES=true

usage() {
  cat <<'EOF'
Usage:
  deploy_java_staging.sh --release-version VERSION --rc N --run-id RUN_ID [options]

Deploy Apache Paimon Mosaic Java RC artifacts to Apache Nexus staging from a
committer/RM machine. Pass the GitHub Actions run id that built the RC native
libraries; the script verifies the run, downloads the native artifacts, and then
runs the local Maven deploy.

Required:
  --release-version VERSION  Release version in java/pom.xml, for example 0.3.0.
  --rc N                     RC number, for example 1 for v0.3.0-rc1.
  --run-id RUN_ID            GitHub Actions run id containing native-* artifacts.

Options:
  --tag TAG                  RC tag. Defaults to vVERSION-rcN.
  --repo REPO                GitHub repository. Defaults to apache/paimon-mosaic.
  --dry-run                  Build and verify release artifacts locally only.
                             Does not sign or deploy to Nexus.
  --maven-settings FILE      Maven settings.xml containing apache.releases.https.
  --staging-description TXT  Nexus staging description.
  --no-skip-tests            Run Maven tests.
  --no-cleanup               Keep java/src/main/resources/native after exit.
  --skip-native-file-check   Do not check native binary file formats.
  -h, --help                 Show this help.

Validate with the real RC artifacts before publishing:
  ./tools/deploy_java_staging.sh --release-version 0.3.0 --rc 1 \
    --run-id 12345678901 --dry-run

Publish staging after the dry run succeeds:
  ./tools/deploy_java_staging.sh --release-version 0.3.0 --rc 1 \
    --run-id 12345678901

Maven/GPG requirements:
  Real deploy uses the committer's local GPG setup and Maven credentials for
  server id apache.releases.https. Configure ~/.m2/settings.xml, pass
  --maven-settings FILE, or set NEXUS_STAGE_DEPLOYER_USER and
  NEXUS_STAGE_DEPLOYER_PW for a temporary settings.xml.

gh CLI requirement:
  --run-id uses the GitHub CLI to check the workflow run and fetch native
  artifacts. Run `gh auth login` first.
EOF
}

require_option_value() {
  local option=$1
  local value=${2-}
  if [[ $# -lt 2 || -z "$value" ]]; then
    echo "$option requires a value" >&2
    usage >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release-version)
      require_option_value "$@"
      RELEASE_VERSION=$2
      shift 2
      ;;
    --rc)
      require_option_value "$@"
      RC_NUMBER=$2
      shift 2
      ;;
    --tag)
      require_option_value "$@"
      TAG=$2
      shift 2
      ;;
    --run-id)
      require_option_value "$@"
      RUN_ID=$2
      shift 2
      ;;
    --repo)
      require_option_value "$@"
      REPO=$2
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --maven-settings)
      require_option_value "$@"
      MAVEN_SETTINGS=$2
      shift 2
      ;;
    --staging-description)
      require_option_value "$@"
      STAGING_DESCRIPTION=$2
      shift 2
      ;;
    --no-skip-tests)
      SKIP_TESTS=false
      shift
      ;;
    --no-cleanup)
      CLEANUP_NATIVE_RESOURCES=false
      shift
      ;;
    --skip-native-file-check)
      CHECK_NATIVE_FILES=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_value() {
  local name=$1
  local value=$2
  if [[ -z "$value" ]]; then
    echo "$name is required" >&2
    usage >&2
    exit 1
  fi
}

REPO=${REPO:-apache/paimon-mosaic}

require_value "--release-version" "$RELEASE_VERSION"

if [[ -z "$TAG" ]]; then
  require_value "--rc" "$RC_NUMBER"
  TAG="v${RELEASE_VERSION}-rc${RC_NUMBER}"
fi

if [[ -z "$STAGING_DESCRIPTION" ]]; then
  if [[ -n "$RC_NUMBER" ]]; then
    STAGING_DESCRIPTION="Apache Paimon Mosaic, version ${RELEASE_VERSION}, release candidate ${RC_NUMBER}"
  else
    STAGING_DESCRIPTION="Apache Paimon Mosaic, version ${RELEASE_VERSION}, release candidate ${TAG#*-rc}"
  fi
fi

if [[ -z "$RUN_ID" ]]; then
  echo "--run-id is required" >&2
  usage >&2
  exit 1
fi

NATIVE_DIR="$SCRIPT_DIR/release/java-native-${TAG}"

if [[ -n "$MAVEN_SETTINGS" && ! -f "$MAVEN_SETTINGS" ]]; then
  echo "--maven-settings does not exist: $MAVEN_SETTINGS" >&2
  exit 1
fi

POM_VERSION=$(
  sed -n 's#.*<version>\([^<]*\)</version>.*#\1#p' "$REPO_DIR/java/pom.xml" |
    sed -n '2p'
)
if [[ "$POM_VERSION" != "$RELEASE_VERSION" ]]; then
  echo "java/pom.xml version is $POM_VERSION, expected $RELEASE_VERSION" >&2
  echo "Check out the RC tag after bumping versions, then run this script again." >&2
  exit 1
fi

if ! git -C "$REPO_DIR" rev-parse -q --verify "$TAG^{commit}" >/dev/null; then
  echo "Tag $TAG does not exist locally." >&2
  echo "Run: git fetch --tags && git checkout $TAG" >&2
  exit 1
else
  TAG_COMMIT=$(git -C "$REPO_DIR" rev-parse "$TAG^{commit}")
  HEAD_COMMIT=$(git -C "$REPO_DIR" rev-parse HEAD)
  if [[ "$TAG_COMMIT" != "$HEAD_COMMIT" ]]; then
    echo "Current HEAD is not $TAG." >&2
    echo "Run: git checkout $TAG" >&2
    exit 1
  fi
fi

check_java_package_inputs_clean() {
  local paths=(java DEPENDENCIES.rust.tsv)
  local untracked

  if ! git -C "$REPO_DIR" diff --quiet -- "${paths[@]}" ||
     ! git -C "$REPO_DIR" diff --cached --quiet -- "${paths[@]}"; then
    echo "Java package inputs have local changes. Commit or revert them before publishing." >&2
    git -C "$REPO_DIR" status --short -- "${paths[@]}" >&2
    exit 1
  fi

  untracked=$(git -C "$REPO_DIR" ls-files --others --exclude-standard -- "${paths[@]}")
  if [[ -n "$untracked" ]]; then
    echo "Java package inputs contain untracked files. Remove or commit them before publishing." >&2
    printf '%s\n' "$untracked" >&2
    exit 1
  fi
}

check_java_package_inputs_clean

validate_github_run() {
  local run_output
  local run_info
  if ! run_output=$(
    gh run view "$RUN_ID" \
      --repo "$REPO" \
      --json status,conclusion,headSha,headBranch,workflowName,event \
      --template '{{printf "%s\n%s\n%s\n%s\n%s\n%s\n" .status .conclusion .headSha (or .headBranch "") (or .workflowName "") (or .event "")}}'
  ); then
    echo "Failed to read GitHub Actions run: $RUN_ID" >&2
    exit 1
  fi
  mapfile -t run_info <<<"$run_output"

  local run_status=${run_info[0]:-}
  local run_conclusion=${run_info[1]:-}
  local run_head_sha=${run_info[2]:-}
  local run_head_branch=${run_info[3]:-}
  local run_workflow_name=${run_info[4]:-}
  local run_event=${run_info[5]:-}

  if [[ "$run_status" != "completed" || "$run_conclusion" != "success" ]]; then
    echo "GitHub Actions run $RUN_ID is not a successful completed run." >&2
    echo "status=$run_status conclusion=$run_conclusion" >&2
    exit 1
  fi

  if [[ "$run_workflow_name" != "Release" ]]; then
    echo "GitHub Actions run $RUN_ID is from workflow '$run_workflow_name', expected 'Release'." >&2
    exit 1
  fi

  if [[ "$run_event" != "push" ]]; then
    echo "GitHub Actions run $RUN_ID was triggered by '$run_event', expected a tag push." >&2
    exit 1
  fi

  if [[ "$run_head_sha" != "$TAG_COMMIT" ]]; then
    echo "GitHub Actions run $RUN_ID does not match $TAG." >&2
    echo "run headSha: $run_head_sha" >&2
    echo "tag commit:  $TAG_COMMIT" >&2
    exit 1
  fi

  echo "Using GitHub Actions run $RUN_ID for native artifacts:"
  echo "  workflow: ${run_workflow_name:-unknown}"
  echo "  event:    ${run_event:-unknown}"
  echo "  ref:      ${run_head_branch:-unknown}"
  echo "  headSha:  ${run_head_sha:-unknown}"
}

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required when --run-id is used" >&2
  exit 1
fi

validate_github_run

rm -rf "$NATIVE_DIR"
mkdir -p "$NATIVE_DIR"
for artifact in \
  native-linux-x86_64 \
  native-linux-aarch64 \
  native-macos-aarch64 \
  native-windows-x86_64
do
  gh run download "$RUN_ID" \
    --repo "$REPO" \
    --name "$artifact" \
    --dir "$NATIVE_DIR/$artifact"
done

if [[ ! -d "$NATIVE_DIR" ]]; then
  echo "Native artifact download directory does not exist: $NATIVE_DIR" >&2
  exit 1
fi

find_native() {
  local artifact_layout=$1
  local resource_layout=$2
  local resource_without_native=${resource_layout#native/}

  for candidate in \
    "$NATIVE_DIR/$artifact_layout" \
    "$NATIVE_DIR/$resource_layout" \
    "$NATIVE_DIR/$resource_without_native"
  do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "Missing native artifact. Tried:" >&2
  echo "  $NATIVE_DIR/$artifact_layout" >&2
  echo "  $NATIVE_DIR/$resource_layout" >&2
  echo "  $NATIVE_DIR/$resource_without_native" >&2
  exit 1
}

validate_native_file() {
  local source_file=$1
  local label=$2

  if [[ "$CHECK_NATIVE_FILES" != "true" ]]; then
    return
  fi

  if ! command -v file >/dev/null 2>&1; then
    echo "WARNING: 'file' command not found; skipping native file format checks." >&2
    return
  fi

  local info
  info=$(file "$source_file")
  case "$label" in
    linux-x86_64)
      if ! grep -Eq 'ELF 64-bit.*(x86-64|x86_64)' <<<"$info"; then
        echo "Unexpected linux x86_64 native file: $info" >&2
        exit 1
      fi
      ;;
    linux-aarch64)
      if ! grep -Eq 'ELF 64-bit.*(ARM aarch64|AArch64|aarch64|ARM64)' <<<"$info"; then
        echo "Unexpected linux aarch64 native file: $info" >&2
        exit 1
      fi
      ;;
    macos-aarch64)
      if ! grep -Eq 'Mach-O 64-bit.*(arm64|aarch64)' <<<"$info"; then
        echo "Unexpected macOS aarch64 native file: $info" >&2
        exit 1
      fi
      ;;
    windows-x86_64)
      if ! grep -Eq 'PE32\+.*(x86-64|x86_64)' <<<"$info"; then
        echo "Unexpected windows x86_64 native file: $info" >&2
        exit 1
      fi
      ;;
    *)
      echo "Unknown native file label: $label" >&2
      exit 1
      ;;
  esac
}

copy_native() {
  local source_file=$1
  local target_rel=$2
  local label=$3
  local target_file="$REPO_DIR/java/src/main/resources/$target_rel"

  validate_native_file "$source_file" "$label"
  mkdir -p "$(dirname "$target_file")"
  cp "$source_file" "$target_file"
}

cleanup_native_resources() {
  if [[ "$CLEANUP_NATIVE_RESOURCES" == "true" ]]; then
    rm -rf "$REPO_DIR/java/src/main/resources/native"
  fi
}

TEMP_SETTINGS=
cleanup_temp_settings() {
  if [[ -n "$TEMP_SETTINGS" ]]; then
    rm -f "$TEMP_SETTINGS"
  fi
}

xml_escape() {
  printf '%s' "$1" |
    sed \
      -e 's/&/\&amp;/g' \
      -e 's/</\&lt;/g' \
      -e 's/>/\&gt;/g'
}

cleanup_all() {
  cleanup_native_resources
  cleanup_temp_settings
}
trap cleanup_all EXIT

rm -rf "$REPO_DIR/java/src/main/resources/native"

copy_native \
  "$(find_native native-linux-x86_64/libpaimon_mosaic_jni.so native/linux/x86_64/libpaimon_mosaic_jni.so)" \
  native/linux/x86_64/libpaimon_mosaic_jni.so \
  linux-x86_64
copy_native \
  "$(find_native native-linux-aarch64/libpaimon_mosaic_jni.so native/linux/aarch64/libpaimon_mosaic_jni.so)" \
  native/linux/aarch64/libpaimon_mosaic_jni.so \
  linux-aarch64
copy_native \
  "$(find_native native-macos-aarch64/libpaimon_mosaic_jni.dylib native/macos/aarch64/libpaimon_mosaic_jni.dylib)" \
  native/macos/aarch64/libpaimon_mosaic_jni.dylib \
  macos-aarch64
copy_native \
  "$(find_native native-windows-x86_64/paimon_mosaic_jni.dll native/windows/x86_64/paimon_mosaic_jni.dll)" \
  native/windows/x86_64/paimon_mosaic_jni.dll \
  windows-x86_64

echo "Native libraries staged for Java package:"
find "$REPO_DIR/java/src/main/resources/native" -type f | sort

if [[ "$DRY_RUN" != "true" &&
      -z "$MAVEN_SETTINGS" &&
      ( -n "${NEXUS_STAGE_DEPLOYER_USER:-}" || -n "${NEXUS_STAGE_DEPLOYER_PW:-}" ) ]]; then
  if [[ -z "${NEXUS_STAGE_DEPLOYER_USER:-}" || -z "${NEXUS_STAGE_DEPLOYER_PW:-}" ]]; then
    echo "Both NEXUS_STAGE_DEPLOYER_USER and NEXUS_STAGE_DEPLOYER_PW are required" >&2
    exit 1
  fi

  TEMP_SETTINGS=$(mktemp)
  NEXUS_STAGE_DEPLOYER_USER_XML=$(xml_escape "$NEXUS_STAGE_DEPLOYER_USER")
  NEXUS_STAGE_DEPLOYER_PW_XML=$(xml_escape "$NEXUS_STAGE_DEPLOYER_PW")
  cat > "$TEMP_SETTINGS" <<EOF
<settings>
  <servers>
    <server>
      <id>apache.releases.https</id>
      <username>${NEXUS_STAGE_DEPLOYER_USER_XML}</username>
      <password>${NEXUS_STAGE_DEPLOYER_PW_XML}</password>
    </server>
  </servers>
</settings>
EOF
  MAVEN_SETTINGS="$TEMP_SETTINGS"
fi

MVN_BASE_CMD=("$MVN")
if [[ -n "$MAVEN_SETTINGS" ]]; then
  MVN_BASE_CMD+=("-s" "$MAVEN_SETTINGS")
fi

VERIFY_CMD=("${MVN_BASE_CMD[@]}" clean verify -Prelease -Dgpg.skip=true)
if [[ "$SKIP_TESTS" == "true" ]]; then
  VERIFY_CMD+=(-DskipTests)
fi

validate_maven_artifacts() {
  local jar_file="$REPO_DIR/java/target/mosaic-${RELEASE_VERSION}.jar"
  local sources_jar="$REPO_DIR/java/target/mosaic-${RELEASE_VERSION}-sources.jar"
  local javadoc_jar="$REPO_DIR/java/target/mosaic-${RELEASE_VERSION}-javadoc.jar"
  local artifact
  local native_entry

  for artifact in "$jar_file" "$sources_jar" "$javadoc_jar"; do
    if [[ ! -f "$artifact" ]]; then
      echo "Expected Maven artifact is missing: $artifact" >&2
      exit 1
    fi
  done

  for native_entry in \
    native/linux/x86_64/libpaimon_mosaic_jni.so \
    native/linux/aarch64/libpaimon_mosaic_jni.so \
    native/macos/aarch64/libpaimon_mosaic_jni.dylib \
    native/windows/x86_64/paimon_mosaic_jni.dll
  do
    if ! jar tf "$jar_file" | grep -qx "$native_entry"; then
      echo "Packaged jar is missing native entry: $native_entry" >&2
      exit 1
    fi
  done
}

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry-running Java staging build. No artifacts will be deployed to Nexus."
else
  echo "Running Java staging preflight before deploying to Apache Nexus."
  echo "Staging description: $STAGING_DESCRIPTION"
fi

(
  cd "$REPO_DIR/java"
  "${VERIFY_CMD[@]}"
)

validate_maven_artifacts

echo ""
if [[ "$DRY_RUN" == "true" ]]; then
  echo "Java staging dry run finished successfully."
else
  DEPLOY_CMD=("${MVN_BASE_CMD[@]}" deploy -Prelease "-DstagingDescription=$STAGING_DESCRIPTION")
  if [[ "$SKIP_TESTS" == "true" ]]; then
    DEPLOY_CMD+=(-DskipTests)
  fi

  echo "Preflight passed. Deploying Java artifacts to Apache Nexus staging."
  (
    cd "$REPO_DIR/java"
    "${DEPLOY_CMD[@]}"
  )
  validate_maven_artifacts

  echo ""
  echo "Java staging deploy finished."
  echo "Check the Maven output for the orgapachepaimon-XXXX staging repository id."
fi
