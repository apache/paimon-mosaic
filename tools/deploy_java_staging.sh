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
PYTHON=${PYTHON:-python3}
GPG=${GPG:-gpg}
CURL=${CURL:-curl}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

RELEASE_VERSION=
RC_NUMBER=
RUN_ID=
TAG=
REPOSITORY=apache/paimon-mosaic
DRY_RUN=false
RUN_TESTS=false
MAVEN_SETTINGS=
GPG_KEYNAME=
KEYS_FILE=
STAGING_DESCRIPTION=
BUILD_ROOT=
GITHUB_HOST=github.com
RELEASE_WORKFLOW_PATH=.github/workflows/release.yml

usage() {
  cat <<'EOF'
Usage:
  deploy_java_staging.sh --release-version VERSION --rc N --run-id RUN_ID [options]

Build, verify, sign, and deploy Apache Paimon Mosaic Java RC artifacts from an
isolated archive of the exact signed RC tag. The native inputs are downloaded
from the exact successful top-level Release workflow run.

Required:
  --release-version VERSION  Release version, for example 0.3.0.
  --rc N                     RC number, for example 1.
  --run-id RUN_ID            Successful RC tag-push Release workflow run id.

Options:
  --tag TAG                  Defaults to vVERSION-rcN.
  --repo OWNER/REPO          Defaults to apache/paimon-mosaic.
  --dry-run                  Run clean verify without signing or Nexus deploy.
  --maven-settings FILE      Maven settings file for Nexus credentials.
  --gpg-keyname FINGERPRINT  Full signing-key fingerprint passed to Maven.
  --keys-file FILE           ASF Paimon KEYS file; otherwise download it.
  --staging-description TXT  Nexus staging description.
  --run-tests                Run Maven tests.
  -h, --help                 Show this help.
EOF
}

require_option_value() {
  local option=$1
  local value=${2-}
  if [[ $# -lt 2 || -z "$value" || "$value" == -* ]]; then
    echo "$option requires a value that is not another option" >&2
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
    --run-id)
      require_option_value "$@"
      RUN_ID=$2
      shift 2
      ;;
    --tag)
      require_option_value "$@"
      TAG=$2
      shift 2
      ;;
    --repo)
      require_option_value "$@"
      REPOSITORY=$2
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
    --gpg-keyname)
      require_option_value "$@"
      GPG_KEYNAME=$2
      shift 2
      ;;
    --keys-file)
      require_option_value "$@"
      KEYS_FILE=$2
      shift 2
      ;;
    --staging-description)
      require_option_value "$@"
      STAGING_DESCRIPTION=$2
      shift 2
      ;;
    --run-tests)
      RUN_TESTS=true
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
  local option=$1
  local value=$2
  if [[ -z "$value" ]]; then
    echo "$option is required" >&2
    usage >&2
    exit 1
  fi
}

require_value "--release-version" "$RELEASE_VERSION"
require_value "--rc" "$RC_NUMBER"
require_value "--run-id" "$RUN_ID"

if [[ ! "$RELEASE_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "--release-version must use X.Y.Z numeric form: $RELEASE_VERSION" >&2
  exit 1
fi
if [[ ! "$RC_NUMBER" =~ ^[1-9][0-9]*$ ]]; then
  echo "--rc must be a positive integer: $RC_NUMBER" >&2
  exit 1
fi
if [[ ! "$RUN_ID" =~ ^[1-9][0-9]*$ ]]; then
  echo "--run-id must be a positive integer: $RUN_ID" >&2
  exit 1
fi

EXPECTED_TAG="v${RELEASE_VERSION}-rc${RC_NUMBER}"
if [[ -z "$TAG" ]]; then
  TAG=$EXPECTED_TAG
elif [[ "$TAG" != "$EXPECTED_TAG" ]]; then
  echo "--tag must match --release-version and --rc." >&2
  echo "tag:      $TAG" >&2
  echo "expected: $EXPECTED_TAG" >&2
  exit 1
fi

if [[ -z "$STAGING_DESCRIPTION" ]]; then
  STAGING_DESCRIPTION="Apache Paimon Mosaic ${RELEASE_VERSION} RC${RC_NUMBER}"
fi

if [[ "$DRY_RUN" != true && "$REPOSITORY" != apache/paimon-mosaic ]]; then
  echo "Real Nexus deployment requires a run from the official apache/paimon-mosaic repository." >&2
  exit 1
fi
if [[ "$DRY_RUN" != true ]]; then
  require_value "--gpg-keyname" "$GPG_KEYNAME"
  if [[ ! "$GPG_KEYNAME" =~ ^([0-9A-Fa-f]{40}|[0-9A-Fa-f]{64})$ ]]; then
    echo "--gpg-keyname must be a full 40- or 64-hex OpenPGP fingerprint." >&2
    exit 1
  fi
  GPG_KEYNAME=$(printf '%s' "$GPG_KEYNAME" | tr '[:lower:]' '[:upper:]')
fi

if [[ -n "$MAVEN_SETTINGS" ]]; then
  if [[ ! -f "$MAVEN_SETTINGS" ]]; then
    echo "--maven-settings does not exist: $MAVEN_SETTINGS" >&2
    exit 1
  fi
  MAVEN_SETTINGS=$(cd "$(dirname "$MAVEN_SETTINGS")" && pwd)/$(basename "$MAVEN_SETTINGS")
fi
if [[ -n "$KEYS_FILE" ]]; then
  if [[ ! -f "$KEYS_FILE" ]]; then
    echo "--keys-file does not exist: $KEYS_FILE" >&2
    exit 1
  fi
  KEYS_FILE=$(cd "$(dirname "$KEYS_FILE")" && pwd)/$(basename "$KEYS_FILE")
fi

for command in git gh tar "$PYTHON" "$MVN"; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command not found: $command" >&2
    exit 1
  fi
done
if [[ "$DRY_RUN" != true ]]; then
  required_release_commands=("$GPG")
  if [[ -z "$KEYS_FILE" ]]; then
    required_release_commands+=("$CURL")
  fi
  for command in "${required_release_commands[@]}"; do
    if ! command -v "$command" >/dev/null 2>&1; then
      echo "Required command not found: $command" >&2
      exit 1
    fi
  done
fi

git_exact() {
  GIT_NO_REPLACE_OBJECTS=1 GIT_ATTR_NOSYSTEM=1 git -C "$REPO_DIR" "$@"
}

gh_exact() {
  GH_HOST="$GITHUB_HOST" gh "$@"
}

REPLACEMENT_REFS=$(
  git_exact for-each-ref --format='%(refname)' refs/replace
)
if [[ -n "$REPLACEMENT_REFS" ]]; then
  echo "Git replacement refs are not allowed during Java staging." >&2
  printf '%s\n' "$REPLACEMENT_REFS" >&2
  exit 1
fi

ARCHIVE_ATTRIBUTES=$(git_exact rev-parse --git-path info/attributes)
case "$ARCHIVE_ATTRIBUTES" in
  /*) ;;
  *) ARCHIVE_ATTRIBUTES="$REPO_DIR/$ARCHIVE_ATTRIBUTES" ;;
esac
if [[ -s "$ARCHIVE_ATTRIBUTES" ]]; then
  echo "Non-empty repository-local Git attributes are not allowed during Java staging." >&2
  echo "$ARCHIVE_ATTRIBUTES" >&2
  exit 1
fi

if ! git_exact rev-parse -q --verify "$TAG^{commit}" >/dev/null; then
  echo "Tag $TAG does not exist locally." >&2
  exit 1
fi
TAG_COMMIT=$(git_exact rev-parse "$TAG^{commit}")
HEAD_COMMIT=$(git_exact rev-parse HEAD)
if [[ "$HEAD_COMMIT" != "$TAG_COMMIT" ]]; then
  echo "Current HEAD is not $TAG." >&2
  echo "Check out the RC tag before staging Java artifacts." >&2
  exit 1
fi

INDEX_FLAGGED_PATHS=$(
  git_exact ls-files -v |
    awk 'substr($0, 1, 1) == "S" || substr($0, 1, 1) ~ /[a-z]/'
)
if [[ -n "$INDEX_FLAGGED_PATHS" ]]; then
  echo "Git index flags such as assume-unchanged or skip-worktree are not allowed." >&2
  printf '%s\n' "$INDEX_FLAGGED_PATHS" >&2
  exit 1
fi

WORKTREE_STATUS=$(
  git_exact status --porcelain=v1 --untracked-files=all --ignored=matching
)
if [[ -n "$WORKTREE_STATUS" ]]; then
  echo "The RC-tag worktree must be completely clean before Java staging." >&2
  printf '%s\n' "$WORKTREE_STATUS" >&2
  exit 1
fi

validate_github_run() {
  local key
  local value
  local output
  local status=
  local conclusion=
  local head_sha=
  local head_branch=
  local workflow_name=
  local workflow_path=
  local event=

  output=$(
    gh_exact api \
      "repos/$REPOSITORY/actions/runs/$RUN_ID" \
      --jq '[
        "status=\(.status // "")",
        "conclusion=\(.conclusion // "")",
        "head_sha=\(.head_sha // "")",
        "head_branch=\(.head_branch // "")",
        "workflow_name=\(.name // "")",
        "workflow_path=\(.path // "")",
        "event=\(.event // "")"
      ] | .[]'
  )
  while IFS='=' read -r key value; do
    case "$key" in
      status) status=$value ;;
      conclusion) conclusion=$value ;;
      head_sha) head_sha=$value ;;
      head_branch) head_branch=$value ;;
      workflow_name) workflow_name=$value ;;
      workflow_path) workflow_path=$value ;;
      event) event=$value ;;
      *)
        echo "Unexpected GitHub Actions run field: $key" >&2
        exit 1
        ;;
    esac
  done <<EOF
$output
EOF

  if [[ "$status" != completed || "$conclusion" != success ]]; then
    echo "GitHub Actions run $RUN_ID is not successfully completed." >&2
    exit 1
  fi
  if [[ "$workflow_name" != Release || "$event" != push ]]; then
    echo "GitHub Actions run $RUN_ID is not a Release tag-push run." >&2
    exit 1
  fi
  if [[ "$workflow_path" != "$RELEASE_WORKFLOW_PATH" ]]; then
    echo "GitHub Actions run $RUN_ID does not use the canonical Release workflow." >&2
    echo "workflow path: $workflow_path" >&2
    echo "expected:      $RELEASE_WORKFLOW_PATH" >&2
    exit 1
  fi
  if [[ "$head_branch" != "$TAG" || "$head_sha" != "$TAG_COMMIT" ]]; then
    echo "GitHub Actions run $RUN_ID does not match $TAG at $TAG_COMMIT." >&2
    exit 1
  fi
}

validate_github_run

BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/paimon-mosaic-java-staging.XXXXXX")
cleanup() {
  if [[ -z "$BUILD_ROOT" ]]; then
    return
  fi
  case "$BUILD_ROOT" in
    "${TMPDIR:-/tmp}"/paimon-mosaic-java-staging.*)
      rm -rf -- "$BUILD_ROOT"
      ;;
    *)
      echo "Refusing to remove unexpected staging path: $BUILD_ROOT" >&2
      ;;
  esac
}
trap cleanup EXIT

validate_signing_key() {
  local keys_fingerprints
  local local_fingerprints

  if ! local_fingerprints=$(
    "$GPG" --batch --with-colons --list-secret-keys --fingerprint "$GPG_KEYNAME" |
      awk -F: '$1 == "fpr" {print toupper($10)}'
  ); then
    echo "Unable to read local secret key $GPG_KEYNAME." >&2
    exit 1
  fi
  if ! printf '%s\n' "$local_fingerprints" | grep -Fxq "$GPG_KEYNAME"; then
    echo "Local GPG secret keys do not contain $GPG_KEYNAME." >&2
    exit 1
  fi

  if [[ -z "$KEYS_FILE" ]]; then
    KEYS_FILE="$BUILD_ROOT/PAIMON_KEYS"
    "$CURL" \
      --proto '=https' \
      --tlsv1.2 \
      --location \
      --fail \
      --silent \
      --show-error \
      --output "$KEYS_FILE" \
      https://downloads.apache.org/paimon/KEYS
  fi
  if ! keys_fingerprints=$(
    "$GPG" \
      --batch \
      --with-colons \
      --import-options show-only \
      --import "$KEYS_FILE" |
      awk -F: '$1 == "fpr" {print toupper($10)}'
  ); then
    echo "Unable to read OpenPGP fingerprints from $KEYS_FILE." >&2
    exit 1
  fi
  if ! printf '%s\n' "$keys_fingerprints" | grep -Fxq "$GPG_KEYNAME"; then
    echo "Signing key $GPG_KEYNAME is not present in the ASF Paimon KEYS file." >&2
    exit 1
  fi
}

if [[ "$DRY_RUN" != true ]]; then
  validate_signing_key
  "$PYTHON" "$REPO_DIR/tools/validate_release_tag.py" \
    "$TAG" \
    --keys-file "$KEYS_FILE" \
    --repository "$REPO_DIR" \
    --expected-commit "$TAG_COMMIT"
fi

ARCHIVE_PREFIX="paimon-mosaic-${RELEASE_VERSION}"
ARCHIVE_PATH="$BUILD_ROOT/source.tar"
GIT_NO_REPLACE_OBJECTS=1 GIT_ATTR_NOSYSTEM=1 \
  git -C "$REPO_DIR" \
    -c core.attributesFile=/dev/null \
    archive \
    --format=tar \
    "--prefix=${ARCHIVE_PREFIX}/" \
    "$TAG_COMMIT" > "$ARCHIVE_PATH"
tar -xf "$ARCHIVE_PATH" -C "$BUILD_ROOT"
BUILD_REPO_DIR="$BUILD_ROOT/$ARCHIVE_PREFIX"

POM_VERSION=$(
  "$PYTHON" - "$BUILD_REPO_DIR/java/pom.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

pom = ET.parse(sys.argv[1]).getroot()
namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
version = pom.findtext("m:version", namespaces=namespace)
if version is None:
    version = pom.findtext("version")
if not version:
    raise ValueError("java/pom.xml does not define a project version")
print(version)
PY
)
if [[ "$POM_VERSION" != "$RELEASE_VERSION" ]]; then
  echo "RC tag java/pom.xml version is $POM_VERSION, expected $RELEASE_VERSION" >&2
  exit 1
fi

NATIVE_DIR="$BUILD_REPO_DIR/java/src/main/resources/native"
mkdir -p "$NATIVE_DIR"
gh_exact run download "$RUN_ID" \
  --repo "$REPOSITORY" \
  --name java-release-native-inputs \
  --dir "$NATIVE_DIR"

EXPECTED_NATIVE_FILES=$(cat <<'EOF'
linux/aarch64/libpaimon_mosaic_jni.so
linux/x86_64/libpaimon_mosaic_jni.so
macos/aarch64/libpaimon_mosaic_jni.dylib
windows/x86_64/paimon_mosaic_jni.dll
EOF
)
ACTUAL_NATIVE_FILES=$(
  cd "$NATIVE_DIR"
  find . -type f |
    sed 's#^\./##' |
    LC_ALL=C sort
)
if [[ "$ACTUAL_NATIVE_FILES" != "$EXPECTED_NATIVE_FILES" ]]; then
  echo "Downloaded Java native inputs differ from the four expected files." >&2
  echo "Expected:" >&2
  printf '%s\n' "$EXPECTED_NATIVE_FILES" >&2
  echo "Actual:" >&2
  printf '%s\n' "$ACTUAL_NATIVE_FILES" >&2
  exit 1
fi
if find "$NATIVE_DIR" -type l | grep -q .; then
  echo "Downloaded Java native inputs must not contain symbolic links." >&2
  exit 1
fi

"$PYTHON" - "$BUILD_REPO_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "tools"))
from native_binary import verify_native_target
from verify_java_jars import NATIVE_ENTRIES

native_root = root / "java/src/main/resources"
for archive_path, target in NATIVE_ENTRIES.items():
    path = native_root / archive_path
    verify_native_target(path.read_bytes(), target, archive_path)
print("Verified four Java native inputs.")
PY

MAVEN_CMD=("$MVN")
if [[ -n "$MAVEN_SETTINGS" ]]; then
  MAVEN_CMD+=("-s" "$MAVEN_SETTINGS")
elif [[ "$DRY_RUN" != true ]]; then
  MAVEN_CMD+=("-s" "$BUILD_REPO_DIR/deploysettings.xml")
fi

if [[ "$DRY_RUN" == true ]]; then
  MAVEN_CMD+=(clean verify -Prelease -Dexec.skip=false -Dgpg.skip=true)
else
  MAVEN_CMD+=(
    clean deploy -Prelease
    -Dexec.skip=false
    -Dgpg.skip=false
    -DskipLocalStaging=false
    -DskipNexusStagingDeployMojo=false
    -DskipRemoteStaging=false
    -DskipStaging=false
    -DskipStagingRepositoryClose=false
    -Dmaven.wagon.http.ssl.allowall=false
    -Dmaven.wagon.http.ssl.insecure=false
    "-DstagingDescription=$STAGING_DESCRIPTION"
  )
fi
if [[ "$RUN_TESTS" != true ]]; then
  MAVEN_CMD+=(-DskipTests)
fi
if [[ "$DRY_RUN" != true ]]; then
  MAVEN_CMD+=("-Dgpg.keyname=${GPG_KEYNAME}!")
fi

(
  cd "$BUILD_REPO_DIR/java"
  MAVEN_ARGS= "${MAVEN_CMD[@]}"
)

if [[ "$DRY_RUN" == true ]]; then
  echo "Java staging dry run finished successfully."
else
  for artifact in \
    "$BUILD_REPO_DIR/java/target/mosaic-${POM_VERSION}.jar" \
    "$BUILD_REPO_DIR/java/target/mosaic-${POM_VERSION}-sources.jar" \
    "$BUILD_REPO_DIR/java/target/mosaic-${POM_VERSION}-javadoc.jar" \
    "$BUILD_REPO_DIR/java/target/mosaic-${POM_VERSION}.pom"; do
    signature="${artifact}.asc"
    if [[ ! -f "$artifact" || ! -f "$signature" ]]; then
      echo "Missing signed Maven artifact pair: $artifact and $signature" >&2
      exit 1
    fi
    valid_fingerprint=$(
      "$GPG" --batch --status-fd 1 --verify "$signature" "$artifact" 2>/dev/null |
        awk '$1 == "[GNUPG:]" && $2 == "VALIDSIG" {print toupper($3); exit}'
    )
    if [[ "$valid_fingerprint" != "$GPG_KEYNAME" ]]; then
      echo "Unexpected signer for $artifact: ${valid_fingerprint:-none}" >&2
      exit 1
    fi
  done
  echo "Java staging deploy finished."
fi
