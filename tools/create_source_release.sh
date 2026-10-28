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

# Create ASF source release artifacts under tools/release/{RC_TAG}/{commit}/:
#   apache-paimon-mosaic-{version}-src.tgz
#   apache-paimon-mosaic-{version}-src.tgz.asc
#   apache-paimon-mosaic-{version}-src.tgz.sha512
#
# Usage:
#   cd tools
#   RELEASE_VERSION=0.3.0 RC_TAG=v0.3.0-rc1 ./create_source_release.sh
#
# Diagnostics go to stderr. Only a successful run prints the verified output
# directory to stdout, for capture by the explicit SVN upload command.

# fail immediately
set -o errexit
set -o nounset
set -o pipefail

# Keep stdout empty on every failure, including failures in child commands.
exec 3>&1 1>&2

# Do not let inherited Git variables redirect checks or replace signature
# verification.
unset \
  GIT_ALTERNATE_OBJECT_DIRECTORIES \
  GIT_COMMON_DIR \
  GIT_CONFIG_COUNT \
  GIT_CONFIG_GLOBAL \
  GIT_CONFIG_PARAMETERS \
  GIT_CONFIG_SYSTEM \
  GIT_DIR \
  GIT_INDEX_FILE \
  GIT_NAMESPACE \
  GIT_OBJECT_DIRECTORY \
  GIT_WORK_TREE
export GIT_ATTR_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1
export GIT_NO_REPLACE_OBJECTS=1

CURR_DIR=$(pwd -P)
if [[ $(basename "${CURR_DIR}") != "tools" ]] ; then
  echo "You have to call the script from the tools/ dir"
  exit 1
fi

RELEASE_VERSION=${RELEASE_VERSION:-}
RC_TAG=${RC_TAG:-}
if [[ -z "${RELEASE_VERSION}" ]]; then
  echo "RELEASE_VERSION is unset" >&2
  exit 1
fi
if [[ -z "${RC_TAG}" ]]; then
  echo "RC_TAG is unset" >&2
  exit 1
fi
if [[ ! "${RC_TAG}" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)-rc[0-9]+$ ]]; then
  echo "RC_TAG must have the form vX.Y.Z-rcN: ${RC_TAG}" >&2
  exit 1
fi
if [[ "${BASH_REMATCH[1]}" != "${RELEASE_VERSION}" ]]; then
  echo \
    "RC_TAG ${RC_TAG} does not match RELEASE_VERSION ${RELEASE_VERSION}" \
    >&2
  exit 1
fi

cd ..
REPOSITORY=$(pwd -P)

if ! python3 -c 'import tomllib' >/dev/null 2>&1; then
  echo "Source release checks require python3 with tomllib (Python 3.11+). Select that interpreter as python3 and retry." >&2
  exit 1
fi

if [[ -n $(git status --porcelain --untracked-files=all) ]]; then
  echo "The source release must be created from a clean Git worktree" >&2
  git status --short >&2
  exit 1
fi

HEAD_COMMIT=$(git rev-parse --verify 'HEAD^{commit}')
TAG_OBJECT_ID=$(git rev-parse --verify "${RC_TAG}^{tag}")
TAG_OBJECT=$(git cat-file tag "${TAG_OBJECT_ID}")
TAG_OBJECT_NAMES=$(
  awk '
    BEGIN { separator = "" }
    /^$/ { exit }
    /^tag / {
      printf "%s%s", separator, substr($0, 5)
      separator = ", "
    }
    END {
      if (separator == "") {
        printf "<missing>"
      }
    }
  ' <<< "${TAG_OBJECT}"
)
if [[ "${TAG_OBJECT_NAMES}" != "${RC_TAG}" ]]; then
  echo \
    "signed tag object names ${TAG_OBJECT_NAMES}, not ${RC_TAG}" \
    >&2
  exit 1
fi
TAG_COMMIT=$(git rev-parse --verify "${TAG_OBJECT_ID}^{commit}")
if [[ "${TAG_COMMIT}" != "${HEAD_COMMIT}" ]]; then
  echo \
    "RC_TAG ${RC_TAG} does not resolve to current HEAD ${HEAD_COMMIT}" \
    >&2
  exit 1
fi
if ! grep -q '^-----BEGIN PGP SIGNATURE-----$' <<< "${TAG_OBJECT}" ||
  ! git -c gpg.program=gpg verify-tag "${TAG_OBJECT_ID}"
then
  echo \
    "RC_TAG ${RC_TAG} is not a locally verifiable GPG-signed tag" \
    >&2
  exit 1
fi

ARCHIVE="apache-paimon-mosaic-${RELEASE_VERSION}-src.tgz"
SIGNATURE="${ARCHIVE}.asc"
CHECKSUM="${ARCHIVE}.sha512"
ARCHIVE_ROOT="paimon-mosaic-${RELEASE_VERSION}"
RELEASE_DIR="${CURR_DIR}/release/${RC_TAG}/${TAG_COMMIT}"

if [[ -e "${RELEASE_DIR}" || -L "${RELEASE_DIR}" ]]; then
  echo "Release output already exists and will not be overwritten: ${RELEASE_DIR}" >&2
  exit 1
fi

STAGING_DIR=$(mktemp -d "${CURR_DIR}/.source-release.XXXXXX")
ARTIFACT_DIR="${STAGING_DIR}/release"
mkdir "${ARTIFACT_DIR}"
cleanup() {
  status=$?
  rm -rf "${STAGING_DIR}"
  exit "${status}"
}
trap cleanup EXIT

echo "Creating source package from signed tag ${RC_TAG}"
python3 tools/verify_source_archive.py create \
  --repository "${REPOSITORY}" \
  --commit "${TAG_COMMIT}" \
  --prefix "${ARCHIVE_ROOT}/" \
  --output "${ARTIFACT_DIR}/${ARCHIVE}"

mkdir "${STAGING_DIR}/extracted"
tar xzf "${ARTIFACT_DIR}/${ARCHIVE}" -C "${STAGING_DIR}/extracted"
# Bash 3.2 does not reliably propagate errexit through subshells, so preserve
# the explicit status checks in both verification blocks.
if (
  cd "${STAGING_DIR}/extracted/${ARCHIVE_ROOT}" &&
    python3 tools/verify_release_versions.py "${RC_TAG}"
); then
  :
else
  status=$?
  exit "${status}"
fi

gpg \
  --armor \
  --detach-sign \
  --output "${ARTIFACT_DIR}/${SIGNATURE}" \
  "${ARTIFACT_DIR}/${ARCHIVE}"

python3 tools/release_checks.py signatures \
  --signed-file "${ARTIFACT_DIR}/${SIGNATURE}" "${ARTIFACT_DIR}/${ARCHIVE}"

if (
  cd "${ARTIFACT_DIR}" &&
  if [[ $(uname) == "Darwin" ]]; then
    shasum -a 512 "${ARCHIVE}" > "${CHECKSUM}" &&
    shasum -a 512 -c "${CHECKSUM}"
  else
    sha512sum "${ARCHIVE}" > "${CHECKSUM}" &&
    sha512sum -c "${CHECKSUM}"
  fi &&
    gpg --verify "${SIGNATURE}" "${ARCHIVE}"
); then
  :
else
  status=$?
  exit "${status}"
fi

python3 tools/verify_source_archive.py verify \
  --repository "${REPOSITORY}" \
  --commit "${TAG_COMMIT}" \
  --prefix "${ARCHIVE_ROOT}/" \
  --archive "${ARTIFACT_DIR}/${ARCHIVE}"

# The staging and release directories share a filesystem, so this publishes
# the fully verified artifact set with one atomic directory rename.
mkdir -p "$(dirname "${RELEASE_DIR}")"
python3 -c \
  'import os, sys; os.rename(sys.argv[1], sys.argv[2])' \
  "${ARTIFACT_DIR}" \
  "${RELEASE_DIR}"

trap - EXIT
rm -rf "${STAGING_DIR}"

echo ""
echo "Source release created successfully. Artifacts in ${RELEASE_DIR}:"
ls -la "${RELEASE_DIR}"/apache-paimon-mosaic-*
echo ""
echo "Next: upload contents to SVN (see docs/creating-a-release.html)."
printf '%s\n' "${RELEASE_DIR}" >&3
