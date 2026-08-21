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

# Create ASF source release artifacts under tools/release/:
#   apache-paimon-mosaic-{version}-src.tgz
#   apache-paimon-mosaic-{version}-src.tgz.asc
#   apache-paimon-mosaic-{version}-src.tgz.sha512
#
# Usage: cd tools && RELEASE_VERSION=0.1.0 ./create_source_release.sh

##
## Variables with defaults (if not overwritten by environment)
##
# fail immediately
set -o errexit
set -o nounset
set -o pipefail
# print command before executing
set -o xtrace

CURR_DIR=$(pwd -P)
if [[ $(basename "${CURR_DIR}") != "tools" ]] ; then
  echo "You have to call the script from the tools/ dir"
  exit 1
fi

if [ "$(uname)" == "Darwin" ]; then
    SHASUM="shasum -a 512"
else
    SHASUM="sha512sum"
fi

###########################

RELEASE_VERSION=${RELEASE_VERSION:-}
if [[ -z "${RELEASE_VERSION}" ]]; then
  echo "RELEASE_VERSION is unset" >&2
  exit 1
fi

cd ..

INDEX_FLAGGED_PATHS=$(
  git ls-files -v |
    awk 'substr($0, 1, 1) == "S" || substr($0, 1, 1) ~ /[a-z]/'
)
if [[ -n "${INDEX_FLAGGED_PATHS}" ]]; then
  echo "Git index flags such as assume-unchanged or skip-worktree are not allowed." >&2
  printf '%s\n' "${INDEX_FLAGGED_PATHS}" >&2
  exit 1
fi

if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  echo "The source release must be created from a clean Git worktree" >&2
  git status --short >&2
  exit 1
fi

git rev-parse --verify 'HEAD^{commit}' > /dev/null

rm -rf tools/release
mkdir tools/release

echo "Creating source package"

ARCHIVE="apache-paimon-mosaic-${RELEASE_VERSION}-src.tgz"
ARCHIVE_PATH="tools/release/${ARCHIVE}"
ARCHIVE_ROOT="paimon-mosaic-${RELEASE_VERSION}"
python3 tools/verify_source_archive.py create \
  --repository . \
  --commit HEAD \
  --prefix "${ARCHIVE_ROOT}/" \
  --output "${ARCHIVE_PATH}"

CHECK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/paimon-source-check.XXXXXX")
trap 'rm -rf "${CHECK_DIR}"' EXIT
tar xzf "${ARCHIVE_PATH}" -C "${CHECK_DIR}"
(
  cd "${CHECK_DIR}/${ARCHIVE_ROOT}"
  python3 tools/verify_release_versions.py "${RELEASE_VERSION}"

  echo "Verifying locked dependencies and generated legal metadata"
  cargo metadata --locked --format-version 1 --no-deps > /dev/null
  python3 tools/dependencies.py check
  python3 tools/generate_license_reports.py --check
)
rm -rf "${CHECK_DIR}"
trap - EXIT

cd tools/release

gpg --armor --detach-sig "${ARCHIVE}"
$SHASUM "${ARCHIVE}" > "${ARCHIVE}.sha512"

echo "Verifying GPG signature"
gpg --verify "${ARCHIVE}.asc" "${ARCHIVE}"

echo "Verifying tarball integrity"
tar tzf "${ARCHIVE}" > /dev/null

for REQUIRED_FILE in \
  Cargo.lock \
  LICENSE \
  NOTICE \
  core/LICENSE \
  core/NOTICE \
  DEPENDENCIES.rust.tsv
do
  tar tzf "${ARCHIVE}" "${ARCHIVE_ROOT}/${REQUIRED_FILE}" > /dev/null
done

echo ""
echo "Source release created successfully. Artifacts in tools/release/:"
ls -la "${CURR_DIR}"/release/apache-paimon-mosaic-*
echo ""
echo "Next: upload contents to SVN (see docs/creating-a-release.html)."
