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

if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  echo "The source release must be created from a clean Git worktree" >&2
  git status --short >&2
  exit 1
fi

git rev-parse --verify 'HEAD^{commit}' > /dev/null

rm -rf tools/release
mkdir tools/release

python3 tools/verify_release_versions.py "${RELEASE_VERSION}"

echo "Verifying locked dependencies and generated legal metadata"
cargo metadata --locked --format-version 1 --no-deps > /dev/null
python3 tools/dependencies.py check
python3 tools/generate_license_reports.py --check

echo "Creating source package"

ARCHIVE="apache-paimon-mosaic-${RELEASE_VERSION}-src.tgz"
ARCHIVE_PATH="tools/release/${ARCHIVE}"
FIRST_ARCHIVE=$(mktemp "${ARCHIVE_PATH}.first.XXXXXX")
SECOND_ARCHIVE=$(mktemp "${ARCHIVE_PATH}.second.XXXXXX")

cleanup_archives() {
  rm -f "${FIRST_ARCHIVE}" "${SECOND_ARCHIVE}"
}
trap cleanup_archives EXIT

create_archive() {
  local output=$1

  # Archive the commit, rather than only its tree, so Git uses the commit timestamp
  # and records the exact source commit in the tar metadata. gzip -n removes the
  # gzip header timestamp and original filename.
  git archive --format=tar --prefix="paimon-mosaic-${RELEASE_VERSION}/" HEAD . \
    ':(exclude).gitignore' ':(exclude).gitattributes' \
    ':(exclude).asf.yaml' ':(exclude).github' \
    ':(exclude)deploysettings.xml' ':(exclude)target' \
    ':(exclude).idea' ':(exclude)*.iml' ':(exclude).DS_Store' \
    | gzip -n > "${output}"
}

create_archive "${FIRST_ARCHIVE}"
create_archive "${SECOND_ARCHIVE}"
cmp "${FIRST_ARCHIVE}" "${SECOND_ARCHIVE}"
mv "${FIRST_ARCHIVE}" "${ARCHIVE_PATH}"
chmod 0644 "${ARCHIVE_PATH}"
rm -f "${SECOND_ARCHIVE}"
trap - EXIT

cd tools/release

gpg --armor --detach-sig "${ARCHIVE}"
$SHASUM "${ARCHIVE}" > "${ARCHIVE}.sha512"

echo "Verifying GPG signature"
gpg --verify "${ARCHIVE}.asc" "${ARCHIVE}"

echo "Verifying tarball integrity"
tar tzf "${ARCHIVE}" > /dev/null

ARCHIVE_ROOT="paimon-mosaic-${RELEASE_VERSION}"
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
