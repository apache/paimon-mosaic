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
MVN=${MVN:-mvn}

# fail immediately
set -o errexit
set -o nounset
set -o pipefail
# print command before executing
set -o xtrace

CURR_DIR=`pwd`
if [[ `basename $CURR_DIR` != "tools" ]] ; then
  echo "You have to call the script from the tools/ dir"
  exit 1
fi

if [ "$(uname)" == "Darwin" ]; then
    SHASUM="shasum -a 512"
else
    SHASUM="sha512sum"
fi

###########################

RELEASE_VERSION=${RELEASE_VERSION}

if [ -z "${RELEASE_VERSION}" ]; then
	echo "RELEASE_VERSION is unset"
	exit 1
fi

rm -rf release
mkdir release
cd ..

echo "Creating source package"

ARCHIVE="apache-paimon-mosaic-${RELEASE_VERSION}-src.tgz"
# Archive from Git objects so filesystem metadata such as macOS xattrs is not included.
git archive --format=tar --prefix="paimon-mosaic-${RELEASE_VERSION}/" 'HEAD^{tree}' . \
  ':(exclude).gitignore' ':(exclude).gitattributes' \
  ':(exclude).asf.yaml' ':(exclude).github' \
  ':(exclude)deploysettings.xml' ':(exclude)target' \
  ':(exclude).idea' ':(exclude)*.iml' ':(exclude).DS_Store' \
  | gzip -n > "tools/release/${ARCHIVE}"

cd tools/release

gpg --armor --detach-sig "${ARCHIVE}"
$SHASUM "${ARCHIVE}" > "${ARCHIVE}.sha512"

echo "Verifying GPG signature"
gpg --verify "${ARCHIVE}.asc" "${ARCHIVE}"

echo "Verifying tarball integrity"
tar tzf "${ARCHIVE}" > /dev/null

echo ""
echo "Source release created successfully. Artifacts in tools/release/:"
ls -la ${CURR_DIR}/release/apache-paimon-mosaic-*
echo ""
echo "Next: upload contents to SVN (see docs/creating-a-release.html)."
