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

###########################

OLD_VERSION=${OLD_VERSION:-}
NEW_VERSION=${NEW_VERSION:-}

if [[ -z "${OLD_VERSION}" ]]; then
  echo "OLD_VERSION is unset" >&2
  exit 1
fi

if [[ -z "${NEW_VERSION}" ]]; then
  echo "NEW_VERSION is unset" >&2
  exit 1
fi

cd ..

if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  echo "Version updates must start from a clean Git worktree" >&2
  git status --short >&2
  exit 1
fi

# Cargo.toml and pyproject.toml never carry the -SNAPSHOT suffix, so strip it
# from both old and new versions when matching/replacing there.
OLD_VERSION_CLEAN=$(echo "$OLD_VERSION" | sed 's/-SNAPSHOT//')
NEW_VERSION_CLEAN=$(echo "$NEW_VERSION" | sed 's/-SNAPSHOT//')

# Change version in all pom files (match both exact and -SNAPSHOT suffix).
export OLD_VERSION_CLEAN NEW_VERSION
find . -name 'pom.xml' -not -path '*/target/*' -type f \
  -exec perl -pi -e '
    BEGIN {
      $old = quotemeta($ENV{"OLD_VERSION_CLEAN"});
      $new = $ENV{"NEW_VERSION"};
    }
    s{<version>${old}(?:-SNAPSHOT)?</version>}{<version>${new}</version>}g;
  ' {} +

# Change workspace package versions and versioned path dependencies together.
# The TOML-aware helper also supports retrying a partially completed bump.
python3 tools/verify_release_versions.py \
  --update-cargo "$OLD_VERSION_CLEAN" "$NEW_VERSION_CLEAN"

# Change version in pyproject.toml.
export NEW_VERSION_CLEAN
perl -pi -e '
  BEGIN {
    $old = quotemeta($ENV{"OLD_VERSION_CLEAN"});
    $new = $ENV{"NEW_VERSION_CLEAN"};
  }
  s{^version = "${old}"$}{version = "${new}"};
' python/pyproject.toml

# Refresh the lockfile without upgrading registry dependencies, then regenerate
# every checked-in report containing workspace package versions.
cargo update --workspace --offline
python3 tools/dependencies.py generate
python3 tools/generate_license_reports.py
if [[ "$NEW_VERSION" == *-SNAPSHOT ]]; then
  python3 tools/verify_release_versions.py "$NEW_VERSION_CLEAN" --java-snapshot
else
  python3 tools/verify_release_versions.py "$NEW_VERSION_CLEAN"
fi

git add \
  Cargo.toml \
  core/Cargo.toml \
  ffi/Cargo.toml \
  jni/Cargo.toml \
  cli/Cargo.toml \
  Cargo.lock \
  DEPENDENCIES.rust.tsv \
  core/DEPENDENCIES.rust.tsv \
  ffi/DEPENDENCIES.rust.tsv \
  jni/DEPENDENCIES.rust.tsv \
  cli/DEPENDENCIES.rust.tsv \
  java/pom.xml \
  java/src/main/binary-resources \
  python/pyproject.toml \
  python/licenses
git commit -m "Update version to $NEW_VERSION"

if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  echo "Version update left unstaged or uncommitted files" >&2
  git status --short >&2
  exit 1
fi

echo "Don't forget to push the change."
