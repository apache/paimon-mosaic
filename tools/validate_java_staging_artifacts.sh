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

if [[ $# -ne 2 ]]; then
  echo "Usage: validate_java_staging_artifacts.sh TARGET_DIR VERSION" >&2
  exit 1
fi

TARGET_DIR=$1
VERSION=$2
JAR_FILE="$TARGET_DIR/mosaic-${VERSION}.jar"
SOURCES_JAR="$TARGET_DIR/mosaic-${VERSION}-sources.jar"
JAVADOC_JAR="$TARGET_DIR/mosaic-${VERSION}-javadoc.jar"
JAR_ENTRIES=

cleanup() {
  if [[ -n "$JAR_ENTRIES" ]]; then
    rm -f -- "$JAR_ENTRIES"
  fi
}
trap cleanup EXIT

if ! command -v jar >/dev/null 2>&1; then
  echo "jar command is required to validate Java staging artifacts" >&2
  exit 1
fi

for artifact in "$JAR_FILE" "$SOURCES_JAR" "$JAVADOC_JAR"; do
  if [[ ! -f "$artifact" ]]; then
    echo "Expected Maven artifact is missing: $artifact" >&2
    exit 1
  fi
done

JAR_ENTRIES=$(mktemp "${TMPDIR:-/tmp}/paimon-mosaic-jar-entries.XXXXXX")
jar tf "$JAR_FILE" > "$JAR_ENTRIES"

for native_entry in \
  native/linux/x86_64/libpaimon_mosaic_jni.so \
  native/linux/aarch64/libpaimon_mosaic_jni.so \
  native/macos/aarch64/libpaimon_mosaic_jni.dylib \
  native/windows/x86_64/paimon_mosaic_jni.dll
do
  if ! grep -Fxq -- "$native_entry" "$JAR_ENTRIES"; then
    echo "Packaged jar is missing native entry: $native_entry" >&2
    exit 1
  fi
done

echo "Validated Java staging artifacts."
