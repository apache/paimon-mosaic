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

##
## Variables with defaults (if not overwritten by environment)
##
MVN=${MVN:-mvn}

# fail immediately
set -o errexit
set -o nounset
# print command before executing
set -o xtrace

CURR_DIR=`pwd`
if [[ `basename $CURR_DIR` != "tools" ]] ; then
  echo "You have to call the script from the tools/ dir"
  exit 1
fi

###########################

cd ..

# Detect OS
case "$(uname -s)" in
  Linux*)  OS_NAME="linux";;
  Darwin*) OS_NAME="macos";;
  *)       echo "Unsupported OS: $(uname -s)"; exit 1;;
esac

# Detect architecture
case "$(uname -m)" in
  x86_64|amd64)   ARCH="x86_64";;
  aarch64|arm64)   ARCH="aarch64";;
  *)               echo "Unsupported arch: $(uname -m)"; exit 1;;
esac

# Determine library file name
if [ "$OS_NAME" = "linux" ]; then
  LIB_FILE="libmosaic_jni.so"
elif [ "$OS_NAME" = "macos" ]; then
  LIB_FILE="libmosaic_jni.dylib"
fi

echo "Building native JNI library for ${OS_NAME}/${ARCH}"
cargo build --release -p mosaic-jni

RESOURCE_DIR="java/src/main/resources/native/${OS_NAME}/${ARCH}"
mkdir -p "$RESOURCE_DIR"
cp "target/release/${LIB_FILE}" "$RESOURCE_DIR/"

echo "Native library copied to ${RESOURCE_DIR}/${LIB_FILE}"

###########################
COMMON_OPTIONS="-Prelease -DskipTests -DretryFailedDeploymentCount=10 "

cd java

echo "Deploying to repository.apache.org"
$MVN clean deploy $COMMON_OPTIONS
