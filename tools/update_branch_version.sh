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

OLD_VERSION=${OLD_VERSION}
NEW_VERSION=${NEW_VERSION}


if [ -z "${OLD_VERSION}" ]; then
	echo "OLD_VERSION is unset"
	exit 1
fi

if [ -z "${NEW_VERSION}" ]; then
	echo "NEW_VERSION is unset"
	exit 1
fi

cd ..

# Cargo.toml and pyproject.toml never carry the -SNAPSHOT suffix, so strip it
# from both old and new versions when matching/replacing there.
OLD_VERSION_CLEAN=$(echo "$OLD_VERSION" | sed 's/-SNAPSHOT//')
NEW_VERSION_CLEAN=$(echo "$NEW_VERSION" | sed 's/-SNAPSHOT//')

#change version in all pom files (match both exact and -SNAPSHOT suffix)
find . -name 'pom.xml' -type f -exec perl -pi -e 's#<version>'$OLD_VERSION'(-SNAPSHOT)?</version>#<version>'$NEW_VERSION'</version>#' {} \;

#change version in Cargo.toml files
find . -name 'Cargo.toml' -not -path '*/target/*' -type f -exec perl -pi -e 's#^version = "'$OLD_VERSION_CLEAN'"#version = "'$NEW_VERSION_CLEAN'"#' {} \;

#change version in pyproject.toml
perl -pi -e 's#^version = "'$OLD_VERSION_CLEAN'"#version = "'$NEW_VERSION_CLEAN'"#' python/pyproject.toml

git commit -am "Update version to $NEW_VERSION"

echo "Don't forget to push the change."
