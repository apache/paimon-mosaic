#!/usr/bin/env python3

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

"""Validate .asf.yaml against known ASF infrastructure schema.

Reference: https://github.com/apache/infrastructure-asfyaml/blob/main/README.md
"""

import sys

try:
    import yaml
except ImportError:
    sys.exit(
        "PyYAML is required: pip install pyyaml\n"
        "Or use: python3 -c \"import pip; pip.main(['install', 'pyyaml'])\""
    )

ASF_YAML_PATH = ".asf.yaml"

VALID_TOP_LEVEL_KEYS = {
    "github",
    "notifications",
    "staging",
    "publish",
    "pelican",
}

VALID_GITHUB_KEYS = {
    "description",
    "homepage",
    "labels",
    "features",
    "enabled_merge_buttons",
    "protected_branches",
    "collaborators",
    "autolinks",
    "environments",
    "dependabot_alerts",
    "dependabot_updates",
    "code_scanning",
    "del_branch_on_merge",
    "ghp_branch",
    "ghp_path",
}

VALID_FEATURES_KEYS = {
    "issues",
    "discussions",
    "wiki",
    "projects",
}

VALID_MERGE_BUTTON_KEYS = {
    "squash",
    "merge",
    "rebase",
}

VALID_NOTIFICATIONS_KEYS = {
    "commits",
    "issues",
    "pullrequests",
    "jira_options",
    "jobs",
    "discussions",
}


def validate():
    errors = []

    try:
        with open(ASF_YAML_PATH, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"SKIP: {ASF_YAML_PATH} not found")
        return 0
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML syntax in {ASF_YAML_PATH}: {e}")
        return 1

    if not isinstance(data, dict):
        print(f"ERROR: {ASF_YAML_PATH} root must be a mapping")
        return 1

    for key in data:
        if key not in VALID_TOP_LEVEL_KEYS:
            errors.append(f"unexpected top-level key '{key}'")

    github = data.get("github")
    if isinstance(github, dict):
        for key in github:
            if key not in VALID_GITHUB_KEYS:
                errors.append(f"unexpected key 'github.{key}'")

        features = github.get("features")
        if isinstance(features, dict):
            for key in features:
                if key not in VALID_FEATURES_KEYS:
                    errors.append(
                        f"unexpected key 'github.features.{key}' "
                        f"(allowed: {', '.join(sorted(VALID_FEATURES_KEYS))})"
                    )
                elif not isinstance(features[key], bool):
                    errors.append(
                        f"'github.features.{key}' must be a boolean, "
                        f"got {type(features[key]).__name__}"
                    )

        merge_buttons = github.get("enabled_merge_buttons")
        if isinstance(merge_buttons, dict):
            for key in merge_buttons:
                if key not in VALID_MERGE_BUTTON_KEYS:
                    errors.append(
                        f"unexpected key 'github.enabled_merge_buttons.{key}' "
                        f"(allowed: {', '.join(sorted(VALID_MERGE_BUTTON_KEYS))})"
                    )

    notifications = data.get("notifications")
    if isinstance(notifications, dict):
        for key in notifications:
            if key not in VALID_NOTIFICATIONS_KEYS:
                errors.append(f"unexpected key 'notifications.{key}'")

    if errors:
        print(f"ERROR: {ASF_YAML_PATH} validation failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"OK: {ASF_YAML_PATH} is valid")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
