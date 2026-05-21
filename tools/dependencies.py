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

"""Check and generate Rust dependency license information for ASF release compliance.

Requires cargo-deny: cargo install cargo-deny
Requires Python 3.11+ (uses tomllib).

Usage:
    python3 tools/dependencies.py check      # Verify all deps have approved licenses
    python3 tools/dependencies.py generate   # Generate DEPENDENCIES.rust.tsv
"""

import sys

if sys.version_info < (3, 11):
    sys.exit(
        "This script requires Python 3.11 or newer (uses tomllib). "
        f"Current: {sys.version}."
    )

import subprocess
import tomllib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

PACKAGES = ["."]
root_cargo = ROOT_DIR / "Cargo.toml"
if root_cargo.exists():
    with open(root_cargo, "rb") as f:
        data = tomllib.load(f)
    members = data.get("workspace", {}).get("members", [])
    if isinstance(members, list):
        for m in members:
            if isinstance(m, str) and m:
                PACKAGES.append(m)


def check_single_package(root):
    pkg_dir = ROOT_DIR / root if root != "." else ROOT_DIR
    if (pkg_dir / "Cargo.toml").exists():
        print(f"Checking dependencies of {root}")
        subprocess.run(
            ["cargo", "deny", "check", "license"],
            cwd=pkg_dir,
            check=True,
        )
    else:
        print(f"Skipping {root} as Cargo.toml does not exist")


def check_deps():
    for d in PACKAGES:
        check_single_package(d)


def generate_single_package(root):
    pkg_dir = ROOT_DIR / root if root != "." else ROOT_DIR
    if (pkg_dir / "Cargo.toml").exists():
        print(f"Generating dependencies for {root}")
        result = subprocess.run(
            ["cargo", "deny", "list", "-f", "tsv", "-t", "0.6"],
            cwd=pkg_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"cargo deny list failed in {root}: {result.stderr or result.stdout}"
            )
        out_file = pkg_dir / "DEPENDENCIES.rust.tsv"
        out_file.write_text(result.stdout)
        print(f"  Written to {out_file}")
    else:
        print(f"Skipping {root} as Cargo.toml does not exist")


def generate_deps():
    for d in PACKAGES:
        generate_single_package(d)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Check and generate Rust dependency license information",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(func=parser.print_help)
    subparsers = parser.add_subparsers()

    parser_check = subparsers.add_parser(
        "check", description="Check dependencies", help="Check dependency licenses"
    )
    parser_check.set_defaults(func=check_deps)

    parser_generate = subparsers.add_parser(
        "generate",
        description="Generate dependencies",
        help="Generate DEPENDENCIES.rust.tsv",
    )
    parser_generate.set_defaults(func=generate_deps)

    args = parser.parse_args()
    arg_dict = dict(vars(args))
    del arg_dict["func"]
    args.func(**arg_dict)
