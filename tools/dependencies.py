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
    python3 tools/dependencies.py check      # Verify licenses and report freshness
    python3 tools/dependencies.py generate   # Generate DEPENDENCIES.rust.tsv
"""

import sys

if sys.version_info < (3, 11):
    sys.exit(
        "This script requires Python 3.11 or newer (uses tomllib). "
        f"Current: {sys.version}."
    )

import difflib
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


def package_dir(root):
    return ROOT_DIR / root if root != "." else ROOT_DIR


def normalized_report(root):
    pkg_dir = package_dir(root)
    result = subprocess.run(
        ["cargo", "deny", "--locked", "list", "-f", "tsv", "-t", "0.6"],
        cwd=pkg_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cargo deny list failed in {root}: {result.stderr or result.stdout}"
        )
    return "\n".join(line.rstrip() for line in result.stdout.splitlines()) + "\n"


def check_deps():
    subprocess.run(
        ["cargo", "deny", "--locked", "check", "licenses"],
        cwd=ROOT_DIR,
        check=True,
    )

    stale = False
    for legal_file in ("LICENSE", "NOTICE"):
        source = ROOT_DIR / legal_file
        packaged = ROOT_DIR / "core" / legal_file
        if not packaged.is_file() or packaged.read_bytes() != source.read_bytes():
            print(f"Stale packaged Rust legal file: {packaged}")
            stale = True

    for root in PACKAGES:
        pkg_dir = package_dir(root)
        if not (pkg_dir / "Cargo.toml").exists():
            print(f"Skipping {root} as Cargo.toml does not exist")
            continue

        print(f"Checking generated dependencies of {root}")
        out_file = pkg_dir / "DEPENDENCIES.rust.tsv"
        expected = normalized_report(root)
        if not out_file.is_file():
            print(f"Missing generated dependency report: {out_file}")
            stale = True
            continue

        actual = out_file.read_text()
        if actual == expected:
            continue

        stale = True
        print(f"Stale generated dependency report: {out_file}")
        for line in difflib.unified_diff(
            actual.splitlines(),
            expected.splitlines(),
            fromfile=str(out_file.relative_to(ROOT_DIR)),
            tofile=f"generated/{out_file.relative_to(ROOT_DIR)}",
            lineterm="",
        ):
            print(line)

    if stale:
        raise RuntimeError(
            "Generated dependency reports are stale; run "
            "'python3 tools/dependencies.py generate'."
        )


def generate_single_package(root):
    pkg_dir = package_dir(root)
    if (pkg_dir / "Cargo.toml").exists():
        print(f"Generating dependencies for {root}")
        out_file = pkg_dir / "DEPENDENCIES.rust.tsv"
        out_file.write_text(normalized_report(root))
        print(f"  Written to {out_file}")
    else:
        print(f"Skipping {root} as Cargo.toml does not exist")


def generate_deps():
    for legal_file in ("LICENSE", "NOTICE"):
        source = ROOT_DIR / legal_file
        packaged = ROOT_DIR / "core" / legal_file
        packaged.write_bytes(source.read_bytes())
        print(f"Copied {source} to {packaged}")

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
