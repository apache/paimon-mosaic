#!/usr/bin/env python3

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

"""Rewrite the project version in a Maven POM without touching other versions."""

from __future__ import annotations

import argparse
import sys
import xml.parsers.expat
from pathlib import Path


class PomBumpError(RuntimeError):
    """Raised when a POM cannot be bumped safely."""


# A textual search cannot tell the project version from a dependency or plugin
# that happens to carry the same string, so locate the element structurally.
PROJECT_VERSION_PATHS = (("project", "version"), ("project", "parent", "version"))


def project_version_spans(document: bytes) -> list[tuple[int, int]]:
    parser = xml.parsers.expat.ParserCreate()
    path: list[str] = []
    spans: list[tuple[int, int]] = []
    opened: list[int] = []

    def start(name: str, _attributes: dict[str, str]) -> None:
        path.append(name.split("}")[-1])
        opened.append(parser.CurrentByteIndex)

    def end(name: str) -> None:
        start_index = opened.pop()
        if tuple(path) in PROJECT_VERSION_PATHS:
            spans.append((start_index, parser.CurrentByteIndex + len(f"</{name}>")))
        path.pop()

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    try:
        parser.Parse(document, True)
    except xml.parsers.expat.ExpatError as error:
        raise PomBumpError(f"POM is not well-formed XML: {error}") from error
    return spans


def bump_pom_version(pom: Path, old_version: str, new_version: str) -> bool:
    document = pom.read_bytes()
    replaced = False
    # Rewrite back to front so earlier spans keep their offsets.
    for start_index, end_index in sorted(project_version_spans(document), reverse=True):
        element = document[start_index:end_index].decode("utf-8")
        if element.strip() != f"<version>{old_version}</version>":
            continue
        document = (
            document[:start_index]
            + f"<version>{new_version}</version>".encode("utf-8")
            + document[end_index:]
        )
        replaced = True

    if not replaced:
        raise PomBumpError(
            f"{pom}: no project or parent version element holds {old_version}"
        )
    pom.write_bytes(document)
    return replaced


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_version")
    parser.add_argument("new_version")
    parser.add_argument("pom", nargs="+", type=Path)
    arguments = parser.parse_args(argv)

    try:
        for pom in arguments.pom:
            bump_pom_version(pom, arguments.old_version, arguments.new_version)
    except (PomBumpError, OSError) as error:
        print(f"pom version bump failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
