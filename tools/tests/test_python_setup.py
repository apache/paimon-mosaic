# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "paimon_mosaic_setup",
    ROOT / "python/setup.py",
)
assert SPEC is not None and SPEC.loader is not None
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


@pytest.mark.parametrize(
    ("system", "machine", "target"),
    (
        ("Linux", "x86_64", "x86_64-unknown-linux-gnu"),
        ("Linux", "aarch64", "aarch64-unknown-linux-gnu"),
        ("Darwin", "arm64", "aarch64-apple-darwin"),
        ("Windows", "AMD64", "x86_64-pc-windows-msvc"),
    ),
)
def test_release_wheel_platform_mapping(
    system: str, machine: str, target: str
) -> None:
    assert SETUP._target_triple(system, machine) == target


def test_release_wheel_platform_mapping_rejects_unsupported_target() -> None:
    with pytest.raises(RuntimeError, match="unsupported release wheel platform"):
        SETUP._target_triple("Darwin", "x86_64")
