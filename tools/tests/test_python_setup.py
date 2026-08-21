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

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SETUP_PY = REPOSITORY_ROOT / "python/setup.py"


def load_setup_module(monkeypatch):
    setuptools = types.ModuleType("setuptools")
    setuptools.Distribution = type("Distribution", (), {})
    setuptools.setup = lambda **_kwargs: None

    setuptools_command = types.ModuleType("setuptools.command")
    setuptools_build_py = types.ModuleType("setuptools.command.build_py")
    setuptools_build_py.build_py = type("build_py", (), {})

    wheel = types.ModuleType("wheel")
    wheel_bdist = types.ModuleType("wheel.bdist_wheel")
    wheel_bdist.bdist_wheel = type("bdist_wheel", (), {})

    monkeypatch.setitem(sys.modules, "setuptools", setuptools)
    monkeypatch.setitem(sys.modules, "setuptools.command", setuptools_command)
    monkeypatch.setitem(
        sys.modules, "setuptools.command.build_py", setuptools_build_py
    )
    monkeypatch.setitem(sys.modules, "wheel", wheel)
    monkeypatch.setitem(sys.modules, "wheel.bdist_wheel", wheel_bdist)
    return runpy.run_path(str(SETUP_PY))


def test_find_native_lib_prefers_release_over_debug_and_packaged(
    tmp_path, monkeypatch
):
    setup_module = load_setup_module(monkeypatch)
    python_directory = tmp_path / "python"
    setup_module["_find_native_lib"].__globals__["__file__"] = str(
        python_directory / "setup.py"
    )
    library_name = setup_module["_lib_name"]()
    packaged = python_directory / "mosaic" / library_name
    release = tmp_path / "target/release" / library_name
    debug = tmp_path / "target/debug" / library_name
    packaged.parent.mkdir(parents=True)
    release.parent.mkdir(parents=True)
    debug.parent.mkdir(parents=True)
    packaged.write_bytes(b"stale")
    release.write_bytes(b"release")
    debug.write_bytes(b"debug")
    monkeypatch.delenv("MOSAIC_LIB_PATH", raising=False)

    assert Path(setup_module["_find_native_lib"]()).resolve() == release.resolve()
