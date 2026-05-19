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

"""Build helper: copies the pre-built native library into the package directory."""

import os
import platform
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


def _lib_name():
    system = platform.system()
    if system == "Darwin":
        return "libmosaic_ffi.dylib"
    elif system == "Windows":
        return "mosaic_ffi.dll"
    return "libmosaic_ffi.so"


def _find_native_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    lib = _lib_name()

    env_path = os.environ.get("MOSAIC_LIB_PATH")
    if env_path:
        candidate = os.path.join(env_path, lib)
        if os.path.isfile(candidate):
            return candidate

    for profile in ["release", "debug"]:
        candidate = os.path.join(here, "..", "target", profile, lib)
        if os.path.isfile(candidate):
            return candidate

    return None


class BuildPyWithNativeLib(build_py):
    def run(self):
        src = _find_native_lib()
        if src:
            dst = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "mosaic", _lib_name()
            )
            shutil.copy2(src, dst)
        super().run()


setup(cmdclass={"build_py": BuildPyWithNativeLib})
