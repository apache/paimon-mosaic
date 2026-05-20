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

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


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


class PlatformWheel(bdist_wheel):
    """Tag wheel as py3-none-{platform} since this is a ctypes package."""

    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = bdist_wheel.get_tag(self)
        return "py3", "none", plat


class BinaryDistribution(Distribution):
    """Force the wheel to be platform-specific."""

    def has_ext_modules(self):
        return True


setup(
    cmdclass={"build_py": BuildPyWithNativeLib, "bdist_wheel": PlatformWheel},
    distclass=BinaryDistribution,
)
