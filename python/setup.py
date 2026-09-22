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

"""Build helper: stages the native library and binary-distribution legal files."""

import os
import platform
import shutil

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


LEGAL_FILES = ("LICENSE", "NOTICE", "LICENSE-binary")
LEGAL_SOURCE_FILES = {
    "LICENSE": "../LICENSE",
    "NOTICE": "../java/src/main/binary-resources/META-INF/NOTICE",
    "LICENSE-binary": "../LICENSE-binary-ffi",
}


def _lib_name():
    system = platform.system()
    if system == "Darwin":
        return "libpaimon_mosaic_ffi.dylib"
    elif system == "Windows":
        return "paimon_mosaic_ffi.dll"
    return "libpaimon_mosaic_ffi.so"


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


def _stage_legal_files(destination_dir):
    here = os.path.dirname(os.path.abspath(__file__))
    staged = []

    for name in LEGAL_FILES:
        source = os.path.join(here, LEGAL_SOURCE_FILES[name])
        destination = os.path.join(destination_dir, name)
        if not os.path.isfile(source):
            raise RuntimeError(f"required binary legal file is missing: {source}")
        if os.path.exists(destination):
            with open(source, "rb") as source_file:
                source_bytes = source_file.read()
            with open(destination, "rb") as destination_file:
                destination_bytes = destination_file.read()
            if source_bytes != destination_bytes:
                raise RuntimeError(
                    f"staged binary legal file does not match repository copy: {destination}"
                )
            continue
        shutil.copy2(source, destination)
        staged.append(destination)

    return staged


class BuildPyWithNativeLib(build_py):
    def run(self):
        src = _find_native_lib()
        if src:
            dst = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "mosaic", _lib_name()
            )
            shutil.copy2(src, dst)
        staged_legal_files = _stage_legal_files(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "mosaic")
        )
        try:
            super().run()
        finally:
            for path in staged_legal_files:
                os.remove(path)


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


staged_distribution_legal_files = _stage_legal_files(
    os.path.dirname(os.path.abspath(__file__))
)
try:
    setup(
        cmdclass={"build_py": BuildPyWithNativeLib, "bdist_wheel": PlatformWheel},
        distclass=BinaryDistribution,
    )
finally:
    for path in staged_distribution_legal_files:
        os.remove(path)
