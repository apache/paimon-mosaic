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
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


TARGET_BY_PLATFORM = {
    ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("Linux", "amd64"): "x86_64-unknown-linux-gnu",
    ("Linux", "aarch64"): "aarch64-unknown-linux-gnu",
    ("Linux", "arm64"): "aarch64-unknown-linux-gnu",
    ("Darwin", "arm64"): "aarch64-apple-darwin",
    ("Darwin", "aarch64"): "aarch64-apple-darwin",
    ("Windows", "amd64"): "x86_64-pc-windows-msvc",
    ("Windows", "x86_64"): "x86_64-pc-windows-msvc",
}
LEGAL_FILES = ("LICENSE", "NOTICE", "THIRD-PARTY-LICENSES.html")


def _target_triple(system=None, machine=None):
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    target = TARGET_BY_PLATFORM.get((system, machine))
    if target is None:
        raise RuntimeError(
            f"unsupported release wheel platform: system={system!r}, "
            f"machine={machine!r}"
        )
    return target


def _lib_name(system=None):
    system = system or platform.system()
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


def _stage_release_payload():
    here = Path(__file__).resolve().parent
    package_dir = here / "mosaic"
    package_dir.mkdir(parents=True, exist_ok=True)
    created = []

    native_name = _lib_name()
    native_destination = package_dir / native_name
    native_source = _find_native_lib()
    if native_destination.is_file():
        pass
    elif native_source:
        shutil.copy2(native_source, native_destination)
        created.append(native_destination)
    else:
        return []

    legal_dir = here / "licenses" / _target_triple()
    for name in LEGAL_FILES:
        source = legal_dir / name
        if not source.is_file():
            raise RuntimeError(f"missing release wheel legal file: {source}")
        destination = package_dir / name
        shutil.copy2(source, destination)
        created.append(destination)
    return created


class BuildPyWithNativeLib(build_py):
    def run(self):
        created = _stage_release_payload()
        try:
            super().run()
        finally:
            for path in created:
                path.unlink(missing_ok=True)


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


if __name__ == "__main__":
    setup(
        cmdclass={"build_py": BuildPyWithNativeLib, "bdist_wheel": PlatformWheel},
        distclass=BinaryDistribution,
    )
