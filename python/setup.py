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

"""Build helper for an artifact-exact native wheel."""

import os
import platform
import shutil

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


def _package_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "mosaic")


def _rust_target():
    configured = os.environ.get("PAIMON_MOSAIC_RUST_TARGET")
    supported = {
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "aarch64-apple-darwin",
        "x86_64-pc-windows-msvc",
    }
    system = platform.system()
    machine = platform.machine().lower()
    detected = {
        ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
        ("Linux", "amd64"): "x86_64-unknown-linux-gnu",
        ("Linux", "aarch64"): "aarch64-unknown-linux-gnu",
        ("Linux", "arm64"): "aarch64-unknown-linux-gnu",
        ("Darwin", "aarch64"): "aarch64-apple-darwin",
        ("Darwin", "arm64"): "aarch64-apple-darwin",
        ("Windows", "x86_64"): "x86_64-pc-windows-msvc",
        ("Windows", "amd64"): "x86_64-pc-windows-msvc",
    }.get((system, machine))
    if not detected:
        raise RuntimeError(
            f"Unsupported wheel build platform: system={system}, machine={machine}"
        )
    if configured:
        if configured not in supported:
            raise RuntimeError("Unsupported PAIMON_MOSAIC_RUST_TARGET: " + configured)
        if configured != detected:
            raise RuntimeError(
                f"PAIMON_MOSAIC_RUST_TARGET {configured} does not match "
                f"build platform {detected}"
            )
        return configured
    return detected


def _license_files():
    target = _rust_target()
    return [
        f"licenses/{target}/LICENSE",
        f"licenses/{target}/NOTICE",
        f"licenses/{target}/THIRD-PARTY-LICENSES.html",
    ]


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

    packaged = os.path.join(_package_dir(), lib)
    if os.path.isfile(packaged):
        return packaged

    return None


class BuildPyWithNativeLib(build_py):
    def run(self):
        super().run()

        src = _find_native_lib()
        if not src:
            raise RuntimeError(
                "The pre-built paimon-mosaic FFI native library was not found"
            )

        build_package = os.path.join(self.build_lib, "mosaic")
        os.makedirs(build_package, exist_ok=True)
        shutil.copy2(src, os.path.join(build_package, _lib_name()))

        license_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "licenses",
            _rust_target(),
        )
        for metadata_file in ["LICENSE", "NOTICE", "THIRD-PARTY-LICENSES.html"]:
            shutil.copy2(
                os.path.join(license_dir, metadata_file),
                os.path.join(build_package, metadata_file),
            )


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
    license_files=_license_files(),
)
