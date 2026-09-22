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

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from mosaic._ffi import lib


ROOT = Path(__file__).resolve().parents[2]


def test_sdist_to_wheel_preserves_canonical_legal_files(tmp_path):
    repository = tmp_path / "repository"
    package = repository / "python"
    shutil.copytree(
        ROOT / "python/mosaic",
        package / "mosaic",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ("setup.py", "pyproject.toml"):
        shutil.copy2(ROOT / "python" / name, package / name)
    legal_sources = {
        "LICENSE": "LICENSE",
        "NOTICE": "java/src/main/binary-resources/META-INF/NOTICE",
        "LICENSE-binary": "LICENSE-binary-ffi",
    }
    for source in legal_sources.values():
        destination = repository / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)

    # The default build extracts its sdist outside the repository before
    # building the wheel, so only the bundled legal files remain available.
    result = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", str(package)],
        env={**os.environ, "MOSAIC_LIB_PATH": str(Path(lib._name).resolve().parent)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    sdist, = (package / "dist").glob("*.tar.gz")
    with tarfile.open(sdist) as archive:
        prefix = sdist.name.removesuffix(".tar.gz")
        for name, source in legal_sources.items():
            with archive.extractfile(f"{prefix}/{name}") as legal_file:
                assert legal_file.read() == (ROOT / source).read_bytes()

    wheel, = (package / "dist").glob("*.whl")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/verify_binary_artifact.py"),
            "--wheel",
            str(wheel),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
