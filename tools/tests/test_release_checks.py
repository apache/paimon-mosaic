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

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_checks as checks  # noqa: E402


@pytest.fixture(scope="module")
def signing_keys(tmp_path_factory):
    if shutil.which("gpg") is None:
        pytest.skip("gpg is required")
    home = tmp_path_factory.mktemp("signature-keys")
    home.chmod(0o700)
    command = ["gpg", "--homedir", str(home), "--batch", "--pinentry-mode", "loopback", "--passphrase", ""]

    def run(*args):
        return subprocess.check_output(command + list(args), stderr=subprocess.PIPE, text=True)

    def fingerprints():
        return [line.split(":")[9] for line in run("--with-colons", "--list-secret-keys").splitlines() if line.startswith("fpr:")]

    run("--quick-gen-key", "Release Policy Test <policy@example.invalid>", "rsa2048", "cert", "0")
    primary = fingerprints()[0]
    keys = {}
    for algorithm in ("rsa2048", "rsa1024", "ed25519"):
        before = set(fingerprints())
        run("--quick-add-key", primary, algorithm, "sign", "0")
        keys[algorithm] = next(iter(set(fingerprints()) - before))
    yield home, command, keys
    subprocess.run(["gpgconf", "--homedir", str(home), "--kill", "gpg-agent"], capture_output=True)


@pytest.mark.parametrize("algorithm", ["rsa2048", "rsa1024", "ed25519"])
def test_checks_actual_signing_subkey(signing_keys, tmp_path, algorithm):
    home, command, keys = signing_keys
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"release bytes\n")
    signature = Path(str(artifact) + ".asc")
    subprocess.run(command + ["--armor", "--local-user", keys[algorithm] + "!", "--detach-sign", "--output", str(signature), str(artifact)], check=True, capture_output=True)
    if algorithm == "rsa2048":
        checks.verify_signatures([(signature, artifact)], homedir=str(home))
    else:
        # The primary key is RSA2048 in all three cases. Checking it alone
        # would incorrectly approve these weak/non-RSA signing subkeys.
        with pytest.raises(ValueError, match="RSA >=2048"):
            checks.verify_signatures([(signature, artifact)], homedir=str(home))


def test_tampered_signed_artifact_is_rejected(signing_keys, tmp_path):
    home, command, keys = signing_keys
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"original")
    signature = Path(str(artifact) + ".asc")
    subprocess.run(command + ["--armor", "--local-user", keys["rsa2048"] + "!", "--detach-sign", "--output", str(signature), str(artifact)], check=True, capture_output=True)
    artifact.write_bytes(b"modified")
    with pytest.raises(ValueError):
        checks.verify_signatures([(signature, artifact)], homedir=str(home))


def test_binary_signature_is_rejected_before_verification(tmp_path):
    signature = tmp_path / "binary.asc"
    signature.write_bytes(b"\x89binary")
    with pytest.raises(ValueError, match="ASCII-armored"):
        checks.verify_signatures([(signature, tmp_path / "artifact")])


def test_unset_maven_properties_use_gpg_defaults():
    assert checks.gpg_command("${gpg.executable}", "${gpg.homedir}") == ["gpg", "--batch"]


def test_java_home_uses_maven_jdk_instead_of_path_java(tmp_path, monkeypatch):
    jdk = tmp_path / "Maven JDK"
    (jdk / "bin").mkdir(parents=True)
    for name in ("java", "javac", "javadoc", "jar"):
        (jdk / "bin" / name).touch()
    commands = []

    def output(command, **kwargs):
        commands.append(command)
        assert command[0] == str(jdk / "bin/java")
        return "java.version = 1.8.0_502\nos.name = Mac OS X\nos.arch = aarch64\n"

    monkeypatch.setattr(checks, "output", output)
    monkeypatch.setenv("JAVA_HOME", "/wrong/path/jdk11")
    version = f"Java version: 1.8.0_502, vendor: Test, runtime: {jdk}/jre\n"
    assert checks.java_home(version) == jdk
    assert len(commands) == 1
