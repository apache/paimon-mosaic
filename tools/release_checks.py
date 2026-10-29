#!/usr/bin/env python3
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

"""Check the local release JVM and actual artifact signatures before upload."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


def output(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise ValueError(result.stderr.strip() or result.stdout.strip())
    return result.stdout + result.stderr


def java_home(version):
    version = re.sub(r"\x1b\[[0-9;]*m", "", version)
    if not re.search(r"Java version: 1\.8\.", version):
        raise ValueError("Java release packaging requires Maven running on JDK 8.\n" + version)
    runtime = re.search(r"(?:runtime:|Java home:)\s*([^\r\n]+)", version)
    if runtime is None:
        raise ValueError("Cannot determine Maven's Java home from mvn --version")
    home = Path(runtime[1].strip()).resolve()
    if home.name == "jre":
        home = home.parent
    suffix = ".exe" if os.name == "nt" else ""
    tools = ("java", "javac", "javadoc", "jar")
    if not all((home / "bin" / (tool + suffix)).is_file() for tool in tools):
        raise ValueError("Maven must use a complete JDK 8: " + str(home))
    settings = output([str(home / "bin" / ("java" + suffix)), "-XshowSettings:properties", "-version"])
    properties = dict(re.findall(r"(?m)^\s*(java.version|os.name|os.arch)\s*=\s*(.+?)\s*$", settings))
    if not properties.get("java.version", "").startswith("1.8."):
        raise ValueError("Maven's Java home does not contain JDK 8")
    name = properties.get("os.name", "").lower()
    arch = properties.get("os.arch", "").lower()
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(arch, arch)
    if "mac" in name or "darwin" in name:
        platform = "macos"
    elif "win" in name:
        platform = "windows"
    else:
        platform = name
    supported = {
        ("linux", "x86_64"), ("linux", "aarch64"),
        ("macos", "aarch64"), ("windows", "x86_64"),
    }
    if (platform, arch) not in supported:
        raise ValueError(
            "No release JNI library for Maven's JVM: " + platform + "/" + arch
            + ". On Apple silicon select an ARM64 JDK 8."
        )
    print("Verified Maven JDK 8: " + str(home) + " (" + platform + "/" + arch + ")", file=sys.stderr)
    return home


def gpg_command(executable=None, homedir=None):
    # Maven leaves an unresolved expression literal when the optional property
    # is unset. In that case use GPG's normal executable/home (including GNUPGHOME).
    command = [executable if executable and executable != "${gpg.executable}" else "gpg", "--batch"]
    if homedir and homedir != "${gpg.homedir}":
        command += ["--homedir", homedir]
    return command


def verify_signatures(signed_files, executable=None, homedir=None):
    gpg = gpg_command(executable, homedir)
    for signature, artifact in signed_files:
        signature, artifact = Path(signature), Path(artifact)
        if not signature.read_bytes().lstrip().startswith(b"-----BEGIN PGP SIGNATURE-----"):
            raise ValueError("Expected an ASCII-armored detached signature: " + str(signature))
        result = output(gpg + ["--status-fd", "1", "--verify", str(signature), str(artifact)])
        signatures = [line.split() for line in result.splitlines() if line.startswith("[GNUPG:] VALIDSIG ")]
        invalid = ("[GNUPG:] EXPKEYSIG ", "[GNUPG:] EXPSIG ", "[GNUPG:] REVKEYSIG ")
        if not signatures or any(token in result for token in invalid):
            raise ValueError("No valid current signature: " + str(signature))
        for status in signatures:
            fingerprint = status[2]
            listing = output(gpg + ["--with-colons", "--fixed-list-mode", "--list-keys", fingerprint])
            keys = {}
            key = None
            for line in listing.splitlines():
                fields = line.split(":")
                if fields[0] in ("pub", "sub"):
                    key = (int(fields[3]), int(fields[2]))
                elif fields[0] == "fpr" and key is not None:
                    keys[fields[9]] = key
                    key = None
            algorithm, bits = keys.get(fingerprint, (0, 0))
            # Check the actual signing subkey, not merely its primary key.
            if int(status[8]) not in (1, 3) or algorithm not in (1, 3) or bits < 2048:
                raise ValueError(
                    "ASF release signatures require RSA >=2048 bits; actual signer "
                    + fingerprint + " uses algorithm " + str(algorithm)
                    + ", " + str(bits) + " bits: " + str(artifact)
                )
            print("Verified RSA release signature: " + str(artifact) + " (" + fingerprint + ")")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("java-home", help="Read Maven --version from stdin")
    signatures = subparsers.add_parser("signatures")
    signatures.add_argument(
        "--signed-file", nargs=2, action="append", required=True,
        metavar=("SIGNATURE", "ARTIFACT"),
    )
    signatures.add_argument("--gpg-executable")
    signatures.add_argument("--gpg-home")
    args = parser.parse_args()
    try:
        if args.command == "java-home":
            print(java_home(sys.stdin.read()))
        else:
            verify_signatures(args.signed_file, args.gpg_executable, args.gpg_home)
    except (OSError, ValueError, IndexError) as error:
        print("Release check failed: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
