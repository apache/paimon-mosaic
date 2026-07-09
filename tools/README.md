<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# Release tools

Release automation requires Python 3.11 or newer. The published Python
binding has a separate runtime floor of Python 3.9.

## Dependency and artifact licensing

The committed lockfile freezes the dependency closure used by source, Rust,
Java, and Python release artifacts. Install the pinned generators and refresh
the checked-in dependency and binary-license reports with:

```bash
cargo install cargo-deny --version 0.19.0 --locked
cargo install cargo-about --version 0.9.1 --locked
cargo fetch --locked
python3 tools/dependencies.py generate
python3 tools/generate_license_reports.py
```

CI verifies that the reports still match `Cargo.lock`:

```bash
python3 tools/dependencies.py check
python3 tools/generate_license_reports.py --check
```

Before signing or publishing a tag, verify that the Java, Rust, Python, and
lockfile versions all match the intended release:

```bash
python3 tools/verify_release_versions.py 0.3.0
```

Release CI also downloads the Apache Paimon `KEYS` file into an isolated GPG
keyring and validates the Git tag:

```bash
python3 tools/validate_release_tag.py v0.3.0-rc1 --keys-file /path/to/KEYS
python3 tools/validate_release_tag.py v0.3.0 --keys-file /path/to/KEYS
```

Tags must be canonical signed annotated tags. A final `vX.Y.Z` tag is accepted
only when it points to the same commit as at least one valid signed
`vX.Y.Z-rcN` tag.

The Java binary JAR carries all four target-specific reports. Sources and
javadoc JARs retain only the source LICENSE and NOTICE. Each Python wheel
carries exactly one target report and installs its legal files both in the
`mosaic` package and in the standard `.dist-info/licenses/` directory. PyPI
publishing is intentionally wheel-only; the signed ASF tarball is the source
release.

Release workflows verify assembled artifacts with:

```bash
python3 tools/verify_java_jars.py \
  --main MAIN.jar --sources SOURCES.jar --javadoc JAVADOC.jar \
  --require-all-natives
python3 tools/verify_python_wheels.py --require-all-targets dist/*.whl
```
