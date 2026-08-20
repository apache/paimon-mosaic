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

import re
import sys
from pathlib import Path


TESTS = Path(__file__).resolve().parent
TOOLS = TESTS.parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import native_binary as verifier  # noqa: E402


JNI_PREFIX = "Java_org_apache_paimon_mosaic_NativeLib_"


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_jni_symbol_contract_matches_java_rust_and_verifier():
    java_methods = set(
        re.findall(
            r"\bstatic\s+native\s+[^;]*?\b(native[A-Za-z0-9_]+)\s*\(",
            source(
                "java/src/main/java/org/apache/paimon/mosaic/NativeLib.java"
            ),
            flags=re.DOTALL,
        )
    )
    rust_source = source("jni/src/lib.rs")
    rust_exports = set(
        re.findall(
            r'\bpub\s+extern\s+"system"\s+fn\s+'
            rf"({JNI_PREFIX}[A-Za-z0-9_]+)\b",
            rust_source,
        )
    )
    generic_rust_exports = set(
        re.findall(
            r'\bpub\s+extern\s+"system"\s+fn\s+'
            rf"({JNI_PREFIX}[A-Za-z0-9_]+)\s*<",
            rust_source,
        )
    )
    java_exports = {JNI_PREFIX + method for method in java_methods}

    assert len(java_exports) == 24
    assert len(rust_exports) == 24
    assert len(generic_rust_exports) == 6
    assert generic_rust_exports <= rust_exports
    assert rust_exports == java_exports
    assert verifier.MOSAIC_SYMBOL_FAMILIES["JNI"] == java_exports


def test_ffi_symbol_contract_matches_header_rust_and_verifier():
    header_without_comments = re.sub(
        r"/\*.*?\*/",
        "",
        source("include/mosaic.h"),
        flags=re.DOTALL,
    )
    header_exports = set(
        re.findall(
            r"\b(mosaic_[A-Za-z0-9_]+)\s*\(",
            header_without_comments,
        )
    )
    rust_exports = set(
        re.findall(
            r'\bpub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+'
            r"(mosaic_[A-Za-z0-9_]+)\b",
            source("ffi/src/lib.rs"),
        )
    )

    assert len(header_exports) == 26
    assert len(rust_exports) == 26
    assert rust_exports == header_exports
    assert verifier.MOSAIC_SYMBOL_FAMILIES["FFI"] == header_exports
