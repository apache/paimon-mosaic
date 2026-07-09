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

import struct
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import native_binary as verifier  # noqa: E402


JNI_SYMBOLS = verifier.MOSAIC_SYMBOL_FAMILIES["JNI"]
FFI_SYMBOLS = verifier.MOSAIC_SYMBOL_FAMILIES["FFI"]


def align(value, alignment):
    return (value + alignment - 1) & -alignment


def build_elf(machine=62, symbols=JNI_SYMBOLS, file_type=3, interpreter=False):
    dynamic = b"\0" * 16
    strings = bytearray(b"\0")
    name_offsets = {}
    for symbol in sorted(symbols):
        name_offsets[symbol] = len(strings)
        strings.extend(symbol.encode() + b"\0")

    program_count = 3 if interpreter else 2
    dynamic_offset = 64 + program_count * 56
    interpreter_bytes = b"/lib64/ld-linux-x86-64.so.2\0" if interpreter else b""
    interpreter_offset = dynamic_offset + len(dynamic)
    strings_offset = interpreter_offset + len(interpreter_bytes)
    symbols_offset = align(strings_offset + len(strings), 8)
    symbol_table = bytearray(b"\0" * 24)
    for symbol in sorted(symbols):
        symbol_table.extend(
            struct.pack(
                "<IBBHQQ",
                name_offsets[symbol],
                0x12,
                0,
                1,
                0x1000,
                1,
            )
        )
    section_offset = align(symbols_offset + len(symbol_table), 8)
    file_size = section_offset + 3 * 64

    data = bytearray(file_size)
    data[:16] = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into(
        "<HHIQQQIHHHHHH",
        data,
        16,
        file_type,
        machine,
        1,
        0,
        64,
        section_offset,
        0,
        64,
        56,
        program_count,
        64,
        3,
        0,
    )
    struct.pack_into(
        "<IIQQQQQQ",
        data,
        64,
        1,
        5,
        0,
        0,
        0,
        file_size,
        file_size,
        0x1000,
    )
    struct.pack_into(
        "<IIQQQQQQ",
        data,
        120,
        2,
        4,
        dynamic_offset,
        dynamic_offset,
        dynamic_offset,
        len(dynamic),
        len(dynamic),
        8,
    )
    if interpreter:
        struct.pack_into(
            "<IIQQQQQQ",
            data,
            176,
            3,
            4,
            interpreter_offset,
            interpreter_offset,
            interpreter_offset,
            len(interpreter_bytes),
            len(interpreter_bytes),
            1,
        )
    data[dynamic_offset : dynamic_offset + len(dynamic)] = dynamic
    data[
        interpreter_offset : interpreter_offset + len(interpreter_bytes)
    ] = interpreter_bytes
    data[strings_offset : strings_offset + len(strings)] = strings
    data[symbols_offset : symbols_offset + len(symbol_table)] = symbol_table

    struct.pack_into(
        "<IIQQQQIIQQ",
        data,
        section_offset + 64,
        0,
        3,
        0,
        0,
        strings_offset,
        len(strings),
        0,
        0,
        1,
        0,
    )
    struct.pack_into(
        "<IIQQQQIIQQ",
        data,
        section_offset + 128,
        0,
        11,
        2,
        0,
        symbols_offset,
        len(symbol_table),
        1,
        1,
        8,
        24,
    )
    return bytes(data)


def build_pe(symbols=JNI_SYMBOLS, dll=True, optional_magic=0x20B):
    pe_offset = 0x80
    optional_size = 240
    section_table_offset = pe_offset + 24 + optional_size
    headers_size = 0x200
    section_rva = 0x1000
    section_offset = headers_size
    section_size = 0x400
    data = bytearray(section_offset + section_size)

    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, pe_offset)
    data[pe_offset : pe_offset + 4] = b"PE\0\0"
    characteristics = 0x0022 | (0x2000 if dll else 0)
    struct.pack_into(
        "<HHIIIHH",
        data,
        pe_offset + 4,
        0x8664,
        1,
        0,
        0,
        0,
        optional_size,
        characteristics,
    )

    optional_offset = pe_offset + 24
    struct.pack_into("<H", data, optional_offset, optional_magic)
    struct.pack_into("<I", data, optional_offset + 32, 0x1000)
    struct.pack_into("<I", data, optional_offset + 36, 0x200)
    struct.pack_into("<I", data, optional_offset + 56, 0x2000)
    struct.pack_into("<I", data, optional_offset + 60, headers_size)
    struct.pack_into("<I", data, optional_offset + 108, 16)
    struct.pack_into("<II", data, optional_offset + 112, section_rva, 0x300)
    struct.pack_into(
        "<8sIIIIIIHHI",
        data,
        section_table_offset,
        b".rdata\0\0",
        section_size,
        section_rva,
        section_size,
        section_offset,
        0,
        0,
        0,
        0,
        0x40000040,
    )

    symbol_list = sorted(symbols)
    export_offset = section_offset
    module_offset = section_offset + 0x40
    functions_offset = section_offset + 0x80
    names_offset = section_offset + 0xA0
    ordinals_offset = section_offset + 0xC0
    string_offset = section_offset + 0x100

    def rva(offset):
        return section_rva + offset - section_offset

    struct.pack_into(
        "<IIHHIIIIIII",
        data,
        export_offset,
        0,
        0,
        0,
        0,
        rva(module_offset),
        1,
        len(symbol_list),
        len(symbol_list),
        rva(functions_offset),
        rva(names_offset),
        rva(ordinals_offset),
    )
    module_name = b"paimon_mosaic_test.dll\0"
    data[module_offset : module_offset + len(module_name)] = module_name
    for index, symbol in enumerate(symbol_list):
        encoded = symbol.encode() + b"\0"
        struct.pack_into(
            "<I", data, functions_offset + index * 4, section_rva + 0x380
        )
        struct.pack_into(
            "<I", data, names_offset + index * 4, rva(string_offset)
        )
        struct.pack_into("<H", data, ordinals_offset + index * 2, index)
        data[string_offset : string_offset + len(encoded)] = encoded
        string_offset += len(encoded)
    return bytes(data)


def build_macho(cpu_type=0x0100000C, symbols=JNI_SYMBOLS, file_type=6):
    symbol_list = sorted(symbols)
    segment_size = 72 + 80
    commands_size = segment_size + 24
    code_offset = 32 + commands_size
    symbols_offset = align(code_offset + 1, 8)

    strings = bytearray(b"\0")
    name_offsets = {}
    for symbol in symbol_list:
        name_offsets[symbol] = len(strings)
        strings.extend(b"_" + symbol.encode() + b"\0")
    strings_offset = symbols_offset + len(symbol_list) * 16
    file_size = strings_offset + len(strings)
    data = bytearray(file_size)

    struct.pack_into(
        "<IiiIIIII",
        data,
        0,
        0xFEEDFACF,
        cpu_type,
        0,
        file_type,
        2,
        commands_size,
        0x80,
        0,
    )
    struct.pack_into(
        "<II16sQQQQiiII",
        data,
        32,
        0x19,
        segment_size,
        b"__TEXT\0" + b"\0" * 9,
        0,
        file_size,
        0,
        file_size,
        7,
        5,
        1,
        0,
    )
    struct.pack_into(
        "<16s16sQQIIIIIIII",
        data,
        32 + 72,
        b"__text\0" + b"\0" * 9,
        b"__TEXT\0" + b"\0" * 9,
        0x1000,
        1,
        code_offset,
        0,
        0,
        0,
        0x80000400,
        0,
        0,
        0,
    )
    struct.pack_into(
        "<IIIIII",
        data,
        32 + segment_size,
        0x02,
        24,
        symbols_offset,
        len(symbol_list),
        strings_offset,
        len(strings),
    )
    data[code_offset] = 0xC3
    for index, symbol in enumerate(symbol_list):
        struct.pack_into(
            "<IBBHQ",
            data,
            symbols_offset + index * 16,
            name_offsets[symbol],
            0x0F,
            1,
            0,
            0x1000,
        )
    data[strings_offset:] = strings
    return bytes(data)


def build_fat_macho(slices):
    entry_size = 20
    table_end = 8 + len(slices) * entry_size
    offsets = []
    offset = align(table_end, 0x1000)
    for cpu_type, image in slices:
        offsets.append(offset)
        offset = align(offset + len(image), 0x1000)
    data = bytearray(offset)
    struct.pack_into(">II", data, 0, 0xCAFEBABE, len(slices))
    for index, ((cpu_type, image), slice_offset) in enumerate(
        zip(slices, offsets)
    ):
        struct.pack_into(
            ">IIIII",
            data,
            8 + index * entry_size,
            cpu_type,
            0,
            slice_offset,
            len(image),
            12,
        )
        data[slice_offset : slice_offset + len(image)] = image
    return bytes(data)


@pytest.mark.parametrize(
    "target,path,data",
    (
        (
            "x86_64-unknown-linux-gnu",
            "native/linux/x86_64/libpaimon_mosaic_jni.so",
            build_elf(machine=62, symbols=JNI_SYMBOLS),
        ),
        (
            "aarch64-unknown-linux-gnu",
            "mosaic/libpaimon_mosaic_ffi.so",
            build_elf(machine=183, symbols=FFI_SYMBOLS),
        ),
        (
            "aarch64-apple-darwin",
            "native/macos/aarch64/libpaimon_mosaic_jni.dylib",
            build_macho(symbols=JNI_SYMBOLS),
        ),
        (
            "x86_64-pc-windows-msvc",
            "mosaic/paimon_mosaic_ffi.dll",
            build_pe(symbols=FFI_SYMBOLS),
        ),
    ),
)
def test_verify_native_target_accepts_four_release_targets(target, path, data):
    verifier.verify_native_target(data, target, path)


@pytest.mark.parametrize(
    "data,error",
    (
        (b"\x7fELF", "truncated ELF header"),
        (build_elf(file_type=2), "not ET_DYN"),
        (build_elf(interpreter=True), "PT_INTERP"),
    ),
)
def test_elf_rejects_truncated_and_executable_images(data, error):
    with pytest.raises(ValueError, match=error):
        verifier.verify_native_target(
            data,
            "x86_64-unknown-linux-gnu",
            "libpaimon_mosaic_jni.so",
        )


def test_elf_rejects_header_only_image():
    data = bytearray(64)
    data[:16] = b"\x7fELF\x02\x01\x01" + b"\0" * 9
    struct.pack_into(
        "<HHIQQQIHHHHHH",
        data,
        16,
        3,
        62,
        1,
        0,
        64,
        0,
        0,
        64,
        56,
        0,
        64,
        0,
        0,
    )

    with pytest.raises(ValueError, match="program header count"):
        verifier.verify_native_target(
            bytes(data),
            "x86_64-unknown-linux-gnu",
            "libpaimon_mosaic_jni.so",
        )


@pytest.mark.parametrize(
    "data,error",
    (
        (b"MZ", "truncated DOS header"),
        (build_pe(dll=False), "DLL characteristic"),
        (build_pe(optional_magic=0x10B), "not PE32\\+"),
    ),
)
def test_pe_rejects_truncated_executable_and_pe32_images(data, error):
    with pytest.raises(ValueError, match=error):
        verifier.verify_native_target(
            data,
            "x86_64-pc-windows-msvc",
            "paimon_mosaic_jni.dll",
        )


@pytest.mark.parametrize(
    "data,error",
    (
        (b"\xcf\xfa\xed\xfe", "truncated Mach-O header"),
        (build_macho(file_type=2), "not MH_DYLIB"),
    ),
)
def test_macho_rejects_truncated_and_executable_images(data, error):
    with pytest.raises(ValueError, match=error):
        verifier.verify_native_target(
            data,
            "aarch64-apple-darwin",
            "libpaimon_mosaic_jni.dylib",
        )


def test_macho_rejects_truncated_load_commands():
    data = bytearray(build_macho())
    struct.pack_into("<I", data, 32 + 4, len(data))

    with pytest.raises(ValueError, match="load command"):
        verifier.verify_native_target(
            bytes(data),
            "aarch64-apple-darwin",
            "libpaimon_mosaic_jni.dylib",
        )


def test_rejects_unexpected_extra_macho_architecture():
    data = build_fat_macho(
        (
            (0x0100000C, build_macho(cpu_type=0x0100000C)),
            (0x01000007, build_macho(cpu_type=0x01000007)),
        )
    )

    with pytest.raises(ValueError, match="expected only aarch64"):
        verifier.verify_native_target(
            data,
            "aarch64-apple-darwin",
            "libpaimon_mosaic_jni.dylib",
        )


def test_rejects_macho_fat_slice_with_mismatched_cpu_type():
    data = build_fat_macho(
        ((0x01000007, build_macho(cpu_type=0x0100000C)),)
    )

    with pytest.raises(ValueError, match="CPU type does not match"):
        verifier.verify_native_target(
            data,
            "aarch64-apple-darwin",
            "libpaimon_mosaic_jni.dylib",
        )


def test_rejects_truncated_macho_fat_slice():
    data = bytearray(
        build_fat_macho(
            ((0x0100000C, build_macho(cpu_type=0x0100000C)),)
        )
    )
    slice_size_offset = 8 + 12
    struct.pack_into(
        ">I",
        data,
        slice_size_offset,
        struct.unpack_from(">I", data, slice_size_offset)[0] + len(data),
    )

    with pytest.raises(ValueError, match="fat slice 0.*out of bounds"):
        verifier.verify_native_target(
            bytes(data),
            "aarch64-apple-darwin",
            "libpaimon_mosaic_jni.dylib",
        )


@pytest.mark.parametrize(
    "target,path,data",
    (
        (
            "x86_64-unknown-linux-gnu",
            "libpaimon_mosaic_jni.so",
            build_elf(symbols={"unrelated_export"}),
        ),
        (
            "x86_64-pc-windows-msvc",
            "paimon_mosaic_ffi.dll",
            build_pe(symbols={"unrelated_export"}),
        ),
        (
            "aarch64-apple-darwin",
            "libpaimon_mosaic_jni.dylib",
            build_macho(symbols={"unrelated_export"}),
        ),
    ),
)
def test_rejects_binary_without_expected_mosaic_exports(target, path, data):
    with pytest.raises(ValueError, match="missing expected Mosaic"):
        verifier.verify_native_target(data, target, path)


def test_raw_symbol_strings_do_not_count_as_elf_exports():
    raw_names = b"\0".join(symbol.encode() for symbol in sorted(JNI_SYMBOLS))
    data = build_elf(symbols={"unrelated_export"}) + raw_names

    with pytest.raises(ValueError, match="missing expected Mosaic JNI exports"):
        verifier.verify_native_target(
            data,
            "x86_64-unknown-linux-gnu",
            "libpaimon_mosaic_jni.so",
        )
