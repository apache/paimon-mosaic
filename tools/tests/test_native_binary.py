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


def gnu_hash(name):
    value = 5381
    for byte in name:
        value = (value * 33 + byte) & 0xFFFFFFFF
    return value


def build_elf(
    machine=62,
    symbols=JNI_SYMBOLS,
    file_type=3,
    interpreter=False,
    loader_symbols=True,
    hash_symbol_count=None,
    hash_reachable=True,
    hash_style="sysv",
    gnu_hash_reachable=None,
    symbol_info=0x12,
    symbol_value=0x100,
    load_flags=5,
):
    symbol_list = sorted(symbols)
    strings = bytearray(b"\0")
    name_offsets = {}
    for symbol in symbol_list:
        name_offsets[symbol] = len(strings)
        strings.extend(symbol.encode() + b"\0")

    symbol_table = bytearray(b"\0" * 24)
    for symbol in symbol_list:
        symbol_table.extend(
            struct.pack(
                "<IBBHQQ",
                name_offsets[symbol],
                symbol_info,
                0,
                1,
                symbol_value,
                1,
            )
        )

    hash_tables = []
    if hash_style in ("sysv", "both"):
        if hash_symbol_count is None:
            hash_symbol_count = len(symbol_list) + 1
        hash_buckets = [1 if hash_reachable and hash_symbol_count > 1 else 0]
        hash_chains = [0] * hash_symbol_count
        for symbol_index in range(1, hash_symbol_count - 1):
            hash_chains[symbol_index] = symbol_index + 1
        hash_tables.append(
            (
                4,
                5,
                4,
                b"".join(
                    (
                        struct.pack(
                            "<II", len(hash_buckets), hash_symbol_count
                        ),
                        struct.pack(
                            f"<{len(hash_buckets)}I", *hash_buckets
                        ),
                        struct.pack(
                            f"<{len(hash_chains)}I", *hash_chains
                        ),
                    )
                ),
            )
        )
    if hash_style in ("gnu", "both"):
        if hash_symbol_count is not None and hash_style == "gnu":
            raise ValueError("hash_symbol_count only applies to SysV hashes")
        if gnu_hash_reachable is None:
            gnu_hash_reachable = hash_reachable
        symbol_hashes = [gnu_hash(symbol.encode()) for symbol in symbol_list]
        bloom_word = 0
        for symbol_hash in symbol_hashes:
            bloom_word |= 1 << (symbol_hash % 64)
            bloom_word |= 1 << ((symbol_hash >> 5) % 64)
        stored_hashes = (
            symbol_hashes
            if gnu_hash_reachable
            else [0] * len(symbol_list)
        )
        hash_chains = [
            (symbol_hash & 0xFFFFFFFE)
            | (1 if index == len(stored_hashes) - 1 else 0)
            for index, symbol_hash in enumerate(stored_hashes)
        ]
        hash_tables.append(
            (
                0x6FFFFEF5,
                0x6FFFFFF6,
                0,
                b"".join(
                    (
                        struct.pack("<IIII", 1, 1, 1, 5),
                        struct.pack("<Q", bloom_word),
                        struct.pack("<I", 1 if symbol_list else 0),
                        struct.pack(
                            f"<{len(hash_chains)}I", *hash_chains
                        ),
                    )
                ),
            )
        )
    if not hash_tables:
        raise ValueError(f"unsupported hash style {hash_style}")

    program_count = 3 if interpreter else 2
    dynamic_offset = 64 + program_count * 56
    dynamic_size = (5 + len(hash_tables)) * 16 if loader_symbols else 16
    interpreter_bytes = b"/lib64/ld-linux-x86-64.so.2\0" if interpreter else b""
    interpreter_offset = dynamic_offset + dynamic_size
    next_offset = align(interpreter_offset + len(interpreter_bytes), 8)
    hash_offsets = []
    for _tag, _section_type, _entry_size, hash_table in hash_tables:
        hash_offsets.append(next_offset)
        next_offset = align(next_offset + len(hash_table), 8)
    strings_offset = next_offset
    symbols_offset = align(strings_offset + len(strings), 8)
    if loader_symbols:
        dynamic = b"".join(
            (
                *(
                    struct.pack("<QQ", hash_table[0], hash_offset)
                    for hash_table, hash_offset in zip(
                        hash_tables, hash_offsets
                    )
                ),
                struct.pack("<QQ", 5, strings_offset),
                struct.pack("<QQ", 6, symbols_offset),
                struct.pack("<QQ", 10, len(strings)),
                struct.pack("<QQ", 11, 24),
                b"\0" * 16,
            )
        )
    else:
        dynamic = b"\0" * 16
    section_offset = align(symbols_offset + len(symbol_table), 8)
    section_count = 3 + len(hash_tables)
    file_size = section_offset + section_count * 64

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
        section_count,
        0,
    )
    struct.pack_into(
        "<IIQQQQQQ",
        data,
        64,
        1,
        load_flags,
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
    for hash_table, hash_offset in zip(hash_tables, hash_offsets):
        table_data = hash_table[3]
        data[hash_offset : hash_offset + len(table_data)] = table_data
    data[strings_offset : strings_offset + len(strings)] = strings
    data[symbols_offset : symbols_offset + len(symbol_table)] = symbol_table

    struct.pack_into(
        "<IIQQQQIIQQ",
        data,
        section_offset + 64,
        0,
        3,
        2,
        strings_offset,
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
        symbols_offset,
        symbols_offset,
        len(symbol_table),
        1,
        1,
        8,
        24,
    )
    for index, (hash_table, hash_offset) in enumerate(
        zip(hash_tables, hash_offsets)
    ):
        _tag, hash_section_type, hash_entry_size, table_data = hash_table
        struct.pack_into(
            "<IIQQQQIIQQ",
            data,
            section_offset + (3 + index) * 64,
            0,
            hash_section_type,
            2,
            hash_offset,
            hash_offset,
            len(table_data),
            2,
            0,
            8,
            hash_entry_size,
        )
    return bytes(data)


def build_pe(
    symbols=JNI_SYMBOLS,
    dll=True,
    optional_magic=0x20B,
    function_rva=None,
    section_characteristics=0x60000020,
    forwarder=None,
    symbol_rvas=None,
    symbol_forwarders=None,
):
    pe_offset = 0x80
    optional_size = 240
    section_table_offset = pe_offset + 24 + optional_size
    headers_size = 0x200
    section_rva = 0x1000
    section_offset = headers_size
    section_size = 0x400
    text_rva = 0x2000
    text_offset = section_offset + section_size
    text_size = 0x200
    data = bytearray(text_offset + text_size)

    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, pe_offset)
    data[pe_offset : pe_offset + 4] = b"PE\0\0"
    characteristics = 0x0022 | (0x2000 if dll else 0)
    struct.pack_into(
        "<HHIIIHH",
        data,
        pe_offset + 4,
        0x8664,
        2,
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
    struct.pack_into("<I", data, optional_offset + 56, 0x3000)
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
    struct.pack_into(
        "<8sIIIIIIHHI",
        data,
        section_table_offset + 40,
        b".text\0\0\0",
        text_size,
        text_rva,
        text_size,
        text_offset,
        0,
        0,
        0,
        0,
        section_characteristics,
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

    forwarder_offset = section_offset + 0x2C0
    if forwarder is not None:
        function_rva = rva(forwarder_offset)
    if function_rva is None:
        function_rva = text_rva

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
        symbol_forwarder = (
            (symbol_forwarders or {}).get(symbol)
            if forwarder is None
            else forwarder
        )
        if symbol_forwarder is not None:
            encoded_forwarder = symbol_forwarder.encode() + b"\0"
            current_function_rva = rva(forwarder_offset)
            data[
                forwarder_offset : forwarder_offset + len(encoded_forwarder)
            ] = encoded_forwarder
            forwarder_offset += len(encoded_forwarder)
        else:
            current_function_rva = (symbol_rvas or {}).get(
                symbol, function_rva
            )
        struct.pack_into(
            "<I", data, functions_offset + index * 4, current_function_rva
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


def test_elf_rejects_dynsym_not_referenced_by_pt_dynamic():
    data = build_elf(loader_symbols=False)

    with pytest.raises(ValueError, match="DT_SYMTAB"):
        verifier.verify_native_target(
            data,
            "x86_64-unknown-linux-gnu",
            "libpaimon_mosaic_jni.so",
        )


def test_elf_rejects_dynsym_entries_beyond_dt_hash_symbol_count():
    data = build_elf(hash_symbol_count=1)

    with pytest.raises(ValueError, match="DT_HASH.*symbol count"):
        verifier.verify_native_target(
            data,
            "x86_64-unknown-linux-gnu",
            "libpaimon_mosaic_jni.so",
        )


def test_elf_does_not_accept_exports_unreachable_from_dt_hash():
    data = build_elf(hash_reachable=False)

    with pytest.raises(ValueError, match="missing expected Mosaic JNI exports"):
        verifier.verify_native_target(
            data,
            "x86_64-unknown-linux-gnu",
            "libpaimon_mosaic_jni.so",
        )


def test_elf_accepts_exports_reachable_from_dt_gnu_hash():
    verifier.verify_native_target(
        build_elf(hash_style="gnu"),
        "x86_64-unknown-linux-gnu",
        "libpaimon_mosaic_jni.so",
    )


def test_elf_does_not_accept_exports_unreachable_from_dt_gnu_hash():
    data = build_elf(hash_style="gnu", hash_reachable=False)

    with pytest.raises(ValueError, match="missing expected Mosaic JNI exports"):
        verifier.verify_native_target(
            data,
            "x86_64-unknown-linux-gnu",
            "libpaimon_mosaic_jni.so",
        )


def test_elf_requires_exports_reachable_from_each_loader_hash():
    data = build_elf(
        hash_style="both",
        hash_reachable=True,
        gnu_hash_reachable=False,
    )

    with pytest.raises(ValueError, match="missing expected Mosaic JNI exports"):
        verifier.verify_native_target(
            data,
            "x86_64-unknown-linux-gnu",
            "libpaimon_mosaic_jni.so",
        )


def test_elf_does_not_accept_object_symbols_as_function_exports():
    data = build_elf(symbol_info=0x11)

    with pytest.raises(ValueError, match="missing expected Mosaic JNI exports"):
        verifier.verify_native_target(
            data,
            "x86_64-unknown-linux-gnu",
            "libpaimon_mosaic_jni.so",
        )


def test_elf_rejects_function_export_outside_load_segments():
    data = build_elf(symbol_value=0xFFFFFFFF)

    with pytest.raises(ValueError, match="function.*not mapped"):
        verifier.verify_native_target(
            data,
            "x86_64-unknown-linux-gnu",
            "libpaimon_mosaic_jni.so",
        )


def test_elf_rejects_function_export_in_non_executable_segment():
    data = build_elf(load_flags=4)

    with pytest.raises(ValueError, match="function.*not mapped"):
        verifier.verify_native_target(
            data,
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


def test_pe_rejects_named_export_with_unmapped_function_rva():
    data = build_pe(function_rva=0xFFFFFFFF)

    with pytest.raises(ValueError, match="function RVA.*not mapped"):
        verifier.verify_native_target(
            data,
            "x86_64-pc-windows-msvc",
            "paimon_mosaic_jni.dll",
        )


def test_pe_rejects_named_export_in_non_executable_section():
    data = build_pe(section_characteristics=0x40000040)

    with pytest.raises(ValueError, match="missing expected Mosaic JNI exports"):
        verifier.verify_native_target(
            data,
            "x86_64-pc-windows-msvc",
            "paimon_mosaic_jni.dll",
        )


def test_pe_rejects_named_forwarded_export():
    data = build_pe(forwarder="other_module.mosaic_writer_open")

    with pytest.raises(ValueError, match="missing expected Mosaic JNI exports"):
        verifier.verify_native_target(
            data,
            "x86_64-pc-windows-msvc",
            "paimon_mosaic_jni.dll",
        )


def test_pe_accepts_required_functions_with_unrelated_data_export():
    unrelated = "unrelated_data"
    data = build_pe(
        symbols=JNI_SYMBOLS | {unrelated},
        symbol_rvas={unrelated: 0x1350},
    )

    verifier.verify_native_target(
        data,
        "x86_64-pc-windows-msvc",
        "paimon_mosaic_jni.dll",
    )


def test_pe_accepts_required_functions_with_unrelated_forwarder():
    unrelated = "unrelated_forwarder"
    data = build_pe(
        symbols=JNI_SYMBOLS | {unrelated},
        symbol_forwarders={unrelated: "KERNEL32.Sleep"},
    )

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
