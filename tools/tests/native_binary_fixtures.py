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

from native_binary import MOSAIC_SYMBOL_FAMILIES


JNI_SYMBOLS = MOSAIC_SYMBOL_FAMILIES["JNI"]
FFI_SYMBOLS = MOSAIC_SYMBOL_FAMILIES["FFI"]


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
