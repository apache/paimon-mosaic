#!/usr/bin/env python3

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

"""Validate native-library format, architecture, structure, and exports."""

from __future__ import annotations

import struct
from dataclasses import dataclass


TARGET_ARCHITECTURE = {
    "x86_64-unknown-linux-gnu": ("ELF", "x86_64"),
    "aarch64-unknown-linux-gnu": ("ELF", "aarch64"),
    "aarch64-apple-darwin": ("Mach-O", "aarch64"),
    "x86_64-pc-windows-msvc": ("PE", "x86_64"),
}

MACHINE_ARCHITECTURE = {
    62: "x86_64",
    183: "aarch64",
}
PE_MACHINE_ARCHITECTURE = {
    0x8664: "x86_64",
    0xAA64: "aarch64",
}
MACHO_CPU_ARCHITECTURE = {
    0x01000007: "x86_64",
    0x0100000C: "aarch64",
}

# A Mosaic native library exports on the order of a hundred symbols; the bound
# exists to keep hostile input from driving quadratic hash-membership work.
MAX_DYNAMIC_SYMBOLS = 100_000

MOSAIC_SYMBOL_FAMILIES = {
    "JNI": {
        "Java_org_apache_paimon_mosaic_NativeLib_nativeReaderExportSchema",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeReaderFree",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeReaderNumRowGroups",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeReaderOpen",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeReaderOpenRowGroup",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeReaderRowGroupNumRows",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeReaderRowGroupStatMaxs",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeReaderRowGroupStatMins",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeReaderRowGroupStatNames",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeReaderRowGroupStatNullCounts",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeReaderSetProjection",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeRowGroupReaderFree",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeRowGroupReaderNumRows",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeRowGroupReaderReadColumns",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeWriterClose",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeWriterEstimatedSize",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeWriterFree",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeWriterNumRowGroups",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeWriterOpen",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeWriterRowGroupStatMaxs",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeWriterRowGroupStatMins",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeWriterRowGroupStatNames",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeWriterRowGroupStatNullCounts",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeWriterWriteBatch",
    },
    "FFI": {
        "mosaic_last_error",
        "mosaic_reader_export_schema",
        "mosaic_reader_free",
        "mosaic_reader_num_row_groups",
        "mosaic_reader_open",
        "mosaic_reader_open_row_group",
        "mosaic_reader_row_group_num_rows",
        "mosaic_reader_row_group_num_stats",
        "mosaic_reader_row_group_stats",
        "mosaic_reader_set_projection",
        "mosaic_record_batch_export",
        "mosaic_record_batch_free",
        "mosaic_record_batch_num_columns",
        "mosaic_record_batch_num_rows",
        "mosaic_row_group_reader_free",
        "mosaic_row_group_reader_num_rows",
        "mosaic_row_group_reader_read_columns",
        "mosaic_writer_close",
        "mosaic_writer_estimated_file_size",
        "mosaic_writer_free",
        "mosaic_writer_num_row_groups",
        "mosaic_writer_open",
        "mosaic_writer_options_default",
        "mosaic_writer_row_group_num_stats",
        "mosaic_writer_row_group_stats",
        "mosaic_writer_write_batch",
    },
}


@dataclass(frozen=True)
class NativeBinary:
    binary_format: str
    architectures: frozenset[str]
    exported_symbols: frozenset[str]


@dataclass(frozen=True)
class ElfSection:
    section_type: int
    flags: int
    address: int
    offset: int
    size: int
    link: int
    entry_size: int


def require_range(data: bytes, offset: int, size: int, description: str) -> None:
    if (
        offset < 0
        or size < 0
        or offset > len(data)
        or size > len(data) - offset
    ):
        raise ValueError(f"{description} is out of bounds")


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def c_string_bytes(
    data: bytes, offset: int, limit: int, description: str
) -> bytes:
    if offset < 0 or offset >= limit or limit > len(data):
        raise ValueError(f"{description} is out of bounds")
    terminator = data.find(b"\0", offset, limit)
    if terminator < 0:
        raise ValueError(f"{description} is not null-terminated")
    return data[offset:terminator]


def ascii_symbol(raw_name: bytes) -> str | None:
    try:
        return raw_name.decode("ascii")
    except UnicodeDecodeError:
        return None


def elf_virtual_range(
    data: bytes,
    address: int,
    size: int,
    load_segments: list[tuple[int, int, int, int]],
    description: str,
) -> int:
    mapped_offsets = set()
    memory_mapped = False
    for file_offset, virtual_address, file_size, memory_size in load_segments:
        if address < virtual_address:
            continue
        delta = address - virtual_address
        if delta > memory_size or size > memory_size - delta:
            continue
        memory_mapped = True
        if delta <= file_size and size <= file_size - delta:
            mapped_offsets.add(file_offset + delta)

    if not mapped_offsets:
        if memory_mapped:
            raise ValueError(f"{description} is not file-backed")
        raise ValueError(f"{description} is not mapped by an ELF PT_LOAD segment")
    if len(mapped_offsets) != 1:
        raise ValueError(f"{description} has ambiguous ELF PT_LOAD mappings")

    offset = mapped_offsets.pop()
    require_range(data, offset, size, description)
    return offset


def elf_loader_section(
    data: bytes,
    sections: list[ElfSection],
    load_segments: list[tuple[int, int, int, int]],
    address: int,
    section_type: int,
    minimum_size: int,
    linked_section: int | None,
    description: str,
    section_name: str,
) -> tuple[int, ElfSection]:
    offset = elf_virtual_range(
        data, address, minimum_size, load_segments, description
    )
    matches = [
        (index, section)
        for index, section in enumerate(sections)
        if section.section_type == section_type
        and section.address == address
        and section.offset == offset
    ]
    if not matches:
        raise ValueError(
            f"{description} does not reference an {section_name} section"
        )
    if len(matches) != 1:
        raise ValueError(f"{description} references multiple {section_name} sections")

    index, section = matches[0]
    if not section.flags & 0x2:
        raise ValueError(f"{description} section is not allocated")
    if section.size < minimum_size:
        raise ValueError(f"{description} is truncated")
    if linked_section is not None and section.link != linked_section:
        raise ValueError(f"{description} section does not link to DT_SYMTAB")
    if (
        elf_virtual_range(
            data, address, section.size, load_segments, description
        )
        != section.offset
    ):
        raise ValueError(f"{description} has inconsistent file mapping")
    return index, section


def elf_sysv_hash(name: bytes) -> int:
    value = 0
    for byte in name:
        value = (value << 4) + byte
        high = value & 0xF0000000
        if high:
            value ^= high >> 24
            value &= ~high
    return value & 0xFFFFFFFF


def elf_gnu_hash(name: bytes) -> int:
    value = 5381
    for byte in name:
        value = (value * 33 + byte) & 0xFFFFFFFF
    return value


@dataclass(frozen=True)
class ElfSysvHash:
    buckets: tuple[int, ...]
    chains: tuple[int, ...]

    @property
    def symbol_count(self) -> int:
        return len(self.chains)

    def contains(self, symbol_index: int, name: bytes) -> bool:
        index = self.buckets[elf_sysv_hash(name) % len(self.buckets)]
        while index:
            if index == symbol_index:
                return True
            index = self.chains[index]
        return False


@dataclass(frozen=True)
class ElfGnuHash:
    symbol_offset: int
    bloom_shift: int
    bloom: tuple[int, ...]
    buckets: tuple[int, ...]
    chains: tuple[int, ...]
    symbol_count: int

    def contains(self, symbol_index: int, name: bytes) -> bool:
        name_hash = elf_gnu_hash(name)
        bloom_word = self.bloom[(name_hash // 64) % len(self.bloom)]
        bloom_mask = (1 << (name_hash % 64)) | (
            1 << ((name_hash >> self.bloom_shift) % 64)
        )
        if bloom_word & bloom_mask != bloom_mask:
            return False

        index = self.buckets[name_hash % len(self.buckets)]
        if (
            index == 0
            or symbol_index < index
            or symbol_index < self.symbol_offset
        ):
            return False
        while True:
            chain_index = index - self.symbol_offset
            if chain_index >= len(self.chains):
                return False
            chain_hash = self.chains[chain_index]
            if index == symbol_index:
                return (chain_hash | 1) == (name_hash | 1)
            if chain_hash & 1:
                return False
            index += 1


def parse_elf_sysv_hash(data: bytes, section: ElfSection) -> ElfSysvHash:
    bucket_count, symbol_count = struct.unpack_from("<II", data, section.offset)
    if bucket_count == 0 or symbol_count == 0:
        raise ValueError("ELF DT_HASH has invalid bucket or symbol count")
    if section.size != 8 + (bucket_count + symbol_count) * 4:
        raise ValueError("ELF DT_HASH has an inconsistent table size")

    buckets_offset = section.offset + 8
    chains_offset = buckets_offset + bucket_count * 4
    buckets = struct.unpack_from(f"<{bucket_count}I", data, buckets_offset)
    chains = struct.unpack_from(f"<{symbol_count}I", data, chains_offset)
    if chains[0] != 0:
        raise ValueError("ELF DT_HASH chain zero is not a terminator")
    if any(index >= symbol_count for index in buckets):
        raise ValueError("ELF DT_HASH bucket index is out of bounds")
    if any(index >= symbol_count for index in chains):
        raise ValueError("ELF DT_HASH chain index is out of bounds")
    # Each dynamic symbol belongs to exactly one bucket's chain. Recording the
    # owning bucket keeps this linear and distinguishes a cycle within one chain
    # from two buckets aliasing the same node, which would let contains() resolve
    # a symbol through the wrong bucket.
    owner: dict[int, int] = {}
    for bucket_index, bucket in enumerate(buckets):
        index = bucket
        while index:
            previous = owner.get(index)
            if previous == bucket_index:
                raise ValueError("ELF DT_HASH contains a chain cycle")
            if previous is not None:
                raise ValueError("ELF DT_HASH bucket chains alias")
            owner[index] = bucket_index
            index = chains[index]
    return ElfSysvHash(buckets, chains)


def parse_elf_gnu_hash(data: bytes, section: ElfSection) -> ElfGnuHash:
    (
        bucket_count,
        symbol_offset,
        bloom_count,
        bloom_shift,
    ) = struct.unpack_from("<IIII", data, section.offset)
    if bucket_count == 0 or bloom_count == 0:
        raise ValueError("ELF DT_GNU_HASH has an invalid header")

    bloom_offset = section.offset + 16
    buckets_offset = bloom_offset + bloom_count * 8
    chains_offset = buckets_offset + bucket_count * 4
    section_end = section.offset + section.size
    if chains_offset > section_end or (section_end - chains_offset) % 4:
        raise ValueError("ELF DT_GNU_HASH has an inconsistent table size")

    bloom = struct.unpack_from(f"<{bloom_count}Q", data, bloom_offset)
    buckets = struct.unpack_from(f"<{bucket_count}I", data, buckets_offset)
    chain_count = (section_end - chains_offset) // 4
    chains = struct.unpack_from(f"<{chain_count}I", data, chains_offset)
    symbol_count = symbol_offset
    verified = set()
    for bucket in buckets:
        if bucket == 0:
            continue
        if bucket < symbol_offset:
            raise ValueError("ELF DT_GNU_HASH bucket precedes the symbol offset")
        chain_index = bucket - symbol_offset
        while chain_index not in verified:
            if chain_index >= chain_count:
                raise ValueError("ELF DT_GNU_HASH chain is not terminated")
            verified.add(chain_index)
            symbol_count = max(symbol_count, symbol_offset + chain_index + 1)
            if chains[chain_index] & 1:
                break
            chain_index += 1
    return ElfGnuHash(
        symbol_offset,
        bloom_shift,
        bloom,
        buckets,
        chains,
        symbol_count,
    )


def parse_elf(data: bytes) -> NativeBinary | None:
    if not data.startswith(b"\x7fELF"):
        return None
    if len(data) < 64:
        raise ValueError("truncated ELF header")
    if data[4] != 2:
        raise ValueError(f"unsupported ELF class {data[4]}")
    if data[5] != 1:
        raise ValueError(f"unsupported ELF byte order {data[5]}")
    if data[6] != 1:
        raise ValueError(f"unsupported ELF identification version {data[6]}")

    (
        file_type,
        machine,
        version,
        _entry,
        program_offset,
        section_offset,
        _flags,
        header_size,
        program_entry_size,
        program_count,
        section_entry_size,
        section_count,
        section_names_index,
    ) = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)

    if file_type != 3:
        raise ValueError(f"ELF file type {file_type} is not ET_DYN")
    architecture = MACHINE_ARCHITECTURE.get(machine)
    if architecture is None:
        raise ValueError(f"unsupported ELF machine {machine}")
    if version != 1:
        raise ValueError(f"unsupported ELF version {version}")
    if header_size != 64:
        raise ValueError(f"invalid ELF header size {header_size}")
    if program_count in (0, 0xFFFF):
        raise ValueError(f"invalid ELF program header count {program_count}")
    if program_entry_size != 56:
        raise ValueError(
            f"invalid ELF program header entry size {program_entry_size}"
        )
    if program_offset < header_size:
        raise ValueError("ELF program header table overlaps the file header")
    require_range(
        data,
        program_offset,
        program_count * program_entry_size,
        "ELF program header table",
    )

    has_load_segment = False
    load_segments = []
    executable_load_segments = []
    dynamic_segments = []
    for index in range(program_count):
        offset = program_offset + index * program_entry_size
        (
            segment_type,
            segment_flags,
            file_offset,
            virtual_address,
            _physical_address,
            file_size,
            memory_size,
            alignment,
        ) = struct.unpack_from("<IIQQQQQQ", data, offset)
        if alignment not in (0, 1) and not is_power_of_two(alignment):
            raise ValueError(
                f"ELF program header {index} has invalid alignment {alignment}"
            )
        if file_size:
            require_range(
                data,
                file_offset,
                file_size,
                f"ELF program header {index} contents",
            )
        if segment_type == 1:
            if file_size > memory_size:
                raise ValueError(
                    f"ELF load segment {index} is larger on disk than in memory"
                )
            if (
                alignment not in (0, 1)
                and virtual_address % alignment != file_offset % alignment
            ):
                raise ValueError(
                    f"ELF load segment {index} has inconsistent alignment"
                )
            has_load_segment = has_load_segment or file_size > 0
            load_segments.append(
                (file_offset, virtual_address, file_size, memory_size)
            )
            if segment_flags & 0x1:
                executable_load_segments.append(
                    (file_offset, virtual_address, file_size, memory_size)
                )
        elif segment_type == 2:
            if file_size == 0 or file_size % 16:
                raise ValueError(f"ELF dynamic segment {index} has invalid size")
            if file_size > memory_size:
                raise ValueError(
                    f"ELF dynamic segment {index} is larger on disk than in memory"
                )
            dynamic_segments.append(
                (index, file_offset, virtual_address, file_size)
            )
        elif segment_type == 3:
            raise ValueError("ELF ET_DYN file contains PT_INTERP and is executable")

    if not has_load_segment:
        raise ValueError("ELF shared object has no non-empty PT_LOAD segment")
    if not dynamic_segments:
        raise ValueError("ELF shared object has no PT_DYNAMIC segment")
    if len(dynamic_segments) != 1:
        raise ValueError("ELF shared object has multiple PT_DYNAMIC segments")

    (
        dynamic_index,
        dynamic_offset,
        dynamic_address,
        dynamic_size,
    ) = dynamic_segments[0]
    mapped_dynamic_offset = elf_virtual_range(
        data,
        dynamic_address,
        dynamic_size,
        load_segments,
        f"ELF dynamic segment {dynamic_index}",
    )
    if mapped_dynamic_offset != dynamic_offset:
        raise ValueError(
            f"ELF dynamic segment {dynamic_index} has inconsistent file mapping"
        )

    required_dynamic_tag_names = {
        6: "DT_SYMTAB",
        5: "DT_STRTAB",
        10: "DT_STRSZ",
        11: "DT_SYMENT",
    }
    hash_dynamic_tag_names = {
        4: "DT_HASH",
        0x6FFFFEF5: "DT_GNU_HASH",
    }
    dynamic_tag_names = {
        **required_dynamic_tag_names,
        **hash_dynamic_tag_names,
    }
    dynamic_values = {}
    dynamic_terminated = False
    for entry_index in range(dynamic_size // 16):
        tag, value = struct.unpack_from(
            "<qQ", data, dynamic_offset + entry_index * 16
        )
        if tag == 0:
            dynamic_terminated = True
            break
        if tag in dynamic_tag_names:
            if tag in dynamic_values:
                raise ValueError(
                    f"ELF PT_DYNAMIC contains duplicate {dynamic_tag_names[tag]}"
                )
            dynamic_values[tag] = value
    if not dynamic_terminated:
        raise ValueError("ELF PT_DYNAMIC is not terminated by DT_NULL")
    for tag, name in required_dynamic_tag_names.items():
        if tag not in dynamic_values:
            raise ValueError(f"ELF PT_DYNAMIC is missing {name}")
    if not any(tag in dynamic_values for tag in hash_dynamic_tag_names):
        raise ValueError("ELF PT_DYNAMIC is missing DT_HASH or DT_GNU_HASH")

    string_address = dynamic_values[5]
    symbol_address = dynamic_values[6]
    string_size = dynamic_values[10]
    symbol_entry_size = dynamic_values[11]
    if string_size == 0:
        raise ValueError("ELF DT_STRSZ is zero")
    if symbol_entry_size != 24:
        raise ValueError(f"ELF DT_SYMENT has invalid size {symbol_entry_size}")
    string_offset = elf_virtual_range(
        data,
        string_address,
        string_size,
        load_segments,
        "ELF DT_STRTAB",
    )
    symbol_offset = elf_virtual_range(
        data,
        symbol_address,
        symbol_entry_size,
        load_segments,
        "ELF DT_SYMTAB",
    )

    if section_offset == 0:
        if section_count != 0 or section_names_index != 0:
            raise ValueError("invalid ELF section header metadata")
        return NativeBinary("ELF", frozenset({architecture}), frozenset())
    if section_count in (0, 0xFFFF):
        raise ValueError(f"invalid ELF section header count {section_count}")
    if section_names_index == 0xFFFF:
        raise ValueError("extended ELF section-name indexes are unsupported")
    if section_entry_size != 64:
        raise ValueError(
            f"invalid ELF section header entry size {section_entry_size}"
        )
    if section_offset < header_size:
        raise ValueError("ELF section header table overlaps the file header")
    require_range(
        data,
        section_offset,
        section_count * section_entry_size,
        "ELF section header table",
    )
    if section_names_index >= section_count:
        raise ValueError(
            f"invalid ELF section-name string table index {section_names_index}"
        )

    sections = []
    for index in range(section_count):
        offset = section_offset + index * section_entry_size
        (
            _name,
            section_type,
            section_flags,
            address,
            file_offset,
            size,
            link,
            _info,
            alignment,
            entry_size,
        ) = struct.unpack_from("<IIQQQQIIQQ", data, offset)
        if alignment not in (0, 1) and not is_power_of_two(alignment):
            raise ValueError(
                f"ELF section header {index} has invalid alignment {alignment}"
            )
        if section_type != 8 and size:
            require_range(
                data,
                file_offset,
                size,
                f"ELF section header {index} contents",
            )
        sections.append(
            ElfSection(
                section_type,
                section_flags,
                address,
                file_offset,
                size,
                link,
                entry_size,
            )
        )

    matching_symbol_sections = [
        (index, section)
        for index, section in enumerate(sections)
        if section.section_type == 11
        and section.address == symbol_address
        and section.offset == symbol_offset
    ]
    if not matching_symbol_sections:
        raise ValueError("ELF DT_SYMTAB does not reference an SHT_DYNSYM section")
    if len(matching_symbol_sections) != 1:
        raise ValueError("ELF DT_SYMTAB references multiple SHT_DYNSYM sections")

    symbol_section_index, symbol_section = matching_symbol_sections[0]
    if not symbol_section.flags & 0x2:
        raise ValueError(
            f"ELF dynamic symbol section {symbol_section_index} is not allocated"
        )
    if (
        symbol_section.entry_size != symbol_entry_size
        or symbol_section.size < symbol_entry_size
        or symbol_section.size % symbol_entry_size
    ):
        raise ValueError(
            f"ELF dynamic symbol section {symbol_section_index} is malformed"
        )
    # Membership in the loader hash is checked per symbol, and a table that
    # funnels every symbol into one bucket makes each check linear. Bound the
    # entry count so a crafted library cannot force quadratic work.
    if symbol_section.size // symbol_entry_size > MAX_DYNAMIC_SYMBOLS:
        raise ValueError(
            f"ELF dynamic symbol section {symbol_section_index} declares more "
            f"than {MAX_DYNAMIC_SYMBOLS} symbols"
        )
    if (
        elf_virtual_range(
            data,
            symbol_address,
            symbol_section.size,
            load_segments,
            f"ELF dynamic symbol section {symbol_section_index}",
        )
        != symbol_section.offset
    ):
        raise ValueError(
            f"ELF dynamic symbol section {symbol_section_index} "
            "has inconsistent file mapping"
        )
    loader_hashes = []
    if 4 in dynamic_values:
        hash_address = dynamic_values[4]
        _hash_section_index, hash_section = elf_loader_section(
            data,
            sections,
            load_segments,
            hash_address,
            5,
            8,
            symbol_section_index,
            "ELF DT_HASH",
            "SHT_HASH",
        )
        sysv_hash = parse_elf_sysv_hash(data, hash_section)
        if symbol_section.size != sysv_hash.symbol_count * symbol_entry_size:
            raise ValueError(
                "ELF DT_HASH symbol count does not match the SHT_DYNSYM size"
            )
        loader_hashes.append(sysv_hash)
    if 0x6FFFFEF5 in dynamic_values:
        hash_address = dynamic_values[0x6FFFFEF5]
        _hash_section_index, hash_section = elf_loader_section(
            data,
            sections,
            load_segments,
            hash_address,
            0x6FFFFFF6,
            16,
            symbol_section_index,
            "ELF DT_GNU_HASH",
            "SHT_GNU_HASH",
        )
        gnu_hash = parse_elf_gnu_hash(data, hash_section)
        if gnu_hash.symbol_count * symbol_entry_size > symbol_section.size:
            raise ValueError(
                "ELF DT_GNU_HASH symbol count exceeds the SHT_DYNSYM size"
            )
        loader_hashes.append(gnu_hash)
    if symbol_section.link >= section_count:
        raise ValueError(
            f"ELF dynamic symbol section {symbol_section_index} "
            "has invalid string-table link"
        )

    string_section = sections[symbol_section.link]
    if string_section.section_type != 3:
        raise ValueError(
            f"ELF dynamic symbol section {symbol_section_index} "
            "does not link to a string table"
        )
    if not string_section.flags & 0x2:
        raise ValueError(
            f"ELF dynamic string section {symbol_section.link} is not allocated"
        )
    if (
        string_section.address != string_address
        or string_section.offset != string_offset
        or string_section.size != string_size
    ):
        raise ValueError(
            f"ELF dynamic symbol section {symbol_section_index} "
            "does not link to DT_STRTAB"
        )
    if data[string_offset] != 0:
        raise ValueError("ELF DT_STRTAB does not start with a null byte")
    string_limit = string_offset + string_size

    exported_symbols = set()
    for symbol_index in range(symbol_section.size // symbol_entry_size):
        entry_offset = symbol_section.offset + symbol_index * symbol_entry_size
        (
            name_offset,
            info,
            other,
            symbol_section_index_value,
            value,
            symbol_size,
        ) = struct.unpack_from("<IBBHQQ", data, entry_offset)
        if name_offset >= string_size:
            raise ValueError(
                f"ELF dynamic symbol {symbol_index} has an invalid name offset"
            )
        if name_offset == 0:
            continue
        binding = info >> 4
        symbol_type = info & 0x0F
        visibility = other & 0x03
        if (
            binding not in (1, 2)
            or symbol_type not in (2, 10)
            or symbol_section_index_value == 0
            or visibility not in (0, 3)
        ):
            continue
        elf_virtual_range(
            data,
            value,
            max(symbol_size, 1),
            executable_load_segments,
            f"ELF dynamic symbol {symbol_index} function",
        )
        raw_name = c_string_bytes(
            data,
            string_offset + name_offset,
            string_limit,
            f"ELF dynamic symbol {symbol_index} name",
        )
        name = ascii_symbol(raw_name)
        if name and all(
            loader_hash.contains(symbol_index, raw_name)
            for loader_hash in loader_hashes
        ):
            exported_symbols.add(name)

    return NativeBinary(
        "ELF", frozenset({architecture}), frozenset(exported_symbols)
    )


def pe_rva_span(
    rva: int,
    sections: list[tuple[int, int, int, int, int]],
    headers_size: int,
    data_size: int,
    description: str,
) -> tuple[int, int]:
    if rva < headers_size:
        if rva >= data_size:
            raise ValueError(f"{description} RVA is out of bounds")
        return rva, min(headers_size, data_size) - rva

    for (
        virtual_address,
        virtual_size,
        file_offset,
        file_size,
        _characteristics,
    ) in sections:
        mapped_size = max(virtual_size, file_size)
        if virtual_address <= rva < virtual_address + mapped_size:
            delta = rva - virtual_address
            if delta >= file_size:
                raise ValueError(f"{description} RVA is not file-backed")
            return file_offset + delta, file_size - delta
    raise ValueError(f"{description} RVA is not mapped by a PE section")


def pe_rva_range(
    data: bytes,
    rva: int,
    size: int,
    sections: list[tuple[int, int, int, int, int]],
    headers_size: int,
    description: str,
) -> int:
    offset, available = pe_rva_span(
        rva, sections, headers_size, len(data), description
    )
    if size > available:
        raise ValueError(f"{description} is out of bounds")
    require_range(data, offset, size, description)
    return offset


def pe_section_rva_range(
    data: bytes,
    rva: int,
    size: int,
    sections: list[tuple[int, int, int, int, int]],
    description: str,
) -> tuple[int, bool]:
    for (
        virtual_address,
        virtual_size,
        file_offset,
        file_size,
        characteristics,
    ) in sections:
        mapped_size = max(virtual_size, file_size)
        if virtual_address <= rva < virtual_address + mapped_size:
            delta = rva - virtual_address
            if delta >= file_size or size > file_size - delta:
                raise ValueError(f"{description} is not file-backed")
            offset = file_offset + delta
            require_range(data, offset, size, description)
            return offset, bool(characteristics & 0x20000000)
    raise ValueError(f"{description} is not mapped by a PE section")


def parse_pe(data: bytes) -> NativeBinary | None:
    if not data.startswith(b"MZ"):
        return None
    if len(data) < 0x40:
        raise ValueError("truncated DOS header")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_offset < 0x40:
        raise ValueError(f"invalid PE header offset 0x{pe_offset:x}")
    require_range(data, pe_offset, 24, "PE signature and COFF header")
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise ValueError("invalid PE signature")

    (
        machine,
        section_count,
        _timestamp,
        _symbol_table_offset,
        _symbol_count,
        optional_size,
        characteristics,
    ) = struct.unpack_from("<HHIIIHH", data, pe_offset + 4)
    architecture = PE_MACHINE_ARCHITECTURE.get(machine)
    if architecture is None:
        raise ValueError(f"unsupported PE machine 0x{machine:04x}")
    if section_count == 0:
        raise ValueError("PE image has no sections")
    if not characteristics & 0x2000:
        raise ValueError("PE image does not have the DLL characteristic")

    optional_offset = pe_offset + 24
    if optional_size < 112:
        raise ValueError(f"truncated PE optional header ({optional_size} bytes)")
    require_range(data, optional_offset, optional_size, "PE optional header")
    optional_magic = struct.unpack_from("<H", data, optional_offset)[0]
    if optional_magic != 0x20B:
        raise ValueError(
            f"PE optional header magic 0x{optional_magic:04x} is not PE32+"
        )

    section_alignment = struct.unpack_from("<I", data, optional_offset + 32)[0]
    file_alignment = struct.unpack_from("<I", data, optional_offset + 36)[0]
    image_size = struct.unpack_from("<I", data, optional_offset + 56)[0]
    headers_size = struct.unpack_from("<I", data, optional_offset + 60)[0]
    directory_count = struct.unpack_from("<I", data, optional_offset + 108)[0]
    available_directories = (optional_size - 112) // 8
    if directory_count > available_directories:
        raise ValueError(
            "PE optional header does not contain all declared data directories"
        )
    if not is_power_of_two(section_alignment):
        raise ValueError(f"invalid PE section alignment {section_alignment}")
    if not is_power_of_two(file_alignment):
        raise ValueError(f"invalid PE file alignment {file_alignment}")
    if section_alignment < file_alignment:
        raise ValueError("PE section alignment is smaller than file alignment")

    section_table_offset = optional_offset + optional_size
    section_table_size = section_count * 40
    require_range(
        data, section_table_offset, section_table_size, "PE section table"
    )
    section_table_end = section_table_offset + section_table_size
    if headers_size < section_table_end or headers_size > len(data):
        raise ValueError(f"invalid PE SizeOfHeaders {headers_size}")
    if image_size < headers_size:
        raise ValueError(f"invalid PE SizeOfImage {image_size}")

    sections = []
    has_file_backed_section = False
    for index in range(section_count):
        offset = section_table_offset + index * 40
        (
            _name,
            virtual_size,
            virtual_address,
            file_size,
            file_offset,
            _relocations,
            _line_numbers,
            _relocation_count,
            _line_number_count,
            section_characteristics,
        ) = struct.unpack_from("<8sIIIIIIHHI", data, offset)
        if virtual_address % section_alignment:
            raise ValueError(f"PE section {index} has an unaligned virtual address")
        if virtual_address + max(virtual_size, file_size) > image_size:
            raise ValueError(f"PE section {index} exceeds SizeOfImage")
        if file_size:
            if file_offset < headers_size or file_offset % file_alignment:
                raise ValueError(f"PE section {index} has an invalid file offset")
            require_range(
                data, file_offset, file_size, f"PE section {index} contents"
            )
            has_file_backed_section = True
        sections.append(
            (
                virtual_address,
                virtual_size,
                file_offset,
                file_size,
                section_characteristics,
            )
        )
    if not has_file_backed_section:
        raise ValueError("PE DLL has no file-backed sections")

    exported_symbols = set()
    if directory_count:
        export_rva, export_size = struct.unpack_from(
            "<II", data, optional_offset + 112
        )
        if bool(export_rva) != bool(export_size):
            raise ValueError("PE export directory has an incomplete RVA/size pair")
        if export_rva:
            if export_size < 40:
                raise ValueError("truncated PE export directory")
            export_offset = pe_rva_range(
                data,
                export_rva,
                export_size,
                sections,
                headers_size,
                "PE export directory",
            )
            (
                _export_flags,
                _export_timestamp,
                _major_version,
                _minor_version,
                module_name_rva,
                _ordinal_base,
                function_count,
                name_count,
                functions_rva,
                names_rva,
                ordinals_rva,
            ) = struct.unpack_from("<IIHHIIIIIII", data, export_offset)
            if name_count > function_count:
                raise ValueError(
                    "PE export directory has more names than functions"
                )
            if function_count:
                functions_offset = pe_rva_range(
                    data,
                    functions_rva,
                    function_count * 4,
                    sections,
                    headers_size,
                    "PE export address table",
                )
            else:
                functions_offset = 0
            if name_count:
                names_offset = pe_rva_range(
                    data,
                    names_rva,
                    name_count * 4,
                    sections,
                    headers_size,
                    "PE export name table",
                )
                ordinals_offset = pe_rva_range(
                    data,
                    ordinals_rva,
                    name_count * 2,
                    sections,
                    headers_size,
                    "PE export ordinal table",
                )
            else:
                names_offset = 0
                ordinals_offset = 0

            if module_name_rva:
                module_offset, module_available = pe_rva_span(
                    module_name_rva,
                    sections,
                    headers_size,
                    len(data),
                    "PE export module name",
                )
                c_string_bytes(
                    data,
                    module_offset,
                    module_offset + module_available,
                    "PE export module name",
                )

            previous_name = None
            for index in range(name_count):
                ordinal = struct.unpack_from(
                    "<H", data, ordinals_offset + index * 2
                )[0]
                if ordinal >= function_count:
                    raise ValueError(
                        f"PE export name {index} has invalid ordinal {ordinal}"
                    )
                function_rva = struct.unpack_from(
                    "<I", data, functions_offset + ordinal * 4
                )[0]
                if function_rva == 0:
                    raise ValueError(
                        f"PE export name {index} points to a null function RVA"
                    )
                name_rva = struct.unpack_from(
                    "<I", data, names_offset + index * 4
                )[0]
                name_offset, name_available = pe_rva_span(
                    name_rva,
                    sections,
                    headers_size,
                    len(data),
                    f"PE export name {index}",
                )
                raw_name = c_string_bytes(
                    data,
                    name_offset,
                    name_offset + name_available,
                    f"PE export name {index}",
                )
                name = ascii_symbol(raw_name)
                if name is None:
                    raise ValueError(f"PE export name {index} is not ASCII")
                if previous_name is not None and raw_name <= previous_name:
                    raise ValueError(
                        "PE export names are not strictly increasing"
                    )
                previous_name = raw_name
                if export_rva <= function_rva < export_rva + export_size:
                    forwarder_offset = export_offset + function_rva - export_rva
                    forwarder = ascii_symbol(
                        c_string_bytes(
                            data,
                            forwarder_offset,
                            export_offset + export_size,
                            f"PE export name {index} forwarder",
                        )
                    )
                    if forwarder is None:
                        raise ValueError(
                            f"PE export name {index} forwarder is not ASCII"
                        )
                    continue
                _function_offset, executable = pe_section_rva_range(
                    data,
                    function_rva,
                    1,
                    sections,
                    f"PE export name {index} function RVA",
                )
                if executable:
                    exported_symbols.add(name)

    return NativeBinary(
        "PE", frozenset({architecture}), frozenset(exported_symbols)
    )


def macho_uleb128(
    data: bytes, offset: int, limit: int, description: str
) -> tuple[int, int]:
    value = 0
    for index in range(10):
        if offset >= limit:
            raise ValueError(f"{description} ULEB128 is truncated")
        byte = data[offset]
        offset += 1
        if index == 9 and byte > 1:
            raise ValueError(f"{description} ULEB128 overflows 64 bits")
        value |= (byte & 0x7F) << (index * 7)
        if not byte & 0x80:
            return value, offset
    raise ValueError(f"{description} ULEB128 overflows 64 bits")


def parse_macho_export_trie(
    data: bytes, trie_offset: int, trie_size: int
) -> frozenset[str]:
    require_range(data, trie_offset, trie_size, "Mach-O export trie")
    if trie_size == 0:
        return frozenset()

    trie_end = trie_offset + trie_size
    exported_symbols = set()
    active_nodes = set()
    visited_nodes = set()
    stack = [(False, 0, b"")]
    while stack:
        leaving, node_offset, prefix = stack.pop()
        if leaving:
            active_nodes.remove(node_offset)
            continue
        if node_offset in active_nodes:
            raise ValueError("Mach-O export trie contains a cycle")
        if node_offset in visited_nodes:
            raise ValueError(
                "Mach-O export trie references a node more than once"
            )
        if node_offset >= trie_size:
            raise ValueError(
                "Mach-O export trie child offset is out of bounds"
            )
        active_nodes.add(node_offset)
        visited_nodes.add(node_offset)
        stack.append((True, node_offset, b""))

        cursor = trie_offset + node_offset
        terminal_size, cursor = macho_uleb128(
            data,
            cursor,
            trie_end,
            f"Mach-O export trie node {node_offset} terminal size",
        )
        if terminal_size > trie_end - cursor:
            raise ValueError("Mach-O export trie terminal is out of bounds")
        terminal_end = cursor + terminal_size
        if terminal_size:
            flags, cursor = macho_uleb128(
                data,
                cursor,
                terminal_end,
                "Mach-O export trie terminal flags",
            )
            if flags & 0x03 == 0x03:
                raise ValueError(
                    "Mach-O export trie terminal has an invalid export kind"
                )
            if flags & 0x08 and flags & 0x10:
                raise ValueError(
                    "Mach-O export trie terminal combines re-export "
                    "and stub/resolver flags"
                )
            if flags & 0x08:
                _ordinal, cursor = macho_uleb128(
                    data,
                    cursor,
                    terminal_end,
                    "Mach-O export trie re-export ordinal",
                )
                import_name = c_string_bytes(
                    data,
                    cursor,
                    terminal_end,
                    "Mach-O export trie re-export name",
                )
                cursor += len(import_name) + 1
            else:
                _address, cursor = macho_uleb128(
                    data,
                    cursor,
                    terminal_end,
                    "Mach-O export trie terminal address",
                )
                if flags & 0x10:
                    _resolver, cursor = macho_uleb128(
                        data,
                        cursor,
                        terminal_end,
                        "Mach-O export trie resolver address",
                    )
            if cursor != terminal_end:
                raise ValueError(
                    "Mach-O export trie terminal has trailing payload data"
                )
            name = ascii_symbol(prefix)
            if name:
                exported_symbols.add(name)

        cursor = terminal_end
        if cursor >= trie_end:
            raise ValueError(
                "Mach-O export trie child count is out of bounds"
            )
        child_count = data[cursor]
        cursor += 1
        child_edges = set()
        children = []
        for child_index in range(child_count):
            edge = c_string_bytes(
                data,
                cursor,
                trie_end,
                f"Mach-O export trie child {child_index} edge",
            )
            cursor += len(edge) + 1
            if not edge:
                raise ValueError(
                    f"Mach-O export trie child {child_index} has an empty edge"
                )
            if edge in child_edges:
                raise ValueError(
                    "Mach-O export trie node has duplicate child edges"
                )
            child_edges.add(edge)
            child_offset, cursor = macho_uleb128(
                data,
                cursor,
                trie_end,
                f"Mach-O export trie child {child_index} offset",
            )
            if child_offset >= trie_size:
                raise ValueError(
                    "Mach-O export trie child offset is out of bounds"
                )
            child_prefix = prefix + edge
            if len(child_prefix) > trie_size:
                raise ValueError(
                    "Mach-O export trie symbol path is unreasonably long"
                )
            children.append((child_offset, child_prefix))

        for child_offset, child_prefix in reversed(children):
            stack.append((False, child_offset, child_prefix))

    return frozenset(exported_symbols)


def parse_macho_thin(data: bytes) -> NativeBinary | None:
    if len(data) < 4:
        return None
    magic = data[:4]
    if magic in (b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe"):
        raise ValueError("unsupported 32-bit Mach-O image")
    byte_order = {
        b"\xcf\xfa\xed\xfe": "<",
        b"\xfe\xed\xfa\xcf": ">",
    }.get(magic)
    if byte_order is None:
        return None
    if byte_order != "<":
        raise ValueError("unsupported big-endian Mach-O image")
    if len(data) < 32:
        raise ValueError("truncated Mach-O header")

    (
        _magic,
        cpu_type,
        _cpu_subtype,
        file_type,
        command_count,
        commands_size,
        _flags,
        _reserved,
    ) = struct.unpack_from("<IiiIIIII", data, 0)
    architecture = MACHO_CPU_ARCHITECTURE.get(cpu_type & 0xFFFFFFFF)
    if architecture is None:
        raise ValueError(
            f"unsupported Mach-O CPU type 0x{cpu_type & 0xFFFFFFFF:08x}"
        )
    if file_type != 6:
        raise ValueError(f"Mach-O file type {file_type} is not MH_DYLIB")
    if command_count == 0:
        raise ValueError("Mach-O dylib has no load commands")
    if commands_size < command_count * 8:
        raise ValueError("Mach-O load-command region is too small")
    require_range(data, 32, commands_size, "Mach-O load commands")

    command_offset = 32
    commands_end = 32 + commands_size
    has_file_backed_segment = False
    section_count = 0
    symbol_table = None
    id_dylib_count = 0
    dyld_info_export_trie = None
    dedicated_export_trie = None
    for index in range(command_count):
        require_range(
            data, command_offset, 8, f"Mach-O load command {index}"
        )
        command, command_size = struct.unpack_from("<II", data, command_offset)
        if command_size < 8 or command_size % 8:
            raise ValueError(
                f"Mach-O load command {index} has invalid size {command_size}"
            )
        if command_size > commands_end - command_offset:
            raise ValueError(f"Mach-O load command {index} is out of bounds")

        if command == 0x19:
            if command_size < 72:
                raise ValueError(f"truncated Mach-O LC_SEGMENT_64 command {index}")
            (
                _command,
                _command_size,
                _segment_name,
                _virtual_address,
                _virtual_size,
                file_offset,
                file_size,
                _maximum_protection,
                _initial_protection,
                segment_section_count,
                _segment_flags,
            ) = struct.unpack_from(
                "<II16sQQQQiiII", data, command_offset
            )
            expected_size = 72 + segment_section_count * 80
            if command_size != expected_size:
                raise ValueError(
                    f"Mach-O segment command {index} has invalid section data"
                )
            if file_size:
                require_range(
                    data,
                    file_offset,
                    file_size,
                    f"Mach-O segment {index} contents",
                )
                has_file_backed_segment = True

            for section_index in range(segment_section_count):
                section_offset = command_offset + 72 + section_index * 80
                (
                    _section_name,
                    _section_segment_name,
                    _address,
                    size,
                    file_data_offset,
                    _alignment,
                    relocations_offset,
                    relocation_count,
                    section_flags,
                    _reserved1,
                    _reserved2,
                    _reserved3,
                ) = struct.unpack_from(
                    "<16s16sQQIIIIIIII", data, section_offset
                )
                section_type = section_flags & 0xFF
                if size and section_type not in (1, 12, 18):
                    require_range(
                        data,
                        file_data_offset,
                        size,
                        f"Mach-O section {section_count} contents",
                    )
                if relocation_count:
                    require_range(
                        data,
                        relocations_offset,
                        relocation_count * 8,
                        f"Mach-O section {section_count} relocations",
                    )
                section_count += 1
        elif command == 0x02:
            if command_size != 24:
                raise ValueError(f"invalid Mach-O LC_SYMTAB command {index}")
            if symbol_table is not None:
                raise ValueError("Mach-O image contains multiple symbol tables")
            (
                _command,
                _command_size,
                symbols_offset,
                symbol_count,
                strings_offset,
                strings_size,
            ) = struct.unpack_from("<IIIIII", data, command_offset)
            require_range(
                data,
                symbols_offset,
                symbol_count * 16,
                "Mach-O symbol table",
            )
            require_range(
                data,
                strings_offset,
                strings_size,
                "Mach-O symbol string table",
            )
            symbol_table = (
                symbols_offset,
                symbol_count,
                strings_offset,
                strings_size,
            )
        elif command == 0x0D:
            if command_size < 24:
                raise ValueError(
                    f"Mach-O LC_ID_DYLIB command {index} has invalid size "
                    f"{command_size}"
                )
            id_dylib_count += 1
            if id_dylib_count > 1:
                raise ValueError(
                    "Mach-O image contains multiple LC_ID_DYLIB commands"
                )
            name_offset = struct.unpack_from(
                "<I", data, command_offset + 8
            )[0]
            if name_offset < 24 or name_offset >= command_size:
                raise ValueError(
                    f"Mach-O LC_ID_DYLIB command {index} has invalid name offset"
                )
            install_name = c_string_bytes(
                data,
                command_offset + name_offset,
                command_offset + command_size,
                f"Mach-O LC_ID_DYLIB command {index} name",
            )
            if not install_name:
                raise ValueError(
                    f"Mach-O LC_ID_DYLIB command {index} has an empty name"
                )
        elif command in (0x22, 0x80000022):
            if command_size != 48:
                raise ValueError(
                    f"invalid Mach-O LC_DYLD_INFO command {index}"
                )
            if dyld_info_export_trie is not None:
                raise ValueError(
                    "Mach-O image contains multiple LC_DYLD_INFO commands"
                )
            export_offset, export_size = struct.unpack_from(
                "<II", data, command_offset + 40
            )
            require_range(
                data,
                export_offset,
                export_size,
                "Mach-O export trie",
            )
            dyld_info_export_trie = (export_offset, export_size)
        elif command == 0x80000033:
            if command_size != 16:
                raise ValueError(
                    f"invalid Mach-O LC_DYLD_EXPORTS_TRIE command {index}"
                )
            if dedicated_export_trie is not None:
                raise ValueError(
                    "Mach-O image contains multiple "
                    "LC_DYLD_EXPORTS_TRIE commands"
                )
            export_offset, export_size = struct.unpack_from(
                "<II", data, command_offset + 8
            )
            require_range(
                data,
                export_offset,
                export_size,
                "Mach-O export trie",
            )
            dedicated_export_trie = (export_offset, export_size)

        command_offset += command_size

    if command_offset != commands_end:
        raise ValueError("Mach-O load-command sizes do not match sizeofcmds")
    if not has_file_backed_segment:
        raise ValueError("Mach-O dylib has no non-empty file-backed segment")
    if id_dylib_count == 0:
        raise ValueError("Mach-O dylib is missing LC_ID_DYLIB")

    exported_symbols = set()
    export_trie = (
        dedicated_export_trie
        if dedicated_export_trie is not None
        else dyld_info_export_trie
    )
    if export_trie is not None:
        exported_symbols.update(
            parse_macho_export_trie(data, export_trie[0], export_trie[1])
        )
    elif symbol_table is not None:
        (
            symbols_offset,
            symbol_count,
            strings_offset,
            strings_size,
        ) = symbol_table
        strings_end = strings_offset + strings_size
        for index in range(symbol_count):
            offset = symbols_offset + index * 16
            name_offset, symbol_type, symbol_section, _description, _value = (
                struct.unpack_from("<IBBHQ", data, offset)
            )
            if name_offset >= strings_size:
                raise ValueError(
                    f"Mach-O symbol {index} has an invalid name offset"
                )
            if symbol_type & 0xE0:
                continue
            basic_type = symbol_type & 0x0E
            if basic_type == 0x0E and not 1 <= symbol_section <= section_count:
                raise ValueError(
                    f"Mach-O symbol {index} has an invalid section index"
                )
            if (
                name_offset == 0
                or not symbol_type & 0x01
                or symbol_type & 0x10
                # N_INDR aliases another symbol and N_PBUD is a prebound
                # undefined reference, so neither one defines code here.
                or basic_type not in (0x02, 0x0E)
            ):
                continue
            name = ascii_symbol(
                c_string_bytes(
                    data,
                    strings_offset + name_offset,
                    strings_end,
                    f"Mach-O symbol {index} name",
                )
            )
            if name:
                exported_symbols.add(name)

    return NativeBinary(
        "Mach-O",
        frozenset({architecture}),
        frozenset(exported_symbols),
    )


def parse_macho(data: bytes) -> NativeBinary | None:
    if len(data) < 4:
        return None
    fat = {
        b"\xca\xfe\xba\xbe": (">", 20),
        b"\xbe\xba\xfe\xca": ("<", 20),
        b"\xca\xfe\xba\xbf": (">", 32),
        b"\xbf\xba\xfe\xca": ("<", 32),
    }.get(data[:4])
    if fat is None:
        return parse_macho_thin(data)
    if len(data) < 8:
        raise ValueError("truncated Mach-O fat header")

    byte_order, entry_size = fat
    architecture_count = struct.unpack_from(f"{byte_order}I", data, 4)[0]
    if architecture_count == 0 or architecture_count > 64:
        raise ValueError(
            f"invalid Mach-O fat architecture count {architecture_count}"
        )
    table_size = architecture_count * entry_size
    require_range(data, 8, table_size, "Mach-O fat architecture table")
    table_end = 8 + table_size

    slices = []
    for index in range(architecture_count):
        offset = 8 + index * entry_size
        cpu_type = struct.unpack_from(f"{byte_order}I", data, offset)[0]
        architecture = MACHO_CPU_ARCHITECTURE.get(cpu_type)
        if architecture is None:
            raise ValueError(
                f"unsupported Mach-O CPU type 0x{cpu_type:08x}"
            )
        if entry_size == 20:
            slice_offset, slice_size, alignment = struct.unpack_from(
                f"{byte_order}III", data, offset + 8
            )
        else:
            slice_offset, slice_size, alignment, _reserved = struct.unpack_from(
                f"{byte_order}QQII", data, offset + 8
            )
        if slice_size == 0:
            raise ValueError(f"Mach-O fat slice {index} is empty")
        if slice_offset < table_end:
            raise ValueError(
                f"Mach-O fat slice {index} overlaps the architecture table"
            )
        if alignment >= 63 or slice_offset % (1 << alignment):
            raise ValueError(f"Mach-O fat slice {index} is misaligned")
        require_range(
            data,
            slice_offset,
            slice_size,
            f"Mach-O fat slice {index}",
        )
        slices.append((slice_offset, slice_size, architecture, index))

    previous_end = table_end
    for slice_offset, slice_size, _architecture, index in sorted(slices):
        if slice_offset < previous_end:
            raise ValueError(f"Mach-O fat slice {index} overlaps another slice")
        previous_end = slice_offset + slice_size

    architectures = set()
    exported_symbols = set()
    for slice_offset, slice_size, architecture, index in slices:
        parsed = parse_macho_thin(
            data[slice_offset : slice_offset + slice_size]
        )
        if parsed is None:
            raise ValueError(f"Mach-O fat slice {index} is not a Mach-O image")
        if parsed.architectures != frozenset({architecture}):
            raise ValueError(
                f"Mach-O fat slice {index} CPU type does not match its image"
            )
        if architecture in architectures:
            raise ValueError(
                f"Mach-O fat image contains duplicate {architecture} slices"
            )
        architectures.add(architecture)
        exported_symbols.update(parsed.exported_symbols)

    return NativeBinary(
        "Mach-O", frozenset(architectures), frozenset(exported_symbols)
    )


def native_binary(data: bytes) -> NativeBinary:
    for parser in (parse_elf, parse_pe, parse_macho):
        parsed = parser(data)
        if parsed is not None:
            return parsed
    raise ValueError("unrecognized native binary format")


def verify_native_target(
    data: bytes,
    target: str,
    path: str,
    *,
    symbol_family: str,
) -> None:
    expected_format, expected_architecture = TARGET_ARCHITECTURE[target]
    parsed = native_binary(data)
    if parsed.binary_format != expected_format:
        raise ValueError(
            f"{path} is {parsed.binary_format}, expected {expected_format} "
            f"for {target}"
        )
    expected_architectures = {expected_architecture}
    if set(parsed.architectures) != expected_architectures:
        raise ValueError(
            f"{path} has architectures {sorted(parsed.architectures)}, "
            f"expected only {expected_architecture} for {target}"
        )

    normalized_symbols = {
        symbol[1:] if symbol.startswith("_") else symbol
        for symbol in parsed.exported_symbols
    }
    missing = sorted(
        MOSAIC_SYMBOL_FAMILIES[symbol_family] - normalized_symbols
    )
    if missing:
        raise ValueError(
            f"{path} is missing expected Mosaic {symbol_family} exports: {missing}"
        )
