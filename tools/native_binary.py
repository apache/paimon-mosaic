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

MOSAIC_SYMBOL_FAMILIES = {
    "JNI": {
        "Java_org_apache_paimon_mosaic_NativeLib_nativeReaderOpen",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeWriterOpen",
        "Java_org_apache_paimon_mosaic_NativeLib_nativeWriterWriteBatch",
    },
    "FFI": {
        "mosaic_last_error",
        "mosaic_reader_open",
        "mosaic_writer_open",
        "mosaic_writer_write_batch",
    },
}


@dataclass(frozen=True)
class NativeBinary:
    binary_format: str
    architectures: frozenset[str]
    exported_symbols: frozenset[str]


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
    has_dynamic_segment = False
    for index in range(program_count):
        offset = program_offset + index * program_entry_size
        (
            segment_type,
            _segment_flags,
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
        elif segment_type == 2:
            if file_size == 0 or file_size % 16:
                raise ValueError(f"ELF dynamic segment {index} has invalid size")
            has_dynamic_segment = True
        elif segment_type == 3:
            raise ValueError("ELF ET_DYN file contains PT_INTERP and is executable")

    if not has_load_segment:
        raise ValueError("ELF shared object has no non-empty PT_LOAD segment")
    if not has_dynamic_segment:
        raise ValueError("ELF shared object has no PT_DYNAMIC segment")

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
            _section_flags,
            _address,
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
        sections.append((section_type, file_offset, size, link, entry_size))

    exported_symbols = set()
    for index, (section_type, file_offset, size, link, entry_size) in enumerate(
        sections
    ):
        if section_type != 11:
            continue
        if entry_size != 24 or size % entry_size:
            raise ValueError(f"ELF dynamic symbol section {index} is malformed")
        if link >= section_count:
            raise ValueError(
                f"ELF dynamic symbol section {index} has invalid string-table link"
            )
        string_type, string_offset, string_size, _link, _entry_size = sections[
            link
        ]
        if string_type != 3:
            raise ValueError(
                f"ELF dynamic symbol section {index} does not link to a string table"
            )
        string_limit = string_offset + string_size

        for symbol_index in range(size // entry_size):
            symbol_offset = file_offset + symbol_index * entry_size
            (
                name_offset,
                info,
                other,
                symbol_section,
                _value,
                _symbol_size,
            ) = struct.unpack_from("<IBBHQQ", data, symbol_offset)
            if name_offset >= string_size:
                raise ValueError(
                    f"ELF dynamic symbol {symbol_index} has an invalid name offset"
                )
            if name_offset == 0:
                continue
            binding = info >> 4
            visibility = other & 0x03
            if (
                binding not in (1, 2)
                or symbol_section == 0
                or visibility not in (0, 3)
            ):
                continue
            name = ascii_symbol(
                c_string_bytes(
                    data,
                    string_offset + name_offset,
                    string_limit,
                    f"ELF dynamic symbol {symbol_index} name",
                )
            )
            if name:
                exported_symbols.add(name)

    return NativeBinary(
        "ELF", frozenset({architecture}), frozenset(exported_symbols)
    )


def pe_rva_span(
    rva: int,
    sections: list[tuple[int, int, int, int]],
    headers_size: int,
    data_size: int,
    description: str,
) -> tuple[int, int]:
    if rva < headers_size:
        if rva >= data_size:
            raise ValueError(f"{description} RVA is out of bounds")
        return rva, min(headers_size, data_size) - rva

    for virtual_address, virtual_size, file_offset, file_size in sections:
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
    sections: list[tuple[int, int, int, int]],
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
            _section_characteristics,
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
            (virtual_address, virtual_size, file_offset, file_size)
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
                name = ascii_symbol(
                    c_string_bytes(
                        data,
                        name_offset,
                        name_offset + name_available,
                        f"PE export name {index}",
                    )
                )
                if name is None:
                    raise ValueError(f"PE export name {index} is not ASCII")
                exported_symbols.add(name)

    return NativeBinary(
        "PE", frozenset({architecture}), frozenset(exported_symbols)
    )


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

        command_offset += command_size

    if command_offset != commands_end:
        raise ValueError("Mach-O load-command sizes do not match sizeofcmds")
    if not has_file_backed_segment:
        raise ValueError("Mach-O dylib has no non-empty file-backed segment")

    exported_symbols = set()
    if symbol_table is not None:
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
                or basic_type == 0
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


def elf_architectures(data: bytes) -> set[str] | None:
    parsed = parse_elf(data)
    return None if parsed is None else set(parsed.architectures)


def pe_architectures(data: bytes) -> set[str] | None:
    parsed = parse_pe(data)
    return None if parsed is None else set(parsed.architectures)


def macho_architectures(data: bytes) -> set[str] | None:
    parsed = parse_macho(data)
    return None if parsed is None else set(parsed.architectures)


def native_format_and_architectures(data: bytes) -> tuple[str, set[str]]:
    parsed = native_binary(data)
    return parsed.binary_format, set(parsed.architectures)


def expected_symbol_family(path: str) -> str | None:
    filename = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if "paimon_mosaic_jni" in filename:
        return "JNI"
    if "paimon_mosaic_ffi" in filename:
        return "FFI"
    return None


def verify_native_target(data: bytes, target: str, path: str) -> None:
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

    family = expected_symbol_family(path)
    if family is None:
        return
    normalized_symbols = {
        symbol[1:] if symbol.startswith("_") else symbol
        for symbol in parsed.exported_symbols
    }
    missing = sorted(MOSAIC_SYMBOL_FAMILIES[family] - normalized_symbols)
    if missing:
        raise ValueError(
            f"{path} is missing expected Mosaic {family} exports: {missing}"
        )
