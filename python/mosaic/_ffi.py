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

import ctypes
import ctypes.util
import os
import platform
import sys
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_int8,
    c_int16,
    c_int32,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)


def _load_library():
    system = platform.system()
    if system == "Darwin":
        lib_name = "libpaimon_mosaic_ffi.dylib"
    elif system == "Windows":
        lib_name = "paimon_mosaic_ffi.dll"
    else:
        lib_name = "libpaimon_mosaic_ffi.so"

    env_path = os.environ.get("MOSAIC_LIB_PATH")
    search_paths = []
    if env_path:
        search_paths.append(env_path)

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths.append(pkg_dir)
    search_paths.append(os.path.join(pkg_dir, "..", ".."))

    for rel in [
        os.path.join("..", "target", "release"),
        os.path.join("..", "target", "debug"),
        os.path.join("..", "..", "target", "release"),
        os.path.join("..", "..", "target", "debug"),
    ]:
        search_paths.append(os.path.join(pkg_dir, rel))

    for d in search_paths:
        candidate = os.path.join(d, lib_name)
        if os.path.isfile(candidate):
            return ctypes.CDLL(candidate)

    try:
        return ctypes.CDLL(lib_name)
    except OSError:
        raise OSError(
            f"Cannot find {lib_name}. Build the native library first with "
            f"'cargo build --release -p paimon-mosaic-ffi', or set MOSAIC_LIB_PATH "
            f"to the directory containing {lib_name}."
        )


lib = _load_library()

# ======================== Callback types ========================

WRITE_FN = CFUNCTYPE(c_int32, c_void_p, POINTER(c_uint8), c_size_t)
FLUSH_FN = CFUNCTYPE(c_int32, c_void_p)
GET_POS_FN = CFUNCTYPE(c_int64, c_void_p)

READ_AT_FN = CFUNCTYPE(c_int32, c_void_p, c_uint64, POINTER(c_uint8), c_size_t)
LENGTH_FN = CFUNCTYPE(c_uint64, c_void_p)


class MosaicOutputFile(Structure):
    _fields_ = [
        ("ctx", c_void_p),
        ("write_fn", WRITE_FN),
        ("flush_fn", FLUSH_FN),
        ("get_pos_fn", GET_POS_FN),
    ]


class MosaicInputFile(Structure):
    _fields_ = [
        ("ctx", c_void_p),
        ("read_at_fn", READ_AT_FN),
        ("length_fn", LENGTH_FN),
    ]


class MosaicWriterOptions(Structure):
    _fields_ = [
        ("compression", c_uint8),
        ("zstd_level", c_int),
        ("num_buckets", c_uint32),
        ("row_group_max_size", c_uint64),
        ("max_dict_total_bytes", c_uint32),
        ("max_dict_entries", c_uint32),
        ("stats_columns", POINTER(c_char_p)),
        ("num_stats_columns", c_uint32),
        ("page_size_threshold", c_uint32),
    ]


# ======================== Writer Options ========================

lib.mosaic_writer_options_default.argtypes = []
lib.mosaic_writer_options_default.restype = MosaicWriterOptions

# ======================== Writer ========================

lib.mosaic_writer_open.argtypes = [MosaicOutputFile, c_void_p, MosaicWriterOptions]
lib.mosaic_writer_open.restype = c_void_p

lib.mosaic_writer_close.argtypes = [c_void_p]
lib.mosaic_writer_close.restype = c_int

lib.mosaic_writer_free.argtypes = [c_void_p]
lib.mosaic_writer_free.restype = None

lib.mosaic_writer_estimated_file_size.argtypes = [c_void_p, POINTER(c_int64)]
lib.mosaic_writer_estimated_file_size.restype = c_int

lib.mosaic_writer_write_batch.argtypes = [c_void_p, c_void_p, c_void_p]
lib.mosaic_writer_write_batch.restype = c_int

# ======================== Writer Stats ========================

lib.mosaic_writer_num_row_groups.argtypes = [c_void_p, POINTER(c_uint32)]
lib.mosaic_writer_num_row_groups.restype = c_int

lib.mosaic_writer_row_group_num_stats.argtypes = [c_void_p, c_uint32, POINTER(c_uint32)]
lib.mosaic_writer_row_group_num_stats.restype = c_int

lib.mosaic_writer_row_group_stats.argtypes = [
    c_void_p, c_uint32,
    POINTER(c_char_p), POINTER(c_uint64),
    POINTER(POINTER(c_uint8)), POINTER(c_size_t),
    POINTER(POINTER(c_uint8)), POINTER(c_size_t),
]
lib.mosaic_writer_row_group_stats.restype = c_int

# ======================== Reader ========================

lib.mosaic_reader_open.argtypes = [MosaicInputFile]
lib.mosaic_reader_open.restype = c_void_p

lib.mosaic_reader_free.argtypes = [c_void_p]
lib.mosaic_reader_free.restype = None

lib.mosaic_reader_export_schema.argtypes = [c_void_p, c_void_p]
lib.mosaic_reader_export_schema.restype = c_int

lib.mosaic_reader_num_row_groups.argtypes = [c_void_p, POINTER(c_uint32)]
lib.mosaic_reader_num_row_groups.restype = c_int

# ======================== Row Group Reader ========================

lib.mosaic_reader_open_row_group.argtypes = [c_void_p, c_uint32]
lib.mosaic_reader_open_row_group.restype = c_void_p

lib.mosaic_reader_set_projection.argtypes = [
    c_void_p, POINTER(c_char_p), c_uint32,
]
lib.mosaic_reader_set_projection.restype = c_int32

lib.mosaic_row_group_reader_free.argtypes = [c_void_p]
lib.mosaic_row_group_reader_free.restype = None

lib.mosaic_row_group_reader_num_rows.argtypes = [c_void_p, POINTER(c_uint32)]
lib.mosaic_row_group_reader_num_rows.restype = c_int

# ======================== Record Batch (Arrow C Data Interface) ========================

lib.mosaic_row_group_reader_read_columns.argtypes = [c_void_p]
lib.mosaic_row_group_reader_read_columns.restype = c_void_p

lib.mosaic_record_batch_num_rows.argtypes = [c_void_p, POINTER(c_uint32)]
lib.mosaic_record_batch_num_rows.restype = c_int

lib.mosaic_record_batch_num_columns.argtypes = [c_void_p, POINTER(c_uint32)]
lib.mosaic_record_batch_num_columns.restype = c_int

lib.mosaic_record_batch_export.argtypes = [c_void_p, c_void_p, c_void_p]
lib.mosaic_record_batch_export.restype = c_int

lib.mosaic_record_batch_free.argtypes = [c_void_p]
lib.mosaic_record_batch_free.restype = None

# ======================== Row Group Num Rows ========================

lib.mosaic_reader_row_group_num_rows.argtypes = [c_void_p, c_uint32, POINTER(c_uint32)]
lib.mosaic_reader_row_group_num_rows.restype = c_int

# ======================== Row Group Stats ========================

lib.mosaic_reader_row_group_num_stats.argtypes = [c_void_p, c_uint32, POINTER(c_uint32)]
lib.mosaic_reader_row_group_num_stats.restype = c_int

lib.mosaic_reader_row_group_stats.argtypes = [
    c_void_p, c_uint32,
    POINTER(c_char_p), POINTER(c_uint64),
    POINTER(POINTER(c_uint8)), POINTER(c_size_t),
    POINTER(POINTER(c_uint8)), POINTER(c_size_t),
]
lib.mosaic_reader_row_group_stats.restype = c_int

# ======================== Error ========================

lib.mosaic_last_error.argtypes = []
lib.mosaic_last_error.restype = c_char_p
