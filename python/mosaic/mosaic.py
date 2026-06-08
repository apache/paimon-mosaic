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

import pyarrow as pa

from . import _ffi
from ._ffi import lib


class _ArrowSchema(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("metadata", ctypes.c_char_p),
        ("flags", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


class _ArrowArray(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_int64),
        ("null_count", ctypes.c_int64),
        ("offset", ctypes.c_int64),
        ("n_buffers", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("buffers", ctypes.c_void_p),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


def _check_error(msg="operation failed"):
    err = lib.mosaic_last_error()
    if err:
        raise RuntimeError(err.decode("utf-8", errors="replace"))
    raise RuntimeError(msg)


def _fetch_rg_stats(num_stats_fn, stats_fn, handle, rg_index):
    n_out = ctypes.c_uint32(0)
    rc = num_stats_fn(handle, rg_index, ctypes.byref(n_out))
    if rc != 0:
        _check_error("num_stats failed")
    n = n_out.value
    if n == 0:
        return {}
    names = (ctypes.c_char_p * n)()
    null_counts = (ctypes.c_uint64 * n)()
    min_ptrs = (ctypes.POINTER(ctypes.c_uint8) * n)()
    min_lens = (ctypes.c_size_t * n)()
    max_ptrs = (ctypes.POINTER(ctypes.c_uint8) * n)()
    max_lens = (ctypes.c_size_t * n)()
    rc = stats_fn(handle, rg_index, names, null_counts, min_ptrs, min_lens, max_ptrs, max_lens)
    if rc != 0:
        _check_error("row_group_stats failed")
    result = {}
    for i in range(n):
        col_name = names[i].decode("utf-8")
        min_val = ctypes.string_at(min_ptrs[i], min_lens[i]) if min_ptrs[i] else None
        max_val = ctypes.string_at(max_ptrs[i], max_lens[i]) if max_ptrs[i] else None
        result[col_name] = ColumnStatistics(null_counts[i], min_val, max_val)
    return result


class WriterOptions:
    COMPRESSION_NONE = 0
    COMPRESSION_ZSTD = 1

    def __init__(
        self,
        compression=1,
        zstd_level=1,
        num_buckets=0,
        row_group_max_size=256 * 1024 * 1024,
        max_dict_total_bytes=32 * 1024,
        max_dict_entries=255,
        stats_columns=None,
        page_size_threshold=32 * 1024,
    ):
        self.compression = compression
        self.zstd_level = zstd_level
        self.num_buckets = num_buckets
        self.row_group_max_size = row_group_max_size
        self.max_dict_total_bytes = max_dict_total_bytes
        self.max_dict_entries = max_dict_entries
        self.stats_columns = stats_columns or []
        self.page_size_threshold = page_size_threshold

    def _to_ffi(self):
        opts = _ffi.MosaicWriterOptions()
        opts.compression = self.compression
        opts.zstd_level = self.zstd_level
        opts.num_buckets = self.num_buckets
        opts.row_group_max_size = self.row_group_max_size
        opts.max_dict_total_bytes = self.max_dict_total_bytes
        opts.max_dict_entries = self.max_dict_entries
        refs = []
        if self.stats_columns:
            encoded = [s.encode("utf-8") for s in self.stats_columns]
            arr = (ctypes.c_char_p * len(encoded))(*encoded)
            refs.append(arr)
            refs.append(encoded)
            opts.stats_columns = arr
            opts.num_stats_columns = len(self.stats_columns)
        else:
            opts.stats_columns = None
            opts.num_stats_columns = 0
        opts.page_size_threshold = self.page_size_threshold
        return opts, refs


class MosaicWriter:

    def __init__(self, stream, schema, options=None):
        if not isinstance(schema, pa.Schema):
            raise TypeError(f"expected pyarrow.Schema, got {type(schema)}")

        self._stream = stream
        self._closed = False
        self._row_group_stats = None

        self._write_callback = _ffi.WRITE_FN(self._on_write)
        self._flush_callback = _ffi.FLUSH_FN(self._on_flush)
        self._get_pos_callback = _ffi.GET_POS_FN(self._on_get_pos)
        self._pos = 0

        c_stream = _ffi.MosaicOutputFile()
        c_stream.ctx = None
        c_stream.write_fn = self._write_callback
        c_stream.flush_fn = self._flush_callback
        c_stream.get_pos_fn = self._get_pos_callback

        c_opts, opts_refs = options._to_ffi() if options else WriterOptions()._to_ffi()

        c_schema = _ArrowSchema()
        schema_ptr = ctypes.addressof(c_schema)
        schema._export_to_c(schema_ptr)

        self._handle = lib.mosaic_writer_open(c_stream, ctypes.c_void_p(schema_ptr), c_opts)
        del opts_refs
        if not self._handle:
            _check_error("failed to open writer")

    def _on_write(self, ctx, data, length):
        try:
            buf = (ctypes.c_char * length).from_address(ctypes.cast(data, ctypes.c_void_p).value)
            self._stream.write(buf)
            self._pos += length
            return 0
        except Exception:
            return -1

    def _on_flush(self, ctx):
        try:
            self._stream.flush()
            return 0
        except Exception:
            return -1

    def _on_get_pos(self, ctx):
        return self._pos

    def write(self, data):
        is_table = isinstance(data, pa.Table)
        if not is_table and not isinstance(data, pa.RecordBatch):
            raise TypeError(f"expected pyarrow.RecordBatch or pyarrow.Table, got {type(data)}")
        if self._closed or not self._handle:
            raise RuntimeError("writer is closed")

        if is_table:
            for record_batch in data.to_batches():
                self._write_single_batch(record_batch)
        else:
            self._write_single_batch(data)

    def _write_single_batch(self, batch):
        c_schema = _ArrowSchema()
        c_array = _ArrowArray()
        schema_ptr = ctypes.addressof(c_schema)
        array_ptr = ctypes.addressof(c_array)
        batch._export_to_c(array_ptr, schema_ptr)
        rc = lib.mosaic_writer_write_batch(
            self._handle,
            ctypes.c_void_p(array_ptr),
            ctypes.c_void_p(schema_ptr),
        )
        if rc != 0:
            _check_error("write_batch failed")

    def estimated_file_size(self):
        out = ctypes.c_int64(0)
        rc = lib.mosaic_writer_estimated_file_size(self._handle, ctypes.byref(out))
        if rc != 0:
            _check_error("estimated_file_size failed")
        return out.value

    @property
    def num_row_groups(self):
        if self._row_group_stats is None:
            raise RuntimeError("writer is not closed yet")
        return len(self._row_group_stats)

    def get_row_group_statistics(self, rg_index):
        """Returns column statistics for the given row group, keyed by column name."""
        if self._row_group_stats is None:
            raise RuntimeError("writer is not closed yet")
        return self._row_group_stats[rg_index]

    def close(self):
        if not self._closed and self._handle:
            self._closed = True
            rc = lib.mosaic_writer_close(self._handle)
            if rc != 0:
                lib.mosaic_writer_free(self._handle)
                self._handle = None
                _check_error("close failed")
            self._collect_statistics()
            lib.mosaic_writer_free(self._handle)
            self._handle = None

    def _collect_statistics(self):
        n_rg = ctypes.c_uint32(0)
        lib.mosaic_writer_num_row_groups(self._handle, ctypes.byref(n_rg))
        all_stats = []
        for rg in range(n_rg.value):
            all_stats.append(_fetch_rg_stats(
                lib.mosaic_writer_row_group_num_stats,
                lib.mosaic_writer_row_group_stats,
                self._handle, rg))
        self._row_group_stats = all_stats

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class ColumnStatistics:
    def __init__(self, null_count, min, max):
        self.null_count = null_count
        self.min = min
        self.max = max

    @property
    def has_min_max(self):
        return self.min is not None


def _build_projected_schema(original_schema, column_names):
    """Build a projected PyArrow schema from dotted column paths.

    Uses exact-match-first strategy (like ORC/Parquet): if a column name
    exactly matches a top-level field (even if it contains '.'), it is
    returned directly. Otherwise, the name is split on '.' and resolved
    as a STRUCT field path.

    Merges multiple leaves under the same nested STRUCT parent recursively.
    For example, ["info.addr.city", "info.addr.zip"] produces
    info: struct<addr: struct<city, zip>>.
    """
    all_top_names = {original_schema.field(i).name for i in range(len(original_schema))}
    seen = set()
    output_fields = []

    for col in column_names:
        # Exact match first: handles top-level column names containing '.'
        if col in all_top_names:
            if col not in seen:
                seen.add(col)
                output_fields.append(original_schema.field(col))
            continue

        parts = col.split(".")
        root = parts[0]
        if root in seen:
            continue

        try:
            top_field = original_schema.field(root)
        except KeyError:
            continue

        if len(parts) == 1 or not isinstance(top_field.type, pa.StructType):
            seen.add(root)
            output_fields.append(top_field)
            continue

        seen.add(root)
        same_root = [c.split(".")[1:] for c in column_names
                     if c.startswith(root + ".") and c not in all_top_names]
        projected = _project_struct_type(top_field.type, same_root)
        output_fields.append(pa.field(root, projected, nullable=top_field.nullable))

    return pa.schema(output_fields, metadata=original_schema.metadata)


def _project_struct_type(struct_type, sub_paths):
    """Build a partial struct type from a list of relative sub-paths.

    Each sub_path is a list of path segments, e.g. ["addr", "city"].
    Paths referring to the same nested struct are merged recursively.
    """
    children = {}
    child_order = []

    for path in sub_paths:
        name = path[0]
        rest = path[1:]

        if name not in children:
            idx = struct_type.get_field_index(name)
            if idx < 0:
                continue
            field = struct_type.field(idx)
            children[name] = {"field": field, "sub_paths": []}
            child_order.append(name)

        if rest:
            children[name]["sub_paths"].append(rest)

    result_fields = []
    for name in child_order:
        entry = children[name]
        field = entry["field"]
        subs = entry["sub_paths"]

        if not subs or not isinstance(field.type, pa.StructType):
            result_fields.append(field)
        else:
            projected = _project_struct_type(field.type, subs)
            result_fields.append(pa.field(name, projected, nullable=field.nullable))

    return pa.struct(result_fields)


class MosaicReader:

    def __init__(self, handle, refs=None):
        self._handle = handle
        self._refs = refs
        c_schema = _ArrowSchema()
        schema_ptr = ctypes.addressof(c_schema)
        rc = lib.mosaic_reader_export_schema(handle, ctypes.c_void_p(schema_ptr))
        if rc != 0:
            _check_error("export_schema failed")
        self._schema = pa.Schema._import_from_c(schema_ptr)
        self._projected_schema = None

    @staticmethod
    def from_input_file(read_at_fn, file_length):
        """Create a MosaicReader from a callable and file length.

        ``read_at_fn(offset, length) -> bytes`` must be thread-safe: the
        reader may call it concurrently from multiple threads to perform
        parallel IO.
        """
        @_ffi.READ_AT_FN
        def c_read_at(ctx, offset, buf, length):
            try:
                data = read_at_fn(offset, length)
                if len(data) != length:
                    return -1
                ctypes.memmove(buf, data, length)
                return 0
            except Exception:
                return -1

        @_ffi.LENGTH_FN
        def c_length(ctx):
            return file_length

        input_file = _ffi.MosaicInputFile()
        input_file.ctx = None
        input_file.read_at_fn = c_read_at
        input_file.length_fn = c_length

        handle = lib.mosaic_reader_open(input_file)
        if not handle:
            _check_error("failed to open reader")
        return MosaicReader(handle, refs=(c_read_at, c_length, input_file))

    @property
    def schema(self):
        return self._schema

    @property
    def num_row_groups(self):
        out = ctypes.c_uint32(0)
        rc = lib.mosaic_reader_num_row_groups(self._handle, ctypes.byref(out))
        if rc != 0:
            _check_error("num_row_groups failed")
        return out.value

    def project(self, columns):
        """Set projection on the reader. Subsequent reads only return the named columns.

        Supports top-level column names, original STRUCT names (e.g. ``"info"``),
        STRUCT leaf paths (e.g. ``"info.name"``), and nested STRUCT paths
        (e.g. ``"info.addr"``).
        """
        column_names = list(columns)
        c_strs = [c.encode("utf-8") for c in column_names]
        arr = (ctypes.c_char_p * len(column_names))(*c_strs)
        rc = lib.mosaic_reader_set_projection(self._handle, arr, len(column_names))
        if rc != 0:
            _check_error("set_projection failed")
        self._projected_schema = _build_projected_schema(self._schema, column_names)

    def read_row_group(self, rg_index):
        rg_handle = lib.mosaic_reader_open_row_group(self._handle, rg_index)
        if not rg_handle:
            _check_error(f"failed to open row group {rg_index}")
        rb_handle = lib.mosaic_row_group_reader_read_columns(rg_handle)
        lib.mosaic_row_group_reader_free(rg_handle)
        if not rb_handle:
            _check_error("read_columns failed")
        try:
            c_schema = _ArrowSchema()
            c_array = _ArrowArray()
            schema_ptr = ctypes.addressof(c_schema)
            array_ptr = ctypes.addressof(c_array)
            rc = lib.mosaic_record_batch_export(
                rb_handle,
                ctypes.c_void_p(array_ptr),
                ctypes.c_void_p(schema_ptr),
            )
            if rc != 0:
                _check_error("record_batch_export failed")
            return pa.RecordBatch._import_from_c(array_ptr, schema_ptr)
        finally:
            lib.mosaic_record_batch_free(rb_handle)

    def read_all(self):
        batches = []
        for rg in range(self.num_row_groups):
            batches.append(self.read_row_group(rg))
        if batches:
            return pa.Table.from_batches(batches, schema=batches[0].schema)
        schema = (
            self._projected_schema
            if self._projected_schema is not None
            else self._schema
        )
        return pa.Table.from_batches([], schema=schema)

    def row_group_num_rows(self, rg_index):
        out = ctypes.c_uint32(0)
        rc = lib.mosaic_reader_row_group_num_rows(self._handle, rg_index, ctypes.byref(out))
        if rc != 0:
            _check_error("row_group_num_rows failed")
        return out.value

    def get_row_group_statistics(self, rg_index):
        """Returns column statistics for the given row group, keyed by column name."""
        return _fetch_rg_stats(
            lib.mosaic_reader_row_group_num_stats,
            lib.mosaic_reader_row_group_stats,
            self._handle, rg_index)

    def close(self):
        if self._handle:
            lib.mosaic_reader_free(self._handle)
            self._handle = None
        if self._refs and isinstance(self._refs, tuple) and hasattr(self._refs[0], "close"):
            self._refs[0].close()
            self._refs = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


def write_table(table, stream, options=None):
    if not isinstance(table, pa.Table):
        raise TypeError(f"expected pyarrow.Table, got {type(table)}")
    with MosaicWriter(stream, table.schema, options) as writer:
        writer.write(table)


def read_table(read_at_fn, file_length, columns=None):
    with MosaicReader.from_input_file(read_at_fn, file_length) as reader:
        if columns is not None:
            reader.project(columns)
        return reader.read_all()
