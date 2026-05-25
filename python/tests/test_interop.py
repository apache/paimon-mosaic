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

"""
Cross-language interoperability tests for Mosaic.

These tests read .mosaic files written by the Rust interop_write_test,
verifying that the Python binding can correctly read files produced by Rust.
One test also writes a file from Python for Rust to read back.
"""

import io
import os
from decimal import Decimal

import pyarrow as pa
import pytest

from mosaic import MosaicReader, MosaicWriter, WriterOptions

INTEROP_DIR = "/tmp/mosaic_interop"


def _open_file(filename):
    """Open a mosaic file from disk and return a MosaicReader."""
    path = os.path.join(INTEROP_DIR, filename)
    f = open(path, "rb")
    data = f.read()
    f.close()
    file_length = len(data)

    def read_at(offset, length):
        return data[offset : offset + length]

    return MosaicReader.from_input_file(read_at, file_length)


def _interop_file_exists(filename):
    return os.path.exists(os.path.join(INTEROP_DIR, filename))


class TestInteropRead:
    """Tests that read .mosaic files written by the Rust test."""

    # ======================== 1. Read int_data.mosaic ========================

    @pytest.mark.skipif(
        not _interop_file_exists("int_data.mosaic"),
        reason="Interop file not found; run Rust interop_write_test first",
    )
    def test_read_rust_int_data(self):
        with _open_file("int_data.mosaic") as reader:
            assert len(reader.schema) == 2

            total_rows = 0
            for rg in range(reader.num_row_groups):
                rb = reader.read_row_group(rg)
                ids = rb.column("id").to_pylist()
                values = rb.column("value").to_pylist()

                for i in range(rb.num_rows):
                    row_idx = total_rows + i
                    assert ids[i] == row_idx, f"id mismatch at row {row_idx}"
                    assert values[i] == row_idx * 10, f"value mismatch at row {row_idx}"

                total_rows += rb.num_rows

            assert total_rows == 10000
        print("test_read_rust_int_data: PASSED (10000 rows)")

    # ======================== 2. Read string_data.mosaic ========================

    @pytest.mark.skipif(
        not _interop_file_exists("string_data.mosaic"),
        reason="Interop file not found; run Rust interop_write_test first",
    )
    def test_read_rust_string_data(self):
        with _open_file("string_data.mosaic") as reader:
            assert len(reader.schema) == 3

            total_rows = 0
            name_nulls = 0
            data_nulls = 0
            for rg in range(reader.num_row_groups):
                rb = reader.read_row_group(rg)
                ids = rb.column("id").to_pylist()
                names = rb.column("name").to_pylist()
                datas = rb.column("data").to_pylist()

                for i in range(rb.num_rows):
                    row_idx = total_rows + i
                    assert ids[i] == row_idx

                    if row_idx % 7 == 0:
                        assert names[i] is None, f"name should be null at row {row_idx}"
                        name_nulls += 1
                    else:
                        assert names[i] == f"name_{row_idx}", f"name mismatch at row {row_idx}"

                    if row_idx % 5 == 0:
                        assert datas[i] is None, f"data should be null at row {row_idx}"
                        data_nulls += 1
                    else:
                        assert datas[i] == f"bin_{row_idx}".encode(), f"data mismatch at row {row_idx}"

                total_rows += rb.num_rows

            assert total_rows == 10000
            assert name_nulls > 0
            assert data_nulls > 0
        print("test_read_rust_string_data: PASSED")

    # ======================== 3. Read all_types.mosaic ========================

    @pytest.mark.skipif(
        not _interop_file_exists("all_types.mosaic"),
        reason="Interop file not found; run Rust interop_write_test first",
    )
    def test_read_rust_all_types(self):
        with _open_file("all_types.mosaic") as reader:
            assert len(reader.schema) == 11

            total_rows = 0
            for rg in range(reader.num_row_groups):
                rb = reader.read_row_group(rg)

                bools = rb.column("f_bool").to_pylist()
                i8s = rb.column("f_int8").to_pylist()
                i16s = rb.column("f_int16").to_pylist()
                i32s = rb.column("f_int32").to_pylist()
                i64s = rb.column("f_int64").to_pylist()
                f32s = rb.column("f_float32").to_pylist()
                f64s = rb.column("f_float64").to_pylist()
                dates = rb.column("f_date32").to_pylist()
                strs = rb.column("f_utf8").to_pylist()
                bins = rb.column("f_binary").to_pylist()
                decs = rb.column("f_decimal").to_pylist()

                for i in range(rb.num_rows):
                    row_idx = total_rows + i

                    # Boolean: null every 13th
                    if row_idx % 13 == 0:
                        assert bools[i] is None
                    else:
                        assert bools[i] == (row_idx % 2 == 0)

                    # Int8: null every 11th
                    if row_idx % 11 == 0:
                        assert i8s[i] is None
                    else:
                        # i8 wraps: (row_idx % 256) as i8
                        expected = row_idx % 256
                        if expected >= 128:
                            expected -= 256
                        assert i8s[i] == expected, f"i8 mismatch at row {row_idx}: {i8s[i]} != {expected}"

                    # Int16: null every 17th
                    if row_idx % 17 == 0:
                        assert i16s[i] is None
                    else:
                        assert i16s[i] == row_idx % 30000

                    # Int32: null every 19th
                    if row_idx % 19 == 0:
                        assert i32s[i] is None
                    else:
                        assert i32s[i] == row_idx * 100

                    # Int64: null every 23rd
                    if row_idx % 23 == 0:
                        assert i64s[i] is None
                    else:
                        assert i64s[i] == row_idx * 1000

                    # Float32: null every 29th
                    if row_idx % 29 == 0:
                        assert f32s[i] is None
                    else:
                        assert abs(f32s[i] - row_idx * 0.1) < 1e-4, f"f32 mismatch at row {row_idx}"

                    # Float64: null every 31st
                    if row_idx % 31 == 0:
                        assert f64s[i] is None
                    else:
                        assert abs(f64s[i] - row_idx * 0.001) < 1e-9

                    # Date32: null every 37th (date32 comes as datetime.date)
                    if row_idx % 37 == 0:
                        assert dates[i] is None
                    else:
                        import datetime

                        expected_date = datetime.date(1970, 1, 1) + datetime.timedelta(
                            days=18000 + (row_idx % 3650)
                        )
                        assert dates[i] == expected_date, f"date mismatch at row {row_idx}: {dates[i]} != {expected_date}"

                    # Utf8: null every 41st
                    if row_idx % 41 == 0:
                        assert strs[i] is None
                    else:
                        assert strs[i] == f"str_{row_idx}"

                    # Binary: null every 43rd
                    if row_idx % 43 == 0:
                        assert bins[i] is None
                    else:
                        expected_bin = bytes([row_idx % 256]) * 4
                        assert bins[i] == expected_bin

                    # Decimal128(10,2): null every 47th
                    if row_idx % 47 == 0:
                        assert decs[i] is None
                    else:
                        expected_dec = Decimal(str(row_idx * 100)) / Decimal("100")
                        assert decs[i] == expected_dec, f"decimal mismatch at row {row_idx}: {decs[i]} != {expected_dec}"

                total_rows += rb.num_rows

            assert total_rows == 5000
        print("test_read_rust_all_types: PASSED (5000 rows, 11 columns)")

    # ======================== 4. Read constant_data.mosaic ========================

    @pytest.mark.skipif(
        not _interop_file_exists("constant_data.mosaic"),
        reason="Interop file not found; run Rust interop_write_test first",
    )
    def test_read_rust_constant_data(self):
        with _open_file("constant_data.mosaic") as reader:
            total_rows = 0
            for rg in range(reader.num_row_groups):
                rb = reader.read_row_group(rg)
                ints = rb.column("c_int").to_pylist()
                strs = rb.column("c_str").to_pylist()
                floats = rb.column("c_float").to_pylist()

                for i in range(rb.num_rows):
                    assert ints[i] == 42, f"c_int mismatch at row {total_rows + i}"
                    assert strs[i] == "constant_value", f"c_str mismatch at row {total_rows + i}"
                    assert abs(floats[i] - 3.14) < 1e-9, f"c_float mismatch at row {total_rows + i}"

                total_rows += rb.num_rows

            assert total_rows == 10000
        print("test_read_rust_constant_data: PASSED (10000 rows, all constant)")

    # ======================== 5. Read null_heavy.mosaic ========================

    @pytest.mark.skipif(
        not _interop_file_exists("null_heavy.mosaic"),
        reason="Interop file not found; run Rust interop_write_test first",
    )
    def test_read_rust_null_heavy(self):
        with _open_file("null_heavy.mosaic") as reader:
            total_rows = 0
            int_nulls = 0
            str_nulls = 0
            float_nulls = 0

            for rg in range(reader.num_row_groups):
                rb = reader.read_row_group(rg)
                ints = rb.column("n_int64").to_pylist()
                strs = rb.column("n_utf8").to_pylist()
                floats = rb.column("n_float64").to_pylist()

                for i in range(rb.num_rows):
                    row_idx = total_rows + i

                    if row_idx % 5 != 0:
                        assert ints[i] is None
                        int_nulls += 1
                    else:
                        assert ints[i] == row_idx

                    if row_idx % 5 != 1:
                        assert strs[i] is None
                        str_nulls += 1
                    else:
                        assert strs[i] == f"val_{row_idx}"

                    if row_idx % 5 != 2:
                        assert floats[i] is None
                        float_nulls += 1
                    else:
                        assert abs(floats[i] - row_idx * 0.5) < 1e-9

                total_rows += rb.num_rows

            assert total_rows == 10000
            assert int_nulls == 8000
            assert str_nulls == 8000
            assert float_nulls == 8000
        print("test_read_rust_null_heavy: PASSED (10000 rows, 80% nulls)")

    # ======================== 6. Read compressed_none.mosaic ========================

    @pytest.mark.skipif(
        not _interop_file_exists("compressed_none.mosaic"),
        reason="Interop file not found; run Rust interop_write_test first",
    )
    def test_read_rust_no_compression(self):
        with _open_file("compressed_none.mosaic") as reader:
            total_rows = 0
            for rg in range(reader.num_row_groups):
                rb = reader.read_row_group(rg)
                ids = rb.column("id").to_pylist()
                values = rb.column("value").to_pylist()

                for i in range(rb.num_rows):
                    row_idx = total_rows + i
                    assert ids[i] == row_idx
                    assert values[i] == row_idx * 10

                total_rows += rb.num_rows

            assert total_rows == 10000
        print("test_read_rust_no_compression: PASSED")

    # ======================== 7. Read multi_rg.mosaic ========================

    @pytest.mark.skipif(
        not _interop_file_exists("multi_rg.mosaic"),
        reason="Interop file not found; run Rust interop_write_test first",
    )
    def test_read_rust_multi_row_group(self):
        with _open_file("multi_rg.mosaic") as reader:
            assert reader.num_row_groups > 1, (
                f"Expected multiple row groups, got {reader.num_row_groups}"
            )

            total_rows = 0
            for rg in range(reader.num_row_groups):
                rb = reader.read_row_group(rg)
                ids = rb.column("id").to_pylist()
                values = rb.column("value").to_pylist()

                for i in range(rb.num_rows):
                    row_idx = total_rows + i
                    assert ids[i] == row_idx
                    assert values[i] == row_idx * 10

                total_rows += rb.num_rows

            assert total_rows == 10000
            num_rgs = reader.num_row_groups
        print(
            f"test_read_rust_multi_row_group: PASSED ({num_rgs} row groups)"
        )


class TestInteropWrite:
    """Test writing a file from Python for Rust to read back."""

    # ======================== 8. Write from Python ========================

    def test_write_file_read_from_rust(self):
        num_rows = 5000
        pa_schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("name", pa.utf8()),
                pa.field("score", pa.float64(), nullable=False),
            ]
        )

        ids = list(range(num_rows))
        names = [None if i % 3 == 0 else f"py_name_{i}" for i in range(num_rows)]
        scores = [i * 2.5 for i in range(num_rows)]

        batch = pa.record_batch(
            [
                pa.array(ids, type=pa.int64()),
                pa.array(names, type=pa.utf8()),
                pa.array(scores, type=pa.float64()),
            ],
            names=["id", "name", "score"],
        )

        opts = WriterOptions(num_buckets=1)
        buf = io.BytesIO()
        with MosaicWriter(buf, pa_schema, opts) as writer:
            writer.write(batch)

        data = buf.getvalue()
        assert len(data) > 0

        # Write to disk
        os.makedirs(INTEROP_DIR, exist_ok=True)
        path = os.path.join(INTEROP_DIR, "python_written.mosaic")
        with open(path, "wb") as f:
            f.write(data)

        # Immediately verify we can read it back in Python
        def read_at(offset, length):
            return data[offset : offset + length]

        with MosaicReader.from_input_file(read_at, len(data)) as reader:
            total_rows = 0
            for rg in range(reader.num_row_groups):
                rb = reader.read_row_group(rg)
                rb_ids = rb.column("id").to_pylist()
                rb_names = rb.column("name").to_pylist()
                rb_scores = rb.column("score").to_pylist()

                for i in range(rb.num_rows):
                    row_idx = total_rows + i
                    assert rb_ids[i] == row_idx
                    if row_idx % 3 == 0:
                        assert rb_names[i] is None
                    else:
                        assert rb_names[i] == f"py_name_{row_idx}"
                    assert abs(rb_scores[i] - row_idx * 2.5) < 1e-9

                total_rows += rb.num_rows

            assert total_rows == num_rows

        print(f"test_write_file_read_from_rust: PASSED ({num_rows} rows written to {path})")
