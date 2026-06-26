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

import io
import struct

import pyarrow as pa
import pytest

from mosaic import (
    ColumnStatistics,
    MosaicReader,
    MosaicWriter,
    WriterOptions,
    read_table,
    write_table,
)


def _write_to_bytes(pa_schema, data, options=None):
    buf = io.BytesIO()
    with MosaicWriter(buf, pa_schema, options) as writer:
        writer.write(data)
    return buf.getvalue()


def _reader_from_bytes(data):
    return MosaicReader.from_input_file(
        lambda offset, length: data[offset : offset + length], len(data)
    )


class TestRoundtrip:
    def test_basic_roundtrip(self):
        pa_schema = pa.schema(
            [
                pa.field("id", pa.int32(), nullable=False),
                pa.field("name", pa.utf8()),
                pa.field("score", pa.float64()),
            ]
        )

        batch = pa.record_batch(
            [
                pa.array(list(range(50)), type=pa.int32()),
                pa.array([f"user_{i}" for i in range(50)]),
                pa.array([i * 1.5 for i in range(50)]),
            ],
            names=["id", "name", "score"],
        )

        data = _write_to_bytes(pa_schema, batch)
        assert len(data) > 32
        assert data[-4:] == b"MOSA"

        with _reader_from_bytes(data) as reader:
            assert reader.num_row_groups >= 1

            total_rows = 0
            for rg in range(reader.num_row_groups):
                rb = reader.read_row_group(rg)
                total_rows += rb.num_rows

                ids = rb.column("id").to_pylist()
                names = rb.column("name").to_pylist()
                scores = rb.column("score").to_pylist()

                for j in range(rb.num_rows):
                    idx = ids[j]
                    assert names[j] == f"user_{idx}"
                    assert abs(scores[j] - idx * 1.5) < 1e-9

            assert total_rows == 50

    def test_null_values(self):
        pa_schema = pa.schema(
            [
                pa.field("id", pa.int32()),
                pa.field("name", pa.utf8()),
                pa.field("value", pa.float64()),
            ]
        )

        batch = pa.record_batch(
            [
                pa.array([1, 2, 3, 4], type=pa.int32()),
                pa.array(["hello", None, "world", None]),
                pa.array([1.0, 2.0, None, None]),
            ],
            names=["id", "name", "value"],
        )

        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            rb = reader.read_row_group(0)
            assert rb.num_rows == 4

            names = rb.column("name")
            values = rb.column("value")

            assert names[0].as_py() == "hello"
            assert names[1].as_py() is None
            assert names[2].as_py() == "world"
            assert names[3].as_py() is None

            assert values[0].as_py() == 1.0
            assert values[1].as_py() == 2.0
            assert values[2].as_py() is None
            assert values[3].as_py() is None

    def test_compression_none(self):
        pa_schema = pa.schema(
            [pa.field("x", pa.int32()), pa.field("y", pa.utf8())]
        )
        batch = pa.record_batch(
            [
                pa.array(list(range(20)), type=pa.int32()),
                pa.array([f"v_{i}" for i in range(20)]),
            ],
            names=["x", "y"],
        )
        opts = WriterOptions(compression=WriterOptions.COMPRESSION_NONE)
        data = _write_to_bytes(pa_schema, batch, opts)

        with _reader_from_bytes(data) as reader:
            rb = reader.read_row_group(0)
            assert rb.num_rows == 20
            assert rb.column("x").to_pylist() == list(range(20))

    def test_compression_zstd(self):
        pa_schema = pa.schema(
            [pa.field("x", pa.int32()), pa.field("y", pa.utf8())]
        )
        batch = pa.record_batch(
            [
                pa.array(list(range(100)), type=pa.int32()),
                pa.array([f"v_{i}" for i in range(100)]),
            ],
            names=["x", "y"],
        )
        opts = WriterOptions(compression=WriterOptions.COMPRESSION_ZSTD, zstd_level=3)
        data = _write_to_bytes(pa_schema, batch, opts)

        with _reader_from_bytes(data) as reader:
            rb = reader.read_row_group(0)
            assert rb.num_rows == 100
            assert rb.column("x").to_pylist() == list(range(100))

    def test_all_types(self):
        pa_schema = pa.schema(
            [
                pa.field("f_bool", pa.bool_()),
                pa.field("f_int8", pa.int8()),
                pa.field("f_int16", pa.int16()),
                pa.field("f_int32", pa.int32()),
                pa.field("f_int64", pa.int64()),
                pa.field("f_float32", pa.float32()),
                pa.field("f_float64", pa.float64()),
                pa.field("f_utf8", pa.utf8()),
                pa.field("f_binary", pa.binary()),
                pa.field("f_decimal", pa.decimal128(10, 2)),
                pa.field("f_date", pa.date32()),
                pa.field("f_timestamp", pa.timestamp("ms")),
                pa.field("f_timestamp_ns", pa.timestamp("ns")),
                pa.field("f_timestamp_ns_tz", pa.timestamp("ns", tz="Asia/Shanghai")),
            ]
        )

        ts_ns_values = [1700000000000000123, -1]
        batch = pa.record_batch(
            [
                pa.array([True, False]),
                pa.array([42, -1], type=pa.int8()),
                pa.array([1234, -5678], type=pa.int16()),
                pa.array([100000, -200000], type=pa.int32()),
                pa.array([9999999999, -9999999999], type=pa.int64()),
                pa.array([3.14, -2.71], type=pa.float32()),
                pa.array([2.718281828, -3.141592653]),
                pa.array(["hello", "world"]),
                pa.array([b"\x01\x02\x03", b"\xff\x00"], type=pa.binary()),
                pa.array([1234567, -9876543], type=pa.decimal128(10, 2)),
                pa.array([19000, 0], type=pa.date32()),
                pa.array([1700000000000, 0], type=pa.timestamp("ms")),
                pa.array(ts_ns_values, type=pa.timestamp("ns")),
                pa.array(ts_ns_values, type=pa.timestamp("ns", tz="Asia/Shanghai")),
            ],
            names=[
                "f_bool", "f_int8", "f_int16", "f_int32", "f_int64",
                "f_float32", "f_float64", "f_utf8", "f_binary",
                "f_decimal", "f_date", "f_timestamp",
                "f_timestamp_ns", "f_timestamp_ns_tz",
            ],
        )

        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            rb = reader.read_row_group(0)
            assert rb.num_rows == 2

            assert rb.column("f_bool").to_pylist() == [True, False]
            assert rb.column("f_int8").to_pylist() == [42, -1]
            assert rb.column("f_int16").to_pylist() == [1234, -5678]
            assert rb.column("f_int32").to_pylist() == [100000, -200000]
            assert rb.column("f_int64").to_pylist() == [9999999999, -9999999999]
            assert rb.column("f_utf8").to_pylist() == ["hello", "world"]
            assert rb.column("f_binary").to_pylist() == [b"\x01\x02\x03", b"\xff\x00"]

            f32 = rb.column("f_float32").to_pylist()
            assert abs(f32[0] - 3.14) < 1e-5
            assert abs(f32[1] - (-2.71)) < 1e-5

            f64 = rb.column("f_float64").to_pylist()
            assert abs(f64[0] - 2.718281828) < 1e-9
            assert abs(f64[1] - (-3.141592653)) < 1e-9

            assert rb.schema.field("f_timestamp_ns").type == pa.timestamp("ns")
            assert rb.column("f_timestamp_ns").cast(pa.int64()).to_pylist() == ts_ns_values
            assert rb.schema.field("f_timestamp_ns_tz").type == pa.timestamp(
                "ns", tz="Asia/Shanghai"
            )
            assert rb.column("f_timestamp_ns_tz").cast(pa.int64()).to_pylist() == ts_ns_values

    def test_timestamp_nanos_mixed_batches_updates_dictionary(self):
        pa_schema = pa.schema([pa.field("ts", pa.timestamp("ns"))])
        first_values = [1, None, 2]
        second_values = [3 + (i % 3) for i in range(120)]

        buf = io.BytesIO()
        with MosaicWriter(buf, pa_schema) as writer:
            writer.write(
                pa.record_batch(
                    [pa.array(first_values, type=pa.timestamp("ns"))],
                    names=["ts"],
                )
            )
            writer.write(
                pa.record_batch(
                    [pa.array(second_values, type=pa.timestamp("ns"))],
                    names=["ts"],
                )
            )

        with _reader_from_bytes(buf.getvalue()) as reader:
            rb = reader.read_row_group(0)
            assert rb.column("ts").cast(pa.int64()).to_pylist() == (
                first_values + second_values
            )

    def test_multiple_row_groups(self):
        pa_schema = pa.schema(
            [pa.field("id", pa.int32()), pa.field("data", pa.int64())]
        )

        opts = WriterOptions(
            compression=WriterOptions.COMPRESSION_NONE,
            num_buckets=1,
            row_group_max_size=200,
        )
        buf = io.BytesIO()
        with MosaicWriter(buf, pa_schema, opts) as writer:
            for start in range(0, 500, 50):
                batch = pa.record_batch(
                    [
                        pa.array(list(range(start, start + 50)), type=pa.int32()),
                        pa.array(
                            [i * 3 for i in range(start, start + 50)],
                            type=pa.int64(),
                        ),
                    ],
                    names=["id", "data"],
                )
                writer.write(batch)
        data = buf.getvalue()

        with _reader_from_bytes(data) as reader:
            assert reader.num_row_groups > 1

            offset = 0
            for rg in range(reader.num_row_groups):
                rb = reader.read_row_group(rg)
                ids = rb.column("id").to_pylist()
                datas = rb.column("data").to_pylist()
                for j in range(rb.num_rows):
                    assert ids[j] == offset + j
                    assert datas[j] == (offset + j) * 3
                offset += rb.num_rows

            assert offset == 500

    def test_multiple_writes(self):
        pa_schema = pa.schema(
            [pa.field("x", pa.int32()), pa.field("y", pa.utf8())]
        )

        buf = io.BytesIO()
        with MosaicWriter(buf, pa_schema) as writer:
            for start in [0, 10, 20]:
                batch = pa.record_batch(
                    [
                        pa.array(list(range(start, start + 10)), type=pa.int32()),
                        pa.array([f"r_{i}" for i in range(start, start + 10)]),
                    ],
                    names=["x", "y"],
                )
                writer.write(batch)

        data = buf.getvalue()
        with _reader_from_bytes(data) as reader:
            table = reader.read_all()
            assert table.num_rows == 30
            xs = table.column("x").to_pylist()
            assert xs == list(range(30))

    def test_single_row(self):
        pa_schema = pa.schema([pa.field("v", pa.int32())])
        batch = pa.record_batch(
            [pa.array([42], type=pa.int32())], names=["v"]
        )
        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            rb = reader.read_row_group(0)
            assert rb.num_rows == 1
            assert rb.column("v")[0].as_py() == 42

    def test_zero_rows(self):
        pa_schema = pa.schema(
            [
                pa.field("v", pa.int32(), nullable=False),
                pa.field("s", pa.utf8(), nullable=True),
            ]
        )
        batch = pa.record_batch(
            [pa.array([], type=pa.int32()), pa.array([], type=pa.utf8())],
            names=["v", "s"],
        )
        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            table = reader.read_all()
            assert table.num_rows == 0
            assert table.schema.names == ["v", "s"]
            assert table.schema.field("v").type == pa.int32()
            assert table.schema.field("s").type == pa.utf8()
            assert not table.schema.field("v").nullable
            assert table.schema.field("s").nullable


class TestProjection:
    def test_projection_subset(self):
        pa_schema = pa.schema(
            [
                pa.field("a", pa.int32()),
                pa.field("b", pa.utf8()),
                pa.field("c", pa.float64()),
                pa.field("d", pa.utf8()),
            ]
        )

        batch = pa.record_batch(
            [
                pa.array(list(range(20)), type=pa.int32()),
                pa.array([f"val_{i}" for i in range(20)]),
                pa.array([float(i) for i in range(20)]),
                pa.array([f"extra_{i}" for i in range(20)]),
            ],
            names=["a", "b", "c", "d"],
        )

        opts = WriterOptions(num_buckets=2)
        data = _write_to_bytes(pa_schema, batch, opts)

        with _reader_from_bytes(data) as reader:
            reader.project(["a", "b"])

            total_rows = 0
            for rg in range(reader.num_row_groups):
                rb = reader.read_row_group(rg)
                assert rb.num_columns == 2
                total_rows += rb.num_rows

            assert total_rows == 20

    def test_projection_single_column(self):
        pa_schema = pa.schema(
            [
                pa.field("a", pa.int32()),
                pa.field("b", pa.utf8()),
                pa.field("c", pa.float64()),
            ]
        )

        batch = pa.record_batch(
            [
                pa.array(list(range(10)), type=pa.int32()),
                pa.array([f"v_{i}" for i in range(10)]),
                pa.array([float(i) for i in range(10)]),
            ],
            names=["a", "b", "c"],
        )

        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            reader.project(["b"])
            rb = reader.read_row_group(0)
            assert rb.num_columns == 1
            assert rb.num_rows == 10

    def test_projection_preserves_order(self):
        pa_schema = pa.schema(
            [
                pa.field("a", pa.int32()),
                pa.field("b", pa.utf8()),
                pa.field("c", pa.float64()),
            ]
        )

        batch = pa.record_batch(
            [
                pa.array(list(range(10)), type=pa.int32()),
                pa.array([f"s{i}" for i in range(10)]),
                pa.array([float(i) * 0.5 for i in range(10)]),
            ],
            names=["a", "b", "c"],
        )

        opts = WriterOptions(num_buckets=2)
        data = _write_to_bytes(pa_schema, batch, opts)

        with _reader_from_bytes(data) as reader:
            reader.project(["c", "a", "b"])
            rb = reader.read_row_group(0)
            assert rb.num_columns == 3
            assert rb.schema.names == ["c", "a", "b"]
            assert rb.column("c").to_pylist() == [i * 0.5 for i in range(10)]
            assert rb.column("a").to_pylist() == list(range(10))
            assert rb.column("b").to_pylist() == [f"s{i}" for i in range(10)]

    def test_projection_empty(self):
        pa_schema = pa.schema(
            [
                pa.field("a", pa.int32()),
                pa.field("b", pa.utf8()),
            ]
        )

        batch = pa.record_batch(
            [
                pa.array(list(range(5)), type=pa.int32()),
                pa.array([f"v{i}" for i in range(5)]),
            ],
            names=["a", "b"],
        )

        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            reader.project([])
            rb = reader.read_row_group(0)
            assert rb.num_columns == 0
            assert rb.num_rows == 5


class TestSchema:
    def test_schema_roundtrip(self):
        pa_schema = pa.schema(
            [
                pa.field("name", pa.utf8(), nullable=True),
                pa.field("id", pa.int32(), nullable=False),
                pa.field("score", pa.float64(), nullable=True),
            ]
        )

        batch = pa.record_batch(
            [
                pa.array(["x"]),
                pa.array([1], type=pa.int32()),
                pa.array([1.0]),
            ],
            names=["name", "id", "score"],
        )

        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            s = reader.schema
            assert len(s) == 3
            assert s.names == ["name", "id", "score"]
            assert s.field("id").type == pa.int32()
            assert s.field("name").type == pa.utf8()
            assert s.field("score").type == pa.float64()
            assert not s.field("id").nullable
            assert s.field("name").nullable


class TestStatistics:
    def test_stats_basic(self):
        pa_schema = pa.schema(
            [
                pa.field("id", pa.int32()),
                pa.field("name", pa.utf8()),
                pa.field("score", pa.float64()),
            ]
        )

        batch = pa.record_batch(
            [
                pa.array([i * 10 for i in range(10)], type=pa.int32()),
                pa.array([f"item_{i}" for i in range(10)]),
                pa.array([i * 1.1 for i in range(10)]),
            ],
            names=["id", "name", "score"],
        )

        opts = WriterOptions(stats_columns=["id", "score"])
        data = _write_to_bytes(pa_schema, batch, opts)

        with _reader_from_bytes(data) as reader:
            for rg in range(reader.num_row_groups):
                stats = reader.get_row_group_statistics(rg)
                assert len(stats) > 0
                assert isinstance(stats, dict)
                for name, stat in stats.items():
                    assert isinstance(stat, ColumnStatistics)
                    assert name in ("id", "score")
                    assert stat.null_count == 0
                    assert stat.has_min_max
                    assert stat.min is not None
                    assert stat.max is not None
                    assert len(stat.min) > 0
                    assert len(stat.max) > 0

                id_stat = stats["id"]
                min_id = struct.unpack(">i", id_stat.min)[0]
                max_id = struct.unpack(">i", id_stat.max)[0]
                assert min_id == 0
                assert max_id == 90

    def test_stats_with_nulls(self):
        pa_schema = pa.schema(
            [pa.field("a", pa.int32()), pa.field("b", pa.int64())]
        )

        batch = pa.record_batch(
            [
                pa.array([10, None, 5, 20], type=pa.int32()),
                pa.array([None, None, 100, 50], type=pa.int64()),
            ],
            names=["a", "b"],
        )

        opts = WriterOptions(stats_columns=["a", "b"], num_buckets=1)
        data = _write_to_bytes(pa_schema, batch, opts)

        with _reader_from_bytes(data) as reader:
            stats = reader.get_row_group_statistics(0)
            assert len(stats) == 2

            a_stat = stats["a"]
            assert a_stat.null_count == 1
            assert a_stat.has_min_max
            min_a = struct.unpack(">i", a_stat.min)[0]
            max_a = struct.unpack(">i", a_stat.max)[0]
            assert min_a == 5
            assert max_a == 20

            b_stat = stats["b"]
            assert b_stat.null_count == 2
            assert b_stat.has_min_max
            min_b = struct.unpack(">q", b_stat.min)[0]
            max_b = struct.unpack(">q", b_stat.max)[0]
            assert min_b == 50
            assert max_b == 100

    def test_stats_all_null(self):
        pa_schema = pa.schema([pa.field("x", pa.int32())])

        batch = pa.record_batch(
            [pa.array([None, None, None], type=pa.int32())], names=["x"]
        )

        opts = WriterOptions(stats_columns=["x"], num_buckets=1)
        data = _write_to_bytes(pa_schema, batch, opts)

        with _reader_from_bytes(data) as reader:
            stats = reader.get_row_group_statistics(0)
            assert len(stats) == 1
            assert stats["x"].null_count == 3
            assert not stats["x"].has_min_max
            assert stats["x"].min is None
            assert stats["x"].max is None


    def test_stats_empty_string_min(self):
        pa_schema = pa.schema([pa.field("s", pa.utf8())])

        batch = pa.record_batch(
            [pa.array(["", "b"])], names=["s"]
        )

        opts = WriterOptions(stats_columns=["s"], num_buckets=1)
        buf = io.BytesIO()
        with MosaicWriter(buf, pa_schema, opts) as writer:
            writer.write(batch)

        writer_stats = writer.get_row_group_statistics(0)
        assert "s" in writer_stats
        s_stat = writer_stats["s"]
        assert s_stat.has_min_max
        assert s_stat.min == b""
        assert s_stat.max == b"b"
        assert s_stat.null_count == 0

        data = buf.getvalue()
        with _reader_from_bytes(data) as reader:
            stats = reader.get_row_group_statistics(0)
            assert stats["s"].has_min_max
            assert stats["s"].min == b""
            assert stats["s"].max == b"b"


class TestConvenience:
    def test_write_table_read_table(self):
        table = pa.table(
            {
                "id": pa.array(list(range(30)), type=pa.int32()),
                "name": pa.array([f"user_{i}" for i in range(30)]),
            }
        )

        buf = io.BytesIO()
        write_table(table, buf)

        data = buf.getvalue()
        result = read_table(
            lambda offset, length: data[offset : offset + length], len(data)
        )

        assert result.num_rows == 30
        assert result.column("id").to_pylist() == list(range(30))
        assert result.column("name").to_pylist() == [f"user_{i}" for i in range(30)]

    def test_read_table_empty_projection(self):
        table = pa.table(
            {
                "id": pa.array(list(range(30)), type=pa.int32()),
                "name": pa.array([f"user_{i}" for i in range(30)]),
            }
        )

        buf = io.BytesIO()
        write_table(table, buf)

        data = buf.getvalue()
        result = read_table(
            lambda offset, length: data[offset : offset + length],
            len(data),
            columns=[],
        )

        assert result.num_columns == 0
        assert result.num_rows == 30
        assert result.schema.names == []

    def test_read_table_projection_with_zero_row_groups(self):
        pa_schema = pa.schema(
            [
                pa.field("id", pa.int32()),
                pa.field("name", pa.utf8()),
            ]
        )

        buf = io.BytesIO()
        with MosaicWriter(buf, pa_schema):
            pass

        data = buf.getvalue()

        empty_projection = read_table(
            lambda offset, length: data[offset : offset + length],
            len(data),
            columns=[],
        )
        assert empty_projection.num_rows == 0
        assert empty_projection.num_columns == 0
        assert empty_projection.schema.names == []

        name_projection = read_table(
            lambda offset, length: data[offset : offset + length],
            len(data),
            columns=["name"],
        )
        assert name_projection.num_rows == 0
        assert name_projection.num_columns == 1
        assert name_projection.schema.names == ["name"]

        duplicate_projection = read_table(
            lambda offset, length: data[offset : offset + length],
            len(data),
            columns=["name", "name"],
        )
        assert duplicate_projection.num_rows == 0
        assert duplicate_projection.num_columns == 1
        assert duplicate_projection.schema.names == ["name"]

        with _reader_from_bytes(data) as reader:
            assert reader.num_row_groups == 0
            reader.project(["name"])
            assert reader.schema.names == ["id", "name"]
            result = reader.read_all()
            assert result.schema.names == ["name"]

    def test_read_all(self):
        pa_schema = pa.schema(
            [pa.field("x", pa.int32()), pa.field("y", pa.utf8())]
        )

        batch = pa.record_batch(
            [
                pa.array(list(range(25)), type=pa.int32()),
                pa.array([f"row_{i}" for i in range(25)]),
            ],
            names=["x", "y"],
        )

        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            table = reader.read_all()
            assert isinstance(table, pa.Table)
            assert table.num_rows == 25
            assert table.column("x").to_pylist() == list(range(25))


class TestWriter:
    def test_estimated_file_size(self):
        pa_schema = pa.schema(
            [pa.field("x", pa.int32()), pa.field("y", pa.utf8())]
        )

        batch = pa.record_batch(
            [
                pa.array(list(range(100)), type=pa.int32()),
                pa.array([f"value_{i}" for i in range(100)]),
            ],
            names=["x", "y"],
        )

        buf = io.BytesIO()
        with MosaicWriter(buf, pa_schema) as writer:
            writer.write(batch)
            est = writer.estimated_file_size()
            assert est > 0

    def test_write_after_close_fails_before_export(self):
        pa_schema = pa.schema([pa.field("x", pa.int32())])
        batch = pa.record_batch(
            [pa.array([1, 2, 3], type=pa.int32())], names=["x"]
        )

        writer = MosaicWriter(io.BytesIO(), pa_schema)
        writer.close()

        with pytest.raises(RuntimeError, match="writer is closed"):
            writer.write(batch)

    def test_writer_stats_basic(self):
        pa_schema = pa.schema(
            [
                pa.field("id", pa.int32()),
                pa.field("name", pa.utf8()),
                pa.field("score", pa.float64()),
            ]
        )

        batch = pa.record_batch(
            [
                pa.array([i * 10 for i in range(10)], type=pa.int32()),
                pa.array([f"item_{i}" for i in range(10)]),
                pa.array([i * 1.1 for i in range(10)]),
            ],
            names=["id", "name", "score"],
        )

        opts = WriterOptions(stats_columns=["id", "score"])
        buf = io.BytesIO()
        writer = MosaicWriter(buf, pa_schema, opts)
        writer.write(batch)
        writer.close()

        assert writer.num_row_groups >= 1
        stats = writer.get_row_group_statistics(0)
        assert len(stats) > 0
        assert isinstance(stats, dict)
        for name, stat in stats.items():
            assert isinstance(stat, ColumnStatistics)
            assert name in ("id", "score")
            assert stat.null_count == 0
            assert stat.has_min_max
            assert stat.min is not None
            assert stat.max is not None

        id_stat = stats["id"]
        min_id = struct.unpack(">i", id_stat.min)[0]
        max_id = struct.unpack(">i", id_stat.max)[0]
        assert min_id == 0
        assert max_id == 90

    def test_writer_stats_with_nulls(self):
        pa_schema = pa.schema(
            [pa.field("a", pa.int32()), pa.field("b", pa.int64())]
        )

        batch = pa.record_batch(
            [
                pa.array([10, None, 5, 20], type=pa.int32()),
                pa.array([None, None, 100, 50], type=pa.int64()),
            ],
            names=["a", "b"],
        )

        opts = WriterOptions(stats_columns=["a", "b"], num_buckets=1)
        buf = io.BytesIO()
        writer = MosaicWriter(buf, pa_schema, opts)
        writer.write(batch)
        writer.close()

        assert writer.num_row_groups == 1
        stats = writer.get_row_group_statistics(0)
        assert len(stats) == 2

        a_stat = stats["a"]
        assert a_stat.null_count == 1
        assert a_stat.has_min_max
        min_a = struct.unpack(">i", a_stat.min)[0]
        max_a = struct.unpack(">i", a_stat.max)[0]
        assert min_a == 5
        assert max_a == 20

        b_stat = stats["b"]
        assert b_stat.null_count == 2
        assert b_stat.has_min_max
        min_b = struct.unpack(">q", b_stat.min)[0]
        max_b = struct.unpack(">q", b_stat.max)[0]
        assert min_b == 50
        assert max_b == 100

    def test_writer_stats_all_null(self):
        pa_schema = pa.schema([pa.field("x", pa.int32())])

        batch = pa.record_batch(
            [pa.array([None, None, None], type=pa.int32())], names=["x"]
        )

        opts = WriterOptions(stats_columns=["x"], num_buckets=1)
        buf = io.BytesIO()
        writer = MosaicWriter(buf, pa_schema, opts)
        writer.write(batch)
        writer.close()

        assert writer.num_row_groups == 1
        stats = writer.get_row_group_statistics(0)
        assert len(stats) == 1
        assert stats["x"].null_count == 3
        assert not stats["x"].has_min_max
        assert stats["x"].min is None
        assert stats["x"].max is None

    def test_writer_stats_matches_reader_stats(self):
        pa_schema = pa.schema(
            [
                pa.field("id", pa.int32()),
                pa.field("score", pa.float64()),
            ]
        )

        batch = pa.record_batch(
            [
                pa.array(list(range(50)), type=pa.int32()),
                pa.array([i * 2.0 for i in range(50)]),
            ],
            names=["id", "score"],
        )

        opts = WriterOptions(stats_columns=["id", "score"], num_buckets=1)
        buf = io.BytesIO()
        writer = MosaicWriter(buf, pa_schema, opts)
        writer.write(batch)
        writer.close()

        data = buf.getvalue()
        with _reader_from_bytes(data) as reader:
            for rg in range(writer.num_row_groups):
                w_stats = writer.get_row_group_statistics(rg)
                r_stats = reader.get_row_group_statistics(rg)
                assert len(w_stats) == len(r_stats)
                assert set(w_stats.keys()) == set(r_stats.keys())
                for name in w_stats:
                    assert w_stats[name].null_count == r_stats[name].null_count
                    assert w_stats[name].has_min_max == r_stats[name].has_min_max
                    assert w_stats[name].min == r_stats[name].min
                    assert w_stats[name].max == r_stats[name].max

    def test_row_group_num_rows(self):
        pa_schema = pa.schema(
            [pa.field("id", pa.int32()), pa.field("data", pa.int64())]
        )

        opts = WriterOptions(
            compression=WriterOptions.COMPRESSION_NONE,
            num_buckets=1,
            row_group_max_size=200,
        )
        buf = io.BytesIO()
        with MosaicWriter(buf, pa_schema, opts) as writer:
            for start in range(0, 500, 50):
                batch = pa.record_batch(
                    [
                        pa.array(list(range(start, start + 50)), type=pa.int32()),
                        pa.array(
                            [i * 2 for i in range(start, start + 50)],
                            type=pa.int64(),
                        ),
                    ],
                    names=["id", "data"],
                )
                writer.write(batch)
        data = buf.getvalue()

        with _reader_from_bytes(data) as reader:
            assert reader.num_row_groups > 1
            total = 0
            for rg in range(reader.num_row_groups):
                num_rows = reader.row_group_num_rows(rg)
                assert num_rows > 0
                rb = reader.read_row_group(rg)
                assert num_rows == rb.num_rows
                total += num_rows
            assert total == 500

    def test_row_group_num_rows_single(self):
        pa_schema = pa.schema([pa.field("x", pa.int32())])
        batch = pa.record_batch(
            [pa.array(list(range(10)), type=pa.int32())], names=["x"]
        )
        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            assert reader.num_row_groups == 1
            assert reader.row_group_num_rows(0) == 10

    def test_array_type(self):
        pa_schema = pa.schema(
            [
                pa.field("id", pa.int32(), nullable=False),
                pa.field("tags", pa.list_(pa.int32())),
            ]
        )

        batch = pa.record_batch(
            [
                pa.array([1, 2, 3, 4], type=pa.int32()),
                pa.array([[10, 20, 30], [40, 50], [], None], type=pa.list_(pa.int32())),
            ],
            names=["id", "tags"],
        )

        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            rb = reader.read_row_group(0)
            assert rb.num_rows == 4

            ids = rb.column("id").to_pylist()
            assert ids == [1, 2, 3, 4]

            tags = rb.column("tags").to_pylist()
            assert tags[0] == [10, 20, 30]
            assert tags[1] == [40, 50]
            assert tags[2] == []
            assert tags[3] is None

    def test_array_with_null_elements(self):
        pa_schema = pa.schema(
            [pa.field("arr", pa.list_(pa.int64()))]
        )

        batch = pa.record_batch(
            [pa.array([[100, None, 300], [None, None], [999]], type=pa.list_(pa.int64()))],
            names=["arr"],
        )

        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            rb = reader.read_row_group(0)
            arr = rb.column("arr").to_pylist()
            assert arr[0] == [100, None, 300]
            assert arr[1] == [None, None]
            assert arr[2] == [999]

    def test_array_string_elements(self):
        pa_schema = pa.schema(
            [pa.field("arr", pa.list_(pa.utf8()))]
        )

        batch = pa.record_batch(
            [pa.array([["hello", "world"], [None, "foo"], []], type=pa.list_(pa.utf8()))],
            names=["arr"],
        )

        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            rb = reader.read_row_group(0)
            arr = rb.column("arr").to_pylist()
            assert arr[0] == ["hello", "world"]
            assert arr[1] == [None, "foo"]
            assert arr[2] == []

    def test_array_nested(self):
        pa_schema = pa.schema(
            [pa.field("nested", pa.list_(pa.list_(pa.int32())))]
        )

        batch = pa.record_batch(
            [
                pa.array(
                    [[[1, 2], [3]], [[4]], None],
                    type=pa.list_(pa.list_(pa.int32())),
                )
            ],
            names=["nested"],
        )

        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            rb = reader.read_row_group(0)
            nested = rb.column("nested").to_pylist()
            assert nested[0] == [[1, 2], [3]]
            assert nested[1] == [[4]]
            assert nested[2] is None

    def test_map_type(self):
        pa_schema = pa.schema(
            [
                pa.field("id", pa.int32(), nullable=False),
                pa.field("props", pa.map_(pa.int32(), pa.utf8())),
            ]
        )

        batch = pa.record_batch(
            [
                pa.array([1, 2, 3], type=pa.int32()),
                pa.array(
                    [[(1, "a"), (2, "b")], None, []],
                    type=pa.map_(pa.int32(), pa.utf8()),
                ),
            ],
            names=["id", "props"],
        )

        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            rb = reader.read_row_group(0)
            assert rb.num_rows == 3

            ids = rb.column("id").to_pylist()
            assert ids == [1, 2, 3]

            props = rb.column("props").to_pylist()
            assert props[0] == [(1, "a"), (2, "b")]
            assert props[1] is None
            assert props[2] == []

    def test_map_with_null_values(self):
        pa_schema = pa.schema(
            [pa.field("m", pa.map_(pa.utf8(), pa.int64()))]
        )

        batch = pa.record_batch(
            [
                pa.array(
                    [[("x", 10), ("y", None)], [("z", 30)]],
                    type=pa.map_(pa.utf8(), pa.int64()),
                )
            ],
            names=["m"],
        )

        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            rb = reader.read_row_group(0)
            m = rb.column("m").to_pylist()
            assert m[0] == [("x", 10), ("y", None)]
            assert m[1] == [("z", 30)]
