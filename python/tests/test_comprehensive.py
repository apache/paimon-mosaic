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
import os
import random

import pyarrow as pa
import pytest

from mosaic import MosaicReader, MosaicWriter, WriterOptions


def _write_to_bytes(pa_schema, data, options=None):
    buf = io.BytesIO()
    with MosaicWriter(buf, pa_schema, options) as writer:
        writer.write(data)
    return buf.getvalue()


def _reader_from_bytes(data):
    return MosaicReader.from_input_file(
        lambda offset, length: data[offset : offset + length], len(data)
    )


class TestComprehensive:
    def test_million_rows_roundtrip(self):
        """1M rows roundtrip with Int64 + Utf8 + Float64 + Boolean."""
        total_rows = 1_000_000
        batch_size = 100_000
        pa_schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.utf8()),
                pa.field("score", pa.float64()),
                pa.field("flag", pa.bool_()),
            ]
        )

        buf = io.BytesIO()
        with MosaicWriter(buf, pa_schema) as writer:
            for start in range(0, total_rows, batch_size):
                count = min(batch_size, total_rows - start)
                ids = list(range(start, start + count))
                batch = pa.record_batch(
                    [
                        pa.array(ids, type=pa.int64()),
                        pa.array([f"row_{i}" for i in ids]),
                        pa.array([i * 0.001 for i in ids]),
                        pa.array([i % 2 == 0 for i in ids]),
                    ],
                    names=["id", "name", "score", "flag"],
                )
                writer.write(batch)

        data = buf.getvalue()
        assert len(data) > 0

        with _reader_from_bytes(data) as reader:
            table = reader.read_all()
            assert table.num_rows == total_rows
            # Spot-check some values
            ids = table.column("id").to_pylist()
            names = table.column("name").to_pylist()
            scores = table.column("score").to_pylist()
            flags = table.column("flag").to_pylist()
            for i in [0, 999, 500_000, 999_999]:
                assert ids[i] == i
                assert names[i] == f"row_{i}"
                assert abs(scores[i] - i * 0.001) < 1e-9
                assert flags[i] == (i % 2 == 0)

    def test_all_constant_data(self):
        """All constant values should produce a small output file."""
        total_rows = 500_000
        batch_size = 100_000
        pa_schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.utf8()),
                pa.field("score", pa.float64()),
                pa.field("flag", pa.bool_()),
            ]
        )

        buf = io.BytesIO()
        with MosaicWriter(buf, pa_schema) as writer:
            for start in range(0, total_rows, batch_size):
                count = min(batch_size, total_rows - start)
                batch = pa.record_batch(
                    [
                        pa.array([42] * count, type=pa.int64()),
                        pa.array(["constant"] * count),
                        pa.array([3.14] * count),
                        pa.array([True] * count),
                    ],
                    names=["id", "name", "score", "flag"],
                )
                writer.write(batch)

        data = buf.getvalue()
        # Constant data should compress very well
        assert len(data) < 500_000, f"Constant data file too large: {len(data)} bytes"

        with _reader_from_bytes(data) as reader:
            table = reader.read_all()
            assert table.num_rows == total_rows
            assert all(v == 42 for v in table.column("id").to_pylist())
            assert all(v == "constant" for v in table.column("name").to_pylist())
            assert all(abs(v - 3.14) < 1e-9 for v in table.column("score").to_pylist())
            assert all(v is True for v in table.column("flag").to_pylist())

    def test_high_null_rate(self):
        """95% null data across all columns."""
        total_rows = 500_000
        batch_size = 100_000
        pa_schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.utf8()),
                pa.field("score", pa.float64()),
            ]
        )

        rng = random.Random(12345)
        all_ids = []
        all_names = []
        all_scores = []

        for i in range(total_rows):
            all_ids.append(i if rng.random() >= 0.95 else None)
            all_names.append(f"name_{i}" if rng.random() >= 0.95 else None)
            all_scores.append(i * 0.5 if rng.random() >= 0.95 else None)

        buf = io.BytesIO()
        with MosaicWriter(buf, pa_schema) as writer:
            for start in range(0, total_rows, batch_size):
                end = min(start + batch_size, total_rows)
                batch = pa.record_batch(
                    [
                        pa.array(all_ids[start:end], type=pa.int64()),
                        pa.array(all_names[start:end], type=pa.utf8()),
                        pa.array(all_scores[start:end], type=pa.float64()),
                    ],
                    names=["id", "name", "score"],
                )
                writer.write(batch)

        data = buf.getvalue()
        with _reader_from_bytes(data) as reader:
            table = reader.read_all()
            assert table.num_rows == total_rows

            read_ids = table.column("id").to_pylist()
            read_names = table.column("name").to_pylist()
            read_scores = table.column("score").to_pylist()

            assert read_ids == all_ids
            assert read_names == all_names
            for i in range(total_rows):
                if all_scores[i] is None:
                    assert read_scores[i] is None
                else:
                    assert abs(read_scores[i] - all_scores[i]) < 1e-9

    def test_wide_table_100_columns(self):
        """100 columns with mixed types."""
        num_cols = 100
        total_rows = 50_000

        fields = []
        for c in range(num_cols):
            if c % 4 == 0:
                fields.append(pa.field(f"col_{c}", pa.int32()))
            elif c % 4 == 1:
                fields.append(pa.field(f"col_{c}", pa.int64()))
            elif c % 4 == 2:
                fields.append(pa.field(f"col_{c}", pa.float64()))
            else:
                fields.append(pa.field(f"col_{c}", pa.utf8()))

        pa_schema = pa.schema(fields)

        batch_size = 10_000
        buf = io.BytesIO()
        with MosaicWriter(buf, pa_schema) as writer:
            for start in range(0, total_rows, batch_size):
                count = min(batch_size, total_rows - start)
                arrays = []
                for c in range(num_cols):
                    vals = list(range(start, start + count))
                    if c % 4 == 0:
                        arrays.append(pa.array(vals, type=pa.int32()))
                    elif c % 4 == 1:
                        arrays.append(pa.array([v * 100 for v in vals], type=pa.int64()))
                    elif c % 4 == 2:
                        arrays.append(pa.array([v * 0.1 for v in vals]))
                    else:
                        arrays.append(pa.array([f"v{v}" for v in vals]))
                batch = pa.record_batch(
                    arrays, names=[f"col_{c}" for c in range(num_cols)]
                )
                writer.write(batch)

        data = buf.getvalue()
        with _reader_from_bytes(data) as reader:
            assert len(reader.schema) == num_cols
            table = reader.read_all()
            assert table.num_rows == total_rows
            assert table.num_columns == num_cols

    def test_sequential_data(self):
        """Strictly sequential values should compress well."""
        total_rows = 100_000
        pa_schema = pa.schema([pa.field("val", pa.int64())])

        batch = pa.record_batch(
            [pa.array(list(range(total_rows)), type=pa.int64())], names=["val"]
        )
        seq_data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(seq_data) as reader:
            table = reader.read_all()
            assert table.num_rows == total_rows
            assert table.column("val").to_pylist() == list(range(total_rows))

    def test_random_data(self):
        """Random values roundtrip and should be larger than sequential."""
        total_rows = 100_000
        pa_schema = pa.schema([pa.field("val", pa.int64())])

        rng = random.Random(42)
        rand_vals = [rng.randint(-(2**62), 2**62) for _ in range(total_rows)]
        batch = pa.record_batch(
            [pa.array(rand_vals, type=pa.int64())], names=["val"]
        )
        rand_data = _write_to_bytes(pa_schema, batch)

        # Sequential for comparison
        seq_batch = pa.record_batch(
            [pa.array(list(range(total_rows)), type=pa.int64())], names=["val"]
        )
        seq_data = _write_to_bytes(pa_schema, seq_batch)

        assert len(seq_data) < len(rand_data), (
            f"Sequential ({len(seq_data)}) should be smaller than random ({len(rand_data)})"
        )

        with _reader_from_bytes(rand_data) as reader:
            table = reader.read_all()
            assert table.num_rows == total_rows
            assert table.column("val").to_pylist() == rand_vals

    def test_many_small_batches(self):
        """5000 batches of 20 rows each."""
        num_batches = 5000
        batch_size = 20
        total_rows = num_batches * batch_size
        pa_schema = pa.schema(
            [pa.field("id", pa.int32()), pa.field("val", pa.utf8())]
        )

        buf = io.BytesIO()
        with MosaicWriter(buf, pa_schema) as writer:
            for b in range(num_batches):
                start = b * batch_size
                batch = pa.record_batch(
                    [
                        pa.array(
                            list(range(start, start + batch_size)), type=pa.int32()
                        ),
                        pa.array([f"r{i}" for i in range(start, start + batch_size)]),
                    ],
                    names=["id", "val"],
                )
                writer.write(batch)

        data = buf.getvalue()
        with _reader_from_bytes(data) as reader:
            table = reader.read_all()
            assert table.num_rows == total_rows

    def test_projection_subset(self):
        """Read only a few columns from a table with many columns."""
        num_cols = 50
        total_rows = 10_000

        fields = [pa.field(f"col_{c}", pa.int32()) for c in range(num_cols)]
        pa_schema = pa.schema(fields)

        arrays = []
        for c in range(num_cols):
            arrays.append(
                pa.array([c * 1000 + i for i in range(total_rows)], type=pa.int32())
            )
        batch = pa.record_batch(arrays, names=[f"col_{c}" for c in range(num_cols)])

        opts = WriterOptions(num_buckets=2)
        data = _write_to_bytes(pa_schema, batch, opts)

        with _reader_from_bytes(data) as reader:
            reader.project(pa.schema([
                pa.field("col_0", pa.int32()),
                pa.field("col_25", pa.int32()),
                pa.field("col_49", pa.int32()),
            ]))
            table = reader.read_all()
            assert table.num_columns == 3
            assert table.num_rows == total_rows
            assert table.schema.names == ["col_0", "col_25", "col_49"]

            col0 = table.column("col_0").to_pylist()
            col25 = table.column("col_25").to_pylist()
            col49 = table.column("col_49").to_pylist()
            for i in range(total_rows):
                orig_i = col0[i]  # col_0 value = 0*1000 + original_i
                assert col25[i] == 25 * 1000 + orig_i
                assert col49[i] == 49 * 1000 + orig_i

    def test_unicode_strings(self):
        """Chinese, emoji, and mixed script strings."""
        pa_schema = pa.schema(
            [pa.field("id", pa.int32()), pa.field("text", pa.utf8())]
        )

        unicode_strings = [
            "Hello World",
            "你好世界",
            "こんにちは世界",
            "안녕하세요",
            "Привет мир",
            "مرحبا بالعالم",
            "🎉🎊🎈🎁",
            "Mixed: 你好 hello こんにちは 🌍",
            "",
            "a" * 1000,
            "中" * 500,
            "🎵" * 200,
        ]

        total_rows = 10_000
        ids = list(range(total_rows))
        texts = [unicode_strings[i % len(unicode_strings)] for i in range(total_rows)]

        batch = pa.record_batch(
            [
                pa.array(ids, type=pa.int32()),
                pa.array(texts),
            ],
            names=["id", "text"],
        )
        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            table = reader.read_all()
            assert table.num_rows == total_rows
            read_texts = table.column("text").to_pylist()
            for i in range(total_rows):
                assert read_texts[i] == texts[i], (
                    f"Mismatch at row {i}: expected {texts[i]!r}, got {read_texts[i]!r}"
                )

    def test_large_binary_values(self):
        """Binary values of 1KB-10KB each."""
        pa_schema = pa.schema(
            [pa.field("id", pa.int32()), pa.field("blob", pa.binary())]
        )

        total_rows = 5_000
        rng = random.Random(99)
        blobs = []
        for i in range(total_rows):
            size = rng.randint(1024, 10240)
            blobs.append(bytes(rng.getrandbits(8) for _ in range(size)))

        batch = pa.record_batch(
            [
                pa.array(list(range(total_rows)), type=pa.int32()),
                pa.array(blobs, type=pa.binary()),
            ],
            names=["id", "blob"],
        )
        data = _write_to_bytes(pa_schema, batch)

        with _reader_from_bytes(data) as reader:
            table = reader.read_all()
            assert table.num_rows == total_rows
            read_blobs = table.column("blob").to_pylist()
            for i in range(total_rows):
                assert read_blobs[i] == blobs[i], f"Blob mismatch at row {i}"
