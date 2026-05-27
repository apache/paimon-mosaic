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

import pyarrow as pa

from mosaic import (
    BloomFilterConfig,
    MosaicReader,
    MosaicWriter,
    WriterOptions,
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


class TestBloomFilter:
    def test_int64_column_hits_and_misses(self):
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        batch = pa.record_batch([pa.array(list(range(2000)), type=pa.int64())], names=["id"])
        opts = WriterOptions(
            num_buckets=2,
            bloom_filter_columns=[BloomFilterConfig("id", ndv=2000, fpp=0.01)],
        )
        data = _write_to_bytes(schema, batch, opts)
        with _reader_from_bytes(data) as reader:
            for v in range(2000):
                assert reader.bloom_might_contain(0, "id", v), f"present {v} missed"
            probe = 5000
            fp = 0
            for v in range(1_000_000, 1_000_000 + probe):
                if reader.bloom_might_contain(0, "id", v):
                    fp += 1
            assert fp / probe < 0.05, f"observed fpp {fp / probe} too high"

    def test_string_column_rejects_absent(self):
        schema = pa.schema([pa.field("name", pa.utf8(), nullable=False)])
        names = ["alice", "bob", "carol", "dave", "eve"]
        batch = pa.record_batch([pa.array(names)], names=["name"])
        opts = WriterOptions(
            num_buckets=1,
            bloom_filter_columns=[BloomFilterConfig("name", ndv=1024, fpp=0.001)],
        )
        data = _write_to_bytes(schema, batch, opts)
        with _reader_from_bytes(data) as reader:
            for n in names:
                assert reader.bloom_might_contain(0, "name", n), f"present {n} missed"
            assert not reader.bloom_might_contain(0, "name", "zachary")

    def test_no_bloom_returns_true_conservatively(self):
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        batch = pa.record_batch([pa.array([1, 2, 3], type=pa.int64())], names=["id"])
        data = _write_to_bytes(schema, batch, WriterOptions(num_buckets=1))
        with _reader_from_bytes(data) as reader:
            assert reader.bloom_might_contain(0, "id", 99999)
