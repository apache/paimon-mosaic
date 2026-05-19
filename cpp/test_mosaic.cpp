/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "mosaic.hpp"

#include <arrow/api.h>
#include <arrow/c/bridge.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

#define ASSERT_EQ(a, b) do { if ((a) != (b)) { \
    fprintf(stderr, "FAIL %s:%d: %s != %s\n", __FILE__, __LINE__, #a, #b); abort(); } } while(0)
#define ASSERT_TRUE(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); abort(); } } while(0)

struct MemBuffer {
    std::vector<uint8_t> data;
    size_t pos = 0;
};

static mosaic::OutputFile make_output(MemBuffer& buf) {
    mosaic::OutputFile out;
    out.write_fn = [&buf](const uint8_t* data, size_t len) -> int {
        buf.data.insert(buf.data.end(), data, data + len);
        buf.pos += len;
        return 0;
    };
    out.flush_fn = [&buf]() -> int { return 0; };
    out.get_pos_fn = [&buf]() -> int64_t { return static_cast<int64_t>(buf.pos); };
    return out;
}

static mosaic::InputFile make_input(const MemBuffer& buf) {
    mosaic::InputFile in;
    in.read_at_fn = [&buf](uint64_t offset, uint8_t* dst, size_t len) -> int {
        if (offset + len > buf.data.size()) return -1;
        memcpy(dst, buf.data.data() + offset, len);
        return 0;
    };
    in.file_length = buf.data.size();
    return in;
}

static std::vector<uint8_t> write_and_get(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::shared_ptr<arrow::RecordBatch>& batch,
    mosaic::WriterOptions opts = {})
{
    MemBuffer buf;

    struct ArrowSchema c_schema;
    auto st = arrow::ExportSchema(*schema, &c_schema);
    assert(st.ok());

    mosaic::Writer writer(make_output(buf), &c_schema, opts);

    struct ArrowArray c_array;
    struct ArrowSchema c_batch_schema;
    st = arrow::ExportRecordBatch(*batch, &c_array, &c_batch_schema);
    assert(st.ok());

    writer.write(&c_array, &c_batch_schema);
    writer.close();
    return buf.data;
}

static std::shared_ptr<arrow::RecordBatch> read_row_group(
    mosaic::Reader& reader, uint32_t rg,
    const uint32_t* cols = nullptr, uint32_t num_cols = 0)
{
    struct ArrowArray c_array;
    struct ArrowSchema c_schema;

    if (cols && num_cols > 0) {
        reader.read_row_group(rg, cols, num_cols, &c_array, &c_schema);
    } else {
        reader.read_row_group(rg, &c_array, &c_schema);
    }

    auto result = arrow::ImportRecordBatch(&c_array, &c_schema);
    assert(result.ok());
    return result.ValueUnsafe();
}

// ======================== Tests ========================

static void test_basic_roundtrip() {
    auto schema = arrow::schema({
        arrow::field("id", arrow::int32(), false),
        arrow::field("name", arrow::utf8()),
        arrow::field("score", arrow::float64()),
    });

    arrow::Int32Builder id_b;
    arrow::StringBuilder name_b;
    arrow::DoubleBuilder score_b;
    for (int i = 0; i < 50; i++) {
        assert(id_b.Append(i).ok());
        assert(name_b.Append("user_" + std::to_string(i)).ok());
        assert(score_b.Append(i * 1.5).ok());
    }
    auto batch = arrow::RecordBatch::Make(schema, 50, {
        id_b.Finish().ValueUnsafe(),
        name_b.Finish().ValueUnsafe(),
        score_b.Finish().ValueUnsafe(),
    });

    mosaic::WriterOptions opts;
    opts.num_buckets = 2;
    auto data_vec = write_and_get(schema, batch, opts);
    ASSERT_TRUE(data_vec.size() > 32);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());
    ASSERT_TRUE(reader.num_row_groups() >= 1);

    auto rb = read_row_group(reader, 0);
    ASSERT_EQ(rb->num_rows(), 50);
    ASSERT_EQ(rb->num_columns(), 3);

    auto ids = std::static_pointer_cast<arrow::Int32Array>(rb->column(0));
    auto names = std::static_pointer_cast<arrow::StringArray>(rb->column(1));
    auto scores = std::static_pointer_cast<arrow::DoubleArray>(rb->column(2));

    for (int i = 0; i < 50; i++) {
        ASSERT_EQ(ids->Value(i), i);
        ASSERT_EQ(names->GetString(i), "user_" + std::to_string(i));
        ASSERT_TRUE(std::abs(scores->Value(i) - i * 1.5) < 1e-9);
    }
    printf("  PASS test_basic_roundtrip\n");
}

static void test_null_values() {
    auto schema = arrow::schema({
        arrow::field("id", arrow::int32()),
        arrow::field("name", arrow::utf8()),
    });

    arrow::Int32Builder id_b;
    assert(id_b.Append(1).ok());
    assert(id_b.Append(2).ok());
    assert(id_b.Append(3).ok());

    arrow::StringBuilder name_b;
    assert(name_b.Append("hello").ok());
    assert(name_b.AppendNull().ok());
    assert(name_b.Append("world").ok());

    auto batch = arrow::RecordBatch::Make(schema, 3, {
        id_b.Finish().ValueUnsafe(), name_b.Finish().ValueUnsafe()});
    auto data_vec = write_and_get(schema, batch);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());
    auto rb = read_row_group(reader, 0);
    ASSERT_EQ(rb->num_rows(), 3);

    auto names = std::static_pointer_cast<arrow::StringArray>(rb->column(1));
    ASSERT_TRUE(!names->IsNull(0));
    ASSERT_EQ(names->GetString(0), "hello");
    ASSERT_TRUE(names->IsNull(1));
    ASSERT_TRUE(!names->IsNull(2));
    ASSERT_EQ(names->GetString(2), "world");
    printf("  PASS test_null_values\n");
}

static void test_all_types() {
    auto schema = arrow::schema({
        arrow::field("f_bool", arrow::boolean()),
        arrow::field("f_int8", arrow::int8()),
        arrow::field("f_int16", arrow::int16()),
        arrow::field("f_int32", arrow::int32()),
        arrow::field("f_int64", arrow::int64()),
        arrow::field("f_float32", arrow::float32()),
        arrow::field("f_float64", arrow::float64()),
        arrow::field("f_utf8", arrow::utf8()),
        arrow::field("f_binary", arrow::binary()),
    });

    arrow::BooleanBuilder bool_b;
    assert(bool_b.Append(true).ok());
    arrow::Int8Builder i8_b;
    assert(i8_b.Append(42).ok());
    arrow::Int16Builder i16_b;
    assert(i16_b.Append(1234).ok());
    arrow::Int32Builder i32_b;
    assert(i32_b.Append(100000).ok());
    arrow::Int64Builder i64_b;
    assert(i64_b.Append(9999999999LL).ok());
    arrow::FloatBuilder f32_b;
    assert(f32_b.Append(3.14f).ok());
    arrow::DoubleBuilder f64_b;
    assert(f64_b.Append(2.718281828).ok());
    arrow::StringBuilder utf8_b;
    assert(utf8_b.Append("hello").ok());
    arrow::BinaryBuilder bin_b;
    uint8_t bin_data[] = {0x01, 0x02};
    assert(bin_b.Append(bin_data, 2).ok());

    auto batch = arrow::RecordBatch::Make(schema, 1, {
        bool_b.Finish().ValueUnsafe(),
        i8_b.Finish().ValueUnsafe(),
        i16_b.Finish().ValueUnsafe(),
        i32_b.Finish().ValueUnsafe(),
        i64_b.Finish().ValueUnsafe(),
        f32_b.Finish().ValueUnsafe(),
        f64_b.Finish().ValueUnsafe(),
        utf8_b.Finish().ValueUnsafe(),
        bin_b.Finish().ValueUnsafe(),
    });

    auto data_vec = write_and_get(schema, batch);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());
    auto rb = read_row_group(reader, 0);
    ASSERT_EQ(rb->num_rows(), 1);
    ASSERT_EQ(rb->num_columns(), 9);

    ASSERT_TRUE(std::static_pointer_cast<arrow::BooleanArray>(rb->column(0))->Value(0));
    ASSERT_EQ(std::static_pointer_cast<arrow::Int8Array>(rb->column(1))->Value(0), 42);
    ASSERT_EQ(std::static_pointer_cast<arrow::Int16Array>(rb->column(2))->Value(0), 1234);
    ASSERT_EQ(std::static_pointer_cast<arrow::Int32Array>(rb->column(3))->Value(0), 100000);
    ASSERT_EQ(std::static_pointer_cast<arrow::Int64Array>(rb->column(4))->Value(0), 9999999999LL);
    ASSERT_TRUE(std::abs(std::static_pointer_cast<arrow::FloatArray>(rb->column(5))->Value(0) - 3.14f) < 1e-5f);
    ASSERT_TRUE(std::abs(std::static_pointer_cast<arrow::DoubleArray>(rb->column(6))->Value(0) - 2.718281828) < 1e-9);
    ASSERT_EQ(std::static_pointer_cast<arrow::StringArray>(rb->column(7))->GetString(0), "hello");
    printf("  PASS test_all_types\n");
}

static void test_projection() {
    auto schema = arrow::schema({
        arrow::field("a", arrow::int32()),
        arrow::field("b", arrow::utf8()),
        arrow::field("c", arrow::float64()),
        arrow::field("d", arrow::utf8()),
    });

    arrow::Int32Builder ab;
    arrow::StringBuilder bb, db;
    arrow::DoubleBuilder cb;
    for (int i = 0; i < 20; i++) {
        assert(ab.Append(i).ok());
        assert(bb.Append("val_" + std::to_string(i)).ok());
        assert(cb.Append(static_cast<double>(i)).ok());
        assert(db.Append("extra_" + std::to_string(i)).ok());
    }
    auto batch = arrow::RecordBatch::Make(schema, 20, {
        ab.Finish().ValueUnsafe(), bb.Finish().ValueUnsafe(),
        cb.Finish().ValueUnsafe(), db.Finish().ValueUnsafe(),
    });

    mosaic::WriterOptions opts;
    opts.num_buckets = 2;
    auto data_vec = write_and_get(schema, batch, opts);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());

    uint32_t cols[] = {0, 1};
    auto rb = read_row_group(reader, 0, cols, 2);
    ASSERT_EQ(rb->num_columns(), 2);
    ASSERT_EQ(rb->num_rows(), 20);
    printf("  PASS test_projection\n");
}

static void test_statistics() {
    auto schema = arrow::schema({
        arrow::field("id", arrow::int32()),
        arrow::field("name", arrow::utf8()),
        arrow::field("score", arrow::float64()),
    });

    arrow::Int32Builder id_b;
    arrow::StringBuilder name_b;
    arrow::DoubleBuilder score_b;
    for (int i = 0; i < 10; i++) {
        assert(id_b.Append(i * 10).ok());
        assert(name_b.Append("item_" + std::to_string(i)).ok());
        assert(score_b.Append(i * 1.1).ok());
    }
    auto batch = arrow::RecordBatch::Make(schema, 10, {
        id_b.Finish().ValueUnsafe(), name_b.Finish().ValueUnsafe(),
        score_b.Finish().ValueUnsafe(),
    });

    mosaic::WriterOptions opts;
    uint32_t stats_cols[] = {0, 2};
    opts.stats_columns = stats_cols;
    opts.num_stats_columns = 2;
    auto data_vec = write_and_get(schema, batch, opts);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());

    auto stats = reader.get_row_group_statistics(0);
    ASSERT_TRUE(stats.size() > 0);

    for (auto& s : stats) {
        ASSERT_TRUE(s.column_index == 0 || s.column_index == 2);
        ASSERT_EQ(s.null_count, 0u);
        ASSERT_TRUE(s.has_min_max());
    }
    printf("  PASS test_statistics\n");
}

static void test_compression_zstd() {
    auto schema = arrow::schema({
        arrow::field("x", arrow::int32()),
        arrow::field("y", arrow::utf8()),
    });

    arrow::Int32Builder xb;
    arrow::StringBuilder yb;
    for (int i = 0; i < 100; i++) {
        assert(xb.Append(i).ok());
        assert(yb.Append("v_" + std::to_string(i)).ok());
    }
    auto batch = arrow::RecordBatch::Make(schema, 100, {
        xb.Finish().ValueUnsafe(), yb.Finish().ValueUnsafe(),
    });

    mosaic::WriterOptions opts;
    opts.compression = 1;
    opts.zstd_level = 3;
    auto data_vec = write_and_get(schema, batch, opts);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());
    auto rb = read_row_group(reader, 0);
    ASSERT_EQ(rb->num_rows(), 100);

    auto xs = std::static_pointer_cast<arrow::Int32Array>(rb->column(0));
    for (int i = 0; i < 100; i++) {
        ASSERT_EQ(xs->Value(i), i);
    }
    printf("  PASS test_compression_zstd\n");
}

static void test_schema_roundtrip() {
    auto schema = arrow::schema({
        arrow::field("id", arrow::int32(), false),
        arrow::field("name", arrow::utf8(), true),
        arrow::field("score", arrow::float64(), true),
    });

    arrow::Int32Builder sr_id_b;
    assert(sr_id_b.Append(1).ok());
    arrow::StringBuilder sr_name_b;
    assert(sr_name_b.Append("x").ok());
    arrow::DoubleBuilder sr_score_b;
    assert(sr_score_b.Append(1.0).ok());

    auto batch = arrow::RecordBatch::Make(schema, 1, {
        sr_id_b.Finish().ValueUnsafe(),
        sr_name_b.Finish().ValueUnsafe(),
        sr_score_b.Finish().ValueUnsafe(),
    });

    auto data_vec = write_and_get(schema, batch);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());

    struct ArrowSchema c_schema;
    reader.export_schema(&c_schema);
    auto imported = arrow::ImportSchema(&c_schema);
    assert(imported.ok());
    auto read_schema = imported.ValueUnsafe();

    ASSERT_EQ(read_schema->num_fields(), 3);
    ASSERT_EQ(read_schema->field(0)->name(), "id");
    ASSERT_EQ(read_schema->field(1)->name(), "name");
    ASSERT_EQ(read_schema->field(2)->name(), "score");
    ASSERT_TRUE(!read_schema->field(0)->nullable());
    ASSERT_TRUE(read_schema->field(1)->nullable());
    printf("  PASS test_schema_roundtrip\n");
}

static void test_multiple_row_groups() {
    auto schema = arrow::schema({
        arrow::field("id", arrow::int32()),
        arrow::field("data", arrow::int64()),
    });

    mosaic::WriterOptions opts;
    opts.compression = 0;
    opts.num_buckets = 1;
    opts.row_group_max_size = 200;

    MemBuffer write_buf;
    struct ArrowSchema c_schema;
    auto st = arrow::ExportSchema(*schema, &c_schema);
    assert(st.ok());
    mosaic::Writer writer(make_output(write_buf), &c_schema, opts);

    const int total_rows = 500;
    const int batch_size = 50;
    for (int start = 0; start < total_rows; start += batch_size) {
        int end = std::min(start + batch_size, total_rows);
        int n = end - start;
        arrow::Int32Builder id_b;
        arrow::Int64Builder data_b;
        for (int i = start; i < end; i++) {
            assert(id_b.Append(i).ok());
            assert(data_b.Append(static_cast<int64_t>(i) * 3).ok());
        }
        auto batch = arrow::RecordBatch::Make(schema, n, {
            id_b.Finish().ValueUnsafe(), data_b.Finish().ValueUnsafe(),
        });
        struct ArrowArray c_array;
        struct ArrowSchema c_batch_schema;
        st = arrow::ExportRecordBatch(*batch, &c_array, &c_batch_schema);
        assert(st.ok());
        writer.write(&c_array, &c_batch_schema);
    }
    writer.close();
    auto data_vec = write_buf.data;

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());
    ASSERT_TRUE(reader.num_row_groups() > 1);

    int offset = 0;
    for (uint32_t rg = 0; rg < reader.num_row_groups(); rg++) {
        auto rb = read_row_group(reader, rg);
        auto ids = std::static_pointer_cast<arrow::Int32Array>(rb->column(0));
        auto datas = std::static_pointer_cast<arrow::Int64Array>(rb->column(1));
        for (int64_t i = 0; i < rb->num_rows(); i++) {
            ASSERT_EQ(ids->Value(i), offset + static_cast<int>(i));
            ASSERT_EQ(datas->Value(i), static_cast<int64_t>(offset + i) * 3);
        }
        offset += static_cast<int>(rb->num_rows());
    }
    ASSERT_EQ(offset, 500);
    printf("  PASS test_multiple_row_groups\n");
}

int main() {
    printf("Running Mosaic C++ tests...\n");
    test_basic_roundtrip();
    test_null_values();
    test_all_types();
    test_projection();
    test_statistics();
    test_compression_zstd();
    test_schema_roundtrip();
    test_multiple_row_groups();
    printf("All %d tests passed.\n", 8);
    return 0;
}
