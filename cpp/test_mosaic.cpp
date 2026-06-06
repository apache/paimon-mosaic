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
    mosaic::Reader& reader, uint32_t rg)
{
    struct ArrowArray c_array;
    struct ArrowSchema c_schema;
    reader.read_row_group(rg, &c_array, &c_schema);
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

    auto ids = std::static_pointer_cast<arrow::Int32Array>(rb->GetColumnByName("id"));
    auto names = std::static_pointer_cast<arrow::StringArray>(rb->GetColumnByName("name"));
    auto scores = std::static_pointer_cast<arrow::DoubleArray>(rb->GetColumnByName("score"));

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

    auto names = std::static_pointer_cast<arrow::StringArray>(rb->GetColumnByName("name"));
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

    ASSERT_TRUE(std::static_pointer_cast<arrow::BooleanArray>(rb->GetColumnByName("f_bool"))->Value(0));
    ASSERT_EQ(std::static_pointer_cast<arrow::Int8Array>(rb->GetColumnByName("f_int8"))->Value(0), 42);
    ASSERT_EQ(std::static_pointer_cast<arrow::Int16Array>(rb->GetColumnByName("f_int16"))->Value(0), 1234);
    ASSERT_EQ(std::static_pointer_cast<arrow::Int32Array>(rb->GetColumnByName("f_int32"))->Value(0), 100000);
    ASSERT_EQ(std::static_pointer_cast<arrow::Int64Array>(rb->GetColumnByName("f_int64"))->Value(0), 9999999999LL);
    ASSERT_TRUE(std::abs(std::static_pointer_cast<arrow::FloatArray>(rb->GetColumnByName("f_float32"))->Value(0) - 3.14f) < 1e-5f);
    ASSERT_TRUE(std::abs(std::static_pointer_cast<arrow::DoubleArray>(rb->GetColumnByName("f_float64"))->Value(0) - 2.718281828) < 1e-9);
    ASSERT_EQ(std::static_pointer_cast<arrow::StringArray>(rb->GetColumnByName("f_utf8"))->GetString(0), "hello");
    printf("  PASS test_all_types\n");
}

static void test_timestamp_ns_roundtrip() {
    auto ts_ns_type = arrow::timestamp(arrow::TimeUnit::NANO);
    auto ts_ns_tz_type = arrow::timestamp(arrow::TimeUnit::NANO, "Asia/Shanghai");
    auto schema = arrow::schema({
        arrow::field("ts_ns", ts_ns_type),
        arrow::field("ts_ns_tz", ts_ns_tz_type),
    });

    const int64_t values[] = {1700000000000000123LL, -1LL};

    arrow::TimestampBuilder ts_ns_b(ts_ns_type, arrow::default_memory_pool());
    assert(ts_ns_b.Append(values[0]).ok());
    assert(ts_ns_b.AppendNull().ok());
    assert(ts_ns_b.Append(values[1]).ok());

    arrow::TimestampBuilder ts_ns_tz_b(ts_ns_tz_type, arrow::default_memory_pool());
    assert(ts_ns_tz_b.Append(values[0]).ok());
    assert(ts_ns_tz_b.AppendNull().ok());
    assert(ts_ns_tz_b.Append(values[1]).ok());

    auto batch = arrow::RecordBatch::Make(schema, 3, {
        ts_ns_b.Finish().ValueUnsafe(),
        ts_ns_tz_b.Finish().ValueUnsafe(),
    });

    auto data_vec = write_and_get(schema, batch);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());
    auto rb = read_row_group(reader, 0);

    ASSERT_TRUE(rb->schema()->field(0)->type()->Equals(ts_ns_type));
    ASSERT_TRUE(rb->schema()->field(1)->type()->Equals(ts_ns_tz_type));

    auto ts_ns = std::static_pointer_cast<arrow::TimestampArray>(rb->column(0));
    auto ts_ns_tz = std::static_pointer_cast<arrow::TimestampArray>(rb->column(1));
    ASSERT_EQ(ts_ns->Value(0), values[0]);
    ASSERT_TRUE(ts_ns->IsNull(1));
    ASSERT_EQ(ts_ns->Value(2), values[1]);
    ASSERT_EQ(ts_ns_tz->Value(0), values[0]);
    ASSERT_TRUE(ts_ns_tz->IsNull(1));
    ASSERT_EQ(ts_ns_tz->Value(2), values[1]);
    printf("  PASS test_timestamp_ns_roundtrip\n");
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

    const char* cols[] = {"c", "a", "b"};
    reader.set_projection(cols, 3);
    auto rb = read_row_group(reader, 0);
    ASSERT_EQ(rb->num_columns(), 3);
    ASSERT_EQ(rb->num_rows(), 20);
    ASSERT_EQ(rb->schema()->field(0)->name(), "c");
    ASSERT_EQ(rb->schema()->field(1)->name(), "a");
    ASSERT_EQ(rb->schema()->field(2)->name(), "b");
    printf("  PASS test_projection\n");
}

static void test_projection_empty() {
    auto schema = arrow::schema({
        arrow::field("a", arrow::int32()),
        arrow::field("b", arrow::utf8()),
    });

    arrow::Int32Builder ab;
    arrow::StringBuilder bb;
    for (int i = 0; i < 5; i++) {
        assert(ab.Append(i).ok());
        assert(bb.Append("v" + std::to_string(i)).ok());
    }
    auto batch = arrow::RecordBatch::Make(schema, 5, {
        ab.Finish().ValueUnsafe(), bb.Finish().ValueUnsafe(),
    });

    auto data_vec = write_and_get(schema, batch);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());

    reader.set_projection(nullptr, 0);
    auto rb = read_row_group(reader, 0);
    ASSERT_EQ(rb->num_columns(), 0);
    ASSERT_EQ(rb->num_rows(), 5);
    printf("  PASS test_projection_empty\n");
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
    const char* stats_cols[] = {"id", "score"};
    opts.stats_columns = stats_cols;
    opts.num_stats_columns = 2;
    auto data_vec = write_and_get(schema, batch, opts);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());

    auto stats = reader.get_row_group_statistics(0);
    ASSERT_TRUE(stats.size() > 0);

    for (auto& s : stats) {
        ASSERT_TRUE(s.column_name == "id" || s.column_name == "score");
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

    auto xs = std::static_pointer_cast<arrow::Int32Array>(rb->GetColumnByName("x"));
    for (int i = 0; i < 100; i++) {
        ASSERT_EQ(xs->Value(i), i);
    }
    printf("  PASS test_compression_zstd\n");
}

static void test_schema_roundtrip() {
    auto schema = arrow::schema({
        arrow::field("name", arrow::utf8(), true),
        arrow::field("id", arrow::int32(), false),
        arrow::field("score", arrow::float64(), true),
    });

    arrow::StringBuilder sr_name_b;
    assert(sr_name_b.Append("x").ok());
    arrow::Int32Builder sr_id_b;
    assert(sr_id_b.Append(1).ok());
    arrow::DoubleBuilder sr_score_b;
    assert(sr_score_b.Append(1.0).ok());

    auto batch = arrow::RecordBatch::Make(schema, 1, {
        sr_name_b.Finish().ValueUnsafe(),
        sr_id_b.Finish().ValueUnsafe(),
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
    ASSERT_EQ(read_schema->field(0)->name(), "name");
    ASSERT_EQ(read_schema->field(1)->name(), "id");
    ASSERT_EQ(read_schema->field(2)->name(), "score");
    ASSERT_TRUE(read_schema->field(0)->nullable());
    ASSERT_TRUE(!read_schema->field(1)->nullable());
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
        auto ids = std::static_pointer_cast<arrow::Int32Array>(rb->GetColumnByName("id"));
        auto datas = std::static_pointer_cast<arrow::Int64Array>(rb->GetColumnByName("data"));
        for (int64_t i = 0; i < rb->num_rows(); i++) {
            ASSERT_EQ(ids->Value(i), offset + static_cast<int>(i));
            ASSERT_EQ(datas->Value(i), static_cast<int64_t>(offset + i) * 3);
        }
        offset += static_cast<int>(rb->num_rows());
    }
    ASSERT_EQ(offset, 500);
    printf("  PASS test_multiple_row_groups\n");
}

static void test_writer_stats() {
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
    const char* stats_cols[] = {"id", "score"};
    opts.stats_columns = stats_cols;
    opts.num_stats_columns = 2;

    MemBuffer write_buf;
    struct ArrowSchema c_schema;
    auto st = arrow::ExportSchema(*schema, &c_schema);
    assert(st.ok());
    mosaic::Writer writer(make_output(write_buf), &c_schema, opts);

    struct ArrowArray c_array;
    struct ArrowSchema c_batch_schema;
    st = arrow::ExportRecordBatch(*batch, &c_array, &c_batch_schema);
    assert(st.ok());
    writer.write(&c_array, &c_batch_schema);
    writer.close();

    ASSERT_EQ(writer.num_row_groups(), 1u);
    auto stats = writer.get_row_group_statistics(0);
    ASSERT_TRUE(stats.size() > 0);

    for (auto& s : stats) {
        ASSERT_TRUE(s.column_name == "id" || s.column_name == "score");
        ASSERT_EQ(s.null_count, 0u);
        ASSERT_TRUE(s.has_min_max());
    }

    auto id_stat = std::find_if(stats.begin(), stats.end(),
        [](const mosaic::ColumnStatistics& s) { return s.column_name == "id"; });
    ASSERT_TRUE(id_stat != stats.end());
    ASSERT_EQ(id_stat->min_value.size(), 4u);
    ASSERT_EQ(id_stat->max_value.size(), 4u);
    int32_t min_id = 0, max_id = 0;
    memcpy(&min_id, id_stat->min_value.data(), 4);
    memcpy(&max_id, id_stat->max_value.data(), 4);
    min_id = __builtin_bswap32(min_id);
    max_id = __builtin_bswap32(max_id);
    ASSERT_EQ(min_id, 0);
    ASSERT_EQ(max_id, 90);
    printf("  PASS test_writer_stats\n");
}

static void test_writer_stats_with_nulls() {
    auto schema = arrow::schema({
        arrow::field("a", arrow::int32()),
        arrow::field("b", arrow::int64()),
    });

    arrow::Int32Builder a_b;
    assert(a_b.Append(10).ok());
    assert(a_b.AppendNull().ok());
    assert(a_b.Append(5).ok());
    assert(a_b.Append(20).ok());

    arrow::Int64Builder b_b;
    assert(b_b.AppendNull().ok());
    assert(b_b.AppendNull().ok());
    assert(b_b.Append(100).ok());
    assert(b_b.Append(50).ok());

    auto batch = arrow::RecordBatch::Make(schema, 4, {
        a_b.Finish().ValueUnsafe(), b_b.Finish().ValueUnsafe(),
    });

    mosaic::WriterOptions opts;
    opts.num_buckets = 1;
    const char* stats_cols[] = {"a", "b"};
    opts.stats_columns = stats_cols;
    opts.num_stats_columns = 2;

    MemBuffer write_buf;
    struct ArrowSchema c_schema;
    auto st = arrow::ExportSchema(*schema, &c_schema);
    assert(st.ok());
    mosaic::Writer writer(make_output(write_buf), &c_schema, opts);

    struct ArrowArray c_array;
    struct ArrowSchema c_batch_schema;
    st = arrow::ExportRecordBatch(*batch, &c_array, &c_batch_schema);
    assert(st.ok());
    writer.write(&c_array, &c_batch_schema);
    writer.close();

    ASSERT_EQ(writer.num_row_groups(), 1u);
    auto stats = writer.get_row_group_statistics(0);
    ASSERT_EQ(stats.size(), 2u);

    auto a_stat = std::find_if(stats.begin(), stats.end(),
        [](const mosaic::ColumnStatistics& s) { return s.column_name == "a"; });
    ASSERT_TRUE(a_stat != stats.end());
    ASSERT_EQ(a_stat->null_count, 1u);
    ASSERT_TRUE(a_stat->has_min_max());
    int32_t min_a = 0, max_a = 0;
    memcpy(&min_a, a_stat->min_value.data(), 4);
    memcpy(&max_a, a_stat->max_value.data(), 4);
    min_a = __builtin_bswap32(min_a);
    max_a = __builtin_bswap32(max_a);
    ASSERT_EQ(min_a, 5);
    ASSERT_EQ(max_a, 20);

    auto b_stat = std::find_if(stats.begin(), stats.end(),
        [](const mosaic::ColumnStatistics& s) { return s.column_name == "b"; });
    ASSERT_TRUE(b_stat != stats.end());
    ASSERT_EQ(b_stat->null_count, 2u);
    ASSERT_TRUE(b_stat->has_min_max());
    int64_t min_b = 0, max_b = 0;
    memcpy(&min_b, b_stat->min_value.data(), 8);
    memcpy(&max_b, b_stat->max_value.data(), 8);
    min_b = __builtin_bswap64(min_b);
    max_b = __builtin_bswap64(max_b);
    ASSERT_EQ(min_b, 50);
    ASSERT_EQ(max_b, 100);
    printf("  PASS test_writer_stats_with_nulls\n");
}

static void test_writer_stats_all_null() {
    auto schema = arrow::schema({
        arrow::field("x", arrow::int32()),
    });

    arrow::Int32Builder x_b;
    assert(x_b.AppendNull().ok());
    assert(x_b.AppendNull().ok());
    assert(x_b.AppendNull().ok());

    auto batch = arrow::RecordBatch::Make(schema, 3, {
        x_b.Finish().ValueUnsafe(),
    });

    mosaic::WriterOptions opts;
    opts.num_buckets = 1;
    const char* stats_cols[] = {"x"};
    opts.stats_columns = stats_cols;
    opts.num_stats_columns = 1;

    MemBuffer write_buf;
    struct ArrowSchema c_schema;
    auto st = arrow::ExportSchema(*schema, &c_schema);
    assert(st.ok());
    mosaic::Writer writer(make_output(write_buf), &c_schema, opts);

    struct ArrowArray c_array;
    struct ArrowSchema c_batch_schema;
    st = arrow::ExportRecordBatch(*batch, &c_array, &c_batch_schema);
    assert(st.ok());
    writer.write(&c_array, &c_batch_schema);
    writer.close();

    ASSERT_EQ(writer.num_row_groups(), 1u);
    auto stats = writer.get_row_group_statistics(0);
    ASSERT_EQ(stats.size(), 1u);
    ASSERT_EQ(stats[0].column_name, "x");
    ASSERT_EQ(stats[0].null_count, 3u);
    ASSERT_TRUE(!stats[0].has_min_max());
    printf("  PASS test_writer_stats_all_null\n");
}

static void test_writer_stats_matches_reader() {
    auto schema = arrow::schema({
        arrow::field("id", arrow::int32()),
        arrow::field("value", arrow::float64()),
    });

    arrow::Int32Builder id_b;
    arrow::DoubleBuilder val_b;
    for (int i = 0; i < 20; i++) {
        assert(id_b.Append(i * 5).ok());
        assert(val_b.Append(i * 2.5).ok());
    }
    auto batch = arrow::RecordBatch::Make(schema, 20, {
        id_b.Finish().ValueUnsafe(), val_b.Finish().ValueUnsafe(),
    });

    mosaic::WriterOptions opts;
    opts.num_buckets = 1;
    const char* stats_cols[] = {"id", "value"};
    opts.stats_columns = stats_cols;
    opts.num_stats_columns = 2;

    MemBuffer write_buf;
    struct ArrowSchema c_schema;
    auto st = arrow::ExportSchema(*schema, &c_schema);
    assert(st.ok());
    mosaic::Writer writer(make_output(write_buf), &c_schema, opts);

    struct ArrowArray c_array;
    struct ArrowSchema c_batch_schema;
    st = arrow::ExportRecordBatch(*batch, &c_array, &c_batch_schema);
    assert(st.ok());
    writer.write(&c_array, &c_batch_schema);
    writer.close();

    auto writer_stats = writer.get_row_group_statistics(0);

    MemBuffer buf;
    buf.data = write_buf.data;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());
    auto reader_stats = reader.get_row_group_statistics(0);

    ASSERT_EQ(writer_stats.size(), reader_stats.size());
    for (size_t i = 0; i < writer_stats.size(); i++) {
        ASSERT_EQ(writer_stats[i].column_name, reader_stats[i].column_name);
        ASSERT_EQ(writer_stats[i].null_count, reader_stats[i].null_count);
        ASSERT_EQ(writer_stats[i].min_value, reader_stats[i].min_value);
        ASSERT_EQ(writer_stats[i].max_value, reader_stats[i].max_value);
    }
    printf("  PASS test_writer_stats_matches_reader\n");
}

static void test_stats_empty_string_min() {
    auto schema = arrow::schema({
        arrow::field("s", arrow::utf8()),
    });

    arrow::StringBuilder s_b;
    assert(s_b.Append("").ok());
    assert(s_b.Append("b").ok());

    auto batch = arrow::RecordBatch::Make(schema, 2, {
        s_b.Finish().ValueUnsafe(),
    });

    mosaic::WriterOptions opts;
    opts.num_buckets = 1;
    const char* stats_cols[] = {"s"};
    opts.stats_columns = stats_cols;
    opts.num_stats_columns = 1;

    MemBuffer write_buf;
    struct ArrowSchema c_schema;
    auto st = arrow::ExportSchema(*schema, &c_schema);
    assert(st.ok());
    mosaic::Writer writer(make_output(write_buf), &c_schema, opts);

    struct ArrowArray c_array;
    struct ArrowSchema c_batch_schema;
    st = arrow::ExportRecordBatch(*batch, &c_array, &c_batch_schema);
    assert(st.ok());
    writer.write(&c_array, &c_batch_schema);
    writer.close();

    // Writer stats
    ASSERT_EQ(writer.num_row_groups(), 1u);
    auto writer_stats = writer.get_row_group_statistics(0);
    ASSERT_EQ(writer_stats.size(), 1u);
    ASSERT_EQ(writer_stats[0].column_name, "s");
    ASSERT_TRUE(writer_stats[0].has_min_max());
    ASSERT_EQ(writer_stats[0].min_value.size(), 0u);
    ASSERT_EQ(writer_stats[0].max_value, (std::vector<uint8_t>{'b'}));
    ASSERT_EQ(writer_stats[0].null_count, 0u);

    // Reader stats
    MemBuffer buf;
    buf.data = write_buf.data;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());
    auto reader_stats = reader.get_row_group_statistics(0);
    ASSERT_EQ(reader_stats.size(), 1u);
    ASSERT_EQ(reader_stats[0].column_name, "s");
    ASSERT_TRUE(reader_stats[0].has_min_max());
    ASSERT_EQ(reader_stats[0].min_value.size(), 0u);
    ASSERT_EQ(reader_stats[0].max_value, (std::vector<uint8_t>{'b'}));
    ASSERT_EQ(reader_stats[0].null_count, 0u);
    printf("  PASS test_stats_empty_string_min\n");
}

static void test_array_type() {
    auto schema = arrow::schema({
        arrow::field("id", arrow::int32(), false),
        arrow::field("tags", arrow::list(arrow::field("item", arrow::int32()))),
    });

    arrow::Int32Builder id_b;
    assert(id_b.Append(1).ok());
    assert(id_b.Append(2).ok());
    assert(id_b.Append(3).ok());
    assert(id_b.Append(4).ok());

    auto val_builder = std::make_shared<arrow::Int32Builder>();
    arrow::ListBuilder list_b(arrow::default_memory_pool(), val_builder);

    // Row 0: [10, 20, 30]
    assert(list_b.Append().ok());
    assert(val_builder->Append(10).ok());
    assert(val_builder->Append(20).ok());
    assert(val_builder->Append(30).ok());

    // Row 1: [40, 50]
    assert(list_b.Append().ok());
    assert(val_builder->Append(40).ok());
    assert(val_builder->Append(50).ok());

    // Row 2: []
    assert(list_b.Append().ok());

    // Row 3: null
    assert(list_b.AppendNull().ok());

    auto batch = arrow::RecordBatch::Make(schema, 4, {
        id_b.Finish().ValueUnsafe(),
        list_b.Finish().ValueUnsafe(),
    });

    auto data_vec = write_and_get(schema, batch);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());
    auto rb = read_row_group(reader, 0);
    ASSERT_EQ(rb->num_rows(), 4);

    auto ids = std::static_pointer_cast<arrow::Int32Array>(rb->GetColumnByName("id"));
    ASSERT_EQ(ids->Value(0), 1);
    ASSERT_EQ(ids->Value(1), 2);
    ASSERT_EQ(ids->Value(2), 3);
    ASSERT_EQ(ids->Value(3), 4);

    auto tags = std::static_pointer_cast<arrow::ListArray>(rb->GetColumnByName("tags"));
    ASSERT_TRUE(!tags->IsNull(0));
    ASSERT_TRUE(!tags->IsNull(1));
    ASSERT_TRUE(!tags->IsNull(2));
    ASSERT_TRUE(tags->IsNull(3));

    // Row 0: [10, 20, 30]
    auto row0 = std::static_pointer_cast<arrow::Int32Array>(tags->value_slice(0));
    ASSERT_EQ(row0->length(), 3);
    ASSERT_EQ(row0->Value(0), 10);
    ASSERT_EQ(row0->Value(1), 20);
    ASSERT_EQ(row0->Value(2), 30);

    // Row 1: [40, 50]
    auto row1 = std::static_pointer_cast<arrow::Int32Array>(tags->value_slice(1));
    ASSERT_EQ(row1->length(), 2);
    ASSERT_EQ(row1->Value(0), 40);
    ASSERT_EQ(row1->Value(1), 50);

    // Row 2: []
    auto row2 = std::static_pointer_cast<arrow::Int32Array>(tags->value_slice(2));
    ASSERT_EQ(row2->length(), 0);

    printf("  PASS test_array_type\n");
}

static void test_array_with_null_elements() {
    auto schema = arrow::schema({
        arrow::field("arr", arrow::list(arrow::field("item", arrow::int64()))),
    });

    auto val_builder = std::make_shared<arrow::Int64Builder>();
    arrow::ListBuilder list_b(arrow::default_memory_pool(), val_builder);

    // [100, null, 300]
    assert(list_b.Append().ok());
    assert(val_builder->Append(100).ok());
    assert(val_builder->AppendNull().ok());
    assert(val_builder->Append(300).ok());

    // [null, null]
    assert(list_b.Append().ok());
    assert(val_builder->AppendNull().ok());
    assert(val_builder->AppendNull().ok());

    // [999]
    assert(list_b.Append().ok());
    assert(val_builder->Append(999).ok());

    auto batch = arrow::RecordBatch::Make(schema, 3, {
        list_b.Finish().ValueUnsafe(),
    });

    auto data_vec = write_and_get(schema, batch);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());
    auto rb = read_row_group(reader, 0);

    auto arr = std::static_pointer_cast<arrow::ListArray>(rb->column(0));

    auto row0 = std::static_pointer_cast<arrow::Int64Array>(arr->value_slice(0));
    ASSERT_EQ(row0->length(), 3);
    ASSERT_EQ(row0->Value(0), 100);
    ASSERT_TRUE(row0->IsNull(1));
    ASSERT_EQ(row0->Value(2), 300);

    auto row1 = std::static_pointer_cast<arrow::Int64Array>(arr->value_slice(1));
    ASSERT_EQ(row1->length(), 2);
    ASSERT_TRUE(row1->IsNull(0));
    ASSERT_TRUE(row1->IsNull(1));

    auto row2 = std::static_pointer_cast<arrow::Int64Array>(arr->value_slice(2));
    ASSERT_EQ(row2->length(), 1);
    ASSERT_EQ(row2->Value(0), 999);

    printf("  PASS test_array_with_null_elements\n");
}

static void test_array_string_elements() {
    auto schema = arrow::schema({
        arrow::field("arr", arrow::list(arrow::field("item", arrow::utf8()))),
    });

    auto val_builder = std::make_shared<arrow::StringBuilder>();
    arrow::ListBuilder list_b(arrow::default_memory_pool(), val_builder);

    // ["hello", "world"]
    assert(list_b.Append().ok());
    assert(val_builder->Append("hello").ok());
    assert(val_builder->Append("world").ok());

    // [null, "foo"]
    assert(list_b.Append().ok());
    assert(val_builder->AppendNull().ok());
    assert(val_builder->Append("foo").ok());

    // []
    assert(list_b.Append().ok());

    auto batch = arrow::RecordBatch::Make(schema, 3, {
        list_b.Finish().ValueUnsafe(),
    });

    auto data_vec = write_and_get(schema, batch);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());
    auto rb = read_row_group(reader, 0);

    auto arr = std::static_pointer_cast<arrow::ListArray>(rb->column(0));

    auto row0 = std::static_pointer_cast<arrow::StringArray>(arr->value_slice(0));
    ASSERT_EQ(row0->length(), 2);
    ASSERT_EQ(row0->GetString(0), "hello");
    ASSERT_EQ(row0->GetString(1), "world");

    auto row1 = std::static_pointer_cast<arrow::StringArray>(arr->value_slice(1));
    ASSERT_EQ(row1->length(), 2);
    ASSERT_TRUE(row1->IsNull(0));
    ASSERT_EQ(row1->GetString(1), "foo");

    auto row2 = std::static_pointer_cast<arrow::StringArray>(arr->value_slice(2));
    ASSERT_EQ(row2->length(), 0);

    printf("  PASS test_array_string_elements\n");
}

static void test_map_type() {
    auto schema = arrow::schema({
        arrow::field("id", arrow::int32(), false),
        arrow::field("props", arrow::map(arrow::int32(), arrow::utf8())),
    });

    arrow::Int32Builder id_b;
    assert(id_b.Append(1).ok());
    assert(id_b.Append(2).ok());
    assert(id_b.Append(3).ok());

    auto key_builder = std::make_shared<arrow::Int32Builder>();
    auto val_builder = std::make_shared<arrow::StringBuilder>();
    arrow::MapBuilder map_b(arrow::default_memory_pool(), key_builder, val_builder);

    // Row 0: {1: "a", 2: "b"}
    assert(map_b.Append().ok());
    assert(key_builder->Append(1).ok());
    assert(val_builder->Append("a").ok());
    assert(key_builder->Append(2).ok());
    assert(val_builder->Append("b").ok());

    // Row 1: null
    assert(map_b.AppendNull().ok());

    // Row 2: {} (empty)
    assert(map_b.Append().ok());

    auto batch = arrow::RecordBatch::Make(schema, 3, {
        id_b.Finish().ValueUnsafe(),
        map_b.Finish().ValueUnsafe(),
    });

    auto data_vec = write_and_get(schema, batch);

    MemBuffer buf;
    buf.data = data_vec;
    auto reader = mosaic::make_reader(make_input(buf), buf.data.size());
    auto rb = read_row_group(reader, 0);
    ASSERT_EQ(rb->num_rows(), 3);

    auto props = std::static_pointer_cast<arrow::MapArray>(rb->GetColumnByName("props"));
    ASSERT_TRUE(!props->IsNull(0));
    ASSERT_TRUE(props->IsNull(1));
    ASSERT_TRUE(!props->IsNull(2));

    // Row 0: 2 entries
    ASSERT_EQ(props->value_length(0), 2);
    // Row 2: empty
    ASSERT_EQ(props->value_length(2), 0);

    printf("  PASS test_map_type\n");
}

int main() {
    printf("Running Mosaic C++ tests...\n");
    test_basic_roundtrip();
    test_null_values();
    test_all_types();
    test_timestamp_ns_roundtrip();
    test_projection();
    test_projection_empty();
    test_statistics();
    test_compression_zstd();
    test_schema_roundtrip();
    test_multiple_row_groups();
    test_writer_stats();
    test_writer_stats_with_nulls();
    test_writer_stats_all_null();
    test_writer_stats_matches_reader();
    test_stats_empty_string_min();
    test_array_type();
    test_array_with_null_elements();
    test_array_string_elements();
    test_map_type();
    printf("All %d tests passed.\n", 19);
    return 0;
}
