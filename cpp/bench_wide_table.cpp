// Wide-table write benchmark: Parquet vs Mosaic, in-memory and on-disk.
//
// Uses Google Benchmark for proper P50/P95/P99 distribution measurement.
// Synthetic data matches /root/tmp/ characteristics:
//   ~60% null, 79% i8, 11% f64, 4.5% i16-wide, 3% i16, ~2% i32-wide.
// (uint8/uint16/uint32 mapped to same-byte signed types; mosaic has no unsigned.)
//
// Build:    cmake --build cpp/build --target bench_wide_table
// Run:      ./bench_wide_table --benchmark_repetitions=5

#include "mosaic.hpp"

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr int kDefaultNumRows = 1500;
constexpr int kZstdLevel = 3;
constexpr unsigned kSeed = 42;
constexpr const char* kDiskPath = "/tmp/bench_wide_table.out";

enum class ColType { Timestamp, Int8, Int16, Int32, Float64 };
enum class Sparsity { AllNull, AlwaysPresent, MostlyNull };

struct ColSpec {
    ColType type;
    Sparsity sparsity;
    std::string name;
};

std::vector<ColSpec> build_specs(int n_cols) {
    std::vector<ColSpec> out;
    out.reserve(n_cols);
    out.push_back({ColType::Timestamp, Sparsity::AlwaysPresent, "timestamp"});

    int data_cols = n_cols - 1;
    int n_i8 = data_cols * 794 / 1000;           // uint8 in real data
    int n_f64 = data_cols * 113 / 1000;
    int n_i16_wide = data_cols * 45 / 1000;      // uint16 in real data
    int n_i16 = data_cols * 31 / 1000;
    int n_i32_wide = data_cols - n_i8 - n_f64 - n_i16_wide - n_i16;  // uint32 in real

    auto add = [&](ColType t, int n) {
        for (int i = 0; i < n; i++) {
            int idx = static_cast<int>(out.size());
            char buf[80];
            std::snprintf(buf, sizeof(buf),
                "054.O1O4NO5.NN2g60ObDkkukO5Jz0g1uaaDU8DSCZ%c.col_%05d_pad",
                'A' + (idx % 26), idx);
            Sparsity s;
            int x = idx % 100;
            if (x < 18) s = Sparsity::AllNull;
            else if (x < 34) s = Sparsity::AlwaysPresent;
            else s = Sparsity::MostlyNull;
            out.push_back({t, s, std::string(buf)});
        }
    };
    add(ColType::Int8, n_i8);
    add(ColType::Float64, n_f64);
    add(ColType::Int16, n_i16_wide);
    add(ColType::Int16, n_i16);
    add(ColType::Int32, n_i32_wide);
    return out;
}

std::shared_ptr<arrow::DataType> arrow_type_for(ColType t) {
    switch (t) {
        case ColType::Timestamp: return arrow::int64();
        case ColType::Int8:      return arrow::int8();
        case ColType::Int16:     return arrow::int16();
        case ColType::Int32:     return arrow::int32();
        case ColType::Float64:   return arrow::float64();
    }
    return arrow::null();
}

std::shared_ptr<arrow::RecordBatch> build_batch(const std::vector<ColSpec>& specs,
                                                 int n_rows) {
    arrow::FieldVector fields;
    fields.reserve(specs.size());
    for (const auto& s : specs) {
        fields.push_back(arrow::field(
            s.name, arrow_type_for(s.type),
            s.sparsity != Sparsity::AlwaysPresent));
    }
    auto schema = arrow::schema(std::move(fields));

    std::mt19937 rng(kSeed);
    std::uniform_int_distribution<int> d_pres(0, 99);
    std::uniform_int_distribution<int> d_i8(-128, 127);
    std::uniform_int_distribution<int> d_i16(-32768, 32767);
    std::uniform_int_distribution<int32_t> d_i32(INT32_MIN, INT32_MAX);
    std::uniform_real_distribution<double> d_f64(-256.0, 255.9);

    auto present = [&](Sparsity s) {
        if (s == Sparsity::AllNull) return false;
        if (s == Sparsity::AlwaysPresent) return true;
        return d_pres(rng) < 25;
    };

    constexpr int64_t base_ts = 1778218500000LL;

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(specs.size());

    for (const auto& spec : specs) {
        std::shared_ptr<arrow::Array> arr;
        switch (spec.type) {
            case ColType::Timestamp: {
                arrow::Int64Builder b;
                (void)b.Reserve(n_rows);
                for (int r = 0; r < n_rows; r++)
                    (void)b.Append(base_ts + r * 200);
                (void)b.Finish(&arr);
                break;
            }
            case ColType::Int8: {
                arrow::Int8Builder b;
                (void)b.Reserve(n_rows);
                for (int r = 0; r < n_rows; r++) {
                    if (present(spec.sparsity)) (void)b.Append((int8_t)d_i8(rng));
                    else (void)b.AppendNull();
                }
                (void)b.Finish(&arr);
                break;
            }
            case ColType::Int16: {
                arrow::Int16Builder b;
                (void)b.Reserve(n_rows);
                for (int r = 0; r < n_rows; r++) {
                    if (present(spec.sparsity)) (void)b.Append((int16_t)d_i16(rng));
                    else (void)b.AppendNull();
                }
                (void)b.Finish(&arr);
                break;
            }
            case ColType::Int32: {
                arrow::Int32Builder b;
                (void)b.Reserve(n_rows);
                for (int r = 0; r < n_rows; r++) {
                    if (present(spec.sparsity)) (void)b.Append(d_i32(rng));
                    else (void)b.AppendNull();
                }
                (void)b.Finish(&arr);
                break;
            }
            case ColType::Float64: {
                arrow::DoubleBuilder b;
                (void)b.Reserve(n_rows);
                for (int r = 0; r < n_rows; r++) {
                    if (present(spec.sparsity)) (void)b.Append(d_f64(rng));
                    else (void)b.AppendNull();
                }
                (void)b.Finish(&arr);
                break;
            }
        }
        arrays.push_back(std::move(arr));
    }
    return arrow::RecordBatch::Make(schema, n_rows, std::move(arrays));
}

// ============== Batch cache (build once per column count) ==============

std::shared_ptr<arrow::RecordBatch> get_or_build_batch(int n_cols, int n_rows) {
    using Key = std::pair<int, int>;
    static std::mutex mu;
    static std::vector<std::pair<Key, std::shared_ptr<arrow::RecordBatch>>> cache;
    std::lock_guard<std::mutex> lock(mu);
    Key k{n_cols, n_rows};
    for (auto& [kk, v] : cache) if (kk == k) return v;
    auto specs = build_specs(n_cols);
    auto b = build_batch(specs, n_rows);
    cache.emplace_back(k, b);
    return b;
}

// ============== Writers ==============

// Write Parquet to in-memory buffer; return bytes written.
size_t write_parquet_to(std::shared_ptr<arrow::io::OutputStream> sink,
                        const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto table = arrow::Table::FromRecordBatches({batch}).ValueOrDie();
    int64_t rg_rows = batch->num_rows();
    auto props = parquet::WriterProperties::Builder()
        .compression(parquet::Compression::ZSTD)
        ->compression_level(kZstdLevel)
        ->max_row_group_length(rg_rows)
        ->build();
    auto arrow_props = parquet::ArrowWriterProperties::Builder().build();
    auto st = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                          sink, rg_rows, props, arrow_props);
    if (!st.ok()) throw std::runtime_error("parquet write: " + st.ToString());
    return 0;
}

size_t write_mosaic_to(const std::function<int(const uint8_t*, size_t)>& sink_write,
                       const std::function<int()>& sink_flush,
                       const std::function<int64_t()>& sink_pos,
                       const std::shared_ptr<arrow::RecordBatch>& batch) {
    mosaic::OutputFile of;
    of.write_fn = sink_write;
    of.flush_fn = sink_flush;
    of.get_pos_fn = sink_pos;

    struct ArrowSchema c_schema;
    auto st = arrow::ExportSchema(*batch->schema(), &c_schema);
    if (!st.ok()) throw std::runtime_error("ExportSchema: " + st.ToString());

    mosaic::WriterOptions opts;
    opts.compression = 1;
    opts.zstd_level = kZstdLevel;

    mosaic::Writer writer(of, &c_schema, opts);
    struct ArrowArray c_array;
    struct ArrowSchema c_batch_schema;
    st = arrow::ExportRecordBatch(*batch, &c_array, &c_batch_schema);
    if (!st.ok()) throw std::runtime_error("ExportRecordBatch: " + st.ToString());
    writer.write(&c_array, &c_batch_schema);
    writer.close();
    return static_cast<size_t>(sink_pos());
}

// ============== Helpers ==============

long peak_rss_mb() {
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    return u.ru_maxrss / 1024;
}

// fsync wrapper for disk writes (matters for car eMMC fairness).
void fsync_path(const char* path) {
    int fd = ::open(path, O_RDONLY);
    if (fd >= 0) { ::fsync(fd); ::close(fd); }
}

// ============== Benchmarks ==============

static void BM_Parquet_Mem(benchmark::State& state) {
    auto batch = get_or_build_batch(state.range(0), state.range(1));
    size_t bytes = 0;
    for (auto _ : state) {
        auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
        write_parquet_to(sink, batch);
        auto buf = sink->Finish().ValueOrDie();
        bytes = buf->size();
        benchmark::DoNotOptimize(buf);
    }
    state.counters["out_MB"] = bytes / 1048576.0;
    state.counters["peak_rss_MB"] = peak_rss_mb();
}

static void BM_Mosaic_Mem(benchmark::State& state) {
    auto batch = get_or_build_batch(state.range(0), state.range(1));
    size_t bytes = 0;
    for (auto _ : state) {
        std::vector<uint8_t> data;
        data.reserve(64 * 1024 * 1024);
        size_t pos = 0;
        bytes = write_mosaic_to(
            [&](const uint8_t* p, size_t n) { data.insert(data.end(), p, p + n); pos += n; return 0; },
            []() { return 0; },
            [&]() { return (int64_t)pos; },
            batch);
        benchmark::DoNotOptimize(data);
    }
    state.counters["out_MB"] = bytes / 1048576.0;
    state.counters["peak_rss_MB"] = peak_rss_mb();
}

static void BM_Parquet_Disk(benchmark::State& state) {
    auto batch = get_or_build_batch(state.range(0), state.range(1));
    size_t bytes = 0;
    for (auto _ : state) {
        ::unlink(kDiskPath);
        auto sink = arrow::io::FileOutputStream::Open(kDiskPath).ValueOrDie();
        write_parquet_to(sink, batch);
        auto st = sink->Close();
        if (!st.ok()) throw std::runtime_error(st.ToString());
        fsync_path(kDiskPath);
        struct stat sb; ::stat(kDiskPath, &sb);
        bytes = sb.st_size;
    }
    state.counters["out_MB"] = bytes / 1048576.0;
    state.counters["peak_rss_MB"] = peak_rss_mb();
}

static void BM_Mosaic_Disk(benchmark::State& state) {
    auto batch = get_or_build_batch(state.range(0), state.range(1));
    size_t bytes = 0;
    for (auto _ : state) {
        ::unlink(kDiskPath);
        FILE* fp = std::fopen(kDiskPath, "wb");
        if (!fp) throw std::runtime_error("fopen failed");
        size_t pos = 0;
        bytes = write_mosaic_to(
            [&](const uint8_t* p, size_t n) {
                size_t w = std::fwrite(p, 1, n, fp);
                pos += w;
                return w == n ? 0 : -1;
            },
            [&]() { return std::fflush(fp) == 0 ? 0 : -1; },
            [&]() { return (int64_t)pos; },
            batch);
        if (std::fflush(fp) != 0) throw std::runtime_error("fflush");
        if (::fsync(::fileno(fp)) != 0) throw std::runtime_error("fsync");
        std::fclose(fp);
    }
    state.counters["out_MB"] = bytes / 1048576.0;
    state.counters["peak_rss_MB"] = peak_rss_mb();
}

// Custom statistic: percentile.
template <int Pct>
double percentile(const std::vector<double>& v) {
    if (v.empty()) return 0;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    double idx = (Pct / 100.0) * (s.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return s[lo];
    return s[lo] + (idx - lo) * (s[hi] - s[lo]);
}

}  // namespace

// Row-count sweep at fixed 30K cols (customer target).
// Args = {n_cols, n_rows}.
#define REGISTER(BM)                                                    \
    BENCHMARK(BM)                                                        \
        ->Args({30000, 100})                                             \
        ->Args({30000, 500})                                             \
        ->Args({30000, 1500})                                            \
        ->Args({30000, 5000})                                            \
        ->Args({30000, 10000})                                           \
        ->Unit(benchmark::kMillisecond)                                  \
        ->Iterations(1)                                                  \
        ->Repetitions(10)                                                \
        ->ComputeStatistics("p50", percentile<50>)                       \
        ->ComputeStatistics("p95", percentile<95>)                       \
        ->ComputeStatistics("p99", percentile<99>)                       \
        ->ReportAggregatesOnly(true)                                     \
        ->UseRealTime()

REGISTER(BM_Parquet_Mem);
REGISTER(BM_Mosaic_Mem);
REGISTER(BM_Parquet_Disk);
REGISTER(BM_Mosaic_Disk);

BENCHMARK_MAIN();
