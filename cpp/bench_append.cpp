// Append-pattern (streaming) benchmark: Parquet vs Mosaic.
//
// Simulates car-like continuous writes: open writer once, call write() N times
// with small batches, close. Compares against writing the same total data in
// fewer large batches.
//
// Fixed: 30K cols, 1500 total rows
// Variable: rows_per_call ∈ {1, 10, 50, 100, 500, 1500}
//
// Measures:
//   - Total wall time (open + N writes + close)
//   - Per-call write() latency distribution (P50 / P99)
//   - Output file size
//
// Build: cmake --build cpp/build --target bench_append
// Run:   ./bench_append

#include "bench_common.hpp"
#include "mosaic.hpp"

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kNumCols = 30000;
constexpr int kTotalRows = 1500;
constexpr int kZstdLevel = 3;
constexpr int kReps = 5;

double now_ms() {
    auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(t).count();
}

template <int Pct>
double percentile(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double idx = (Pct / 100.0) * (v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return v[lo];
    return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}

struct AppendStats {
    double total_ms = 0;
    double open_ms = 0;
    double close_ms = 0;
    std::vector<double> write_call_ms;
    size_t output_bytes = 0;
};

AppendStats run_mosaic_append(
    const std::shared_ptr<arrow::RecordBatch>& batch, int n_calls)
{
    AppendStats s;
    s.write_call_ms.reserve(n_calls);

    std::vector<uint8_t> data;
    data.reserve(8 * 1024 * 1024);
    size_t pos = 0;

    mosaic::OutputFile of;
    of.write_fn = [&](const uint8_t* p, size_t n) -> int {
        data.insert(data.end(), p, p + n);
        pos += n;
        return 0;
    };
    of.flush_fn = []() -> int { return 0; };
    of.get_pos_fn = [&]() -> int64_t { return static_cast<int64_t>(pos); };

    struct ArrowSchema c_schema;
    auto st = arrow::ExportSchema(*batch->schema(), &c_schema);
    if (!st.ok()) throw std::runtime_error("ExportSchema: " + st.ToString());

    mosaic::WriterOptions opts;
    opts.compression = 1;
    opts.zstd_level = kZstdLevel;
    // Default row_group_max_size=256MB; all 1500 rows × 30K cols will fit
    // in a single row group regardless of how we split the writes.

    double t_start = now_ms();
    double t0 = now_ms();
    mosaic::Writer writer(of, &c_schema, opts);
    s.open_ms = now_ms() - t0;

    for (int i = 0; i < n_calls; i++) {
        struct ArrowArray c_array;
        struct ArrowSchema c_batch_schema;
        st = arrow::ExportRecordBatch(*batch, &c_array, &c_batch_schema);
        if (!st.ok()) throw std::runtime_error("ExportRecordBatch: " + st.ToString());

        double t1 = now_ms();
        writer.write(&c_array, &c_batch_schema);
        s.write_call_ms.push_back(now_ms() - t1);
    }

    t0 = now_ms();
    writer.close();
    s.close_ms = now_ms() - t0;

    s.total_ms = now_ms() - t_start;
    s.output_bytes = data.size();
    return s;
}

AppendStats run_parquet_append(
    const std::shared_ptr<arrow::RecordBatch>& batch, int n_calls)
{
    AppendStats s;
    s.write_call_ms.reserve(n_calls);

    auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();

    auto props = parquet::WriterProperties::Builder()
        .compression(parquet::Compression::ZSTD)
        ->compression_level(kZstdLevel)
        ->max_row_group_length(kTotalRows * 2)  // ensure single row group
        ->build();
    auto arrow_props = parquet::ArrowWriterProperties::Builder().build();

    double t_start = now_ms();
    double t0 = now_ms();
    auto writer_res = parquet::arrow::FileWriter::Open(
        *batch->schema(), arrow::default_memory_pool(),
        sink, props, arrow_props);
    if (!writer_res.ok()) throw std::runtime_error("parquet Open: " + writer_res.status().ToString());
    auto writer = std::move(writer_res).ValueOrDie();
    s.open_ms = now_ms() - t0;

    for (int i = 0; i < n_calls; i++) {
        double t1 = now_ms();
        auto st = writer->WriteRecordBatch(*batch);
        if (!st.ok()) throw std::runtime_error("parquet WriteRecordBatch: " + st.ToString());
        s.write_call_ms.push_back(now_ms() - t1);
    }

    t0 = now_ms();
    auto st = writer->Close();
    if (!st.ok()) throw std::runtime_error("parquet Close: " + st.ToString());
    s.close_ms = now_ms() - t0;

    s.total_ms = now_ms() - t_start;
    auto buf = sink->Finish().ValueOrDie();
    s.output_bytes = static_cast<size_t>(buf->size());
    return s;
}

void report(const char* fmt, int rpc, int n_calls,
            const std::vector<double>& totals,
            const std::vector<double>& opens,
            const std::vector<double>& closes,
            const std::vector<double>& all_call_times,
            size_t bytes)
{
    std::printf("%-8s %5d %5d  | %7.1f %7.1f %7.1f  | %7.2f %7.2f %7.2f  | %6.2f\n",
        fmt, rpc, n_calls,
        percentile<50>(totals), percentile<99>(totals),
        *std::max_element(totals.begin(), totals.end()),
        percentile<50>(all_call_times),
        percentile<99>(all_call_times),
        *std::max_element(all_call_times.begin(), all_call_times.end()),
        bytes / 1048576.0);
    (void)opens; (void)closes;
}

}  // namespace

int main() {
    std::printf("Append-pattern benchmark (%d cols × %d total rows, zstd-%d, in-memory)\n",
                kNumCols, kTotalRows, kZstdLevel);
    std::printf("Each row group fits in a single group regardless of call splitting.\n\n");

    std::vector<int> rpc_values = {1, 10, 50, 100, 500, 1500};

    std::printf("%-8s %-5s %-5s  | %-23s  | %-23s  | %s\n",
        "format", "rpc", "calls",
        "total_ms (p50/p99/max)",
        "per-call_ms (p50/p99/max)",
        "size_MB");
    std::printf("%s\n", std::string(95, '-').c_str());

    for (int rpc : rpc_values) {
        int n_calls = kTotalRows / rpc;
        std::fprintf(stderr, "[build batch] cols=%d rows=%d\n", kNumCols, rpc);
        auto specs = bench::build_specs(kNumCols);
        auto batch = bench::build_batch(specs, rpc);

        std::vector<double> p_totals, p_opens, p_closes, p_all_calls;
        size_t p_bytes = 0;
        std::vector<double> m_totals, m_opens, m_closes, m_all_calls;
        size_t m_bytes = 0;

        for (int r = 0; r < kReps; r++) {
            auto ps = run_parquet_append(batch, n_calls);
            p_totals.push_back(ps.total_ms);
            p_opens.push_back(ps.open_ms);
            p_closes.push_back(ps.close_ms);
            p_all_calls.insert(p_all_calls.end(),
                ps.write_call_ms.begin(), ps.write_call_ms.end());
            p_bytes = ps.output_bytes;

            auto ms = run_mosaic_append(batch, n_calls);
            m_totals.push_back(ms.total_ms);
            m_opens.push_back(ms.open_ms);
            m_closes.push_back(ms.close_ms);
            m_all_calls.insert(m_all_calls.end(),
                ms.write_call_ms.begin(), ms.write_call_ms.end());
            m_bytes = ms.output_bytes;
        }

        report("parquet", rpc, n_calls, p_totals, p_opens, p_closes, p_all_calls, p_bytes);
        report("mosaic",  rpc, n_calls, m_totals, m_opens, m_closes, m_all_calls, m_bytes);
        std::printf("\n");
    }
    return 0;
}
