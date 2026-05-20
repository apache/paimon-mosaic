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

#pragma once

extern "C" {
#include "mosaic.h"
}
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mosaic {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

inline void check(int result) {
    if (result != 0) {
        const char* err = mosaic_last_error();
        throw Error(err ? err : "unknown error");
    }
}

// OutputFile adapter: wraps a C++ object into C callbacks
struct OutputFile {
    std::function<int(const uint8_t*, size_t)> write_fn;
    std::function<int()> flush_fn;
    std::function<int64_t()> get_pos_fn;
};

namespace detail {

inline int32_t stream_write(void* ctx, const uint8_t* data, size_t len) noexcept {
    try {
        auto* cbs = static_cast<OutputFile*>(ctx);
        return cbs->write_fn(data, len);
    } catch (...) {
        return -1;
    }
}

inline int32_t stream_flush(void* ctx) noexcept {
    try {
        auto* cbs = static_cast<OutputFile*>(ctx);
        return cbs->flush_fn();
    } catch (...) {
        return -1;
    }
}

inline int64_t stream_get_pos(void* ctx) noexcept {
    try {
        auto* cbs = static_cast<OutputFile*>(ctx);
        return cbs->get_pos_fn();
    } catch (...) {
        return -1;
    }
}

} // namespace detail

struct WriterOptions {
    uint8_t compression = 1;  // ZSTD
    int zstd_level = 1;
    uint32_t num_buckets = 0;
    uint64_t row_group_max_size = 256ULL * 1024 * 1024;
    uint32_t max_dict_total_bytes = 32 * 1024;
    uint32_t max_dict_entries = 255;
    const uint32_t* stats_columns = nullptr;
    uint32_t num_stats_columns = 0;
    uint32_t page_size_threshold = 32 * 1024;
};

// ======================== Statistics ========================

struct ColumnStatistics {
    uint32_t column_index;
    uint64_t null_count;
    std::vector<uint8_t> min_value;
    std::vector<uint8_t> max_value;
    bool has_min_max() const { return !min_value.empty(); }
};

class Writer {
public:
    /// Construct a writer. `arrow_schema` is a pointer to an ArrowSchema (Arrow C Data Interface).
    Writer(OutputFile callbacks, void* arrow_schema, WriterOptions opts = {})
        : callbacks_(std::make_shared<OutputFile>(std::move(callbacks))) {
        MosaicOutputFile stream;
        stream.ctx = callbacks_.get();
        stream.write_fn = detail::stream_write;
        stream.flush_fn = detail::stream_flush;
        stream.get_pos_fn = detail::stream_get_pos;

        MosaicWriterOptions c_opts = mosaic_writer_options_default();
        c_opts.compression = opts.compression;
        c_opts.zstd_level = opts.zstd_level;
        c_opts.num_buckets = opts.num_buckets;
        c_opts.row_group_max_size = opts.row_group_max_size;
        c_opts.max_dict_total_bytes = opts.max_dict_total_bytes;
        c_opts.max_dict_entries = opts.max_dict_entries;
        c_opts.stats_columns = opts.stats_columns;
        c_opts.num_stats_columns = opts.num_stats_columns;
        c_opts.page_size_threshold = opts.page_size_threshold;

        handle_ = mosaic_writer_open(stream, static_cast<ArrowSchema*>(arrow_schema), c_opts);
        if (!handle_) throw Error("failed to open writer");
    }

    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    Writer(Writer&& other) noexcept
        : callbacks_(std::move(other.callbacks_)), handle_(other.handle_), closed_(other.closed_) {
        other.handle_ = nullptr;
    }
    Writer& operator=(Writer&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                if (!closed_) mosaic_writer_close(handle_);
                mosaic_writer_free(handle_);
            }
            callbacks_ = std::move(other.callbacks_);
            handle_ = other.handle_;
            closed_ = other.closed_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Writer() {
        if (handle_) {
            if (!closed_) mosaic_writer_close(handle_);
            mosaic_writer_free(handle_);
        }
    }

    void write(void* ffi_array, void* ffi_schema) {
        check(mosaic_writer_write_batch(
            handle_,
            static_cast<ArrowArray*>(ffi_array),
            static_cast<ArrowSchema*>(ffi_schema)));
    }

    int64_t estimated_file_size() const {
        int64_t out = 0;
        check(mosaic_writer_estimated_file_size(handle_, &out));
        return out;
    }

    void close() {
        if (!closed_) {
            check(mosaic_writer_close(handle_));
            closed_ = true;
        }
    }

    uint32_t num_row_groups() const {
        uint32_t out = 0;
        check(mosaic_writer_num_row_groups(handle_, &out));
        return out;
    }

    std::vector<ColumnStatistics> get_row_group_statistics(uint32_t rg_index) const {
        uint32_t n = 0;
        check(mosaic_writer_row_group_num_stats(handle_, rg_index, &n));
        std::vector<ColumnStatistics> result;
        result.reserve(n);
        for (uint32_t i = 0; i < n; i++) {
            ColumnStatistics s;
            check(mosaic_writer_row_group_stat_column_index(handle_, rg_index, i, &s.column_index));
            check(mosaic_writer_row_group_stat_null_count(handle_, rg_index, i, &s.null_count));
            size_t min_len = 0, max_len = 0;
            const uint8_t* min_ptr = mosaic_writer_row_group_stat_min(handle_, rg_index, i, &min_len);
            const uint8_t* max_ptr = mosaic_writer_row_group_stat_max(handle_, rg_index, i, &max_len);
            if (min_ptr && min_len > 0)
                s.min_value.assign(min_ptr, min_ptr + min_len);
            if (max_ptr && max_len > 0)
                s.max_value.assign(max_ptr, max_ptr + max_len);
            result.push_back(std::move(s));
        }
        return result;
    }

private:
    std::shared_ptr<OutputFile> callbacks_;
    MosaicWriterHandle* handle_ = nullptr;
    bool closed_ = false;
};

// ======================== Reader ========================

/// Input file for reading Mosaic files.
///
/// `read_at_fn` must be thread-safe: the reader may call it concurrently
/// from multiple threads to perform parallel IO.
struct InputFile {
    std::function<int(uint64_t offset, uint8_t* buf, size_t len)> read_at_fn;
    uint64_t file_length = 0;
};

namespace detail {

inline int32_t input_read_at(void* ctx, uint64_t offset, uint8_t* buf, size_t len) noexcept {
    try {
        auto* cbs = static_cast<InputFile*>(ctx);
        return cbs->read_at_fn(offset, buf, len);
    } catch (...) {
        return -1;
    }
}

inline uint64_t input_length(void* ctx) noexcept {
    auto* cbs = static_cast<InputFile*>(ctx);
    return cbs->file_length;
}

} // namespace detail

class Reader {
public:
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    Reader(Reader&& other) noexcept
        : callbacks_(std::move(other.callbacks_)), handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    ~Reader() {
        if (handle_) mosaic_reader_free(handle_);
    }

    uint32_t num_row_groups() const {
        uint32_t out = 0;
        check(mosaic_reader_num_row_groups(handle_, &out));
        return out;
    }

    void export_schema(void* out_schema) const {
        check(mosaic_reader_export_schema(handle_, static_cast<ArrowSchema*>(out_schema)));
    }

    void read_row_group(uint32_t rg_index, void* out_array, void* out_schema) {
        auto* rg = mosaic_reader_open_row_group(handle_, rg_index);
        if (!rg) throw Error("failed to open row group");
        auto* rb = mosaic_row_group_reader_read_columns(rg);
        mosaic_row_group_reader_free(rg);
        if (!rb) throw Error("read_columns failed");
        int rc = mosaic_record_batch_export(rb,
            static_cast<ArrowArray*>(out_array),
            static_cast<ArrowSchema*>(out_schema));
        mosaic_record_batch_free(rb);
        if (rc != 0) throw Error("record_batch_export failed");
    }

    void read_row_group(uint32_t rg_index, const uint32_t* cols, uint32_t num_cols,
                        void* out_array, void* out_schema) {
        auto* rg = mosaic_reader_open_row_group_projected(handle_, rg_index, cols, num_cols);
        if (!rg) throw Error("failed to open row group");
        auto* rb = mosaic_row_group_reader_read_columns(rg);
        mosaic_row_group_reader_free(rg);
        if (!rb) throw Error("read_columns failed");
        int rc = mosaic_record_batch_export(rb,
            static_cast<ArrowArray*>(out_array),
            static_cast<ArrowSchema*>(out_schema));
        mosaic_record_batch_free(rb);
        if (rc != 0) throw Error("record_batch_export failed");
    }

    uint32_t row_group_num_rows(uint32_t rg_index) const {
        uint32_t out = 0;
        check(mosaic_reader_row_group_num_rows(handle_, rg_index, &out));
        return out;
    }

    std::vector<ColumnStatistics> get_row_group_statistics(uint32_t rg_index) const {
        uint32_t n = 0;
        check(mosaic_reader_row_group_num_stats(handle_, rg_index, &n));
        std::vector<ColumnStatistics> result;
        result.reserve(n);
        for (uint32_t i = 0; i < n; i++) {
            ColumnStatistics s;
            check(mosaic_reader_row_group_stat_column_index(handle_, rg_index, i, &s.column_index));
            check(mosaic_reader_row_group_stat_null_count(handle_, rg_index, i, &s.null_count));
            size_t min_len = 0, max_len = 0;
            const uint8_t* min_ptr = mosaic_reader_row_group_stat_min(handle_, rg_index, i, &min_len);
            const uint8_t* max_ptr = mosaic_reader_row_group_stat_max(handle_, rg_index, i, &max_len);
            if (min_ptr && min_len > 0)
                s.min_value.assign(min_ptr, min_ptr + min_len);
            if (max_ptr && max_len > 0)
                s.max_value.assign(max_ptr, max_ptr + max_len);
            result.push_back(std::move(s));
        }
        return result;
    }

private:
    friend Reader make_reader(InputFile callbacks, uint64_t len);
    Reader(std::shared_ptr<InputFile> cbs, MosaicReaderHandle* h)
        : callbacks_(std::move(cbs)), handle_(h) {}
    std::shared_ptr<InputFile> callbacks_;
    MosaicReaderHandle* handle_;
};

inline Reader make_reader(InputFile callbacks, uint64_t len) {
    callbacks.file_length = len;
    auto cbs = std::make_shared<InputFile>(std::move(callbacks));
    MosaicInputFile input;
    input.ctx = cbs.get();
    input.read_at_fn = detail::input_read_at;
    input.length_fn = detail::input_length;
    auto* handle = mosaic_reader_open(input);
    if (!handle) throw Error("failed to open reader");
    return Reader(std::move(cbs), handle);
}

} // namespace mosaic
