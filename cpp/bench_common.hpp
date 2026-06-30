// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Shared synthetic-data generation for wide-table benchmarks.
// Matches /root/tmp/ characteristics: ~60% null, mostly int8/16/32 and double.

#pragma once

#include <arrow/api.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace bench {

inline constexpr unsigned kSeed = 42;

enum class ColType { Timestamp, Int8, Int16, Int32, Float64 };
enum class Sparsity { AllNull, AlwaysPresent, MostlyNull };

struct ColSpec {
    ColType type;
    Sparsity sparsity;
    std::string name;
};

inline std::vector<ColSpec> build_specs(int n_cols) {
    std::vector<ColSpec> out;
    out.reserve(n_cols);
    out.push_back({ColType::Timestamp, Sparsity::AlwaysPresent, "timestamp"});

    int data_cols = n_cols - 1;
    int n_i8 = data_cols * 794 / 1000;
    int n_f64 = data_cols * 113 / 1000;
    int n_i16_wide = data_cols * 45 / 1000;
    int n_i16 = data_cols * 31 / 1000;
    int n_i32_wide = data_cols - n_i8 - n_f64 - n_i16_wide - n_i16;

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

inline std::shared_ptr<arrow::DataType> arrow_type_for(ColType t) {
    switch (t) {
        case ColType::Timestamp: return arrow::int64();
        case ColType::Int8:      return arrow::int8();
        case ColType::Int16:     return arrow::int16();
        case ColType::Int32:     return arrow::int32();
        case ColType::Float64:   return arrow::float64();
    }
    return arrow::null();
}

inline std::shared_ptr<arrow::RecordBatch> build_batch(
    const std::vector<ColSpec>& specs, int n_rows) {

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

}  // namespace bench
