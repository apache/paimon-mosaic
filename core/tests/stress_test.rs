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

#![allow(
    clippy::approx_constant,
    clippy::unnecessary_cast,
    clippy::cloned_ref_to_slice_refs,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of
)]

use std::io;
use std::sync::Arc;

use arrow_array::*;
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use paimon_mosaic_core::reader::{InputFile, MosaicReader, ReaderAccess};
use paimon_mosaic_core::writer::{MosaicWriter, OutputFile, WriterOptions};

struct MemOutputFile {
    pub buf: Vec<u8>,
}

impl MemOutputFile {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
}

impl OutputFile for MemOutputFile {
    fn write(&mut self, data: &[u8]) -> io::Result<()> {
        self.buf.extend_from_slice(data);
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
    fn pos(&self) -> u64 {
        self.buf.len() as u64
    }
}

struct ByteArrayInputFile {
    data: Vec<u8>,
}

impl InputFile for ByteArrayInputFile {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let start = offset as usize;
        let end = start + buf.len();
        if end > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "read past end",
            ));
        }
        buf.copy_from_slice(&self.data[start..end]);
        Ok(())
    }
}

fn roundtrip(schema: &Schema, batches: &[RecordBatch], options: WriterOptions) -> Vec<RecordBatch> {
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(out, schema, options).unwrap();
    for batch in batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();

    let data = writer.output().buf.clone();
    let file_len = data.len() as u64;
    let input = ByteArrayInputFile { data };
    let reader = MosaicReader::new(input, file_len).unwrap();

    let mut result = Vec::new();
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader(rg).unwrap();
        result.push(rg_reader.read_columns().unwrap());
    }
    result
}

fn assert_batches_equal(expected: &[RecordBatch], actual: &[RecordBatch]) {
    let expected_rows: usize = expected.iter().map(|b| b.num_rows()).sum();
    let actual_rows: usize = actual.iter().map(|b| b.num_rows()).sum();
    assert_eq!(expected_rows, actual_rows, "total row count mismatch");

    let mut exp_offset = 0;
    let mut act_offset = 0;
    let mut exp_batch_idx = 0;
    let mut act_batch_idx = 0;

    let num_cols = expected[0].num_columns();
    let exp_schema = expected[0].schema();
    let mut row = 0;
    while row < expected_rows {
        let exp_batch = &expected[exp_batch_idx];
        let act_batch = &actual[act_batch_idx];
        let exp_remaining = exp_batch.num_rows() - exp_offset;
        let act_remaining = act_batch.num_rows() - act_offset;
        let chunk = exp_remaining.min(act_remaining);

        for col in 0..num_cols {
            let col_name = exp_schema.field(col).name();
            let act_col_idx = act_batch.schema().index_of(col_name).unwrap();
            let exp_col = exp_batch.column(col).slice(exp_offset, chunk);
            let act_col = act_batch.column(act_col_idx).slice(act_offset, chunk);
            assert_eq!(
                &exp_col,
                &act_col,
                "mismatch at column {} rows {}..{}",
                col_name,
                row,
                row + chunk
            );
        }

        exp_offset += chunk;
        act_offset += chunk;
        row += chunk;
        if exp_offset == exp_batch.num_rows() {
            exp_batch_idx += 1;
            exp_offset = 0;
        }
        if act_offset == act_batch.num_rows() {
            act_batch_idx += 1;
            act_offset = 0;
        }
    }
}

// ============== Test: Large Int64 file (~50MB uncompressed) ==============
#[test]
fn test_large_int64_file() {
    let num_rows = 3_000_000;
    let batch_size = 100_000;
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int64, true),
    ]);

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;
        let ids: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();
        let values: Vec<Option<i64>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 7 == 0 {
                    None
                } else {
                    Some((batch_start + i) as i64 * 17)
                }
            })
            .collect();
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(values)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 10,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&batches, &result);
}

// ============== Test: Wide table (200 columns) ==============
#[test]
fn test_wide_table_200_columns() {
    let num_cols = 200;
    let num_rows = 50_000;
    let batch_size = 10_000;

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| {
            let dt = match i % 4 {
                0 => DataType::Int32,
                1 => DataType::Int64,
                2 => DataType::Float64,
                _ => DataType::Utf8,
            };
            Field::new(format!("col_{:04}", i), dt, true)
        })
        .collect();
    let schema = Schema::new(fields);

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;
        let mut arrays: Vec<Arc<dyn Array>> = Vec::new();
        for col in 0..num_cols {
            let arr: Arc<dyn Array> = match col % 4 {
                0 => {
                    let vals: Vec<Option<i32>> = (0..count)
                        .map(|i| {
                            if (batch_start + i) % 11 == 0 {
                                None
                            } else {
                                Some((batch_start + i) as i32)
                            }
                        })
                        .collect();
                    Arc::new(Int32Array::from(vals))
                }
                1 => {
                    let vals: Vec<Option<i64>> = (0..count)
                        .map(|i| {
                            if (batch_start + i) % 13 == 0 {
                                None
                            } else {
                                Some((batch_start + i) as i64 * col as i64)
                            }
                        })
                        .collect();
                    Arc::new(Int64Array::from(vals))
                }
                2 => {
                    let vals: Vec<Option<f64>> = (0..count)
                        .map(|i| {
                            if (batch_start + i) % 17 == 0 {
                                None
                            } else {
                                Some((batch_start + i) as f64 / (col + 1) as f64)
                            }
                        })
                        .collect();
                    Arc::new(Float64Array::from(vals))
                }
                _ => {
                    let vals: Vec<Option<&str>> = (0..count)
                        .map(|i| {
                            if (batch_start + i) % 19 == 0 {
                                None
                            } else {
                                Some(["alpha", "beta", "gamma", "delta"][(batch_start + i) % 4])
                            }
                        })
                        .collect();
                    Arc::new(StringArray::from(vals))
                }
            };
            arrays.push(arr);
        }
        batches.push(RecordBatch::try_new(Arc::new(schema.clone()), arrays).unwrap());
    }

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 50,
            row_group_max_size: 8 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&batches, &result);
}

// ============== Test: All data types roundtrip at scale ==============
#[test]
fn test_all_types_at_scale() {
    let num_rows = 500_000;
    let schema = Schema::new(vec![
        Field::new("bool_col", DataType::Boolean, true),
        Field::new("i8_col", DataType::Int8, true),
        Field::new("i16_col", DataType::Int16, true),
        Field::new("i32_col", DataType::Int32, true),
        Field::new("i64_col", DataType::Int64, true),
        Field::new("f32_col", DataType::Float32, true),
        Field::new("f64_col", DataType::Float64, true),
        Field::new("date_col", DataType::Date32, true),
        Field::new("time_col", DataType::Time32(TimeUnit::Millisecond), true),
        Field::new("str_col", DataType::Utf8, true),
        Field::new("bin_col", DataType::Binary, true),
        Field::new("dec_col", DataType::Decimal128(10, 2), true),
        Field::new(
            "ts_ms_col",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ),
        Field::new(
            "ts_us_col",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "ts_ns_col",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            true,
        ),
    ]);

    let batch_size = 50_000;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let bools: Vec<Option<bool>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 10 == 0 {
                    None
                } else {
                    Some((batch_start + i) % 2 == 0)
                }
            })
            .collect();

        let i8s: Vec<Option<i8>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 15 == 0 {
                    None
                } else {
                    Some(((batch_start + i) % 256) as i8)
                }
            })
            .collect();

        let i16s: Vec<Option<i16>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 12 == 0 {
                    None
                } else {
                    Some(((batch_start + i) % 30000) as i16)
                }
            })
            .collect();

        let i32s: Vec<Option<i32>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 9 == 0 {
                    None
                } else {
                    Some((batch_start + i) as i32 * 3)
                }
            })
            .collect();

        let i64s: Vec<Option<i64>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 8 == 0 {
                    None
                } else {
                    Some((batch_start + i) as i64 * 1_000_000)
                }
            })
            .collect();

        let f32s: Vec<Option<f32>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 14 == 0 {
                    None
                } else {
                    Some((batch_start + i) as f32 * 0.1)
                }
            })
            .collect();

        let f64s: Vec<Option<f64>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 11 == 0 {
                    None
                } else {
                    Some((batch_start + i) as f64 * 3.14159)
                }
            })
            .collect();

        let dates: Vec<Option<i32>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 20 == 0 {
                    None
                } else {
                    Some(18000 + (batch_start + i) as i32 % 3650)
                }
            })
            .collect();

        let times: Vec<Option<i32>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 25 == 0 {
                    None
                } else {
                    Some(((batch_start + i) as i32 % 86_400_000).abs())
                }
            })
            .collect();

        let strings: Vec<Option<String>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 6 == 0 {
                    None
                } else {
                    Some(format!("row_{:08}", batch_start + i))
                }
            })
            .collect();
        let str_refs: Vec<Option<&str>> = strings.iter().map(|s| s.as_deref()).collect();

        let binaries: Vec<Option<Vec<u8>>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 7 == 0 {
                    None
                } else {
                    let len = (batch_start + i) % 32 + 1;
                    Some(
                        (0..len)
                            .map(|j| ((batch_start + i + j) % 256) as u8)
                            .collect(),
                    )
                }
            })
            .collect();
        let bin_refs: Vec<Option<&[u8]>> = binaries
            .iter()
            .map(|b| b.as_ref().map(|v| v.as_slice()))
            .collect();

        let decimals: Vec<Option<i128>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 13 == 0 {
                    None
                } else {
                    Some((batch_start + i) as i128 * 100 + 99)
                }
            })
            .collect();

        let ts_ms: Vec<Option<i64>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 16 == 0 {
                    None
                } else {
                    Some(1_700_000_000_000i64 + (batch_start + i) as i64 * 1000)
                }
            })
            .collect();

        let ts_us: Vec<Option<i64>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 18 == 0 {
                    None
                } else {
                    Some(1_700_000_000_000_000i64 + (batch_start + i) as i64)
                }
            })
            .collect();

        let ts_ns: Vec<Option<i64>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 21 == 0 {
                    None
                } else {
                    let millis = 1_700_000_000_000i64 + (batch_start + i) as i64;
                    let nanos = ((batch_start + i) % 1_000_000) as i32;
                    Some(paimon_mosaic_core::types::millis_nanos_to_ns(millis, nanos).unwrap())
                }
            })
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(BooleanArray::from(bools)),
                Arc::new(Int8Array::from(i8s)),
                Arc::new(Int16Array::from(i16s)),
                Arc::new(Int32Array::from(i32s)),
                Arc::new(Int64Array::from(i64s)),
                Arc::new(Float32Array::from(f32s)),
                Arc::new(Float64Array::from(f64s)),
                Arc::new(Date32Array::from(dates)),
                Arc::new(Time32MillisecondArray::from(times)),
                Arc::new(StringArray::from(str_refs)),
                Arc::new(BinaryArray::from(bin_refs)),
                Arc::new(
                    Decimal128Array::from(decimals)
                        .with_precision_and_scale(10, 2)
                        .unwrap(),
                ),
                Arc::new(TimestampMillisecondArray::from(ts_ms)),
                Arc::new(TimestampMicrosecondArray::from(ts_us)),
                Arc::new(TimestampNanosecondArray::from(ts_ns)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 20,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&batches, &result);
}

// ============== Test: Large strings (variable-length stress) ==============
#[test]
fn test_large_strings() {
    let num_rows = 200_000;
    let batch_size = 20_000;
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("short_str", DataType::Utf8, true),
        Field::new("long_str", DataType::Utf8, true),
        Field::new("binary_data", DataType::Binary, true),
    ]);

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let ids: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();

        let short_strs: Vec<Option<String>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 5 == 0 {
                    None
                } else {
                    Some(format!("s_{}", (batch_start + i) % 100))
                }
            })
            .collect();
        let short_refs: Vec<Option<&str>> = short_strs.iter().map(|s| s.as_deref()).collect();

        let long_strs: Vec<Option<String>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 3 == 0 {
                    None
                } else {
                    let len = 100 + (batch_start + i) % 900;
                    Some("x".repeat(len))
                }
            })
            .collect();
        let long_refs: Vec<Option<&str>> = long_strs.iter().map(|s| s.as_deref()).collect();

        let bin_data: Vec<Option<Vec<u8>>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 4 == 0 {
                    None
                } else {
                    let len = 50 + (batch_start + i) % 500;
                    Some(
                        (0..len)
                            .map(|j| ((batch_start + i + j) % 256) as u8)
                            .collect(),
                    )
                }
            })
            .collect();
        let bin_refs: Vec<Option<&[u8]>> = bin_data
            .iter()
            .map(|b| b.as_ref().map(|v| v.as_slice()))
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(short_refs)),
                Arc::new(StringArray::from(long_refs)),
                Arc::new(BinaryArray::from(bin_refs)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 4,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&batches, &result);
}

// ============== Test: Many small batches (simulating streaming writes) ==============
#[test]
fn test_many_small_batches() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8, true),
    ]);

    let mut batches = Vec::new();
    for i in 0..5000 {
        let count = 10 + i % 50;
        let a_vals: Vec<Option<i32>> = (0..count)
            .map(|j| {
                if j % 3 == 0 {
                    None
                } else {
                    Some(i as i32 * 100 + j as i32)
                }
            })
            .collect();
        let b_vals: Vec<Option<&str>> = (0..count)
            .map(|j| if j % 4 == 0 { None } else { Some("hello") })
            .collect();
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(a_vals)),
                Arc::new(StringArray::from(b_vals)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 2,
            row_group_max_size: 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&batches, &result);
}

// ============== Test: Extreme null patterns ==============
#[test]
fn test_extreme_null_patterns() {
    let schema = Schema::new(vec![
        Field::new("all_null", DataType::Int64, true),
        Field::new("mostly_null", DataType::Utf8, true),
        Field::new("rarely_null", DataType::Float64, true),
        Field::new("no_null", DataType::Int32, false),
    ]);

    let num_rows = 1_000_000;
    let batch_size = 100_000;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let all_null: Vec<Option<i64>> = vec![None; count];
        let mostly_null: Vec<Option<&str>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 1000 == 0 {
                    Some("rare_value")
                } else {
                    None
                }
            })
            .collect();
        let rarely_null: Vec<Option<f64>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 100_000 == 0 {
                    None
                } else {
                    Some((batch_start + i) as f64)
                }
            })
            .collect();
        let no_null: Vec<i32> = (0..count).map(|i| (batch_start + i) as i32).collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(all_null)),
                Arc::new(StringArray::from(mostly_null)),
                Arc::new(Float64Array::from(rarely_null)),
                Arc::new(Int32Array::from(no_null)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 10,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&batches, &result);
}

// ============== Test: Constant columns at scale ==============
#[test]
fn test_constant_columns_at_scale() {
    let schema = Schema::new(vec![
        Field::new("const_int", DataType::Int32, false),
        Field::new("const_str", DataType::Utf8, false),
        Field::new("const_f64", DataType::Float64, false),
        Field::new("varying", DataType::Int64, false),
    ]);

    let num_rows = 2_000_000;
    let batch_size = 200_000;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let const_ints: Vec<i32> = vec![42; count];
        let const_strs: Vec<&str> = vec!["constant_value"; count];
        let const_f64s: Vec<f64> = vec![3.14159; count];
        let varying: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(const_ints)),
                Arc::new(StringArray::from(const_strs)),
                Arc::new(Float64Array::from(const_f64s)),
                Arc::new(Int64Array::from(varying)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 4,
            row_group_max_size: 64 * 1024 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    for batch in &batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();

    let data = writer.output().buf.clone();
    let file_size = data.len();

    // Constant columns should compress extremely well
    // Raw data would be ~30MB+ but with CONST encoding should be much smaller
    assert!(
        file_size < 20 * 1024 * 1024,
        "file size {} bytes is too large for constant columns",
        file_size
    );

    let file_len = data.len() as u64;
    let input = ByteArrayInputFile { data };
    let reader = MosaicReader::new(input, file_len).unwrap();

    let mut result = Vec::new();
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader(rg).unwrap();
        result.push(rg_reader.read_columns().unwrap());
    }

    assert_batches_equal(&batches, &result);
}

// ============== Test: Dict encoding stress (many distinct values near limit) ==============
#[test]
fn test_dict_encoding_boundary() {
    let schema = Schema::new(vec![
        Field::new("low_card", DataType::Int32, true),
        Field::new("med_card", DataType::Utf8, true),
        Field::new("high_card", DataType::Int64, false),
    ]);

    let num_rows = 500_000;
    let batch_size = 100_000;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        // 5 distinct values -> dict
        let low_card: Vec<Option<i32>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 20 == 0 {
                    None
                } else {
                    Some(((batch_start + i) % 5) as i32)
                }
            })
            .collect();

        // 200 distinct values -> near dict boundary
        let med_card: Vec<Option<String>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 10 == 0 {
                    None
                } else {
                    Some(format!("val_{:03}", (batch_start + i) % 200))
                }
            })
            .collect();
        let med_refs: Vec<Option<&str>> = med_card.iter().map(|s| s.as_deref()).collect();

        // All unique values -> plain
        let high_card: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(low_card)),
                Arc::new(StringArray::from(med_refs)),
                Arc::new(Int64Array::from(high_card)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 3,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&batches, &result);
}

// ============== Test: Column projection at scale ==============
#[test]
fn test_column_projection_at_scale() {
    let num_cols = 50;
    let num_rows = 200_000;
    let batch_size = 50_000;

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| Field::new(format!("col_{:03}", i), DataType::Int64, true))
        .collect();
    let schema = Schema::new(fields);

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let arrays: Vec<Arc<dyn Array>> = (0..num_cols)
            .map(|col| {
                let vals: Vec<Option<i64>> = (0..count)
                    .map(|i| {
                        if (batch_start + i + col) % 10 == 0 {
                            None
                        } else {
                            Some((batch_start + i) as i64 * (col + 1) as i64)
                        }
                    })
                    .collect();
                Arc::new(Int64Array::from(vals)) as Arc<dyn Array>
            })
            .collect();

        batches.push(RecordBatch::try_new(Arc::new(schema.clone()), arrays).unwrap());
    }

    // Write
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 20,
            row_group_max_size: 8 * 1024 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    for batch in &batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();

    let data = writer.output().buf.clone();
    let file_len = data.len() as u64;
    let input = ByteArrayInputFile { data };
    let reader = MosaicReader::new(input, file_len).unwrap();

    // Read only a few projected columns
    let projected_cols: Vec<usize> = vec![0, 10, 25, 49];
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader
            .row_group_reader_projected(rg, &projected_cols)
            .unwrap();
        let batch = rg_reader.read_columns().unwrap();
        // Should only have 4 columns
        assert_eq!(batch.num_columns(), projected_cols.len());
        assert!(batch.num_rows() > 0);
    }
}

// ============== Test: Multiple row groups with varying sizes ==============
#[test]
fn test_multiple_row_groups_varying_sizes() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, true),
    ]);

    let mut batches = Vec::new();
    let mut total_rows = 0usize;

    // Write batches of varying sizes
    let sizes = [
        1, 10, 100, 1000, 10000, 50000, 100000, 50000, 10000, 1000, 100, 10, 1,
    ];
    for (idx, &size) in sizes.iter().enumerate() {
        let ids: Vec<i64> = (total_rows as i64..(total_rows + size) as i64).collect();
        let data: Vec<Option<String>> = (0..size)
            .map(|i| {
                if i % 3 == 0 {
                    None
                } else {
                    Some(format!("batch_{}_row_{}", idx, i))
                }
            })
            .collect();
        let data_refs: Vec<Option<&str>> = data.iter().map(|s| s.as_deref()).collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(data_refs)),
            ],
        )
        .unwrap();
        batches.push(batch);
        total_rows += size;
    }

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 2,
            row_group_max_size: 512 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&batches, &result);
}

// ============== Test: Edge values ==============
#[test]
fn test_edge_values() {
    let schema = Schema::new(vec![
        Field::new("i32_col", DataType::Int32, true),
        Field::new("i64_col", DataType::Int64, true),
        Field::new("f32_col", DataType::Float32, true),
        Field::new("f64_col", DataType::Float64, true),
        Field::new("str_col", DataType::Utf8, true),
    ]);

    let num_rows = 100_000;
    let i32_vals: Vec<Option<i32>> = (0..num_rows)
        .map(|i| match i % 7 {
            0 => None,
            1 => Some(i32::MIN),
            2 => Some(i32::MAX),
            3 => Some(0),
            4 => Some(-1),
            5 => Some(1),
            _ => Some(i as i32),
        })
        .collect();

    let i64_vals: Vec<Option<i64>> = (0..num_rows)
        .map(|i| match i % 7 {
            0 => None,
            1 => Some(i64::MIN),
            2 => Some(i64::MAX),
            3 => Some(0),
            4 => Some(-1),
            5 => Some(1),
            _ => Some(i as i64 * 1_000_000_000),
        })
        .collect();

    let f32_vals: Vec<Option<f32>> = (0..num_rows)
        .map(|i| match i % 8 {
            0 => None,
            1 => Some(f32::MIN),
            2 => Some(f32::MAX),
            3 => Some(0.0),
            4 => Some(-0.0),
            5 => Some(f32::INFINITY),
            6 => Some(f32::NEG_INFINITY),
            _ => Some(i as f32 * 0.001),
        })
        .collect();

    let f64_vals: Vec<Option<f64>> = (0..num_rows)
        .map(|i| match i % 8 {
            0 => None,
            1 => Some(f64::MIN),
            2 => Some(f64::MAX),
            3 => Some(0.0),
            4 => Some(-0.0),
            5 => Some(f64::INFINITY),
            6 => Some(f64::NEG_INFINITY),
            _ => Some(i as f64 * 0.000001),
        })
        .collect();

    let str_vals: Vec<Option<String>> = (0..num_rows)
        .map(|i| match i % 6 {
            0 => None,
            1 => Some(String::new()),
            2 => Some("a".repeat(1000)),
            3 => Some("\u{4e2d}\u{6587}\u{6d4b}\u{8bd5}".to_string()),
            4 => Some("\0\0\0".to_string()),
            _ => Some(format!("normal_{}", i)),
        })
        .collect();
    let str_refs: Vec<Option<&str>> = str_vals.iter().map(|s| s.as_deref()).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(i32_vals)),
            Arc::new(Int64Array::from(i64_vals)),
            Arc::new(Float32Array::from(f32_vals)),
            Arc::new(Float64Array::from(f64_vals)),
            Arc::new(StringArray::from(str_refs)),
        ],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 5,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
}

// ============== Test: No compression at scale ==============
#[test]
fn test_no_compression_large() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Float64, true),
        Field::new("c", DataType::Utf8, true),
    ]);

    let num_rows = 500_000;
    let batch_size = 100_000;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let a: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();
        let b: Vec<Option<f64>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 5 == 0 {
                    None
                } else {
                    Some((batch_start + i) as f64 * 2.5)
                }
            })
            .collect();
        let c: Vec<Option<String>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 4 == 0 {
                    None
                } else {
                    Some(format!("item_{}", batch_start + i))
                }
            })
            .collect();
        let c_refs: Vec<Option<&str>> = c.iter().map(|s| s.as_deref()).collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(a)),
                Arc::new(Float64Array::from(b)),
                Arc::new(StringArray::from(c_refs)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 3,
            compression: 0, // COMPRESSION_NONE
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&batches, &result);
}

// ============== Test: Single row batches (edge case) ==============
#[test]
fn test_single_row_batches() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, true),
    ]);

    let mut batches = Vec::new();
    for i in 0..10_000 {
        let val: Option<&str> = if i % 3 == 0 { None } else { Some("x") };
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![i as i32])),
                Arc::new(StringArray::from(vec![val])),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 2,
            row_group_max_size: 64 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&batches, &result);
}

// ============== Test: High bucket count ==============
#[test]
fn test_high_bucket_count() {
    let num_cols = 150;
    let num_rows = 100_000;

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| Field::new(format!("c_{:03}", i), DataType::Int32, true))
        .collect();
    let schema = Schema::new(fields);

    let arrays: Vec<Arc<dyn Array>> = (0..num_cols)
        .map(|col| {
            let vals: Vec<Option<i32>> = (0..num_rows)
                .map(|i| {
                    if (i + col) % 8 == 0 {
                        None
                    } else {
                        Some((i * col) as i32)
                    }
                })
                .collect();
            Arc::new(Int32Array::from(vals)) as Arc<dyn Array>
        })
        .collect();

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), arrays).unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 100,
            row_group_max_size: 64 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
}

// ============== Test: Paged bucket encoding at scale ==============
#[test]
fn test_paged_bucket_large() {
    let num_cols = 30;
    let num_rows = 500_000;
    let batch_size = 100_000;

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| {
            let dt = if i % 3 == 0 {
                DataType::Utf8
            } else {
                DataType::Int64
            };
            Field::new(format!("field_{:03}", i), dt, true)
        })
        .collect();
    let schema = Schema::new(fields);

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let arrays: Vec<Arc<dyn Array>> = (0..num_cols)
            .map(|col| {
                if col % 3 == 0 {
                    let vals: Vec<Option<String>> = (0..count)
                        .map(|i| {
                            if (batch_start + i) % 7 == 0 {
                                None
                            } else {
                                Some(format!("field{}_{}", col, (batch_start + i) % 50))
                            }
                        })
                        .collect();
                    let refs: Vec<Option<&str>> = vals.iter().map(|s| s.as_deref()).collect();
                    Arc::new(StringArray::from(refs)) as Arc<dyn Array>
                } else {
                    let vals: Vec<Option<i64>> = (0..count)
                        .map(|i| {
                            if (batch_start + i + col) % 9 == 0 {
                                None
                            } else {
                                Some((batch_start + i) as i64 * col as i64)
                            }
                        })
                        .collect();
                    Arc::new(Int64Array::from(vals)) as Arc<dyn Array>
                }
            })
            .collect();

        batches.push(RecordBatch::try_new(Arc::new(schema.clone()), arrays).unwrap());
    }

    // Use small page_size_threshold to force paged encoding
    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 5,
            row_group_max_size: 64 * 1024 * 1024,
            page_size_threshold: 4096,
            ..Default::default()
        },
    );

    assert_batches_equal(&batches, &result);
}

// ============== Test: 200MB file with mixed types ==============
#[test]
fn test_200mb_mixed_types() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("category", DataType::Utf8, true),
        Field::new("amount", DataType::Float64, true),
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ),
        Field::new("description", DataType::Utf8, true),
        Field::new("flag", DataType::Boolean, true),
        Field::new("counter", DataType::Int32, true),
        Field::new("price", DataType::Decimal128(12, 4), true),
    ]);

    let total_rows = 5_000_000;
    let batch_size = 250_000;
    let mut batches = Vec::new();

    for batch_start in (0..total_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(total_rows);
        let count = end - batch_start;

        let ids: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();

        let categories: Vec<Option<&str>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 20 == 0 {
                    None
                } else {
                    Some(
                        [
                            "electronics",
                            "clothing",
                            "food",
                            "books",
                            "toys",
                            "home",
                            "sports",
                            "music",
                        ][(batch_start + i) % 8],
                    )
                }
            })
            .collect();

        let amounts: Vec<Option<f64>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 15 == 0 {
                    None
                } else {
                    Some(((batch_start + i) as f64 % 10000.0) * 0.01)
                }
            })
            .collect();

        let timestamps: Vec<Option<i64>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 25 == 0 {
                    None
                } else {
                    Some(1_700_000_000_000i64 + (batch_start + i) as i64 * 60_000)
                }
            })
            .collect();

        let descriptions: Vec<Option<String>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 4 == 0 {
                    None
                } else {
                    let len = 20 + (batch_start + i) % 80;
                    Some(format!("{:_>width$}", batch_start + i, width = len))
                }
            })
            .collect();
        let desc_refs: Vec<Option<&str>> = descriptions.iter().map(|s| s.as_deref()).collect();

        let flags: Vec<Option<bool>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 12 == 0 {
                    None
                } else {
                    Some((batch_start + i) % 2 == 0)
                }
            })
            .collect();

        let counters: Vec<Option<i32>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 9 == 0 {
                    None
                } else {
                    Some(((batch_start + i) % 1000) as i32)
                }
            })
            .collect();

        let prices: Vec<Option<i128>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 11 == 0 {
                    None
                } else {
                    Some(((batch_start + i) % 1_000_000) as i128 * 10000 + 5000)
                }
            })
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(categories)),
                Arc::new(Float64Array::from(amounts)),
                Arc::new(TimestampMillisecondArray::from(timestamps)),
                Arc::new(StringArray::from(desc_refs)),
                Arc::new(BooleanArray::from(flags)),
                Arc::new(Int32Array::from(counters)),
                Arc::new(
                    Decimal128Array::from(prices)
                        .with_precision_and_scale(12, 4)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 20,
            row_group_max_size: 64 * 1024 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    for batch in &batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();

    let data = writer.output().buf.clone();
    let file_size = data.len();
    println!(
        "200MB test: {} rows, file size = {} MB",
        total_rows,
        file_size / (1024 * 1024)
    );

    // Verify readable
    let file_len = data.len() as u64;
    let input = ByteArrayInputFile { data };
    let reader = MosaicReader::new(input, file_len).unwrap();

    let mut total_read_rows = 0;
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader(rg).unwrap();
        let batch = rg_reader.read_columns().unwrap();
        assert_eq!(batch.num_columns(), 8);
        total_read_rows += batch.num_rows();
    }
    assert_eq!(total_read_rows, total_rows);

    // Also verify projection works
    let projected = vec![0, 2, 5];
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader_projected(rg, &projected).unwrap();
        let batch = rg_reader.read_columns().unwrap();
        assert_eq!(batch.num_columns(), 3);
    }
}

// ============== Test: Decimal128 with high precision (>18) ==============
#[test]
fn test_large_decimal_precision() {
    let schema = Schema::new(vec![
        Field::new("small_dec", DataType::Decimal128(10, 2), true),
        Field::new("large_dec", DataType::Decimal128(38, 10), true),
    ]);

    let num_rows = 100_000;
    let small_vals: Vec<Option<i128>> = (0..num_rows)
        .map(|i| {
            if i % 7 == 0 {
                None
            } else {
                Some(i as i128 * 100 - 5_000_000)
            }
        })
        .collect();

    let large_vals: Vec<Option<i128>> = (0..num_rows)
        .map(|i| {
            if i % 9 == 0 {
                None
            } else {
                Some(i as i128 * 10_000_000_000i128 + 123_456_789)
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(
                Decimal128Array::from(small_vals)
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from(large_vals)
                    .with_precision_and_scale(38, 10)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 2,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
}

// ============== Test: Timestamp with timezone ==============
#[test]
fn test_timestamp_with_timezone() {
    let tz: Arc<str> = Arc::from("America/New_York");
    let schema = Schema::new(vec![
        Field::new(
            "ts_ms_tz",
            DataType::Timestamp(TimeUnit::Millisecond, Some(tz.clone())),
            true,
        ),
        Field::new(
            "ts_us_tz",
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz.clone())),
            true,
        ),
    ]);

    let num_rows = 200_000;
    let ms_vals: Vec<Option<i64>> = (0..num_rows)
        .map(|i| {
            if i % 10 == 0 {
                None
            } else {
                Some(1_700_000_000_000i64 + i as i64 * 1000)
            }
        })
        .collect();

    let us_vals: Vec<Option<i64>> = (0..num_rows)
        .map(|i| {
            if i % 8 == 0 {
                None
            } else {
                Some(1_700_000_000_000_000i64 + i as i64)
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(TimestampMillisecondArray::from(ms_vals).with_timezone(tz.clone())),
            Arc::new(TimestampMicrosecondArray::from(us_vals).with_timezone(tz.clone())),
        ],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 2,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
}

// ============== Test: Stats correctness ==============
#[test]
fn test_stats_at_scale() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int32, true),
        Field::new("name", DataType::Utf8, true),
    ]);

    let num_rows = 1_000_000;
    let batch_size = 200_000;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let ids: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();
        let values: Vec<Option<i32>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 10 == 0 {
                    None
                } else {
                    Some(((batch_start + i) % 5000) as i32)
                }
            })
            .collect();
        let names: Vec<Option<&str>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 5 == 0 {
                    None
                } else {
                    Some("test")
                }
            })
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int32Array::from(values)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    // Write with stats on columns 0 and 1
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 10,
            row_group_max_size: 32 * 1024 * 1024,
            stats_columns: vec!["id".to_string(), "value".to_string()],
            ..Default::default()
        },
    )
    .unwrap();
    for batch in &batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();

    let data = writer.output().buf.clone();
    let file_len = data.len() as u64;
    let input = ByteArrayInputFile { data };
    let reader = MosaicReader::new(input, file_len).unwrap();

    // Verify stats exist and are sensible
    for rg in 0..reader.num_row_groups() {
        let stats = reader.row_group_stats(rg).unwrap();
        assert!(!stats.is_empty(), "stats should exist for row group {}", rg);
    }
}

// ============== Test: Concurrent/parallel read_ranges ==============
#[test]
fn test_parallel_read_ranges() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Utf8, true),
        Field::new("c", DataType::Float64, true),
        Field::new("d", DataType::Int32, true),
    ]);

    let num_rows = 1_000_000;
    let batch_size = 200_000;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let a: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();
        let b: Vec<Option<&str>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 3 == 0 {
                    None
                } else {
                    Some("data")
                }
            })
            .collect();
        let c: Vec<Option<f64>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 5 == 0 {
                    None
                } else {
                    Some(i as f64)
                }
            })
            .collect();
        let d: Vec<Option<i32>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 7 == 0 {
                    None
                } else {
                    Some(i as i32)
                }
            })
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(a)),
                Arc::new(StringArray::from(b)),
                Arc::new(Float64Array::from(c)),
                Arc::new(Int32Array::from(d)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 30,
            row_group_max_size: 8 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&batches, &result);
}

// ============== Test: Empty strings and binary ==============
#[test]
fn test_empty_strings_binary() {
    let schema = Schema::new(vec![
        Field::new("str_col", DataType::Utf8, true),
        Field::new("bin_col", DataType::Binary, true),
    ]);

    let num_rows = 100_000;
    let str_vals: Vec<Option<&str>> = (0..num_rows)
        .map(|i| match i % 5 {
            0 => None,
            1 => Some(""),
            2 => Some("a"),
            3 => Some("hello world this is a test string"),
            _ => Some("x"),
        })
        .collect();

    let bin_val_a: [u8; 0] = [];
    let bin_val_b: [u8; 1] = [0u8];
    let bin_val_c: [u8; 5] = [1, 2, 3, 4, 5];
    let bin_val_d: [u8; 1] = [255u8];
    let bin_vals: Vec<Option<&[u8]>> = (0..num_rows)
        .map(|i| match i % 5 {
            0 => None,
            1 => Some(bin_val_a.as_slice()),
            2 => Some(bin_val_b.as_slice()),
            3 => Some(bin_val_c.as_slice()),
            _ => Some(bin_val_d.as_slice()),
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(StringArray::from(str_vals)),
            Arc::new(BinaryArray::from(bin_vals)),
        ],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 2,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
}
