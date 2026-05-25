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

// ======================== Test Infrastructure ========================

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

fn simple_hash(seed: u64, i: usize) -> u64 {
    let mut x = seed.wrapping_add(i as u64);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
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

// ======================== Test 1: 10M sequential Int64 ========================

#[test]
fn test_10_million_int64_sequential() {
    let num_rows = 10_000_000usize;
    let batch_size = 500_000usize;
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]);

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;
        let ids: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();
        let values: Vec<i64> = (0..count).map(|i| (batch_start + i) as i64 * 7 + 3).collect();
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

    println!("test_10_million_int64_sequential: writing {} rows...", num_rows);

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 10,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    println!(
        "test_10_million_int64_sequential: verifying {} row groups...",
        result.len()
    );

    assert_batches_equal(&batches, &result);
    println!("test_10_million_int64_sequential: PASSED");
}

// ======================== Test 2: 5M mixed types random ========================

#[test]
fn test_5_million_mixed_types_random() {
    let num_rows = 5_000_000usize;
    let batch_size = 250_000usize;
    let schema = Schema::new(vec![
        Field::new("i32_col", DataType::Int32, true),
        Field::new("i64_col", DataType::Int64, true),
        Field::new("f64_col", DataType::Float64, true),
        Field::new("str_col", DataType::Utf8, true),
        Field::new("bool_col", DataType::Boolean, true),
        Field::new("bin_col", DataType::Binary, true),
    ]);

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let i32_vals: Vec<Option<i32>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                let h = simple_hash(1, idx);
                if h % 11 == 0 {
                    None
                } else {
                    Some(h as i32)
                }
            })
            .collect();

        let i64_vals: Vec<Option<i64>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                let h = simple_hash(2, idx);
                if h % 13 == 0 {
                    None
                } else {
                    Some(h as i64)
                }
            })
            .collect();

        let f64_vals: Vec<Option<f64>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                let h = simple_hash(3, idx);
                if h % 9 == 0 {
                    None
                } else {
                    Some(h as f64 / 1000.0)
                }
            })
            .collect();

        let str_vals: Vec<Option<String>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                let h = simple_hash(4, idx);
                if h % 7 == 0 {
                    None
                } else {
                    let len = (h % 20) as usize + 1;
                    let s: String = (0..len)
                        .map(|j| (b'a' + (simple_hash(5, idx * 20 + j) % 26) as u8) as char)
                        .collect();
                    Some(s)
                }
            })
            .collect();
        let str_refs: Vec<Option<&str>> = str_vals.iter().map(|s| s.as_deref()).collect();

        let bool_vals: Vec<Option<bool>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                let h = simple_hash(6, idx);
                if h % 8 == 0 {
                    None
                } else {
                    Some(h % 2 == 0)
                }
            })
            .collect();

        let bin_vals: Vec<Option<Vec<u8>>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                let h = simple_hash(7, idx);
                if h % 10 == 0 {
                    None
                } else {
                    let len = (h % 16) as usize + 1;
                    Some(
                        (0..len)
                            .map(|j| simple_hash(8, idx * 16 + j) as u8)
                            .collect(),
                    )
                }
            })
            .collect();
        let bin_refs: Vec<Option<&[u8]>> = bin_vals
            .iter()
            .map(|b| b.as_ref().map(|v| v.as_slice()))
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(i32_vals)),
                Arc::new(Int64Array::from(i64_vals)),
                Arc::new(Float64Array::from(f64_vals)),
                Arc::new(StringArray::from(str_refs)),
                Arc::new(BooleanArray::from(bool_vals)),
                Arc::new(BinaryArray::from(bin_refs)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    println!("test_5_million_mixed_types_random: writing {} rows...", num_rows);

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 15,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    println!(
        "test_5_million_mixed_types_random: verifying {} row groups...",
        result.len()
    );

    assert_batches_equal(&batches, &result);
    println!("test_5_million_mixed_types_random: PASSED");
}

// ======================== Test 3: 2M all same value (CONST encoding) ========================

#[test]
fn test_2_million_all_same_value() {
    let num_rows = 2_000_000usize;
    let batch_size = 500_000usize;
    let schema = Schema::new(vec![
        Field::new("const_i32", DataType::Int32, false),
        Field::new("const_i64", DataType::Int64, false),
        Field::new("const_f64", DataType::Float64, false),
        Field::new("const_str", DataType::Utf8, false),
        Field::new("const_bool", DataType::Boolean, false),
        Field::new("const_bin", DataType::Binary, false),
    ]);

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![42i32; count])),
                Arc::new(Int64Array::from(vec![999_999_999i64; count])),
                Arc::new(Float64Array::from(vec![3.14159; count])),
                Arc::new(StringArray::from(vec!["constant_value"; count])),
                Arc::new(BooleanArray::from(vec![true; count])),
                Arc::new(BinaryArray::from(vec![&[0xDE, 0xAD, 0xBE, 0xEF][..]; count])),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    println!("test_2_million_all_same_value: writing {} rows...", num_rows);

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 6,
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
        "test_2_million_all_same_value: file size = {} bytes ({} KB) for {} rows",
        file_size,
        file_size / 1024,
        num_rows
    );

    // CONST encoding should make this extremely small -- well under 1MB
    assert!(
        file_size < 1024 * 1024,
        "file size {} bytes is too large for constant columns (expected < 1MB)",
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
    println!("test_2_million_all_same_value: PASSED");
}

// ======================== Test 4: 1000 columns, 100K rows ========================

#[test]
fn test_1000_columns_100k_rows() {
    let num_cols = 1000usize;
    let num_rows = 100_000usize;
    let batch_size = 25_000usize;

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| Field::new(format!("c_{:04}", i), DataType::Int32, true))
        .collect();
    let schema = Schema::new(fields);

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let arrays: Vec<Arc<dyn Array>> = (0..num_cols)
            .map(|col| {
                let vals: Vec<Option<i32>> = (0..count)
                    .map(|i| {
                        let idx = batch_start + i;
                        let h = simple_hash(col as u64, idx);
                        if h % 13 == 0 {
                            None
                        } else {
                            Some(h as i32)
                        }
                    })
                    .collect();
                Arc::new(Int32Array::from(vals)) as Arc<dyn Array>
            })
            .collect();

        batches.push(RecordBatch::try_new(Arc::new(schema.clone()), arrays).unwrap());
    }

    println!(
        "test_1000_columns_100k_rows: writing {} cols x {} rows...",
        num_cols, num_rows
    );

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 100,
            row_group_max_size: 64 * 1024 * 1024,
            ..Default::default()
        },
    );

    println!(
        "test_1000_columns_100k_rows: verifying {} row groups...",
        result.len()
    );

    assert_batches_equal(&batches, &result);
    println!("test_1000_columns_100k_rows: PASSED");
}

// ======================== Test 5: 500K rows large strings ========================

#[test]
fn test_500k_rows_large_strings() {
    let num_rows = 500_000usize;
    let batch_size = 50_000usize;
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("big_str", DataType::Utf8, true),
    ]);

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let ids: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();

        let str_vals: Vec<Option<String>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                let h = simple_hash(10, idx);
                if h % 20 == 0 {
                    None
                } else {
                    // String length between 100 and 1000
                    let len = 100 + (h % 901) as usize;
                    let s: String = (0..len)
                        .map(|j| (b'A' + (simple_hash(11, idx * 1000 + j) % 26) as u8) as char)
                        .collect();
                    Some(s)
                }
            })
            .collect();
        let str_refs: Vec<Option<&str>> = str_vals.iter().map(|s| s.as_deref()).collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(str_refs)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    println!("test_500k_rows_large_strings: writing {} rows...", num_rows);

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 10,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    println!(
        "test_500k_rows_large_strings: verifying {} row groups...",
        result.len()
    );

    assert_batches_equal(&batches, &result);
    println!("test_500k_rows_large_strings: PASSED");
}

// ======================== Test 6: Monotonic sequential data ========================

#[test]
fn test_monotonic_sequential_data() {
    let num_rows = 3_000_000usize;
    let batch_size = 300_000usize;
    let schema = Schema::new(vec![
        Field::new("seq_i64", DataType::Int64, false),
        Field::new("seq_f64", DataType::Float64, false),
        Field::new("seq_date", DataType::Date32, false),
    ]);

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let i64_vals: Vec<i64> = (0..count)
            .map(|i| (batch_start + i) as i64 * 3 + 100)
            .collect();
        let f64_vals: Vec<f64> = (0..count)
            .map(|i| (batch_start + i) as f64 * 0.001 + 1.0)
            .collect();
        let date_vals: Vec<i32> = (0..count)
            .map(|i| 18000 + (batch_start + i) as i32)
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(i64_vals)),
                Arc::new(Float64Array::from(f64_vals)),
                Arc::new(Date32Array::from(date_vals)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    println!(
        "test_monotonic_sequential_data: writing {} rows...",
        num_rows
    );

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 10,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    println!(
        "test_monotonic_sequential_data: verifying {} row groups...",
        result.len()
    );

    assert_batches_equal(&batches, &result);
    println!("test_monotonic_sequential_data: PASSED");
}

// ======================== Test 7: Repeated pattern data ========================

#[test]
fn test_repeated_pattern_data() {
    let num_rows = 2_000_000usize;
    let batch_size = 250_000usize;
    let schema = Schema::new(vec![
        Field::new("pattern_7", DataType::Int32, false),
        Field::new("pattern_13", DataType::Utf8, false),
        Field::new("pattern_100", DataType::Int64, false),
    ]);

    let pattern_13_strs: Vec<String> = (0..13).map(|i| format!("pat_{:02}", i)).collect();

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let p7_vals: Vec<i32> = (0..count)
            .map(|i| ((batch_start + i) % 7) as i32 * 100 + 10)
            .collect();

        let p13_vals: Vec<&str> = (0..count)
            .map(|i| pattern_13_strs[(batch_start + i) % 13].as_str())
            .collect();

        let p100_vals: Vec<i64> = (0..count)
            .map(|i| ((batch_start + i) % 100) as i64 * 9999 + 1)
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(p7_vals)),
                Arc::new(StringArray::from(p13_vals)),
                Arc::new(Int64Array::from(p100_vals)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    println!("test_repeated_pattern_data: writing {} rows...", num_rows);

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 8,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    println!(
        "test_repeated_pattern_data: verifying {} row groups...",
        result.len()
    );

    assert_batches_equal(&batches, &result);
    println!("test_repeated_pattern_data: PASSED");
}

// ======================== Test 8: Alternating null data ========================

#[test]
fn test_alternating_null_data() {
    let num_rows = 1_000_000usize;
    let batch_size = 200_000usize;
    let schema = Schema::new(vec![
        Field::new("every_other", DataType::Int32, true),
        Field::new("every_3rd", DataType::Int64, true),
        Field::new("every_5th", DataType::Utf8, true),
        Field::new("every_10th", DataType::Float64, true),
        Field::new("every_100th", DataType::Boolean, true),
    ]);

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let every_other: Vec<Option<i32>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 2 == 0 {
                    None
                } else {
                    Some(idx as i32)
                }
            })
            .collect();

        let every_3rd: Vec<Option<i64>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 3 == 0 {
                    None
                } else {
                    Some(idx as i64 * 7)
                }
            })
            .collect();

        let every_5th_vals: Vec<Option<String>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 5 == 0 {
                    None
                } else {
                    Some(format!("v_{}", idx % 50))
                }
            })
            .collect();
        let every_5th_refs: Vec<Option<&str>> =
            every_5th_vals.iter().map(|s| s.as_deref()).collect();

        let every_10th: Vec<Option<f64>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 10 == 0 {
                    None
                } else {
                    Some(idx as f64 * 1.5)
                }
            })
            .collect();

        let every_100th: Vec<Option<bool>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 100 == 0 {
                    None
                } else {
                    Some(idx % 2 == 0)
                }
            })
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(every_other)),
                Arc::new(Int64Array::from(every_3rd)),
                Arc::new(StringArray::from(every_5th_refs)),
                Arc::new(Float64Array::from(every_10th)),
                Arc::new(BooleanArray::from(every_100th)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    println!("test_alternating_null_data: writing {} rows...", num_rows);

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 10,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    println!(
        "test_alternating_null_data: verifying {} row groups...",
        result.len()
    );

    assert_batches_equal(&batches, &result);
    println!("test_alternating_null_data: PASSED");
}

// ======================== Test 9: Massive binary data (~100MB raw) ========================

#[test]
fn test_massive_binary_data() {
    let num_rows = 200_000usize;
    let batch_size = 10_000usize;
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("blob", DataType::Binary, true),
    ]);

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let ids: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();

        let bin_vals: Vec<Option<Vec<u8>>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                let h = simple_hash(20, idx);
                if h % 25 == 0 {
                    None
                } else {
                    // Average ~500 bytes (range 200-800)
                    let len = 200 + (h % 601) as usize;
                    Some(
                        (0..len)
                            .map(|j| simple_hash(21, idx * 800 + j) as u8)
                            .collect(),
                    )
                }
            })
            .collect();
        let bin_refs: Vec<Option<&[u8]>> = bin_vals
            .iter()
            .map(|b| b.as_ref().map(|v| v.as_slice()))
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(BinaryArray::from(bin_refs)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    println!("test_massive_binary_data: writing {} rows...", num_rows);

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 5,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    println!(
        "test_massive_binary_data: verifying {} row groups...",
        result.len()
    );

    assert_batches_equal(&batches, &result);
    println!("test_massive_binary_data: PASSED");
}

// ======================== Test 10: 10M rows all types ========================

#[test]
fn test_all_types_10_million() {
    let num_rows = 10_000_000usize;
    let batch_size = 500_000usize;
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
        Field::new("dec_small", DataType::Decimal128(10, 2), true),
        Field::new("dec_large", DataType::Decimal128(38, 10), true),
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

    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let bools: Vec<Option<bool>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 10 == 0 {
                    None
                } else {
                    Some(idx % 2 == 0)
                }
            })
            .collect();

        let i8s: Vec<Option<i8>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 15 == 0 {
                    None
                } else {
                    Some((idx % 256) as i8)
                }
            })
            .collect();

        let i16s: Vec<Option<i16>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 12 == 0 {
                    None
                } else {
                    Some((idx % 30000) as i16)
                }
            })
            .collect();

        let i32s: Vec<Option<i32>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 9 == 0 {
                    None
                } else {
                    Some(idx as i32 * 3)
                }
            })
            .collect();

        let i64s: Vec<Option<i64>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 8 == 0 {
                    None
                } else {
                    Some(idx as i64 * 1_000_000)
                }
            })
            .collect();

        let f32s: Vec<Option<f32>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 14 == 0 {
                    None
                } else {
                    Some(idx as f32 * 0.1)
                }
            })
            .collect();

        let f64s: Vec<Option<f64>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 11 == 0 {
                    None
                } else {
                    Some(idx as f64 * 3.14159)
                }
            })
            .collect();

        let dates: Vec<Option<i32>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 20 == 0 {
                    None
                } else {
                    Some(18000 + idx as i32 % 3650)
                }
            })
            .collect();

        let times: Vec<Option<i32>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 25 == 0 {
                    None
                } else {
                    Some((idx as i32 % 86_400_000).abs())
                }
            })
            .collect();

        let strings: Vec<Option<String>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 6 == 0 {
                    None
                } else {
                    Some(format!("r_{:08}", idx))
                }
            })
            .collect();
        let str_refs: Vec<Option<&str>> = strings.iter().map(|s| s.as_deref()).collect();

        let binaries: Vec<Option<Vec<u8>>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 7 == 0 {
                    None
                } else {
                    let len = idx % 32 + 1;
                    Some((0..len).map(|j| ((idx + j) % 256) as u8).collect())
                }
            })
            .collect();
        let bin_refs: Vec<Option<&[u8]>> = binaries
            .iter()
            .map(|b| b.as_ref().map(|v| v.as_slice()))
            .collect();

        let dec_small: Vec<Option<i128>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 13 == 0 {
                    None
                } else {
                    Some(idx as i128 * 100 + 99)
                }
            })
            .collect();

        let dec_large: Vec<Option<i128>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 17 == 0 {
                    None
                } else {
                    Some(idx as i128 * 10_000_000_000i128 + 123_456_789)
                }
            })
            .collect();

        let ts_ms: Vec<Option<i64>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 16 == 0 {
                    None
                } else {
                    Some(1_700_000_000_000i64 + idx as i64 * 1000)
                }
            })
            .collect();

        let ts_us: Vec<Option<i64>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 18 == 0 {
                    None
                } else {
                    Some(1_700_000_000_000_000i64 + idx as i64)
                }
            })
            .collect();

        let ts_ns: Vec<Option<i64>> = (0..count)
            .map(|i| {
                let idx = batch_start + i;
                if idx % 21 == 0 {
                    None
                } else {
                    let millis = 1_700_000_000_000i64 + idx as i64;
                    let nanos = (idx % 1_000_000) as i32;
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
                    Decimal128Array::from(dec_small)
                        .with_precision_and_scale(10, 2)
                        .unwrap(),
                ),
                Arc::new(
                    Decimal128Array::from(dec_large)
                        .with_precision_and_scale(38, 10)
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

    println!("test_all_types_10_million: writing {} rows...", num_rows);

    let result = roundtrip(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 20,
            row_group_max_size: 64 * 1024 * 1024,
            ..Default::default()
        },
    );

    println!(
        "test_all_types_10_million: verifying {} row groups...",
        result.len()
    );

    assert_batches_equal(&batches, &result);
    println!("test_all_types_10_million: PASSED");
}
