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

//! Data pattern tests: exercises various shapes of data to stress encoding,
//! compression, null handling, and roundtrip correctness in the Mosaic format.

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
                "mismatch at column '{}' rows {}..{}",
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

// ======================== 1. Sawtooth Pattern ========================

#[test]
fn test_sawtooth_pattern() {
    let num_rows = 500_000;
    let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);

    let vals: Vec<i32> = (0..num_rows).map(|i| (i % 100) as i32).collect();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vals))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_sawtooth_pattern: PASSED ({} rows)", num_rows);
}

// ======================== 2. Fibonacci-like Values ========================

#[test]
fn test_fibonacci_like_values() {
    let num_rows = 100_000;
    let modulus: i64 = 1_000_000_000;
    let schema = Schema::new(vec![Field::new("fib", DataType::Int64, false)]);

    let mut fib_vals: Vec<i64> = Vec::with_capacity(num_rows);
    if num_rows > 0 {
        fib_vals.push(0);
    }
    if num_rows > 1 {
        fib_vals.push(1);
    }
    for i in 2..num_rows {
        let next = (fib_vals[i - 1] + fib_vals[i - 2]) % modulus;
        fib_vals.push(next);
    }

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(fib_vals))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_fibonacci_like_values: PASSED ({} rows)", num_rows);
}

// ======================== 3. String Length Distribution ========================

#[test]
fn test_string_length_distribution() {
    let num_rows = 200_000;
    let schema = Schema::new(vec![Field::new("s", DataType::Utf8, false)]);

    let str_vals: Vec<String> = (0..num_rows)
        .map(|i| {
            let h = simple_hash(77, i);
            let pct = h % 100;
            let len = if pct < 50 {
                // 50%: 1-10 chars
                (simple_hash(78, i) % 10 + 1) as usize
            } else if pct < 80 {
                // 30%: 10-100 chars
                (simple_hash(79, i) % 91 + 10) as usize
            } else if pct < 95 {
                // 15%: 100-1000 chars
                (simple_hash(80, i) % 901 + 100) as usize
            } else {
                // 5%: 1000-5000 chars
                (simple_hash(81, i) % 4001 + 1000) as usize
            };
            (0..len)
                .map(|j| (b'a' + (simple_hash(82, i * 5000 + j) % 26) as u8) as char)
                .collect()
        })
        .collect();

    let str_refs: Vec<&str> = str_vals.iter().map(|s| s.as_str()).collect();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(StringArray::from(str_refs))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 64 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!(
        "test_string_length_distribution: PASSED ({} rows)",
        num_rows
    );
}

// ======================== 4. Correlated Null Columns ========================

#[test]
fn test_correlated_null_columns() {
    let num_rows = 500_000;
    let schema = Schema::new(vec![
        Field::new("col_a", DataType::Int32, true),
        Field::new("col_b", DataType::Int64, true),
        Field::new("col_c", DataType::Float64, true),
    ]);

    let a_vals: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            if simple_hash(10, i) % 5 == 0 {
                None
            } else {
                Some(i as i32)
            }
        })
        .collect();

    // When col_a is null, col_b is always null too
    let b_vals: Vec<Option<i64>> = (0..num_rows)
        .map(|i| {
            if a_vals[i].is_none() || simple_hash(20, i) % 7 == 0 {
                None
            } else {
                Some(i as i64 * 100)
            }
        })
        .collect();

    // col_c is independently nullable
    let c_vals: Vec<Option<f64>> = (0..num_rows)
        .map(|i| {
            if simple_hash(30, i) % 4 == 0 {
                None
            } else {
                Some(i as f64 * 0.5)
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(a_vals)),
            Arc::new(Int64Array::from(b_vals)),
            Arc::new(Float64Array::from(c_vals)),
        ],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 3,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_correlated_null_columns: PASSED ({} rows)", num_rows);
}

// ======================== 5. All Extremes Int ========================

#[test]
fn test_all_extremes_int() {
    let num_rows = 200_000;
    let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);

    let vals: Vec<i32> = (0..num_rows)
        .map(|i| {
            match simple_hash(50, i) % 6 {
                0 => i32::MIN,
                1 => i32::MAX,
                2 => 0,
                3 => -1,
                4 => 1,
                _ => simple_hash(51, i) as i32, // random
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vals))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_all_extremes_int: PASSED ({} rows)", num_rows);
}

// ======================== 6. Unicode Strings ========================

#[test]
fn test_unicode_strings() {
    let num_rows = 100_000;
    let schema = Schema::new(vec![Field::new("text", DataType::Utf8, false)]);

    let unicode_fragments: &[&str] = &[
        "\u{4e2d}\u{6587}\u{6d4b}\u{8bd5}",                 // Chinese
        "\u{65e5}\u{672c}\u{8a9e}\u{30c6}\u{30b9}\u{30c8}", // Japanese
        "\u{d55c}\u{ad6d}\u{c5b4}\u{d14c}\u{c2a4}\u{d2b8}", // Korean
        "\u{0627}\u{0644}\u{0639}\u{0631}\u{0628}\u{064a}\u{0629}", // Arabic
        "\u{1f600}\u{1f60d}\u{1f680}\u{1f4a5}\u{2764}\u{fe0f}", // Emoji
        "Hello\u{4e16}\u{754c}\u{d558}\u{c138}\u{c694}",    // Mixed scripts
        "\u{00e9}\u{00e8}\u{00ea}\u{00eb}\u{00fc}\u{00f6}", // European accents
        "\u{0410}\u{0411}\u{0412}\u{0413}\u{0414}",         // Cyrillic
        "\u{0e01}\u{0e02}\u{0e03}\u{0e04}\u{0e05}",         // Thai
        "\u{05d0}\u{05d1}\u{05d2}\u{05d3}\u{05d4}",         // Hebrew
    ];

    let str_vals: Vec<String> = (0..num_rows)
        .map(|i| {
            let idx = simple_hash(60, i) as usize % unicode_fragments.len();
            let repeat = (simple_hash(61, i) % 5 + 1) as usize;
            let base = unicode_fragments[idx];
            let mut s = String::new();
            for _ in 0..repeat {
                s.push_str(base);
            }
            // Add some unique suffix to vary cardinality
            s.push_str(&format!("_{}", i));
            s
        })
        .collect();

    let str_refs: Vec<&str> = str_vals.iter().map(|s| s.as_str()).collect();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(StringArray::from(str_refs))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_unicode_strings: PASSED ({} rows)", num_rows);
}

// ======================== 7. Zigzag Int Pattern ========================

#[test]
fn test_zigzag_int_pattern() {
    let num_rows = 500_000;
    let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);

    // 1,-1,2,-2,3,-3,...
    let vals: Vec<i32> = (0..num_rows)
        .map(|i| {
            let n = (i / 2 + 1) as i32;
            if i % 2 == 0 {
                n
            } else {
                -n
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vals))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_zigzag_int_pattern: PASSED ({} rows)", num_rows);
}

// ======================== 8. Power of Two Values ========================

#[test]
fn test_power_of_two_values() {
    let num_rows = 500_000;
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);

    // Powers of 2: 1, 2, 4, 8, ..., 2^62, then cycle
    let num_powers = 63; // 2^0 through 2^62
    let vals: Vec<i64> = (0..num_rows).map(|i| 1i64 << (i % num_powers)).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vals))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_power_of_two_values: PASSED ({} rows)", num_rows);
}

// ======================== 9. Sparse Data (99% null) ========================

#[test]
fn test_sparse_data() {
    let num_rows = 1_000_000;
    let schema = Schema::new(vec![
        Field::new("int_col", DataType::Int64, true),
        Field::new("str_col", DataType::Utf8, true),
        Field::new("bin_col", DataType::Binary, true),
    ]);

    let int_vals: Vec<Option<i64>> = (0..num_rows)
        .map(|i| {
            if simple_hash(90, i) % 100 == 0 {
                Some(i as i64)
            } else {
                None
            }
        })
        .collect();

    let str_owned: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if simple_hash(91, i) % 100 == 0 {
                Some(format!("sparse_{}", i))
            } else {
                None
            }
        })
        .collect();
    let str_refs: Vec<Option<&str>> = str_owned.iter().map(|s| s.as_deref()).collect();

    let bin_owned: Vec<Option<Vec<u8>>> = (0..num_rows)
        .map(|i| {
            if simple_hash(92, i) % 100 == 0 {
                Some(vec![(i % 256) as u8; 10])
            } else {
                None
            }
        })
        .collect();
    let bin_refs: Vec<Option<&[u8]>> = bin_owned
        .iter()
        .map(|b| b.as_ref().map(|v| v.as_slice()))
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(int_vals)),
            Arc::new(StringArray::from(str_refs)),
            Arc::new(BinaryArray::from(bin_refs)),
        ],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 3,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_sparse_data: PASSED ({} rows)", num_rows);
}

// ======================== 10. Dense Data No Nulls ========================

#[test]
fn test_dense_data_no_nulls() {
    let num_rows = 500_000;
    let schema = Schema::new(vec![
        Field::new("bool_col", DataType::Boolean, false),
        Field::new("i8_col", DataType::Int8, false),
        Field::new("i16_col", DataType::Int16, false),
        Field::new("i32_col", DataType::Int32, false),
        Field::new("i64_col", DataType::Int64, false),
        Field::new("f32_col", DataType::Float32, false),
        Field::new("f64_col", DataType::Float64, false),
        Field::new("str_col", DataType::Utf8, false),
        Field::new("bin_col", DataType::Binary, false),
        Field::new("date_col", DataType::Date32, false),
        Field::new("time_col", DataType::Time32(TimeUnit::Millisecond), false),
        Field::new("dec_col", DataType::Decimal128(10, 2), false),
        Field::new(
            "ts_col",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
    ]);

    let bools: Vec<bool> = (0..num_rows).map(|i| i % 2 == 0).collect();
    let i8s: Vec<i8> = (0..num_rows).map(|i| (i % 256) as i8).collect();
    let i16s: Vec<i16> = (0..num_rows).map(|i| (i % 30000) as i16).collect();
    let i32s: Vec<i32> = (0..num_rows).map(|i| i as i32).collect();
    let i64s: Vec<i64> = (0..num_rows).map(|i| i as i64 * 1000).collect();
    let f32s: Vec<f32> = (0..num_rows).map(|i| i as f32 * 0.1).collect();
    let f64s: Vec<f64> = (0..num_rows).map(|i| i as f64 * 0.001).collect();
    let str_vals: Vec<String> = (0..num_rows).map(|i| format!("row_{:06}", i)).collect();
    let str_refs: Vec<&str> = str_vals.iter().map(|s| s.as_str()).collect();
    let bin_vals: Vec<Vec<u8>> = (0..num_rows).map(|i| vec![(i % 256) as u8; 8]).collect();
    let bin_refs: Vec<&[u8]> = bin_vals.iter().map(|v| v.as_slice()).collect();
    let dates: Vec<i32> = (0..num_rows).map(|i| 18000 + (i % 3650) as i32).collect();
    let times: Vec<i32> = (0..num_rows)
        .map(|i| (i as i32 % 86_400_000).abs())
        .collect();
    let decimals: Vec<i128> = (0..num_rows).map(|i| i as i128 * 100).collect();
    let timestamps: Vec<i64> = (0..num_rows)
        .map(|i| 1_700_000_000_000i64 + i as i64)
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
            Arc::new(StringArray::from(str_refs)),
            Arc::new(BinaryArray::from(bin_refs)),
            Arc::new(Date32Array::from(dates)),
            Arc::new(Time32MillisecondArray::from(times)),
            Arc::new(
                Decimal128Array::from(decimals)
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
            Arc::new(TimestampMillisecondArray::from(timestamps)),
        ],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 10,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_dense_data_no_nulls: PASSED ({} rows)", num_rows);
}

// ======================== 11. Binary All Zeros ========================

#[test]
fn test_binary_all_zeros() {
    let num_rows = 100_000;
    let schema = Schema::new(vec![Field::new("data", DataType::Binary, false)]);

    let bin_vals: Vec<Vec<u8>> = (0..num_rows)
        .map(|i| {
            let len = (simple_hash(100, i) % 64 + 1) as usize;
            vec![0u8; len]
        })
        .collect();
    let bin_refs: Vec<&[u8]> = bin_vals.iter().map(|v| v.as_slice()).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(BinaryArray::from(bin_refs))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_binary_all_zeros: PASSED ({} rows)", num_rows);
}

// ======================== 12. Binary All 0xFF ========================

#[test]
fn test_binary_all_0xff() {
    let num_rows = 100_000;
    let schema = Schema::new(vec![Field::new("data", DataType::Binary, false)]);

    let bin_vals: Vec<Vec<u8>> = (0..num_rows)
        .map(|i| {
            let len = (simple_hash(110, i) % 64 + 1) as usize;
            vec![0xFFu8; len]
        })
        .collect();
    let bin_refs: Vec<&[u8]> = bin_vals.iter().map(|v| v.as_slice()).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(BinaryArray::from(bin_refs))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_binary_all_0xff: PASSED ({} rows)", num_rows);
}

// ======================== 13. String With Special Chars ========================

#[test]
fn test_string_with_special_chars() {
    let num_rows = 100_000;
    let schema = Schema::new(vec![Field::new("s", DataType::Utf8, false)]);

    let special_chars = ['\0', '\n', '\r', '\t', '\\', '"', '\''];

    let str_vals: Vec<String> = (0..num_rows)
        .map(|i| {
            let len = (simple_hash(120, i) % 50 + 1) as usize;
            (0..len)
                .map(|j| {
                    let h = simple_hash(121, i * 50 + j);
                    if h % 5 == 0 {
                        special_chars[(h as usize / 5) % special_chars.len()]
                    } else {
                        (b'a' + (h % 26) as u8) as char
                    }
                })
                .collect()
        })
        .collect();

    let str_refs: Vec<&str> = str_vals.iter().map(|s| s.as_str()).collect();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(StringArray::from(str_refs))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_string_with_special_chars: PASSED ({} rows)", num_rows);
}

// ======================== 14. Decimal Near Overflow ========================

#[test]
fn test_decimal_near_overflow() {
    let num_rows = 50_000;
    let schema = Schema::new(vec![Field::new("dec", DataType::Decimal128(38, 0), true)]);

    let max_decimal38: i128 = 99_999_999_999_999_999_999_999_999_999_999_999_999;

    let vals: Vec<Option<i128>> = (0..num_rows)
        .map(|i| match simple_hash(130, i) % 7 {
            0 => None,
            1 => Some(max_decimal38),
            2 => Some(-max_decimal38),
            3 => Some(max_decimal38 - i as i128),
            4 => Some(-max_decimal38 + i as i128),
            5 => Some(0),
            _ => Some((simple_hash(131, i) as i128) * 1_000_000_000_000_000_000),
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(
            Decimal128Array::from(vals)
                .with_precision_and_scale(38, 0)
                .unwrap(),
        )],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_decimal_near_overflow: PASSED ({} rows)", num_rows);
}

// ======================== 15. Timestamp Boundaries ========================

#[test]
fn test_timestamp_boundaries() {
    let num_rows = 100_000;
    let schema = Schema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Millisecond, None),
        true,
    )]);

    let epoch_0: i64 = 0;
    let year_1970: i64 = 0; // same as epoch 0
    let year_2000: i64 = 946_684_800_000;
    let year_2038: i64 = 2_145_916_800_000; // Unix Y2038
    let year_9999: i64 = 253_402_300_799_000;

    let boundary_vals = [epoch_0, year_1970, year_2000, year_2038, year_9999];

    let vals: Vec<Option<i64>> = (0..num_rows)
        .map(|i| {
            let h = simple_hash(140, i);
            if h % 10 == 0 {
                None
            } else if h % 3 == 0 {
                Some(boundary_vals[(h as usize) % boundary_vals.len()])
            } else {
                // Random timestamp between year 2000 and year 2038
                Some(year_2000 + (h as i64 % (year_2038 - year_2000)))
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(TimestampMillisecondArray::from(vals))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_timestamp_boundaries: PASSED ({} rows)", num_rows);
}

// ======================== 16. Monotonic Then Random ========================

#[test]
fn test_monotonic_then_random() {
    let half = 100_000;
    let num_rows = half * 2;
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);

    // First 100K: monotonically increasing
    let mut vals: Vec<i64> = (0..half as i64).collect();
    // Next 100K: random
    for i in 0..half {
        vals.push(simple_hash(150, i) as i64);
    }

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vals))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 512 * 1024, // small row groups to force encoding transitions
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_monotonic_then_random: PASSED ({} rows)", num_rows);
}

// ======================== 17. Single Value Then Varied ========================

#[test]
fn test_single_value_then_varied() {
    let half = 250_000;
    let num_rows = half * 2;
    let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);

    // First 250K: all value 42
    let mut vals: Vec<i32> = vec![42i32; half];
    // Next 250K: all different values
    for i in 0..half {
        vals.push(i as i32);
    }

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vals))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 512 * 1024, // small to force transitions
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_single_value_then_varied: PASSED ({} rows)", num_rows);
}

// ======================== 18. Run-Length Encoding Friendly ========================

#[test]
fn test_run_length_encoding_friendly() {
    let num_rows = 500_000;
    let run_length = 1000;
    let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);

    let vals: Vec<i32> = (0..num_rows).map(|i| (i / run_length) as i32).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vals))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!(
        "test_run_length_encoding_friendly: PASSED ({} rows)",
        num_rows
    );
}

// ======================== 19. High Cardinality Strings ========================

#[test]
fn test_high_cardinality_strings() {
    let num_rows = 200_000;
    let schema = Schema::new(vec![Field::new("uuid", DataType::Utf8, false)]);

    // Generate UUID-like strings: every string is unique
    let str_vals: Vec<String> = (0..num_rows)
        .map(|i| {
            let a = simple_hash(200, i);
            let b = simple_hash(201, i);
            let c = simple_hash(202, i);
            let d = simple_hash(203, i);
            format!(
                "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                (a & 0xFFFFFFFF) as u32,
                (b & 0xFFFF) as u16,
                (c & 0xFFFF) as u16,
                (d & 0xFFFF) as u16,
                a ^ b ^ c ^ d
            )
        })
        .collect();

    let str_refs: Vec<&str> = str_vals.iter().map(|s| s.as_str()).collect();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(StringArray::from(str_refs))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_high_cardinality_strings: PASSED ({} rows)", num_rows);
}

// ======================== 20. Low Cardinality With Skew ========================

#[test]
fn test_low_cardinality_with_skew() {
    let num_rows = 500_000;
    let schema = Schema::new(vec![Field::new("category", DataType::Utf8, false)]);

    // 5 distinct values with heavy skew:
    // 90% "A", 5% "B", 3% "C", 1.5% "D", 0.5% "E"
    let str_vals: Vec<&str> = (0..num_rows)
        .map(|i| {
            let h = simple_hash(210, i) % 1000;
            if h < 900 {
                "A"
            } else if h < 950 {
                "B"
            } else if h < 980 {
                "C"
            } else if h < 995 {
                "D"
            } else {
                "E"
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(StringArray::from(str_vals))],
    )
    .unwrap();

    let result = roundtrip(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );

    assert_batches_equal(&[batch], &result);
    println!("test_low_cardinality_with_skew: PASSED ({} rows)", num_rows);
}
