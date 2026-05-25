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

//! Comprehensive tests for encoding strategies, compression behavior,
//! projection correctness, and writer options interactions.

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
use paimon_mosaic_core::spec;
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

fn write_file(schema: &Schema, batches: &[RecordBatch], options: WriterOptions) -> Vec<u8> {
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(out, schema, options).unwrap();
    for batch in batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();
    writer.output().buf.clone()
}

fn read_all(data: &[u8]) -> Vec<RecordBatch> {
    let input = ByteArrayInputFile {
        data: data.to_vec(),
    };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();
    let mut result = Vec::new();
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader(rg).unwrap();
        result.push(rg_reader.read_columns().unwrap());
    }
    result
}

fn concat_column_i32(batches: &[RecordBatch], col_name: &str) -> Vec<Option<i32>> {
    let mut out = Vec::new();
    for batch in batches {
        let idx = batch.schema().index_of(col_name).unwrap();
        let arr = batch
            .column(idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..arr.len() {
            if arr.is_null(i) {
                out.push(None);
            } else {
                out.push(Some(arr.value(i)));
            }
        }
    }
    out
}

fn concat_column_i64(batches: &[RecordBatch], col_name: &str) -> Vec<Option<i64>> {
    let mut out = Vec::new();
    for batch in batches {
        let idx = batch.schema().index_of(col_name).unwrap();
        let arr = batch
            .column(idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..arr.len() {
            if arr.is_null(i) {
                out.push(None);
            } else {
                out.push(Some(arr.value(i)));
            }
        }
    }
    out
}

fn concat_column_string(batches: &[RecordBatch], col_name: &str) -> Vec<Option<String>> {
    let mut out = Vec::new();
    for batch in batches {
        let idx = batch.schema().index_of(col_name).unwrap();
        let arr = batch
            .column(idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..arr.len() {
            if arr.is_null(i) {
                out.push(None);
            } else {
                out.push(Some(arr.value(i).to_string()));
            }
        }
    }
    out
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
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

// ======================== Encoding Tests ========================

// 1. test_const_encoding_all_types
#[test]
fn test_const_encoding_all_types() {
    let num_rows = 100_000;

    // Boolean constant
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Boolean, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(BooleanArray::from(vec![true; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        assert!(
            data.len() < 4096,
            "Boolean const file too large: {} bytes",
            data.len()
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Int8 constant
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Int8, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int8Array::from(vec![42i8; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        assert!(
            data.len() < 4096,
            "Int8 const file too large: {} bytes",
            data.len()
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Int16 constant
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Int16, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int16Array::from(vec![1234i16; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        assert!(
            data.len() < 4096,
            "Int16 const file too large: {} bytes",
            data.len()
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Int32 constant
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int32Array::from(vec![99999i32; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        assert!(
            data.len() < 4096,
            "Int32 const file too large: {} bytes",
            data.len()
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Int64 constant
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![
                999_999_999_999i64;
                num_rows
            ]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        assert!(
            data.len() < 4096,
            "Int64 const file too large: {} bytes",
            data.len()
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Float32 constant
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Float32, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Float32Array::from(vec![3.14f32; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        assert!(
            data.len() < 4096,
            "Float32 const file too large: {} bytes",
            data.len()
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Float64 constant
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Float64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Float64Array::from(vec![2.71828f64; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        assert!(
            data.len() < 4096,
            "Float64 const file too large: {} bytes",
            data.len()
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Utf8 constant
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Utf8, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(StringArray::from(vec!["hello_world"; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        assert!(
            data.len() < 4096,
            "Utf8 const file too large: {} bytes",
            data.len()
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Binary constant
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Binary, false)]);
        let val: &[u8] = b"constant_binary_value";
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(BinaryArray::from(vec![val; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        assert!(
            data.len() < 4096,
            "Binary const file too large: {} bytes",
            data.len()
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Date32 constant
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Date32, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Date32Array::from(vec![19000i32; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        assert!(
            data.len() < 4096,
            "Date32 const file too large: {} bytes",
            data.len()
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Time32 constant
    {
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::Time32(TimeUnit::Millisecond),
            false,
        )]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Time32MillisecondArray::from(vec![
                3_600_000i32;
                num_rows
            ]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        assert!(
            data.len() < 4096,
            "Time32 const file too large: {} bytes",
            data.len()
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Decimal128 constant
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Decimal128(10, 2), false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(
                Decimal128Array::from(vec![12345i128; num_rows])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            )],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        assert!(
            data.len() < 4096,
            "Decimal128 const file too large: {} bytes",
            data.len()
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    println!("test_const_encoding_all_types: PASSED - all types produce small const-encoded files");
}

// 2. test_dict_encoding_boundary_253_254_255_256
#[test]
fn test_dict_encoding_boundary_253_254_255_256() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);

    for num_distinct in [253, 254, 255, 256] {
        let num_rows = 50_000;
        let vals: Vec<i32> = (0..num_rows).map(|i| (i % num_distinct) as i32).collect();
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int32Array::from(vals.clone()))],
        )
        .unwrap();

        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );

        let result = read_all(&data);
        let out_vals: Vec<i32> = result
            .iter()
            .flat_map(|b| {
                let idx = b.schema().index_of("v").unwrap();
                b.column(idx)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(
            out_vals, vals,
            "roundtrip failed for {} distinct values",
            num_distinct
        );
        println!(
            "  dict boundary {}: file size = {} bytes, roundtrip OK",
            num_distinct,
            data.len()
        );
    }

    println!("test_dict_encoding_boundary_253_254_255_256: PASSED");
}

// 3. test_dict_encoding_string_cardinality
#[test]
fn test_dict_encoding_string_cardinality() {
    for num_distinct in [5, 50, 100, 200, 255] {
        let num_rows = 100_000;
        let schema = Schema::new(vec![Field::new("v", DataType::Utf8, false)]);

        // Generate distinct string values
        let distinct_vals: Vec<String> = (0..num_distinct)
            .map(|i| format!("str_value_{:05}", i))
            .collect();

        let str_vals: Vec<&str> = (0..num_rows)
            .map(|i| distinct_vals[i % num_distinct].as_str())
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(StringArray::from(str_vals.clone()))],
        )
        .unwrap();

        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );

        let result = read_all(&data);
        let out_vals = concat_column_string(&result, "v");
        assert_eq!(out_vals.len(), num_rows);
        for (i, v) in out_vals.iter().enumerate() {
            assert_eq!(
                v.as_deref(),
                Some(str_vals[i]),
                "mismatch at row {} for cardinality {}",
                i,
                num_distinct
            );
        }
        println!(
            "  string cardinality {}: file size = {} bytes, roundtrip OK",
            num_distinct,
            data.len()
        );
    }

    println!("test_dict_encoding_string_cardinality: PASSED");
}

// 4. test_dict_budget_exceeded
#[test]
fn test_dict_budget_exceeded() {
    let schema = Schema::new(vec![Field::new("v", DataType::Utf8, false)]);

    // Create strings that are each 120+ bytes, with enough distinct values
    // to blow past max_dict_total_bytes=1024
    let num_distinct = 20; // 20 * 120 = 2400 bytes > 1024
    let num_rows = 10_000;

    let distinct_vals: Vec<String> = (0..num_distinct)
        .map(|i| format!("long_string_value_{:0>100}", i))
        .collect();
    let str_vals: Vec<&str> = (0..num_rows)
        .map(|i| distinct_vals[i % num_distinct].as_str())
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(StringArray::from(str_vals.clone()))],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            max_dict_total_bytes: 1024,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    // Verify the data still roundtrips correctly (fallback to plain)
    let result = read_all(&data);
    let out_vals = concat_column_string(&result, "v");
    assert_eq!(out_vals.len(), num_rows);
    for (i, v) in out_vals.iter().enumerate() {
        assert_eq!(v.as_deref(), Some(str_vals[i]), "mismatch at row {}", i);
    }

    println!(
        "test_dict_budget_exceeded: PASSED - file size = {} bytes, data roundtrips",
        data.len()
    );
}

// 5. test_all_null_encoding
#[test]
fn test_all_null_encoding() {
    let num_rows = 100_000;

    // Boolean all null
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Boolean, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(BooleanArray::from(vec![None::<bool>; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
        // All values should be null
        for b in &result {
            let idx = b.schema().index_of("v").unwrap();
            let arr = b
                .column(idx)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap();
            assert_eq!(arr.null_count(), arr.len());
        }
    }

    // Int32 all null
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Int32, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int32Array::from(vec![None::<i32>; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
        for b in &result {
            let idx = b.schema().index_of("v").unwrap();
            let arr = b.column(idx).as_any().downcast_ref::<Int32Array>().unwrap();
            assert_eq!(arr.null_count(), arr.len());
        }
    }

    // Int64 all null
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Int64, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![None::<i64>; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Float64 all null
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Float64, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Float64Array::from(vec![None::<f64>; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Utf8 all null
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Utf8, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(StringArray::from(vec![None::<&str>; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Binary all null
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Binary, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(BinaryArray::from(vec![None::<&[u8]>; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Date32 all null
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Date32, true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Date32Array::from(vec![None::<i32>; num_rows]))],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    // Decimal128 all null
    {
        let schema = Schema::new(vec![Field::new("v", DataType::Decimal128(10, 2), true)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(
                Decimal128Array::from(vec![None::<i128>; num_rows])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            )],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        );
        let result = read_all(&data);
        assert_eq!(total_rows(&result), num_rows);
    }

    println!("test_all_null_encoding: PASSED - all types with all null values roundtrip");
}

// 6. test_encoding_mixed_in_same_file
#[test]
fn test_encoding_mixed_in_same_file() {
    let num_rows = 200_000;
    let schema = Schema::new(vec![
        Field::new("const_col", DataType::Int64, false), // CONST: single value
        Field::new("dict_col", DataType::Utf8, false),   // DICT: few distinct values
        Field::new("plain_col", DataType::Int64, false), // PLAIN: all unique values
        Field::new("null_col", DataType::Float64, true), // ALL_NULL
    ]);

    let batch_size = 50_000;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let const_vals: Vec<i64> = vec![42i64; count];
        let dict_vals: Vec<&str> = (0..count)
            .map(|i| match (batch_start + i) % 5 {
                0 => "alpha",
                1 => "beta",
                2 => "gamma",
                3 => "delta",
                _ => "epsilon",
            })
            .collect();
        let plain_vals: Vec<i64> = (batch_start..batch_start + count)
            .map(|i| i as i64)
            .collect();
        let null_vals: Vec<Option<f64>> = vec![None; count];

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(const_vals)),
                Arc::new(StringArray::from(dict_vals)),
                Arc::new(Int64Array::from(plain_vals)),
                Arc::new(Float64Array::from(null_vals)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let data = write_file(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let result = read_all(&data);
    assert_batches_equal(&batches, &result);

    println!(
        "test_encoding_mixed_in_same_file: PASSED - file size = {} bytes, {} row groups",
        data.len(),
        result.len()
    );
}

// ======================== Compression Tests ========================

// 7. test_no_compression_roundtrip_all_types
#[test]
fn test_no_compression_roundtrip_all_types() {
    let num_rows = 200_000;
    let batch_size = 50_000;
    let schema = Schema::new(vec![
        Field::new("bool_col", DataType::Boolean, true),
        Field::new("i8_col", DataType::Int8, true),
        Field::new("i16_col", DataType::Int16, true),
        Field::new("i32_col", DataType::Int32, true),
        Field::new("i64_col", DataType::Int64, true),
        Field::new("f32_col", DataType::Float32, true),
        Field::new("f64_col", DataType::Float64, true),
        Field::new("str_col", DataType::Utf8, true),
        Field::new("bin_col", DataType::Binary, true),
        Field::new("date_col", DataType::Date32, true),
        Field::new("time_col", DataType::Time32(TimeUnit::Millisecond), true),
        Field::new("dec_col", DataType::Decimal128(10, 2), true),
    ]);

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
        let decimals: Vec<Option<i128>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 13 == 0 {
                    None
                } else {
                    Some((batch_start + i) as i128 * 100 + 99)
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
                Arc::new(StringArray::from(str_refs)),
                Arc::new(BinaryArray::from(bin_refs)),
                Arc::new(Date32Array::from(dates)),
                Arc::new(Time32MillisecondArray::from(times)),
                Arc::new(
                    Decimal128Array::from(decimals)
                        .with_precision_and_scale(10, 2)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let data = write_file(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 4,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let result = read_all(&data);
    assert_batches_equal(&batches, &result);

    println!(
        "test_no_compression_roundtrip_all_types: PASSED - file size = {} bytes",
        data.len()
    );
}

// 8. test_zstd_compression_levels
#[test]
fn test_zstd_compression_levels() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, false),
    ]);

    let num_rows = 50_000;
    let ids: Vec<i64> = (0..num_rows as i64).collect();
    let strs: Vec<String> = (0..num_rows)
        .map(|i| format!("record_{:08}_padding_{}", i, "x".repeat(20)))
        .collect();
    let str_refs: Vec<&str> = strs.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids.clone())),
            Arc::new(StringArray::from(str_refs)),
        ],
    )
    .unwrap();

    let levels = [1, 3, 7, 15, 22];
    let mut sizes = Vec::new();

    for &level in &levels {
        let data = write_file(
            &schema,
            &[batch.clone()],
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_ZSTD,
                zstd_level: level,
                ..Default::default()
            },
        );

        // Verify roundtrip
        let result = read_all(&data);
        let out_ids = concat_column_i64(&result, "id");
        assert_eq!(out_ids.len(), num_rows);
        for (i, v) in out_ids.iter().enumerate() {
            assert_eq!(
                *v,
                Some(ids[i]),
                "id mismatch at row {} for level {}",
                i,
                level
            );
        }

        sizes.push((level, data.len()));
        println!("  zstd level {}: file size = {} bytes", level, data.len());
    }

    // Higher levels should generally produce smaller or equal files
    // (not strictly guaranteed but usually true for this kind of data)
    println!("test_zstd_compression_levels: PASSED - all levels roundtrip correctly");
}

// 9. test_compression_highly_compressible
#[test]
fn test_compression_highly_compressible() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);

    let num_rows = 200_000;
    let vals: Vec<i64> = (0..num_rows as i64).collect();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vals.clone()))],
    )
    .unwrap();

    let data_none = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let data_zstd = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_ZSTD,
            ..Default::default()
        },
    );

    // Verify both roundtrip
    let result_none = read_all(&data_none);
    let result_zstd = read_all(&data_zstd);
    assert_eq!(total_rows(&result_none), num_rows);
    assert_eq!(total_rows(&result_zstd), num_rows);

    let ratio = data_none.len() as f64 / data_zstd.len() as f64;
    println!(
        "  NONE: {} bytes, ZSTD: {} bytes, ratio: {:.2}x",
        data_none.len(),
        data_zstd.len(),
        ratio
    );
    assert!(
        ratio > 2.0,
        "expected compression ratio > 2x for sequential integers, got {:.2}x",
        ratio
    );

    println!("test_compression_highly_compressible: PASSED");
}

// 10. test_compression_incompressible
#[test]
fn test_compression_incompressible() {
    let schema = Schema::new(vec![Field::new("v", DataType::Binary, false)]);

    let num_rows = 10_000;
    let bin_vals: Vec<Vec<u8>> = (0..num_rows)
        .map(|i| (0..64).map(|j| simple_hash(42, i * 64 + j) as u8).collect())
        .collect();
    let bin_refs: Vec<&[u8]> = bin_vals.iter().map(|v| v.as_slice()).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(BinaryArray::from(bin_refs))],
    )
    .unwrap();

    let data_none = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let data_zstd = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_ZSTD,
            ..Default::default()
        },
    );

    // Verify both roundtrip
    let result_none = read_all(&data_none);
    let result_zstd = read_all(&data_zstd);
    assert_eq!(total_rows(&result_none), num_rows);
    assert_eq!(total_rows(&result_zstd), num_rows);

    // ZSTD should not dramatically inflate random data
    // Allow up to 10% overhead
    let max_allowed = (data_none.len() as f64 * 1.10) as usize;
    assert!(
        data_zstd.len() <= max_allowed,
        "ZSTD inflated random data too much: NONE={}, ZSTD={}, max_allowed={}",
        data_none.len(),
        data_zstd.len(),
        max_allowed
    );

    println!(
        "test_compression_incompressible: PASSED - NONE: {} bytes, ZSTD: {} bytes",
        data_none.len(),
        data_zstd.len()
    );
}

// 11. test_compression_empty_strings
#[test]
fn test_compression_empty_strings() {
    let schema = Schema::new(vec![Field::new("v", DataType::Utf8, false)]);

    let num_rows = 100_000;
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(StringArray::from(vec![""; num_rows]))],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_ZSTD,
            ..Default::default()
        },
    );

    let result = read_all(&data);
    assert_eq!(total_rows(&result), num_rows);

    // Verify all empty strings come back
    let out_vals = concat_column_string(&result, "v");
    for (i, v) in out_vals.iter().enumerate() {
        assert_eq!(v.as_deref(), Some(""), "row {} not empty string", i);
    }

    println!(
        "test_compression_empty_strings: PASSED - file size = {} bytes",
        data.len()
    );
}

// ======================== Projection Tests ========================

// 12. test_projection_single_column_from_100
#[test]
fn test_projection_single_column_from_100() {
    let num_cols = 100;
    let num_rows = 10_000;

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| Field::new(format!("col_{:04}", i), DataType::Int32, false))
        .collect();
    let schema = Schema::new(fields);

    let mut arrays: Vec<Arc<dyn Array>> = Vec::new();
    for col in 0..num_cols {
        let vals: Vec<i32> = (0..num_rows).map(|i| (col * 10000 + i) as i32).collect();
        arrays.push(Arc::new(Int32Array::from(vals)));
    }

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), arrays).unwrap();
    let file_data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 10,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    // Read only column 50
    let target_col = 50;
    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();

    let mut all_vals: Vec<i32> = Vec::new();
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader
            .row_group_reader_projected(rg, &[target_col])
            .unwrap();
        let result = rg_reader.read_columns().unwrap();
        assert_eq!(result.num_columns(), 1);
        let col_name = format!("col_{:04}", target_col);
        let idx = result.schema().index_of(&col_name).unwrap();
        let arr = result
            .column(idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        all_vals.extend(arr.values().iter().copied());
    }

    let expected: Vec<i32> = (0..num_rows)
        .map(|i| (target_col * 10000 + i) as i32)
        .collect();
    assert_eq!(all_vals, expected);

    println!("test_projection_single_column_from_100: PASSED");
}

// 13. test_projection_last_column
#[test]
fn test_projection_last_column() {
    let num_cols = 50;
    let num_rows = 10_000;

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| Field::new(format!("col_{:04}", i), DataType::Int64, false))
        .collect();
    let schema = Schema::new(fields);

    let mut arrays: Vec<Arc<dyn Array>> = Vec::new();
    for col in 0..num_cols {
        let vals: Vec<i64> = (0..num_rows).map(|i| (col * 100000 + i) as i64).collect();
        arrays.push(Arc::new(Int64Array::from(vals)));
    }

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), arrays).unwrap();
    let file_data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 5,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    // Read only the last column
    let last_col = num_cols - 1;
    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();

    let mut all_vals: Vec<i64> = Vec::new();
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader_projected(rg, &[last_col]).unwrap();
        let result = rg_reader.read_columns().unwrap();
        assert_eq!(result.num_columns(), 1);
        let col_name = format!("col_{:04}", last_col);
        let idx = result.schema().index_of(&col_name).unwrap();
        let arr = result
            .column(idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        all_vals.extend(arr.values().iter().copied());
    }

    let expected: Vec<i64> = (0..num_rows)
        .map(|i| (last_col * 100000 + i) as i64)
        .collect();
    assert_eq!(all_vals, expected);

    println!("test_projection_last_column: PASSED");
}

// 14. test_projection_first_column
#[test]
fn test_projection_first_column() {
    let num_cols = 50;
    let num_rows = 10_000;

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| Field::new(format!("col_{:04}", i), DataType::Int64, false))
        .collect();
    let schema = Schema::new(fields);

    let mut arrays: Vec<Arc<dyn Array>> = Vec::new();
    for col in 0..num_cols {
        let vals: Vec<i64> = (0..num_rows).map(|i| (col * 100000 + i) as i64).collect();
        arrays.push(Arc::new(Int64Array::from(vals)));
    }

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), arrays).unwrap();
    let file_data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 5,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    // Read only the first column
    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();

    let mut all_vals: Vec<i64> = Vec::new();
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader_projected(rg, &[0]).unwrap();
        let result = rg_reader.read_columns().unwrap();
        assert_eq!(result.num_columns(), 1);
        let col_name = "col_0000";
        let idx = result.schema().index_of(col_name).unwrap();
        let arr = result
            .column(idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        all_vals.extend(arr.values().iter().copied());
    }

    let expected: Vec<i64> = (0..num_rows).map(|i| i as i64).collect();
    assert_eq!(all_vals, expected);

    println!("test_projection_first_column: PASSED");
}

// 15. test_projection_reverse_order
#[test]
fn test_projection_reverse_order() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int64, false),
        Field::new("c", DataType::Utf8, false),
        Field::new("d", DataType::Float64, false),
    ]);

    let num_rows = 10_000;
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from((0..num_rows as i32).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(
                (0..num_rows as i64).map(|i| i * 100).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..num_rows)
                    .map(|i| format!("s_{}", i))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (0..num_rows).map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();

    let file_data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 4,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    // Project in reverse order [3, 2, 1, 0]
    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();

    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader
            .row_group_reader_projected(rg, &[3, 2, 1, 0])
            .unwrap();
        let result = rg_reader.read_columns().unwrap();
        assert_eq!(result.num_columns(), 4);

        // All four columns should be present
        assert!(result.schema().index_of("a").is_ok());
        assert!(result.schema().index_of("b").is_ok());
        assert!(result.schema().index_of("c").is_ok());
        assert!(result.schema().index_of("d").is_ok());
    }

    println!("test_projection_reverse_order: PASSED");
}

// 16. test_projection_same_column_twice
#[test]
fn test_projection_same_column_twice() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int64, false),
    ]);

    let num_rows = 5_000;
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from((0..num_rows as i32).collect::<Vec<_>>())),
            Arc::new(Int64Array::from((0..num_rows as i64).collect::<Vec<_>>())),
        ],
    )
    .unwrap();

    let file_data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();

    // Project [0, 0] - same column twice. The implementation marks projected[0]=true
    // which deduplicates, so we get the column once. This is implementation-defined behavior.
    for rg in 0..reader.num_row_groups() {
        let rg_result = reader.row_group_reader_projected(rg, &[0, 0]);
        match rg_result {
            Ok(mut rg_reader) => {
                let result = rg_reader.read_columns().unwrap();
                // Should at least contain column "a"
                assert!(result.schema().index_of("a").is_ok());
                assert!(result.num_rows() > 0);
            }
            Err(_) => {
                // Some implementations may reject duplicate indices
                println!("  duplicate projection indices rejected (acceptable)");
            }
        }
    }

    println!("test_projection_same_column_twice: PASSED");
}

// 17. test_projection_by_name
#[test]
fn test_projection_by_name() {
    let schema = Schema::new(vec![
        Field::new("alpha", DataType::Int32, false),
        Field::new("beta", DataType::Utf8, false),
        Field::new("gamma", DataType::Float64, false),
        Field::new("delta", DataType::Int64, false),
    ]);

    let num_rows = 10_000;
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from((0..num_rows as i32).collect::<Vec<_>>())),
            Arc::new(StringArray::from(
                (0..num_rows)
                    .map(|i| format!("val_{}", i))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (0..num_rows).map(|i| i as f64).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from((0..num_rows as i64).collect::<Vec<_>>())),
        ],
    )
    .unwrap();

    let file_data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 4,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();

    // Project by names: only "beta" and "delta"
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader
            .row_group_reader_by_names(rg, &["beta", "delta"])
            .unwrap();
        let result = rg_reader.read_columns().unwrap();
        assert_eq!(result.num_columns(), 2);
        assert!(result.schema().index_of("beta").is_ok());
        assert!(result.schema().index_of("delta").is_ok());
    }

    println!("test_projection_by_name: PASSED");
}

// 18. test_projection_with_all_encodings
#[test]
fn test_projection_with_all_encodings() {
    let num_rows = 50_000;
    let schema = Schema::new(vec![
        Field::new("const_col", DataType::Int32, false), // CONST
        Field::new("dict_col", DataType::Utf8, false),   // DICT
        Field::new("null_col", DataType::Float64, true), // ALL_NULL
        Field::new("plain_col", DataType::Int64, false), // PLAIN
    ]);

    let const_vals: Vec<i32> = vec![42; num_rows];
    let dict_vals: Vec<&str> = (0..num_rows)
        .map(|i| match i % 3 {
            0 => "foo",
            1 => "bar",
            _ => "baz",
        })
        .collect();
    let null_vals: Vec<Option<f64>> = vec![None; num_rows];
    let plain_vals: Vec<i64> = (0..num_rows as i64).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(const_vals)),
            Arc::new(StringArray::from(dict_vals)),
            Arc::new(Float64Array::from(null_vals)),
            Arc::new(Int64Array::from(plain_vals.clone())),
        ],
    )
    .unwrap();

    let file_data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();

    // Project only CONST column
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader
            .row_group_reader_by_names(rg, &["const_col"])
            .unwrap();
        let result = rg_reader.read_columns().unwrap();
        assert_eq!(result.num_columns(), 1);
        let idx = result.schema().index_of("const_col").unwrap();
        let arr = result
            .column(idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..arr.len() {
            assert_eq!(arr.value(i), 42);
        }
    }

    // Project only PLAIN column
    let input2 = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader2 = MosaicReader::new(input2, file_data.len() as u64).unwrap();
    for rg in 0..reader2.num_row_groups() {
        let mut rg_reader = reader2
            .row_group_reader_by_names(rg, &["plain_col"])
            .unwrap();
        let result = rg_reader.read_columns().unwrap();
        assert_eq!(result.num_columns(), 1);
    }

    // Project CONST + ALL_NULL
    let input3 = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader3 = MosaicReader::new(input3, file_data.len() as u64).unwrap();
    for rg in 0..reader3.num_row_groups() {
        let mut rg_reader = reader3
            .row_group_reader_by_names(rg, &["const_col", "null_col"])
            .unwrap();
        let result = rg_reader.read_columns().unwrap();
        assert_eq!(result.num_columns(), 2);
    }

    println!("test_projection_with_all_encodings: PASSED");
}

// 19. test_projection_across_multiple_row_groups
#[test]
fn test_projection_across_multiple_row_groups() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8, false),
        Field::new("c", DataType::Int64, false),
    ]);

    let num_rows = 100_000;
    let batch_size = 1000;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let a_vals: Vec<i32> = (batch_start..batch_start + count)
            .map(|i| i as i32)
            .collect();
        let b_vals: Vec<String> = (batch_start..batch_start + count)
            .map(|i| format!("s_{:06}", i))
            .collect();
        let b_refs: Vec<&str> = b_vals.iter().map(|s| s.as_str()).collect();
        let c_vals: Vec<i64> = (batch_start..batch_start + count)
            .map(|i| i as i64 * 1000)
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(a_vals)),
                Arc::new(StringArray::from(b_refs)),
                Arc::new(Int64Array::from(c_vals)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    // Write with small row group size to force many row groups
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 32 * 1024, // 32KB
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    )
    .unwrap();
    for batch in &batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();
    let file_data = writer.output().buf.clone();

    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();

    assert!(
        reader.num_row_groups() > 1,
        "expected multiple row groups, got {}",
        reader.num_row_groups()
    );

    // Project different columns for different row groups
    let mut a_total = 0usize;
    let mut c_total = 0usize;

    for rg in 0..reader.num_row_groups() {
        if rg % 2 == 0 {
            // Even row groups: project "a"
            let mut rg_reader = reader.row_group_reader_by_names(rg, &["a"]).unwrap();
            let result = rg_reader.read_columns().unwrap();
            assert_eq!(result.num_columns(), 1);
            assert!(result.schema().index_of("a").is_ok());
            a_total += result.num_rows();
        } else {
            // Odd row groups: project "c"
            let mut rg_reader = reader.row_group_reader_by_names(rg, &["c"]).unwrap();
            let result = rg_reader.read_columns().unwrap();
            assert_eq!(result.num_columns(), 1);
            assert!(result.schema().index_of("c").is_ok());
            c_total += result.num_rows();
        }
    }

    assert_eq!(a_total + c_total, num_rows);
    println!(
        "test_projection_across_multiple_row_groups: PASSED - {} row groups, a_rows={}, c_rows={}",
        reader.num_row_groups(),
        a_total,
        c_total
    );
}

// ======================== Writer Options Interactions ========================

// 20. test_bucket_count_1
#[test]
fn test_bucket_count_1() {
    let num_rows = 200_000;
    let schema = Schema::new(vec![
        Field::new("i32_col", DataType::Int32, true),
        Field::new("i64_col", DataType::Int64, false),
        Field::new("str_col", DataType::Utf8, true),
        Field::new("f64_col", DataType::Float64, false),
    ]);

    let batch_size = 50_000;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let i32_vals: Vec<Option<i32>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 7 == 0 {
                    None
                } else {
                    Some((batch_start + i) as i32)
                }
            })
            .collect();
        let i64_vals: Vec<i64> = (0..count).map(|i| (batch_start + i) as i64 * 100).collect();
        let str_vals: Vec<Option<String>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 5 == 0 {
                    None
                } else {
                    Some(format!("val_{}", (batch_start + i) % 50))
                }
            })
            .collect();
        let str_refs: Vec<Option<&str>> = str_vals.iter().map(|s| s.as_deref()).collect();
        let f64_vals: Vec<f64> = (0..count)
            .map(|i| (batch_start + i) as f64 * 0.01)
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(i32_vals)),
                Arc::new(Int64Array::from(i64_vals)),
                Arc::new(StringArray::from(str_refs)),
                Arc::new(Float64Array::from(f64_vals)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let data = write_file(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();
    assert_eq!(reader.schema().num_buckets, 1);

    let result = read_all(&data);
    assert_batches_equal(&batches, &result);

    println!(
        "test_bucket_count_1: PASSED - file size = {} bytes",
        data.len()
    );
}

// 21. test_bucket_count_equals_columns
#[test]
fn test_bucket_count_equals_columns() {
    let num_cols = 10;
    let num_rows = 10_000;

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| Field::new(format!("col_{:02}", i), DataType::Int32, false))
        .collect();
    let schema = Schema::new(fields);

    let mut arrays: Vec<Arc<dyn Array>> = Vec::new();
    for col in 0..num_cols {
        let vals: Vec<i32> = (0..num_rows).map(|i| (col * 10000 + i) as i32).collect();
        arrays.push(Arc::new(Int32Array::from(vals)));
    }

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), arrays).unwrap();

    let data = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: num_cols,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();
    assert_eq!(reader.schema().num_buckets, num_cols);

    let result = read_all(&data);
    assert_batches_equal(&[batch], &result);

    println!("test_bucket_count_equals_columns: PASSED");
}

// 22. test_bucket_count_exceeds_columns
#[test]
fn test_bucket_count_exceeds_columns() {
    let num_cols = 10;
    let num_rows = 5_000;

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| Field::new(format!("col_{:02}", i), DataType::Int32, false))
        .collect();
    let schema = Schema::new(fields);

    let mut arrays: Vec<Arc<dyn Array>> = Vec::new();
    for col in 0..num_cols {
        let vals: Vec<i32> = (0..num_rows).map(|i| (col * 10000 + i) as i32).collect();
        arrays.push(Arc::new(Int32Array::from(vals)));
    }

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), arrays).unwrap();

    let data = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 500, // Way more than 10 columns
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();
    // Should be capped to num_cols
    assert_eq!(
        reader.schema().num_buckets,
        num_cols,
        "num_buckets should be capped to column count"
    );

    let result = read_all(&data);
    assert_batches_equal(&[batch], &result);

    println!("test_bucket_count_exceeds_columns: PASSED");
}

// 23. test_tiny_row_group_max
#[test]
fn test_tiny_row_group_max() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, false),
    ]);

    let num_rows = 50_000;
    let batch_size = 1_000;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let ids: Vec<i64> = (batch_start..batch_start + count)
            .map(|i| i as i64)
            .collect();
        let strs: Vec<String> = (batch_start..batch_start + count)
            .map(|i| format!("row_{:06}_pad", i))
            .collect();
        let str_refs: Vec<&str> = strs.iter().map(|s| s.as_str()).collect();

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

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 1024, // 1KB - very small
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    )
    .unwrap();
    for batch in &batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();
    let file_data = writer.output().buf.clone();

    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();

    // Should have many row groups due to tiny max size
    println!(
        "  tiny row group: {} row groups created",
        reader.num_row_groups()
    );
    assert!(
        reader.num_row_groups() > 5,
        "expected many row groups with 1KB max, got {}",
        reader.num_row_groups()
    );

    // Verify all data reads back
    let result = read_all(&file_data);
    assert_eq!(total_rows(&result), num_rows);

    // Verify data correctness
    let all_ids = concat_column_i64(&result, "id");
    for (i, v) in all_ids.iter().enumerate() {
        assert_eq!(*v, Some(i as i64), "id mismatch at row {}", i);
    }

    println!("test_tiny_row_group_max: PASSED");
}

// 24. test_huge_row_group_max
#[test]
fn test_huge_row_group_max() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, false),
    ]);

    let num_rows = 50_000;
    let ids: Vec<i64> = (0..num_rows as i64).collect();
    let strs: Vec<String> = (0..num_rows).map(|i| format!("row_{:06}", i)).collect();
    let str_refs: Vec<&str> = strs.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(str_refs)),
        ],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 1_073_741_824, // 1GB
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();
    assert_eq!(
        reader.num_row_groups(),
        1,
        "expected single row group with 1GB max"
    );

    let result = read_all(&data);
    assert_eq!(total_rows(&result), num_rows);

    println!("test_huge_row_group_max: PASSED");
}

// 25. test_page_size_threshold_small
#[test]
fn test_page_size_threshold_small() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, false),
    ]);

    let num_rows = 200_000;
    let batch_size = 50_000;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let ids: Vec<i64> = (batch_start..batch_start + count)
            .map(|i| i as i64)
            .collect();
        let strs: Vec<String> = (0..count)
            .map(|i| {
                let h = simple_hash(99, batch_start + i);
                format!("data_{:08}_{}", batch_start + i, h % 1000)
            })
            .collect();
        let str_refs: Vec<&str> = strs.iter().map(|s| s.as_str()).collect();

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

    let data = write_file(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 1,
            page_size_threshold: 256, // Very small, forces paged encoding
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let result = read_all(&data);
    assert_batches_equal(&batches, &result);

    println!(
        "test_page_size_threshold_small: PASSED - file size = {} bytes",
        data.len()
    );
}

// 26. test_page_size_threshold_large
#[test]
fn test_page_size_threshold_large() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, false),
    ]);

    let num_rows = 50_000;
    let ids: Vec<i64> = (0..num_rows as i64).collect();
    let strs: Vec<String> = (0..num_rows).map(|i| format!("row_{:08}", i)).collect();
    let str_refs: Vec<&str> = strs.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(str_refs)),
        ],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            page_size_threshold: 1_048_576, // 1MB - monolithic
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let result = read_all(&data);
    assert_batches_equal(&[batch], &result);

    println!(
        "test_page_size_threshold_large: PASSED - file size = {} bytes",
        data.len()
    );
}

// 27. test_stats_all_columns
#[test]
fn test_stats_all_columns() {
    let schema = Schema::new(vec![
        Field::new("i32_col", DataType::Int32, true),
        Field::new("i64_col", DataType::Int64, true),
        Field::new("f64_col", DataType::Float64, true),
        Field::new("str_col", DataType::Utf8, true),
        Field::new("date_col", DataType::Date32, true),
    ]);

    let num_rows = 10_000;
    let i32_vals: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            if i % 10 == 0 {
                None
            } else {
                Some(i as i32 - 5000)
            }
        })
        .collect();
    let i64_vals: Vec<Option<i64>> = (0..num_rows)
        .map(|i| {
            if i % 8 == 0 {
                None
            } else {
                Some(i as i64 * 100)
            }
        })
        .collect();
    let f64_vals: Vec<Option<f64>> = (0..num_rows)
        .map(|i| {
            if i % 12 == 0 {
                None
            } else {
                Some(i as f64 * 0.01)
            }
        })
        .collect();
    let str_owned: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if i % 5 == 0 {
                None
            } else {
                Some(format!("val_{:04}", i % 100))
            }
        })
        .collect();
    let str_refs: Vec<Option<&str>> = str_owned.iter().map(|s| s.as_deref()).collect();
    let date_vals: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            if i % 15 == 0 {
                None
            } else {
                Some(18000 + i as i32 % 365)
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(i32_vals)),
            Arc::new(Int64Array::from(i64_vals)),
            Arc::new(Float64Array::from(f64_vals)),
            Arc::new(StringArray::from(str_refs)),
            Arc::new(Date32Array::from(date_vals)),
        ],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            stats_columns: vec![
                "i32_col".to_string(),
                "i64_col".to_string(),
                "f64_col".to_string(),
                "str_col".to_string(),
                "date_col".to_string(),
            ],
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();

    let data = writer.output().buf.clone();
    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();

    for rg in 0..reader.num_row_groups() {
        let stats = reader.row_group_stats(rg).unwrap();
        assert!(
            !stats.is_empty(),
            "rg={}: expected stats for all columns",
            rg
        );
        println!("  rg={}: {} stats entries", rg, stats.len());

        // Verify stats have sensible null counts
        for stat in stats {
            // null_count should be non-negative and less than total rows
            let rg_rows = reader.row_group_num_rows(rg).unwrap();
            assert!(
                stat.null_count <= rg_rows,
                "null_count {} exceeds row count {} for column {}",
                stat.null_count,
                rg_rows,
                stat.column_index
            );
        }
    }

    println!("test_stats_all_columns: PASSED");
}

// 28. test_stats_subset_columns
#[test]
fn test_stats_subset_columns() {
    let num_cols = 10;
    let num_rows = 10_000;

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| Field::new(format!("col_{:02}", i), DataType::Int32, true))
        .collect();
    let schema = Schema::new(fields);

    let mut arrays: Vec<Arc<dyn Array>> = Vec::new();
    for col in 0..num_cols {
        let vals: Vec<Option<i32>> = (0..num_rows)
            .map(|i| {
                if i % (col + 3) == 0 {
                    None
                } else {
                    Some((col * 10000 + i) as i32)
                }
            })
            .collect();
        arrays.push(Arc::new(Int32Array::from(vals)));
    }

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), arrays).unwrap();

    // Enable stats for only col_02 and col_07
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            stats_columns: vec!["col_02".to_string(), "col_07".to_string()],
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();

    let data = writer.output().buf.clone();
    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();

    for rg in 0..reader.num_row_groups() {
        let stats = reader.row_group_stats(rg).unwrap();
        // Should have stats for exactly 2 columns
        assert_eq!(
            stats.len(),
            2,
            "expected stats for 2 columns, got {}",
            stats.len()
        );
    }

    // Also verify the data roundtrips
    let result = read_all(&data);
    assert_eq!(total_rows(&result), num_rows);

    println!("test_stats_subset_columns: PASSED");
}

// ======================== Row Group Boundary Tests ========================

// 29. test_exact_row_group_boundary
#[test]
fn test_exact_row_group_boundary() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, false),
    ]);

    // Write enough data to hit near the row group boundary.
    // We use many small batches so the writer can split at batch boundaries.
    let num_rows = 20_000;
    let batch_size = 100;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let ids: Vec<i64> = (batch_start..batch_start + count)
            .map(|i| i as i64)
            .collect();
        // Each row is about 20 bytes of string data
        let strs: Vec<String> = (0..count)
            .map(|i| format!("data_{:010}", batch_start + i))
            .collect();
        let str_refs: Vec<&str> = strs.iter().map(|s| s.as_str()).collect();

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

    // Use row_group_max_size that creates a boundary somewhere in the middle
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024, // 16KB
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    )
    .unwrap();
    for batch in &batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();
    let file_data = writer.output().buf.clone();

    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();

    // Verify all data reads back in order
    let result = read_all(&file_data);
    let all_ids = concat_column_i64(&result, "id");
    assert_eq!(all_ids.len(), num_rows);
    for (i, v) in all_ids.iter().enumerate() {
        assert_eq!(*v, Some(i as i64), "id mismatch at row {}", i);
    }

    println!(
        "test_exact_row_group_boundary: PASSED - {} row groups",
        reader.num_row_groups()
    );
}

// 30. test_data_split_across_many_row_groups
#[test]
fn test_data_split_across_many_row_groups() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int32, false),
        Field::new("label", DataType::Utf8, false),
    ]);

    let num_rows = 200_000;
    let batch_size = 500;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;

        let ids: Vec<i64> = (batch_start..batch_start + count)
            .map(|i| i as i64)
            .collect();
        let vals: Vec<i32> = (0..count)
            .map(|i| simple_hash(77, batch_start + i) as i32)
            .collect();
        let labels: Vec<String> = (0..count)
            .map(|i| format!("label_{:08}_{}", batch_start + i, "x".repeat(30)))
            .collect();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int32Array::from(vals)),
                Arc::new(StringArray::from(label_refs)),
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
            num_buckets: 1,
            row_group_max_size: 8 * 1024, // 8KB - very small to force many row groups
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    )
    .unwrap();
    for batch in &batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();
    let file_data = writer.output().buf.clone();

    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();

    println!(
        "  data split: {} row groups created",
        reader.num_row_groups()
    );
    assert!(
        reader.num_row_groups() >= 50,
        "expected 50+ row groups with 8KB max, got {}",
        reader.num_row_groups()
    );

    // Read all row groups and verify concatenated data
    let result = read_all(&file_data);
    let total = total_rows(&result);
    assert_eq!(total, num_rows, "total rows mismatch");

    // Verify IDs are consecutive
    let all_ids = concat_column_i64(&result, "id");
    for (i, v) in all_ids.iter().enumerate() {
        assert_eq!(*v, Some(i as i64), "id mismatch at row {}", i);
    }

    // Verify values match expected hash
    let all_vals = concat_column_i32(&result, "value");
    for (i, v) in all_vals.iter().enumerate() {
        let expected = simple_hash(77, i) as i32;
        assert_eq!(*v, Some(expected), "value mismatch at row {}", i);
    }

    println!("test_data_split_across_many_row_groups: PASSED");
}
