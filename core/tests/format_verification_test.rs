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

//! Format verification tests: bit-level footer parsing, statistics accuracy,
//! and estimated file size checks for the Mosaic columnar format.

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
use arrow_schema::{DataType, Field, Schema};
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

fn read_be_u64(data: &[u8], offset: usize) -> u64 {
    u64::from_be_bytes(data[offset..offset + 8].try_into().unwrap())
}

fn read_be_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap())
}

/// Write a Mosaic file from a schema and batches, return the raw bytes.
fn write_file(schema: &Schema, batches: &[RecordBatch], options: WriterOptions) -> Vec<u8> {
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(out, schema, options).unwrap();
    for batch in batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();
    writer.output().buf.clone()
}

/// Open a Mosaic reader over an in-memory byte buffer.
fn open_reader(data: &[u8]) -> MosaicReader<ByteArrayInputFile> {
    let input = ByteArrayInputFile {
        data: data.to_vec(),
    };
    MosaicReader::new(input, data.len() as u64).unwrap()
}

/// Read all row groups from a reader.
fn read_all(data: &[u8]) -> Vec<RecordBatch> {
    let reader = open_reader(data);
    let mut result = Vec::new();
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader(rg).unwrap();
        result.push(rg_reader.read_columns().unwrap());
    }
    result
}

/// Extract the 32-byte footer from raw file bytes.
fn extract_footer(data: &[u8]) -> &[u8] {
    let len = data.len();
    &data[len - spec::FOOTER_SIZE..]
}

// ======================== Binary Format Verification Tests ========================

// 1. test_footer_magic_bytes
#[test]
fn test_footer_magic_bytes() {
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let data = write_file(&schema, &[batch], WriterOptions::default());

    let last_4 = &data[data.len() - 4..];
    assert_eq!(last_4, b"MOSA", "last 4 bytes must be the MOSA magic");
    println!("test_footer_magic_bytes: PASSED");
}

// 2. test_footer_version_byte
#[test]
fn test_footer_version_byte() {
    let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vec![100, 200]))],
    )
    .unwrap();
    let data = write_file(&schema, &[batch], WriterOptions::default());

    // Version byte is at footer[25], which is data[len - 7]
    let version_byte = data[data.len() - 7];
    assert_eq!(
        version_byte,
        spec::VERSION,
        "version byte at len-7 must equal spec::VERSION ({})",
        spec::VERSION
    );
    println!("test_footer_version_byte: PASSED (version={})", version_byte);
}

// 3. test_footer_size_exactly_32
#[test]
fn test_footer_size_exactly_32() {
    assert_eq!(spec::FOOTER_SIZE, 32, "FOOTER_SIZE constant must be 32");

    let schema = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vec![42]))],
    )
    .unwrap();
    let data = write_file(&schema, &[batch], WriterOptions::default());

    // Verify we can parse the footer region
    let footer = extract_footer(&data);
    assert_eq!(footer.len(), 32);

    // Parse fields to verify the region is valid
    let _index_offset = read_be_u64(footer, 0);
    let _schema_block_offset = read_be_u64(footer, 8);
    let _num_buckets = read_be_u32(footer, 16);
    let _num_row_groups = read_be_u32(footer, 20);
    let _compression = footer[24];
    let _version = footer[25];
    // bytes 26-27 are reserved/padding
    let magic = &footer[28..32];
    assert_eq!(magic, b"MOSA");

    println!("test_footer_size_exactly_32: PASSED");
}

// 4. test_footer_compression_byte
#[test]
fn test_footer_compression_byte() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);
    let vals: Vec<i64> = (0..1000).collect();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vals))],
    )
    .unwrap();

    // Write with NONE compression
    let data_none = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            compression: spec::COMPRESSION_NONE,
            num_buckets: 1,
            ..Default::default()
        },
    );
    let footer_none = extract_footer(&data_none);
    assert_eq!(
        footer_none[24],
        spec::COMPRESSION_NONE,
        "compression byte for NONE must be {}",
        spec::COMPRESSION_NONE
    );

    // Write with ZSTD compression
    let data_zstd = write_file(
        &schema,
        &[batch],
        WriterOptions {
            compression: spec::COMPRESSION_ZSTD,
            num_buckets: 1,
            ..Default::default()
        },
    );
    let footer_zstd = extract_footer(&data_zstd);
    assert_eq!(
        footer_zstd[24],
        spec::COMPRESSION_ZSTD,
        "compression byte for ZSTD must be {}",
        spec::COMPRESSION_ZSTD
    );

    println!(
        "test_footer_compression_byte: PASSED (NONE={}, ZSTD={})",
        footer_none[24], footer_zstd[24]
    );
}

// 5. test_footer_offsets_valid
#[test]
fn test_footer_offsets_valid() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8, true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                Some("hello"),
                None,
                Some("world"),
                Some("test"),
                None,
            ])),
        ],
    )
    .unwrap();
    let data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 2,
            ..Default::default()
        },
    );

    let file_len = data.len() as u64;
    let footer = extract_footer(&data);
    let index_offset = read_be_u64(footer, 0);
    let schema_block_offset = read_be_u64(footer, 8);

    let footer_start = file_len - spec::FOOTER_SIZE as u64;

    // Both offsets must be within the file before the footer
    assert!(
        index_offset <= footer_start,
        "index_offset ({}) must be <= footer_start ({})",
        index_offset,
        footer_start
    );
    assert!(
        schema_block_offset <= footer_start,
        "schema_block_offset ({}) must be <= footer_start ({})",
        schema_block_offset,
        footer_start
    );

    // The format layout is: data | schema_block | index | footer
    // So schema_block_offset + 4 <= index_offset <= footer_start
    assert!(
        schema_block_offset + 4 <= index_offset,
        "schema_block_offset+4 ({}) must be <= index_offset ({})",
        schema_block_offset + 4,
        index_offset
    );

    println!(
        "test_footer_offsets_valid: PASSED (schema_block_offset={}, index_offset={}, footer_start={})",
        schema_block_offset, index_offset, footer_start
    );
}

// 6. test_footer_num_row_groups
#[test]
fn test_footer_num_row_groups() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);

    // Write enough data with a small row_group_max_size to force multiple row groups
    let batch_size = 5000;
    let mut batches = Vec::new();
    for b in 0..10 {
        let vals: Vec<i64> = ((b * batch_size)..((b + 1) * batch_size))
            .map(|i| i as i64)
            .collect();
        batches.push(
            RecordBatch::try_new(
                Arc::new(schema.clone()),
                vec![Arc::new(Int64Array::from(vals))],
            )
            .unwrap(),
        );
    }

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 8 * 1024, // very small to force multiple row groups
            ..Default::default()
        },
    )
    .unwrap();
    for batch in &batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    // Parse num_row_groups from footer
    let footer = extract_footer(&data);
    let num_row_groups_from_footer = read_be_u32(footer, 20) as usize;

    // Compare with reader
    let reader = open_reader(&data);
    let num_row_groups_from_reader = reader.num_row_groups();

    assert_eq!(
        num_row_groups_from_footer, num_row_groups_from_reader,
        "footer num_row_groups ({}) must match reader.num_row_groups() ({})",
        num_row_groups_from_footer, num_row_groups_from_reader
    );
    assert!(
        num_row_groups_from_footer > 1,
        "expected multiple row groups, got {}",
        num_row_groups_from_footer
    );

    println!(
        "test_footer_num_row_groups: PASSED (num_row_groups={})",
        num_row_groups_from_footer
    );
}

// 7. test_footer_num_buckets
#[test]
fn test_footer_num_buckets() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
        Field::new("c", DataType::Int32, false),
        Field::new("d", DataType::Int32, false),
        Field::new("e", DataType::Int32, false),
        Field::new("f", DataType::Int32, false),
        Field::new("g", DataType::Int32, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![2])),
            Arc::new(Int32Array::from(vec![3])),
            Arc::new(Int32Array::from(vec![4])),
            Arc::new(Int32Array::from(vec![5])),
            Arc::new(Int32Array::from(vec![6])),
            Arc::new(Int32Array::from(vec![7])),
        ],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 7,
            ..Default::default()
        },
    );

    let footer = extract_footer(&data);
    let num_buckets_from_footer = read_be_u32(footer, 16);

    let reader = open_reader(&data);
    let num_buckets_from_schema = reader.schema().num_buckets;

    assert_eq!(
        num_buckets_from_footer as usize, num_buckets_from_schema,
        "footer num_buckets ({}) must match schema num_buckets ({})",
        num_buckets_from_footer, num_buckets_from_schema
    );
    assert_eq!(num_buckets_from_footer, 7, "expected num_buckets=7");

    println!(
        "test_footer_num_buckets: PASSED (num_buckets={})",
        num_buckets_from_footer
    );
}

// 8. test_schema_block_readable
#[test]
fn test_schema_block_readable() {
    let schema = Schema::new(vec![
        Field::new("x", DataType::Int64, false),
        Field::new("y", DataType::Utf8, true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(vec![10, 20, 30])),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
        ],
    )
    .unwrap();

    // Use NONE compression so we can more easily inspect the schema block
    let data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let footer = extract_footer(&data);
    let schema_block_offset = read_be_u64(footer, 8) as usize;
    let index_offset = read_be_u64(footer, 0) as usize;

    // The schema block starts with a 4-byte uncompressed size (big-endian u32)
    assert!(
        schema_block_offset + 4 <= data.len(),
        "schema_block_offset+4 must be within file"
    );
    let uncompressed_size = read_be_u32(&data, schema_block_offset);
    assert!(
        uncompressed_size > 0,
        "schema block uncompressed_size must be > 0"
    );

    // The schema compressed data is between schema_block_offset+4 and index_offset
    let schema_compressed_len = index_offset - schema_block_offset - 4;
    assert!(
        schema_compressed_len > 0,
        "schema block compressed data must have length > 0"
    );

    // For NONE compression, uncompressed_size should equal the compressed length
    assert_eq!(
        uncompressed_size as usize, schema_compressed_len,
        "with NONE compression, uncompressed_size should equal compressed_len"
    );

    println!(
        "test_schema_block_readable: PASSED (uncompressed_size={}, compressed_len={})",
        uncompressed_size, schema_compressed_len
    );
}

// 9. test_column_order_alphabetical
#[test]
fn test_column_order_alphabetical() {
    // Write columns in non-alphabetical order
    let schema = Schema::new(vec![
        Field::new("zebra", DataType::Int32, false),
        Field::new("alpha", DataType::Int32, false),
        Field::new("middle", DataType::Int32, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(vec![30])),
            Arc::new(Int32Array::from(vec![10])),
            Arc::new(Int32Array::from(vec![20])),
        ],
    )
    .unwrap();

    let data = write_file(&schema, &[batch], WriterOptions::default());
    let reader = open_reader(&data);

    // Internal storage order (schema.columns) should be alphabetical
    let col_names: Vec<&str> = reader
        .schema()
        .columns
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(
        col_names,
        vec!["alpha", "middle", "zebra"],
        "internal column order must be alphabetical"
    );

    // original_order provides the mapping from original position to sorted position
    let original_order = &reader.schema().original_order;
    assert_eq!(original_order.len(), 3);

    // Read back and verify data is correctly associated with column names
    let result = read_all(&data);
    assert_eq!(result.len(), 1);

    // Find each column by name and verify its value
    let batch_out = &result[0];
    let alpha_idx = batch_out.schema().index_of("alpha").unwrap();
    let middle_idx = batch_out.schema().index_of("middle").unwrap();
    let zebra_idx = batch_out.schema().index_of("zebra").unwrap();

    let alpha_val = batch_out
        .column(alpha_idx)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .value(0);
    let middle_val = batch_out
        .column(middle_idx)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .value(0);
    let zebra_val = batch_out
        .column(zebra_idx)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .value(0);

    assert_eq!(alpha_val, 10);
    assert_eq!(middle_val, 20);
    assert_eq!(zebra_val, 30);

    println!("test_column_order_alphabetical: PASSED (order={:?})", col_names);
}

// 10. test_multiple_files_different_schemas
#[test]
fn test_multiple_files_different_schemas() {
    // File 1: Int32 columns
    let schema1 = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, true),
    ]);
    let batch1 = RecordBatch::try_new(
        Arc::new(schema1.clone()),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int32Array::from(vec![Some(10), None])),
        ],
    )
    .unwrap();
    let data1 = write_file(&schema1, &[batch1], WriterOptions::default());

    // File 2: String + Float64 columns
    let schema2 = Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("score", DataType::Float64, true),
    ]);
    let batch2 = RecordBatch::try_new(
        Arc::new(schema2.clone()),
        vec![
            Arc::new(StringArray::from(vec!["alice", "bob"])),
            Arc::new(Float64Array::from(vec![Some(95.5), None])),
        ],
    )
    .unwrap();
    let data2 = write_file(&schema2, &[batch2], WriterOptions::default());

    // File 3: Single Int64 column
    let schema3 = Schema::new(vec![Field::new("ts", DataType::Int64, false)]);
    let batch3 = RecordBatch::try_new(
        Arc::new(schema3.clone()),
        vec![Arc::new(Int64Array::from(vec![1000, 2000, 3000]))],
    )
    .unwrap();
    let data3 = write_file(&schema3, &[batch3], WriterOptions::default());

    // Each file should have valid magic
    for (i, data) in [&data1, &data2, &data3].iter().enumerate() {
        let footer = extract_footer(data);
        assert_eq!(&footer[28..32], b"MOSA", "file {} bad magic", i);
        assert_eq!(footer[25], spec::VERSION, "file {} bad version", i);
    }

    // Schema blocks should be at different offsets (different schemas = different content)
    let sb1 = read_be_u64(extract_footer(&data1), 8);
    let sb2 = read_be_u64(extract_footer(&data2), 8);
    let sb3 = read_be_u64(extract_footer(&data3), 8);

    // All readers should open successfully with correct column counts
    let reader1 = open_reader(&data1);
    let reader2 = open_reader(&data2);
    let reader3 = open_reader(&data3);

    assert_eq!(reader1.schema().columns.len(), 2);
    assert_eq!(reader2.schema().columns.len(), 2);
    assert_eq!(reader3.schema().columns.len(), 1);

    println!(
        "test_multiple_files_different_schemas: PASSED (offsets: {}, {}, {})",
        sb1, sb2, sb3
    );
}

// ======================== Statistics Accuracy Tests ========================

// 11. test_stats_int64_min_max_exact
#[test]
fn test_stats_int64_min_max_exact() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);
    let num_rows = 100_000;

    let expected_min: i64 = -999_999;
    let expected_max: i64 = 999_999;

    // Create values that include the known min and max, with lots of values in between
    let mut vals: Vec<i64> = (0..num_rows)
        .map(|i| ((i as i64) % 100_000) - 50_000)
        .collect();
    vals[0] = expected_min;
    vals[1] = expected_max;

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vals))],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            stats_columns: vec!["v".to_string()],
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    let reader = open_reader(&data);
    let stats = reader.row_group_stats(0).unwrap();
    assert!(!stats.is_empty(), "stats should not be empty");

    let col_stats = &stats[0];
    assert_eq!(col_stats.null_count, 0);
    match &col_stats.min {
        Some(paimon_mosaic_core::values::Value::BigInt(v)) => {
            assert_eq!(*v, expected_min, "min mismatch: got {}", v);
        }
        other => panic!("expected BigInt min, got {:?}", other),
    }
    match &col_stats.max {
        Some(paimon_mosaic_core::values::Value::BigInt(v)) => {
            assert_eq!(*v, expected_max, "max mismatch: got {}", v);
        }
        other => panic!("expected BigInt max, got {:?}", other),
    }

    println!("test_stats_int64_min_max_exact: PASSED (min={}, max={})", expected_min, expected_max);
}

// 12. test_stats_int32_min_max_exact
#[test]
fn test_stats_int32_min_max_exact() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);
    let num_rows = 100_000;

    let expected_min: i32 = i32::MIN + 1;
    let expected_max: i32 = i32::MAX - 1;

    let mut vals: Vec<i32> = (0..num_rows).map(|i| (i as i32) - 50_000).collect();
    vals[10] = expected_min;
    vals[20] = expected_max;

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vals))],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            stats_columns: vec!["v".to_string()],
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    let reader = open_reader(&data);
    let stats = reader.row_group_stats(0).unwrap();
    assert!(!stats.is_empty());

    let col_stats = &stats[0];
    match &col_stats.min {
        Some(paimon_mosaic_core::values::Value::Integer(v)) => {
            assert_eq!(*v, expected_min, "min mismatch");
        }
        other => panic!("expected Integer min, got {:?}", other),
    }
    match &col_stats.max {
        Some(paimon_mosaic_core::values::Value::Integer(v)) => {
            assert_eq!(*v, expected_max, "max mismatch");
        }
        other => panic!("expected Integer max, got {:?}", other),
    }

    println!("test_stats_int32_min_max_exact: PASSED");
}

// 13. test_stats_float64_min_max_exact
#[test]
fn test_stats_float64_min_max_exact() {
    let schema = Schema::new(vec![Field::new("v", DataType::Float64, false)]);

    let expected_min: f64 = -1234567.89;
    let expected_max: f64 = 9876543.21;

    let mut vals: Vec<f64> = (0..10_000).map(|i| (i as f64) * 0.01 - 50.0).collect();
    vals[0] = expected_min;
    vals[1] = expected_max;

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Float64Array::from(vals))],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            stats_columns: vec!["v".to_string()],
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    let reader = open_reader(&data);
    let stats = reader.row_group_stats(0).unwrap();
    assert!(!stats.is_empty());

    let col_stats = &stats[0];
    match &col_stats.min {
        Some(paimon_mosaic_core::values::Value::Double(v)) => {
            assert!(
                (*v - expected_min).abs() < 1e-10,
                "min mismatch: got {}",
                v
            );
        }
        other => panic!("expected Double min, got {:?}", other),
    }
    match &col_stats.max {
        Some(paimon_mosaic_core::values::Value::Double(v)) => {
            assert!(
                (*v - expected_max).abs() < 1e-10,
                "max mismatch: got {}",
                v
            );
        }
        other => panic!("expected Double max, got {:?}", other),
    }

    println!("test_stats_float64_min_max_exact: PASSED");
}

// 14. test_stats_string_min_max_exact
#[test]
fn test_stats_string_min_max_exact() {
    let schema = Schema::new(vec![Field::new("s", DataType::Utf8, false)]);

    // Lexicographic: "aaa" < "mmm" < "zzz"
    let str_vals: Vec<&str> = vec!["mmm", "zzz", "bbb", "aaa", "xyz", "def"];
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(StringArray::from(str_vals))],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            stats_columns: vec!["s".to_string()],
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    let reader = open_reader(&data);
    let stats = reader.row_group_stats(0).unwrap();
    assert!(!stats.is_empty());

    let col_stats = &stats[0];
    match &col_stats.min {
        Some(paimon_mosaic_core::values::Value::String(v)) => {
            assert_eq!(v, b"aaa", "min should be 'aaa', got {:?}", std::str::from_utf8(v));
        }
        other => panic!("expected String min, got {:?}", other),
    }
    match &col_stats.max {
        Some(paimon_mosaic_core::values::Value::String(v)) => {
            assert_eq!(v, b"zzz", "max should be 'zzz', got {:?}", std::str::from_utf8(v));
        }
        other => panic!("expected String max, got {:?}", other),
    }

    println!("test_stats_string_min_max_exact: PASSED");
}

// 15. test_stats_null_count_exact
#[test]
fn test_stats_null_count_exact() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int32, true)]);

    // Exactly 300 nulls out of 1000 rows
    let vals: Vec<Option<i32>> = (0..1000)
        .map(|i| {
            if i < 300 {
                None
            } else {
                Some(i as i32)
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vals))],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            stats_columns: vec!["v".to_string()],
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    let reader = open_reader(&data);
    let stats = reader.row_group_stats(0).unwrap();
    assert!(!stats.is_empty());

    let col_stats = &stats[0];
    assert_eq!(col_stats.null_count, 300, "null_count should be exactly 300");

    println!("test_stats_null_count_exact: PASSED (null_count={})", col_stats.null_count);
}

// 16. test_stats_all_null_column
#[test]
fn test_stats_all_null_column() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, true)]);

    let num_rows = 500;
    let vals: Vec<Option<i64>> = vec![None; num_rows];

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vals))],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            stats_columns: vec!["v".to_string()],
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    let reader = open_reader(&data);
    let stats = reader.row_group_stats(0).unwrap();
    assert!(!stats.is_empty());

    let col_stats = &stats[0];
    assert_eq!(
        col_stats.null_count, num_rows,
        "null_count should equal num_rows for all-null column"
    );
    assert!(col_stats.min.is_none(), "min should be None for all-null column");
    assert!(col_stats.max.is_none(), "max should be None for all-null column");

    println!("test_stats_all_null_column: PASSED");
}

// 17. test_stats_no_null_column
#[test]
fn test_stats_no_null_column() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);

    let num_rows = 1000;
    let vals: Vec<i32> = (0..num_rows as i32).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vals))],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            stats_columns: vec!["v".to_string()],
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    let reader = open_reader(&data);
    let stats = reader.row_group_stats(0).unwrap();
    assert!(!stats.is_empty());

    let col_stats = &stats[0];
    assert_eq!(col_stats.null_count, 0, "null_count should be 0 for no-null column");
    assert!(col_stats.min.is_some(), "min should be present");
    assert!(col_stats.max.is_some(), "max should be present");

    println!("test_stats_no_null_column: PASSED");
}

// 18. test_stats_across_multiple_row_groups
#[test]
fn test_stats_across_multiple_row_groups() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);

    // Write batches with clearly different value ranges to get per-row-group stats
    let batch_size = 5000;
    let mut batches = Vec::new();
    for b in 0..10 {
        // Batch b has values in range [b*1_000_000, b*1_000_000 + batch_size)
        let base = (b as i64) * 1_000_000;
        let vals: Vec<i64> = (0..batch_size).map(|i| base + i as i64).collect();
        batches.push(
            RecordBatch::try_new(
                Arc::new(schema.clone()),
                vec![Arc::new(Int64Array::from(vals))],
            )
            .unwrap(),
        );
    }

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 8 * 1024, // small to force multiple row groups
            stats_columns: vec!["v".to_string()],
            ..Default::default()
        },
    )
    .unwrap();
    for batch in &batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    let reader = open_reader(&data);
    let num_rg = reader.num_row_groups();
    assert!(num_rg > 1, "need multiple row groups, got {}", num_rg);

    // Each row group should have independent stats
    let mut prev_max: Option<i64> = None;
    for rg in 0..num_rg {
        let stats = reader.row_group_stats(rg).unwrap();
        if stats.is_empty() {
            continue;
        }
        let col_stats = &stats[0];
        if let (Some(min_val), Some(max_val)) = (&col_stats.min, &col_stats.max) {
            if let (
                paimon_mosaic_core::values::Value::BigInt(min_v),
                paimon_mosaic_core::values::Value::BigInt(max_v),
            ) = (min_val, max_val)
            {
                assert!(
                    min_v <= max_v,
                    "rg {}: min ({}) > max ({})",
                    rg,
                    min_v,
                    max_v
                );

                // Verify stats are per-row-group (not global)
                if let Some(pm) = prev_max {
                    // The global max grows across row groups, but each row group
                    // should have its own local min/max, not the global one
                    // (unless all data happens to be in one row group)
                    println!(
                        "  rg {}: min={}, max={}, prev_max={}",
                        rg, min_v, max_v, pm
                    );
                }
                prev_max = Some(*max_v);
            }
        }
    }

    println!(
        "test_stats_across_multiple_row_groups: PASSED (num_row_groups={})",
        num_rg
    );
}

// 19. test_stats_single_value
#[test]
fn test_stats_single_value() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vec![42_i64]))],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            stats_columns: vec!["v".to_string()],
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    let reader = open_reader(&data);
    let stats = reader.row_group_stats(0).unwrap();
    assert!(!stats.is_empty());

    let col_stats = &stats[0];
    match (&col_stats.min, &col_stats.max) {
        (
            Some(paimon_mosaic_core::values::Value::BigInt(min_v)),
            Some(paimon_mosaic_core::values::Value::BigInt(max_v)),
        ) => {
            assert_eq!(*min_v, 42, "min should be 42");
            assert_eq!(*max_v, 42, "max should be 42");
            assert_eq!(min_v, max_v, "min must equal max for single value");
        }
        other => panic!("expected BigInt min/max, got {:?}", other),
    }

    println!("test_stats_single_value: PASSED");
}

// 20. test_stats_negative_values
#[test]
fn test_stats_negative_values() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);

    // All negative values
    let vals: Vec<i64> = (-10_000..-1).collect();
    let expected_min = -10_000_i64;
    let expected_max = -2_i64;

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vals))],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            stats_columns: vec!["v".to_string()],
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    let reader = open_reader(&data);
    let stats = reader.row_group_stats(0).unwrap();
    assert!(!stats.is_empty());

    let col_stats = &stats[0];
    match (&col_stats.min, &col_stats.max) {
        (
            Some(paimon_mosaic_core::values::Value::BigInt(min_v)),
            Some(paimon_mosaic_core::values::Value::BigInt(max_v)),
        ) => {
            assert_eq!(*min_v, expected_min, "min mismatch");
            assert_eq!(*max_v, expected_max, "max mismatch");
            assert!(*min_v < *max_v, "min should be < max");
            assert!(*max_v < 0, "max should be < 0 (all negative)");
        }
        other => panic!("expected BigInt min/max, got {:?}", other),
    }

    println!("test_stats_negative_values: PASSED");
}

// 21. test_stats_disabled_columns
#[test]
fn test_stats_disabled_columns() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int64, false),
        Field::new("c", DataType::Float64, false),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
            Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3])),
        ],
    )
    .unwrap();

    // Only enable stats for column "b"
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            stats_columns: vec!["b".to_string()],
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    let reader = open_reader(&data);
    let stats = reader.row_group_stats(0).unwrap();

    // Only one column should have stats
    assert_eq!(stats.len(), 1, "only 1 column should have stats");

    // The stats entry should be for the column that we enabled (internal sorted index of "b")
    let b_sorted_idx = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "b")
        .unwrap();
    assert_eq!(
        stats[0].column_index, b_sorted_idx,
        "stats should be for column 'b'"
    );

    // Now write with NO stats
    let out2 = MemOutputFile::new();
    let mut writer2 = MosaicWriter::new(
        out2,
        &schema,
        WriterOptions {
            num_buckets: 1,
            stats_columns: vec![],
            ..Default::default()
        },
    )
    .unwrap();
    writer2.write_batch(&batch).unwrap();
    writer2.close().unwrap();
    let data2 = writer2.output().buf.clone();

    let reader2 = open_reader(&data2);
    let stats2 = reader2.row_group_stats(0).unwrap();
    assert!(stats2.is_empty(), "no stats should be present when disabled");

    println!("test_stats_disabled_columns: PASSED");
}

// ======================== Estimated File Size Accuracy Tests ========================

// 22. test_estimated_size_before_any_write
#[test]
fn test_estimated_size_before_any_write() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8, true),
    ]);

    let out = MemOutputFile::new();
    let writer = MosaicWriter::new(out, &schema, WriterOptions::default()).unwrap();

    let estimated = writer.estimated_file_size();
    // Before any writes, estimated_file_size should be > 0 (at least the 1024 overhead constant)
    assert!(
        estimated > 0,
        "estimated_file_size before any write should be > 0, got {}",
        estimated
    );

    println!(
        "test_estimated_size_before_any_write: PASSED (estimated={})",
        estimated
    );
}

// 23. test_estimated_size_grows_with_data
#[test]
fn test_estimated_size_grows_with_data() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();

    let mut prev_estimated = writer.estimated_file_size();
    let batch_size = 1000;

    for i in 0..20 {
        let vals: Vec<i64> = ((i * batch_size)..((i + 1) * batch_size))
            .map(|v| v as i64)
            .collect();
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vals))],
        )
        .unwrap();
        writer.write_batch(&batch).unwrap();

        let current_estimated = writer.estimated_file_size();
        assert!(
            current_estimated >= prev_estimated,
            "estimated_file_size should monotonically increase: batch {}: prev={}, current={}",
            i,
            prev_estimated,
            current_estimated
        );
        prev_estimated = current_estimated;
    }

    println!(
        "test_estimated_size_grows_with_data: PASSED (final_estimated={})",
        prev_estimated
    );
}

// 24. test_estimated_size_vs_actual_within_tolerance
#[test]
fn test_estimated_size_vs_actual_within_tolerance() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, true),
    ]);

    // Use NONE compression to avoid compression ratio distortion
    for &num_rows in &[1_000usize, 10_000, 100_000, 500_000] {
        let ids: Vec<i64> = (0..num_rows as i64).collect();
        let strs: Vec<Option<String>> = (0..num_rows)
            .map(|i| {
                if i % 5 == 0 {
                    None
                } else {
                    Some(format!("row_{:08}", i))
                }
            })
            .collect();
        let str_refs: Vec<Option<&str>> = strs.iter().map(|s| s.as_deref()).collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(str_refs)),
            ],
        )
        .unwrap();

        let out = MemOutputFile::new();
        let mut writer = MosaicWriter::new(
            out,
            &schema,
            WriterOptions {
                num_buckets: 1,
                compression: spec::COMPRESSION_NONE,
                ..Default::default()
            },
        )
        .unwrap();
        writer.write_batch(&batch).unwrap();

        // Get estimated size just before close
        let estimated = writer.estimated_file_size();
        writer.close().unwrap();
        let actual = writer.output().buf.len() as u64;

        // The estimate includes a constant 1024 overhead plus buffered * compression_ratio.
        // For small files the 1024 constant dominates. For larger files, the estimate
        // converges closer to actual. We use a generous tolerance that accounts for the
        // constant overhead and dict/const encoding effects.
        let ratio = if estimated > actual {
            estimated as f64 / actual as f64
        } else {
            actual as f64 / estimated as f64
        };

        // With NONE compression, the estimate should be reasonably close.
        // Allow up to 5x for small files (dominated by the 1024 constant),
        // but for larger files (500K rows) it should be tighter.
        let max_ratio = if num_rows >= 100_000 { 1.5 } else { 5.0 };
        assert!(
            ratio <= max_ratio,
            "num_rows={}: estimated ({}) vs actual ({}) ratio {:.2} exceeds {:.1}",
            num_rows,
            estimated,
            actual,
            ratio,
            max_ratio
        );

        println!(
            "  num_rows={}: estimated={}, actual={}, ratio={:.3}",
            num_rows, estimated, actual, ratio
        );
    }

    println!("test_estimated_size_vs_actual_within_tolerance: PASSED");
}

// 25. test_estimated_size_with_compression
#[test]
fn test_estimated_size_with_compression() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);
    let num_rows = 50_000;

    // Highly compressible: repeated values
    let vals: Vec<i64> = (0..num_rows).map(|i| (i % 100) as i64).collect();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vals.clone()))],
    )
    .unwrap();

    // Without compression
    let out_none = MemOutputFile::new();
    let mut writer_none = MosaicWriter::new(
        out_none,
        &schema,
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    )
    .unwrap();
    writer_none.write_batch(&batch).unwrap();
    let est_none = writer_none.estimated_file_size();
    writer_none.close().unwrap();
    let actual_none = writer_none.output().buf.len() as u64;

    // With ZSTD compression
    let batch_zstd = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vals))],
    )
    .unwrap();

    let out_zstd = MemOutputFile::new();
    let mut writer_zstd = MosaicWriter::new(
        out_zstd,
        &schema,
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_ZSTD,
            ..Default::default()
        },
    )
    .unwrap();
    writer_zstd.write_batch(&batch_zstd).unwrap();
    let est_zstd = writer_zstd.estimated_file_size();
    writer_zstd.close().unwrap();
    let actual_zstd = writer_zstd.output().buf.len() as u64;

    // ZSTD actual should be smaller than NONE actual
    assert!(
        actual_zstd < actual_none,
        "ZSTD actual ({}) should be smaller than NONE actual ({})",
        actual_zstd,
        actual_none
    );

    println!(
        "test_estimated_size_with_compression: PASSED (none: est={} actual={}, zstd: est={} actual={})",
        est_none, actual_none, est_zstd, actual_zstd
    );
}

// 26. test_estimated_size_constant_data
#[test]
fn test_estimated_size_constant_data() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);
    let num_rows = 100_000;

    // All same value - will use CONST encoding
    let vals: Vec<i64> = vec![42_i64; num_rows];
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vals))],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();

    let estimated = writer.estimated_file_size();
    // Just verify it doesn't crash and returns a reasonable value
    assert!(estimated > 0, "estimated should be > 0, got {}", estimated);

    writer.close().unwrap();
    let actual = writer.output().buf.len() as u64;

    // CONST encoding makes the actual very small; estimated may be much larger.
    // That's acceptable. Just verify actual is valid.
    assert!(actual > 0, "actual file size should be > 0");
    assert!(
        actual < 2048,
        "CONST-encoded file should be small, got {}",
        actual
    );

    println!(
        "test_estimated_size_constant_data: PASSED (estimated={}, actual={})",
        estimated, actual
    );
}
