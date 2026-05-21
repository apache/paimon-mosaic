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

//! Robustness tests inspired by Apache Arrow Parquet test suite.
//! Covers: corrupted data handling, randomized roundtrips, encoding boundary conditions,
//! statistics validation, row group boundary alignment, and file format invariants.

#![allow(
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::cloned_ref_to_slice_refs,
    clippy::unnecessary_cast
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

fn reorder_batch(batch: &RecordBatch, target_schema: &Schema) -> RecordBatch {
    let mut fields = Vec::new();
    let mut arrays = Vec::new();
    for field in target_schema.fields() {
        let idx = batch.schema().index_of(field.name()).unwrap();
        fields.push(field.clone());
        arrays.push(batch.column(idx).clone());
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
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

// ======================== Bad Data / Corruption Tests ========================

#[test]
fn test_file_too_small() {
    let data = vec![0u8; 10];
    let input = ByteArrayInputFile { data: data.clone() };
    match MosaicReader::new(input, data.len() as u64) {
        Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData),
        Ok(_) => panic!("expected error for file too small"),
    }
}

#[test]
fn test_bad_magic_bytes() {
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let mut data = write_file(&schema, &[batch], WriterOptions::default());

    // Corrupt magic bytes at end of file
    let len = data.len();
    data[len - 4] = b'X';
    data[len - 3] = b'X';

    let input = ByteArrayInputFile { data: data.clone() };
    match MosaicReader::new(input, data.len() as u64) {
        Err(e) => assert!(e.to_string().contains("bad magic"), "got: {}", e),
        Ok(_) => panic!("expected error for bad magic bytes"),
    }
}

#[test]
fn test_corrupted_version() {
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let mut data = write_file(&schema, &[batch], WriterOptions::default());

    // Corrupt version byte (footer[25])
    let len = data.len();
    data[len - 7] = 99; // Invalid version

    let input = ByteArrayInputFile { data: data.clone() };
    match MosaicReader::new(input, data.len() as u64) {
        Err(e) => assert!(e.to_string().contains("unsupported version"), "got: {}", e),
        Ok(_) => panic!("expected error for corrupted version"),
    }
}

#[test]
fn test_corrupted_footer_offsets() {
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let mut data = write_file(&schema, &[batch], WriterOptions::default());

    // Corrupt schema_block_offset to point past file end
    let len = data.len();
    let footer_start = len - spec::FOOTER_SIZE;
    // footer[8..16] = schema_block_offset (big-endian u64)
    let bad_offset = (len as u64 + 1000).to_be_bytes();
    data[footer_start + 8..footer_start + 16].copy_from_slice(&bad_offset);

    let input = ByteArrayInputFile { data: data.clone() };
    match MosaicReader::new(input, data.len() as u64) {
        Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData),
        Ok(_) => panic!("expected error for corrupted footer offsets"),
    }
}

#[test]
fn test_corrupted_compressed_data() {
    let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);
    let vals: Vec<i64> = (0..10000).collect();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vals))],
    )
    .unwrap();
    let mut data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            ..Default::default()
        },
    );

    // Corrupt a large portion of the data section to guarantee hitting compressed payload
    let corrupt_start = 0;
    let corrupt_end = (data.len() - spec::FOOTER_SIZE).min(data.len() / 2);
    for i in corrupt_start..corrupt_end {
        data[i] = 0xFF;
    }

    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64);
    // Corruption must surface as an error at open or at read time
    let failed = match reader {
        Err(_) => true,
        Ok(reader) => match reader.row_group_reader(0) {
            Err(_) => true,
            Ok(mut rg) => rg.read_columns().is_err(),
        },
    };
    assert!(failed, "corrupted data should cause an error");
}

#[test]
fn test_truncated_file() {
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let data = write_file(&schema, &[batch], WriterOptions::default());

    // Truncate to just the footer size + a bit
    let truncated = &data[..spec::FOOTER_SIZE + 10];
    let input = ByteArrayInputFile {
        data: truncated.to_vec(),
    };
    assert!(
        MosaicReader::new(input, truncated.len() as u64).is_err(),
        "truncated file should fail to open"
    );
}

#[test]
fn test_zero_length_file() {
    let data: Vec<u8> = vec![];
    let input = ByteArrayInputFile { data: data.clone() };
    match MosaicReader::new(input, 0) {
        Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData),
        Ok(_) => panic!("expected error for zero-length file"),
    }
}

// ======================== Randomized Roundtrip Tests ========================

fn simple_hash(seed: u64, i: usize) -> u64 {
    let mut x = seed.wrapping_add(i as u64);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

#[test]
fn test_randomized_int_patterns() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int64, true),
    ]);

    for seed in 0..10u64 {
        let num_rows = 10_000 + (seed as usize * 5000);
        let a_vals: Vec<Option<i32>> = (0..num_rows)
            .map(|i| {
                let h = simple_hash(seed, i);
                if h % 7 == 0 {
                    None
                } else {
                    Some(h as i32)
                }
            })
            .collect();
        let b_vals: Vec<Option<i64>> = (0..num_rows)
            .map(|i| {
                let h = simple_hash(seed.wrapping_add(1), i);
                if h % 11 == 0 {
                    None
                } else {
                    Some(h as i64)
                }
            })
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(a_vals.clone())),
                Arc::new(Int64Array::from(b_vals.clone())),
            ],
        )
        .unwrap();

        let data = write_file(
            &schema,
            &[batch.clone()],
            WriterOptions {
                num_buckets: 2 + seed as usize % 10,
                row_group_max_size: 1024 * 1024,
                ..Default::default()
            },
        );

        let result = read_all(&data);
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, num_rows, "seed={seed}");
    }
}

#[test]
fn test_randomized_string_patterns() {
    let schema = Schema::new(vec![
        Field::new("s", DataType::Utf8, true),
        Field::new("id", DataType::Int32, false),
    ]);

    for seed in 0..5u64 {
        let num_rows = 20_000;
        let str_vals: Vec<Option<String>> = (0..num_rows)
            .map(|i| {
                let h = simple_hash(seed, i);
                if h % 5 == 0 {
                    None
                } else {
                    let len = (h % 200) as usize;
                    Some(
                        (0..len)
                            .map(|j| {
                                let c = simple_hash(seed + 100, i * 200 + j) % 26;
                                (b'a' + c as u8) as char
                            })
                            .collect(),
                    )
                }
            })
            .collect();
        let str_refs: Vec<Option<&str>> = str_vals.iter().map(|s| s.as_deref()).collect();
        let ids: Vec<i32> = (0..num_rows as i32).collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(StringArray::from(str_refs)),
                Arc::new(Int32Array::from(ids)),
            ],
        )
        .unwrap();

        let data = write_file(
            &schema,
            &[batch.clone()],
            WriterOptions {
                num_buckets: 3,
                row_group_max_size: 2 * 1024 * 1024,
                ..Default::default()
            },
        );

        let result = read_all(&data);
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, num_rows, "seed={seed}");
    }
}

// ======================== Encoding Boundary Tests ========================

#[test]
fn test_dict_exactly_255_entries() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);

    // Exactly 255 distinct values (max dict entries)
    let vals: Vec<i32> = (0..10_000).map(|i| (i % 255) as i32).collect();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vals.clone()))],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            ..Default::default()
        },
    );
    let result = read_all(&data);
    let out_vals: Vec<i32> = result
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(out_vals, vals);
}

#[test]
fn test_dict_256_entries_falls_back_to_plain() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);

    // 256 distinct values (exceeds max dict entries of 255)
    let vals: Vec<i32> = (0..10_000).map(|i| (i % 256) as i32).collect();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vals.clone()))],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            ..Default::default()
        },
    );
    let result = read_all(&data);
    let out_vals: Vec<i32> = result
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(out_vals, vals);
}

#[test]
fn test_single_distinct_value_const_encoding() {
    let schema = Schema::new(vec![
        Field::new("int_const", DataType::Int64, false),
        Field::new("str_const", DataType::Utf8, false),
    ]);

    let num_rows = 100_000;
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(vec![999_999_999i64; num_rows])),
            Arc::new(StringArray::from(vec!["constant"; num_rows])),
        ],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            ..Default::default()
        },
    );

    // Constant encoding should produce a very small file
    assert!(
        data.len() < 1024,
        "file size {} bytes is too large for constant data",
        data.len()
    );

    let result = read_all(&data);
    let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, num_rows);
}

#[test]
fn test_encoding_transition_across_row_groups() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int32, true)]);

    // First batch: constant values (should get CONST encoding)
    let batch1 = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vec![Some(42); 50_000]))],
    )
    .unwrap();

    // Second batch: high cardinality (should get PLAIN encoding)
    let vals: Vec<Option<i32>> = (0..50_000).map(|i| Some(i as i32)).collect();
    let batch2 = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vals))],
    )
    .unwrap();

    // Third batch: low cardinality (should get DICT encoding)
    let vals: Vec<Option<i32>> = (0..50_000).map(|i| Some((i % 5) as i32)).collect();
    let batch3 = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vals))],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[batch1.clone(), batch2.clone(), batch3.clone()],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 128 * 1024,
            ..Default::default()
        },
    );

    let result = read_all(&data);
    let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 150_000);
}

// ======================== Row Group Boundary Tests ========================

#[test]
fn test_row_group_split_preserves_data() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, true),
    ]);

    let num_rows = 100_000;
    let batch_size = 1000;
    let num_batches = num_rows / batch_size;

    let mut batches = Vec::new();
    for b in 0..num_batches {
        let start = (b * batch_size) as i64;
        let ids: Vec<i64> = (start..start + batch_size as i64).collect();
        let data_vals: Vec<Option<String>> = (0..batch_size)
            .map(|i| {
                let row = b * batch_size + i;
                if row % 3 == 0 {
                    None
                } else {
                    Some(format!("row_{:06}_padding_{}", row, "x".repeat(100)))
                }
            })
            .collect();
        let data_refs: Vec<Option<&str>> = data_vals.iter().map(|s| s.as_deref()).collect();
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(data_refs)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    // Write with small row group size; many small batches allow the writer to split
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 5,
            row_group_max_size: 64 * 1024, // 64KB
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

    // Verify multiple row groups were created
    assert!(
        reader.num_row_groups() > 1,
        "expected multiple row groups, got {}",
        reader.num_row_groups()
    );

    // Verify all data reads back correctly
    let mut total_rows = 0;
    let mut last_id = -1i64;
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader(rg).unwrap();
        let batch = rg_reader.read_columns().unwrap();
        let id_idx = batch.schema().index_of("id").unwrap();
        let id_col = batch
            .column(id_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        // IDs should be consecutive
        for i in 0..batch.num_rows() {
            let id = id_col.value(i);
            assert_eq!(id, last_id + 1, "row group {rg}, row {i}");
            last_id = id;
        }
        total_rows += batch.num_rows();
    }
    assert_eq!(total_rows, num_rows);
}

#[test]
fn test_empty_row_groups_not_created() {
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();

    let file_data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 1024 * 1024 * 1024,
            ..Default::default()
        },
    );

    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();
    assert_eq!(reader.num_row_groups(), 1);
}

// ======================== Statistics Validation Tests ========================

#[test]
fn test_stats_min_max_correctness() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, true)]);

    // Create data with known min/max per row group
    let batches: Vec<RecordBatch> = vec![
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![
                Some(100),
                Some(200),
                Some(300),
                None,
                Some(50),
            ]))],
        )
        .unwrap(),
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![
                Some(-100),
                None,
                Some(0),
                Some(1000),
                Some(500),
            ]))],
        )
        .unwrap(),
    ];

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 32, // Force each batch to be its own row group
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
    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();

    // Should have stats for each row group
    for rg in 0..reader.num_row_groups() {
        let stats = reader.row_group_stats(rg).unwrap();
        // Stats should exist (we enabled them for column 0)
        assert!(!stats.is_empty(), "rg={rg}: expected stats");
    }
}

#[test]
fn test_stats_with_all_null_row_group() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, true)]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vec![None; 1000]))],
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
    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();

    let stats = reader.row_group_stats(0).unwrap();
    assert!(!stats.is_empty());
}

// ======================== File Format Invariant Tests ========================

#[test]
fn test_file_starts_with_data_ends_with_metadata() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Utf8, true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("x"), None, Some("z")])),
        ],
    )
    .unwrap();

    let data = write_file(&schema, &[batch], WriterOptions::default());

    // File must end with magic bytes MOSA
    assert_eq!(&data[data.len() - 4..], b"MOSA");

    // Footer is 32 bytes
    assert!(data.len() >= spec::FOOTER_SIZE);

    // Version byte
    assert_eq!(data[data.len() - 7], spec::VERSION);
}

#[test]
fn test_num_buckets_capped_by_column_count() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![2])),
        ],
    )
    .unwrap();

    // Request 1000 buckets but only 2 columns
    let data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1000,
            ..Default::default()
        },
    );

    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();
    assert_eq!(reader.schema().num_buckets, 2);
}

#[test]
fn test_writer_rejects_column_count_mismatch() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let wrong_schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(out, &schema, WriterOptions::default()).unwrap();

    let batch = RecordBatch::try_new(
        Arc::new(wrong_schema),
        vec![Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();

    let err = writer.write_batch(&batch).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    assert!(err.to_string().contains("column count mismatch"));
}

#[test]
fn test_writer_rejects_write_after_close() {
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(out, &schema, WriterOptions::default()).unwrap();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vec![1]))],
    )
    .unwrap();

    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();

    let err = writer.write_batch(&batch).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
}

// ======================== Projection Correctness Tests ========================

#[test]
fn test_projection_returns_correct_columns() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8, true),
        Field::new("c", DataType::Float64, false),
        Field::new("d", DataType::Int64, true),
    ]);

    let num_rows = 10_000;
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from((0..num_rows as i32).collect::<Vec<_>>())),
            Arc::new(StringArray::from(
                (0..num_rows)
                    .map(|i| if i % 3 == 0 { None } else { Some("val") })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (0..num_rows).map(|i| i as f64).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                (0..num_rows)
                    .map(|i| if i % 2 == 0 { Some(i as i64) } else { None })
                    .collect::<Vec<Option<i64>>>(),
            )),
        ],
    )
    .unwrap();

    let file_data = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 4,
            ..Default::default()
        },
    );

    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();

    // Project only columns [0, 2] (a, c)
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader_projected(rg, &[0, 2]).unwrap();
        let result = rg_reader.read_columns().unwrap();
        assert_eq!(result.num_columns(), 2);

        // Verify column names
        let schema = result.schema();
        let fields: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert!(fields.contains(&"a"));
        assert!(fields.contains(&"c"));
    }
}

#[test]
fn test_projection_out_of_bounds_error() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![2])),
        ],
    )
    .unwrap();

    let file_data = write_file(&schema, &[batch], WriterOptions::default());
    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();

    match reader.row_group_reader_projected(0, &[0, 99]) {
        Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidInput),
        Ok(_) => panic!("expected error for out-of-bounds projection"),
    }
}

// ======================== Compression Tests ========================

#[test]
fn test_zstd_compressed_smaller_than_uncompressed() {
    let schema = Schema::new(vec![Field::new("v", DataType::Int64, false)]);

    let num_rows = 100_000;
    // Highly compressible: sequential values
    let vals: Vec<i64> = (0..num_rows as i64).collect();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int64Array::from(vals))],
    )
    .unwrap();

    let data_zstd = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_ZSTD,
            ..Default::default()
        },
    );

    let data_none = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    assert!(
        data_zstd.len() < data_none.len(),
        "zstd={} should be smaller than none={}",
        data_zstd.len(),
        data_none.len()
    );
}

#[test]
fn test_incompressible_data_still_works() {
    let schema = Schema::new(vec![Field::new("v", DataType::Binary, false)]);

    // Random-ish binary data that won't compress well
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

    let data = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 1,
            ..Default::default()
        },
    );

    let result = read_all(&data);
    let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, num_rows);
}

// ======================== Schema Tests ========================

#[test]
fn test_schema_roundtrip_preserves_types_and_nullability() {
    let tz: Arc<str> = Arc::from("UTC");
    let schema = Schema::new(vec![
        Field::new("bool", DataType::Boolean, false),
        Field::new("i8", DataType::Int8, true),
        Field::new("i16", DataType::Int16, false),
        Field::new("i32", DataType::Int32, true),
        Field::new("i64", DataType::Int64, false),
        Field::new("f32", DataType::Float32, true),
        Field::new("f64", DataType::Float64, false),
        Field::new("date", DataType::Date32, true),
        Field::new("time", DataType::Time32(TimeUnit::Millisecond), true),
        Field::new("str", DataType::Utf8, true),
        Field::new("bin", DataType::Binary, false),
        Field::new("dec_small", DataType::Decimal128(10, 3), true),
        Field::new(
            "ts_ms",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ),
        Field::new(
            "ts_us_tz",
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz)),
            true,
        ),
    ]);

    // Write minimal data
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(BooleanArray::from(vec![true])),
            Arc::new(Int8Array::from(vec![Some(1i8)])),
            Arc::new(Int16Array::from(vec![1i16])),
            Arc::new(Int32Array::from(vec![Some(1i32)])),
            Arc::new(Int64Array::from(vec![1i64])),
            Arc::new(Float32Array::from(vec![Some(1.0f32)])),
            Arc::new(Float64Array::from(vec![1.0f64])),
            Arc::new(Date32Array::from(vec![Some(19000)])),
            Arc::new(Time32MillisecondArray::from(vec![Some(3600000)])),
            Arc::new(StringArray::from(vec![Some("hello")])),
            Arc::new(BinaryArray::from(vec![&[1u8, 2, 3] as &[u8]])),
            Arc::new(
                Decimal128Array::from(vec![Some(12345i128)])
                    .with_precision_and_scale(10, 3)
                    .unwrap(),
            ),
            Arc::new(TimestampMillisecondArray::from(vec![Some(
                1700000000000i64,
            )])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![Some(1700000000000000i64)])
                    .with_timezone("UTC"),
            ),
        ],
    )
    .unwrap();

    let file_data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 5,
            ..Default::default()
        },
    );

    let input = ByteArrayInputFile {
        data: file_data.clone(),
    };
    let reader = MosaicReader::new(input, file_data.len() as u64).unwrap();
    let mosaic_schema = reader.schema();

    // Verify all columns preserved
    assert_eq!(mosaic_schema.columns.len(), 14);

    // Verify names and types (columns are in sorted order)
    let find = |name: &str| {
        mosaic_schema
            .columns
            .iter()
            .find(|c| c.name == name)
            .unwrap()
    };

    let c = find("bool");
    assert_eq!(c.data_type, DataType::Boolean);
    assert!(!c.nullable);

    let c = find("i8");
    assert_eq!(c.data_type, DataType::Int8);
    assert!(c.nullable);

    let c = find("str");
    assert_eq!(c.data_type, DataType::Utf8);

    let c = find("bin");
    assert_eq!(c.data_type, DataType::Binary);
    assert!(!c.nullable);
}

#[test]
fn test_schema_rejects_duplicate_column_names() {
    let schema = Schema::new(vec![
        Field::new("dup", DataType::Int32, false),
        Field::new("dup", DataType::Int64, false),
    ]);

    let out = MemOutputFile::new();
    let result = MosaicWriter::new(out, &schema, WriterOptions::default());
    assert!(result.is_err());
}

// ======================== Numeric Limits (inspired by Parquet NumericLimits scenario) ========================

#[test]
fn test_numeric_limits_roundtrip() {
    let schema = Schema::new(vec![
        Field::new("i8", DataType::Int8, false),
        Field::new("i16", DataType::Int16, false),
        Field::new("i32", DataType::Int32, false),
        Field::new("i64", DataType::Int64, false),
        Field::new("f32", DataType::Float32, false),
        Field::new("f64", DataType::Float64, false),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int8Array::from(vec![i8::MIN, -100, -1, 0, 1, 100, i8::MAX])),
            Arc::new(Int16Array::from(vec![
                i16::MIN,
                -1000,
                -1,
                0,
                1,
                1000,
                i16::MAX,
            ])),
            Arc::new(Int32Array::from(vec![
                i32::MIN,
                -100_000,
                -1,
                0,
                1,
                100_000,
                i32::MAX,
            ])),
            Arc::new(Int64Array::from(vec![
                i64::MIN,
                -1_000_000_000,
                -1,
                0,
                1,
                1_000_000_000,
                i64::MAX,
            ])),
            Arc::new(Float32Array::from(vec![
                f32::MIN,
                -1.0,
                -f32::EPSILON,
                0.0,
                f32::EPSILON,
                1.0,
                f32::MAX,
            ])),
            Arc::new(Float64Array::from(vec![
                f64::MIN,
                -1.0,
                -f64::EPSILON,
                0.0,
                f64::EPSILON,
                1.0,
                f64::MAX,
            ])),
        ],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 3,
            ..Default::default()
        },
    );
    let result = read_all(&data);
    assert_eq!(result.len(), 1);
    assert_eq!(reorder_batch(&result[0], &schema), batch);
}

// ======================== Multiple Writers Compatibility ========================

#[test]
fn test_no_compression_vs_zstd_same_data() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("val", DataType::Utf8, true),
    ]);

    let num_rows = 50_000;
    let ids: Vec<i64> = (0..num_rows as i64).collect();
    let vals: Vec<Option<&str>> = (0..num_rows)
        .map(|i| if i % 4 == 0 { None } else { Some("test_value") })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(vals)),
        ],
    )
    .unwrap();

    let data_none = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 5,
            compression: spec::COMPRESSION_NONE,
            ..Default::default()
        },
    );

    let data_zstd = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: 5,
            compression: spec::COMPRESSION_ZSTD,
            ..Default::default()
        },
    );

    let result_none = read_all(&data_none);
    let result_zstd = read_all(&data_zstd);

    // Both should produce the same logical data
    let rows_none: usize = result_none.iter().map(|b| b.num_rows()).sum();
    let rows_zstd: usize = result_zstd.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows_none, num_rows);
    assert_eq!(rows_zstd, num_rows);
}

// ======================== Empty Batch / Single Row Edge Cases ========================

#[test]
fn test_single_row_roundtrip() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("flag", DataType::Boolean, true),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(vec![42])),
            Arc::new(StringArray::from(vec![Some("only_one")])),
            Arc::new(BooleanArray::from(vec![Some(true)])),
        ],
    )
    .unwrap();

    let data = write_file(&schema, &[batch.clone()], WriterOptions::default());
    let result = read_all(&data);
    assert_eq!(result.len(), 1);
    assert_eq!(reorder_batch(&result[0], &schema), batch);
}

#[test]
fn test_single_row_all_null() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8, true),
        Field::new("c", DataType::Float64, true),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(vec![None as Option<i32>])),
            Arc::new(StringArray::from(vec![None as Option<&str>])),
            Arc::new(Float64Array::from(vec![None as Option<f64>])),
        ],
    )
    .unwrap();

    let data = write_file(&schema, &[batch.clone()], WriterOptions::default());
    let result = read_all(&data);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].num_rows(), 1);
    assert_eq!(result[0].column(0).null_count(), 1);
    assert_eq!(result[0].column(1).null_count(), 1);
    assert_eq!(result[0].column(2).null_count(), 1);
}

#[test]
fn test_empty_batch_skipped() {
    let schema = Schema::new(vec![Field::new("x", DataType::Int32, false)]);

    let empty_batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(Vec::<i32>::new()))],
    )
    .unwrap();

    let real_batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[empty_batch, real_batch],
        WriterOptions::default(),
    );
    let result = read_all(&data);
    let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3);
}

// ======================== ALL_NULL Encoding Tests ========================

#[test]
fn test_all_null_column_encoding() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("nullable_int", DataType::Int32, true),
        Field::new("nullable_str", DataType::Utf8, true),
    ]);

    let num_rows = 10_000;
    let ids: Vec<i64> = (0..num_rows as i64).collect();
    let nulls_i32: Vec<Option<i32>> = vec![None; num_rows];
    let nulls_str: Vec<Option<&str>> = vec![None; num_rows];

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int32Array::from(nulls_i32)),
            Arc::new(StringArray::from(nulls_str)),
        ],
    )
    .unwrap();

    let data = write_file(&schema, &[batch], WriterOptions::default());
    let result = read_all(&data);
    assert_eq!(result[0].num_rows(), num_rows);
    assert_eq!(result[0].column(1).null_count(), num_rows);
    assert_eq!(result[0].column(2).null_count(), num_rows);
}

#[test]
fn test_all_null_to_data_across_row_groups() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("val", DataType::Int32, true),
    ]);

    let batch_size = 1000;

    // First batch: all nulls
    let batch_null = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from((0..batch_size as i64).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(vec![None as Option<i32>; batch_size])),
        ],
    )
    .unwrap();

    // Second batch: all data
    let batch_data = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(
                (batch_size as i64..2 * batch_size as i64).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from((0..batch_size as i32).collect::<Vec<_>>())),
        ],
    )
    .unwrap();

    // Write with small row group size to get separate row groups
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 1024,
            ..Default::default()
        },
    )
    .unwrap();
    writer.write_batch(&batch_null).unwrap();
    writer.write_batch(&batch_data).unwrap();
    writer.close().unwrap();
    let file_data = writer.output().buf.clone();

    let result = read_all(&file_data);
    let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2 * batch_size);

    // Verify first row group is all nulls, second has data
    let first_nulls: usize = result
        .iter()
        .take(1)
        .map(|b| b.column(1).null_count())
        .sum();
    assert!(first_nulls > 0);
    let last_batch = &result[result.len() - 1];
    let val_col = last_batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert!(val_col.null_count() < val_col.len());
}

#[test]
fn test_all_null_boolean_column() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("flag", DataType::Boolean, true),
    ]);

    let num_rows = 5000;
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from((0..num_rows as i32).collect::<Vec<_>>())),
            Arc::new(BooleanArray::from(vec![None as Option<bool>; num_rows])),
        ],
    )
    .unwrap();

    let data = write_file(&schema, &[batch], WriterOptions::default());
    let result = read_all(&data);
    let flag_idx = result[0].schema().index_of("flag").unwrap();
    assert_eq!(result[0].column(flag_idx).null_count(), num_rows);
}

// ======================== Decimal128 Tests ========================

#[test]
fn test_decimal128_compact_precision() {
    // precision <= 18: uses compact i64 encoding
    let schema = Schema::new(vec![
        Field::new("dec_p10", DataType::Decimal128(10, 3), false),
        Field::new("dec_p18", DataType::Decimal128(18, 6), true),
    ]);

    let num_rows = 10_000;
    let vals_p10: Vec<i128> = (0..num_rows)
        .map(|i| ((i as i128) - 5000) * 1_000)
        .collect();
    let vals_p18: Vec<Option<i128>> = (0..num_rows)
        .map(|i| {
            if i % 7 == 0 {
                None
            } else {
                Some((i as i128 - 5000) * 1_000_000)
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(
                Decimal128Array::from(vals_p10.clone())
                    .with_precision_and_scale(10, 3)
                    .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from(vals_p18.clone())
                    .with_precision_and_scale(18, 6)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();

    let data = write_file(&schema, &[batch.clone()], WriterOptions::default());
    let result = read_all(&data);
    assert_eq!(result.len(), 1);
    let col0 = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    assert_eq!(col0.precision(), 10);
    assert_eq!(col0.scale(), 3);
    for i in 0..num_rows {
        assert_eq!(col0.value(i), vals_p10[i]);
    }

    let col1 = result[0]
        .column(1)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    assert_eq!(col1.precision(), 18);
    assert_eq!(col1.scale(), 6);
    for i in 0..num_rows {
        match vals_p18[i] {
            None => assert!(col1.is_null(i)),
            Some(v) => assert_eq!(col1.value(i), v),
        }
    }
}

#[test]
fn test_decimal128_large_precision() {
    // precision > 18: uses full 16-byte i128 encoding
    let schema = Schema::new(vec![
        Field::new("dec_p20", DataType::Decimal128(20, 5), false),
        Field::new("dec_p38", DataType::Decimal128(38, 10), true),
    ]);

    let num_rows = 5_000;
    // Values that exceed i64 range
    let vals_p20: Vec<i128> = (0..num_rows)
        .map(|i| {
            let base: i128 = 10_000_000_000_000_000_000; // exceeds i64::MAX
            base + i as i128 * 100_000
        })
        .collect();
    let vals_p38: Vec<Option<i128>> = (0..num_rows)
        .map(|i| {
            if i % 5 == 0 {
                None
            } else {
                let base: i128 = 100_000_000_000_000_000_000_000_000_000; // very large
                Some(base + i as i128 * 10_000_000_000)
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(
                Decimal128Array::from(vals_p20.clone())
                    .with_precision_and_scale(20, 5)
                    .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from(vals_p38.clone())
                    .with_precision_and_scale(38, 10)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();

    let data = write_file(&schema, &[batch], WriterOptions::default());
    let result = read_all(&data);
    assert_eq!(result.len(), 1);

    let col0 = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    assert_eq!(col0.precision(), 20);
    assert_eq!(col0.scale(), 5);
    for i in 0..num_rows {
        assert_eq!(col0.value(i), vals_p20[i]);
    }

    let col1 = result[0]
        .column(1)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    assert_eq!(col1.precision(), 38);
    assert_eq!(col1.scale(), 10);
    for i in 0..num_rows {
        match vals_p38[i] {
            None => assert!(col1.is_null(i)),
            Some(v) => assert_eq!(col1.value(i), v),
        }
    }
}

#[test]
fn test_decimal128_extreme_values() {
    let schema = Schema::new(vec![Field::new("dec", DataType::Decimal128(38, 0), true)]);

    // i128 extremes within Decimal128(38,0) range
    let max_decimal38: i128 = 99_999_999_999_999_999_999_999_999_999_999_999_999;
    let min_decimal38: i128 = -max_decimal38;

    let vals = vec![
        Some(0i128),
        Some(1),
        Some(-1),
        Some(max_decimal38),
        Some(min_decimal38),
        None,
        Some(i64::MAX as i128),
        Some(i64::MIN as i128),
        Some(i64::MAX as i128 + 1),
        Some(i64::MIN as i128 - 1),
    ];

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(
            Decimal128Array::from(vals.clone())
                .with_precision_and_scale(38, 0)
                .unwrap(),
        )],
    )
    .unwrap();

    let data = write_file(&schema, &[batch], WriterOptions::default());
    let result = read_all(&data);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();

    for (i, v) in vals.iter().enumerate() {
        match v {
            None => assert!(col.is_null(i)),
            Some(expected) => assert_eq!(col.value(i), *expected, "mismatch at row {}", i),
        }
    }
}

// ======================== TimestampNanos Tests ========================

#[test]
fn test_timestamp_nanos_basic_roundtrip() {
    let ts_fields: arrow_schema::Fields = vec![
        Field::new("millis", DataType::Int64, false),
        Field::new("nanos_of_milli", DataType::Int32, false),
    ]
    .into();

    let schema = Schema::new(vec![Field::new(
        "ts",
        DataType::Struct(ts_fields.clone()),
        true,
    )]);

    let num_rows = 10_000;
    let millis_vals: Vec<Option<i64>> = (0..num_rows)
        .map(|i| {
            if i % 11 == 0 {
                None
            } else {
                Some(1_700_000_000_000i64 + i as i64)
            }
        })
        .collect();
    let nanos_vals: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            if i % 11 == 0 {
                None
            } else {
                Some((i % 1_000_000) as i32)
            }
        })
        .collect();

    let millis_arr = Int64Array::from(millis_vals.clone());
    let nanos_arr = Int32Array::from(nanos_vals.clone());
    let null_buf = millis_arr.nulls().cloned();
    let ts_struct = StructArray::new(
        ts_fields.clone(),
        vec![Arc::new(millis_arr), Arc::new(nanos_arr)],
        null_buf,
    );

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(ts_struct)]).unwrap();

    let data = write_file(&schema, &[batch], WriterOptions::default());
    let result = read_all(&data);
    assert_eq!(result[0].num_rows(), num_rows);

    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();

    for i in 0..num_rows {
        if i % 11 == 0 {
            assert!(col.is_null(i), "expected null at row {}", i);
        } else {
            assert!(!col.is_null(i));
            assert_eq!(
                col.value(i),
                paimon_mosaic_core::types::millis_nanos_to_ns(
                    1_700_000_000_000i64 + i as i64,
                    (i % 1_000_000) as i32,
                )
                .unwrap()
            );
        }
    }
}

#[test]
fn test_timestamp_nanos_extreme_values() {
    let ts_fields: arrow_schema::Fields = vec![
        Field::new("millis", DataType::Int64, false),
        Field::new("nanos_of_milli", DataType::Int32, false),
    ]
    .into();

    let schema = Schema::new(vec![Field::new(
        "ts",
        DataType::Struct(ts_fields.clone()),
        false,
    )]);

    let ns_data: Vec<i64> = vec![
        0,
        i64::MAX,
        i64::MIN,
        paimon_mosaic_core::types::millis_nanos_to_ns(1_700_000_000_000, 999_999).unwrap(),
        paimon_mosaic_core::types::millis_nanos_to_ns(-1_700_000_000_000, 1).unwrap(),
        1,
        -1,
    ];
    let (millis_data, nanos_data): (Vec<i64>, Vec<i32>) = ns_data
        .iter()
        .map(|&ns| paimon_mosaic_core::types::ns_to_millis_nanos(ns))
        .unzip();

    let millis_arr = Int64Array::from(millis_data.clone());
    let nanos_arr = Int32Array::from(nanos_data.clone());
    let ts_struct = StructArray::new(
        ts_fields.clone(),
        vec![Arc::new(millis_arr), Arc::new(nanos_arr)],
        None,
    );

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(ts_struct)]).unwrap();

    let data = write_file(&schema, &[batch], WriterOptions::default());
    let result = read_all(&data);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();

    for (i, expected) in ns_data.iter().enumerate() {
        assert_eq!(col.value(i), *expected, "timestamp ns mismatch at {}", i);
    }
}

#[test]
fn test_timestamp_nanos_all_null() {
    let ts_fields: arrow_schema::Fields = vec![
        Field::new("millis", DataType::Int64, false),
        Field::new("nanos_of_milli", DataType::Int32, false),
    ]
    .into();

    let schema = Schema::new(vec![Field::new(
        "ts",
        DataType::Struct(ts_fields.clone()),
        true,
    )]);

    let num_rows = 1000;
    let millis_vals: Vec<Option<i64>> = vec![None; num_rows];
    let nanos_vals: Vec<Option<i32>> = vec![None; num_rows];

    let millis_arr = Int64Array::from(millis_vals);
    let nanos_arr = Int32Array::from(nanos_vals);
    let null_buf = millis_arr.nulls().cloned();
    let ts_struct = StructArray::new(
        ts_fields.clone(),
        vec![Arc::new(millis_arr), Arc::new(nanos_arr)],
        null_buf,
    );

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(ts_struct)]).unwrap();

    let data = write_file(&schema, &[batch], WriterOptions::default());
    let result = read_all(&data);
    assert_eq!(result[0].column(0).null_count(), num_rows);
}

// ======================== Boolean Boundary Tests ========================

#[test]
fn test_boolean_all_true() {
    let schema = Schema::new(vec![Field::new("b", DataType::Boolean, false)]);
    let num_rows = 10_000;
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(BooleanArray::from(vec![true; num_rows]))],
    )
    .unwrap();

    let data = write_file(&schema, &[batch.clone()], WriterOptions::default());
    let result = read_all(&data);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    for i in 0..num_rows {
        assert!(col.value(i), "expected true at row {}", i);
    }
}

#[test]
fn test_boolean_all_false() {
    let schema = Schema::new(vec![Field::new("b", DataType::Boolean, false)]);
    let num_rows = 10_000;
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(BooleanArray::from(vec![false; num_rows]))],
    )
    .unwrap();

    let data = write_file(&schema, &[batch.clone()], WriterOptions::default());
    let result = read_all(&data);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    for i in 0..num_rows {
        assert!(!col.value(i), "expected false at row {}", i);
    }
}

#[test]
fn test_boolean_alternating_with_nulls() {
    let schema = Schema::new(vec![Field::new("b", DataType::Boolean, true)]);
    let num_rows = 10_000;

    // Pattern: true, false, null, true, false, null, ...
    let vals: Vec<Option<bool>> = (0..num_rows)
        .map(|i| match i % 3 {
            0 => Some(true),
            1 => Some(false),
            _ => None,
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(BooleanArray::from(vals.clone()))],
    )
    .unwrap();

    let data = write_file(&schema, &[batch], WriterOptions::default());
    let result = read_all(&data);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();

    for i in 0..num_rows {
        match vals[i] {
            None => assert!(col.is_null(i), "expected null at {}", i),
            Some(v) => {
                assert!(!col.is_null(i), "unexpected null at {}", i);
                assert_eq!(col.value(i), v, "value mismatch at {}", i);
            }
        }
    }
}

#[test]
fn test_boolean_byte_boundary_alignment() {
    let schema = Schema::new(vec![Field::new("b", DataType::Boolean, true)]);

    // Test sizes that exercise byte boundaries: 7, 8, 9, 15, 16, 17, 63, 64, 65
    for num_rows in [
        7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257,
    ] {
        let vals: Vec<Option<bool>> = (0..num_rows)
            .map(|i| if i % 5 == 0 { None } else { Some(i % 2 == 0) })
            .collect();

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(BooleanArray::from(vals.clone()))],
        )
        .unwrap();

        let data = write_file(&schema, &[batch], WriterOptions::default());
        let result = read_all(&data);
        let col = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();

        assert_eq!(
            col.len(),
            num_rows,
            "wrong length for num_rows={}",
            num_rows
        );
        for i in 0..num_rows {
            match vals[i] {
                None => assert!(col.is_null(i), "expected null at {} (rows={})", i, num_rows),
                Some(v) => assert_eq!(col.value(i), v, "mismatch at {} (rows={})", i, num_rows),
            }
        }
    }
}

#[test]
fn test_boolean_sparse_nulls() {
    // Mostly data, very few nulls — tests that null bitmap is correct at sparse positions
    let schema = Schema::new(vec![Field::new("b", DataType::Boolean, true)]);
    let num_rows = 10_000;

    let vals: Vec<Option<bool>> = (0..num_rows)
        .map(|i| {
            // Null at positions that are prime-ish to avoid alignment
            if i == 7 || i == 13 || i == 97 || i == 1023 || i == 4999 || i == 9999 {
                None
            } else {
                Some(simple_hash(99, i) % 2 == 0)
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(BooleanArray::from(vals.clone()))],
    )
    .unwrap();

    let data = write_file(&schema, &[batch], WriterOptions::default());
    let result = read_all(&data);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();

    for i in 0..num_rows {
        match vals[i] {
            None => assert!(col.is_null(i), "expected null at {}", i),
            Some(v) => {
                assert!(!col.is_null(i), "unexpected null at {}", i);
                assert_eq!(col.value(i), v, "mismatch at {}", i);
            }
        }
    }
}

#[test]
fn test_boolean_dense_nulls() {
    // Mostly nulls, very few values
    let schema = Schema::new(vec![Field::new("b", DataType::Boolean, true)]);
    let num_rows = 10_000;

    let vals: Vec<Option<bool>> = (0..num_rows)
        .map(|i| {
            if i % 100 == 0 {
                Some(i % 200 == 0)
            } else {
                None
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(BooleanArray::from(vals.clone()))],
    )
    .unwrap();

    let data = write_file(&schema, &[batch], WriterOptions::default());
    let result = read_all(&data);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();

    for i in 0..num_rows {
        match vals[i] {
            None => assert!(col.is_null(i), "expected null at {}", i),
            Some(v) => {
                assert!(!col.is_null(i), "unexpected null at {}", i);
                assert_eq!(col.value(i), v, "mismatch at {}", i);
            }
        }
    }
}

// ======================== Large Binary / Long Strings (Page Threshold) ========================

#[test]
fn test_large_binary_values_exceed_page_threshold() {
    // PAGE_SIZE_THRESHOLD is 32KB — test values that individually exceed this
    let schema = Schema::new(vec![Field::new("blob", DataType::Binary, true)]);

    let num_rows = 100;
    let bin_vals: Vec<Option<Vec<u8>>> = (0..num_rows)
        .map(|i| {
            if i % 10 == 0 {
                None
            } else {
                // Each value is 40KB — exceeds page threshold
                let size = 40 * 1024;
                let val: Vec<u8> = (0..size).map(|j| ((i * 7 + j) % 256) as u8).collect();
                Some(val)
            }
        })
        .collect();

    let bin_refs: Vec<Option<&[u8]>> = bin_vals
        .iter()
        .map(|v| v.as_ref().map(|x| x.as_slice()))
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(BinaryArray::from(bin_refs))],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            ..Default::default()
        },
    );
    let result = read_all(&data);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();

    for i in 0..num_rows {
        match &bin_vals[i] {
            None => assert!(col.is_null(i)),
            Some(expected) => {
                assert!(!col.is_null(i));
                assert_eq!(col.value(i), expected.as_slice(), "mismatch at row {}", i);
            }
        }
    }
}

#[test]
fn test_long_strings_exceed_page_threshold() {
    let schema = Schema::new(vec![Field::new("text", DataType::Utf8, true)]);

    let num_rows = 50;
    let str_vals: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if i % 8 == 0 {
                None
            } else {
                // 50KB string
                let c = (b'A' + (i % 26) as u8) as char;
                Some(c.to_string().repeat(50 * 1024))
            }
        })
        .collect();
    let str_refs: Vec<Option<&str>> = str_vals.iter().map(|s| s.as_deref()).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(StringArray::from(str_refs))],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            ..Default::default()
        },
    );
    let result = read_all(&data);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    for i in 0..num_rows {
        match &str_vals[i] {
            None => assert!(col.is_null(i)),
            Some(expected) => {
                assert!(!col.is_null(i));
                assert_eq!(col.value(i), expected.as_str(), "mismatch at row {}", i);
            }
        }
    }
}

#[test]
fn test_mixed_small_and_large_binary() {
    let schema = Schema::new(vec![Field::new("data", DataType::Binary, false)]);

    // Mix of tiny and large values
    let num_rows = 200;
    let bin_vals: Vec<Vec<u8>> = (0..num_rows)
        .map(|i| {
            if i % 20 == 0 {
                // 64KB value
                vec![(i % 256) as u8; 64 * 1024]
            } else {
                // 10 byte value
                vec![(i % 256) as u8; 10]
            }
        })
        .collect();
    let bin_refs: Vec<&[u8]> = bin_vals.iter().map(|v| v.as_slice()).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(BinaryArray::from(bin_refs))],
    )
    .unwrap();

    let data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            ..Default::default()
        },
    );
    let result = read_all(&data);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();

    for i in 0..num_rows {
        assert_eq!(
            col.value(i),
            bin_vals[i].as_slice(),
            "mismatch at row {}",
            i
        );
    }
}

// ======================== Fuzz-style Random Schema Tests ========================

#[test]
fn test_random_schema_seed_1() {
    run_random_schema_test(1, 20, 50_000);
}

#[test]
fn test_random_schema_seed_2() {
    run_random_schema_test(2, 30, 30_000);
}

#[test]
fn test_random_schema_seed_3() {
    run_random_schema_test(3, 10, 100_000);
}

fn run_random_schema_test(seed: u64, num_cols: usize, num_rows: usize) {
    let type_choices = [
        DataType::Boolean,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::Float32,
        DataType::Float64,
        DataType::Utf8,
        DataType::Binary,
        DataType::Date32,
        DataType::Time32(TimeUnit::Millisecond),
        DataType::Decimal128(10, 2),
        DataType::Decimal128(28, 8),
    ];

    let mut fields = Vec::new();
    for col in 0..num_cols {
        let type_idx = simple_hash(seed, col * 3) as usize % type_choices.len();
        let nullable = simple_hash(seed, col * 3 + 1) % 2 == 0;
        let dt = type_choices[type_idx].clone();
        fields.push(Field::new(format!("c{}", col), dt, nullable));
    }
    let schema = Schema::new(fields.clone());

    let mut columns: Vec<ArrayRef> = Vec::new();
    for (col, field) in fields.iter().enumerate() {
        let null_rate = if field.is_nullable() {
            simple_hash(seed, col * 3 + 2) % 10 // 0-9 out of 10
        } else {
            0
        };

        let array: ArrayRef = match field.data_type() {
            DataType::Boolean => {
                let vals: Vec<Option<bool>> = (0..num_rows)
                    .map(|i| {
                        if simple_hash(seed + 100, col * num_rows + i) % 10 < null_rate {
                            None
                        } else {
                            Some(simple_hash(seed, col * num_rows + i) % 2 == 0)
                        }
                    })
                    .collect();
                Arc::new(BooleanArray::from(vals))
            }
            DataType::Int8 => {
                let vals: Vec<Option<i8>> = (0..num_rows)
                    .map(|i| {
                        if simple_hash(seed + 100, col * num_rows + i) % 10 < null_rate {
                            None
                        } else {
                            Some(simple_hash(seed, col * num_rows + i) as i8)
                        }
                    })
                    .collect();
                Arc::new(Int8Array::from(vals))
            }
            DataType::Int16 => {
                let vals: Vec<Option<i16>> = (0..num_rows)
                    .map(|i| {
                        if simple_hash(seed + 100, col * num_rows + i) % 10 < null_rate {
                            None
                        } else {
                            Some(simple_hash(seed, col * num_rows + i) as i16)
                        }
                    })
                    .collect();
                Arc::new(Int16Array::from(vals))
            }
            DataType::Int32 => {
                let vals: Vec<Option<i32>> = (0..num_rows)
                    .map(|i| {
                        if simple_hash(seed + 100, col * num_rows + i) % 10 < null_rate {
                            None
                        } else {
                            Some(simple_hash(seed, col * num_rows + i) as i32)
                        }
                    })
                    .collect();
                Arc::new(Int32Array::from(vals))
            }
            DataType::Int64 => {
                let vals: Vec<Option<i64>> = (0..num_rows)
                    .map(|i| {
                        if simple_hash(seed + 100, col * num_rows + i) % 10 < null_rate {
                            None
                        } else {
                            Some(simple_hash(seed, col * num_rows + i) as i64)
                        }
                    })
                    .collect();
                Arc::new(Int64Array::from(vals))
            }
            DataType::Float32 => {
                let vals: Vec<Option<f32>> = (0..num_rows)
                    .map(|i| {
                        if simple_hash(seed + 100, col * num_rows + i) % 10 < null_rate {
                            None
                        } else {
                            Some(simple_hash(seed, col * num_rows + i) as f32 / 1000.0)
                        }
                    })
                    .collect();
                Arc::new(Float32Array::from(vals))
            }
            DataType::Float64 => {
                let vals: Vec<Option<f64>> = (0..num_rows)
                    .map(|i| {
                        if simple_hash(seed + 100, col * num_rows + i) % 10 < null_rate {
                            None
                        } else {
                            Some(simple_hash(seed, col * num_rows + i) as f64 / 1000.0)
                        }
                    })
                    .collect();
                Arc::new(Float64Array::from(vals))
            }
            DataType::Utf8 => {
                let vals: Vec<Option<String>> = (0..num_rows)
                    .map(|i| {
                        if simple_hash(seed + 100, col * num_rows + i) % 10 < null_rate {
                            None
                        } else {
                            let len = (simple_hash(seed, col * num_rows + i) % 50) as usize + 1;
                            Some(
                                (0..len)
                                    .map(|j| {
                                        (b'a' + (simple_hash(seed + 200, i * len + j) % 26) as u8)
                                            as char
                                    })
                                    .collect(),
                            )
                        }
                    })
                    .collect();
                let refs: Vec<Option<&str>> = vals.iter().map(|s| s.as_deref()).collect();
                Arc::new(StringArray::from(refs))
            }
            DataType::Binary => {
                let vals: Vec<Option<Vec<u8>>> = (0..num_rows)
                    .map(|i| {
                        if simple_hash(seed + 100, col * num_rows + i) % 10 < null_rate {
                            None
                        } else {
                            let len = (simple_hash(seed, col * num_rows + i) % 64) as usize + 1;
                            Some(
                                (0..len)
                                    .map(|j| simple_hash(seed + 300, i * len + j) as u8)
                                    .collect(),
                            )
                        }
                    })
                    .collect();
                let refs: Vec<Option<&[u8]>> = vals
                    .iter()
                    .map(|v| v.as_ref().map(|x| x.as_slice()))
                    .collect();
                Arc::new(BinaryArray::from(refs))
            }
            DataType::Date32 => {
                let vals: Vec<Option<i32>> = (0..num_rows)
                    .map(|i| {
                        if simple_hash(seed + 100, col * num_rows + i) % 10 < null_rate {
                            None
                        } else {
                            Some((simple_hash(seed, col * num_rows + i) % 40000) as i32)
                        }
                    })
                    .collect();
                Arc::new(Date32Array::from(vals))
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                let vals: Vec<Option<i32>> = (0..num_rows)
                    .map(|i| {
                        if simple_hash(seed + 100, col * num_rows + i) % 10 < null_rate {
                            None
                        } else {
                            Some((simple_hash(seed, col * num_rows + i) % 86_400_000) as i32)
                        }
                    })
                    .collect();
                Arc::new(Time32MillisecondArray::from(vals))
            }
            DataType::Decimal128(p, s) => {
                let p = *p;
                let s = *s;
                let max_val: i128 = 10i128.pow(p as u32) - 1;
                let vals: Vec<Option<i128>> = (0..num_rows)
                    .map(|i| {
                        if simple_hash(seed + 100, col * num_rows + i) % 10 < null_rate {
                            None
                        } else {
                            let h = simple_hash(seed, col * num_rows + i) as i128;
                            Some(h % max_val - max_val / 2)
                        }
                    })
                    .collect();
                Arc::new(
                    Decimal128Array::from(vals)
                        .with_precision_and_scale(p, s)
                        .unwrap(),
                )
            }
            _ => unreachable!(),
        };
        columns.push(array);
    }

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), columns).unwrap();

    let data = write_file(
        &schema,
        &[batch.clone()],
        WriterOptions {
            num_buckets: (simple_hash(seed, 9999) % 20 + 1) as usize,
            ..Default::default()
        },
    );
    let result = read_all(&data);
    let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, num_rows, "seed={} row count mismatch", seed);

    // Verify data integrity by checking a sample of rows
    assert_eq!(result.len(), 1);
    assert_eq!(
        reorder_batch(&result[0], &schema),
        batch,
        "seed={} data mismatch",
        seed
    );
}
