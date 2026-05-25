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

//! Determinism, idempotency, and re-roundtrip tests for the Mosaic file format.
//!
//! These tests verify critical properties for format stability:
//! - Same input data with same options produces byte-identical output (determinism).
//! - write -> read -> write -> read cycles produce stable bytes and data (re-roundtrip).
//! - File metadata (size, footer offsets) is consistent across identical writes.

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

fn read_projected(data: &[u8], columns: &[usize]) -> Vec<RecordBatch> {
    let input = ByteArrayInputFile {
        data: data.to_vec(),
    };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();
    let mut result = Vec::new();
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader_projected(rg, columns).unwrap();
        result.push(rg_reader.read_columns().unwrap());
    }
    result
}

fn make_default_options() -> WriterOptions {
    WriterOptions::default()
}

/// Compare two sets of batches for logical equality, handling column reordering.
/// Columns are matched by name. Both sides must have the same total row count
/// and identical column data when concatenated.
fn batches_equal_unordered(a: &[RecordBatch], b: &[RecordBatch]) -> bool {
    let rows_a: usize = a.iter().map(|b| b.num_rows()).sum();
    let rows_b: usize = b.iter().map(|b| b.num_rows()).sum();
    if rows_a != rows_b {
        return false;
    }
    if rows_a == 0 {
        return true;
    }

    // Get column names from the first batch of each side
    let schema_a = a[0].schema();
    let schema_b = b[0].schema();

    // Every column in a must exist in b
    for field_a in schema_a.fields() {
        let col_name = field_a.name();
        if schema_b.index_of(col_name).is_err() {
            return false;
        }
    }

    // Concatenate all arrays for each column from both sides, then compare
    for field_a in schema_a.fields() {
        let col_name = field_a.name();

        // Collect column data from a
        let mut arrays_a: Vec<ArrayRef> = Vec::new();
        for batch in a {
            let idx = batch.schema().index_of(col_name).unwrap();
            arrays_a.push(batch.column(idx).clone());
        }

        // Collect column data from b
        let mut arrays_b: Vec<ArrayRef> = Vec::new();
        for batch in b {
            let idx = batch.schema().index_of(col_name).unwrap();
            arrays_b.push(batch.column(idx).clone());
        }

        // Compare element-by-element
        let mut offset_a = 0usize;
        let mut batch_idx_a = 0usize;
        let mut offset_b = 0usize;
        let mut batch_idx_b = 0usize;

        let mut row = 0;
        while row < rows_a {
            let arr_a = &arrays_a[batch_idx_a];
            let arr_b = &arrays_b[batch_idx_b];
            let remaining_a = arr_a.len() - offset_a;
            let remaining_b = arr_b.len() - offset_b;
            let chunk = remaining_a.min(remaining_b);

            let slice_a = arr_a.slice(offset_a, chunk);
            let slice_b = arr_b.slice(offset_b, chunk);
            if slice_a != slice_b {
                return false;
            }

            offset_a += chunk;
            offset_b += chunk;
            row += chunk;

            if offset_a == arr_a.len() {
                batch_idx_a += 1;
                offset_a = 0;
            }
            if offset_b == arr_b.len() {
                batch_idx_b += 1;
                offset_b = 0;
            }
        }
    }

    true
}

/// Write batches -> read -> write -> read -> write -> return stable cycle data.
///
/// Since the reader returns columns in sorted order, the first write (with original
/// schema) may differ from subsequent writes (with sorted schema). We therefore
/// compare cycle 2 vs cycle 3 for byte-level stability, and cycle 1 vs cycle 2
/// for logical data equality.
///
/// Returns (cycle2_bytes, cycle3_bytes, cycle1_batches, cycle2_batches).
fn reroundtrip(
    schema: &Schema,
    batches: &[RecordBatch],
    options_fn: fn() -> WriterOptions,
) -> (Vec<u8>, Vec<u8>, Vec<RecordBatch>, Vec<RecordBatch>) {
    // Cycle 1: write with original schema
    let data1 = write_file(schema, batches, options_fn());
    let batches1 = read_all(&data1);
    assert!(!batches1.is_empty(), "cycle 1 read produced no batches");

    // Cycle 2: write with sorted schema (from reader)
    let sorted_schema = batches1[0].schema();
    let data2 = write_file(&sorted_schema, &batches1, options_fn());
    let batches2 = read_all(&data2);

    // Cycle 3: write again from cycle 2 read to verify stability
    let sorted_schema2 = batches2[0].schema();
    let data3 = write_file(&sorted_schema2, &batches2, options_fn());

    (data2, data3, batches1, batches2)
}

// ======================== Determinism Tests ========================

#[test]
fn test_deterministic_int_output() {
    let schema = Schema::new(vec![
        Field::new("ax", DataType::Int32, false),
        Field::new("by", DataType::Int64, false),
    ]);

    let num_rows = 100_000;
    let i32_vals: Vec<i32> = (0..num_rows).map(|i| simple_hash(1, i) as i32).collect();
    let i64_vals: Vec<i64> = (0..num_rows).map(|i| simple_hash(2, i) as i64).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(i32_vals)),
            Arc::new(Int64Array::from(i64_vals)),
        ],
    )
    .unwrap();

    let options = || WriterOptions::default();

    let data1 = write_file(&schema, &[batch.clone()], options());
    let data2 = write_file(&schema, &[batch.clone()], options());

    assert_eq!(
        data1.len(),
        data2.len(),
        "file sizes differ: {} vs {}",
        data1.len(),
        data2.len()
    );
    assert_eq!(
        data1, data2,
        "byte-level mismatch for deterministic int output"
    );

    println!(
        "test_deterministic_int_output: PASSED - {} bytes, byte-identical",
        data1.len()
    );
}

#[test]
fn test_deterministic_string_output() {
    let schema = Schema::new(vec![
        Field::new("cx", DataType::Utf8, true),
        Field::new("dy", DataType::Binary, true),
    ]);

    let num_rows = 50_000;
    let str_vals: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if simple_hash(10, i) % 7 == 0 {
                None
            } else {
                let len = (simple_hash(11, i) % 100) as usize + 1;
                Some(
                    (0..len)
                        .map(|j| (b'a' + (simple_hash(12, i * 100 + j) % 26) as u8) as char)
                        .collect(),
                )
            }
        })
        .collect();
    let str_refs: Vec<Option<&str>> = str_vals.iter().map(|s| s.as_deref()).collect();

    let bin_vals: Vec<Option<Vec<u8>>> = (0..num_rows)
        .map(|i| {
            if simple_hash(20, i) % 9 == 0 {
                None
            } else {
                let len = (simple_hash(21, i) % 64) as usize + 1;
                Some(
                    (0..len)
                        .map(|j| simple_hash(22, i * 64 + j) as u8)
                        .collect(),
                )
            }
        })
        .collect();
    let bin_refs: Vec<Option<&[u8]>> = bin_vals
        .iter()
        .map(|v| v.as_ref().map(|x| x.as_slice()))
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(StringArray::from(str_refs)),
            Arc::new(BinaryArray::from(bin_refs)),
        ],
    )
    .unwrap();

    let options = || WriterOptions::default();

    let data1 = write_file(&schema, &[batch.clone()], options());
    let data2 = write_file(&schema, &[batch.clone()], options());

    assert_eq!(data1.len(), data2.len(), "file sizes differ");
    assert_eq!(
        data1, data2,
        "byte-level mismatch for deterministic string output"
    );

    println!(
        "test_deterministic_string_output: PASSED - {} bytes, byte-identical",
        data1.len()
    );
}

#[test]
fn test_deterministic_all_types() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Boolean, true),
        Field::new("b", DataType::Int8, true),
        Field::new("c", DataType::Int16, true),
        Field::new("d", DataType::Int32, true),
        Field::new("e", DataType::Int64, true),
        Field::new("f", DataType::Float32, true),
        Field::new("g", DataType::Float64, true),
        Field::new("h", DataType::Date32, true),
        Field::new("j", DataType::Time32(TimeUnit::Millisecond), true),
        Field::new("k", DataType::Utf8, true),
        Field::new("m", DataType::Binary, true),
        Field::new("n", DataType::Decimal128(10, 2), true),
        Field::new("p", DataType::Decimal128(38, 10), true),
        Field::new("q", DataType::Timestamp(TimeUnit::Millisecond, None), true),
        Field::new("r", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("s", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
    ]);

    let num_rows = 50_000;

    let make_batch = || {
        let bools: Vec<Option<bool>> = (0..num_rows)
            .map(|i| {
                if simple_hash(100, i) % 8 == 0 {
                    None
                } else {
                    Some(simple_hash(101, i) % 2 == 0)
                }
            })
            .collect();
        let i8s: Vec<Option<i8>> = (0..num_rows)
            .map(|i| {
                if simple_hash(110, i) % 10 == 0 {
                    None
                } else {
                    Some(simple_hash(111, i) as i8)
                }
            })
            .collect();
        let i16s: Vec<Option<i16>> = (0..num_rows)
            .map(|i| {
                if simple_hash(120, i) % 10 == 0 {
                    None
                } else {
                    Some(simple_hash(121, i) as i16)
                }
            })
            .collect();
        let i32s: Vec<Option<i32>> = (0..num_rows)
            .map(|i| {
                if simple_hash(130, i) % 10 == 0 {
                    None
                } else {
                    Some(simple_hash(131, i) as i32)
                }
            })
            .collect();
        let i64s: Vec<Option<i64>> = (0..num_rows)
            .map(|i| {
                if simple_hash(140, i) % 10 == 0 {
                    None
                } else {
                    Some(simple_hash(141, i) as i64)
                }
            })
            .collect();
        let f32s: Vec<Option<f32>> = (0..num_rows)
            .map(|i| {
                if simple_hash(150, i) % 10 == 0 {
                    None
                } else {
                    Some(simple_hash(151, i) as f32 / 1000.0)
                }
            })
            .collect();
        let f64s: Vec<Option<f64>> = (0..num_rows)
            .map(|i| {
                if simple_hash(160, i) % 10 == 0 {
                    None
                } else {
                    Some(simple_hash(161, i) as f64 / 1000.0)
                }
            })
            .collect();
        let dates: Vec<Option<i32>> = (0..num_rows)
            .map(|i| {
                if simple_hash(170, i) % 10 == 0 {
                    None
                } else {
                    Some((simple_hash(171, i) % 40000) as i32)
                }
            })
            .collect();
        let times: Vec<Option<i32>> = (0..num_rows)
            .map(|i| {
                if simple_hash(180, i) % 10 == 0 {
                    None
                } else {
                    Some((simple_hash(181, i) % 86_400_000) as i32)
                }
            })
            .collect();
        let strings: Vec<Option<String>> = (0..num_rows)
            .map(|i| {
                if simple_hash(190, i) % 8 == 0 {
                    None
                } else {
                    let len = (simple_hash(191, i) % 30) as usize + 1;
                    Some(
                        (0..len)
                            .map(|j| (b'a' + (simple_hash(192, i * 30 + j) % 26) as u8) as char)
                            .collect(),
                    )
                }
            })
            .collect();
        let str_refs: Vec<Option<&str>> = strings.iter().map(|s| s.as_deref()).collect();
        let bins: Vec<Option<Vec<u8>>> = (0..num_rows)
            .map(|i| {
                if simple_hash(200, i) % 8 == 0 {
                    None
                } else {
                    let len = (simple_hash(201, i) % 32) as usize + 1;
                    Some(
                        (0..len)
                            .map(|j| simple_hash(202, i * 32 + j) as u8)
                            .collect(),
                    )
                }
            })
            .collect();
        let bin_refs: Vec<Option<&[u8]>> = bins
            .iter()
            .map(|v| v.as_ref().map(|x| x.as_slice()))
            .collect();
        let dec_smalls: Vec<Option<i128>> = (0..num_rows)
            .map(|i| {
                if simple_hash(210, i) % 10 == 0 {
                    None
                } else {
                    Some((simple_hash(211, i) % 99_999_999) as i128)
                }
            })
            .collect();
        let dec_larges: Vec<Option<i128>> = (0..num_rows)
            .map(|i| {
                if simple_hash(220, i) % 10 == 0 {
                    None
                } else {
                    Some(simple_hash(221, i) as i128 * 1_000_000_000)
                }
            })
            .collect();
        let ts_ms: Vec<Option<i64>> = (0..num_rows)
            .map(|i| {
                if simple_hash(230, i) % 10 == 0 {
                    None
                } else {
                    Some(simple_hash(231, i) as i64)
                }
            })
            .collect();
        let ts_us: Vec<Option<i64>> = (0..num_rows)
            .map(|i| {
                if simple_hash(240, i) % 10 == 0 {
                    None
                } else {
                    Some(simple_hash(241, i) as i64)
                }
            })
            .collect();
        let ts_ns: Vec<Option<i64>> = (0..num_rows)
            .map(|i| {
                if simple_hash(250, i) % 10 == 0 {
                    None
                } else {
                    Some(simple_hash(251, i) as i64)
                }
            })
            .collect();

        RecordBatch::try_new(
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
                    Decimal128Array::from(dec_smalls)
                        .with_precision_and_scale(10, 2)
                        .unwrap(),
                ),
                Arc::new(
                    Decimal128Array::from(dec_larges)
                        .with_precision_and_scale(38, 10)
                        .unwrap(),
                ),
                Arc::new(TimestampMillisecondArray::from(ts_ms)),
                Arc::new(TimestampMicrosecondArray::from(ts_us)),
                Arc::new(TimestampNanosecondArray::from(ts_ns)),
            ],
        )
        .unwrap()
    };

    let batch = make_batch();

    let options = || WriterOptions::default();

    let data1 = write_file(&schema, &[batch.clone()], options());
    let data2 = write_file(&schema, &[batch.clone()], options());

    assert_eq!(data1.len(), data2.len(), "file sizes differ");
    assert_eq!(
        data1, data2,
        "byte-level mismatch for all-types deterministic output"
    );

    println!(
        "test_deterministic_all_types: PASSED - {} bytes, byte-identical",
        data1.len()
    );
}

#[test]
fn test_deterministic_with_nulls() {
    let schema = Schema::new(vec![
        Field::new("w", DataType::Int32, true),
        Field::new("x", DataType::Utf8, true),
        Field::new("y", DataType::Float64, true),
        Field::new("z", DataType::Binary, true),
    ]);

    let num_rows = 50_000;

    // Various null patterns: dense nulls, sparse nulls, alternating, runs
    let i32_vals: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            // Dense nulls in first quarter
            if i < num_rows / 4 {
                if i % 2 == 0 {
                    None
                } else {
                    Some(i as i32)
                }
            }
            // Sparse nulls in second quarter
            else if i < num_rows / 2 {
                if i % 100 == 0 {
                    None
                } else {
                    Some(i as i32)
                }
            }
            // Runs of nulls in third quarter
            else if i < 3 * num_rows / 4 {
                if (i / 50) % 2 == 0 {
                    None
                } else {
                    Some(i as i32)
                }
            }
            // All non-null in last quarter
            else {
                Some(i as i32)
            }
        })
        .collect();

    let str_vals: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if simple_hash(50, i) % 3 == 0 {
                None
            } else {
                Some(format!("val_{}", i))
            }
        })
        .collect();
    let str_refs: Vec<Option<&str>> = str_vals.iter().map(|s| s.as_deref()).collect();

    let f64_vals: Vec<Option<f64>> = (0..num_rows)
        .map(|i| {
            if simple_hash(60, i) % 5 == 0 {
                None
            } else {
                Some(i as f64 * 0.01)
            }
        })
        .collect();

    let bin_vals: Vec<Option<Vec<u8>>> = (0..num_rows)
        .map(|i| {
            if simple_hash(70, i) % 4 == 0 {
                None
            } else {
                Some(vec![(i % 256) as u8; (i % 20) + 1])
            }
        })
        .collect();
    let bin_refs: Vec<Option<&[u8]>> = bin_vals
        .iter()
        .map(|v| v.as_ref().map(|x| x.as_slice()))
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(i32_vals)),
            Arc::new(StringArray::from(str_refs)),
            Arc::new(Float64Array::from(f64_vals)),
            Arc::new(BinaryArray::from(bin_refs)),
        ],
    )
    .unwrap();

    let options = || WriterOptions::default();

    let data1 = write_file(&schema, &[batch.clone()], options());
    let data2 = write_file(&schema, &[batch.clone()], options());

    assert_eq!(data1.len(), data2.len(), "file sizes differ");
    assert_eq!(
        data1, data2,
        "byte-level mismatch for null-pattern deterministic output"
    );

    println!(
        "test_deterministic_with_nulls: PASSED - {} bytes, byte-identical",
        data1.len()
    );
}

#[test]
fn test_deterministic_with_zstd() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, true),
    ]);

    let num_rows = 50_000;
    let ids: Vec<i64> = (0..num_rows as i64).collect();
    let strs: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if i % 10 == 0 {
                None
            } else {
                Some(format!("row_{:08}_data", i))
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

    let options = || WriterOptions {
        compression: spec::COMPRESSION_ZSTD,
        zstd_level: 3,
        ..Default::default()
    };

    let data1 = write_file(&schema, &[batch.clone()], options());
    let data2 = write_file(&schema, &[batch.clone()], options());

    assert_eq!(data1.len(), data2.len(), "file sizes differ with ZSTD");
    assert_eq!(data1, data2, "byte-level mismatch with ZSTD compression");

    println!(
        "test_deterministic_with_zstd: PASSED - {} bytes, byte-identical",
        data1.len()
    );
}

#[test]
fn test_deterministic_with_no_compression() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int32, true),
        Field::new("label", DataType::Utf8, false),
    ]);

    let num_rows = 50_000;
    let ids: Vec<i64> = (0..num_rows as i64).collect();
    let vals: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            if i % 11 == 0 {
                None
            } else {
                Some(simple_hash(42, i) as i32)
            }
        })
        .collect();
    let labels: Vec<String> = (0..num_rows)
        .map(|i| format!("label_{}", i % 100))
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

    let options = || WriterOptions {
        compression: spec::COMPRESSION_NONE,
        ..Default::default()
    };

    let data1 = write_file(&schema, &[batch.clone()], options());
    let data2 = write_file(&schema, &[batch.clone()], options());

    assert_eq!(
        data1.len(),
        data2.len(),
        "file sizes differ with no compression"
    );
    assert_eq!(data1, data2, "byte-level mismatch with no compression");

    println!(
        "test_deterministic_with_no_compression: PASSED - {} bytes, byte-identical",
        data1.len()
    );
}

#[test]
fn test_deterministic_multiple_row_groups() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, false),
    ]);

    let num_rows = 100_000;
    let batch_size = 1_000;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;
        let ids: Vec<i64> = (batch_start..batch_start + count)
            .map(|i| i as i64)
            .collect();
        let strs: Vec<String> = (0..count)
            .map(|i| format!("row_{:08}_padding_{}", batch_start + i, "x".repeat(50)))
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

    let options = || WriterOptions {
        num_buckets: 1,
        row_group_max_size: 64 * 1024, // 64KB to force multiple row groups
        ..Default::default()
    };

    // Write 1
    let out1 = MemOutputFile::new();
    let mut writer1 = MosaicWriter::new(out1, &schema, options()).unwrap();
    for batch in &batches {
        writer1.write_batch(batch).unwrap();
    }
    writer1.close().unwrap();
    let data1 = writer1.output().buf.clone();

    // Write 2
    let out2 = MemOutputFile::new();
    let mut writer2 = MosaicWriter::new(out2, &schema, options()).unwrap();
    for batch in &batches {
        writer2.write_batch(batch).unwrap();
    }
    writer2.close().unwrap();
    let data2 = writer2.output().buf.clone();

    // Verify multiple row groups were created
    let input = ByteArrayInputFile {
        data: data1.clone(),
    };
    let reader = MosaicReader::new(input, data1.len() as u64).unwrap();
    assert!(
        reader.num_row_groups() > 1,
        "expected multiple row groups, got {}",
        reader.num_row_groups()
    );

    assert_eq!(data1.len(), data2.len(), "file sizes differ");
    assert_eq!(data1, data2, "byte-level mismatch for multiple row groups");

    println!(
        "test_deterministic_multiple_row_groups: PASSED - {} bytes, {} row groups, byte-identical",
        data1.len(),
        reader.num_row_groups()
    );
}

#[test]
fn test_deterministic_different_batch_splits() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int32, false),
    ]);

    let num_rows = 100_000;

    // Generate the same logical data
    let ids: Vec<i64> = (0..num_rows as i64).collect();
    let vals: Vec<i32> = (0..num_rows).map(|i| simple_hash(99, i) as i32).collect();

    // Write 1: single big batch
    let batch_single = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids.clone())),
            Arc::new(Int32Array::from(vals.clone())),
        ],
    )
    .unwrap();

    let options = || WriterOptions {
        num_buckets: 1,
        row_group_max_size: 256 * 1024 * 1024, // large enough for single row group
        ..Default::default()
    };

    let data_single = write_file(&schema, &[batch_single], options());

    // Write 2: 10 batches of 10K each
    let mut batches_split = Vec::new();
    let chunk = num_rows / 10;
    for c in 0..10 {
        let start = c * chunk;
        let end = start + chunk;
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(ids[start..end].to_vec())),
                Arc::new(Int32Array::from(vals[start..end].to_vec())),
            ],
        )
        .unwrap();
        batches_split.push(batch);
    }

    let data_split = write_file(&schema, &batches_split, options());

    // Both should roundtrip correctly (but may or may not be byte-identical
    // since row group boundaries may differ)
    let result_single = read_all(&data_single);
    let result_split = read_all(&data_split);

    assert!(
        batches_equal_unordered(&result_single, &result_split),
        "data mismatch between single batch and split batch writes"
    );

    println!(
        "test_deterministic_different_batch_splits: PASSED - single: {} bytes, split: {} bytes, data matches",
        data_single.len(),
        data_split.len()
    );
}

// ======================== Idempotency / Re-roundtrip Tests ========================

#[test]
fn test_reroundtrip_int_data() {
    let schema = Schema::new(vec![
        Field::new("ax", DataType::Int32, false),
        Field::new("by", DataType::Int64, false),
    ]);

    let num_rows = 100_000;
    let i32_vals: Vec<i32> = (0..num_rows).map(|i| simple_hash(1, i) as i32).collect();
    let i64_vals: Vec<i64> = (0..num_rows).map(|i| simple_hash(2, i) as i64).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(i32_vals)),
            Arc::new(Int64Array::from(i64_vals)),
        ],
    )
    .unwrap();

    let (cycle2, cycle3, batches1, batches2) = reroundtrip(&schema, &[batch], make_default_options);

    // Cycle 1 read should equal cycle 2 read (data stability)
    assert!(
        batches_equal_unordered(&batches1, &batches2),
        "re-roundtrip data mismatch for int data"
    );

    // Cycle 2 bytes should equal cycle 3 bytes (byte-level stability)
    assert_eq!(
        cycle2.len(),
        cycle3.len(),
        "re-roundtrip file sizes differ: {} vs {}",
        cycle2.len(),
        cycle3.len()
    );
    assert_eq!(cycle2, cycle3, "re-roundtrip byte mismatch for int data");

    println!(
        "test_reroundtrip_int_data: PASSED - {} bytes, stable across cycles",
        cycle2.len()
    );
}

#[test]
fn test_reroundtrip_string_data() {
    let schema = Schema::new(vec![
        Field::new("cx", DataType::Utf8, true),
        Field::new("dy", DataType::Binary, true),
    ]);

    let num_rows = 50_000;
    let str_vals: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if simple_hash(30, i) % 7 == 0 {
                None
            } else {
                let len = (simple_hash(31, i) % 50) as usize + 1;
                Some(
                    (0..len)
                        .map(|j| (b'a' + (simple_hash(32, i * 50 + j) % 26) as u8) as char)
                        .collect(),
                )
            }
        })
        .collect();
    let str_refs: Vec<Option<&str>> = str_vals.iter().map(|s| s.as_deref()).collect();

    let bin_vals: Vec<Option<Vec<u8>>> = (0..num_rows)
        .map(|i| {
            if simple_hash(40, i) % 9 == 0 {
                None
            } else {
                let len = (simple_hash(41, i) % 32) as usize + 1;
                Some(
                    (0..len)
                        .map(|j| simple_hash(42, i * 32 + j) as u8)
                        .collect(),
                )
            }
        })
        .collect();
    let bin_refs: Vec<Option<&[u8]>> = bin_vals
        .iter()
        .map(|v| v.as_ref().map(|x| x.as_slice()))
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(StringArray::from(str_refs)),
            Arc::new(BinaryArray::from(bin_refs)),
        ],
    )
    .unwrap();

    let (cycle2, cycle3, batches1, batches2) = reroundtrip(&schema, &[batch], make_default_options);

    assert!(
        batches_equal_unordered(&batches1, &batches2),
        "re-roundtrip data mismatch for string data"
    );
    assert_eq!(cycle2, cycle3, "re-roundtrip byte mismatch for string data");

    println!(
        "test_reroundtrip_string_data: PASSED - {} bytes, stable",
        cycle2.len()
    );
}

#[test]
fn test_reroundtrip_all_types() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Boolean, true),
        Field::new("b", DataType::Int8, true),
        Field::new("c", DataType::Int16, true),
        Field::new("d", DataType::Int32, true),
        Field::new("e", DataType::Int64, true),
        Field::new("f", DataType::Float32, true),
        Field::new("g", DataType::Float64, true),
        Field::new("h", DataType::Date32, true),
        Field::new("j", DataType::Time32(TimeUnit::Millisecond), true),
        Field::new("k", DataType::Utf8, true),
        Field::new("m", DataType::Binary, true),
        Field::new("n", DataType::Decimal128(10, 2), true),
        Field::new("p", DataType::Decimal128(38, 10), true),
        Field::new("q", DataType::Timestamp(TimeUnit::Millisecond, None), true),
        Field::new("r", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("s", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
    ]);

    let num_rows = 30_000;

    let bools: Vec<Option<bool>> = (0..num_rows)
        .map(|i| {
            if simple_hash(300, i) % 8 == 0 {
                None
            } else {
                Some(simple_hash(301, i) % 2 == 0)
            }
        })
        .collect();
    let i8s: Vec<Option<i8>> = (0..num_rows)
        .map(|i| {
            if simple_hash(310, i) % 10 == 0 {
                None
            } else {
                Some(simple_hash(311, i) as i8)
            }
        })
        .collect();
    let i16s: Vec<Option<i16>> = (0..num_rows)
        .map(|i| {
            if simple_hash(320, i) % 10 == 0 {
                None
            } else {
                Some(simple_hash(321, i) as i16)
            }
        })
        .collect();
    let i32s: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            if simple_hash(330, i) % 10 == 0 {
                None
            } else {
                Some(simple_hash(331, i) as i32)
            }
        })
        .collect();
    let i64s: Vec<Option<i64>> = (0..num_rows)
        .map(|i| {
            if simple_hash(340, i) % 10 == 0 {
                None
            } else {
                Some(simple_hash(341, i) as i64)
            }
        })
        .collect();
    let f32s: Vec<Option<f32>> = (0..num_rows)
        .map(|i| {
            if simple_hash(350, i) % 10 == 0 {
                None
            } else {
                Some(simple_hash(351, i) as f32 / 1000.0)
            }
        })
        .collect();
    let f64s: Vec<Option<f64>> = (0..num_rows)
        .map(|i| {
            if simple_hash(360, i) % 10 == 0 {
                None
            } else {
                Some(simple_hash(361, i) as f64 / 1000.0)
            }
        })
        .collect();
    let dates: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            if simple_hash(370, i) % 10 == 0 {
                None
            } else {
                Some((simple_hash(371, i) % 40000) as i32)
            }
        })
        .collect();
    let times: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            if simple_hash(380, i) % 10 == 0 {
                None
            } else {
                Some((simple_hash(381, i) % 86_400_000) as i32)
            }
        })
        .collect();
    let strings: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if simple_hash(390, i) % 8 == 0 {
                None
            } else {
                let len = (simple_hash(391, i) % 20) as usize + 1;
                Some(
                    (0..len)
                        .map(|j| (b'a' + (simple_hash(392, i * 20 + j) % 26) as u8) as char)
                        .collect(),
                )
            }
        })
        .collect();
    let str_refs: Vec<Option<&str>> = strings.iter().map(|s| s.as_deref()).collect();
    let bins: Vec<Option<Vec<u8>>> = (0..num_rows)
        .map(|i| {
            if simple_hash(400, i) % 8 == 0 {
                None
            } else {
                let len = (simple_hash(401, i) % 16) as usize + 1;
                Some(
                    (0..len)
                        .map(|j| simple_hash(402, i * 16 + j) as u8)
                        .collect(),
                )
            }
        })
        .collect();
    let bin_refs: Vec<Option<&[u8]>> = bins
        .iter()
        .map(|v| v.as_ref().map(|x| x.as_slice()))
        .collect();
    let dec_smalls: Vec<Option<i128>> = (0..num_rows)
        .map(|i| {
            if simple_hash(410, i) % 10 == 0 {
                None
            } else {
                Some((simple_hash(411, i) % 99_999_999) as i128)
            }
        })
        .collect();
    let dec_larges: Vec<Option<i128>> = (0..num_rows)
        .map(|i| {
            if simple_hash(420, i) % 10 == 0 {
                None
            } else {
                Some(simple_hash(421, i) as i128 * 1_000_000_000)
            }
        })
        .collect();
    let ts_ms: Vec<Option<i64>> = (0..num_rows)
        .map(|i| {
            if simple_hash(430, i) % 10 == 0 {
                None
            } else {
                Some(simple_hash(431, i) as i64)
            }
        })
        .collect();
    let ts_us: Vec<Option<i64>> = (0..num_rows)
        .map(|i| {
            if simple_hash(440, i) % 10 == 0 {
                None
            } else {
                Some(simple_hash(441, i) as i64)
            }
        })
        .collect();
    let ts_ns: Vec<Option<i64>> = (0..num_rows)
        .map(|i| {
            if simple_hash(450, i) % 10 == 0 {
                None
            } else {
                Some(simple_hash(451, i) as i64)
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
                Decimal128Array::from(dec_smalls)
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from(dec_larges)
                    .with_precision_and_scale(38, 10)
                    .unwrap(),
            ),
            Arc::new(TimestampMillisecondArray::from(ts_ms)),
            Arc::new(TimestampMicrosecondArray::from(ts_us)),
            Arc::new(TimestampNanosecondArray::from(ts_ns)),
        ],
    )
    .unwrap();

    let (cycle2, cycle3, batches1, batches2) = reroundtrip(&schema, &[batch], make_default_options);

    assert!(
        batches_equal_unordered(&batches1, &batches2),
        "re-roundtrip data mismatch for all types"
    );
    assert_eq!(cycle2, cycle3, "re-roundtrip byte mismatch for all types");

    println!(
        "test_reroundtrip_all_types: PASSED - {} bytes, stable",
        cycle2.len()
    );
}

#[test]
fn test_reroundtrip_with_nulls() {
    let schema = Schema::new(vec![
        Field::new("w", DataType::Int32, true),
        Field::new("x", DataType::Utf8, true),
        Field::new("y", DataType::Float64, true),
        Field::new("z", DataType::Int64, true),
    ]);

    let num_rows = 50_000;

    let mostly_null: Vec<Option<i32>> = (0..num_rows)
        .map(|i| if i % 100 == 0 { Some(i as i32) } else { None })
        .collect();
    let half_null: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if i % 2 == 0 {
                None
            } else {
                Some(format!("val_{}", i))
            }
        })
        .collect();
    let half_null_refs: Vec<Option<&str>> = half_null.iter().map(|s| s.as_deref()).collect();
    let rarely_null: Vec<Option<f64>> = (0..num_rows)
        .map(|i| {
            if i == 7 || i == 9999 || i == 49999 {
                None
            } else {
                Some(i as f64 * 0.001)
            }
        })
        .collect();
    let all_null: Vec<Option<i64>> = vec![None; num_rows];

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(mostly_null)),
            Arc::new(StringArray::from(half_null_refs)),
            Arc::new(Float64Array::from(rarely_null)),
            Arc::new(Int64Array::from(all_null)),
        ],
    )
    .unwrap();

    let (cycle2, cycle3, batches1, batches2) = reroundtrip(&schema, &[batch], make_default_options);

    assert!(
        batches_equal_unordered(&batches1, &batches2),
        "re-roundtrip data mismatch with nulls"
    );
    assert_eq!(cycle2, cycle3, "re-roundtrip byte mismatch with nulls");

    println!(
        "test_reroundtrip_with_nulls: PASSED - {} bytes, stable",
        cycle2.len()
    );
}

#[test]
fn test_reroundtrip_multiple_row_groups() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, false),
    ]);

    let num_rows = 50_000;
    let batch_size = 500;
    let mut batches = Vec::new();

    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;
        let ids: Vec<i64> = (batch_start..batch_start + count)
            .map(|i| i as i64)
            .collect();
        let strs: Vec<String> = (0..count)
            .map(|i| format!("row_{:08}_pad_{}", batch_start + i, "x".repeat(40)))
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

    let options_fn = || WriterOptions {
        num_buckets: 1,
        row_group_max_size: 32 * 1024, // 32KB to force multiple row groups
        ..Default::default()
    };

    // First write
    let out1 = MemOutputFile::new();
    let mut writer1 = MosaicWriter::new(out1, &schema, options_fn()).unwrap();
    for batch in &batches {
        writer1.write_batch(batch).unwrap();
    }
    writer1.close().unwrap();
    let data1 = writer1.output().buf.clone();

    // Read all row groups, concatenate
    let batches1 = read_all(&data1);
    let total_rows1: usize = batches1.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows1, num_rows);
    assert!(batches1.len() > 1, "expected multiple row groups");

    // Second write from read data
    let read_schema = batches1[0].schema();
    let out2 = MemOutputFile::new();
    let mut writer2 = MosaicWriter::new(out2, &read_schema, options_fn()).unwrap();
    for batch in &batches1 {
        writer2.write_batch(batch).unwrap();
    }
    writer2.close().unwrap();
    let data2 = writer2.output().buf.clone();

    // Second read
    let batches2 = read_all(&data2);

    assert!(
        batches_equal_unordered(&batches1, &batches2),
        "re-roundtrip data mismatch for multiple row groups"
    );

    println!(
        "test_reroundtrip_multiple_row_groups: PASSED - {} row groups, data matches",
        batches1.len()
    );
}

#[test]
fn test_reroundtrip_with_projection() {
    // Write 20 columns
    let num_cols = 20;
    let num_rows = 10_000;

    let fields: Vec<Field> = (0..num_cols)
        .map(|i| Field::new(format!("c{:02}", i), DataType::Int64, true))
        .collect();
    let schema = Schema::new(fields);

    let mut arrays: Vec<Arc<dyn Array>> = Vec::new();
    for col in 0..num_cols {
        let vals: Vec<Option<i64>> = (0..num_rows)
            .map(|i| {
                if simple_hash(500 + col as u64, i) % 10 == 0 {
                    None
                } else {
                    Some(simple_hash(600 + col as u64, i) as i64)
                }
            })
            .collect();
        arrays.push(Arc::new(Int64Array::from(vals)));
    }

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), arrays).unwrap();

    let data1 = write_file(&schema, &[batch], WriterOptions::default());

    // Read back with projection (only 5 columns: indices 0, 4, 9, 14, 19)
    let projected_cols = vec![0, 4, 9, 14, 19];
    let projected_batches = read_projected(&data1, &projected_cols);
    assert!(!projected_batches.is_empty());

    let projected_schema = projected_batches[0].schema();
    assert_eq!(projected_schema.fields().len(), 5);

    // Write the 5 columns
    let data2 = write_file(
        &projected_schema,
        &projected_batches,
        WriterOptions::default(),
    );

    // Read again
    let reread_batches = read_all(&data2);

    assert!(
        batches_equal_unordered(&projected_batches, &reread_batches),
        "re-roundtrip with projection: data mismatch"
    );

    println!("test_reroundtrip_with_projection: PASSED - 20 cols -> 5 projected, roundtrip stable");
}

#[test]
fn test_reroundtrip_chain_5_times() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int32, true),
        Field::new("label", DataType::Utf8, true),
        Field::new("score", DataType::Float64, true),
    ]);

    let num_rows = 20_000;
    let ids: Vec<i64> = (0..num_rows as i64).collect();
    let vals: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            if simple_hash(700, i) % 7 == 0 {
                None
            } else {
                Some(simple_hash(701, i) as i32)
            }
        })
        .collect();
    let labels: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if simple_hash(710, i) % 5 == 0 {
                None
            } else {
                Some(format!("lbl_{}", i % 200))
            }
        })
        .collect();
    let label_refs: Vec<Option<&str>> = labels.iter().map(|s| s.as_deref()).collect();
    let scores: Vec<Option<f64>> = (0..num_rows)
        .map(|i| {
            if simple_hash(720, i) % 9 == 0 {
                None
            } else {
                Some(simple_hash(721, i) as f64 / 10000.0)
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int32Array::from(vals)),
            Arc::new(StringArray::from(label_refs)),
            Arc::new(Float64Array::from(scores)),
        ],
    )
    .unwrap();

    let options = || WriterOptions::default();

    // Cycle 0: initial write
    let mut current_data = write_file(&schema, &[batch], options());
    let mut current_batches = read_all(&current_data);

    let mut prev_data = current_data.clone();

    // Cycles 1-4
    for cycle in 1..5 {
        let read_schema = current_batches[0].schema();
        current_data = write_file(&read_schema, &current_batches, options());
        current_batches = read_all(&current_data);

        // Each cycle should produce identical bytes
        assert_eq!(
            current_data.len(),
            prev_data.len(),
            "cycle {}: file size changed from {} to {}",
            cycle,
            prev_data.len(),
            current_data.len()
        );
        assert_eq!(
            current_data, prev_data,
            "cycle {}: byte-level mismatch",
            cycle
        );

        prev_data = current_data.clone();
    }

    println!(
        "test_reroundtrip_chain_5_times: PASSED - {} bytes, stable across 5 cycles",
        current_data.len()
    );
}

// ======================== Output Stability Tests ========================

#[test]
fn test_output_size_consistent() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Utf8, true),
        Field::new("count", DataType::Int32, true),
    ]);

    let num_rows = 30_000;
    let ids: Vec<i64> = (0..num_rows as i64).collect();
    let vals: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if i % 8 == 0 {
                None
            } else {
                Some(format!("item_{:06}", i))
            }
        })
        .collect();
    let val_refs: Vec<Option<&str>> = vals.iter().map(|s| s.as_deref()).collect();
    let counts: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            if i % 12 == 0 {
                None
            } else {
                Some(simple_hash(800, i) as i32)
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(val_refs)),
            Arc::new(Int32Array::from(counts)),
        ],
    )
    .unwrap();

    let options = || WriterOptions::default();

    let mut sizes = Vec::new();
    for _ in 0..10 {
        let data = write_file(&schema, &[batch.clone()], options());
        sizes.push(data.len());
    }

    let first_size = sizes[0];
    for (i, &size) in sizes.iter().enumerate() {
        assert_eq!(
            size, first_size,
            "write {} produced {} bytes, expected {}",
            i, size, first_size
        );
    }

    println!(
        "test_output_size_consistent: PASSED - all 10 writes produced exactly {} bytes",
        first_size
    );
}

#[test]
fn test_footer_position_deterministic() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, true),
        Field::new("score", DataType::Float64, false),
    ]);

    let num_rows = 50_000;
    let ids: Vec<i64> = (0..num_rows as i64).collect();
    let strs: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if i % 6 == 0 {
                None
            } else {
                Some(format!("record_{:08}", i))
            }
        })
        .collect();
    let str_refs: Vec<Option<&str>> = strs.iter().map(|s| s.as_deref()).collect();
    let scores: Vec<f64> = (0..num_rows).map(|i| i as f64 * 0.01).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(str_refs)),
            Arc::new(Float64Array::from(scores)),
        ],
    )
    .unwrap();

    let options = || WriterOptions::default();

    let data1 = write_file(&schema, &[batch.clone()], options());
    let data2 = write_file(&schema, &[batch.clone()], options());

    // Parse footer from both files (last 32 bytes)
    let parse_footer = |data: &[u8]| {
        let len = data.len();
        let footer = &data[len - spec::FOOTER_SIZE..];
        let index_offset = u64::from_be_bytes(footer[0..8].try_into().unwrap());
        let schema_block_offset = u64::from_be_bytes(footer[8..16].try_into().unwrap());
        let num_buckets = u32::from_be_bytes(footer[16..20].try_into().unwrap());
        let num_row_groups = u32::from_be_bytes(footer[20..24].try_into().unwrap());
        let compression = footer[24];
        let version = footer[25];
        (
            index_offset,
            schema_block_offset,
            num_buckets,
            num_row_groups,
            compression,
            version,
        )
    };

    let footer1 = parse_footer(&data1);
    let footer2 = parse_footer(&data2);

    assert_eq!(
        footer1.0, footer2.0,
        "index_offset differs: {} vs {}",
        footer1.0, footer2.0
    );
    assert_eq!(
        footer1.1, footer2.1,
        "schema_block_offset differs: {} vs {}",
        footer1.1, footer2.1
    );
    assert_eq!(
        footer1.2, footer2.2,
        "num_buckets differs: {} vs {}",
        footer1.2, footer2.2
    );
    assert_eq!(
        footer1.3, footer2.3,
        "num_row_groups differs: {} vs {}",
        footer1.3, footer2.3
    );
    assert_eq!(
        footer1.4, footer2.4,
        "compression differs: {} vs {}",
        footer1.4, footer2.4
    );
    assert_eq!(
        footer1.5, footer2.5,
        "version differs: {} vs {}",
        footer1.5, footer2.5
    );

    println!(
        "test_footer_position_deterministic: PASSED - index_offset={}, schema_block_offset={}, identical",
        footer1.0, footer1.1
    );
}

// ======================== BPE Determinism Regression Test ========================

/// Regression test for BPE non-determinism when column names share common substrings.
/// The BPE vocabulary builder uses HashMap for pair counting; when multiple pairs have
/// equal frequency, the tie must be broken deterministically (by pair key) rather than
/// relying on HashMap iteration order.
#[test]
fn test_deterministic_shared_substring_columns() {
    let schema = Schema::new(vec![
        Field::new("engine_coolant_temp", DataType::Int32, true),
        Field::new("engine_coolant_pressure", DataType::Float64, true),
        Field::new("engine_oil_temp", DataType::Int32, true),
        Field::new("engine_oil_pressure", DataType::Float64, true),
        Field::new("engine_rpm", DataType::Int64, false),
        Field::new("engine_load", DataType::Float32, true),
        Field::new("transmission_temp", DataType::Int32, true),
        Field::new("transmission_pressure", DataType::Float64, true),
    ]);

    let num_rows = 10_000;
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(
                (0..num_rows)
                    .map(|i| if i % 5 == 0 { None } else { Some(i as i32) })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (0..num_rows)
                    .map(|i| {
                        if i % 7 == 0 {
                            None
                        } else {
                            Some(i as f64 * 0.1)
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                (0..num_rows)
                    .map(|i| if i % 6 == 0 { None } else { Some(i as i32 * 2) })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (0..num_rows)
                    .map(|i| {
                        if i % 8 == 0 {
                            None
                        } else {
                            Some(i as f64 * 0.5)
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from((0..num_rows as i64).collect::<Vec<_>>())),
            Arc::new(Float32Array::from(
                (0..num_rows)
                    .map(|i| if i % 9 == 0 { None } else { Some(i as f32) })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                (0..num_rows)
                    .map(|i| if i % 4 == 0 { None } else { Some(i as i32 * 3) })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (0..num_rows)
                    .map(|i| {
                        if i % 11 == 0 {
                            None
                        } else {
                            Some(i as f64 * 0.3)
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();

    // Write 10 times - all must produce byte-identical output
    let first = write_file(&schema, &[batch.clone()], WriterOptions::default());
    for attempt in 1..10 {
        let again = write_file(&schema, &[batch.clone()], WriterOptions::default());
        assert_eq!(
            first.len(),
            again.len(),
            "attempt {}: file sizes differ: {} vs {}",
            attempt,
            first.len(),
            again.len()
        );
        assert_eq!(
            first, again,
            "attempt {}: byte-level mismatch with shared-substring column names",
            attempt
        );
    }

    // Also verify roundtrip correctness
    let result = read_all(&first);
    let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, num_rows);

    println!(
        "test_deterministic_shared_substring_columns: PASSED - {} bytes, 10 writes all byte-identical",
        first.len()
    );
}
