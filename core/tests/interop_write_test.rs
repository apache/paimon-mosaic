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

//! Interop write tests: writes golden .mosaic files to /tmp/mosaic_interop/
//! that can be read by Java and Python tests to verify cross-language compatibility.

#![allow(clippy::approx_constant, clippy::needless_range_loop)]

use std::fs;
use std::io;
use std::io::Write as IoWrite;
use std::sync::Arc;

use arrow_array::*;
use arrow_schema::{DataType, Field, Schema};
use paimon_mosaic_core::reader::{InputFile, MosaicReader, ReaderAccess};
use paimon_mosaic_core::writer::{MosaicWriter, OutputFile, WriterOptions};

const INTEROP_DIR: &str = "/tmp/mosaic_interop";

// ======================== Infrastructure ========================

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

fn write_and_verify(
    schema: &Schema,
    batches: &[RecordBatch],
    options: WriterOptions,
    filename: &str,
) -> Vec<u8> {
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(out, schema, options).unwrap();
    for batch in batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    // Verify by reading back in Rust
    let file_len = data.len() as u64;
    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, file_len).unwrap();

    let expected_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    let mut actual_rows: usize = 0;
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader(rg).unwrap();
        let batch = rg_reader.read_columns().unwrap();
        actual_rows += batch.num_rows();
    }
    assert_eq!(expected_rows, actual_rows, "row count mismatch for {}", filename);

    // Write to disk
    fs::create_dir_all(INTEROP_DIR).unwrap();
    let path = format!("{}/{}", INTEROP_DIR, filename);
    let mut file = fs::File::create(&path).unwrap();
    file.write_all(&data).unwrap();
    file.flush().unwrap();
    println!("Wrote {} ({} bytes, {} rows)", path, data.len(), expected_rows);

    data
}

// ======================== 1. int_data.mosaic ========================

#[test]
fn test_write_int_data() {
    let num_rows = 10_000;
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int32, false),
    ]);

    let ids: Vec<i64> = (0..num_rows).map(|i| i as i64).collect();
    let values: Vec<i32> = (0..num_rows).map(|i| (i * 10) as i32).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int32Array::from(values)),
        ],
    )
    .unwrap();

    write_and_verify(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
        "int_data.mosaic",
    );
    println!("test_write_int_data: PASSED");
}

// ======================== 2. string_data.mosaic ========================

#[test]
fn test_write_string_data() {
    let num_rows = 10_000;
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("data", DataType::Binary, true),
    ]);

    let ids: Vec<i64> = (0..num_rows).map(|i| i as i64).collect();
    let names: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if i % 7 == 0 {
                None
            } else {
                Some(format!("name_{}", i))
            }
        })
        .collect();
    let name_refs: Vec<Option<&str>> = names.iter().map(|s| s.as_deref()).collect();

    let bin_owned: Vec<Option<Vec<u8>>> = (0..num_rows)
        .map(|i| {
            if i % 5 == 0 {
                None
            } else {
                Some(format!("bin_{}", i).into_bytes())
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
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(name_refs)),
            Arc::new(BinaryArray::from(bin_refs)),
        ],
    )
    .unwrap();

    write_and_verify(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
        "string_data.mosaic",
    );
    println!("test_write_string_data: PASSED");
}

// ======================== 3. all_types.mosaic ========================

#[test]
fn test_write_all_types() {
    let num_rows = 5_000;
    let schema = Schema::new(vec![
        Field::new("f_bool", DataType::Boolean, true),
        Field::new("f_int8", DataType::Int8, true),
        Field::new("f_int16", DataType::Int16, true),
        Field::new("f_int32", DataType::Int32, true),
        Field::new("f_int64", DataType::Int64, true),
        Field::new("f_float32", DataType::Float32, true),
        Field::new("f_float64", DataType::Float64, true),
        Field::new("f_date32", DataType::Date32, true),
        Field::new("f_utf8", DataType::Utf8, true),
        Field::new("f_binary", DataType::Binary, true),
        Field::new("f_decimal", DataType::Decimal128(10, 2), true),
    ]);

    let bools: Vec<Option<bool>> = (0..num_rows)
        .map(|i| {
            if i % 13 == 0 {
                None
            } else {
                Some(i % 2 == 0)
            }
        })
        .collect();

    let i8s: Vec<Option<i8>> = (0..num_rows)
        .map(|i| {
            if i % 11 == 0 {
                None
            } else {
                Some((i % 256) as i8)
            }
        })
        .collect();

    let i16s: Vec<Option<i16>> = (0..num_rows)
        .map(|i| {
            if i % 17 == 0 {
                None
            } else {
                Some((i % 30000) as i16)
            }
        })
        .collect();

    let i32s: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            if i % 19 == 0 {
                None
            } else {
                Some(i as i32 * 100)
            }
        })
        .collect();

    let i64s: Vec<Option<i64>> = (0..num_rows)
        .map(|i| {
            if i % 23 == 0 {
                None
            } else {
                Some(i as i64 * 1000)
            }
        })
        .collect();

    let f32s: Vec<Option<f32>> = (0..num_rows)
        .map(|i| {
            if i % 29 == 0 {
                None
            } else {
                Some(i as f32 * 0.1)
            }
        })
        .collect();

    let f64s: Vec<Option<f64>> = (0..num_rows)
        .map(|i| {
            if i % 31 == 0 {
                None
            } else {
                Some(i as f64 * 0.001)
            }
        })
        .collect();

    let dates: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            if i % 37 == 0 {
                None
            } else {
                Some(18000 + (i % 3650) as i32)
            }
        })
        .collect();

    let str_owned: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if i % 41 == 0 {
                None
            } else {
                Some(format!("str_{}", i))
            }
        })
        .collect();
    let str_refs: Vec<Option<&str>> = str_owned.iter().map(|s| s.as_deref()).collect();

    let bin_owned: Vec<Option<Vec<u8>>> = (0..num_rows)
        .map(|i| {
            if i % 43 == 0 {
                None
            } else {
                Some(vec![(i % 256) as u8; 4])
            }
        })
        .collect();
    let bin_refs: Vec<Option<&[u8]>> = bin_owned
        .iter()
        .map(|b| b.as_ref().map(|v| v.as_slice()))
        .collect();

    let decimals: Vec<Option<i128>> = (0..num_rows)
        .map(|i| {
            if i % 47 == 0 {
                None
            } else {
                Some(i as i128 * 100)
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
            Arc::new(StringArray::from(str_refs)),
            Arc::new(BinaryArray::from(bin_refs)),
            Arc::new(
                Decimal128Array::from(decimals)
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();

    write_and_verify(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
        "all_types.mosaic",
    );
    println!("test_write_all_types: PASSED");
}

// ======================== 4. constant_data.mosaic ========================

#[test]
fn test_write_constant_data() {
    let num_rows = 10_000;
    let schema = Schema::new(vec![
        Field::new("c_int", DataType::Int64, false),
        Field::new("c_str", DataType::Utf8, false),
        Field::new("c_float", DataType::Float64, false),
    ]);

    let ints: Vec<i64> = vec![42i64; num_rows];
    let strs: Vec<&str> = vec!["constant_value"; num_rows];
    let floats: Vec<f64> = vec![3.14; num_rows];

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ints)),
            Arc::new(StringArray::from(strs)),
            Arc::new(Float64Array::from(floats)),
        ],
    )
    .unwrap();

    write_and_verify(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
        "constant_data.mosaic",
    );
    println!("test_write_constant_data: PASSED");
}

// ======================== 5. null_heavy.mosaic ========================

#[test]
fn test_write_null_heavy() {
    let num_rows = 10_000;
    let schema = Schema::new(vec![
        Field::new("n_int64", DataType::Int64, true),
        Field::new("n_utf8", DataType::Utf8, true),
        Field::new("n_float64", DataType::Float64, true),
    ]);

    // 80% nulls
    let ints: Vec<Option<i64>> = (0..num_rows)
        .map(|i| {
            if i % 5 != 0 {
                None
            } else {
                Some(i as i64)
            }
        })
        .collect();

    let str_owned: Vec<Option<String>> = (0..num_rows)
        .map(|i| {
            if i % 5 != 1 {
                None
            } else {
                Some(format!("val_{}", i))
            }
        })
        .collect();
    let str_refs: Vec<Option<&str>> = str_owned.iter().map(|s| s.as_deref()).collect();

    let floats: Vec<Option<f64>> = (0..num_rows)
        .map(|i| {
            if i % 5 != 2 {
                None
            } else {
                Some(i as f64 * 0.5)
            }
        })
        .collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ints)),
            Arc::new(StringArray::from(str_refs)),
            Arc::new(Float64Array::from(floats)),
        ],
    )
    .unwrap();

    write_and_verify(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
        "null_heavy.mosaic",
    );
    println!("test_write_null_heavy: PASSED");
}

// ======================== 6. compressed_none.mosaic ========================

#[test]
fn test_write_compressed_none() {
    let num_rows = 10_000;
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int32, false),
    ]);

    let ids: Vec<i64> = (0..num_rows).map(|i| i as i64).collect();
    let values: Vec<i32> = (0..num_rows).map(|i| (i * 10) as i32).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int32Array::from(values)),
        ],
    )
    .unwrap();

    write_and_verify(
        &schema,
        &[batch],
        WriterOptions {
            compression: 0, // COMPRESSION_NONE
            num_buckets: 1,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
        "compressed_none.mosaic",
    );
    println!("test_write_compressed_none: PASSED");
}

// ======================== 7. multi_rg.mosaic ========================

#[test]
fn test_write_multi_rg() {
    let num_rows = 10_000;
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int32, false),
    ]);

    // Write in small batches to trigger row group flushes with a tiny max size
    let batch_size = 100;
    let num_batches = num_rows / batch_size;
    let mut batches = Vec::new();
    for b in 0..num_batches {
        let start = b * batch_size;
        let ids: Vec<i64> = (start..start + batch_size).map(|i| i as i64).collect();
        let values: Vec<i32> = (start..start + batch_size).map(|i| (i * 10) as i32).collect();
        batches.push(
            RecordBatch::try_new(
                Arc::new(schema.clone()),
                vec![
                    Arc::new(Int64Array::from(ids)),
                    Arc::new(Int32Array::from(values)),
                ],
            )
            .unwrap(),
        );
    }

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            compression: 0, // No compression to make data larger
            num_buckets: 1,
            row_group_max_size: 1024, // Force multiple row groups
            ..Default::default()
        },
    )
    .unwrap();
    for batch in &batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();
    let data = writer.output().buf.clone();

    // Write to disk
    fs::create_dir_all(INTEROP_DIR).unwrap();
    let path = format!("{}/multi_rg.mosaic", INTEROP_DIR);
    let mut file = fs::File::create(&path).unwrap();
    file.write_all(&data).unwrap();
    file.flush().unwrap();

    // Verify by reading back
    let file_len = data.len() as u64;
    let input = ByteArrayInputFile { data: data.clone() };
    let reader_verify = MosaicReader::new(input, file_len).unwrap();
    let mut actual_rows: usize = 0;
    for rg in 0..reader_verify.num_row_groups() {
        let mut rg_reader = reader_verify.row_group_reader(rg).unwrap();
        let rb = rg_reader.read_columns().unwrap();
        actual_rows += rb.num_rows();
    }
    assert_eq!(num_rows, actual_rows, "row count mismatch for multi_rg.mosaic");
    println!("Wrote {} ({} bytes, {} rows)", path, data.len(), num_rows);

    // Verify multiple row groups were created
    let file_len2 = data.len() as u64;
    let input2 = ByteArrayInputFile { data };
    let reader = MosaicReader::new(input2, file_len2).unwrap();
    assert!(
        reader.num_row_groups() > 1,
        "Expected multiple row groups, got {}",
        reader.num_row_groups()
    );
    println!(
        "test_write_multi_rg: PASSED ({} row groups)",
        reader.num_row_groups()
    );
}
