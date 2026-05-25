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

//! Interop read tests: reads .mosaic files written by Java and Python,
//! verifying that Rust can correctly read files produced by other languages.

#![allow(clippy::approx_constant, clippy::needless_range_loop)]

use std::fs;
use std::io;

use arrow_array::*;
use paimon_mosaic_core::reader::{InputFile, MosaicReader, ReaderAccess};

const INTEROP_DIR: &str = "/tmp/mosaic_interop";

// ======================== Infrastructure ========================

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

fn open_file(filename: &str) -> MosaicReader<ByteArrayInputFile> {
    let path = format!("{}/{}", INTEROP_DIR, filename);
    let data = fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "Failed to read {}: {}. Did you run the Java/Python interop tests first?",
            path, e
        );
    });
    let file_len = data.len() as u64;
    let input = ByteArrayInputFile { data };
    MosaicReader::new(input, file_len).unwrap()
}

// ======================== 1. Read java_written.mosaic ========================

#[test]
fn test_read_java_written_file() {
    let reader = open_file("java_written.mosaic");
    let num_rgs = reader.num_row_groups();
    assert!(num_rgs >= 1, "Expected at least 1 row group");

    let mut total_rows = 0usize;
    for rg in 0..num_rgs {
        let mut rg_reader = reader.row_group_reader(rg).unwrap();
        let batch = rg_reader.read_columns().unwrap();

        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let names = batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let scores = batch
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        for i in 0..batch.num_rows() {
            let row_idx = total_rows + i;
            assert_eq!(ids.value(i), row_idx as i64, "id mismatch at row {}", row_idx);

            if row_idx % 3 == 0 {
                assert!(names.is_null(i), "name should be null at row {}", row_idx);
            } else {
                assert!(!names.is_null(i), "name should not be null at row {}", row_idx);
                assert_eq!(
                    names.value(i),
                    format!("java_name_{}", row_idx),
                    "name mismatch at row {}",
                    row_idx
                );
            }

            assert_eq!(
                scores.value(i),
                row_idx as f64 * 2.5,
                "score mismatch at row {}",
                row_idx
            );
        }
        total_rows += batch.num_rows();
    }

    assert_eq!(total_rows, 5000, "Expected 5000 rows from Java-written file");
    println!(
        "test_read_java_written_file: PASSED ({} rows, {} row groups)",
        total_rows, num_rgs
    );
}

// ======================== 2. Read python_written.mosaic ========================

#[test]
fn test_read_python_written_file() {
    let reader = open_file("python_written.mosaic");
    let num_rgs = reader.num_row_groups();
    assert!(num_rgs >= 1, "Expected at least 1 row group");

    let mut total_rows = 0usize;
    for rg in 0..num_rgs {
        let mut rg_reader = reader.row_group_reader(rg).unwrap();
        let batch = rg_reader.read_columns().unwrap();

        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let names = batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let scores = batch
            .column_by_name("score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        for i in 0..batch.num_rows() {
            let row_idx = total_rows + i;
            assert_eq!(ids.value(i), row_idx as i64, "id mismatch at row {}", row_idx);

            if row_idx % 3 == 0 {
                assert!(names.is_null(i), "name should be null at row {}", row_idx);
            } else {
                assert!(!names.is_null(i), "name should not be null at row {}", row_idx);
                assert_eq!(
                    names.value(i),
                    format!("py_name_{}", row_idx),
                    "name mismatch at row {}",
                    row_idx
                );
            }

            assert!(
                (scores.value(i) - row_idx as f64 * 2.5).abs() < 1e-9,
                "score mismatch at row {}: {} != {}",
                row_idx,
                scores.value(i),
                row_idx as f64 * 2.5
            );
        }
        total_rows += batch.num_rows();
    }

    assert_eq!(
        total_rows, 5000,
        "Expected 5000 rows from Python-written file"
    );
    println!(
        "test_read_python_written_file: PASSED ({} rows, {} row groups)",
        total_rows, num_rgs
    );
}
