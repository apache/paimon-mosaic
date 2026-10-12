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

use super::*;
use crate::reader::{InputFile, MosaicReader, ReaderAccess};
use arrow_schema::{DataType, Field, Schema};
use std::sync::{Arc, Mutex};

impl InputFile for Bytes {
    fn read_at(&self, offset: u64, out: &mut [u8]) -> io::Result<()> {
        let start = offset as usize;
        out.copy_from_slice(&self.0[start..start + out.len()]);
        Ok(())
    }
}

#[derive(Default)]
struct Bytes(Vec<u8>);
impl OutputFile for Bytes {
    fn write(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.0.extend_from_slice(bytes);
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
    fn pos(&self) -> u64 {
        self.0.len() as u64
    }
}

#[derive(Default)]
struct OutputCalls {
    bytes: Vec<u8>,
    writes: Vec<usize>,
    flushes: usize,
}

struct RecordingOutput {
    calls: Arc<Mutex<OutputCalls>>,
    fail_on_write: Option<(usize, io::ErrorKind)>,
}

impl OutputFile for RecordingOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<()> {
        let mut calls = self.calls.lock().unwrap();
        calls.writes.push(bytes.len());
        if let Some((_, kind)) = self
            .fail_on_write
            .filter(|(write, _)| *write == calls.writes.len())
        {
            // OutputFile callbacks can fail after writing a prefix. Retrying
            // the buffered bytes would duplicate that prefix in the file.
            calls.bytes.extend_from_slice(&bytes[..bytes.len() / 2]);
            return Err(io::Error::new(kind, "buffered output failure"));
        }
        calls.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.calls.lock().unwrap().flushes += 1;
        Ok(())
    }

    fn pos(&self) -> u64 {
        self.calls.lock().unwrap().bytes.len() as u64
    }
}

fn dictionary_input(rows: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(
        (0..100)
            .map(|col| Field::new(format!("c{col:03}"), DataType::Int32, false))
            .collect::<Vec<_>>(),
    ));
    let values: ArrayRef = Arc::new(Int32Array::from(
        (0..rows).map(|row| (row % 200) as i32).collect::<Vec<_>>(),
    ));
    RecordBatch::try_new(schema, vec![values; 100]).unwrap()
}

#[test]
fn uncompressed_integer_output_coalesces_dictionary_writes() {
    let input = dictionary_input(10_000);
    let options = || WriterOptions {
        compression: COMPRESSION_NONE,
        num_buckets: 1,
        row_group_max_size: 1,
        ..Default::default()
    };
    let calls = Arc::new(Mutex::new(OutputCalls::default()));
    let mut compact = MosaicWriter::new(
        RecordingOutput {
            calls: calls.clone(),
            fail_on_write: None,
        },
        &input.schema(),
        options(),
    )
    .unwrap();
    let mut wide = MosaicWriter::new(Bytes::default(), &input.schema(), options()).unwrap();
    for bucket in wide.bucket_writers.iter_mut().flatten() {
        bucket.use_wide_integer_buffers();
    }
    compact.write_batch(&input).unwrap();
    wide.write_batch(&input).unwrap();
    {
        let calls = calls.lock().unwrap();
        assert_eq!(calls.bytes, wide.output().0);
        assert!(calls.writes.len() <= calls.bytes.len().div_ceil(INTEGER_OUTPUT_BUFFER_SIZE / 2));
        assert!(calls
            .writes
            .iter()
            .all(|&len| len <= INTEGER_OUTPUT_BUFFER_SIZE));
        assert_eq!(
            calls.flushes, 0,
            "bucket completion must not flush the sink"
        );
    }
    compact.close().unwrap();
    wide.close().unwrap();
    let calls = calls.lock().unwrap();
    assert_eq!(calls.bytes, wide.output().0);
    assert_eq!(calls.flushes, 1);
}

#[test]
fn integer_output_failure_does_not_retry_partial_writes() {
    let plain = RecordBatch::try_from_iter([(
        "id",
        Arc::new(Int64Array::from_iter_values(0..200_003)) as ArrayRef,
    )])
    .unwrap();
    // Cover final drain, buffered dictionary writes, direct value chunks,
    // one-shot compression and streaming compression (> 1 MiB).
    for (input, compression, fail_on_write) in [
        (dictionary_input(1), COMPRESSION_NONE, 1),
        (dictionary_input(10_000), COMPRESSION_NONE, 2),
        (plain.clone(), COMPRESSION_NONE, 2),
        (dictionary_input(1_000), COMPRESSION_ZSTD, 1),
        (plain, COMPRESSION_ZSTD, 1),
    ] {
        for kind in [io::ErrorKind::Other, io::ErrorKind::Interrupted] {
            for row_group_max_size in [1, u64::MAX] {
                let calls = Arc::new(Mutex::new(OutputCalls::default()));
                let mut writer = MosaicWriter::new(
                    RecordingOutput {
                        calls: calls.clone(),
                        fail_on_write: Some((fail_on_write, kind)),
                    },
                    &input.schema(),
                    WriterOptions {
                        compression,
                        num_buckets: 1,
                        row_group_max_size,
                        page_size_threshold: usize::MAX,
                        ..Default::default()
                    },
                )
                .unwrap();
                let result = writer.write_batch(&input);
                let error = if row_group_max_size == 1 {
                    result.unwrap_err()
                } else {
                    result.unwrap();
                    writer.close().unwrap_err()
                };
                assert_eq!(error.to_string(), "buffered output failure");
                assert_eq!(calls.lock().unwrap().writes.len(), fail_on_write);
                let bytes_after_failure = calls.lock().unwrap().bytes.clone();
                assert!(writer
                    .write_batch(&input)
                    .unwrap_err()
                    .to_string()
                    .contains("aborted"));
                assert!(writer.close().unwrap_err().to_string().contains("aborted"));
                drop(writer);
                let calls = calls.lock().unwrap();
                assert_eq!(calls.writes.len(), fail_on_write);
                assert_eq!(calls.bytes, bytes_after_failure);
                assert_eq!(calls.flushes, 0);
            }
        }
    }
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("i8", DataType::Int8, true),
        Field::new("i16", DataType::Int16, true),
        Field::new("i32", DataType::Int32, true),
        Field::new("i64", DataType::Int64, true),
        Field::new("float", DataType::Float64, false),
        Field::new("string", DataType::Utf8, true),
    ]))
}

fn batch(schema: Arc<Schema>, mode: usize, batch: usize, rows: usize) -> RecordBatch {
    let value = |row: usize| -> Option<i64> {
        let i = row + batch * rows;
        match mode {
            0 => None,
            1 => Some(255),
            2 => {
                if i.is_multiple_of(7) {
                    None
                } else {
                    Some(65535)
                }
            }
            3 => {
                if i.is_multiple_of(11) {
                    None
                } else {
                    Some((i % 103) as i64)
                }
            }
            _ => {
                if i.is_multiple_of(13) {
                    None
                } else {
                    Some(i as i64)
                }
            }
        }
    };
    let ints: Vec<_> = (0..rows + 2).map(value).collect();
    let i8 = Int8Array::from(ints.iter().map(|v| v.map(|v| v as i8)).collect::<Vec<_>>());
    let i16 = Int16Array::from(
        ints.iter()
            .map(|v| v.map(|v| (v % 256) as i16))
            .collect::<Vec<_>>(),
    );
    let mut i32 = ints
        .iter()
        .map(|v| v.map(|v| (v % 65536) as i32))
        .collect::<Vec<_>>();
    let mut i64 = ints
        .iter()
        .map(|v| v.map(|v| (v * 2654435761) & 0xffff_ffff))
        .collect::<Vec<_>>();
    if mode == 5 && batch == 3 {
        i32[rows - 1] = Some(i32::MIN);
        i32[rows] = Some(i32::MAX);
        i64[rows - 1] = Some(i64::MIN);
        i64[rows] = Some(i64::MAX);
    }
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(i8),
        Arc::new(i16),
        Arc::new(Int32Array::from(i32)),
        Arc::new(Int64Array::from(i64)),
        Arc::new(Float64Array::from(
            (0..rows + 2).map(|i| i as f64 / 7.0).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            (0..rows + 2)
                .map(|i| {
                    if i.is_multiple_of(17) {
                        None
                    } else {
                        Some("mixed-column")
                    }
                })
                .collect::<Vec<_>>(),
        )),
    ];
    // A nonzero offset exercises Arrow slicing and nullable arrays together.
    RecordBatch::try_new(
        schema,
        arrays.into_iter().map(|a| a.slice(1, rows)).collect(),
    )
    .unwrap()
}

#[test]
fn integer_stream_preserves_values_layout_stats_and_compression() {
    let schema = schema();
    for compression in [COMPRESSION_NONE, COMPRESSION_ZSTD] {
        for threshold in [1, 4096, usize::MAX] {
            for row_group in [60_000, u64::MAX] {
                for mode in 0..6 {
                    let options = || WriterOptions {
                        compression,
                        num_buckets: 1,
                        page_size_threshold: threshold,
                        row_group_max_size: row_group,
                        zstd_level: 3,
                        stats_columns: vec!["i16".into(), "i32".into(), "i64".into()],
                        ..Default::default()
                    };
                    let mut compact =
                        MosaicWriter::new(Bytes::default(), &schema, options()).unwrap();
                    let mut wide = MosaicWriter::new(Bytes::default(), &schema, options()).unwrap();
                    for bucket in wide.bucket_writers.iter_mut().flatten() {
                        bucket.use_wide_integer_buffers();
                    }
                    for i in 0..5 {
                        let input = batch(schema.clone(), mode, i, 10_003);
                        compact.write_batch(&input).unwrap();
                        wide.write_batch(&input).unwrap();
                        assert_eq!(compact.current_buffered_size, wide.current_buffered_size);
                        assert_eq!(compact.row_group_metas.len(), wide.row_group_metas.len());
                        assert_eq!(compact.estimated_file_size(), wide.estimated_file_size());
                    }
                    compact.close().unwrap();
                    wide.close().unwrap();
                    let actual = &compact.output().0;
                    let expected = &wide.output().0;
                    assert!(actual.len() <= expected.len(),
                        "larger file: compression={compression}, threshold={threshold}, row_group={row_group}, mode={mode}; {} vs {}", actual.len(), expected.len());
                    if compression == COMPRESSION_NONE {
                        assert!(actual == expected, "uncompressed encoding changed");
                    }
                    let a = MosaicReader::new(Bytes(actual.clone()), actual.len() as u64).unwrap();
                    let b =
                        MosaicReader::new(Bytes(expected.clone()), expected.len() as u64).unwrap();
                    assert_eq!(a.num_row_groups(), b.num_row_groups());
                    assert_eq!(a.schema().serialize(), b.schema().serialize());
                    let mut consumed = 0;
                    for rg in 0..a.num_row_groups() {
                        assert_eq!(
                            a.row_group_num_rows(rg).unwrap(),
                            b.row_group_num_rows(rg).unwrap()
                        );
                        assert_eq!(
                            stats::serialize_stats(
                                a.row_group_stats(rg).unwrap(),
                                &a.schema().columns
                            ),
                            stats::serialize_stats(
                                b.row_group_stats(rg).unwrap(),
                                &b.schema().columns
                            )
                        );
                        let a_batch = a.row_group_reader(rg).unwrap().read_columns().unwrap();
                        let b_batch = b.row_group_reader(rg).unwrap().read_columns().unwrap();
                        assert!(a_batch == b_batch, "decoded values changed");
                        let mut row = 0;
                        while row < a_batch.num_rows() {
                            let input = batch(schema.clone(), mode, consumed / 10_003, 10_003);
                            let offset = consumed % 10_003;
                            let count = (10_003 - offset).min(a_batch.num_rows() - row);
                            for field in schema.fields() {
                                let actual = a_batch
                                    .column_by_name(field.name())
                                    .unwrap()
                                    .slice(row, count);
                                let expected = input
                                    .column_by_name(field.name())
                                    .unwrap()
                                    .slice(offset, count);
                                assert!(
                                    actual.to_data() == expected.to_data(),
                                    "input values changed: {}",
                                    field.name()
                                );
                            }
                            row += count;
                            consumed += count;
                        }
                    }
                    assert_eq!(consumed, 50_015);
                }
            }
        }
    }
}

#[test]
fn integer_only_stream_matches_one_shot_at_zstd_block_boundaries() {
    let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
    for rows in [
        0, 1, 4095, 4096, 4097, 16383, 16384, 16385, 100_003, 200_003,
    ] {
        for level in [-1, 1, 9] {
            for threshold in [1, usize::MAX] {
                let options = || WriterOptions {
                    num_buckets: 1,
                    max_dict_entries: 0,
                    page_size_threshold: threshold,
                    zstd_level: level,
                    row_group_max_size: u64::MAX,
                    ..Default::default()
                };
                let mut compact = MosaicWriter::new(Bytes::default(), &schema, options()).unwrap();
                let mut wide = MosaicWriter::new(Bytes::default(), &schema, options()).unwrap();
                for bucket in wide.bucket_writers.iter_mut().flatten() {
                    bucket.use_wide_integer_buffers();
                }
                let input = RecordBatch::try_new(
                    Arc::new(schema.clone()),
                    vec![Arc::new(Int64Array::from(
                        (0..rows)
                            .map(|i| ((i as u64 * 2654435761) & 0xffff_ffff) as i64)
                            .collect::<Vec<_>>(),
                    ))],
                )
                .unwrap();
                compact.write_batch(&input).unwrap();
                wide.write_batch(&input).unwrap();
                compact.close().unwrap();
                wide.close().unwrap();
                assert!(
                    compact.output().0 == wide.output().0,
                    "rows={rows}, level={level}, threshold={threshold}"
                );
            }
        }
    }
}
