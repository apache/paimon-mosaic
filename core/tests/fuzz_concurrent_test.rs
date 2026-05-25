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

//! Fuzz robustness and concurrent reader tests for the Mosaic file format.
//! Fuzz tests feed random/corrupted data to MosaicReader and verify it never panics.
//! Concurrent tests verify thread-safe parallel reading of Mosaic files.

#![allow(
    clippy::approx_constant,
    clippy::unnecessary_cast,
    clippy::cloned_ref_to_slice_refs,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of
)]

use std::io;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::thread;

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

struct SharedInputFile {
    data: Arc<Vec<u8>>,
}

impl InputFile for SharedInputFile {
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

/// A corrupting input file that returns random data for some reads,
/// simulating I/O corruption at the storage layer.
struct CorruptingInputFile {
    data: Vec<u8>,
    seed: u64,
}

impl InputFile for CorruptingInputFile {
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
        // Corrupt approximately 1 in 3 reads with random data
        let h = simple_hash(self.seed, offset as usize);
        if h % 3 == 0 {
            for i in 0..buf.len() {
                buf[i] = simple_hash(self.seed.wrapping_add(1), offset as usize + i) as u8;
            }
        }
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

/// Write a small valid test file for corruption tests.
/// Uses only numeric columns to avoid arrow-array panics on invalid UTF-8
/// when bytes are corrupted in string data regions.
fn write_small_test_file() -> Vec<u8> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("val", DataType::Int32, true),
        Field::new("score", DataType::Float64, true),
    ]);
    let num_rows = 100;
    let ids: Vec<i64> = (0..num_rows as i64).collect();
    let vals: Vec<Option<i32>> = (0..num_rows)
        .map(|i| {
            if i % 3 == 0 {
                None
            } else {
                Some(i as i32 * 7)
            }
        })
        .collect();
    let scores: Vec<Option<f64>> = (0..num_rows)
        .map(|i| {
            if i % 5 == 0 {
                None
            } else {
                Some(i as f64 * 1.5)
            }
        })
        .collect();
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int32Array::from(vals)),
            Arc::new(Float64Array::from(scores)),
        ],
    )
    .unwrap();
    write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 1,
            ..Default::default()
        },
    )
}

/// Attempt to open and fully read a Mosaic file from raw bytes.
/// Returns Ok with batches on success, Err on any reader error.
fn try_read_file(data: &[u8]) -> io::Result<Vec<RecordBatch>> {
    let input = ByteArrayInputFile {
        data: data.to_vec(),
    };
    let reader = MosaicReader::new(input, data.len() as u64)?;
    let mut result = Vec::new();
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader(rg)?;
        result.push(rg_reader.read_columns()?);
    }
    Ok(result)
}

/// Generate a pseudo-random byte vector of a given length from a seed.
fn random_bytes(seed: u64, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| simple_hash(seed, i) as u8)
        .collect()
}

// ======================== Fuzz Robustness Tests ========================

#[test]
fn test_random_bytes_small() {
    // Feed 100 random byte sequences of length 32-1024 to MosaicReader.
    // It should either return Err or Ok (never panic).
    let mut panics = 0;
    for i in 0..100 {
        let seed = 1000 + i as u64;
        let len = 32 + (simple_hash(seed, 0) as usize % (1024 - 32 + 1));
        let data = random_bytes(seed, len);
        let result = catch_unwind(AssertUnwindSafe(|| try_read_file(&data)));
        if result.is_err() {
            panics += 1;
            eprintln!("PANIC at iteration {} (seed={}, len={})", i, seed, len);
        }
    }
    assert_eq!(panics, 0, "{} out of 100 iterations panicked", panics);
}

#[test]
fn test_random_bytes_large() {
    // Feed 20 random byte sequences of length 1024-65536. Should never panic.
    let mut panics = 0;
    for i in 0..20 {
        let seed = 2000 + i as u64;
        let len = 1024 + (simple_hash(seed, 0) as usize % (65536 - 1024 + 1));
        let data = random_bytes(seed, len);
        let result = catch_unwind(AssertUnwindSafe(|| try_read_file(&data)));
        if result.is_err() {
            panics += 1;
            eprintln!("PANIC at iteration {} (seed={}, len={})", i, seed, len);
        }
    }
    assert_eq!(panics, 0, "{} out of 20 iterations panicked", panics);
}

#[test]
fn test_random_bytes_with_valid_magic() {
    // Generate random bytes but put valid magic "MOSA" at the end and ensure
    // the file is at least FOOTER_SIZE bytes.
    let mut panics = 0;
    for i in 0..50 {
        let seed = 3000 + i as u64;
        let len = spec::FOOTER_SIZE + (simple_hash(seed, 0) as usize % 2048);
        let mut data = random_bytes(seed, len);
        // Place valid magic at the very end
        let dlen = data.len();
        data[dlen - 4] = b'M';
        data[dlen - 3] = b'O';
        data[dlen - 2] = b'S';
        data[dlen - 1] = b'A';
        let result = catch_unwind(AssertUnwindSafe(|| try_read_file(&data)));
        if result.is_err() {
            panics += 1;
            eprintln!("PANIC at iteration {} (seed={}, len={})", i, seed, len);
        }
    }
    assert_eq!(panics, 0, "{} out of 50 iterations panicked", panics);
}

#[test]
fn test_random_bytes_with_valid_footer() {
    // Generate a valid footer (correct magic, version 1, valid offsets within file size)
    // but random data before it. Should not panic.
    let mut panics = 0;
    for i in 0..50 {
        let seed = 4000 + i as u64;
        let total_len = 256 + (simple_hash(seed, 0) as usize % 4096);
        let mut data = random_bytes(seed, total_len);

        // Build a valid footer at the end (32 bytes)
        let footer_start = total_len - spec::FOOTER_SIZE;
        let data_region_end = footer_start;

        // index_offset: point to somewhere in the data region
        let index_offset = (data_region_end / 2) as u64;
        data[footer_start..footer_start + 8].copy_from_slice(&index_offset.to_be_bytes());

        // schema_block_offset: point before index_offset
        let schema_block_offset = (index_offset / 2).min(index_offset.saturating_sub(4));
        data[footer_start + 8..footer_start + 16]
            .copy_from_slice(&schema_block_offset.to_be_bytes());

        // num_buckets: 1
        data[footer_start + 16..footer_start + 20].copy_from_slice(&1u32.to_be_bytes());

        // num_row_groups: 1
        data[footer_start + 20..footer_start + 24].copy_from_slice(&1u32.to_be_bytes());

        // compression: ZSTD (1)
        data[footer_start + 24] = spec::COMPRESSION_ZSTD;

        // version: 1
        data[footer_start + 25] = spec::VERSION;

        // reserved bytes
        data[footer_start + 26] = 0;
        data[footer_start + 27] = 0;

        // magic: MOSA
        data[footer_start + 28] = b'M';
        data[footer_start + 29] = b'O';
        data[footer_start + 30] = b'S';
        data[footer_start + 31] = b'A';

        let result = catch_unwind(AssertUnwindSafe(|| try_read_file(&data)));
        if result.is_err() {
            panics += 1;
            eprintln!("PANIC at iteration {} (seed={}, len={})", i, seed, total_len);
        }
    }
    assert_eq!(panics, 0, "{} out of 50 iterations panicked", panics);
}

#[test]
fn test_bitflip_corruption() {
    // Write a valid file, then flip 1 random bit per attempt, 200 attempts.
    // Reader should either succeed (if bit was in non-critical area) or return error,
    // never panic.
    let original = write_small_test_file();
    let mut panics = 0;
    for i in 0..200 {
        let seed = 5000 + i as u64;
        let mut data = original.clone();
        // Pick a random byte position and bit position
        let byte_pos = simple_hash(seed, 0) as usize % data.len();
        let bit_pos = simple_hash(seed, 1) as usize % 8;
        data[byte_pos] ^= 1 << bit_pos;

        let result = catch_unwind(AssertUnwindSafe(|| try_read_file(&data)));
        if result.is_err() {
            panics += 1;
            eprintln!(
                "PANIC at iteration {} (byte_pos={}, bit_pos={})",
                i, byte_pos, bit_pos
            );
        }
    }
    assert_eq!(panics, 0, "{} out of 200 iterations panicked", panics);
}

#[test]
fn test_byte_substitution_corruption() {
    // Write a valid file, replace a random byte with a random value, 200 attempts.
    // Never panic.
    let original = write_small_test_file();
    let mut panics = 0;
    for i in 0..200 {
        let seed = 6000 + i as u64;
        let mut data = original.clone();
        let byte_pos = simple_hash(seed, 0) as usize % data.len();
        let new_val = simple_hash(seed, 1) as u8;
        data[byte_pos] = new_val;

        let result = catch_unwind(AssertUnwindSafe(|| try_read_file(&data)));
        if result.is_err() {
            panics += 1;
            eprintln!(
                "PANIC at iteration {} (byte_pos={}, new_val={})",
                i, byte_pos, new_val
            );
        }
    }
    assert_eq!(panics, 0, "{} out of 200 iterations panicked", panics);
}

#[test]
fn test_truncation_at_every_byte() {
    // Write a valid file of ~1KB. Try truncating at every position from 0 to len.
    // Reader should error for too-small files, never panic.
    let original = write_small_test_file();
    let file_len = original.len();
    println!(
        "test_truncation_at_every_byte: file is {} bytes, testing {} truncation points",
        file_len, file_len
    );

    let mut panics = 0;
    for trunc_len in 0..file_len {
        let data = original[..trunc_len].to_vec();
        let result = catch_unwind(AssertUnwindSafe(|| try_read_file(&data)));
        if result.is_err() {
            panics += 1;
            eprintln!("PANIC at truncation length {}", trunc_len);
        }
    }
    assert_eq!(
        panics, 0,
        "{} out of {} truncation points panicked",
        panics, file_len
    );
}

#[test]
fn test_extension_with_random_bytes() {
    // Write a valid file, append 1-1000 random bytes.
    // When using the original file length, the reader should still work
    // since the extra bytes are beyond the footer position.
    let original = write_small_test_file();
    let original_len = original.len();

    let mut panics = 0;
    for i in 0..50 {
        let seed = 7000 + i as u64;
        let extra_len = 1 + (simple_hash(seed, 0) as usize % 1000);
        let extra = random_bytes(seed, extra_len);

        let mut extended = original.clone();
        extended.extend_from_slice(&extra);

        // With original file_len, reader should still find the correct footer
        let result = catch_unwind(AssertUnwindSafe(|| {
            let input = ByteArrayInputFile {
                data: extended.clone(),
            };
            let reader = MosaicReader::new(input, original_len as u64)?;
            let mut batches = Vec::new();
            for rg in 0..reader.num_row_groups() {
                let mut rg_reader = reader.row_group_reader(rg)?;
                batches.push(rg_reader.read_columns()?);
            }
            Ok::<Vec<RecordBatch>, io::Error>(batches)
        }));
        match &result {
            Ok(Ok(batches)) => {
                let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
                assert_eq!(
                    total_rows, 100,
                    "iteration {}: expected 100 rows, got {}",
                    i, total_rows
                );
            }
            Ok(Err(_)) => {
                // Error is acceptable
            }
            Err(_) => {
                panics += 1;
                eprintln!(
                    "PANIC at iteration {} (extra_len={})",
                    i, extra_len
                );
            }
        }

        // Also try with extended file_len -- reader should error or work, but not panic
        let result2 = catch_unwind(AssertUnwindSafe(|| try_read_file(&extended)));
        if result2.is_err() {
            panics += 1;
            eprintln!(
                "PANIC at iteration {} with extended file_len (extra_len={})",
                i, extra_len
            );
        }
    }
    assert_eq!(panics, 0, "{} out of 100 attempts panicked", panics);
}

#[test]
fn test_zero_filled_file() {
    // All-zero files of various sizes. Should error, never panic.
    let sizes = [32, 64, 128, 256, 512, 1024, 4096];
    let mut panics = 0;
    for &size in &sizes {
        let data = vec![0u8; size];
        let result = catch_unwind(AssertUnwindSafe(|| try_read_file(&data)));
        match &result {
            Ok(inner) => {
                assert!(inner.is_err(), "zero-filled file of size {} should fail", size);
            }
            Err(_) => {
                panics += 1;
                eprintln!("PANIC with zero-filled file of size {}", size);
            }
        }
    }
    assert_eq!(
        panics, 0,
        "{} out of {} sizes panicked",
        panics,
        sizes.len()
    );
}

#[test]
fn test_all_ones_file() {
    // All 0xFF files of various sizes. Should error, never panic.
    let sizes = [32, 64, 128, 256, 512, 1024, 4096];
    let mut panics = 0;
    for &size in &sizes {
        let data = vec![0xFFu8; size];
        let result = catch_unwind(AssertUnwindSafe(|| try_read_file(&data)));
        match &result {
            Ok(inner) => {
                assert!(inner.is_err(), "all-ones file of size {} should fail", size);
            }
            Err(_) => {
                panics += 1;
                eprintln!("PANIC with all-ones file of size {}", size);
            }
        }
    }
    assert_eq!(
        panics, 0,
        "{} out of {} sizes panicked",
        panics,
        sizes.len()
    );
}

#[test]
fn test_repeated_valid_footer() {
    // Take a valid footer and repeat it 100 times as a file.
    // Should error or produce weird results, never panic.
    let valid_file = write_small_test_file();
    let footer = &valid_file[valid_file.len() - spec::FOOTER_SIZE..];

    let mut data = Vec::new();
    for _ in 0..100 {
        data.extend_from_slice(footer);
    }

    let result = catch_unwind(AssertUnwindSafe(|| try_read_file(&data)));
    assert!(
        result.is_ok(),
        "repeated footer file caused a panic"
    );
    // The reader should either error or return results -- both are acceptable
}

#[test]
fn test_valid_file_random_read_offsets() {
    // Write a valid file. Create a custom InputFile that returns random data for some reads
    // (simulating I/O corruption). Try to read. Should error, never panic.
    let valid_file = write_small_test_file();
    let mut panics = 0;

    for i in 0..50 {
        let seed = 8000 + i as u64;
        let input = CorruptingInputFile {
            data: valid_file.clone(),
            seed,
        };
        let result = catch_unwind(AssertUnwindSafe(|| {
            let reader = MosaicReader::new(input, valid_file.len() as u64)?;
            let mut batches = Vec::new();
            for rg in 0..reader.num_row_groups() {
                let mut rg_reader = reader.row_group_reader(rg)?;
                batches.push(rg_reader.read_columns()?);
            }
            Ok::<Vec<RecordBatch>, io::Error>(batches)
        }));
        if result.is_err() {
            panics += 1;
            eprintln!("PANIC at iteration {} (seed={})", i, seed);
        }
    }
    assert_eq!(panics, 0, "{} out of 50 iterations panicked", panics);
}

// ======================== Concurrent Reader Tests ========================

#[test]
fn test_concurrent_full_read() {
    // Write a file with 1M rows. Wrap data in Arc. Spawn 8 threads, each creates
    // its own MosaicReader and reads ALL row groups. All should get the same total row count.
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("val", DataType::Int32, true),
    ]);
    let num_rows = 1_000_000;
    let batch_size = 200_000;
    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;
        let ids: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();
        let vals: Vec<Option<i32>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 7 == 0 {
                    None
                } else {
                    Some((batch_start + i) as i32)
                }
            })
            .collect();
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int32Array::from(vals)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let file_data = write_file(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 2,
            row_group_max_size: 16 * 1024 * 1024,
            ..Default::default()
        },
    );
    let shared_data = Arc::new(file_data);
    let file_len = shared_data.len() as u64;

    let num_threads = 8;
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let data = Arc::clone(&shared_data);
            thread::spawn(move || {
                let input = SharedInputFile { data };
                let reader = MosaicReader::new(input, file_len).unwrap();
                let mut total = 0usize;
                for rg in 0..reader.num_row_groups() {
                    let mut rg_reader = reader.row_group_reader(rg).unwrap();
                    let batch = rg_reader.read_columns().unwrap();
                    total += batch.num_rows();
                }
                total
            })
        })
        .collect();

    let results: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for (i, &count) in results.iter().enumerate() {
        assert_eq!(
            count, num_rows,
            "thread {} got {} rows, expected {}",
            i, count, num_rows
        );
    }
}

#[test]
fn test_concurrent_different_row_groups() {
    // Write a file with multiple row groups (small row_group_max_size).
    // Spawn threads, each reads different row groups. Verify all row groups covered.
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, true),
    ]);
    let num_rows = 100_000;
    let batch_size = 10_000;
    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;
        let ids: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();
        let data_vals: Vec<Option<String>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 4 == 0 {
                    None
                } else {
                    Some(format!("row_{:06}", batch_start + i))
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

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &schema,
        WriterOptions {
            num_buckets: 2,
            row_group_max_size: 64 * 1024, // Small to force multiple row groups
            ..Default::default()
        },
    )
    .unwrap();
    for batch in &batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();
    let file_data = writer.output().buf.clone();
    let shared_data = Arc::new(file_data);
    let file_len = shared_data.len() as u64;

    // Determine number of row groups
    let probe_input = SharedInputFile {
        data: Arc::clone(&shared_data),
    };
    let probe_reader = MosaicReader::new(probe_input, file_len).unwrap();
    let num_rgs = probe_reader.num_row_groups();
    assert!(
        num_rgs > 1,
        "expected multiple row groups, got {}",
        num_rgs
    );
    println!(
        "test_concurrent_different_row_groups: {} row groups",
        num_rgs
    );

    // Spawn one thread per row group
    let handles: Vec<_> = (0..num_rgs)
        .map(|rg_index| {
            let data = Arc::clone(&shared_data);
            thread::spawn(move || {
                let input = SharedInputFile { data };
                let reader = MosaicReader::new(input, file_len).unwrap();
                let mut rg_reader = reader.row_group_reader(rg_index).unwrap();
                let batch = rg_reader.read_columns().unwrap();
                (rg_index, batch.num_rows())
            })
        })
        .collect();

    let mut total_rows = 0;
    let mut covered_rgs = vec![false; num_rgs];
    for h in handles {
        let (rg_index, rows) = h.join().unwrap();
        assert!(rows > 0, "row group {} had 0 rows", rg_index);
        total_rows += rows;
        covered_rgs[rg_index] = true;
    }
    assert_eq!(total_rows, num_rows);
    assert!(
        covered_rgs.iter().all(|&c| c),
        "not all row groups were covered"
    );
}

#[test]
fn test_concurrent_projection() {
    // Write a 50-column file. Spawn 10 threads, each projecting different column subsets.
    // All should succeed.
    let num_cols = 50;
    let num_rows = 50_000;
    let fields: Vec<Field> = (0..num_cols)
        .map(|i| Field::new(format!("c{:03}", i), DataType::Int64, true))
        .collect();
    let schema = Schema::new(fields);

    let arrays: Vec<Arc<dyn Array>> = (0..num_cols)
        .map(|col| {
            let vals: Vec<Option<i64>> = (0..num_rows)
                .map(|i| {
                    if (i + col) % 10 == 0 {
                        None
                    } else {
                        Some((i * col) as i64)
                    }
                })
                .collect();
            Arc::new(Int64Array::from(vals)) as Arc<dyn Array>
        })
        .collect();

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), arrays).unwrap();
    let file_data = write_file(
        &schema,
        &[batch],
        WriterOptions {
            num_buckets: 10,
            ..Default::default()
        },
    );
    let shared_data = Arc::new(file_data);
    let file_len = shared_data.len() as u64;

    let num_threads = 10;
    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let data = Arc::clone(&shared_data);
            thread::spawn(move || {
                let input = SharedInputFile { data };
                let reader = MosaicReader::new(input, file_len).unwrap();

                // Each thread projects a different subset of columns
                let projected_cols: Vec<usize> = (0..num_cols)
                    .filter(|&c| (c + t) % num_threads == 0)
                    .collect();

                if projected_cols.is_empty() {
                    return (t, 0usize, 0usize);
                }

                let mut total_rows = 0;
                for rg in 0..reader.num_row_groups() {
                    let mut rg_reader = reader
                        .row_group_reader_projected(rg, &projected_cols)
                        .unwrap();
                    let batch = rg_reader.read_columns().unwrap();
                    assert_eq!(
                        batch.num_columns(),
                        projected_cols.len(),
                        "thread {} rg {}: wrong column count",
                        t,
                        rg
                    );
                    total_rows += batch.num_rows();
                }
                (t, total_rows, projected_cols.len())
            })
        })
        .collect();

    for h in handles {
        let (thread_id, total_rows, num_projected) = h.join().unwrap();
        if num_projected > 0 {
            assert_eq!(
                total_rows, num_rows,
                "thread {} got {} rows, expected {}",
                thread_id, total_rows, num_rows
            );
        }
    }
}

#[test]
fn test_concurrent_read_different_files() {
    // Write 5 different files. Spawn threads, each reading a different file. All should succeed.
    let num_files = 5;
    let mut file_datas: Vec<Arc<Vec<u8>>> = Vec::new();
    let mut expected_rows: Vec<usize> = Vec::new();

    for f in 0..num_files {
        let num_rows = 10_000 + f * 5_000;
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "val",
                match f % 3 {
                    0 => DataType::Int32,
                    1 => DataType::Float64,
                    _ => DataType::Utf8,
                },
                true,
            ),
        ]);

        let ids: Vec<i64> = (0..num_rows as i64).collect();
        let val_array: Arc<dyn Array> = match f % 3 {
            0 => {
                let vals: Vec<Option<i32>> = (0..num_rows)
                    .map(|i| if i % 5 == 0 { None } else { Some(i as i32) })
                    .collect();
                Arc::new(Int32Array::from(vals))
            }
            1 => {
                let vals: Vec<Option<f64>> = (0..num_rows)
                    .map(|i| {
                        if i % 7 == 0 {
                            None
                        } else {
                            Some(i as f64 * 1.5)
                        }
                    })
                    .collect();
                Arc::new(Float64Array::from(vals))
            }
            _ => {
                let vals: Vec<Option<String>> = (0..num_rows)
                    .map(|i| {
                        if i % 3 == 0 {
                            None
                        } else {
                            Some(format!("file{}_row{}", f, i))
                        }
                    })
                    .collect();
                let refs: Vec<Option<&str>> = vals.iter().map(|s| s.as_deref()).collect();
                Arc::new(StringArray::from(refs))
            }
        };

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(ids)), val_array],
        )
        .unwrap();
        let data = write_file(
            &schema,
            &[batch],
            WriterOptions {
                num_buckets: 2 + f,
                ..Default::default()
            },
        );
        file_datas.push(Arc::new(data));
        expected_rows.push(num_rows);
    }

    let handles: Vec<_> = (0..num_files)
        .map(|f| {
            let data = Arc::clone(&file_datas[f]);
            let expected = expected_rows[f];
            thread::spawn(move || {
                let file_len = data.len() as u64;
                let input = SharedInputFile { data };
                let reader = MosaicReader::new(input, file_len).unwrap();
                let mut total = 0;
                for rg in 0..reader.num_row_groups() {
                    let mut rg_reader = reader.row_group_reader(rg).unwrap();
                    let batch = rg_reader.read_columns().unwrap();
                    total += batch.num_rows();
                }
                assert_eq!(
                    total, expected,
                    "file {}: got {} rows, expected {}",
                    f, total, expected
                );
                (f, total)
            })
        })
        .collect();

    for h in handles {
        let (file_idx, rows) = h.join().unwrap();
        println!("file {}: {} rows OK", file_idx, rows);
    }
}

#[test]
fn test_concurrent_readers_heavy() {
    // Write a large file (500K rows). Spawn 16 threads, each reading all data.
    // Verify all get identical results.
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int32, true),
        Field::new("label", DataType::Utf8, true),
    ]);
    let num_rows = 500_000;
    let batch_size = 100_000;
    let mut batches = Vec::new();
    for batch_start in (0..num_rows).step_by(batch_size) {
        let end = (batch_start + batch_size).min(num_rows);
        let count = end - batch_start;
        let ids: Vec<i64> = (batch_start as i64..(batch_start + count) as i64).collect();
        let vals: Vec<Option<i32>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 9 == 0 {
                    None
                } else {
                    Some(((batch_start + i) % 1000) as i32)
                }
            })
            .collect();
        let labels: Vec<Option<&str>> = (0..count)
            .map(|i| {
                if (batch_start + i) % 5 == 0 {
                    None
                } else {
                    Some("label")
                }
            })
            .collect();
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int32Array::from(vals)),
                Arc::new(StringArray::from(labels)),
            ],
        )
        .unwrap();
        batches.push(batch);
    }

    let file_data = write_file(
        &schema,
        &batches,
        WriterOptions {
            num_buckets: 3,
            row_group_max_size: 32 * 1024 * 1024,
            ..Default::default()
        },
    );
    let shared_data = Arc::new(file_data);
    let file_len = shared_data.len() as u64;

    let num_threads = 16;
    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let data = Arc::clone(&shared_data);
            thread::spawn(move || {
                let input = SharedInputFile { data };
                let reader = MosaicReader::new(input, file_len).unwrap();
                let num_rgs = reader.num_row_groups();
                let mut total_rows = 0usize;
                let mut rg_row_counts = Vec::with_capacity(num_rgs);
                for rg in 0..num_rgs {
                    let mut rg_reader = reader.row_group_reader(rg).unwrap();
                    let batch = rg_reader.read_columns().unwrap();
                    let rows = batch.num_rows();
                    total_rows += rows;
                    rg_row_counts.push(rows);
                }
                (t, total_rows, num_rgs, rg_row_counts)
            })
        })
        .collect();

    let mut all_results: Vec<(usize, usize, usize, Vec<usize>)> = Vec::new();
    for h in handles {
        all_results.push(h.join().unwrap());
    }

    // Verify all threads got the same total row count
    let expected_total = num_rows;
    for (thread_id, total_rows, num_rgs, rg_row_counts) in &all_results {
        assert_eq!(
            *total_rows, expected_total,
            "thread {} got {} rows, expected {}",
            thread_id, total_rows, expected_total
        );
        println!(
            "thread {}: {} total rows across {} row groups",
            thread_id, total_rows, num_rgs
        );

        // Verify all threads see the same row group structure
        assert_eq!(
            rg_row_counts, &all_results[0].3,
            "thread {} has different row group structure than thread 0",
            thread_id
        );
    }
}
