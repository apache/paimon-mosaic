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

//! Reproducible write-only workload for comparing integer cache memory.
//! Run the same example on the base and candidate revisions in separate processes:
//! cargo run --release -p paimon-mosaic-core --example integer_writer_memory -- 250000 20 paged unsigned output.mosaic
//! Use /usr/bin/time -v on Linux for peak RSS and CPU; this example also reports
//! RSS/PSS snapshots at append/close boundaries. Data is synthetic, not a vehicle trace.

use arrow_array::{ArrayRef, Int16Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use paimon_mosaic_core::writer::{FileSink, MosaicWriter, OutputFile, WriterOptions};
use std::sync::Arc;
use std::time::Instant;

fn memory(stage: &str) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let smaps = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default();
    let field = |text: &str, name: &str| -> u64 {
        text.lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|v| v.split_whitespace().next())
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    println!(
        "stage={stage} rss_kib={} peak_rss_kib={} pss_kib={} anon_kib={}",
        field(&status, "VmRSS:"),
        field(&status, "VmHWM:"),
        field(&smaps, "Pss:"),
        field(&status, "RssAnon:")
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "expected: ROWS_PER_BATCH BATCHES paged|monolithic unsigned|negative|outlier OUTPUT"
                .into(),
        );
    }
    let rows: usize = args[1].parse()?;
    let batches: usize = args[2].parse()?;
    let paged = match args[3].as_str() {
        "paged" => true,
        "monolithic" => false,
        _ => return Err("invalid layout".into()),
    };
    let pattern = args[4].as_str();
    if !matches!(pattern, "unsigned" | "negative" | "outlier") {
        return Err("invalid pattern".into());
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("u8_as_i16", DataType::Int16, false),
        Field::new("u16_as_i32", DataType::Int32, false),
        Field::new("u32_as_i64", DataType::Int64, false),
    ]));
    let options = WriterOptions {
        num_buckets: 1,
        row_group_max_size: u64::MAX,
        zstd_level: 1,
        max_dict_entries: 0,
        page_size_threshold: if paged { 1 } else { usize::MAX },
        ..Default::default()
    };
    let sink = FileSink::create(std::path::Path::new(&args[5]))?;
    let mut writer = MosaicWriter::new(sink, &schema, options)?;
    let mut append_ns = 0u128;
    let mut max_append_ns = 0u128;
    memory("open");
    for batch in 0..batches {
        let mut v16 = Vec::with_capacity(rows);
        let mut v32 = Vec::with_capacity(rows);
        let mut v64 = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut x = (row + batch * rows) as u64;
            x ^= x >> 30;
            x = x.wrapping_mul(0xbf58476d1ce4e5b9);
            x ^= x >> 27;
            x = x.wrapping_mul(0x94d049bb133111eb);
            x ^= x >> 31;
            if pattern == "negative"
                || (pattern == "outlier" && batch + 1 == batches && row + 1 == rows)
            {
                v16.push(x as i16);
                v32.push(x as i32);
                v64.push(x as i64);
            } else {
                v16.push((x as u8) as i16);
                v32.push((x as u16) as i32);
                v64.push((x as u32) as i64);
            }
        }
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Int16Array::from(v16)),
            Arc::new(Int32Array::from(v32)),
            Arc::new(Int64Array::from(v64)),
        ];
        let input = RecordBatch::try_new(schema.clone(), arrays)?;
        let start = Instant::now();
        writer.write_batch(&input)?;
        let elapsed = start.elapsed().as_nanos();
        append_ns += elapsed;
        max_append_ns = max_append_ns.max(elapsed);
        if batch + 1 == batches {
            memory("last_append");
        }
    }
    memory("before_close");
    let start = Instant::now();
    writer.close()?;
    let close_ns = start.elapsed().as_nanos();
    memory("after_close");
    println!(
        "rows={} append_ns={} max_append_ns={} close_ns={} file_bytes={}",
        rows * batches,
        append_ns,
        max_append_ns,
        close_ns,
        writer.output().pos()
    );
    drop(writer);
    memory("after_free");
    Ok(())
}
