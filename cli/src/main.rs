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

mod fmt;
mod input;

use std::path::PathBuf;
use std::process::ExitCode;

use arrow_array::RecordBatch;
use clap::{Parser, Subcommand};
use paimon_mosaic_core::reader::{MosaicReader, ReaderAccess};

use crate::input::FileInput;

/// Mosaic file inspector — the cat/meta/schema/pages toolkit (cf. parquet-cli).
#[derive(Parser)]
#[command(name = "mosaic", version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the column names, types, nullability and bucket assignment.
    Schema {
        file: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Print row-group / bucket / stats metadata.
    Meta {
        file: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Print per-column encoding and slot size for each row group.
    Pages {
        file: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Print the first N rows as a table.
    Cat {
        file: PathBuf,
        /// Number of rows to print.
        #[arg(short = 'n', long, default_value_t = 10)]
        num: usize,
        /// Comma-separated columns to project.
        #[arg(short, long)]
        columns: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let res = match cli.cmd {
        Cmd::Schema { file, json } => schema(&file, json),
        Cmd::Meta { file, json } => meta(&file, json),
        Cmd::Pages { file, json } => pages(&file, json),
        Cmd::Cat { file, num, columns, json } => cat(&file, num, columns, json),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn open(file: &PathBuf) -> std::io::Result<MosaicReader<FileInput>> {
    let input = FileInput::open(file)?;
    let len = input.len();
    MosaicReader::new(input, len)
}

/// Columns in original (write) order rather than the name-sorted layout.
fn original_order(s: &paimon_mosaic_core::schema::MosaicSchema) -> Vec<usize> {
    let mut by_sorted = vec![0usize; s.columns.len()];
    for (orig, &sorted) in s.original_order.iter().enumerate() {
        by_sorted[sorted] = orig;
    }
    let mut cols: Vec<usize> = (0..s.columns.len()).collect();
    cols.sort_by_key(|&i| by_sorted[i]);
    cols
}

fn schema(file: &PathBuf, json: bool) -> std::io::Result<()> {
    let reader = open(file)?;
    let s = reader.schema();
    let cols = original_order(s);
    if json {
        let items: Vec<String> = cols.iter().map(|&i| {
            let c = &s.columns[i];
            format!("{{\"name\":{},\"type\":{},\"nullable\":{},\"bucket\":{}}}",
                fmt::json_str(&c.name), fmt::json_str(&format!("{:?}", c.data_type)), c.nullable, c.bucket_id)
        }).collect();
        println!("{{\"columns\":{},\"buckets\":{},\"fields\":[{}]}}", s.columns.len(), s.num_buckets, items.join(","));
        return Ok(());
    }
    println!("{} columns, {} buckets", s.columns.len(), s.num_buckets);
    for i in cols {
        let c = &s.columns[i];
        let null = if c.nullable { "" } else { " not null" };
        println!("  {}: {:?}{} [bucket {}]", c.name, c.data_type, null, c.bucket_id);
    }
    Ok(())
}

fn meta(file: &PathBuf, json: bool) -> std::io::Result<()> {
    let reader = open(file)?;
    let s = reader.schema();
    let nrg = reader.num_row_groups();
    let total: usize = (0..nrg).map(|i| reader.row_group_num_rows(i).unwrap_or(0)).sum();
    if json {
        let mut rgs = Vec::new();
        for rg in 0..nrg {
            let st: Vec<String> = reader.row_group_stats(rg)?.iter().map(|x| {
                let mm = match (&x.min, &x.max) {
                    (Some(lo), Some(hi)) => format!(",\"min\":{},\"max\":{}", fmt::json_str(&fmt::render_value(lo)), fmt::json_str(&fmt::render_value(hi))),
                    _ => String::new(),
                };
                format!("{{\"column\":{},\"nulls\":{}{}}}", fmt::json_str(&s.columns[x.column_index].name), x.null_count, mm)
            }).collect();
            rgs.push(format!("{{\"rows\":{},\"stats\":[{}]}}", reader.row_group_num_rows(rg)?, st.join(",")));
        }
        println!("{{\"rows\":{},\"columns\":{},\"buckets\":{},\"row_groups\":[{}]}}", total, s.columns.len(), s.num_buckets, rgs.join(","));
        return Ok(());
    }
    println!("file: {} rows, {} columns, {} buckets, {} row groups", total, s.columns.len(), s.num_buckets, nrg);
    for rg in 0..nrg {
        println!("row group {rg}: {} rows", reader.row_group_num_rows(rg)?);
        for st in reader.row_group_stats(rg)? {
            let mm = match (&st.min, &st.max) {
                (Some(lo), Some(hi)) => format!("min={} max={}", fmt::render_value(lo), fmt::render_value(hi)),
                _ => "no min/max".to_string(),
            };
            println!("    {}: nulls={} {}", s.columns[st.column_index].name, st.null_count, mm);
        }
    }
    Ok(())
}

fn pages(file: &PathBuf, json: bool) -> std::io::Result<()> {
    let reader = open(file)?;
    let s = reader.schema();
    let nrg = reader.num_row_groups();
    if json {
        let mut rgs = Vec::new();
        for rg in 0..nrg {
            let items: Vec<String> = reader.page_infos(rg)?.iter().map(|p| {
                format!("{{\"column\":{},\"bucket\":{},\"encoding\":{},\"slot_size\":{}}}",
                    fmt::json_str(&s.columns[p.column_index].name), p.bucket, fmt::json_str(fmt::encoding_name(p.encoding)), p.slot_size)
            }).collect();
            rgs.push(format!("[{}]", items.join(",")));
        }
        println!("{{\"row_groups\":[{}]}}", rgs.join(","));
        return Ok(());
    }
    for rg in 0..nrg {
        println!("row group {rg}:");
        for p in reader.page_infos(rg)? {
            let c = &s.columns[p.column_index];
            println!("    {}: bucket {} encoding={} slot={}B", c.name, p.bucket, fmt::encoding_name(p.encoding), p.slot_size);
        }
    }
    Ok(())
}

fn cat(file: &PathBuf, num: usize, columns: Option<String>, json: bool) -> std::io::Result<()> {
    let mut reader = open(file)?;
    if let Some(list) = &columns {
        let names: Vec<&str> = list.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()).collect();
        reader.project(&names)?;
    }
    let mut batches: Vec<RecordBatch> = Vec::new();
    let mut got = 0usize;
    for rg in 0..reader.num_row_groups() {
        if got >= num {
            break;
        }
        let batch = reader.row_group_reader(rg)?.read_columns()?;
        got += batch.num_rows();
        batches.push(batch);
    }
    if batches.is_empty() {
        if !json {
            println!("(no rows)");
        }
    } else if json {
        print!("{}", fmt::ndjson(&batches, num));
    } else {
        print!("{}", fmt::pretty_table(&batches, num));
    }
    Ok(())
}
