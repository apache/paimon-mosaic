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

mod filter;
mod fmt;
mod jsonout;
mod input;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use arrow::array::RecordBatch;
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
        /// Comma-separated columns to show (default: all).
        #[arg(short, long)]
        columns: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Print all rows as a table (use -n to limit).
    Cat {
        file: PathBuf,
        /// Limit to N rows (default: all).
        #[arg(short = 'n', long)]
        num: Option<usize>,
        /// Comma-separated columns to project.
        #[arg(short, long)]
        columns: Option<String>,
        /// Row filter, e.g. `id>100` or `kind=a` (one condition).
        #[arg(long)]
        r#where: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Print the first N rows (default 10).
    Head {
        file: PathBuf,
        #[arg(short = 'n', long, default_value_t = 10)]
        num: usize,
        #[arg(short, long)]
        columns: Option<String>,
        #[arg(long)]
        r#where: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Print the total row count.
    Count {
        file: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Print the file footer: version, buckets, compression, offsets.
    Footer {
        file: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Print on-disk bytes per column (summed over row groups).
    ColumnSize {
        file: PathBuf,
        /// Comma-separated columns to show (default: all).
        #[arg(short, long)]
        columns: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Print the dictionary of a dict-encoded column.
    Dictionary {
        file: PathBuf,
        /// Column name to dump.
        #[arg(short = 'c', long)]
        column: String,
        #[arg(long)]
        json: bool,
    },
    /// Print bucket layout per row group (Mosaic's column grouping).
    Buckets {
        file: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Import a CSV or JSON file into a new Mosaic file (schema inferred).
    Convert {
        /// Input: CSV (.csv, header row) or JSON lines (.json/.ndjson/.jsonl).
        input: PathBuf,
        /// Output .mosaic path.
        #[arg(short, long)]
        out: PathBuf,
        /// Columns to build min/max stats for (comma-separated).
        #[arg(long)]
        stats: Option<String>,
        /// Overwrite the output file if it already exists.
        #[arg(long)]
        overwrite: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let res = match cli.cmd {
        Cmd::Schema { file, json } => schema(&file, json),
        Cmd::Meta { file, json } => meta(&file, json),
        Cmd::Pages {
            file,
            columns,
            json,
        } => pages(&file, columns, json),
        Cmd::Cat {
            file,
            num,
            columns,
            r#where,
            json,
        } => cat(&file, num.unwrap_or(usize::MAX), columns, r#where, json),
        Cmd::Head {
            file,
            num,
            columns,
            r#where,
            json,
        } => cat(&file, num, columns, r#where, json),
        Cmd::Count { file, json } => count(&file, json),
        Cmd::Footer { file, json } => footer(&file, json),
        Cmd::ColumnSize {
            file,
            columns,
            json,
        } => column_size(&file, columns, json),
        Cmd::Dictionary { file, column, json } => dictionary(&file, &column, json),
        Cmd::Buckets { file, json } => buckets(&file, json),
        Cmd::Convert {
            input,
            out,
            stats,
            overwrite,
        } => convert(&input, &out, stats, overwrite),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn open(file: &Path) -> std::io::Result<MosaicReader<FileInput>> {
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

/// Split a comma list into trimmed, non-empty names (e.g. `-c a, b,` -> [a, b]).
fn parse_comma_list(l: &str) -> Vec<String> {
    l.split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(String::from)
        .collect()
}

/// Parse a `-c a,b` list into a name set, or `None` for "all columns".
fn col_filter(
    columns: &Option<String>,
    s: &paimon_mosaic_core::schema::MosaicSchema,
) -> std::io::Result<Option<std::collections::HashSet<String>>> {
    let Some(l) = columns else { return Ok(None) };
    let set: std::collections::HashSet<String> = parse_comma_list(l).into_iter().collect();
    if let Some(bad) = set
        .iter()
        .find(|n| !s.columns.iter().any(|c| &c.name == *n))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("column '{bad}' not found in schema"),
        ));
    }
    Ok(Some(set))
}

/// Add `total` across `cols`, distributing the remainder so the parts sum exactly.
fn split_evenly(total: usize, cols: &[usize], acc: &mut [usize]) {
    if cols.is_empty() {
        return;
    }
    let share = total / cols.len();
    let mut rem = total % cols.len();
    for &c in cols {
        acc[c] += share
            + if rem > 0 {
                rem -= 1;
                1
            } else {
                0
            };
    }
}

fn schema(file: &Path, json: bool) -> std::io::Result<()> {
    let reader = open(file)?;
    let s = reader.schema();
    let cols = original_order(s);
    if json {
        let fields = cols
            .iter()
            .map(|&i| {
                let c = &s.columns[i];
                jsonout::SchemaField {
                    name: c.name.clone(),
                    ty: format!("{:?}", c.data_type),
                    nullable: c.nullable,
                    bucket: c.bucket_id as u32,
                }
            })
            .collect();
        println!(
            "{}",
            jsonout::line(&jsonout::Schema {
                columns: s.columns.len(),
                buckets: s.num_buckets,
                fields,
            })
        );
        return Ok(());
    }
    println!("{} columns, {} buckets", s.columns.len(), s.num_buckets);
    for i in cols {
        let c = &s.columns[i];
        let null = if c.nullable { "" } else { " not null" };
        println!(
            "  {}: {:?}{} [bucket {}]",
            fmt::safe(&c.name), c.data_type, null, c.bucket_id
        );
    }
    Ok(())
}

fn meta(file: &Path, json: bool) -> std::io::Result<()> {
    let reader = open(file)?;
    let s = reader.schema();
    let nrg = reader.num_row_groups();
    let total: usize = (0..nrg)
        .map(|i| reader.row_group_num_rows(i))
        .sum::<std::io::Result<usize>>()?;
    if json {
        let mut row_groups = Vec::new();
        for rg in 0..nrg {
            let stats = reader
                .row_group_stats(rg)?
                .iter()
                .map(|x| {
                    let (min, max) = match (&x.min, &x.max) {
                        (Some(lo), Some(hi)) => {
                            (Some(fmt::render_json(lo)), Some(fmt::render_json(hi)))
                        }
                        _ => (None, None),
                    };
                    jsonout::Stat {
                        column: s.columns[x.column_index].name.clone(),
                        nulls: x.null_count,
                        min,
                        max,
                    }
                })
                .collect();
            row_groups.push(jsonout::MetaRg {
                rows: reader.row_group_num_rows(rg)?,
                stats,
            });
        }
        println!(
            "{}",
            jsonout::line(&jsonout::Meta {
                rows: total,
                columns: s.columns.len(),
                buckets: s.num_buckets,
                row_groups,
            })
        );
        return Ok(());
    }
    println!(
        "file: {} rows, {} columns, {} buckets, {} row groups",
        total,
        s.columns.len(),
        s.num_buckets,
        nrg
    );
    for rg in 0..nrg {
        println!("row group {rg}: {} rows", reader.row_group_num_rows(rg)?);
        for st in reader.row_group_stats(rg)? {
            let mm = match (&st.min, &st.max) {
                (Some(lo), Some(hi)) => format!(
                    "min={} max={}",
                    fmt::render_value(lo),
                    fmt::render_value(hi)
                ),
                _ => "no min/max".to_string(),
            };
            println!(
                "    {}: nulls={} {}",
                fmt::safe(&s.columns[st.column_index].name), st.null_count, mm
            );
        }
    }
    Ok(())
}

fn pages(file: &Path, columns: Option<String>, json: bool) -> std::io::Result<()> {
    let reader = open(file)?;
    let s = reader.schema();
    let want = col_filter(&columns, s)?;
    let nrg = reader.num_row_groups();
    if json {
        let mut row_groups = Vec::new();
        for rg in 0..nrg {
            let pgs = reader
                .page_infos(rg)?
                .iter()
                .filter(|p| {
                    want.as_ref()
                        .is_none_or(|w| w.contains(&s.columns[p.column_index].name))
                })
                .map(|p| jsonout::Page {
                    column: s.columns[p.column_index].name.clone(),
                    bucket: p.bucket,
                    encoding: fmt::encoding_name(p.encoding),
                    slot_size: p.slot_size,
                })
                .collect();
            row_groups.push(pgs);
        }
        println!("{}", jsonout::line(&jsonout::Pages { row_groups }));
        return Ok(());
    }
    for rg in 0..nrg {
        println!("row group {rg}:");
        for p in reader.page_infos(rg)? {
            let c = &s.columns[p.column_index];
            if want.as_ref().is_some_and(|w| !w.contains(&c.name)) {
                continue;
            }
            println!(
                "    {}: bucket {} encoding={} slot={}B",
                fmt::safe(&c.name),
                p.bucket,
                fmt::encoding_name(p.encoding),
                p.slot_size
            );
        }
    }
    Ok(())
}

fn count(file: &Path, json: bool) -> std::io::Result<()> {
    let reader = open(file)?;
    let n: usize = (0..reader.num_row_groups())
        .map(|i| reader.row_group_num_rows(i))
        .sum::<std::io::Result<usize>>()?;
    if json {
        println!("{}", jsonout::line(&jsonout::Count { rows: n }));
    } else {
        println!("{}", n);
    }
    Ok(())
}

fn convert(
    input: &Path,
    out: &Path,
    stats: Option<String>,
    overwrite: bool,
) -> std::io::Result<()> {
    if out.exists() && !overwrite {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("{} exists (use --overwrite to replace)", out.display()),
        ));
    }
    use arrow::error::ArrowError;
    use paimon_mosaic_core::writer::{MosaicWriter, WriterOptions};
    let bad = |e: ArrowError| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string());
    let is_json = matches!(
        input.extension().and_then(|e| e.to_str()),
        Some("json") | Some("ndjson") | Some("jsonl")
    );
    // Infer schema, then build a batch iterator — CSV (header) or JSON (one object per line).
    type Batches = Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>>>;
    let (schema, reader): (arrow::datatypes::Schema, Batches) = if is_json {
        let mut r = std::io::BufReader::new(std::fs::File::open(input)?);
        let (schema, _) = arrow::json::reader::infer_json_schema(&mut r, None).map_err(bad)?;
        let rd = arrow::json::ReaderBuilder::new(std::sync::Arc::new(schema.clone()))
            .build(std::io::BufReader::new(std::fs::File::open(input)?))
            .map_err(bad)?;
        (schema, Box::new(rd))
    } else {
        let (schema, _) = arrow::csv::reader::Format::default()
            .with_header(true)
            .infer_schema(std::io::BufReader::new(std::fs::File::open(input)?), None)
            .map_err(bad)?;
        let rd = arrow::csv::ReaderBuilder::new(std::sync::Arc::new(schema.clone()))
            .with_header(true)
            .build(std::io::BufReader::new(std::fs::File::open(input)?))
            .map_err(bad)?;
        (schema, Box::new(rd))
    };
    let opts = WriterOptions {
        stats_columns: stats.map(|s| parse_comma_list(&s)).unwrap_or_default(),
        ..Default::default()
    };
    // Write to a temp file and rename on success, so a mid-stream failure never
    // leaves a truncated .mosaic in place.
    let tmp = out.with_extension("mosaic.tmp");
    let mut rows = 0;
    let res = (|| {
        let sink = paimon_mosaic_core::writer::FileSink::create(&tmp)?;
        let mut w = MosaicWriter::new(sink, &schema, opts)?;
        for batch in reader {
            let batch = batch
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            rows += batch.num_rows();
            w.write_batch(&batch)?;
        }
        w.close()
    })();
    if let Err(e) = res {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, out)?;
    let plural = |n: usize, w: &str| {
        if n == 1 {
            format!("1 {w}")
        } else {
            format!("{n} {w}s")
        }
    };
    println!(
        "wrote {} ({}, {})",
        out.display(),
        plural(rows, "row"),
        plural(schema.fields().len(), "column")
    );
    Ok(())
}

fn cat(
    file: &Path,
    num: usize,
    columns: Option<String>,
    filter: Option<String>,
    json: bool,
) -> std::io::Result<()> {
    let mut reader = open(file)?;
    let pred = filter
        .as_deref()
        .map(filter::parse_where)
        .transpose()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    // The display columns; the filter column is read even if projected out, then
    // dropped before printing, so `--where` works on a hidden column.
    let mut display: Vec<String> = Vec::new();
    if let Some(list) = &columns {
        display = parse_comma_list(list);
        let mut read: Vec<&str> = display.iter().map(String::as_str).collect();
        if let Some(p) = &pred {
            if !read.contains(&p.column.as_str()) {
                read.push(&p.column);
            }
        }
        reader.project(&read)?;
    }
    // Column index of the filter target, for stats-based row-group skipping.
    let pred_col = pred.as_ref().and_then(|p| {
        reader
            .schema()
            .columns
            .iter()
            .position(|c| c.name == p.column)
    });
    let mut batches: Vec<RecordBatch> = Vec::new();
    let mut got = 0usize;
    for rg in 0..reader.num_row_groups() {
        if got >= num {
            break;
        }
        // Pushdown: skip a row group when its min/max prove no row can match.
        if let (Some(p), Some(ci)) = (&pred, pred_col) {
            if let Some(st) = reader
                .row_group_stats(rg)?
                .iter()
                .find(|s| s.column_index == ci)
            {
                if filter::stats_exclude(p, &st.min, &st.max) {
                    continue;
                }
            }
        }
        let mut batch = reader.row_group_reader(rg)?.read_columns()?;
        if let Some(p) = &pred {
            batch = filter::apply_where(&batch, p)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        }
        // Drop the filter-only column so it isn't printed when -c excluded it.
        if !display.is_empty() {
            let keep: Vec<usize> = display
                .iter()
                .filter_map(|n| batch.schema().index_of(n).ok())
                .collect();
            batch = batch
                .project(&keep)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
        }
        let batch_rows = batch.num_rows();
        // JSON rows are independent, so stream each group out instead of holding
        // every batch — `cat --json` on a huge file stays bounded. The table path
        // must buffer: column widths need all rows before the first line prints.
        if json {
            print!("{}", fmt::ndjson(&[batch], num - got)?);
        } else {
            batches.push(batch);
        }
        got += batch_rows;
    }
    if json {
        // (no rows) stays silent for JSON; nothing to print.
    } else if batches.iter().all(|b| b.num_rows() == 0) {
        println!("(no rows)");
    } else {
        print!("{}", fmt::pretty_table(&batches, num));
    }
    Ok(())
}

fn footer(file: &Path, json: bool) -> std::io::Result<()> {
    use paimon_mosaic_core::spec::{COMPRESSION_ZSTD, MAGIC, VERSION};
    let reader = open(file)?;
    let s = reader.schema();
    let comp = if reader.compression() == COMPRESSION_ZSTD {
        "zstd"
    } else {
        "none"
    };
    let magic = std::str::from_utf8(&MAGIC).unwrap_or("MOSA");
    if json {
        println!(
            "{}",
            jsonout::line(&jsonout::Footer {
                magic: magic.to_string(),
                version: VERSION as u32,
                buckets: s.num_buckets,
                row_groups: reader.num_row_groups(),
                compression: comp.to_string(),
            })
        );
    } else {
        println!(
            "magic={} version={} buckets={} row_groups={} compression={}",
            magic,
            VERSION,
            s.num_buckets,
            reader.num_row_groups(),
            comp
        );
    }
    Ok(())
}

fn column_size(file: &Path, columns: Option<String>, json: bool) -> std::io::Result<()> {
    let reader = open(file)?;
    let s = reader.schema();
    let want = col_filter(&columns, s)?;
    let mut bytes = vec![0usize; s.columns.len()];
    let mut approx = vec![false; s.columns.len()];
    for rg in 0..reader.num_row_groups() {
        // Paged buckets store each column in its own slot → exact per-column bytes.
        // Read slot sizes from the directory only (no slot decode/decompress).
        for (ci, sz) in reader.slot_sizes(rg)?.into_iter().enumerate() {
            bytes[ci] += sz;
        }
        // Monolithic buckets are one blob; split evenly and mark approximate when
        // more than one column shares the bucket (a single-column bucket is exact).
        for b in reader.bucket_infos(rg)? {
            if b.kind != paimon_mosaic_core::reader::BucketKind::Monolithic || b.columns.is_empty()
            {
                continue;
            }
            split_evenly(b.size, &b.columns, &mut bytes);
            if b.columns.len() > 1 {
                for &c in &b.columns {
                    approx[c] = true;
                }
            }
        }
    }
    let cols: Vec<usize> = original_order(s)
        .into_iter()
        .filter(|&i| want.as_ref().is_none_or(|w| w.contains(&s.columns[i].name)))
        .collect();
    let comp: usize = cols.iter().map(|&i| bytes[i]).sum();
    let any_approx = cols.iter().any(|&i| approx[i]);
    if json {
        let columns = cols
            .iter()
            .map(|&i| jsonout::ColumnBytes {
                column: s.columns[i].name.clone(),
                bytes: bytes[i],
                approximate: approx[i],
            })
            .collect();
        println!(
            "{}",
            jsonout::line(&jsonout::ColumnSize {
                columns,
                total_bytes: comp,
            })
        );
    } else {
        for i in cols {
            println!(
                "  {}: {} B{}",
                fmt::safe(&s.columns[i].name),
                bytes[i],
                if approx[i] { " (approx)" } else { "" }
            );
        }
        println!(
            "  total: {} B{}",
            comp,
            if any_approx {
                " (some columns approximate)"
            } else {
                ""
            }
        );
    }
    Ok(())
}

fn dictionary(file: &Path, column: &str, json: bool) -> std::io::Result<()> {
    let reader = open(file)?;
    let col = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == column)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("column '{column}' not found"),
            )
        })?;
    if json {
        let mut row_groups = Vec::new();
        for rg in 0..reader.num_row_groups() {
            row_groups.push(
                reader
                    .dictionary(rg, col)?
                    .map(|vals| vals.iter().map(fmt::render_json).collect()),
            );
        }
        println!(
            "{}",
            jsonout::line(&jsonout::Dictionary {
                column: column.to_string(),
                row_groups,
            })
        );
        return Ok(());
    }
    for rg in 0..reader.num_row_groups() {
        match reader.dictionary(rg, col)? {
            Some(vals) => {
                println!("row group {rg}: {} entries", vals.len());
                for (i, v) in vals.iter().enumerate() {
                    println!("    {i}: {}", fmt::render_value(v));
                }
            }
            None => println!("row group {rg}: not dict-encoded"),
        }
    }
    Ok(())
}

fn buckets(file: &Path, json: bool) -> std::io::Result<()> {
    let reader = open(file)?;
    let s = reader.schema();
    let name = |i: usize| fmt::safe(&s.columns[i].name);
    let mut rgs = Vec::new();
    for rg in 0..reader.num_row_groups() {
        let infos = reader.bucket_infos(rg)?;
        if json {
            let items = infos
                .iter()
                .map(|b| jsonout::Bucket {
                    bucket: b.bucket,
                    kind: fmt::bucket_kind(b.kind).to_string(),
                    size: b.size,
                    uncompressed: b.uncompressed,
                    columns: b.columns.iter().map(|&i| name(i)).collect(),
                })
                .collect();
            rgs.push(items);
        } else {
            println!("row group {rg}:");
            for b in &infos {
                let cols: Vec<String> = b.columns.iter().map(|&i| name(i)).collect();
                println!(
                    "    bucket {}: {} {}B{} [{}]",
                    b.bucket,
                    fmt::bucket_kind(b.kind),
                    b.size,
                    fmt::ratio(b.size, b.uncompressed),
                    cols.join(", ")
                );
            }
        }
    }
    if json {
        println!("{}", jsonout::line(&jsonout::Buckets { row_groups: rgs }));
    }
    Ok(())
}
