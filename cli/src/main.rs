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
mod input;
mod jsonout;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use arrow::array::timezone::Tz;
use arrow::array::types::{
    ArrowPrimitiveType, Date32Type, Decimal128Type, Float32Type, Float64Type, Int32Type, Int64Type,
    Time32MillisecondType, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType,
};
use arrow::array::{
    new_null_array, ArrayRef, BooleanArray, PrimitiveArray, RecordBatch, StringArray,
};
use arrow::compute::kernels::cast_utils::{string_to_datetime, Parser as ArrowValueParser};
use arrow::datatypes::{DataType, Field, Fields, Schema, TimeUnit};
use clap::{Parser, Subcommand};
use paimon_mosaic_core::reader::{MosaicReader, ReaderAccess};
use serde::de::{DeserializeSeed, Error as _, IgnoredAny, MapAccess, Visitor};
use serde_json::value::RawValue;
use serde_json::Value;

use crate::input::FileInput;

/// Mosaic file inspector — the cat/meta/schema/pages toolkit.
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
    /// Print rows as a table (default: all rows; use -n to limit).
    Cat {
        file: PathBuf,
        /// Limit to N rows.
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
    /// Create a Mosaic file from a JSON data file.
    Convert {
        /// Input JSON data file (.json/.ndjson/.jsonl).
        input: PathBuf,
        /// Output .mosaic path.
        #[arg(short = 'o', long = "output")]
        out: PathBuf,
        /// Avro record schema file (supported subset; see the CLI README).
        #[arg(short = 's', long)]
        schema: Option<PathBuf>,
        /// Columns to keep; each occurrence accepts a comma-separated list.
        #[arg(short = 'c', long = "column")]
        columns: Vec<String>,
        /// Columns to build min/max stats for (comma-separated); `cat --where`
        /// uses them to skip row groups.
        #[arg(long)]
        stats: Option<String>,
        /// Overwrite the output file if it already exists.
        #[arg(long)]
        overwrite: bool,
    },
    /// Create a Mosaic file from CSV data.
    ConvertCsv {
        /// Input CSV path(s).
        inputs: Vec<PathBuf>,
        /// Output .mosaic path.
        #[arg(short = 'o', long = "output")]
        out: PathBuf,
        /// Avro record schema file (supported subset, scalar fields only; see the CLI README).
        #[arg(short = 's', long)]
        schema: Option<PathBuf>,
        /// Do not allow null values for inferred fields; repeat for multiple fields.
        #[arg(long)]
        require: Vec<String>,
        /// Delimiter character.
        #[arg(long, default_value = ",")]
        delimiter: String,
        /// Escape character (disabled by default).
        #[arg(long)]
        escape: Option<String>,
        /// Quote character.
        #[arg(long, default_value = "\"")]
        quote: String,
        /// Don't use first line as CSV header.
        #[arg(long, conflicts_with = "header")]
        no_header: bool,
        /// Line to use as a header. Must match the CSV settings.
        #[arg(long, conflicts_with = "no_header")]
        header: Option<String>,
        /// Lines to skip before CSV start.
        #[arg(long, default_value_t = 0)]
        skip_lines: usize,
        /// Columns to build min/max stats for (comma-separated); `cat --where`
        /// uses them to skip row groups.
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
            schema,
            columns,
            stats,
            overwrite,
        } => convert(
            &input,
            &out,
            schema.as_deref(),
            &columns,
            stats.as_deref(),
            overwrite,
        ),
        Cmd::ConvertCsv {
            inputs,
            out,
            schema,
            require,
            delimiter,
            escape,
            quote,
            no_header,
            header,
            skip_lines,
            stats,
            overwrite,
        } => {
            let options = CsvConvertOptions {
                delimiter,
                escape,
                quote,
                no_header,
                header,
                skip_lines,
            };
            convert_csv(
                &inputs,
                &out,
                schema.as_deref(),
                &require,
                options,
                stats.as_deref(),
                overwrite,
            )
        }
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", fmt::safe(&e.to_string()));
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

/// True when `name` is selected by a `-c` set (`None` = all columns).
fn selected(want: &Option<std::collections::HashSet<String>>, name: &str) -> bool {
    want.as_ref().is_none_or(|w| w.contains(name))
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
            fmt::safe(&c.name),
            c.data_type,
            null,
            c.bucket_id
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
                fmt::safe(&s.columns[st.column_index].name),
                st.null_count,
                mm
            );
        }
    }
    Ok(())
}

fn pages(file: &Path, columns: Option<String>, json: bool) -> std::io::Result<()> {
    let reader = open(file)?;
    let s = reader.schema();
    let want = col_filter(&columns, s)?;
    let cols: Vec<usize> = (0..s.columns.len())
        .filter(|&i| selected(&want, &s.columns[i].name))
        .collect();
    let nrg = reader.num_row_groups();
    if json {
        let mut row_groups = Vec::new();
        for rg in 0..nrg {
            let pgs = reader
                .page_infos_projected(rg, &cols)?
                .iter()
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
        for p in reader.page_infos_projected(rg, &cols)? {
            let c = &s.columns[p.column_index];
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
    schema: Option<&Path>,
    columns: &[String],
    stats: Option<&str>,
    overwrite: bool,
) -> std::io::Result<()> {
    convert_with_json_record_limit(
        input,
        out,
        schema,
        columns,
        stats,
        overwrite,
        MAX_JSON_RECORD_BYTES,
    )
}

fn convert_with_json_record_limit(
    input: &Path,
    out: &Path,
    schema: Option<&Path>,
    columns: &[String],
    stats: Option<&str>,
    overwrite: bool,
    max_record_bytes: usize,
) -> std::io::Result<()> {
    use arrow::error::ArrowError;
    let bad = |e: ArrowError| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string());
    if !is_json_input(input) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "convert only supports JSON inputs (.json/.ndjson/.jsonl); use convert-csv for CSV data",
        ));
    }
    let columns = parse_convert_columns(columns)?;
    ensure_can_write(out, overwrite)?;
    let explicit_schema = schema.map(load_convert_schema).transpose()?;
    let open = || -> std::io::Result<_> {
        Ok(std::io::BufReader::with_capacity(
            JSON_INPUT_BUFFER_BYTES,
            JsonRecordLimitReader::new(std::fs::File::open(input)?, max_record_bytes),
        ))
    };
    let has_explicit_schema = explicit_schema.is_some();
    let schema = match explicit_schema {
        Some(schema) => schema,
        None if columns.is_empty() => arrow::json::reader::infer_json_schema(&mut open()?, None)
            .map(|(schema, _)| schema)
            .map_err(bad)?,
        None => infer_projected_json_schema(open()?, &columns).map_err(bad)?,
    };
    let schema = project_convert_schema(schema, &columns)?;
    reject_null_inferred_fields(&schema)?;
    reject_json_unsupported_fields(&schema)?;
    if has_explicit_schema && schema_needs_json_validation(&schema) {
        return write_mosaic(out, overwrite, &schema, stats, |writer, rows| {
            write_validated_json_input(open()?, &schema, writer, rows)
        });
    }
    let reader = arrow::json::ReaderBuilder::new(Arc::new(schema.clone()))
        .build(open()?)
        .map_err(bad)?;
    write_mosaic(out, overwrite, &schema, stats, |writer, rows| {
        for batch in reader {
            let batch = batch
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            *rows += batch.num_rows();
            writer.write_batch(&batch)?;
        }
        Ok(())
    })
}

fn write_validated_json_input<R: std::io::BufRead>(
    reader: R,
    schema: &Schema,
    writer: &mut paimon_mosaic_core::writer::MosaicWriter<paimon_mosaic_core::writer::FileSink>,
    rows: &mut usize,
) -> std::io::Result<()> {
    for_each_validated_json_batch(reader, schema, TARGET_CONVERT_BATCH_BYTES, |batch| {
        *rows += batch.num_rows();
        writer.write_batch(&batch)
    })
}

const DEFAULT_JSON_BATCH_SIZE: usize = 1024;
const TARGET_CONVERT_BATCH_BYTES: usize = 16 * 1024 * 1024;
const JSON_INPUT_BUFFER_BYTES: usize = 1024 * 1024;
// Hard ceiling on a single raw JSON root value. A streaming lexical reader
// enforces it before serde_json or Arrow can materialize the value.
const MAX_JSON_RECORD_BYTES: usize = 16 * 1024 * 1024;
const MAX_NORMALIZED_JSON_RECORD_BYTES: usize = MAX_JSON_RECORD_BYTES;
const MAX_JSON_STRUCTURAL_UNITS: usize = 100_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JsonRecordScanState {
    BetweenRecords,
    Compound {
        depth: usize,
        in_string: bool,
        escaped: bool,
    },
    String {
        escaped: bool,
    },
    Scalar,
}

struct JsonRecordScanner {
    state: JsonRecordScanState,
    record: usize,
    record_bytes: usize,
    line_bytes: usize,
    max_record_bytes: usize,
    structural_units: usize,
    max_structural_units: usize,
}

impl JsonRecordScanner {
    #[inline]
    fn new(max_record_bytes: usize) -> Self {
        Self::new_with_limits(max_record_bytes, MAX_JSON_STRUCTURAL_UNITS)
    }

    fn new_with_limits(max_record_bytes: usize, max_structural_units: usize) -> Self {
        Self {
            state: JsonRecordScanState::BetweenRecords,
            record: 0,
            record_bytes: 0,
            line_bytes: 0,
            max_record_bytes,
            structural_units: 0,
            max_structural_units,
        }
    }

    #[inline(always)]
    fn count_record_byte(&mut self) -> std::io::Result<()> {
        if self.record_bytes >= self.max_record_bytes {
            return Err(invalid_schema(format!(
                "JSON record {} exceeds the {} byte limit",
                self.record, self.max_record_bytes
            )));
        }
        self.record_bytes += 1;
        Ok(())
    }

    #[inline(always)]
    fn count_structural_units(&mut self, units: usize) -> std::io::Result<()> {
        if units
            > self
                .max_structural_units
                .saturating_sub(self.structural_units)
        {
            return Err(invalid_schema(format!(
                "JSON record {} exceeds the {} structural unit limit",
                self.record, self.max_structural_units
            )));
        }
        self.structural_units += units;
        Ok(())
    }

    #[inline(always)]
    fn scan(&mut self, byte: u8) -> std::io::Result<()> {
        let mut reprocess = true;
        while reprocess {
            reprocess = false;
            match self.state {
                JsonRecordScanState::BetweenRecords => {
                    if is_json_whitespace(byte) {
                        continue;
                    }
                    self.record = self.record.saturating_add(1);
                    self.record_bytes = 0;
                    self.structural_units = 0;
                    self.count_record_byte()?;
                    self.count_structural_units(matches!(byte, b'{' | b'[' | b'"') as usize)?;
                    self.state = match byte {
                        b'{' | b'[' => JsonRecordScanState::Compound {
                            depth: 1,
                            in_string: false,
                            escaped: false,
                        },
                        b'"' => JsonRecordScanState::String { escaped: false },
                        _ => JsonRecordScanState::Scalar,
                    };
                }
                JsonRecordScanState::Compound {
                    mut depth,
                    mut in_string,
                    mut escaped,
                } => {
                    self.count_record_byte()?;
                    if in_string {
                        if escaped {
                            escaped = false;
                        } else if byte == b'\\' {
                            escaped = true;
                        } else if byte == b'"' {
                            in_string = false;
                        }
                    } else {
                        match byte {
                            b'"' => {
                                self.count_structural_units(1)?;
                                in_string = true;
                            }
                            b'{' | b'[' => {
                                self.count_structural_units(1)?;
                                depth = depth.saturating_add(1);
                            }
                            b'}' | b']' => {
                                self.count_structural_units(1)?;
                                depth = depth.saturating_sub(1);
                            }
                            b',' | b':' => self.count_structural_units(1)?,
                            _ => {}
                        }
                    }
                    self.state = if depth == 0 && !in_string {
                        JsonRecordScanState::BetweenRecords
                    } else {
                        JsonRecordScanState::Compound {
                            depth,
                            in_string,
                            escaped,
                        }
                    };
                }
                JsonRecordScanState::String { mut escaped } => {
                    self.count_record_byte()?;
                    if escaped {
                        escaped = false;
                    } else if byte == b'\\' {
                        escaped = true;
                    } else if byte == b'"' {
                        self.state = JsonRecordScanState::BetweenRecords;
                        continue;
                    }
                    self.state = JsonRecordScanState::String { escaped };
                }
                JsonRecordScanState::Scalar => {
                    if is_json_whitespace(byte) {
                        self.state = JsonRecordScanState::BetweenRecords;
                    } else if matches!(byte, b'{' | b'[' | b'"') {
                        self.state = JsonRecordScanState::BetweenRecords;
                        reprocess = true;
                    } else {
                        self.count_record_byte()?;
                    }
                }
            }
        }
        self.count_line_byte(byte)
    }

    #[inline(always)]
    fn count_line_byte(&mut self, byte: u8) -> std::io::Result<()> {
        if self.line_bytes >= self.max_record_bytes {
            return Err(invalid_schema(format!(
                "JSON input line exceeds the {} byte limit",
                self.max_record_bytes
            )));
        }
        self.line_bytes += 1;
        if byte == b'\n' {
            self.line_bytes = 0;
        }
        Ok(())
    }

    #[inline]
    fn scan_chunk(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        let mut offset = 0;
        while offset < bytes.len() {
            let ordinary = self.ordinary_prefix_len(&bytes[offset..]);
            if ordinary == 0 {
                self.scan(bytes[offset])?;
                offset += 1;
            } else {
                self.advance_ordinary(&bytes[offset..offset + ordinary])?;
                offset += ordinary;
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn ordinary_prefix_len(&self, bytes: &[u8]) -> usize {
        let special = match self.state {
            JsonRecordScanState::BetweenRecords => bytes
                .iter()
                .position(|&byte| byte == b'\n' || !matches!(byte, b' ' | b'\r' | b'\t')),
            JsonRecordScanState::Compound {
                in_string: false, ..
            } => bytes
                .iter()
                .position(|&byte| matches!(byte, b'"' | b'{' | b'}' | b'[' | b']' | b'\n')),
            JsonRecordScanState::Compound {
                in_string: true,
                escaped: false,
                ..
            }
            | JsonRecordScanState::String { escaped: false } => {
                memchr::memchr3(b'"', b'\\', b'\n', bytes)
            }
            JsonRecordScanState::Compound {
                in_string: true,
                escaped: true,
                ..
            }
            | JsonRecordScanState::String { escaped: true } => Some(0),
            JsonRecordScanState::Scalar => bytes
                .iter()
                .position(|&byte| is_json_whitespace(byte) || matches!(byte, b'{' | b'[' | b'"')),
        };
        special.unwrap_or(bytes.len())
    }

    #[inline(always)]
    fn advance_ordinary(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        let len = bytes.len();
        if matches!(self.state, JsonRecordScanState::BetweenRecords) {
            return self.advance_line(len);
        }
        if matches!(
            self.state,
            JsonRecordScanState::Compound {
                in_string: false,
                ..
            }
        ) {
            self.count_structural_units(memchr::memchr2_iter(b',', b':', bytes).count())?;
        }

        let record_remaining = self.max_record_bytes.saturating_sub(self.record_bytes);
        let line_remaining = self.max_record_bytes.saturating_sub(self.line_bytes);
        if len <= record_remaining.min(line_remaining) {
            self.record_bytes += len;
            self.line_bytes += len;
            return Ok(());
        }

        if record_remaining <= line_remaining {
            self.record_bytes = self.max_record_bytes.saturating_add(1);
            return Err(invalid_schema(format!(
                "JSON record {} exceeds the {} byte limit",
                self.record, self.max_record_bytes
            )));
        }
        self.line_bytes = self.max_record_bytes.saturating_add(1);
        Err(invalid_schema(format!(
            "JSON input line exceeds the {} byte limit",
            self.max_record_bytes
        )))
    }

    #[inline(always)]
    fn advance_line(&mut self, len: usize) -> std::io::Result<()> {
        let remaining = self.max_record_bytes.saturating_sub(self.line_bytes);
        if len <= remaining {
            self.line_bytes += len;
            return Ok(());
        }
        self.line_bytes = self.max_record_bytes.saturating_add(1);
        Err(invalid_schema(format!(
            "JSON input line exceeds the {} byte limit",
            self.max_record_bytes
        )))
    }
}

fn is_json_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\n' | b'\r' | b'\t')
}

struct JsonRecordLimitReader<R> {
    inner: R,
    scanner: JsonRecordScanner,
}

impl<R> JsonRecordLimitReader<R> {
    fn new(inner: R, max_record_bytes: usize) -> Self {
        Self {
            inner,
            scanner: JsonRecordScanner::new(max_record_bytes),
        }
    }
}

impl<R: std::io::Read> std::io::Read for JsonRecordLimitReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.scanner.scan_chunk(&buffer[..read])?;
        Ok(read)
    }
}

#[cfg(test)]
fn validate_json_record_limits<R: std::io::Read>(
    reader: R,
    max_record_bytes: usize,
) -> std::io::Result<()> {
    std::io::copy(
        &mut JsonRecordLimitReader::new(reader, max_record_bytes),
        &mut std::io::sink(),
    )?;
    Ok(())
}

fn for_each_validated_json_batch<R, F>(
    reader: R,
    schema: &Schema,
    byte_budget: usize,
    mut write: F,
) -> std::io::Result<()>
where
    R: std::io::Read,
    F: FnMut(RecordBatch) -> std::io::Result<()>,
{
    let bad = |e: arrow::error::ArrowError| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
    };
    let build_decoder = || {
        arrow::json::ReaderBuilder::new(Arc::new(schema.clone()))
            .with_batch_size(DEFAULT_JSON_BATCH_SIZE)
            .build_decoder()
            .map_err(bad)
    };
    let fields = json_special_fields(schema);
    let decimal_plan =
        schema_has_decimal(schema).then(|| JsonDecimalStructPlan::from_fields(schema.fields()));
    let mut decoder = build_decoder()?;
    let mut batch_bytes = 0_usize;
    let records = serde_json::Deserializer::from_reader(reader).into_iter::<Box<RawValue>>();

    for (index, raw) in records.enumerate() {
        let record = index + 1;
        let raw = raw.map_err(|e| invalid_schema(format!("invalid JSON record {record}: {e}")))?;
        let raw_bytes = raw.get().as_bytes();
        validate_json_special_values(raw_bytes, &fields, record)?;
        let normalized;
        let decode_bytes = if let Some(plan) = &decimal_plan {
            normalized = normalize_json_decimal_record(&raw, plan, record)?;
            normalized.as_bytes()
        } else {
            raw_bytes
        };
        if !decoder.is_empty()
            && (decoder.len() >= DEFAULT_JSON_BATCH_SIZE
                || batch_bytes.saturating_add(decode_bytes.len()) > byte_budget)
        {
            if let Some(batch) = decoder.flush().map_err(bad)? {
                write(batch)?;
            }
            batch_bytes = 0;
        }

        let decoded = decoder.decode(decode_bytes).map_err(bad)?;
        if decoded != decode_bytes.len() || decoder.has_partial_record() {
            return Err(invalid_schema(format!(
                "invalid JSON record {record}: decoder stopped before the record ended"
            )));
        }
        batch_bytes = batch_bytes.saturating_add(decode_bytes.len());

        if decoder.len() >= DEFAULT_JSON_BATCH_SIZE || batch_bytes >= byte_budget {
            if let Some(batch) = decoder.flush().map_err(bad)? {
                write(batch)?;
            }
            if batch_bytes > byte_budget {
                decoder = build_decoder()?;
            }
            batch_bytes = 0;
        }
    }

    if let Some(batch) = decoder.flush().map_err(bad)? {
        write(batch)?;
    }
    Ok(())
}

fn schema_has_decimal(schema: &Schema) -> bool {
    schema
        .fields()
        .iter()
        .any(|field| data_type_has_decimal(field.data_type()))
}

fn data_type_has_decimal(data_type: &DataType) -> bool {
    match data_type {
        DataType::Decimal128(_, _) => true,
        DataType::List(field) => data_type_has_decimal(field.data_type()),
        DataType::Map(entries, _) => match entries.data_type() {
            DataType::Struct(fields) => fields
                .get(1)
                .is_some_and(|field| data_type_has_decimal(field.data_type())),
            _ => false,
        },
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| data_type_has_decimal(field.data_type())),
        _ => false,
    }
}

struct JsonDecimalStructPlan {
    by_name: std::collections::HashMap<String, Option<JsonDecimalValuePlan>>,
    #[cfg(test)]
    lookup_count: std::cell::Cell<usize>,
}

enum JsonDecimalValuePlan {
    Decimal { precision: u8, scale: i8 },
    List(Box<JsonDecimalValuePlan>),
    Map(Box<JsonDecimalValuePlan>),
    Struct(JsonDecimalStructPlan),
}

impl JsonDecimalStructPlan {
    fn from_fields(fields: &Fields) -> Self {
        let mut by_name = std::collections::HashMap::with_capacity(fields.len());
        for field in fields {
            if let std::collections::hash_map::Entry::Vacant(entry) =
                by_name.entry(field.name().clone())
            {
                entry.insert(JsonDecimalValuePlan::from_data_type(field.data_type()));
            }
        }
        Self {
            by_name,
            #[cfg(test)]
            lookup_count: std::cell::Cell::new(0),
        }
    }

    fn lookup(&self, name: &str) -> Option<&JsonDecimalValuePlan> {
        #[cfg(test)]
        self.lookup_count
            .set(self.lookup_count.get().saturating_add(1));
        self.by_name.get(name).and_then(Option::as_ref)
    }

    #[cfg(test)]
    fn lookup_count(&self) -> usize {
        self.lookup_count.get()
            + self
                .by_name
                .values()
                .filter_map(Option::as_ref)
                .map(JsonDecimalValuePlan::lookup_count)
                .sum::<usize>()
    }
}

impl JsonDecimalValuePlan {
    fn from_data_type(data_type: &DataType) -> Option<Self> {
        if !data_type_has_decimal(data_type) {
            return None;
        }
        match data_type {
            DataType::Decimal128(precision, scale) => Some(Self::Decimal {
                precision: *precision,
                scale: *scale,
            }),
            DataType::List(field) => {
                Self::from_data_type(field.data_type()).map(|plan| Self::List(Box::new(plan)))
            }
            DataType::Map(entries, _) => {
                let DataType::Struct(fields) = entries.data_type() else {
                    return None;
                };
                fields
                    .get(1)
                    .and_then(|field| Self::from_data_type(field.data_type()))
                    .map(|plan| Self::Map(Box::new(plan)))
            }
            DataType::Struct(fields) => {
                Some(Self::Struct(JsonDecimalStructPlan::from_fields(fields)))
            }
            _ => None,
        }
    }

    #[cfg(test)]
    fn lookup_count(&self) -> usize {
        match self {
            Self::Decimal { .. } => 0,
            Self::List(plan) | Self::Map(plan) => plan.lookup_count(),
            Self::Struct(plan) => plan.lookup_count(),
        }
    }
}

fn normalize_json_decimal_record(
    raw: &RawValue,
    plan: &JsonDecimalStructPlan,
    record: usize,
) -> std::io::Result<String> {
    normalize_json_decimal_record_with_limit(raw, plan, record, MAX_NORMALIZED_JSON_RECORD_BYTES)
}

fn normalize_json_decimal_record_with_limit(
    raw: &RawValue,
    plan: &JsonDecimalStructPlan,
    record: usize,
    max_bytes: usize,
) -> std::io::Result<String> {
    let values: std::collections::BTreeMap<std::borrow::Cow<'_, str>, &RawValue> =
        serde_json::from_str(raw.get())
            .map_err(|e| invalid_schema(format!("invalid JSON record {record}: {e}")))?;
    let mut normalized = CappedJsonString::new(raw.get().len().min(max_bytes), max_bytes, record);
    normalized.push_str("{")?;
    for (index, (name, value)) in values.iter().enumerate() {
        if index != 0 {
            normalized.push_str(",")?;
        }
        let encoded_name = serde_json::to_string(name.as_ref())
            .map_err(|e| invalid_schema(format!("invalid JSON field name: {e}")))?;
        normalized.push_str(&encoded_name)?;
        normalized.push_str(":")?;
        if let Some(value_plan) = plan.lookup(name.as_ref()) {
            normalize_json_decimal_value(
                &mut normalized,
                value,
                value_plan,
                name.as_ref(),
                record,
            )?;
        } else {
            normalized.push_str(value.get())?;
        }
    }
    normalized.push_str("}")?;
    Ok(normalized.into_string())
}

struct CappedJsonString {
    value: String,
    max_bytes: usize,
    record: usize,
}

impl CappedJsonString {
    fn new(capacity: usize, max_bytes: usize, record: usize) -> Self {
        Self {
            value: String::with_capacity(capacity),
            max_bytes,
            record,
        }
    }

    fn push_str(&mut self, value: &str) -> std::io::Result<()> {
        if value.len() > self.max_bytes.saturating_sub(self.value.len()) {
            return Err(invalid_schema(format!(
                "normalized JSON record {} exceeds the {} byte limit",
                self.record, self.max_bytes
            )));
        }
        self.value.push_str(value);
        Ok(())
    }

    fn into_string(self) -> String {
        self.value
    }
}

fn normalize_json_decimal_value(
    normalized: &mut CappedJsonString,
    raw: &RawValue,
    plan: &JsonDecimalValuePlan,
    path: &str,
    record: usize,
) -> std::io::Result<()> {
    if raw.get() == "null" {
        return normalized.push_str(raw.get());
    }
    match plan {
        JsonDecimalValuePlan::Decimal { precision, scale } => {
            let raw_text = raw.get();
            let value = if raw_text.starts_with('"') {
                serde_json::from_str::<String>(raw_text)
                    .map_err(|e| invalid_schema(format!("invalid JSON decimal: {e}")))?
            } else {
                raw_text.to_string()
            };
            let unscaled = parse_decimal_exact(&value, *precision, *scale).map_err(|e| {
                let data_type = DataType::Decimal128(*precision, *scale);
                invalid_schema(format!(
                    "cannot parse '{}' as {} for JSON field '{}' at record {record}: {e}",
                    fmt::safe(&value),
                    data_type,
                    fmt::safe(path)
                ))
            })?;
            let encoded = serde_json::to_string(&format_decimal_unscaled(unscaled, *scale))
                .map_err(|e| invalid_schema(format!("invalid JSON decimal: {e}")))?;
            normalized.push_str(&encoded)
        }
        JsonDecimalValuePlan::List(item) => {
            let values: Vec<&RawValue> = serde_json::from_str(raw.get())
                .map_err(|e| invalid_schema(format!("invalid JSON array: {e}")))?;
            let child_path = format!("{path}[]");
            normalized.push_str("[")?;
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    normalized.push_str(",")?;
                }
                normalize_json_decimal_value(normalized, value, item, &child_path, record)?;
            }
            normalized.push_str("]")
        }
        JsonDecimalValuePlan::Map(value_plan) => {
            let values: std::collections::BTreeMap<std::borrow::Cow<'_, str>, &RawValue> =
                serde_json::from_str(raw.get())
                    .map_err(|e| invalid_schema(format!("invalid JSON map: {e}")))?;
            let child_path = format!("{path}{{}}");
            normalized.push_str("{")?;
            for (index, (name, value)) in values.iter().enumerate() {
                if index != 0 {
                    normalized.push_str(",")?;
                }
                let encoded_name = serde_json::to_string(name.as_ref())
                    .map_err(|e| invalid_schema(format!("invalid JSON map key: {e}")))?;
                normalized.push_str(&encoded_name)?;
                normalized.push_str(":")?;
                normalize_json_decimal_value(normalized, value, value_plan, &child_path, record)?;
            }
            normalized.push_str("}")
        }
        JsonDecimalValuePlan::Struct(struct_plan) => {
            let values: std::collections::BTreeMap<std::borrow::Cow<'_, str>, &RawValue> =
                serde_json::from_str(raw.get())
                    .map_err(|e| invalid_schema(format!("invalid JSON object: {e}")))?;
            normalized.push_str("{")?;
            for (index, (name, value)) in values.iter().enumerate() {
                if index != 0 {
                    normalized.push_str(",")?;
                }
                let encoded_name = serde_json::to_string(name.as_ref())
                    .map_err(|e| invalid_schema(format!("invalid JSON field name: {e}")))?;
                normalized.push_str(&encoded_name)?;
                normalized.push_str(":")?;
                if let Some(child) = struct_plan.lookup(name.as_ref()) {
                    let child_path = format!("{path}.{}", name.as_ref());
                    normalize_json_decimal_value(normalized, value, child, &child_path, record)?;
                } else {
                    normalized.push_str(value.get())?;
                }
            }
            normalized.push_str("}")
        }
    }
}

fn format_decimal_unscaled(unscaled: i128, scale: i8) -> String {
    let negative = unscaled < 0;
    let mut digits = unscaled.unsigned_abs().to_string();
    if scale > 0 {
        let scale = scale as usize;
        if digits.len() <= scale {
            digits.insert_str(0, &"0".repeat(scale + 1 - digits.len()));
        }
        digits.insert(digits.len() - scale, '.');
    } else if scale < 0 {
        digits.push_str(&"0".repeat(scale.unsigned_abs() as usize));
    }
    if negative {
        digits.insert(0, '-');
    }
    digits
}

fn schema_needs_json_validation(schema: &Schema) -> bool {
    schema
        .fields()
        .iter()
        .any(|field| field_needs_json_validation(field))
}

const AVRO_LOGICAL_TYPE_METADATA: &str = "paimon.mosaic.avro.logical_type";
const AVRO_UUID_LOGICAL_TYPE: &str = "uuid";

fn field_needs_json_validation(field: &Field) -> bool {
    field_is_avro_uuid(field) || data_type_needs_json_validation(field.data_type())
}

fn field_is_avro_uuid(field: &Field) -> bool {
    field
        .metadata()
        .get(AVRO_LOGICAL_TYPE_METADATA)
        .is_some_and(|value| value == AVRO_UUID_LOGICAL_TYPE)
}

fn data_type_needs_json_validation(data_type: &DataType) -> bool {
    match data_type {
        DataType::Int32
        | DataType::Int64
        | DataType::Date32
        | DataType::Time32(_)
        | DataType::Timestamp(_, _)
        | DataType::Decimal128(_, _) => true,
        DataType::List(field) => field_needs_json_validation(field),
        // Every map must be walked so duplicate keys are rejected even when
        // its values do not otherwise need special Avro validation.
        DataType::Map(_, _) => true,
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| field_needs_json_validation(field)),
        _ => false,
    }
}

fn validate_json_special_values(
    raw: &[u8],
    fields: &std::collections::HashMap<String, Arc<Field>>,
    first_record: usize,
) -> std::io::Result<()> {
    // Borrow only the raw values of relevant fields. Unrelated values are
    // skipped without constructing a second set of Arrow arrays or a Value tree.
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    JsonSpecialRecordSeed {
        fields,
        record: first_record,
    }
    .deserialize(&mut deserializer)
    .map_err(|e| invalid_schema(format!("invalid JSON record {first_record}: {e}")))
}

fn json_special_fields(schema: &Schema) -> std::collections::HashMap<String, Arc<Field>> {
    schema
        .fields()
        .iter()
        .filter(|field| field_needs_json_validation(field))
        .map(|field| (field.name().clone(), Arc::clone(field)))
        .collect()
}

struct JsonSpecialRecordSeed<'a> {
    fields: &'a std::collections::HashMap<String, Arc<Field>>,
    record: usize,
}

impl<'de> DeserializeSeed<'de> for JsonSpecialRecordSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(JsonSpecialRecordVisitor {
            fields: self.fields,
            record: self.record,
        })
    }
}

struct JsonSpecialRecordVisitor<'a> {
    fields: &'a std::collections::HashMap<String, Arc<Field>>,
    record: usize,
}

impl<'de> Visitor<'de> for JsonSpecialRecordVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON object")
    }

    fn visit_map<M>(self, mut map: M) -> Result<(), M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut values = std::collections::BTreeMap::new();
        while let Some(name) = map.next_key::<std::borrow::Cow<'de, str>>()? {
            if self.fields.contains_key(name.as_ref()) {
                let raw: &'de RawValue = map.next_value()?;
                values.insert(name, raw);
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        for (name, raw) in values {
            let field = self
                .fields
                .get(name.as_ref())
                .expect("only schema fields are retained");
            validate_json_special_value(raw, field, name.as_ref(), self.record)
                .map_err(M::Error::custom)?;
        }
        Ok(())
    }
}

struct JsonSpecialMapSeed<'a> {
    value_field: &'a Field,
    path: &'a str,
    record: usize,
}

impl<'de> DeserializeSeed<'de> for JsonSpecialMapSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(JsonSpecialMapVisitor {
            value_field: self.value_field,
            path: self.path,
            record: self.record,
        })
    }
}

struct JsonSpecialMapVisitor<'a> {
    value_field: &'a Field,
    path: &'a str,
    record: usize,
}

impl<'de> Visitor<'de> for JsonSpecialMapVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON map")
    }

    fn visit_map<M>(self, mut map: M) -> Result<(), M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut seen = std::collections::HashSet::new();
        while let Some(key) = map.next_key::<std::borrow::Cow<'de, str>>()? {
            if !seen.insert(key.to_string()) {
                return Err(M::Error::custom(format!(
                    "duplicate JSON map key '{}' in field '{}' at record {}",
                    fmt::safe(key.as_ref()),
                    fmt::safe(self.path),
                    self.record
                )));
            }
            let raw: &RawValue = map.next_value()?;
            validate_json_special_value(raw, self.value_field, self.path, self.record)
                .map_err(M::Error::custom)?;
        }
        Ok(())
    }
}

fn validate_json_special_value(
    raw: &RawValue,
    field: &Field,
    path: &str,
    record: usize,
) -> std::io::Result<()> {
    if raw.get() == "null" || !field_needs_json_validation(field) {
        return Ok(());
    }
    if field_is_avro_uuid(field) {
        let value: String = serde_json::from_str(raw.get()).map_err(|_| {
            invalid_schema(format!(
                "JSON field '{}' at record {record} must be a valid UUID string",
                fmt::safe(path)
            ))
        })?;
        validate_avro_uuid(&value).map_err(|_| {
            invalid_schema(format!(
                "invalid UUID '{}' for JSON field '{}' at record {record}",
                fmt::safe(&value),
                fmt::safe(path)
            ))
        })?;
    }
    let data_type = field.data_type();
    match data_type {
        DataType::Time32(TimeUnit::Millisecond) if !raw.get().starts_with('"') => {
            validate_json_integer(raw, data_type, path, record, 0, i128::from(MAX_TIME_MILLIS))?;
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            let value: String = serde_json::from_str(raw.get()).map_err(|_| {
                invalid_schema(format!(
                    "JSON field '{}' at record {record} must be a valid time-millis value",
                    fmt::safe(path)
                ))
            })?;
            let parsed = Time32MillisecondType::parse(&value).ok_or_else(|| {
                invalid_schema(format!(
                    "JSON field '{}' at record {record} must be a valid time-millis value; got '{}'",
                    fmt::safe(path),
                    fmt::safe(&value)
                ))
            })?;
            if !valid_time_millis(parsed) {
                return Err(invalid_schema(format!(
                    "JSON field '{}' at record {record} is out of range for {data_type}; got '{}'",
                    fmt::safe(path),
                    fmt::safe(&value)
                )));
            }
        }
        DataType::Int32 | DataType::Date32 if !raw.get().starts_with('"') => {
            validate_json_integer(
                raw,
                data_type,
                path,
                record,
                i128::from(i32::MIN),
                i128::from(i32::MAX),
            )?;
        }
        DataType::Int64 | DataType::Timestamp(_, _) if !raw.get().starts_with('"') => {
            validate_json_integer(
                raw,
                data_type,
                path,
                record,
                i128::from(i64::MIN),
                i128::from(i64::MAX),
            )?;
        }
        DataType::Decimal128(precision, scale) => {
            let raw_text = raw.get();
            let value = if raw_text.starts_with('"') {
                std::borrow::Cow::Owned(
                    serde_json::from_str::<String>(raw_text)
                        .map_err(|e| invalid_schema(format!("invalid JSON decimal: {e}")))?,
                )
            } else {
                std::borrow::Cow::Borrowed(raw_text)
            };
            parse_decimal_exact(&value, *precision, *scale).map_err(|e| {
                invalid_schema(format!(
                    "cannot parse '{}' as {data_type} for JSON field '{}' at record {record}: {e}",
                    fmt::safe(&value),
                    fmt::safe(path)
                ))
            })?;
        }
        DataType::List(field) => {
            let values: Vec<&RawValue> = serde_json::from_str(raw.get())
                .map_err(|e| invalid_schema(format!("invalid JSON array: {e}")))?;
            let child_path = format!("{path}[]");
            for value in values {
                validate_json_special_value(value, field, &child_path, record)?;
            }
        }
        DataType::Map(entries, _) => {
            let DataType::Struct(fields) = entries.data_type() else {
                return Ok(());
            };
            let Some(value_field) = fields.get(1) else {
                return Ok(());
            };
            let child_path = format!("{path}{{}}");
            let mut deserializer = serde_json::Deserializer::from_str(raw.get());
            JsonSpecialMapSeed {
                value_field,
                path: &child_path,
                record,
            }
            .deserialize(&mut deserializer)
            .map_err(|e| invalid_schema(format!("invalid JSON map: {e}")))?;
        }
        DataType::Struct(fields) => {
            let values: std::collections::HashMap<String, &RawValue> =
                serde_json::from_str(raw.get())
                    .map_err(|e| invalid_schema(format!("invalid JSON object: {e}")))?;
            for field in fields {
                if let Some(value) = values.get(field.name()) {
                    let child_path = format!("{path}.{}", field.name());
                    validate_json_special_value(value, field, &child_path, record)?;
                }
            }
        }
        DataType::Timestamp(_, None) if raw.get().starts_with('"') => {
            let value: String = serde_json::from_str(raw.get())
                .map_err(|e| invalid_schema(format!("invalid JSON timestamp: {e}")))?;
            if timestamp_has_explicit_timezone(&value) {
                return Err(invalid_schema(format!(
                    "JSON field '{}' at record {record} must not include a timezone for a local timestamp; got '{}'",
                    fmt::safe(path),
                    fmt::safe(&value)
                )));
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_avro_uuid(value: &str) -> Result<(), ()> {
    let bytes = value.as_bytes();
    if bytes.len() != 36 {
        return Err(());
    }
    for (index, byte) in bytes.iter().enumerate() {
        if matches!(index, 8 | 13 | 18 | 23) {
            if *byte != b'-' {
                return Err(());
            }
        } else if !byte.is_ascii_hexdigit() {
            return Err(());
        }
    }
    Ok(())
}

fn validate_json_integer(
    raw: &RawValue,
    data_type: &DataType,
    path: &str,
    record: usize,
    min: i128,
    max: i128,
) -> std::io::Result<()> {
    let value = raw.get();
    let parsed = parse_decimal_exact(value, 38, 0).map_err(|_| {
        invalid_schema(format!(
            "JSON field '{}' at record {record} must be an integer for {data_type}; got '{}'",
            fmt::safe(path),
            fmt::safe(value)
        ))
    })?;
    if parsed < min || parsed > max {
        return Err(invalid_schema(format!(
            "JSON field '{}' at record {record} is out of range for {data_type}; got '{}'",
            fmt::safe(path),
            fmt::safe(value)
        )));
    }
    Ok(())
}

const MAX_TIME_MILLIS: i32 = 86_399_999;

fn valid_time_millis(value: i32) -> bool {
    (0..=MAX_TIME_MILLIS).contains(&value)
}

struct CsvConvertOptions {
    delimiter: String,
    escape: Option<String>,
    quote: String,
    no_header: bool,
    header: Option<String>,
    skip_lines: usize,
}

fn convert_csv(
    inputs: &[PathBuf],
    out: &Path,
    schema: Option<&Path>,
    required_fields: &[String],
    options: CsvConvertOptions,
    stats: Option<&str>,
    overwrite: bool,
) -> std::io::Result<()> {
    if inputs.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "CSV path is required",
        ));
    }
    if schema.is_some() && !required_fields.is_empty() {
        return Err(invalid_schema(
            "--require applies only when the schema is inferred; set nullability in the --schema file instead",
        ));
    }
    ensure_can_write(out, overwrite)?;
    let format = csv_format(&options)?;
    let dialect = CsvDialect::from_options(&options)?;
    let explicit_schema = schema.map(load_convert_schema).transpose()?;
    if let Some(schema) = explicit_schema {
        reject_csv_unsupported_fields(&schema)?;
        let schema_index = csv_schema_index(&schema);
        return write_mosaic(out, overwrite, &schema, stats, |writer, rows| {
            for input in inputs {
                let mut source = OpenCsvSource::open(input, options.skip_lines)?;
                write_explicit_schema_csv_input(
                    writer,
                    rows,
                    &mut source,
                    &schema,
                    &schema_index,
                    &options,
                    dialect,
                )?;
            }
            Ok(())
        });
    }

    let mut sources = inputs
        .iter()
        .map(|input| {
            OpenCsvSource::open(input, options.skip_lines)
                .and_then(|source| prepare_inferred_csv_input(source, &options, dialect))
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    let mut inferred: Option<Schema> = None;
    for source in &mut sources {
        if !source.layout.has_records {
            continue;
        }
        let line_offset = source.line_offset;
        let reader = source.reader()?;
        let (schema, rows) = format
            .infer_schema(reader, None)
            .map_err(|e| csv_data_error(e, line_offset))?;
        // A shard with no data rows has nothing to infer from; it is skipped
        // when reading too.
        if rows == 0 || schema.fields().is_empty() {
            continue;
        }
        let schema =
            promote_second_precision_csv_timestamps(csv_schema_with_csv_names(schema, &options)?);
        inferred = Some(match inferred.take() {
            Some(prev) => merge_csv_inferred_schema(prev, schema, &source.path)?,
            None => schema,
        });
    }
    let schema = inferred
        .ok_or_else(|| invalid_schema("no CSV data to infer a schema from; provide --schema"))?;
    let schema = apply_required_fields(csv_schema_with_null_fallback(schema), required_fields)?;
    reject_csv_unsupported_fields(&schema)?;
    let schema_index = csv_schema_index(&schema);
    write_mosaic(out, overwrite, &schema, stats, |writer, rows| {
        for source in &mut sources {
            let input = source.path.clone();
            let layout = source.layout.clone();
            // Empty and header-only shards contribute no rows, so their header
            // cannot affect the schema inferred from non-empty inputs.
            if !layout.has_records {
                continue;
            }
            let reader_schema = csv_reader_schema(&schema, &schema_index, &layout);
            let source_mapping = csv_output_mapping(&schema, &schema_index, &layout);
            validate_csv_mapping(&schema, &layout, &source_mapping, &input)?;
            let (projection, mapping) = csv_projection(&source_mapping);
            let batch_size =
                inferred_csv_batch_size(reader_schema.fields().len(), source.max_record_bytes);
            let line_offset = source.line_offset;
            let reader = source.reader()?;
            let reader = arrow::csv::ReaderBuilder::new(Arc::new(reader_schema))
                .with_format(format.clone().with_truncated_rows(true))
                .with_batch_size(batch_size)
                .with_projection(projection)
                .build(reader)
                .map_err(|e| csv_data_error(e, line_offset))?;
            for batch in reader {
                let batch = batch.map_err(|e| csv_data_error(e, line_offset))?;
                let batch = align_csv_batch_to_schema(batch, &schema, &mapping, &input)?;
                *rows += batch.num_rows();
                writer.write_batch(&batch)?;
            }
        }
        Ok(())
    })
}

fn write_mosaic<F>(
    out: &Path,
    overwrite: bool,
    schema: &Schema,
    stats: Option<&str>,
    write: F,
) -> std::io::Result<()>
where
    F: FnOnce(
        &mut paimon_mosaic_core::writer::MosaicWriter<paimon_mosaic_core::writer::FileSink>,
        &mut usize,
    ) -> std::io::Result<()>,
{
    use paimon_mosaic_core::writer::{FileSink, MosaicWriter, WriterOptions};
    ensure_can_write(out, overwrite)?;
    let opts = WriterOptions {
        stats_columns: stats.map(parse_comma_list).unwrap_or_default(),
        ..Default::default()
    };
    // Write to a unique sibling temp file and rename on success, so a mid-stream
    // failure never leaves a truncated .mosaic — and a process-unique suffix
    // avoids clobbering an unrelated `out.mosaic.tmp` the user may already have.
    let uniq = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = out.with_extension(format!("mosaic.{}.{uniq}.tmp", std::process::id()));
    let mut rows = 0;
    let res = (|| {
        let sink = FileSink::create(&tmp)?;
        let mut w = MosaicWriter::new(sink, schema, opts)?;
        write(&mut w, &mut rows)?;
        w.close()
    })();
    if let Err(e) = res {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    #[cfg(windows)]
    if out.exists() {
        std::fs::remove_file(out)?;
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

fn project_convert_schema(schema: Schema, columns: &[String]) -> std::io::Result<Schema> {
    if columns.is_empty() {
        return Ok(schema);
    }
    let mut seen = std::collections::HashSet::new();
    let mut fields = Vec::new();
    for name in columns {
        if name.is_empty() {
            return Err(invalid_schema("--column field name cannot be empty"));
        }
        let index = schema
            .index_of(name)
            .map_err(|_| invalid_schema(format!("--column '{name}' not found in schema")))?;
        if seen.insert(index) {
            fields.push(schema.fields()[index].as_ref().clone());
        }
    }
    Ok(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

fn parse_convert_columns(arguments: &[String]) -> std::io::Result<Vec<String>> {
    if arguments.is_empty() {
        return Ok(Vec::new());
    }
    let columns: Vec<String> = arguments
        .iter()
        .flat_map(|argument| parse_comma_list(argument))
        .collect();
    if columns.is_empty() {
        return Err(invalid_schema("--column field name cannot be empty"));
    }
    Ok(columns)
}

fn infer_projected_json_schema<R: std::io::Read>(
    reader: R,
    columns: &[String],
) -> Result<Schema, arrow::error::ArrowError> {
    use arrow::error::ArrowError;

    let values = serde_json::Deserializer::from_reader(reader)
        .into_iter::<Value>()
        .map(|value| {
            let value = value.map_err(|e| ArrowError::JsonError(e.to_string()))?;
            Ok(match value {
                Value::Object(mut object) => {
                    let projected = columns
                        .iter()
                        .filter_map(|name| object.remove(name).map(|value| (name.clone(), value)))
                        .collect();
                    Value::Object(projected)
                }
                value => value,
            })
        });
    arrow::json::reader::infer_json_schema_from_iterator(values)
}

/// Mosaic cannot store Arrow `Null` columns, and JSON inference produces
/// `Null` for a field with no non-null value in the input — fail
/// with the column name instead of the writer's late "unsupported DataType".
fn reject_null_inferred_fields(schema: &Schema) -> std::io::Result<()> {
    for field in schema.fields() {
        if matches!(field.data_type(), DataType::Null) {
            return Err(invalid_schema(format!(
                "cannot infer a type for column '{}' (no non-null value in the records); provide --schema",
                fmt::safe(field.name())
            )));
        }
    }
    Ok(())
}

fn is_json_input(input: &Path) -> bool {
    input
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "json" | "ndjson" | "jsonl"
            )
        })
}

fn ensure_can_write(out: &Path, overwrite: bool) -> std::io::Result<()> {
    if out.exists() && !overwrite {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("{} exists (use --overwrite to replace)", out.display()),
        ));
    }
    Ok(())
}

fn csv_format(options: &CsvConvertOptions) -> std::io::Result<arrow::csv::reader::Format> {
    let dialect = CsvDialect::from_options(options)?;
    let format = arrow::csv::reader::Format::default()
        .with_header(!options.no_header && options.header.is_none())
        .with_delimiter(dialect.delimiter)
        .with_quote(dialect.quote);
    Ok(match dialect.escape {
        Some(escape) => format.with_escape(escape),
        None => format,
    })
}

#[derive(Clone, Copy)]
struct CsvDialect {
    delimiter: u8,
    escape: Option<u8>,
    quote: u8,
}

impl CsvDialect {
    fn from_options(options: &CsvConvertOptions) -> std::io::Result<Self> {
        Ok(Self {
            delimiter: parse_csv_byte(&options.delimiter, "delimiter")?,
            escape: parse_optional_csv_byte(options.escape.as_deref(), "escape")?,
            quote: parse_csv_byte(&options.quote, "quote")?,
        })
    }
}

fn parse_csv_byte(value: &str, name: &str) -> std::io::Result<u8> {
    let bytes = value.as_bytes();
    if bytes.len() == 1 {
        Ok(bytes[0])
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("--{name} must be exactly one byte"),
        ))
    }
}

fn parse_optional_csv_byte(value: Option<&str>, name: &str) -> std::io::Result<Option<u8>> {
    value.map(|value| parse_csv_byte(value, name)).transpose()
}

fn csv_data_error(error: arrow::error::ArrowError, line_offset: u64) -> std::io::Error {
    // Arrow ParseError line numbers are zero-based record indexes, whereas
    // CsvError line numbers are one-based and already include the file header.
    let parse_error = matches!(&error, arrow::error::ArrowError::ParseError(_));
    let line_offset = line_offset.saturating_add(parse_error as u64);
    let message = error.to_string();
    // ParseError appends the complete row. Do not rewrite line-shaped text
    // inside user data such as "note at line 99".
    let search_end = if parse_error {
        message.find(". Row data:").unwrap_or(message.len())
    } else {
        message.len()
    };
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        csv_error_message_with_line_offset(message, line_offset, search_end),
    )
}

fn csv_error_with_line_offset(error: impl ToString, line_offset: u64) -> String {
    let message = error.to_string();
    let search_end = message.len();
    csv_error_message_with_line_offset(message, line_offset, search_end)
}

fn csv_read_error(error: csv::Error, context: &str, line_offset: u64) -> std::io::Error {
    if error.is_io_error() {
        if let csv::ErrorKind::Io(error) = error.into_kind() {
            return error;
        }
        unreachable!("csv::Error::is_io_error must imply ErrorKind::Io");
    }
    invalid_schema(format!(
        "{context}: {}",
        csv_error_with_line_offset(error, line_offset)
    ))
}

fn csv_error_message_with_line_offset(
    message: String,
    line_offset: u64,
    search_end: usize,
) -> String {
    if line_offset == 0 {
        return message;
    }
    for marker in [" at line ", "(line: ", "(line ", "for line "] {
        let Some(index) = message[..search_end].rfind(marker) else {
            continue;
        };
        let value_start = index + marker.len();
        let value_len = message[value_start..]
            .bytes()
            .take_while(u8::is_ascii_digit)
            .count();
        if value_len == 0 {
            continue;
        }
        let Ok(line) = message[value_start..value_start + value_len].parse::<u64>() else {
            continue;
        };
        return format!(
            "{}{}{}",
            &message[..value_start],
            line.saturating_add(line_offset),
            &message[value_start + value_len..]
        );
    }
    message
}

fn add_csv_record_line_offset(record: &mut csv::StringRecord, line_offset: u64) {
    let Some(mut position) = record.position().cloned() else {
        return;
    };
    position.set_line(position.line().saturating_add(line_offset));
    record.set_position(Some(position));
}

#[derive(Clone)]
struct CsvInputLayout {
    header: Option<Vec<String>>,
    columns: usize,
    has_records: bool,
}

struct CsvInput<R> {
    reader: csv::Reader<R>,
    layout: CsvInputLayout,
    first_record: Option<csv::StringRecord>,
}

struct OpenCsvSource {
    path: PathBuf,
    reader: Option<std::io::BufReader<std::fs::File>>,
    line_offset: u64,
}

impl OpenCsvSource {
    fn open(path: &Path, skip_lines: usize) -> std::io::Result<Self> {
        use std::io::BufRead;

        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        let mut skipped = 0_u64;
        for _ in 0..skip_lines {
            if reader.skip_until(b'\n')? == 0 {
                break;
            }
            skipped = skipped.saturating_add(1);
        }
        Ok(Self {
            path: path.to_path_buf(),
            reader: Some(reader),
            line_offset: skipped,
        })
    }

    fn guarded_reader(
        &mut self,
        dialect: CsvDialect,
    ) -> std::io::Result<CsvRecordLimitReader<std::io::BufReader<std::fs::File>>> {
        let reader = self
            .reader
            .take()
            .ok_or_else(|| invalid_schema("CSV source was already consumed"))?;
        Ok(CsvRecordLimitReader::new(
            reader,
            self.path.clone(),
            dialect,
            MAX_CSV_COLUMNS,
            MAX_CSV_RECORD_BYTES,
        ))
    }
}

struct PreparedCsvInput {
    path: PathBuf,
    snapshot: tempfile::TempPath,
    layout: CsvInputLayout,
    line_offset: u64,
    max_record_bytes: usize,
}

impl PreparedCsvInput {
    fn reader(&self) -> std::io::Result<std::io::BufReader<std::fs::File>> {
        Ok(std::io::BufReader::new(std::fs::File::open(
            self.snapshot.as_ref() as &Path,
        )?))
    }
}

fn prepare_inferred_csv_input(
    mut source: OpenCsvSource,
    options: &CsvConvertOptions,
    dialect: CsvDialect,
) -> std::io::Result<PreparedCsvInput> {
    let mut snapshot = tempfile::NamedTempFile::new()?;
    let mut reader = source.guarded_reader(dialect)?;
    std::io::copy(&mut reader, snapshot.as_file_mut())?;
    let max_record_bytes = reader.max_observed_record_bytes();
    let layout = {
        let reader = std::io::BufReader::new(snapshot.reopen()?);
        let CsvInput { layout, .. } =
            open_csv_input_from_reader(&source.path, options, reader, source.line_offset)?;
        layout
    };
    Ok(PreparedCsvInput {
        path: source.path,
        snapshot: snapshot.into_temp_path(),
        layout,
        line_offset: source.line_offset,
        max_record_bytes,
    })
}

const DEFAULT_CSV_BATCH_SIZE: usize = 1024;
// Arrow's CSV RecordDecoder reserves roughly one data byte range and one
// offset per cell before projection is applied. Keep that eager allocation
// bounded for very wide records by reducing the number of rows per batch.
const TARGET_CSV_DECODE_CELLS: usize = 64 * 1024;
// Hard ceiling on the inferred column count. The width comes from the CSV
// header or first no-header logical record, which is attacker-controllable;
// without this cap inference could allocate millions of Arrow Fields.
const MAX_CSV_COLUMNS: usize = 65_535;
// A single decoded CSV record is otherwise allowed to exceed the batch byte
// budget and can allocate several times its payload before it is flushed.
const MAX_CSV_RECORD_BYTES: usize = 64 * 1024 * 1024;
const CSV_LIMIT_SCAN_BUFFER_BYTES: usize = 8 * 1024;

struct CsvRecordLimitReader<R> {
    inner: R,
    scanner: CsvRecordLimitScanner,
}

impl<R> CsvRecordLimitReader<R> {
    fn new(
        inner: R,
        path: PathBuf,
        dialect: CsvDialect,
        max_columns: usize,
        max_record_bytes: usize,
    ) -> Self {
        Self {
            inner,
            scanner: CsvRecordLimitScanner::new(path, dialect, max_columns, max_record_bytes),
        }
    }

    fn max_observed_record_bytes(&self) -> usize {
        self.scanner.max_observed_record_bytes
    }
}

impl<R: std::io::Read> std::io::Read for CsvRecordLimitReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let read = if buffer.is_empty() {
            self.inner.read(buffer)?
        } else {
            let capacity = buffer.len().min(CSV_LIMIT_SCAN_BUFFER_BYTES);
            self.inner.read(&mut buffer[..capacity])?
        };
        if read == 0 {
            self.scanner.finish()?;
        } else {
            self.scanner.scan(&buffer[..read])?;
        }
        Ok(read)
    }
}

struct CsvRecordLimitScanner {
    parser: csv_core::Reader,
    path: PathBuf,
    fields: usize,
    decoded_bytes: usize,
    max_observed_record_bytes: usize,
    record: usize,
    max_columns: usize,
    max_record_bytes: usize,
    scratch: Vec<u8>,
    finished: bool,
}

impl CsvRecordLimitScanner {
    fn new(
        path: PathBuf,
        dialect: CsvDialect,
        max_columns: usize,
        max_record_bytes: usize,
    ) -> Self {
        let mut builder = csv_core::ReaderBuilder::new();
        builder
            .delimiter(dialect.delimiter)
            .quote(dialect.quote)
            .escape(dialect.escape);
        Self {
            parser: builder.build(),
            path,
            fields: 0,
            decoded_bytes: 0,
            max_observed_record_bytes: 0,
            record: 1,
            max_columns,
            max_record_bytes,
            scratch: vec![0; CSV_LIMIT_SCAN_BUFFER_BYTES],
            finished: false,
        }
    }

    fn scan(&mut self, input: &[u8]) -> std::io::Result<()> {
        let mut offset = 0;
        while offset < input.len() {
            let (result, consumed, decoded) =
                self.parser.read_field(&input[offset..], &mut self.scratch);
            self.add_decoded_bytes(decoded)?;
            offset = offset.saturating_add(consumed);
            match result {
                csv_core::ReadFieldResult::InputEmpty => break,
                csv_core::ReadFieldResult::OutputFull => {
                    debug_assert!(consumed > 0 || decoded > 0);
                }
                csv_core::ReadFieldResult::Field { record_end } => {
                    self.finish_field(record_end)?;
                }
                csv_core::ReadFieldResult::End => {
                    self.finished = true;
                    break;
                }
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> std::io::Result<()> {
        if self.finished {
            return Ok(());
        }
        loop {
            let (result, consumed, decoded) = self.parser.read_field(&[], &mut self.scratch);
            debug_assert_eq!(consumed, 0);
            self.add_decoded_bytes(decoded)?;
            match result {
                csv_core::ReadFieldResult::Field { record_end } => {
                    self.finish_field(record_end)?;
                }
                csv_core::ReadFieldResult::End | csv_core::ReadFieldResult::InputEmpty => {
                    self.finished = true;
                    return Ok(());
                }
                csv_core::ReadFieldResult::OutputFull => {
                    debug_assert!(decoded > 0);
                }
            }
        }
    }

    fn add_decoded_bytes(&mut self, bytes: usize) -> std::io::Result<()> {
        self.decoded_bytes = self.decoded_bytes.saturating_add(bytes);
        if self.decoded_bytes > self.max_record_bytes {
            return Err(invalid_schema(format!(
                "CSV record {} in {} exceeds the {} decoded byte limit",
                self.record,
                self.path.display(),
                self.max_record_bytes
            )));
        }
        Ok(())
    }

    fn finish_field(&mut self, record_end: bool) -> std::io::Result<()> {
        self.fields = self.fields.saturating_add(1);
        if self.fields > self.max_columns || (!record_end && self.fields == self.max_columns) {
            let columns = self.fields.saturating_add((!record_end) as usize);
            return Err(invalid_schema(format!(
                "CSV input {} has at least {} columns, exceeds the {} column limit",
                self.path.display(),
                columns,
                self.max_columns
            )));
        }
        if record_end {
            self.max_observed_record_bytes = self.max_observed_record_bytes.max(self.decoded_bytes);
            self.fields = 0;
            self.decoded_bytes = 0;
            self.record = self.record.saturating_add(1);
        }
        Ok(())
    }
}

fn ensure_csv_column_limit(path: &Path, columns: usize) -> std::io::Result<()> {
    if columns > MAX_CSV_COLUMNS {
        return Err(invalid_schema(format!(
            "CSV input {} has {} columns, exceeds the {} column limit",
            path.display(),
            columns,
            MAX_CSV_COLUMNS
        )));
    }
    Ok(())
}

fn csv_batch_size(columns: usize) -> usize {
    if columns == 0 {
        return DEFAULT_CSV_BATCH_SIZE;
    }
    (TARGET_CSV_DECODE_CELLS / columns).clamp(1, DEFAULT_CSV_BATCH_SIZE)
}

fn inferred_csv_batch_size(columns: usize, max_record_bytes: usize) -> usize {
    let cell_bound = csv_batch_size(columns);
    if max_record_bytes == 0 {
        return cell_bound;
    }
    cell_bound.min((TARGET_CONVERT_BATCH_BYTES / max_record_bytes).max(1))
}

fn explicit_csv_row_cells(source_columns: usize, output_columns: usize) -> usize {
    source_columns.max(output_columns)
}

fn open_csv_input_from_reader<R: std::io::Read>(
    path: &Path,
    options: &CsvConvertOptions,
    reader: R,
    line_offset: u64,
) -> std::io::Result<CsvInput<R>> {
    let dialect = CsvDialect::from_options(options)?;
    let mut builder = csv::ReaderBuilder::new();
    builder
        .has_headers(false)
        .flexible(true)
        .delimiter(dialect.delimiter)
        .quote(dialect.quote)
        .escape(dialect.escape);
    let mut reader = builder.from_reader(reader);
    let file_header = options.header.is_none() && !options.no_header;
    let header = if let Some(header) = &options.header {
        Some(parse_csv_header(header, options)?)
    } else if options.no_header {
        None
    } else {
        let mut record = csv::StringRecord::new();
        if reader
            .read_record(&mut record)
            .map_err(|e| csv_read_error(e, "invalid CSV header", line_offset))?
        {
            add_csv_record_line_offset(&mut record, line_offset);
            Some(record.iter().map(ToString::to_string).collect())
        } else {
            Some(Vec::new())
        }
    };
    if let Some(header) = &header {
        ensure_csv_column_limit(path, header.len())?;
    }
    let mut first_record = csv::StringRecord::new();
    let has_records = reader
        .read_record(&mut first_record)
        .map_err(|e| csv_read_error(e, "invalid CSV record", line_offset))?;
    if has_records {
        add_csv_record_line_offset(&mut first_record, line_offset);
        ensure_csv_column_limit(path, first_record.len())?;
    }
    if has_records && file_header {
        validate_csv_header_names(header.as_ref().unwrap())?;
    }
    let columns = header.as_ref().map_or_else(
        || {
            if has_records {
                first_record.len()
            } else {
                0
            }
        },
        Vec::len,
    );
    Ok(CsvInput {
        reader,
        layout: CsvInputLayout {
            header,
            columns,
            has_records,
        },
        first_record: has_records.then_some(first_record),
    })
}

fn write_explicit_schema_csv_input(
    writer: &mut paimon_mosaic_core::writer::MosaicWriter<paimon_mosaic_core::writer::FileSink>,
    rows: &mut usize,
    source: &mut OpenCsvSource,
    schema: &Schema,
    schema_index: &std::collections::HashMap<&str, usize>,
    options: &CsvConvertOptions,
    dialect: CsvDialect,
) -> std::io::Result<()> {
    let input = source.path.clone();
    let line_offset = source.line_offset;
    let reader = source.guarded_reader(dialect)?;
    let mut input_reader = open_csv_input_from_reader(&input, options, reader, line_offset)?;
    if !input_reader.layout.has_records {
        return Ok(());
    }
    let source_mapping = csv_output_mapping(schema, schema_index, &input_reader.layout);
    validate_csv_mapping(schema, &input_reader.layout, &source_mapping, &input)?;

    let first = input_reader.first_record.take().into_iter().map(Ok);
    let rest = std::iter::from_fn(|| {
        let mut record = csv::StringRecord::new();
        match input_reader.reader.read_record(&mut record) {
            Ok(true) => {
                add_csv_record_line_offset(&mut record, line_offset);
                Some(ensure_csv_column_limit(&input, record.len()).map(|_| record))
            }
            Ok(false) => None,
            Err(e) => Some(Err(csv_read_error(
                e,
                &format!("invalid CSV record in {}", input.display()),
                line_offset,
            ))),
        }
    });
    for_each_explicit_csv_batch(
        first.chain(rest),
        input_reader.layout.columns,
        schema,
        TARGET_CONVERT_BATCH_BYTES,
        |records| write_explicit_csv_records(writer, rows, schema, &source_mapping, records),
    )
}

fn for_each_explicit_csv_batch<I, F>(
    records: I,
    source_columns: usize,
    schema: &Schema,
    byte_budget: usize,
    mut write: F,
) -> std::io::Result<()>
where
    I: IntoIterator<Item = std::io::Result<csv::StringRecord>>,
    F: FnMut(&[csv::StringRecord]) -> std::io::Result<()>,
{
    let output_columns = schema.fields().len();
    let mut batch = Vec::with_capacity(csv_batch_size(source_columns.max(output_columns)));
    let mut cells: usize = 0;
    let mut bytes: usize = 0;
    for record in records {
        let record = record?;
        let row_cells = explicit_csv_row_cells(record.len(), output_columns);
        let row_bytes = record.as_slice().len();
        if !batch.is_empty()
            && (batch.len() >= DEFAULT_CSV_BATCH_SIZE
                || cells.saturating_add(row_cells) > TARGET_CSV_DECODE_CELLS
                || bytes.saturating_add(row_bytes) > byte_budget)
        {
            write(&batch)?;
            batch.clear();
            cells = 0;
            bytes = 0;
        }
        cells = cells.saturating_add(row_cells);
        bytes = bytes.saturating_add(row_bytes);
        batch.push(record);
        if row_bytes > byte_budget {
            write(&batch)?;
            batch.clear();
            cells = 0;
            bytes = 0;
        }
    }
    if !batch.is_empty() {
        write(&batch)?;
    }
    Ok(())
}

fn write_explicit_csv_records(
    writer: &mut paimon_mosaic_core::writer::MosaicWriter<paimon_mosaic_core::writer::FileSink>,
    rows: &mut usize,
    schema: &Schema,
    mapping: &[Option<usize>],
    records: &[csv::StringRecord],
) -> std::io::Result<()> {
    let batch = csv_records_to_batch(schema, mapping, records)?;
    *rows += batch.num_rows();
    writer.write_batch(&batch)
}

fn csv_records_to_batch(
    schema: &Schema,
    mapping: &[Option<usize>],
    records: &[csv::StringRecord],
) -> std::io::Result<RecordBatch> {
    let columns = schema
        .fields()
        .iter()
        .zip(mapping)
        .map(|(field, source)| match source {
            Some(source) => csv_column_array(records, *source, field),
            None => Ok(new_null_array(field.data_type(), records.len())),
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    RecordBatch::try_new(Arc::new(schema.clone()), columns)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
}

fn csv_column_array(
    records: &[csv::StringRecord],
    source: usize,
    field: &Field,
) -> std::io::Result<ArrayRef> {
    match field.data_type() {
        DataType::Boolean => {
            let values = records
                .iter()
                .map(|record| {
                    let Some(value) = csv_record_value(record, source) else {
                        return Ok(None);
                    };
                    if value.eq_ignore_ascii_case("true") {
                        Ok(Some(true))
                    } else if value.eq_ignore_ascii_case("false") {
                        Ok(Some(false))
                    } else {
                        Err(csv_value_parse_error(record, field, value))
                    }
                })
                .collect::<std::io::Result<Vec<_>>>()?;
            Ok(Arc::new(BooleanArray::from(values)))
        }
        DataType::Int32 => csv_primitive_column::<Int32Type>(records, source, field),
        DataType::Int64 => csv_primitive_column::<Int64Type>(records, source, field),
        DataType::Float32 => csv_primitive_column::<Float32Type>(records, source, field),
        DataType::Float64 => csv_primitive_column::<Float64Type>(records, source, field),
        DataType::Date32 => csv_primitive_column::<Date32Type>(records, source, field),
        DataType::Time32(TimeUnit::Millisecond) => csv_time_millis_column(records, source, field),
        DataType::Timestamp(unit, timezone) => {
            csv_timestamp_column(records, source, field, unit, timezone.clone())
        }
        DataType::Decimal128(precision, scale) => {
            let values = records
                .iter()
                .map(|record| {
                    csv_record_value(record, source)
                        .map(|value| match parse_decimal_unscaled_exact(
                            value, *precision, *scale,
                        ) {
                            Ok(parsed) => Ok(Some(parsed)),
                            Err(DecimalParseFailure::Inexact) => {
                                Err(invalid_schema(format!(
                                    "decimal value '{}' for CSV field '{}' at line {} cannot be represented exactly with scale {scale}",
                                    fmt::safe(value),
                                    fmt::safe(field.name()),
                                    record
                                        .position()
                                        .map(|position| position.line().to_string())
                                        .unwrap_or_else(|| "unknown".to_string())
                                )))
                            }
                            Err(_) => Err(csv_value_parse_error(record, field, value)),
                        })
                        .unwrap_or(Ok(None))
                })
                .collect::<std::io::Result<Vec<_>>>()?;
            let array: PrimitiveArray<Decimal128Type> = values.into_iter().collect();
            Ok(Arc::new(
                array
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|e| invalid_schema(e.to_string()))?,
            ))
        }
        DataType::Utf8 => {
            let values = records
                .iter()
                .map(|record| {
                    let value = csv_record_value(record, source);
                    if let Some(value) = value {
                        if field_is_avro_uuid(field) && validate_avro_uuid(value).is_err() {
                            return Err(invalid_schema(format!(
                                "invalid UUID '{}' for CSV field '{}' at line {}",
                                fmt::safe(value),
                                fmt::safe(field.name()),
                                record
                                    .position()
                                    .map(|position| position.line().to_string())
                                    .unwrap_or_else(|| "unknown".to_string())
                            )));
                        }
                    }
                    Ok(value)
                })
                .collect::<std::io::Result<StringArray>>()?;
            Ok(Arc::new(values))
        }
        data_type => Err(invalid_schema(format!(
            "CSV conversion does not support field '{}' with type {data_type}",
            fmt::safe(field.name())
        ))),
    }
}

fn csv_time_millis_column(
    records: &[csv::StringRecord],
    source: usize,
    field: &Field,
) -> std::io::Result<ArrayRef> {
    let values = records
        .iter()
        .map(|record| {
            let Some(value) = csv_record_value(record, source) else {
                return Ok(None);
            };
            let parsed = Time32MillisecondType::parse(value)
                .ok_or_else(|| csv_value_parse_error(record, field, value))?;
            if !valid_time_millis(parsed) {
                return Err(invalid_schema(format!(
                    "time-millis value '{}' for CSV field '{}' at line {} must be between 0 and {MAX_TIME_MILLIS}",
                    fmt::safe(value),
                    fmt::safe(field.name()),
                    record
                        .position()
                        .map(|position| position.line().to_string())
                        .unwrap_or_else(|| "unknown".to_string())
                )));
            }
            Ok(Some(parsed))
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    Ok(Arc::new(
        values
            .into_iter()
            .collect::<PrimitiveArray<Time32MillisecondType>>(),
    ))
}

fn csv_primitive_column<T>(
    records: &[csv::StringRecord],
    source: usize,
    field: &Field,
) -> std::io::Result<ArrayRef>
where
    T: ArrowPrimitiveType + ArrowValueParser,
{
    Ok(Arc::new(csv_primitive_array::<T>(records, source, field)?))
}

fn csv_primitive_array<T>(
    records: &[csv::StringRecord],
    source: usize,
    field: &Field,
) -> std::io::Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType + ArrowValueParser,
{
    let values = records
        .iter()
        .map(|record| {
            csv_record_value(record, source)
                .map(|value| {
                    T::parse(value)
                        .map(Some)
                        .ok_or_else(|| csv_value_parse_error(record, field, value))
                })
                .unwrap_or(Ok(None))
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    Ok(values.into_iter().collect())
}

fn csv_timestamp_column(
    records: &[csv::StringRecord],
    source: usize,
    field: &Field,
    unit: &TimeUnit,
    timezone: Option<Arc<str>>,
) -> std::io::Result<ArrayRef> {
    let parser_timezone: Tz = timezone
        .as_deref()
        .unwrap_or("+00:00")
        .parse()
        .map_err(|e| invalid_schema(format!("invalid timestamp timezone: {e}")))?;
    let timezone_policy = timezone.is_none().then_some("a local timestamp");
    let values = records
        .iter()
        .map(|record| {
            let Some(value) = csv_record_value(record, source) else {
                return Ok(None);
            };
            let line = record
                .position()
                .map(|position| position.line().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            parse_csv_timestamp_value(
                value,
                field,
                unit,
                &parser_timezone,
                timezone_policy,
                &format!("at line {line}"),
            )
            .map(Some)
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    csv_timestamp_array(values, unit, timezone)
}

fn parse_csv_timestamp_value(
    value: &str,
    field: &Field,
    unit: &TimeUnit,
    parser_timezone: &Tz,
    timezone_policy: Option<&str>,
    location: &str,
) -> std::io::Result<i64> {
    match timezone_policy {
        Some(policy) if timestamp_has_explicit_timezone(value) => {
            return Err(invalid_schema(format!(
                "CSV field '{}' {location} must not include a timezone for {policy}",
                fmt::safe(field.name())
            )));
        }
        _ => {}
    }
    let parse_error = || {
        invalid_schema(format!(
            "cannot parse '{}' as {} for CSV field '{}' {location}",
            fmt::safe(value),
            field.data_type(),
            fmt::safe(field.name())
        ))
    };
    let datetime = string_to_datetime(parser_timezone, value).map_err(|_| parse_error())?;
    match unit {
        TimeUnit::Millisecond => Ok(datetime.timestamp_millis()),
        TimeUnit::Microsecond => Ok(datetime.timestamp_micros()),
        TimeUnit::Nanosecond => datetime.timestamp_nanos_opt().ok_or_else(parse_error),
        unit => Err(invalid_schema(format!(
            "CSV conversion does not support timestamp unit {unit:?}"
        ))),
    }
}

fn csv_timestamp_array(
    values: Vec<Option<i64>>,
    unit: &TimeUnit,
    timezone: Option<Arc<str>>,
) -> std::io::Result<ArrayRef> {
    Ok(match unit {
        TimeUnit::Millisecond => Arc::new(
            PrimitiveArray::<TimestampMillisecondType>::from(values)
                .with_timezone_opt(timezone.clone()),
        ),
        TimeUnit::Microsecond => Arc::new(
            PrimitiveArray::<TimestampMicrosecondType>::from(values)
                .with_timezone_opt(timezone.clone()),
        ),
        TimeUnit::Nanosecond => Arc::new(
            PrimitiveArray::<TimestampNanosecondType>::from(values).with_timezone_opt(timezone),
        ),
        unit => {
            return Err(invalid_schema(format!(
                "CSV conversion does not support timestamp unit {unit:?}"
            )));
        }
    })
}

fn parse_decimal_exact(
    value: &str,
    precision: u8,
    scale: i8,
) -> Result<i128, arrow::error::ArrowError> {
    parse_decimal_unscaled_exact(value, precision, scale).map_err(|failure| {
        let message = match failure {
            DecimalParseFailure::Invalid => {
                format!("can't parse the string value {value} to decimal")
            }
            DecimalParseFailure::Inexact => {
                format!("cannot be represented exactly with scale {scale}")
            }
            DecimalParseFailure::Overflow => format!("parse decimal overflow ({value})"),
        };
        arrow::error::ArrowError::ParseError(message)
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecimalParseFailure {
    Invalid,
    Inexact,
    Overflow,
}

fn parse_decimal_unscaled_exact(
    value: &str,
    precision: u8,
    scale: i8,
) -> Result<i128, DecimalParseFailure> {
    if precision == 0 || precision > 38 {
        return Err(DecimalParseFailure::Overflow);
    }
    let (negative, unsigned) = match value.as_bytes().first() {
        Some(b'-') => (true, &value[1..]),
        Some(b'+') => (false, &value[1..]),
        _ => (false, value),
    };
    let exponent_index = unsigned.find(['e', 'E']);
    let (mantissa, exponent) = match exponent_index {
        Some(index) => {
            let exponent = unsigned[index + 1..]
                .parse::<i64>()
                .map_err(|_| DecimalParseFailure::Invalid)?;
            (&unsigned[..index], exponent)
        }
        None => (unsigned, 0),
    };

    let mut seen_decimal_point = false;
    let mut has_digit = false;
    let mut total_digits = 0_usize;
    let mut fractional_digits = 0_usize;
    let mut first_nonzero = None;
    for byte in mantissa.bytes() {
        match byte {
            b'0'..=b'9' => {
                has_digit = true;
                if seen_decimal_point {
                    fractional_digits = fractional_digits
                        .checked_add(1)
                        .ok_or(DecimalParseFailure::Overflow)?;
                }
                if byte != b'0' && first_nonzero.is_none() {
                    first_nonzero = Some(total_digits);
                }
                total_digits = total_digits
                    .checked_add(1)
                    .ok_or(DecimalParseFailure::Overflow)?;
            }
            b'.' if !seen_decimal_point => seen_decimal_point = true,
            _ => return Err(DecimalParseFailure::Invalid),
        }
    }
    if !has_digit {
        return Err(DecimalParseFailure::Invalid);
    }
    let Some(first_nonzero) = first_nonzero else {
        return Ok(0);
    };

    let shift = i128::from(exponent) - fractional_digits as i128 + i128::from(scale);
    let (kept_digits, appended_zeros) = if shift >= 0 {
        let appended_zeros = usize::try_from(shift).map_err(|_| DecimalParseFailure::Overflow)?;
        (total_digits, appended_zeros)
    } else {
        let discarded =
            usize::try_from(shift.unsigned_abs()).map_err(|_| DecimalParseFailure::Inexact)?;
        if discarded > total_digits {
            return Err(DecimalParseFailure::Inexact);
        }
        let kept_digits = total_digits - discarded;
        for (digit_index, byte) in mantissa.bytes().filter(u8::is_ascii_digit).enumerate() {
            if digit_index >= kept_digits && byte != b'0' {
                return Err(DecimalParseFailure::Inexact);
            }
        }
        (kept_digits, 0)
    };

    let significant_digits = kept_digits
        .saturating_sub(first_nonzero)
        .checked_add(appended_zeros)
        .ok_or(DecimalParseFailure::Overflow)?;
    if significant_digits > usize::from(precision) {
        return Err(DecimalParseFailure::Overflow);
    }

    let mut result = 0_i128;
    for (digit_index, byte) in mantissa.bytes().filter(u8::is_ascii_digit).enumerate() {
        if (first_nonzero..kept_digits).contains(&digit_index) {
            result = result
                .checked_mul(10)
                .and_then(|value| value.checked_add(i128::from(byte - b'0')))
                .ok_or(DecimalParseFailure::Overflow)?;
        }
    }
    for _ in 0..appended_zeros {
        result = result
            .checked_mul(10)
            .ok_or(DecimalParseFailure::Overflow)?;
    }
    if negative {
        result.checked_neg().ok_or(DecimalParseFailure::Overflow)
    } else {
        Ok(result)
    }
}

fn timestamp_has_explicit_timezone(value: &str) -> bool {
    // Arrow's timestamp grammar is a fixed-width YYYY-MM-DD date, a `T`/space
    // separator, a `HH:MM[:SS[.fraction]]` time, then an optional zone. The
    // separator and time-of-day only contain digits, `:`, and `.`, so after the
    // 10-byte date prefix a trailing `Z`/`z` or any `+`/`-` marks an explicit
    // zone regardless of whether seconds or fractional digits are present. The
    // parser rejects expanded-year forms, so the fixed date width holds; byte
    // scanning keeps malformed multi-byte input from panicking on a slice.
    let bytes = value.trim().as_bytes();
    if bytes.len() <= 10 {
        return false;
    }
    let after_date = &bytes[10..];
    matches!(after_date.last(), Some(b'Z' | b'z'))
        || after_date.iter().any(|b| matches!(b, b'+' | b'-'))
}

fn csv_record_value(record: &csv::StringRecord, source: usize) -> Option<&str> {
    record.get(source).filter(|value| !value.is_empty())
}

fn csv_value_parse_error(record: &csv::StringRecord, field: &Field, value: &str) -> std::io::Error {
    let line = record
        .position()
        .map(|position| position.line().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!(
            "cannot parse '{}' as {} for CSV field '{}' at line {line}",
            fmt::safe(value),
            field.data_type(),
            fmt::safe(field.name())
        ),
    )
}

fn csv_schema_index(schema: &Schema) -> std::collections::HashMap<&str, usize> {
    schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| (field.name().as_str(), index))
        .collect()
}

fn csv_reader_schema(
    output_schema: &Schema,
    schema_index: &std::collections::HashMap<&str, usize>,
    layout: &CsvInputLayout,
) -> Schema {
    let positional = layout.header.is_none();
    let columns = if positional {
        layout.columns.max(output_schema.fields().len())
    } else {
        layout.columns
    };
    let fields: Vec<Field> = (0..columns)
        .map(|i| {
            let source = if let Some(header) = &layout.header {
                header
                    .get(i)
                    .and_then(|name| schema_index.get(name.as_str()).copied())
            } else {
                (i < output_schema.fields().len()).then_some(i)
            };
            if let Some(source) = source {
                let output_field = output_schema.fields()[source].as_ref();
                let data_type = match output_field.data_type() {
                    // Arrow's Float64 decoder would round bare integer tokens
                    // before Mosaic can enforce exactness. Preserve raw text
                    // for every inferred Float64 field, including shards that
                    // already infer as Float64 on their own.
                    DataType::Float64 | DataType::Timestamp(_, _) => DataType::Utf8,
                    data_type => data_type.clone(),
                };
                // Read as nullable: not-null enforcement happens when the batch
                // is re-attached to the output schema, where the error carries
                // the real column name rather than a positional one.
                output_field
                    .clone()
                    .with_data_type(data_type)
                    .with_name(format!("field_{i}"))
                    .with_nullable(true)
            } else {
                Field::new(format!("field_{i}"), DataType::Utf8, true)
            }
        })
        .collect();
    Schema::new(fields)
}

fn csv_output_mapping(
    output_schema: &Schema,
    schema_index: &std::collections::HashMap<&str, usize>,
    layout: &CsvInputLayout,
) -> Vec<Option<usize>> {
    if let Some(header) = &layout.header {
        let mut mapping = vec![None; output_schema.fields().len()];
        for (csv_index, name) in header.iter().enumerate() {
            if let Some(field_index) = schema_index.get(name.as_str()).copied() {
                mapping[field_index] = Some(csv_index);
            }
        }
        mapping
    } else {
        (0..output_schema.fields().len()).map(Some).collect()
    }
}

fn csv_projection(mapping: &[Option<usize>]) -> (Vec<usize>, Vec<Option<usize>>) {
    let mut projection = Vec::new();
    let projected_mapping = mapping
        .iter()
        .map(|source| {
            source.map(|source| {
                let projected = projection.len();
                projection.push(source);
                projected
            })
        })
        .collect();
    (projection, projected_mapping)
}

/// A schema field absent from the CSV header becomes an all-null column, so
/// refuse the conversions that can only be mistakes: a header matching no
/// schema field at all, and a required field that the data cannot supply.
fn validate_csv_mapping(
    schema: &Schema,
    layout: &CsvInputLayout,
    mapping: &[Option<usize>],
    input: &Path,
) -> std::io::Result<()> {
    if layout.header.is_some() && !schema.fields().is_empty() && mapping.iter().all(Option::is_none)
    {
        return Err(invalid_schema(format!(
            "none of the schema fields were found in the CSV header of {}; use --no-header if the file has no header row",
            input.display()
        )));
    }
    for (field, index) in schema.fields().iter().zip(mapping) {
        if index.is_none() && !field.is_nullable() {
            return Err(invalid_schema(format!(
                "required field '{}' was not found in the CSV header of {}",
                fmt::safe(field.name()),
                input.display()
            )));
        }
    }
    Ok(())
}

fn align_csv_batch_to_schema(
    batch: RecordBatch,
    schema: &Schema,
    mapping: &[Option<usize>],
    input: &Path,
) -> std::io::Result<RecordBatch> {
    let columns: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .zip(mapping)
        .map(|(field, index)| match index {
            Some(index) if batch.column(*index).data_type() == &DataType::Utf8 => {
                match field.data_type() {
                    DataType::Float64 => {
                        parse_inferred_csv_float64(batch.column(*index), field, input)
                    }
                    DataType::Timestamp(_, _) => {
                        parse_inferred_csv_timestamp_column(batch.column(*index), field, input)
                    }
                    _ => Ok(batch.column(*index).clone()),
                }
            }
            Some(index) => Ok(batch.column(*index).clone()),
            None => Ok(new_null_array(field.data_type(), batch.num_rows())),
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    RecordBatch::try_new(Arc::new(schema.clone()), columns)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
}

fn parse_inferred_csv_timestamp_column(
    array: &ArrayRef,
    field: &Field,
    input: &Path,
) -> std::io::Result<ArrayRef> {
    let values = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| invalid_schema("expected a Utf8 CSV timestamp column"))?;
    let DataType::Timestamp(unit, timezone) = field.data_type() else {
        return Err(invalid_schema("expected a Timestamp CSV field"));
    };
    let parser_timezone: Tz = timezone
        .as_deref()
        .unwrap_or("+00:00")
        .parse()
        .map_err(|e| invalid_schema(format!("invalid timestamp timezone: {e}")))?;
    let timezone_policy = timezone
        .is_none()
        .then_some("an inferred local timestamp; provide --schema to select timestamp semantics");
    let location = format!("in {}", input.display());
    let values = values
        .iter()
        .map(|value| {
            let Some(value) = value else {
                return Ok(None);
            };
            parse_csv_timestamp_value(
                value,
                field,
                unit,
                &parser_timezone,
                timezone_policy,
                &location,
            )
            .map(Some)
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    csv_timestamp_array(values, unit, timezone.clone())
}

fn parse_inferred_csv_float64(
    array: &ArrayRef,
    field: &Field,
    input: &Path,
) -> std::io::Result<ArrayRef> {
    let values = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| invalid_schema("expected a Utf8 CSV column"))?;
    let values = values
        .iter()
        .map(|value| {
            let Some(value) = value else {
                return Ok(None);
            };
            let promoted = Float64Type::parse(value)
                .ok_or_else(|| csv_inferred_float_parse_error(value, field, input))?;
            // Float-shaped values retain normal floating-point semantics.
            // Integer-shaped tokens must round-trip exactly.
            let unsigned = value
                .strip_prefix('+')
                .or_else(|| value.strip_prefix('-'))
                .unwrap_or(value);
            let integer_token =
                !unsigned.is_empty() && unsigned.bytes().all(|byte| byte.is_ascii_digit());
            if promoted.is_finite() {
                if integer_token {
                    match parse_decimal_unscaled_exact(value, 38, 0) {
                        Ok(exact) if promoted as i128 != exact => {
                            return Err(csv_inferred_float_parse_error(value, field, input));
                        }
                        Err(DecimalParseFailure::Invalid | DecimalParseFailure::Overflow) => {
                            return Err(csv_inferred_float_parse_error(value, field, input));
                        }
                        Ok(_) | Err(DecimalParseFailure::Inexact) => {}
                    }
                }
            } else if !matches!(
                value.to_ascii_lowercase().as_str(),
                "nan" | "inf" | "+inf" | "-inf" | "infinity" | "+infinity" | "-infinity"
            ) {
                return Err(csv_inferred_float_parse_error(value, field, input));
            }
            Ok(Some(promoted))
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    Ok(Arc::new(
        values.into_iter().collect::<PrimitiveArray<Float64Type>>(),
    ))
}

fn csv_inferred_float_parse_error(value: &str, field: &Field, input: &Path) -> std::io::Error {
    invalid_schema(format!(
        "numeric value '{}' in CSV field '{}' of {} cannot be represented exactly as Float64 during CSV schema inference",
        fmt::safe(value),
        fmt::safe(field.name()),
        input.display()
    ))
}

fn csv_schema_with_csv_names(
    schema: Schema,
    options: &CsvConvertOptions,
) -> std::io::Result<Schema> {
    let names = if let Some(header) = &options.header {
        Some(parse_csv_header(header, options)?)
    } else if options.no_header {
        Some(
            (0..schema.fields().len())
                .map(|i| format!("field_{i}"))
                .collect(),
        )
    } else {
        None
    };
    let Some(names) = names else {
        return Ok(schema);
    };
    if names.len() != schema.fields().len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "CSV header has {} fields but inferred schema has {} fields",
                names.len(),
                schema.fields().len()
            ),
        ));
    }
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .zip(names)
        .map(|(field, name)| field.as_ref().clone().with_name(name))
        .collect();
    Ok(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

fn parse_csv_header(header: &str, options: &CsvConvertOptions) -> std::io::Result<Vec<String>> {
    let delimiter = parse_csv_byte(&options.delimiter, "delimiter")?;
    let escape = parse_optional_csv_byte(options.escape.as_deref(), "escape")?;
    let quote = parse_csv_byte(&options.quote, "quote")?;
    let mut builder = csv::ReaderBuilder::new();
    builder
        .has_headers(false)
        .delimiter(delimiter)
        .quote(quote)
        .escape(escape);
    let mut reader = builder.from_reader(header.as_bytes());
    let mut records = reader.records();
    let record = records
        .next()
        .ok_or_else(|| invalid_schema("--header must contain at least one field"))?
        .map_err(|e| invalid_schema(format!("invalid --header CSV: {e}")))?;
    if let Some(next) = records.next() {
        next.map_err(|e| invalid_schema(format!("invalid --header CSV: {e}")))?;
        return Err(invalid_schema(
            "--header must contain exactly one CSV record",
        ));
    }
    let header: Vec<String> = record.iter().map(ToString::to_string).collect();
    validate_csv_header_names(&header)?;
    Ok(header)
}

fn validate_csv_header_names(header: &[String]) -> std::io::Result<()> {
    let mut seen = std::collections::HashSet::new();
    for name in header {
        if name.is_empty() {
            return Err(invalid_schema("empty column name"));
        }
        if !seen.insert(name.as_str()) {
            return Err(invalid_schema(format!(
                "duplicate CSV header field '{}'",
                fmt::safe(name)
            )));
        }
    }
    Ok(())
}

fn csv_schema_with_null_fallback(schema: Schema) -> Schema {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|field| {
            let field = field.as_ref().clone();
            if matches!(field.data_type(), DataType::Null) {
                field.with_data_type(DataType::Utf8)
            } else {
                field
            }
        })
        .collect();
    Schema::new_with_metadata(fields, schema.metadata().clone())
}

fn promote_second_precision_csv_timestamps(schema: Schema) -> Schema {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|field| {
            let field = field.as_ref().clone();
            match field.data_type().clone() {
                DataType::Timestamp(TimeUnit::Second, timezone) => {
                    field.with_data_type(DataType::Timestamp(TimeUnit::Millisecond, timezone))
                }
                _ => field,
            }
        })
        .collect();
    Schema::new_with_metadata(fields, schema.metadata().clone())
}

fn reject_csv_unsupported_fields(schema: &Schema) -> std::io::Result<()> {
    for field in schema.fields() {
        let avro_type = match field.data_type() {
            DataType::Binary => Some("bytes"),
            DataType::List(_) => Some("array"),
            DataType::Map(_, _) => Some("map"),
            _ => None,
        };
        if let Some(avro_type) = avro_type {
            return Err(invalid_schema(format!(
                "CSV conversion does not support Avro '{avro_type}' field '{}'; use a scalar type or a JSON input",
                fmt::safe(field.name()),
            )));
        }
    }
    Ok(())
}

fn reject_json_unsupported_fields(schema: &Schema) -> std::io::Result<()> {
    fn reject(field: &Field, path: &str) -> std::io::Result<()> {
        match field.data_type() {
            DataType::Binary => Err(invalid_schema(format!(
                "JSON conversion does not support raw Avro 'bytes' field '{}'; use a supported logical type or encode the value as a string field",
                fmt::safe(path)
            ))),
            DataType::List(item) => reject(item, &format!("{path}[]")),
            DataType::Map(entries, _) => {
                let DataType::Struct(fields) = entries.data_type() else {
                    return Ok(());
                };
                fields
                    .get(1)
                    .map_or(Ok(()), |value| reject(value, &format!("{path}{{}}")))
            }
            DataType::Struct(fields) => {
                for child in fields {
                    reject(child, &format!("{path}.{}", child.name()))?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    for field in schema.fields() {
        reject(field, field.name())?;
    }
    Ok(())
}

fn merge_csv_inferred_schema(prev: Schema, next: Schema, input: &Path) -> std::io::Result<Schema> {
    if prev.fields().len() != next.fields().len() {
        return Err(csv_schema_mismatch(input));
    }
    let next_fields: std::collections::HashMap<&str, &Field> = next
        .fields()
        .iter()
        .map(|field| (field.name().as_str(), field.as_ref()))
        .collect();
    let fields: Vec<Field> = prev
        .fields()
        .iter()
        .map(|left| {
            let right = next_fields
                .get(left.name().as_str())
                .copied()
                .ok_or_else(|| csv_schema_mismatch(input))?;
            merge_csv_inferred_field(left.as_ref(), right, input)
        })
        .collect::<std::io::Result<_>>()?;
    Ok(Schema::new_with_metadata(fields, prev.metadata().clone()))
}

fn merge_csv_inferred_field(left: &Field, right: &Field, input: &Path) -> std::io::Result<Field> {
    if left.name() != right.name() {
        return Err(csv_schema_mismatch(input));
    }
    let nullable = left.is_nullable() || right.is_nullable();
    let field = match (left.data_type(), right.data_type()) {
        (left_type, right_type) if left_type == right_type => left.clone().with_nullable(nullable),
        (DataType::Null, _) => right.clone().with_nullable(true),
        (_, DataType::Null) => left.clone().with_nullable(true),
        (DataType::Int64, DataType::Float64) | (DataType::Float64, DataType::Int64) => left
            .clone()
            .with_data_type(DataType::Float64)
            .with_nullable(nullable),
        (
            DataType::Timestamp(left_unit, left_timezone),
            DataType::Timestamp(right_unit, right_timezone),
        ) if left_timezone == right_timezone => left
            .clone()
            .with_data_type(DataType::Timestamp(
                finer_timestamp_unit(left_unit, right_unit),
                left_timezone.clone(),
            ))
            .with_nullable(nullable),
        _ => return Err(csv_schema_mismatch(input)),
    };
    Ok(field)
}

fn finer_timestamp_unit(left: &TimeUnit, right: &TimeUnit) -> TimeUnit {
    match (left, right) {
        (TimeUnit::Nanosecond, _) | (_, TimeUnit::Nanosecond) => TimeUnit::Nanosecond,
        (TimeUnit::Microsecond, _) | (_, TimeUnit::Microsecond) => TimeUnit::Microsecond,
        (TimeUnit::Millisecond, _) | (_, TimeUnit::Millisecond) => TimeUnit::Millisecond,
        _ => TimeUnit::Second,
    }
}

fn csv_schema_mismatch(input: &Path) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "{} seems to have a different schema from others. Please specify the correct schema explicitly with the --schema option.",
            input.display()
        ),
    )
}

// Hard ceiling on the `--schema` file. The path is user-supplied, so a
// softlink to /dev/zero or an inflated file must not be able to OOM the CLI.
// 4 MiB is generous for any realistic Avro record schema.
const MAX_SCHEMA_FILE_BYTES: u64 = 4 * 1024 * 1024;

fn load_convert_schema(path: &Path) -> std::io::Result<Schema> {
    use std::io::Read;
    let mut text = String::new();
    std::fs::File::open(path)?
        .take(MAX_SCHEMA_FILE_BYTES + 1)
        .read_to_string(&mut text)?;
    if text.len() as u64 > MAX_SCHEMA_FILE_BYTES {
        return Err(invalid_schema(format!(
            "--schema file {} exceeds the {} byte limit",
            path.display(),
            MAX_SCHEMA_FILE_BYTES
        )));
    }
    parse_avro_schema(&text)
}

fn parse_avro_schema(spec: &str) -> std::io::Result<Schema> {
    let value: Value = serde_json::from_str(spec)
        .map_err(|e| invalid_schema(format!("invalid Avro schema JSON: {e}")))?;
    let obj = value
        .as_object()
        .ok_or_else(|| invalid_schema("Avro schema must be a record object"))?;
    let schema_type = obj
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_schema("Avro schema must have type: \"record\""))?;
    if schema_type != "record" {
        return Err(invalid_schema(format!(
            "Avro schema type must be record, got '{schema_type}'"
        )));
    }
    let record_name = obj
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_schema("Avro record schema must contain a string record name"))?;
    if !is_valid_avro_fullname(record_name) {
        return Err(invalid_schema(format!(
            "invalid Avro record name '{}'",
            fmt::safe(record_name)
        )));
    }
    if let Some(namespace) = obj.get("namespace") {
        let namespace = namespace
            .as_str()
            .ok_or_else(|| invalid_schema("Avro record namespace must be a string"))?;
        if !namespace.is_empty() && !is_valid_avro_fullname(namespace) {
            return Err(invalid_schema(format!(
                "invalid Avro record namespace '{}'",
                fmt::safe(namespace)
            )));
        }
    }
    let avro_fields = obj
        .get("fields")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_schema("Avro record schema must contain a fields array"))?;
    let mut fields = Vec::with_capacity(avro_fields.len());
    let mut field_names = std::collections::HashSet::with_capacity(avro_fields.len());
    for field in avro_fields {
        let field_obj = field
            .as_object()
            .ok_or_else(|| invalid_schema("Avro field must be an object"))?;
        let name = field_obj
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_schema("Avro field must contain a string name"))?;
        if !is_valid_avro_name(name) {
            return Err(invalid_schema(format!(
                "invalid Avro field name '{}'",
                fmt::safe(name)
            )));
        }
        if !field_names.insert(name) {
            return Err(invalid_schema(format!(
                "duplicate Avro field name '{}'",
                fmt::safe(name)
            )));
        }
        let avro_type = field_obj
            .get("type")
            .ok_or_else(|| invalid_schema(format!("Avro field '{name}' is missing type")))?;
        let parsed = parse_avro_type(avro_type)
            .map_err(|e| invalid_schema(format!("Avro field '{name}': {e}")))?;
        fields.push(parsed.into_field(name));
    }
    if fields.is_empty() {
        return Err(invalid_schema(
            "Avro record schema must contain at least one field",
        ));
    }
    Ok(Schema::new(fields))
}

fn is_valid_avro_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|ch| ch == '_' || ch.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn is_valid_avro_fullname(name: &str) -> bool {
    !name.is_empty() && name.split('.').all(is_valid_avro_name)
}

struct ParsedAvroType {
    data_type: DataType,
    nullable: bool,
    logical_type: Option<&'static str>,
}

impl ParsedAvroType {
    fn physical(data_type: DataType) -> Self {
        Self {
            data_type,
            nullable: false,
            logical_type: None,
        }
    }

    fn uuid() -> Self {
        Self {
            data_type: DataType::Utf8,
            nullable: false,
            logical_type: Some(AVRO_UUID_LOGICAL_TYPE),
        }
    }

    fn into_field(self, name: impl Into<String>) -> Field {
        let field = Field::new(name, self.data_type, self.nullable);
        match self.logical_type {
            Some(logical_type) => field.with_metadata(std::collections::HashMap::from([(
                AVRO_LOGICAL_TYPE_METADATA.to_string(),
                logical_type.to_string(),
            )])),
            None => field,
        }
    }
}

fn parse_avro_type(value: &Value) -> Result<ParsedAvroType, String> {
    match value {
        Value::String(name) => parse_avro_named_type(name, None),
        Value::Object(obj) => {
            let name = obj
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| "Avro type object must contain a string type".to_string())?;
            parse_avro_named_type(name, Some(value))
        }
        Value::Array(types) => parse_avro_union(types),
        _ => Err("Avro type must be a string, object, or union array".to_string()),
    }
}

fn parse_avro_union(types: &[Value]) -> Result<ParsedAvroType, String> {
    let mut has_null = false;
    let mut non_null = None;
    for ty in types {
        if matches!(ty, Value::String(s) if s == "null") {
            has_null = true;
            continue;
        }
        let parsed = parse_avro_type(ty)?;
        if parsed.nullable {
            return Err("nested nullable unions are not supported".to_string());
        }
        if non_null.replace(parsed).is_some() {
            return Err("Avro unions with multiple non-null types are not supported".to_string());
        }
    }
    let mut parsed =
        non_null.ok_or_else(|| "pure null Avro fields are not supported".to_string())?;
    parsed.nullable = has_null;
    Ok(parsed)
}

fn parse_avro_named_type(name: &str, full_type: Option<&Value>) -> Result<ParsedAvroType, String> {
    let logical_type = full_type
        .and_then(|value| value.get("logicalType"))
        .and_then(Value::as_str);
    if let Some(logical_type) = logical_type {
        if let Some(result) = parse_avro_logical_type(name, logical_type, full_type.unwrap()) {
            return result;
        }
    }
    match name {
        "boolean" => Ok(ParsedAvroType::physical(DataType::Boolean)),
        "int" => Ok(ParsedAvroType::physical(DataType::Int32)),
        "long" => Ok(ParsedAvroType::physical(DataType::Int64)),
        "float" => Ok(ParsedAvroType::physical(DataType::Float32)),
        "double" => Ok(ParsedAvroType::physical(DataType::Float64)),
        "string" => Ok(ParsedAvroType::physical(DataType::Utf8)),
        "bytes" => Ok(ParsedAvroType::physical(DataType::Binary)),
        "array" => parse_avro_array(
            full_type.ok_or_else(|| "Avro array type must contain items".to_string())?,
        ),
        "map" => parse_avro_map(
            full_type.ok_or_else(|| "Avro map type must contain values".to_string())?,
        ),
        "null" => Err("null is not a supported schema type; use a nullable union".to_string()),
        other => Err(format!("unsupported Avro type '{other}'")),
    }
}

fn parse_avro_array(full_type: &Value) -> Result<ParsedAvroType, String> {
    let items = full_type
        .get("items")
        .ok_or_else(|| "Avro array type must contain items".to_string())?;
    let items = parse_avro_type(items)?;
    Ok(ParsedAvroType::physical(DataType::List(Arc::new(
        items.into_field("item"),
    ))))
}

fn parse_avro_map(full_type: &Value) -> Result<ParsedAvroType, String> {
    let values = full_type
        .get("values")
        .ok_or_else(|| "Avro map type must contain values".to_string())?;
    let values = parse_avro_type(values)?;
    let entries = Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("keys", DataType::Utf8, false),
            values.into_field("values"),
        ])),
        false,
    );
    Ok(ParsedAvroType::physical(DataType::Map(
        Arc::new(entries),
        false,
    )))
}

fn parse_avro_logical_type(
    physical_type: &str,
    logical_type: &str,
    full_type: &Value,
) -> Option<Result<ParsedAvroType, String>> {
    Some(match (physical_type, logical_type) {
        ("int", "date") => Ok(ParsedAvroType::physical(DataType::Date32)),
        ("int", "time-millis") => Ok(ParsedAvroType::physical(DataType::Time32(
            TimeUnit::Millisecond,
        ))),
        ("long", "timestamp-millis") => Ok(ParsedAvroType::physical(DataType::Timestamp(
            TimeUnit::Millisecond,
            Some("+00:00".into()),
        ))),
        ("long", "timestamp-micros") => Ok(ParsedAvroType::physical(DataType::Timestamp(
            TimeUnit::Microsecond,
            Some("+00:00".into()),
        ))),
        ("long", "timestamp-nanos") => Ok(ParsedAvroType::physical(DataType::Timestamp(
            TimeUnit::Nanosecond,
            Some("+00:00".into()),
        ))),
        ("long", "local-timestamp-millis") => Ok(ParsedAvroType::physical(DataType::Timestamp(
            TimeUnit::Millisecond,
            None,
        ))),
        ("long", "local-timestamp-micros") => Ok(ParsedAvroType::physical(DataType::Timestamp(
            TimeUnit::Microsecond,
            None,
        ))),
        ("long", "local-timestamp-nanos") => Ok(ParsedAvroType::physical(DataType::Timestamp(
            TimeUnit::Nanosecond,
            None,
        ))),
        ("bytes", "decimal") => parse_avro_decimal(full_type).map(ParsedAvroType::physical),
        ("fixed", "decimal") => parse_avro_fixed_decimal(full_type).map(ParsedAvroType::physical),
        ("string", "uuid") => Ok(ParsedAvroType::uuid()),
        _ => return None,
    })
}

fn parse_avro_fixed_decimal(full_type: &Value) -> Result<DataType, String> {
    let name = full_type
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| is_valid_avro_fullname(name))
        .ok_or_else(|| "fixed decimal must contain a valid name".to_string())?;
    let size = full_type
        .get("size")
        .and_then(Value::as_u64)
        .filter(|size| *size > 0)
        .ok_or_else(|| "fixed decimal must contain a positive integer size".to_string())?;
    let data_type = parse_avro_decimal(full_type)?;
    let DataType::Decimal128(precision, _) = data_type else {
        unreachable!()
    };
    let max_precision = if size >= 16 {
        38
    } else {
        let max_value = (1_u128 << (size * 8 - 1)) - 1;
        max_value.ilog10() as u8
    };
    if precision > max_precision {
        return Err(format!(
            "fixed decimal precision must be at most {max_precision} for size {size}, got {precision} in '{name}'"
        ));
    }
    Ok(data_type)
}

fn parse_avro_decimal(full_type: &Value) -> Result<DataType, String> {
    let precision = full_type
        .get("precision")
        .and_then(Value::as_u64)
        .ok_or_else(|| "decimal logical type must contain precision".to_string())?;
    let scale = match full_type.get("scale") {
        Some(scale) => scale
            .as_i64()
            .ok_or_else(|| "decimal scale must be an integer".to_string())?,
        None => 0,
    };
    let precision = u8::try_from(precision)
        .map_err(|_| format!("decimal precision must be in 1..38, got {precision}"))?;
    if precision == 0 || precision > 38 {
        return Err(format!(
            "decimal precision must be in 1..38, got {precision}"
        ));
    }
    let scale = i8::try_from(scale).map_err(|_| format!("invalid decimal scale {scale}"))?;
    // Avro requires 0 <= scale <= precision; catch it here rather than as an
    // Arrow error halfway through a conversion.
    if scale < 0 || scale as u8 > precision {
        return Err(format!(
            "decimal scale must be in 0..={precision}, got {scale}"
        ));
    }
    Ok(DataType::Decimal128(precision, scale))
}

fn apply_required_fields(schema: Schema, required_fields: &[String]) -> std::io::Result<Schema> {
    if required_fields.is_empty() {
        return Ok(schema);
    }
    for name in required_fields {
        if name.is_empty() {
            return Err(invalid_schema("--require field name cannot be empty"));
        }
        schema.index_of(name).map_err(|_| {
            invalid_schema(format!("--require column '{name}' not found in schema"))
        })?;
    }
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|field| {
            let field = field.as_ref().clone();
            if required_fields.iter().any(|name| name == field.name()) {
                field.with_nullable(false)
            } else {
                field
            }
        })
        .collect();
    Ok(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

fn invalid_schema(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, msg.into())
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
    let pred_col = match &pred {
        Some(p) => Some(
            reader
                .schema()
                .columns
                .iter()
                .position(|c| c.name == p.column)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("--where: column '{}' not found", p.column),
                    )
                })?,
        ),
        None => None,
    };
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
    let bounded_text = !json && num != usize::MAX;
    let mut batches: Vec<RecordBatch> = Vec::new();
    let mut got = 0usize;
    let mut printed_any = false;
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
        // every batch. Text output only buffers for bounded requests so global
        // column widths can be computed; unbounded `cat` prints one table per
        // row group and stays bounded.
        if json {
            print!("{}", fmt::ndjson(&[batch], num - got)?);
            printed_any = printed_any || batch_rows > 0;
        } else if bounded_text {
            batches.push(batch);
        } else if batch_rows > 0 {
            print!("{}", fmt::pretty_table(&[batch], usize::MAX));
            printed_any = true;
        }
        got += batch_rows;
    }
    if json {
        // (no rows) stays silent for JSON; nothing to print.
    } else if bounded_text {
        if batches.iter().all(|b| b.num_rows() == 0) {
            println!("(no rows)");
        } else {
            print!("{}", fmt::pretty_table(&batches, num));
        }
    } else if !printed_any {
        println!("(no rows)");
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
        .filter(|&i| selected(&want, &s.columns[i].name))
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
    // For nested columns the first physical slot is the ARRAY/MAP length column,
    // not the logical values — its dictionary would mislead. Only primitive
    // leaves have a meaningful one, so reject List/Map rather than print junk.
    use arrow::datatypes::DataType;
    if matches!(
        reader.schema().columns[col].data_type,
        DataType::List(_) | DataType::LargeList(_) | DataType::Map(_, _)
    ) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("dictionary: column '{column}' is nested; only primitive columns supported"),
        ));
    }
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
    let raw_name = |i: usize| s.columns[i].name.clone();
    let text_name = |i: usize| fmt::safe(&s.columns[i].name);
    let mut rgs = Vec::new();
    for rg in 0..reader.num_row_groups() {
        let infos = reader.bucket_infos(rg)?;
        if json {
            let items = infos
                .iter()
                .map(|b| jsonout::Bucket {
                    bucket: b.bucket,
                    kind: fmt::bucket_kind(b.kind),
                    size: b.size,
                    uncompressed: b.uncompressed,
                    columns: b.columns.iter().map(|&i| raw_name(i)).collect(),
                })
                .collect();
            rgs.push(items);
        } else {
            println!("row group {rg}:");
            for b in &infos {
                let cols: Vec<String> = b.columns.iter().map(|&i| text_name(i)).collect();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_avro_schema_accepts_nullable_and_logical_types() {
        let schema = parse_avro_schema(
            r#"{
  "type": "record",
  "name": "T",
  "fields": [
    {"name": "id", "type": "int"},
    {"name": "name", "type": ["null", "string"], "default": null},
    {"name": "amount", "type": {"type": "bytes", "logicalType": "decimal", "precision": 10, "scale": 2}},
    {"name": "ts", "type": {"type": "long", "logicalType": "timestamp-nanos"}},
    {"name": "local_ts", "type": {"type": "long", "logicalType": "local-timestamp-nanos"}}
  ]
}"#,
        )
        .unwrap();
        assert_eq!(schema.fields().len(), 5);
        assert_eq!(schema.fields()[0].data_type(), &DataType::Int32);
        assert!(!schema.fields()[0].is_nullable());
        assert_eq!(schema.fields()[1].data_type(), &DataType::Utf8);
        assert!(schema.fields()[1].is_nullable());
        assert_eq!(schema.fields()[2].data_type(), &DataType::Decimal128(10, 2));
        assert_eq!(
            schema.fields()[3].data_type(),
            &DataType::Timestamp(TimeUnit::Nanosecond, Some("+00:00".into()))
        );
        assert_eq!(
            schema.fields()[4].data_type(),
            &DataType::Timestamp(TimeUnit::Nanosecond, None)
        );
    }

    #[test]
    fn parse_avro_schema_ignores_unknown_logical_types() {
        let schema = parse_avro_schema(
            r#"{
  "type": "record",
  "name": "T",
  "fields": [
    {"name": "id", "type": {"type": "long", "logicalType": "vendor-id"}},
    {"name": "name", "type": {"type": "string", "logicalType": "vendor-name"}}
  ]
}"#,
        )
        .unwrap();
        assert_eq!(schema.fields()[0].data_type(), &DataType::Int64);
        assert_eq!(schema.fields()[1].data_type(), &DataType::Utf8);
        assert!(!field_is_avro_uuid(&schema.fields()[1]));
    }

    #[test]
    fn parse_avro_schema_supports_recursive_array_and_map_types() {
        let schema = parse_avro_schema(
            r#"{
  "type": "record",
  "name": "T",
  "fields": [
    {"name": "tags", "type": {"type": "array", "items": ["null", "string"]}},
    {"name": "props", "type": {"type": "map", "values": {"type": "array", "items": "long"}}}
  ]
}"#,
        )
        .unwrap();
        let DataType::List(items) = schema.fields()[0].data_type() else {
            panic!("expected List");
        };
        assert_eq!(items.data_type(), &DataType::Utf8);
        assert!(items.is_nullable());

        let DataType::Map(entries, false) = schema.fields()[1].data_type() else {
            panic!("expected unsorted Map");
        };
        let DataType::Struct(fields) = entries.data_type() else {
            panic!("expected Map entries struct");
        };
        assert_eq!(fields[0].data_type(), &DataType::Utf8);
        assert!(!fields[0].is_nullable());
        let DataType::List(items) = fields[1].data_type() else {
            panic!("expected List map values");
        };
        assert_eq!(items.data_type(), &DataType::Int64);
        assert!(!items.is_nullable());
    }

    #[test]
    fn parse_avro_schema_preserves_recursive_uuid_validation() {
        let schema = parse_avro_schema(
            r#"{
  "type": "record",
  "name": "T",
  "fields": [
    {"name": "id", "type": ["null", {"type": "string", "logicalType": "uuid"}]},
    {"name": "ids", "type": {"type": "array", "items": {"type": "string", "logicalType": "uuid"}}},
    {"name": "by_name", "type": {"type": "map", "values": {"type": "string", "logicalType": "uuid"}}}
  ]
}"#,
        )
        .unwrap();
        assert!(schema.fields()[0].is_nullable());
        assert!(field_is_avro_uuid(&schema.fields()[0]));

        let DataType::List(items) = schema.fields()[1].data_type() else {
            panic!("expected List");
        };
        assert!(field_is_avro_uuid(items));

        let DataType::Map(entries, false) = schema.fields()[2].data_type() else {
            panic!("expected Map");
        };
        let DataType::Struct(fields) = entries.data_type() else {
            panic!("expected Map entries struct");
        };
        assert!(field_is_avro_uuid(&fields[1]));
    }

    #[test]
    fn avro_uuid_requires_canonical_groups() {
        for value in [
            "550e8400-e29b-41d4-a716-446655440000",
            "550E8400-E29B-41D4-A716-446655440000",
        ] {
            assert!(validate_avro_uuid(value).is_ok(), "{value}");
        }
        for value in [
            "not-a-uuid",
            "550e8400e29b41d4a716446655440000",
            "550e8400-e29b-41d4-a716-44665544000g",
            "550e8400-e29b41d4-a716-446655440000",
        ] {
            assert!(validate_avro_uuid(value).is_err(), "{value}");
        }
    }

    #[test]
    fn convert_columns_split_each_comma_separated_occurrence() {
        assert_eq!(
            parse_convert_columns(&["id, kind".into(), "name".into()]).unwrap(),
            ["id", "kind", "name"]
        );
        assert!(parse_convert_columns(&[", ".into()]).is_err());
    }

    #[test]
    fn wide_csv_batch_size_bounds_decoder_preallocation() {
        assert_eq!(csv_batch_size(0), DEFAULT_CSV_BATCH_SIZE);
        assert_eq!(csv_batch_size(64), DEFAULT_CSV_BATCH_SIZE);
        assert_eq!(csv_batch_size(80_000), 1);
        for columns in [1, 64, 1_000, TARGET_CSV_DECODE_CELLS] {
            let batch_size = csv_batch_size(columns);
            assert!((1..=DEFAULT_CSV_BATCH_SIZE).contains(&batch_size));
            assert!(batch_size * columns <= TARGET_CSV_DECODE_CELLS);
        }
    }

    #[test]
    fn inferred_csv_batch_size_bounds_accumulated_record_payload() {
        assert_eq!(inferred_csv_batch_size(1, 0), DEFAULT_CSV_BATCH_SIZE);
        assert_eq!(
            inferred_csv_batch_size(1, 2 * 1024 * 1024),
            TARGET_CONVERT_BATCH_BYTES / (2 * 1024 * 1024)
        );
        assert_eq!(inferred_csv_batch_size(1, MAX_CSV_RECORD_BYTES), 1);
        assert_eq!(inferred_csv_batch_size(TARGET_CSV_DECODE_CELLS, 1), 1);
    }

    #[test]
    fn explicit_csv_batch_bound_includes_output_schema_width() {
        assert_eq!(explicit_csv_row_cells(1, 4096), 4096);
        assert_eq!(csv_batch_size(explicit_csv_row_cells(1, 4096)), 16);
        assert_eq!(explicit_csv_row_cells(8192, 4096), 8192);

        let records = (0..33).map(|i| Ok(csv::StringRecord::from(vec![i.to_string()])));
        let mut batch_sizes = Vec::new();
        let schema = Schema::new(
            (0..4096)
                .map(|i| Field::new(format!("c{i}"), DataType::Int64, true))
                .collect::<Vec<_>>(),
        );
        for_each_explicit_csv_batch(records, 1, &schema, TARGET_CONVERT_BATCH_BYTES, |batch| {
            batch_sizes.push(batch.len());
            Ok(())
        })
        .unwrap();
        assert_eq!(batch_sizes, [16, 16, 1]);
    }

    #[test]
    fn explicit_csv_batch_flushes_on_payload_bytes_and_isolates_oversized_row() {
        let records = ["aaaa", "bbbbbbbbb", "c"]
            .into_iter()
            .map(|value| Ok(csv::StringRecord::from(vec![value])));
        let schema = Schema::new(vec![Field::new("value", DataType::Utf8, false)]);
        let mut batches = Vec::new();
        for_each_explicit_csv_batch(records, 1, &schema, 8, |batch| {
            batches.push(
                batch
                    .iter()
                    .map(|record| record.as_slice().len())
                    .collect::<Vec<_>>(),
            );
            Ok(())
        })
        .unwrap();
        assert_eq!(batches, [vec![4], vec![9], vec![1]]);
    }

    #[test]
    fn validated_json_batches_use_soft_byte_budget_across_multiline_records() {
        let schema = parse_avro_schema(
            r#"{
  "type": "record",
  "name": "T",
  "fields": [
    {"name": "payload", "type": "string"},
    {"name": "amount", "type": {"type": "bytes", "logicalType": "decimal", "precision": 10, "scale": 2}}
  ]
}"#,
        )
        .unwrap();
        let input = format!(
            "{{\n  \"payload\": \"{}\",\n  \"amount\": \"12.34\"\n}}\n\
             {{\"payload\":\"{}\",\"amount\":1.20}} \
             {{\"payload\":\"{}\",\"amount\":3.40}}",
            "x".repeat(80),
            "b".repeat(24),
            "c".repeat(24)
        );
        let reader = std::io::BufReader::with_capacity(7, std::io::Cursor::new(input.into_bytes()));
        let mut batch_rows = Vec::new();
        for_each_validated_json_batch(reader, &schema, 64, |batch| {
            batch_rows.push(batch.num_rows());
            Ok(())
        })
        .unwrap();
        assert_eq!(batch_rows, [1, 1, 1]);
    }

    #[test]
    fn json_record_limit_stops_at_limit_plus_one_without_reading_the_tail() {
        use std::cell::Cell;
        use std::io::Read;
        use std::rc::Rc;

        struct OneByteCountingReader {
            bytes: std::io::Cursor<Vec<u8>>,
            reads: Rc<Cell<usize>>,
        }

        impl Read for OneByteCountingReader {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let len = buf.len().min(1);
                let read = self.bytes.read(&mut buf[..len])?;
                self.reads.set(self.reads.get() + read);
                Ok(read)
            }
        }

        let reads = Rc::new(Cell::new(0));
        let reader = OneByteCountingReader {
            bytes: std::io::Cursor::new(
                format!(r#"{{"payload":"{}"}}"#, "x".repeat(256)).into_bytes(),
            ),
            reads: Rc::clone(&reads),
        };
        let err = validate_json_record_limits(reader, 16)
            .unwrap_err()
            .to_string();
        assert_eq!(err, "JSON record 1 exceeds the 16 byte limit");
        assert_eq!(reads.get(), 17);
    }

    #[test]
    fn json_record_limit_accepts_multiline_and_adjacent_values() {
        let input = b"{\n\"a\":1\n}{\"b\":[2,3]}1\"cool\"\"stuff\" 3{} [0]";
        let reader = std::io::BufReader::with_capacity(2, std::io::Cursor::new(input));
        validate_json_record_limits(reader, 64).unwrap();
    }

    #[test]
    fn json_record_chunk_scanner_matches_byte_scanner() {
        let inputs: &[&[u8]] = &[
            br#"{"a":[1,{"b":"x\\\"y"}]} {"c":2}"#,
            b"{\n\"a\":\"line\nvalue\"\n}\r\n[1,2,3]",
            br#"1{"a":2}["x"]"y" false null"#,
        ];
        for input in inputs {
            for chunk_size in [1, 2, 3, 7, 64] {
                let mut byte_scanner = JsonRecordScanner::new_with_limits(32, 64);
                let byte_result = input
                    .iter()
                    .try_for_each(|&byte| byte_scanner.scan(byte))
                    .map_err(|error| error.to_string());

                let mut chunk_scanner = JsonRecordScanner::new_with_limits(32, 64);
                let chunk_result = input
                    .chunks(chunk_size)
                    .try_for_each(|chunk| chunk_scanner.scan_chunk(chunk))
                    .map_err(|error| error.to_string());

                assert_eq!(chunk_result, byte_result, "chunk size {chunk_size}");
                assert_eq!(chunk_scanner.state, byte_scanner.state);
                assert_eq!(chunk_scanner.record, byte_scanner.record);
                assert_eq!(chunk_scanner.record_bytes, byte_scanner.record_bytes);
                assert_eq!(chunk_scanner.line_bytes, byte_scanner.line_bytes);
                assert_eq!(
                    chunk_scanner.structural_units,
                    byte_scanner.structural_units
                );
            }
        }
    }

    #[test]
    fn json_record_limit_rejects_dense_structural_input() {
        let mut scanner = JsonRecordScanner::new_with_limits(usize::MAX, 5);
        let err = scanner
            .scan_chunk(br#"{"values":[0,0,0,0]}"#)
            .unwrap_err()
            .to_string();
        assert_eq!(err, "JSON record 1 exceeds the 5 structural unit limit");
    }

    #[test]
    fn json_record_limit_does_not_accumulate_whitespace_across_short_lines() {
        let input = format!("{{}}\n{}{{}}", " \n".repeat(32));
        let reader = std::io::BufReader::with_capacity(3, std::io::Cursor::new(input.into_bytes()));
        validate_json_record_limits(reader, 16).unwrap();
    }

    #[test]
    fn convert_json_record_limit_covers_every_schema_dispatch() {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "mosaic_json_record_limit_{}_{}",
            std::process::id(),
            suffix
        ));
        std::fs::create_dir(&dir).unwrap();
        let input = dir.join("input.jsonl");
        let ordinary_schema = dir.join("ordinary.avsc");
        let validated_schema = dir.join("validated.avsc");
        std::fs::write(
            &input,
            format!(r#"{{"selected":"ok","unselected":"{}"}}"#, "x".repeat(128)),
        )
        .unwrap();
        std::fs::write(
            &ordinary_schema,
            r#"{"type":"record","name":"T","fields":[{"name":"selected","type":"string"},{"name":"unselected","type":"string"}]}"#,
        )
        .unwrap();
        std::fs::write(
            &validated_schema,
            r#"{"type":"record","name":"T","fields":[{"name":"amount","type":{"type":"bytes","logicalType":"decimal","precision":10,"scale":2}},{"name":"unselected","type":"string"}]}"#,
        )
        .unwrap();

        let cases = [
            (None, Vec::new()),
            (None, vec!["selected".to_string()]),
            (Some(ordinary_schema.as_path()), Vec::new()),
            (Some(validated_schema.as_path()), Vec::new()),
        ];
        for (index, (schema, columns)) in cases.into_iter().enumerate() {
            let out = dir.join(format!("out-{index}.mosaic"));
            let err =
                convert_with_json_record_limit(&input, &out, schema, &columns, None, false, 32)
                    .unwrap_err()
                    .to_string();
            assert!(
                err.contains("JSON record 1 exceeds the 32 byte limit"),
                "{err}"
            );
            assert!(!out.exists());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn convert_json_record_limit_bounds_arrow_line_whitespace() {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "mosaic_json_whitespace_limit_{}_{}",
            std::process::id(),
            suffix
        ));
        std::fs::create_dir(&dir).unwrap();
        let input = dir.join("input.jsonl");
        let out = dir.join("out.mosaic");
        std::fs::write(&input, format!("{{}}{}\n", " ".repeat(128))).unwrap();
        let err = convert_with_json_record_limit(&input, &out, None, &[], None, false, 16)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("JSON input line exceeds the 16 byte limit"),
            "{err}"
        );
        assert!(!out.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn convert_json_record_limit_bounds_many_values_on_one_arrow_line() {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "mosaic_json_line_limit_{}_{}",
            std::process::id(),
            suffix
        ));
        std::fs::create_dir(&dir).unwrap();
        let input = dir.join("input.jsonl");
        let out = dir.join("out.mosaic");
        std::fs::write(&input, "{}".repeat(32)).unwrap();
        let err = convert_with_json_record_limit(&input, &out, None, &[], None, false, 16)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("JSON input line exceeds the 16 byte limit"),
            "{err}"
        );
        assert!(!out.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn csv_projection_remaps_output_columns() {
        let (projection, mapping) = csv_projection(&[Some(5), None, Some(1)]);
        assert_eq!(projection, [5, 1]);
        assert_eq!(mapping, [Some(0), None, Some(1)]);
    }

    #[test]
    fn csv_schema_index_drives_wide_reordered_mappings() {
        let schema = Schema::new(
            (0..4096)
                .map(|i| Field::new(format!("field_{i}"), DataType::Int64, true))
                .collect::<Vec<_>>(),
        );
        let index = csv_schema_index(&schema);
        let layout = CsvInputLayout {
            header: Some((0..4096).rev().map(|i| format!("field_{i}")).collect()),
            columns: 4096,
            has_records: true,
        };
        let reader_schema = csv_reader_schema(&schema, &index, &layout);
        let mapping = csv_output_mapping(&schema, &index, &layout);
        assert_eq!(reader_schema.fields()[0].name(), "field_0");
        assert_eq!(reader_schema.fields()[4095].name(), "field_4095");
        assert_eq!(mapping[0], Some(4095));
        assert_eq!(mapping[4095], Some(0));
    }

    #[test]
    fn inferred_csv_float_validates_bare_integer_tokens() {
        let field = Field::new("value", DataType::Float64, true);
        for value in [
            "-9007199254740992",
            "9007199254740992",
            "9007199254740994",
            "-9223372036854775808",
            "1.5",
            "9007199254740993.0",
            "9.007199254740993e15",
            "1.5e30",
            "1e300",
        ] {
            let input: ArrayRef = Arc::new(StringArray::from(vec![value]));
            let output = parse_inferred_csv_float64(&input, &field, Path::new("safe.csv")).unwrap();
            let output = output
                .as_any()
                .downcast_ref::<PrimitiveArray<Float64Type>>()
                .unwrap();
            assert_eq!(
                output.value(0),
                Float64Type::parse(value).unwrap(),
                "{value}"
            );
        }
        let input: ArrayRef = Arc::new(StringArray::from(vec!["NaN", "inf", "-inf"]));
        let output =
            parse_inferred_csv_float64(&input, &field, Path::new("non-finite.csv")).unwrap();
        let output = output
            .as_any()
            .downcast_ref::<PrimitiveArray<Float64Type>>()
            .unwrap();
        assert!(output.value(0).is_nan());
        assert_eq!(output.value(1), f64::INFINITY);
        assert_eq!(output.value(2), f64::NEG_INFINITY);
        for value in [
            "-9007199254740993",
            "9007199254740993",
            "9223372036854775807",
        ] {
            let input: ArrayRef = Arc::new(StringArray::from(vec![value]));
            let err = parse_inferred_csv_float64(&input, &field, Path::new("lossy.csv"))
                .unwrap_err()
                .to_string();
            assert!(err.contains(value), "{err}");
        }
    }

    #[test]
    fn csv_error_lines_include_skipped_lines() {
        assert_eq!(
            csv_error_with_line_offset("bad record at line 3", 2),
            "bad record at line 5"
        );
        assert_eq!(
            csv_error_with_line_offset("CSV error: record 2 (line: 3, byte: 7)", 2),
            "CSV error: record 2 (line: 5, byte: 7)"
        );
        assert_eq!(
            csv_error_with_line_offset("CSV parse error: record 2 (line 3, field: 0)", 2),
            "CSV parse error: record 2 (line 5, field: 0)"
        );
        assert_eq!(
            csv_error_with_line_offset(
                "Csv error: Encountered invalid UTF-8 data for line 3 and field 1",
                2
            ),
            "Csv error: Encountered invalid UTF-8 data for line 5 and field 1"
        );
        assert_eq!(
            csv_data_error(
                arrow::error::ArrowError::ParseError(
                    "bad value at line 1. Row data: '[note at line 99]'".to_string()
                ),
                2
            )
            .to_string(),
            "Parser error: bad value at line 4. Row data: '[note at line 99]'"
        );
        assert_eq!(
            csv_error_with_line_offset("bad record without a position", 2),
            "bad record without a position"
        );
    }

    #[test]
    fn timestamp_precision_merge_is_order_independent() {
        assert_eq!(
            finer_timestamp_unit(&TimeUnit::Millisecond, &TimeUnit::Microsecond),
            TimeUnit::Microsecond
        );
        assert_eq!(
            finer_timestamp_unit(&TimeUnit::Microsecond, &TimeUnit::Millisecond),
            TimeUnit::Microsecond
        );
        assert_eq!(
            finer_timestamp_unit(&TimeUnit::Microsecond, &TimeUnit::Nanosecond),
            TimeUnit::Nanosecond
        );
        assert_eq!(
            finer_timestamp_unit(&TimeUnit::Nanosecond, &TimeUnit::Microsecond),
            TimeUnit::Nanosecond
        );
    }

    #[test]
    fn decimal_exactness_allows_only_zero_discarded_digits() {
        for value in ["12.34", "12.3400", "12.3", "1230e-3", "0.000"] {
            assert!(parse_decimal_exact(value, 10, 2).is_ok(), "{value}");
        }
        for value in ["12.349", "-12.349", "123e-3", "0.001"] {
            assert!(parse_decimal_exact(value, 10, 2).is_err(), "{value}");
        }
    }

    #[test]
    fn decimal_parsing_rejects_unbounded_literals_without_panicking() {
        let oversized_digits = "1".repeat(255);
        let wrapping_divisor = format!("0.{}e1", "0".repeat(129));
        let wrapping_value = format!("0.{}1e1", "0".repeat(128));
        for (value, expected) in [
            ("1e40000", None),
            (oversized_digits.as_str(), None),
            (wrapping_divisor.as_str(), Some(0)),
            (wrapping_value.as_str(), None),
        ] {
            let result = std::panic::catch_unwind(|| parse_decimal_exact(value, 38, 0));
            assert!(result.is_ok(), "decimal parsing panicked for {value}");
            match expected {
                Some(expected) => assert_eq!(result.unwrap().unwrap(), expected, "{value}"),
                None => assert!(result.unwrap().is_err(), "{value}"),
            }
        }
    }

    #[test]
    fn decimal_parsing_accepts_extra_trailing_zero_digits() {
        let value = format!("12.34{}", "0".repeat(40));
        assert_eq!(parse_decimal_exact(&value, 10, 2).unwrap(), 1234);
    }

    #[test]
    fn decimal_normalization_plan_preserves_nested_semantics_and_order() {
        let decimal = DataType::Decimal128(10, 2);
        let map_entries = Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("keys", DataType::Utf8, false),
                Field::new("values", decimal.clone(), false),
            ])),
            false,
        );
        let nested = Field::new(
            "nested",
            DataType::Struct(Fields::from(vec![
                Field::new("decimal", decimal.clone(), false),
                Field::new(
                    "list",
                    DataType::List(Arc::new(Field::new("item", decimal.clone(), false))),
                    false,
                ),
                Field::new("map", DataType::Map(Arc::new(map_entries), false), false),
                Field::new("plain", DataType::Utf8, false),
            ])),
            false,
        );
        let schema = Schema::new(vec![Field::new("top", decimal, false), nested]);
        let plan = JsonDecimalStructPlan::from_fields(schema.fields());
        let raw: Box<RawValue> = serde_json::from_str(
            r#"{"unknown":{"keep":[1]},"top":1.2,"nested":{"unknown":true,"list":[2,3.40],"map":{"z":"4.50","a":5},"decimal":"6.7","plain":"same"}}"#,
        )
        .unwrap();
        assert_eq!(
            normalize_json_decimal_record(&raw, &plan, 1).unwrap(),
            r#"{"nested":{"decimal":"6.70","list":["2.00","3.40"],"map":{"a":"5.00","z":"4.50"},"plain":"same","unknown":true},"top":"1.20","unknown":{"keep":[1]}}"#
        );
    }

    #[test]
    fn decimal_normalization_plan_uses_one_name_lookup_per_object_key() {
        const WIDTH: usize = 1024;
        let decimal = DataType::Decimal128(10, 2);
        let nested_fields = (0..WIDTH)
            .map(|i| Field::new(format!("n{i}"), DataType::Utf8, false))
            .chain(std::iter::once(Field::new(
                "nested_amount",
                decimal.clone(),
                false,
            )))
            .collect::<Vec<_>>();
        let root_fields = (0..WIDTH)
            .map(|i| Field::new(format!("r{i}"), DataType::Utf8, false))
            .chain([
                Field::new("amount", decimal, false),
                Field::new(
                    "nested",
                    DataType::Struct(Fields::from(nested_fields)),
                    false,
                ),
            ])
            .collect::<Vec<_>>();
        let schema = Schema::new(root_fields);
        let plan = JsonDecimalStructPlan::from_fields(schema.fields());

        let mut nested = String::from("{");
        for i in 0..WIDTH {
            if i != 0 {
                nested.push(',');
            }
            nested.push_str(&format!(r#""n{i}":"v""#));
        }
        nested.push_str(r#","nested_amount":1.2,"unknown_nested":true}"#);
        let mut input = String::from("{");
        for i in 0..WIDTH {
            if i != 0 {
                input.push(',');
            }
            input.push_str(&format!(r#""r{i}":"v""#));
        }
        input.push_str(&format!(
            r#","amount":1.2,"nested":{nested},"unknown_root":true}}"#
        ));
        let raw: Box<RawValue> = serde_json::from_str(&input).unwrap();
        normalize_json_decimal_record(&raw, &plan, 1).unwrap();
        assert_eq!(plan.lookup_count(), 2 * WIDTH + 5);
    }

    #[test]
    fn decimal_normalization_plan_keeps_first_duplicate_schema_field() {
        let schema = Schema::new(vec![
            Field::new("amount", DataType::Utf8, false),
            Field::new("amount", DataType::Decimal128(10, 2), false),
        ]);
        let plan = JsonDecimalStructPlan::from_fields(schema.fields());
        let raw: Box<RawValue> = serde_json::from_str(r#"{"amount":1.2}"#).unwrap();
        assert_eq!(
            normalize_json_decimal_record(&raw, &plan, 1).unwrap(),
            r#"{"amount":1.2}"#
        );
    }

    #[test]
    fn decimal_normalization_rejects_expansion_past_hard_limit() {
        let schema = Schema::new(vec![Field::new(
            "amount",
            DataType::Decimal128(10, 2),
            false,
        )]);
        let plan = JsonDecimalStructPlan::from_fields(schema.fields());
        let raw: Box<RawValue> = serde_json::from_str(r#"{"amount":1}"#).unwrap();
        let err = normalize_json_decimal_record_with_limit(&raw, &plan, 7, raw.get().len())
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            format!(
                "normalized JSON record 7 exceeds the {} byte limit",
                raw.get().len()
            )
        );
    }

    #[test]
    fn json_special_validation_uses_the_last_duplicate_record_field() {
        let schema = Schema::new(vec![Field::new(
            "amount",
            DataType::Decimal128(10, 2),
            false,
        )]);
        let fields = json_special_fields(&schema);
        validate_json_special_values(br#"{"amount":"bad","amount":"2.00"}"#, &fields, 1).unwrap();

        let err = validate_json_special_values(br#"{"amount":"2.00","amount":"bad"}"#, &fields, 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("cannot parse 'bad'"), "{err}");
    }

    #[test]
    fn local_timestamp_timezone_detection_ignores_fractional_precision() {
        for value in [
            "2026-08-20T12:34:56Z",
            "2026-08-20T12:34:56+08:00",
            "2026-08-20T12:34:56.123-08:00",
            "2026-08-20T12:34Z",
            "2026-08-20T12:34+05:00",
        ] {
            assert!(timestamp_has_explicit_timezone(value), "{value}");
        }
        for value in [
            "2026-08-20",
            "2026-08-20T12:34",
            "2026-08-20T12:34:56",
            "2026-08-20T12:34:56.123456789",
        ] {
            assert!(!timestamp_has_explicit_timezone(value), "{value}");
        }
    }

    #[test]
    fn explicit_csv_timestamps_floor_before_epoch() {
        let records = [csv::StringRecord::from(vec![
            "1969-12-31T23:59:59.999500Z",
            "1969-12-31T23:59:59.999999500",
        ])];
        let millis_field = Field::new(
            "millis",
            DataType::Timestamp(TimeUnit::Millisecond, Some("+00:00".into())),
            false,
        );
        let micros_field = Field::new(
            "micros",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        );
        let millis = csv_column_array(&records, 0, &millis_field).unwrap();
        let micros = csv_column_array(&records, 1, &micros_field).unwrap();
        assert_eq!(
            millis
                .as_any()
                .downcast_ref::<PrimitiveArray<TimestampMillisecondType>>()
                .unwrap()
                .value(0),
            -1
        );
        assert_eq!(
            micros
                .as_any()
                .downcast_ref::<PrimitiveArray<TimestampMicrosecondType>>()
                .unwrap()
                .value(0),
            -1
        );
    }

    #[test]
    fn duplicate_csv_header_errors_sanitize_control_characters() {
        let name = "bad\u{1b}]2;title\u{7}";
        let err = validate_csv_header_names(&[name.to_string(), name.to_string()])
            .unwrap_err()
            .to_string();
        assert!(!err.chars().any(char::is_control), "{err:?}");
        assert!(err.contains("bad\u{fffd}]2;title\u{fffd}"), "{err:?}");
    }

    #[test]
    fn missing_required_csv_field_errors_sanitize_control_characters() {
        let name = "bad\u{1b}]2;title\u{7}";
        let schema = Schema::new(vec![
            Field::new(name, DataType::Utf8, false),
            Field::new("other", DataType::Utf8, true),
        ]);
        let layout = CsvInputLayout {
            header: Some(vec!["other".to_string()]),
            columns: 1,
            has_records: true,
        };
        let err = validate_csv_mapping(&schema, &layout, &[None, Some(0)], Path::new("input.csv"))
            .unwrap_err()
            .to_string();
        assert!(!err.chars().any(char::is_control), "{err:?}");
        assert!(err.contains("bad\u{fffd}]2;title\u{fffd}"), "{err:?}");
    }

    #[test]
    fn unsupported_csv_field_errors_sanitize_control_characters() {
        let name = "bad\u{1b}]2;title\u{7}";
        let schema = Schema::new(vec![Field::new(name, DataType::Binary, false)]);
        let err = reject_csv_unsupported_fields(&schema)
            .unwrap_err()
            .to_string();
        assert!(!err.chars().any(char::is_control), "{err:?}");
        assert!(err.contains("bad\u{fffd}]2;title\u{fffd}"), "{err:?}");
    }

    #[test]
    fn null_inferred_field_error_sanitizes_control_characters() {
        let name = "bad\u{1b}]2;title\u{7}";
        let schema = Schema::new(vec![Field::new(name, DataType::Null, true)]);
        let err = reject_null_inferred_fields(&schema)
            .unwrap_err()
            .to_string();
        assert!(!err.chars().any(char::is_control), "{err:?}");
        assert!(err.contains("bad\u{fffd}]2;title\u{fffd}"), "{err:?}");
    }

    #[test]
    fn parse_avro_schema_rejects_out_of_range_decimal_scale() {
        for scale in ["-1", "11"] {
            let err = parse_avro_schema(&format!(
                r#"{{
  "type": "record",
  "name": "T",
  "fields": [{{"name": "a", "type": {{"type": "bytes", "logicalType": "decimal", "precision": 10, "scale": {scale}}}}}]
}}"#,
            ))
            .unwrap_err();
            assert!(
                err.to_string().contains("decimal scale must be in 0..=10"),
                "{err}"
            );
        }
    }

    #[test]
    fn parse_avro_schema_rejects_non_integer_decimal_scale() {
        for scale in [r#""2""#, "2.5", "null"] {
            let err = parse_avro_schema(&format!(
                r#"{{
  "type": "record",
  "name": "T",
  "fields": [{{"name": "a", "type": {{"type": "bytes", "logicalType": "decimal", "precision": 10, "scale": {scale}}}}}]
}}"#,
            ))
            .unwrap_err();
            assert!(
                err.to_string().contains("decimal scale must be an integer"),
                "{err}"
            );
        }
    }

    #[test]
    fn parse_avro_schema_rejects_invalid_fixed_decimal() {
        for (fixed, expected) in [
            (
                r#"{"type":"fixed","logicalType":"decimal","size":1,"precision":2}"#,
                "fixed decimal must contain a valid name",
            ),
            (
                r#"{"type":"fixed","logicalType":"decimal","name":"Tiny","precision":2}"#,
                "fixed decimal must contain a positive integer size",
            ),
            (
                r#"{"type":"fixed","logicalType":"decimal","name":"Tiny","size":1,"precision":3}"#,
                "fixed decimal precision must be at most 2 for size 1",
            ),
        ] {
            let err = parse_avro_schema(&format!(
                r#"{{
  "type": "record",
  "name": "T",
  "fields": [{{"name": "amount", "type": {fixed}}}]
}}"#,
            ))
            .unwrap_err()
            .to_string();
            assert!(err.contains(expected), "{err}");
        }
    }

    #[test]
    fn parse_avro_schema_rejects_pure_null_type() {
        let err = parse_avro_schema(
            r#"{
  "type": "record",
  "name": "T",
  "fields": [{"name": "empty", "type": "null"}]
}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("null is not a supported"));
    }

    #[test]
    fn parse_avro_schema_requires_valid_record_and_field_names() {
        for (schema, expected) in [
            (
                r#"{"type":"record","fields":[{"name":"id","type":"long"}]}"#,
                "record name",
            ),
            (
                r#"{"type":"record","name":"bad-name","fields":[{"name":"id","type":"long"}]}"#,
                "invalid Avro record name",
            ),
            (
                r#"{"type":"record","name":"T","fields":[{"name":"bad-name","type":"long"}]}"#,
                "invalid Avro field name",
            ),
        ] {
            let err = parse_avro_schema(schema).unwrap_err().to_string();
            assert!(err.contains(expected), "{err}");
        }
        parse_avro_schema(
            r#"{"type":"record","name":"com.example.T","fields":[{"name":"_id9","type":"long"}]}"#,
        )
        .unwrap();
    }

    #[test]
    fn parse_avro_schema_accepts_empty_namespace() {
        let schema = parse_avro_schema(
            r#"{"type":"record","name":"T","namespace":"","fields":[{"name":"id","type":"long"}]}"#,
        )
        .unwrap();
        assert_eq!(schema.fields()[0].data_type(), &DataType::Int64);
    }

    #[test]
    fn load_convert_schema_rejects_oversized_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("mosaic_test_oversized_schema.avsc");
        let mut spec = String::from(
            r#"{"type":"record","name":"T","fields":[{"name":"id","type":"long","doc":""#,
        );
        spec.push_str(&"x".repeat(MAX_SCHEMA_FILE_BYTES as usize + 4096));
        spec.push_str(r#""}]}"#);
        std::fs::write(&path, spec).unwrap();
        let err = load_convert_schema(&path).unwrap_err().to_string();
        let _ = std::fs::remove_file(&path);
        assert!(
            err.contains("exceeds the") && err.contains("byte limit"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn open_csv_input_rejects_too_many_columns() {
        let dir = std::env::temp_dir();
        let path = dir.join("mosaic_test_wide_header.csv");
        let header = (0..MAX_CSV_COLUMNS + 1)
            .map(|i| format!("c{i}"))
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(&path, format!("{header}\n")).unwrap();
        let options = CsvConvertOptions {
            delimiter: ",".to_string(),
            escape: None,
            quote: "\"".to_string(),
            no_header: false,
            header: None,
            skip_lines: 0,
        };
        let dialect = CsvDialect::from_options(&options).unwrap();
        let source = OpenCsvSource::open(&path, options.skip_lines).unwrap();
        let err = match prepare_inferred_csv_input(source, &options, dialect) {
            Ok(_) => panic!("expected CSV width guard to reject this header"),
            Err(err) => err.to_string(),
        };
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            err,
            format!(
                "CSV input {} has at least {} columns, exceeds the {} column limit",
                path.display(),
                MAX_CSV_COLUMNS + 1,
                MAX_CSV_COLUMNS
            )
        );
    }

    #[test]
    fn open_csv_input_rejects_too_many_no_header_columns() {
        let dir = std::env::temp_dir();
        let path = dir.join("mosaic_test_wide_no_header.csv");
        let record = std::iter::repeat_n("1", MAX_CSV_COLUMNS + 1)
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(&path, format!("{record}\n")).unwrap();
        let options = CsvConvertOptions {
            delimiter: ",".to_string(),
            escape: None,
            quote: "\"".to_string(),
            no_header: true,
            header: None,
            skip_lines: 0,
        };
        let dialect = CsvDialect::from_options(&options).unwrap();
        let source = OpenCsvSource::open(&path, options.skip_lines).unwrap();
        let err = match prepare_inferred_csv_input(source, &options, dialect) {
            Ok(_) => panic!("expected CSV width guard to reject this record"),
            Err(err) => err.to_string(),
        };
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            err,
            format!(
                "CSV input {} has at least {} columns, exceeds the {} column limit",
                path.display(),
                MAX_CSV_COLUMNS + 1,
                MAX_CSV_COLUMNS
            )
        );
    }

    #[test]
    fn open_csv_input_records_no_header_width() {
        let path = std::env::temp_dir().join("mosaic_test_no_header_width.csv");
        std::fs::write(&path, "a,b\n").unwrap();
        let options = CsvConvertOptions {
            delimiter: ",".to_string(),
            escape: None,
            quote: "\"".to_string(),
            no_header: true,
            header: None,
            skip_lines: 0,
        };
        let dialect = CsvDialect::from_options(&options).unwrap();
        let source = OpenCsvSource::open(&path, options.skip_lines).unwrap();
        let input = prepare_inferred_csv_input(source, &options, dialect).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(input.layout.columns, 2);
    }

    #[test]
    fn csv_record_guard_is_quote_and_escape_aware_across_chunks() {
        let dialect = CsvDialect {
            delimiter: b',',
            escape: Some(b'\\'),
            quote: b'"',
        };
        let input = b"\"a,b\",\"line\nvalue\",\"escaped\\\"quote\"\n";
        for chunk_size in [1, 2, 3, 7] {
            let mut scanner =
                CsvRecordLimitScanner::new(PathBuf::from("input.csv"), dialect, 3, 64);
            for chunk in input.chunks(chunk_size) {
                scanner.scan(chunk).unwrap();
            }
            scanner.finish().unwrap();
        }

        let mut scanner = CsvRecordLimitScanner::new(PathBuf::from("input.csv"), dialect, 3, 4);
        let err = scanner.scan(b"\"ab\ncd\",x,y\n").unwrap_err().to_string();
        assert!(err.contains("exceeds the 4 decoded byte limit"), "{err}");
    }

    #[test]
    fn csv_record_guard_rejects_late_width_without_reading_tail() {
        use std::cell::Cell;
        use std::io::Read;
        use std::rc::Rc;

        struct OneByteCountingReader {
            bytes: std::io::Cursor<Vec<u8>>,
            reads: Rc<Cell<usize>>,
        }

        impl Read for OneByteCountingReader {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                let capacity = buffer.len().min(1);
                let read = self.bytes.read(&mut buffer[..capacity])?;
                self.reads.set(self.reads.get().saturating_add(read));
                Ok(read)
            }
        }

        let data = b"a,b\n1,2,3,this-tail-must-not-be-read".to_vec();
        let reads = Rc::new(Cell::new(0));
        let inner = OneByteCountingReader {
            bytes: std::io::Cursor::new(data.clone()),
            reads: Rc::clone(&reads),
        };
        let dialect = CsvDialect {
            delimiter: b',',
            escape: None,
            quote: b'"',
        };
        let mut reader =
            CsvRecordLimitReader::new(inner, PathBuf::from("input.csv"), dialect, 2, 64);
        let err = std::io::copy(&mut reader, &mut std::io::sink())
            .unwrap_err()
            .to_string();
        assert!(err.contains("has at least 3 columns"), "{err}");
        assert!(
            reads.get() < data.len(),
            "{} >= {}",
            reads.get(),
            data.len()
        );
    }

    #[test]
    fn inferred_csv_decode_replays_snapshot_after_source_replacement() {
        use std::io::Read;

        let path = std::env::temp_dir().join(format!(
            "mosaic_csv_snapshot_{}_{}.csv",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, "name\noriginal\n").unwrap();
        let options = CsvConvertOptions {
            delimiter: ",".to_string(),
            escape: None,
            quote: "\"".to_string(),
            no_header: false,
            header: None,
            skip_lines: 0,
        };
        let dialect = CsvDialect::from_options(&options).unwrap();
        let source = OpenCsvSource::open(&path, options.skip_lines).unwrap();
        let prepared = prepare_inferred_csv_input(source, &options, dialect).unwrap();

        std::fs::write(&path, "name\nreplacement\n").unwrap();
        let mut replay = String::new();
        prepared
            .reader()
            .unwrap()
            .read_to_string(&mut replay)
            .unwrap();
        assert_eq!(replay, "name\noriginal\n");
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn convert_csv_rejects_wide_positional_input_before_inference() {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "mosaic_csv_width_{}_{}",
            std::process::id(),
            suffix
        ));
        std::fs::create_dir(&dir).unwrap();
        let input = dir.join("wide.csv");
        let first_record = std::iter::repeat_n("1", MAX_CSV_COLUMNS + 1)
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(&input, format!("{first_record}\nshort\n")).unwrap();
        for (index, (no_header, header)) in [(true, None), (false, Some("only".to_string()))]
            .into_iter()
            .enumerate()
        {
            let out = dir.join(format!("wide-{index}.mosaic"));
            let err = convert_csv(
                std::slice::from_ref(&input),
                &out,
                None,
                &[],
                CsvConvertOptions {
                    delimiter: ",".to_string(),
                    escape: None,
                    quote: "\"".to_string(),
                    no_header,
                    header,
                    skip_lines: 0,
                },
                None,
                false,
            )
            .unwrap_err()
            .to_string();
            assert_eq!(
                err,
                format!(
                    "CSV input {} has at least {} columns, exceeds the {} column limit",
                    input.display(),
                    MAX_CSV_COLUMNS + 1,
                    MAX_CSV_COLUMNS
                )
            );
            assert!(!out.exists());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn convert_csv_skip_lines_discards_non_utf8_bytes() {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "mosaic_csv_skip_bytes_{}_{}",
            std::process::id(),
            suffix
        ));
        std::fs::create_dir(&dir).unwrap();
        let input = dir.join("input.csv");
        let out = dir.join("out.mosaic");
        std::fs::write(&input, [b"\xff\n".as_slice(), b"name\nok\n"].concat()).unwrap();

        convert_csv(
            std::slice::from_ref(&input),
            &out,
            None,
            &[],
            CsvConvertOptions {
                delimiter: ",".to_string(),
                escape: None,
                quote: "\"".to_string(),
                no_header: false,
                header: None,
                skip_lines: 1,
            },
            None,
            false,
        )
        .unwrap();

        let reader = open(&out).unwrap();
        assert_eq!(reader.schema().columns.len(), 1);
        assert_eq!(reader.schema().columns[0].name, "name");
        assert_eq!(reader.row_group_num_rows(0).unwrap(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn convert_csv_rejects_wide_later_record_with_explicit_schema() {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "mosaic_csv_later_width_{}_{}",
            std::process::id(),
            suffix
        ));
        std::fs::create_dir(&dir).unwrap();
        let input = dir.join("input.csv");
        let schema = dir.join("schema.avsc");
        let out = dir.join("out.mosaic");
        let wide_record = std::iter::repeat_n("2", MAX_CSV_COLUMNS + 1)
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(&input, format!("1\n{wide_record}\n")).unwrap();
        std::fs::write(
            &schema,
            r#"{"type":"record","name":"T","fields":[{"name":"id","type":"long"}]}"#,
        )
        .unwrap();

        let err = convert_csv(
            std::slice::from_ref(&input),
            &out,
            Some(&schema),
            &[],
            CsvConvertOptions {
                delimiter: ",".to_string(),
                escape: None,
                quote: "\"".to_string(),
                no_header: true,
                header: None,
                skip_lines: 0,
            },
            None,
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains(&format!("exceeds the {} column limit", MAX_CSV_COLUMNS)),
            "{err}"
        );
        assert!(!out.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
