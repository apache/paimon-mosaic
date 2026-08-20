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
use arrow::compute::kernels::cast_utils::{
    parse_decimal, string_to_datetime, Parser as ArrowValueParser,
};
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
        #[arg(short = 'c', long = "column", visible_alias = "columns")]
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
    let open =
        || -> std::io::Result<_> { Ok(std::io::BufReader::new(std::fs::File::open(input)?)) };
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
    mut reader: R,
    schema: &Schema,
    writer: &mut paimon_mosaic_core::writer::MosaicWriter<paimon_mosaic_core::writer::FileSink>,
    rows: &mut usize,
) -> std::io::Result<()> {
    let bad = |e: arrow::error::ArrowError| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
    };
    let mut decoder = arrow::json::ReaderBuilder::new(Arc::new(schema.clone()))
        .build_decoder()
        .map_err(bad)?;
    let mut raw_batch = Vec::new();

    loop {
        loop {
            let buf = reader.fill_buf()?;
            if buf.is_empty() {
                break;
            }
            let available = buf.len();
            let decoded = decoder.decode(buf).map_err(bad)?;
            raw_batch.extend_from_slice(&buf[..decoded]);
            reader.consume(decoded);
            if decoded != available {
                break;
            }
        }

        let batch = decoder.flush().map_err(bad)?;
        let Some(batch) = batch else {
            break;
        };
        validate_json_special_values(&raw_batch, schema, *rows + 1)?;
        raw_batch.clear();
        *rows += batch.num_rows();
        writer.write_batch(&batch)?;
    }
    Ok(())
}

fn schema_needs_json_validation(schema: &Schema) -> bool {
    schema.fields().iter().any(|field| {
        data_type_has_local_timestamp(field.data_type()) || data_type_has_decimal(field.data_type())
    })
}

fn data_type_has_local_timestamp(data_type: &DataType) -> bool {
    match data_type {
        DataType::Timestamp(_, None) => true,
        DataType::List(field) | DataType::Map(field, _) => {
            data_type_has_local_timestamp(field.data_type())
        }
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| data_type_has_local_timestamp(field.data_type())),
        _ => false,
    }
}

fn data_type_has_decimal(data_type: &DataType) -> bool {
    match data_type {
        DataType::Decimal128(_, _) => true,
        DataType::List(field) | DataType::Map(field, _) => data_type_has_decimal(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| data_type_has_decimal(field.data_type())),
        _ => false,
    }
}

fn data_type_needs_json_validation(data_type: &DataType) -> bool {
    data_type_has_local_timestamp(data_type) || data_type_has_decimal(data_type)
}

fn validate_json_special_values(
    raw: &[u8],
    schema: &Schema,
    first_record: usize,
) -> std::io::Result<()> {
    // Borrow only the raw values of relevant fields. Unrelated values are
    // skipped without constructing a second set of Arrow arrays or a Value tree.
    let fields: std::collections::HashMap<String, DataType> = schema
        .fields()
        .iter()
        .filter(|field| data_type_needs_json_validation(field.data_type()))
        .map(|field| (field.name().clone(), field.data_type().clone()))
        .collect();
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    let mut record = first_record;
    loop {
        let seed = JsonSpecialRecordSeed {
            fields: &fields,
            record,
        };
        match seed.deserialize(&mut deserializer) {
            Ok(()) => record += 1,
            Err(e) if e.is_eof() => break,
            Err(e) => {
                return Err(invalid_schema(format!("invalid JSON record {record}: {e}")));
            }
        }
    }
    Ok(())
}

struct JsonSpecialRecordSeed<'a> {
    fields: &'a std::collections::HashMap<String, DataType>,
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
    fields: &'a std::collections::HashMap<String, DataType>,
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
        while let Some(name) = map.next_key::<std::borrow::Cow<'de, str>>()? {
            if let Some(data_type) = self.fields.get(name.as_ref()) {
                let raw: &RawValue = map.next_value()?;
                validate_json_special_value(raw, data_type, name.as_ref(), self.record)
                    .map_err(M::Error::custom)?;
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        Ok(())
    }
}

struct JsonSpecialMapSeed<'a> {
    value_type: &'a DataType,
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
            value_type: self.value_type,
            path: self.path,
            record: self.record,
        })
    }
}

struct JsonSpecialMapVisitor<'a> {
    value_type: &'a DataType,
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
            validate_json_special_value(raw, self.value_type, self.path, self.record)
                .map_err(M::Error::custom)?;
        }
        Ok(())
    }
}

fn validate_json_special_value(
    raw: &RawValue,
    data_type: &DataType,
    path: &str,
    record: usize,
) -> std::io::Result<()> {
    if raw.get() == "null" || !data_type_needs_json_validation(data_type) {
        return Ok(());
    }
    match data_type {
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
                validate_json_special_value(value, field.data_type(), &child_path, record)?;
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
                value_type: value_field.data_type(),
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
                    validate_json_special_value(value, field.data_type(), &child_path, record)?;
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
    use arrow::error::ArrowError;
    let bad = |e: ArrowError| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string());
    let format = csv_format(&options)?;
    let explicit_schema = schema.map(load_convert_schema).transpose()?;
    let has_explicit_schema = explicit_schema.is_some();
    let mut inferred_input_schemas = vec![None; inputs.len()];
    let schema = match explicit_schema {
        Some(schema) => schema,
        None => {
            let mut inferred: Option<Schema> = None;
            for (index, input) in inputs.iter().enumerate() {
                let (schema, rows) = format
                    .infer_schema(open_csv(input, options.skip_lines)?, None)
                    .map_err(bad)?;
                // A shard with no data rows has nothing to infer from; it is
                // skipped when reading too.
                if rows == 0 || schema.fields().is_empty() {
                    continue;
                }
                let schema = csv_schema_with_csv_names(schema, &options)?;
                inferred_input_schemas[index] = Some(schema.clone());
                inferred = Some(match inferred.take() {
                    Some(prev) => merge_csv_inferred_schema(prev, schema, input)?,
                    None => schema,
                });
            }
            let schema = inferred.ok_or_else(|| {
                invalid_schema("no CSV data to infer a schema from; provide --schema")
            })?;
            apply_required_fields(csv_schema_with_null_fallback(schema), required_fields)?
        }
    };
    reject_csv_unsupported_fields(&schema)?;
    let schema_index = csv_schema_index(&schema);
    let mixed_float_fields = mixed_csv_float_fields(&schema, &inferred_input_schemas);
    write_mosaic(out, overwrite, &schema, stats, |writer, rows| {
        for input in inputs {
            if has_explicit_schema {
                write_explicit_schema_csv_input(
                    writer,
                    rows,
                    input,
                    &schema,
                    &schema_index,
                    &options,
                )?;
                continue;
            }
            let layout = csv_input_layout(input, &options)?;
            // Empty and header-only shards contribute no rows, so their header
            // cannot affect the schema inferred from non-empty inputs.
            if !layout.has_records {
                continue;
            }
            let reader_schema =
                csv_reader_schema(&schema, &schema_index, &mixed_float_fields, &layout);
            let source_mapping = csv_output_mapping(&schema, &schema_index, &layout);
            validate_csv_mapping(&schema, &layout, &source_mapping, input)?;
            let (projection, mapping) = csv_projection(&source_mapping);
            let batch_size = csv_batch_size(reader_schema.fields().len());
            let reader = arrow::csv::ReaderBuilder::new(Arc::new(reader_schema))
                .with_format(format.clone().with_truncated_rows(true))
                .with_batch_size(batch_size)
                .with_projection(projection)
                .build(open_csv(input, options.skip_lines)?)
                .map_err(bad)?;
            for batch in reader {
                let batch = batch.map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                })?;
                let batch = align_csv_batch_to_schema(batch, &schema, &mapping, input)?;
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
    let delimiter = parse_csv_byte(&options.delimiter, "delimiter")?;
    let escape = parse_optional_csv_byte(options.escape.as_deref(), "escape")?;
    let quote = parse_csv_byte(&options.quote, "quote")?;
    let format = arrow::csv::reader::Format::default()
        .with_header(!options.no_header && options.header.is_none())
        .with_delimiter(delimiter)
        .with_quote(quote);
    Ok(match escape {
        Some(escape) => format.with_escape(escape),
        None => format,
    })
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

fn open_csv(path: &Path, skip_lines: usize) -> std::io::Result<std::io::BufReader<std::fs::File>> {
    use std::io::BufRead;
    let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut line = String::new();
    for _ in 0..skip_lines {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
    }
    Ok(reader)
}

struct CsvInputLayout {
    header: Option<Vec<String>>,
    columns: usize,
    has_records: bool,
}

struct CsvInput {
    reader: csv::Reader<std::io::BufReader<std::fs::File>>,
    layout: CsvInputLayout,
    first_record: Option<csv::StringRecord>,
}

const DEFAULT_CSV_BATCH_SIZE: usize = 1024;
// Arrow's CSV RecordDecoder reserves roughly one data byte range and one
// offset per cell before projection is applied. Keep that eager allocation
// bounded for very wide records by reducing the number of rows per batch.
const TARGET_CSV_DECODE_CELLS: usize = 64 * 1024;

fn csv_batch_size(columns: usize) -> usize {
    if columns == 0 {
        return DEFAULT_CSV_BATCH_SIZE;
    }
    (TARGET_CSV_DECODE_CELLS / columns).clamp(1, DEFAULT_CSV_BATCH_SIZE)
}

fn explicit_csv_row_cells(source_columns: usize, output_columns: usize) -> usize {
    source_columns.max(output_columns)
}

fn csv_input_layout(path: &Path, options: &CsvConvertOptions) -> std::io::Result<CsvInputLayout> {
    Ok(open_csv_input(path, options)?.layout)
}

fn open_csv_input(path: &Path, options: &CsvConvertOptions) -> std::io::Result<CsvInput> {
    let delimiter = parse_csv_byte(&options.delimiter, "delimiter")?;
    let escape = parse_optional_csv_byte(options.escape.as_deref(), "escape")?;
    let quote = parse_csv_byte(&options.quote, "quote")?;
    let mut builder = csv::ReaderBuilder::new();
    builder
        .has_headers(false)
        .flexible(true)
        .delimiter(delimiter)
        .quote(quote)
        .escape(escape);
    let mut reader = builder.from_reader(open_csv(path, options.skip_lines)?);
    let file_header = options.header.is_none() && !options.no_header;
    let header = if let Some(header) = &options.header {
        Some(parse_csv_header(header, options)?)
    } else if options.no_header {
        None
    } else {
        let mut record = csv::StringRecord::new();
        if reader
            .read_record(&mut record)
            .map_err(|e| invalid_schema(format!("invalid CSV header: {e}")))?
        {
            Some(record.iter().map(ToString::to_string).collect())
        } else {
            Some(Vec::new())
        }
    };
    let columns = header.as_ref().map_or(0, Vec::len);
    let mut first_record = csv::StringRecord::new();
    let has_records = reader
        .read_record(&mut first_record)
        .map_err(|e| invalid_schema(format!("invalid CSV record: {e}")))?;
    if has_records && file_header {
        validate_csv_header_names(header.as_ref().unwrap())?;
    }
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
    input: &Path,
    schema: &Schema,
    schema_index: &std::collections::HashMap<&str, usize>,
    options: &CsvConvertOptions,
) -> std::io::Result<()> {
    let mut input_reader = open_csv_input(input, options)?;
    if !input_reader.layout.has_records {
        return Ok(());
    }
    let source_mapping = csv_output_mapping(schema, schema_index, &input_reader.layout);
    validate_csv_mapping(schema, &input_reader.layout, &source_mapping, input)?;

    let first = input_reader.first_record.take().into_iter().map(Ok);
    let rest = std::iter::from_fn(|| {
        let mut record = csv::StringRecord::new();
        match input_reader.reader.read_record(&mut record) {
            Ok(true) => Some(Ok(record)),
            Ok(false) => None,
            Err(e) => Some(Err(invalid_schema(format!(
                "invalid CSV record in {}: {e}",
                input.display()
            )))),
        }
    });
    for_each_explicit_csv_batch(
        first.chain(rest),
        input_reader.layout.columns,
        schema,
        |records| write_explicit_csv_records(writer, rows, schema, &source_mapping, records),
    )
}

fn for_each_explicit_csv_batch<I, F>(
    records: I,
    source_columns: usize,
    schema: &Schema,
    mut write: F,
) -> std::io::Result<()>
where
    I: IntoIterator<Item = std::io::Result<csv::StringRecord>>,
    F: FnMut(&[csv::StringRecord]) -> std::io::Result<()>,
{
    let output_columns = schema.fields().len();
    let mut batch = Vec::with_capacity(csv_batch_size(source_columns.max(output_columns)));
    let mut cells: usize = 0;
    for record in records {
        let record = record?;
        let row_cells = explicit_csv_row_cells(record.len(), output_columns);
        if !batch.is_empty()
            && (batch.len() >= DEFAULT_CSV_BATCH_SIZE
                || cells.saturating_add(row_cells) > TARGET_CSV_DECODE_CELLS)
        {
            write(&batch)?;
            batch.clear();
            cells = 0;
        }
        cells = cells.saturating_add(row_cells);
        batch.push(record);
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
        DataType::Time32(TimeUnit::Millisecond) => {
            csv_primitive_column::<Time32MillisecondType>(records, source, field)
        }
        DataType::Timestamp(unit, timezone) => {
            csv_timestamp_column(records, source, field, unit, timezone.clone())
        }
        DataType::Decimal128(precision, scale) => {
            let values = records
                .iter()
                .map(|record| {
                    csv_record_value(record, source)
                        .map(|value| {
                            let parsed =
                                parse_decimal::<Decimal128Type>(value, *precision, *scale)
                                    .map_err(|_| csv_value_parse_error(record, field, value))?;
                            if !decimal_is_exact_at_scale(value, *scale) {
                                return Err(invalid_schema(format!(
                                    "decimal value '{}' for CSV field '{}' at line {} cannot be represented exactly with scale {scale}",
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
        DataType::Utf8 => Ok(Arc::new(
            records
                .iter()
                .map(|record| csv_record_value(record, source))
                .collect::<StringArray>(),
        )),
        data_type => Err(invalid_schema(format!(
            "CSV conversion does not support field '{}' with type {data_type}",
            fmt::safe(field.name())
        ))),
    }
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
    let values = records
        .iter()
        .map(|record| {
            let Some(value) = csv_record_value(record, source) else {
                return Ok(None);
            };
            if timezone.is_none() && timestamp_has_explicit_timezone(value) {
                return Err(invalid_schema(format!(
                    "CSV field '{}' at line {} must not include a timezone for a local timestamp",
                    fmt::safe(field.name()),
                    record
                        .position()
                        .map(|position| position.line().to_string())
                        .unwrap_or_else(|| "unknown".to_string())
                )));
            }
            let datetime = string_to_datetime(&parser_timezone, value)
                .map_err(|_| csv_value_parse_error(record, field, value))?;
            let timestamp = match unit {
                TimeUnit::Millisecond => datetime.timestamp_millis(),
                TimeUnit::Microsecond => datetime.timestamp_micros(),
                TimeUnit::Nanosecond => datetime
                    .timestamp_nanos_opt()
                    .ok_or_else(|| csv_value_parse_error(record, field, value))?,
                unit => {
                    return Err(invalid_schema(format!(
                        "CSV conversion does not support timestamp unit {unit:?}"
                    )));
                }
            };
            Ok(Some(timestamp))
        })
        .collect::<std::io::Result<Vec<_>>>()?;
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
    let parsed = parse_decimal::<Decimal128Type>(value, precision, scale)?;
    if !decimal_is_exact_at_scale(value, scale) {
        return Err(arrow::error::ArrowError::ParseError(format!(
            "cannot be represented exactly with scale {scale}"
        )));
    }
    Ok(parsed)
}

fn decimal_is_exact_at_scale(value: &str, scale: i8) -> bool {
    let value = value
        .strip_prefix('+')
        .or_else(|| value.strip_prefix('-'))
        .unwrap_or(value);
    let exponent_index = value.find(['e', 'E']);
    let (mantissa, exponent) = match exponent_index {
        Some(index) => (&value[..index], parse_decimal_exponent(&value[index + 1..])),
        None => (value, 0),
    };
    let fractional_digits = mantissa
        .split_once('.')
        .map_or(0_i64, |(_, fraction)| fraction.len() as i64);
    let digits: Vec<u8> = mantissa.bytes().filter(u8::is_ascii_digit).collect();
    if digits.is_empty() {
        return false;
    }
    if digits.iter().all(|digit| *digit == b'0') {
        return true;
    }
    let discarded = fractional_digits
        .saturating_sub(exponent)
        .saturating_sub(i64::from(scale));
    if discarded <= 0 {
        return true;
    }
    let discarded = usize::try_from(discarded).unwrap_or(usize::MAX);
    discarded <= digits.len()
        && digits[digits.len() - discarded..]
            .iter()
            .all(|digit| *digit == b'0')
}

fn parse_decimal_exponent(value: &str) -> i64 {
    let (negative, digits) = match value.as_bytes().first() {
        Some(b'-') => (true, &value[1..]),
        Some(b'+') => (false, &value[1..]),
        _ => (false, value),
    };
    let exponent = digits.bytes().fold(0_i64, |value, digit| {
        value
            .saturating_mul(10)
            .saturating_add(i64::from(digit.saturating_sub(b'0')))
    });
    if negative {
        exponent.saturating_neg()
    } else {
        exponent
    }
}

fn timestamp_has_explicit_timezone(value: &str) -> bool {
    let bytes = value.trim().as_bytes();
    if bytes.len() <= 10 {
        return false;
    }
    let mut timezone_start = if bytes.get(13) == Some(&b':') && bytes.get(16) == Some(&b':') {
        19
    } else {
        17
    };
    if bytes.get(timezone_start) == Some(&b'.') {
        timezone_start += 1;
        while bytes.get(timezone_start).is_some_and(u8::is_ascii_digit) {
            timezone_start += 1;
        }
    }
    timezone_start < bytes.len()
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

fn mixed_csv_float_fields(
    output_schema: &Schema,
    input_schemas: &[Option<Schema>],
) -> std::collections::HashSet<String> {
    // These fields are Float64 only because different shards inferred Int64
    // and Float64. Read their raw text in every shard so integer-looking values
    // cannot be rounded before the exactness check.
    output_schema
        .fields()
        .iter()
        .filter(|field| matches!(field.data_type(), DataType::Float64))
        .filter_map(|field| {
            let mut saw_int64 = false;
            let mut saw_float64 = false;
            for source in input_schemas.iter().flatten() {
                if let Ok(source) = source.field_with_name(field.name()) {
                    saw_int64 |= matches!(source.data_type(), DataType::Int64);
                    saw_float64 |= matches!(source.data_type(), DataType::Float64);
                }
            }
            (saw_int64 && saw_float64).then(|| field.name().clone())
        })
        .collect()
}

fn csv_reader_schema(
    output_schema: &Schema,
    schema_index: &std::collections::HashMap<&str, usize>,
    mixed_float_fields: &std::collections::HashSet<String>,
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
                let data_type = if mixed_float_fields.contains(output_field.name()) {
                    DataType::Utf8
                } else {
                    output_field.data_type().clone()
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
                field.name(),
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
            Some(index)
                if batch.column(*index).data_type() == &DataType::Utf8
                    && field.data_type() == &DataType::Float64 =>
            {
                parse_mixed_csv_float64(batch.column(*index), field, input)
            }
            Some(index) => Ok(batch.column(*index).clone()),
            None => Ok(new_null_array(field.data_type(), batch.num_rows())),
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    RecordBatch::try_new(Arc::new(schema.clone()), columns)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
}

fn parse_mixed_csv_float64(
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
                .ok_or_else(|| csv_mixed_float_parse_error(value, field, input))?;
            // Fractional Float64 values retain normal floating-point semantics.
            // Integral decimal tokens must round-trip exactly.
            if decimal_is_exact_at_scale(value, 0) {
                let exact = parse_decimal::<Decimal128Type>(value, 38, 0)
                    .map_err(|_| csv_mixed_float_parse_error(value, field, input))?;
                if !promoted.is_finite() || promoted as i128 != exact {
                    return Err(csv_mixed_float_parse_error(value, field, input));
                }
            }
            Ok(Some(promoted))
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    Ok(Arc::new(
        values.into_iter().collect::<PrimitiveArray<Float64Type>>(),
    ))
}

fn csv_mixed_float_parse_error(value: &str, field: &Field, input: &Path) -> std::io::Error {
    invalid_schema(format!(
        "numeric value '{}' in CSV field '{}' of {} cannot be represented exactly as Float64 during mixed Int64/Float64 schema promotion",
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
                field.name(),
            )));
        }
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
        _ => return Err(csv_schema_mismatch(input)),
    };
    Ok(field)
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

fn load_convert_schema(path: &Path) -> std::io::Result<Schema> {
    let text = std::fs::read_to_string(path)?;
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
        if !is_valid_avro_fullname(namespace) {
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
        let (data_type, nullable) = parse_avro_type(avro_type)
            .map_err(|e| invalid_schema(format!("Avro field '{name}': {e}")))?;
        fields.push(Field::new(name, data_type, nullable));
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

fn parse_avro_type(value: &Value) -> Result<(DataType, bool), String> {
    match value {
        Value::String(name) => parse_avro_named_type(name, None).map(|dt| (dt, false)),
        Value::Object(obj) => {
            let name = obj
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| "Avro type object must contain a string type".to_string())?;
            parse_avro_named_type(name, Some(value)).map(|dt| (dt, false))
        }
        Value::Array(types) => parse_avro_union(types),
        _ => Err("Avro type must be a string, object, or union array".to_string()),
    }
}

fn parse_avro_union(types: &[Value]) -> Result<(DataType, bool), String> {
    let mut has_null = false;
    let mut non_null = None;
    for ty in types {
        if matches!(ty, Value::String(s) if s == "null") {
            has_null = true;
            continue;
        }
        let (dt, nullable) = parse_avro_type(ty)?;
        if nullable {
            return Err("nested nullable unions are not supported".to_string());
        }
        if non_null.replace(dt).is_some() {
            return Err("Avro unions with multiple non-null types are not supported".to_string());
        }
    }
    let dt = non_null.ok_or_else(|| "pure null Avro fields are not supported".to_string())?;
    Ok((dt, has_null))
}

fn parse_avro_named_type(name: &str, full_type: Option<&Value>) -> Result<DataType, String> {
    let logical_type = full_type
        .and_then(|value| value.get("logicalType"))
        .and_then(Value::as_str);
    if let Some(logical_type) = logical_type {
        if let Some(result) = parse_avro_logical_type(name, logical_type, full_type.unwrap()) {
            return result;
        }
    }
    match name {
        "boolean" => Ok(DataType::Boolean),
        "int" => Ok(DataType::Int32),
        "long" => Ok(DataType::Int64),
        "float" => Ok(DataType::Float32),
        "double" => Ok(DataType::Float64),
        "string" => Ok(DataType::Utf8),
        "bytes" => Ok(DataType::Binary),
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

fn parse_avro_array(full_type: &Value) -> Result<DataType, String> {
    let items = full_type
        .get("items")
        .ok_or_else(|| "Avro array type must contain items".to_string())?;
    let (data_type, nullable) = parse_avro_type(items)?;
    Ok(DataType::List(Arc::new(Field::new(
        "item", data_type, nullable,
    ))))
}

fn parse_avro_map(full_type: &Value) -> Result<DataType, String> {
    let values = full_type
        .get("values")
        .ok_or_else(|| "Avro map type must contain values".to_string())?;
    let (value_type, value_nullable) = parse_avro_type(values)?;
    let entries = Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("keys", DataType::Utf8, false),
            Field::new("values", value_type, value_nullable),
        ])),
        false,
    );
    Ok(DataType::Map(Arc::new(entries), false))
}

fn parse_avro_logical_type(
    physical_type: &str,
    logical_type: &str,
    full_type: &Value,
) -> Option<Result<DataType, String>> {
    Some(match (physical_type, logical_type) {
        ("int", "date") => Ok(DataType::Date32),
        ("int", "time-millis") => Ok(DataType::Time32(TimeUnit::Millisecond)),
        ("long", "timestamp-millis") => Ok(DataType::Timestamp(
            TimeUnit::Millisecond,
            Some("+00:00".into()),
        )),
        ("long", "timestamp-micros") => Ok(DataType::Timestamp(
            TimeUnit::Microsecond,
            Some("+00:00".into()),
        )),
        ("long", "timestamp-nanos") => Ok(DataType::Timestamp(
            TimeUnit::Nanosecond,
            Some("+00:00".into()),
        )),
        ("long", "local-timestamp-millis") => Ok(DataType::Timestamp(TimeUnit::Millisecond, None)),
        ("long", "local-timestamp-micros") => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
        ("long", "local-timestamp-nanos") => Ok(DataType::Timestamp(TimeUnit::Nanosecond, None)),
        ("bytes" | "fixed", "decimal") => parse_avro_decimal(full_type),
        ("string", "uuid") => Ok(DataType::Utf8),
        _ => return None,
    })
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
        for_each_explicit_csv_batch(records, 1, &schema, |batch| {
            batch_sizes.push(batch.len());
            Ok(())
        })
        .unwrap();
        assert_eq!(batch_sizes, [16, 16, 1]);
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
        let reader_schema =
            csv_reader_schema(&schema, &index, &std::collections::HashSet::new(), &layout);
        let mapping = csv_output_mapping(&schema, &index, &layout);
        assert_eq!(reader_schema.fields()[0].name(), "field_0");
        assert_eq!(reader_schema.fields()[4095].name(), "field_4095");
        assert_eq!(mapping[0], Some(4095));
        assert_eq!(mapping[4095], Some(0));
    }

    #[test]
    fn csv_int64_to_float64_promotion_requires_exact_round_trip() {
        let field = Field::new("value", DataType::Float64, true);
        for value in [
            "-9007199254740992",
            "9007199254740992",
            "9007199254740994",
            "-9223372036854775808",
            "1.5",
        ] {
            let input: ArrayRef = Arc::new(StringArray::from(vec![value]));
            let output = parse_mixed_csv_float64(&input, &field, Path::new("safe.csv")).unwrap();
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
        let output = parse_mixed_csv_float64(&input, &field, Path::new("non-finite.csv")).unwrap();
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
            "9007199254740993.0",
            "9.007199254740993e15",
            "9223372036854775807",
        ] {
            let input: ArrayRef = Arc::new(StringArray::from(vec![value]));
            let err = parse_mixed_csv_float64(&input, &field, Path::new("lossy.csv"))
                .unwrap_err()
                .to_string();
            assert!(err.contains(value), "{err}");
        }
    }

    #[test]
    fn mixed_csv_float_fields_require_both_source_types() {
        let output = Schema::new(vec![
            Field::new("mixed", DataType::Float64, true),
            Field::new("float_only", DataType::Float64, true),
        ]);
        let inputs = vec![
            Some(Schema::new(vec![
                Field::new("mixed", DataType::Int64, true),
                Field::new("float_only", DataType::Float64, true),
            ])),
            Some(Schema::new(vec![
                Field::new("mixed", DataType::Float64, true),
                Field::new("float_only", DataType::Float64, true),
            ])),
        ];
        assert_eq!(
            mixed_csv_float_fields(&output, &inputs),
            std::collections::HashSet::from(["mixed".to_string()])
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
    fn local_timestamp_timezone_detection_ignores_fractional_precision() {
        for value in [
            "2026-08-20T12:34:56Z",
            "2026-08-20T12:34:56+08:00",
            "2026-08-20T12:34:56.123-08:00",
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
}
