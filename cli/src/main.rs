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

use arrow::array::{new_null_array, ArrayRef, RecordBatch};
use arrow::datatypes::{DataType, Field, Fields, Schema, TimeUnit};
use clap::{Parser, Subcommand};
use paimon_mosaic_core::reader::{MosaicReader, ReaderAccess};
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
        /// Avro record schema file to use instead of inferring one.
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
        /// Avro record schema file to use instead of inferring one (scalar fields only).
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
        #[arg(long)]
        no_header: bool,
        /// Line to use as a header. Must match the CSV settings.
        #[arg(long)]
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
    let schema = match explicit_schema {
        Some(schema) => schema,
        None if columns.is_empty() => arrow::json::reader::infer_json_schema(&mut open()?, None)
            .map(|(schema, _)| schema)
            .map_err(bad)?,
        None => infer_projected_json_schema(open()?, &columns).map_err(bad)?,
    };
    let schema = project_convert_schema(schema, &columns)?;
    reject_null_inferred_fields(&schema)?;
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
    // Arrow schema inference rejects ragged rows, so once inference succeeds
    // every row is as wide as the header; only an explicit schema needs the
    // row widths scanned to size the reader for extra CSV columns.
    let scan_widths = explicit_schema.is_some();
    let schema = match explicit_schema {
        Some(schema) => schema,
        None => {
            let mut inferred: Option<Schema> = None;
            for input in inputs {
                let (schema, rows) = format
                    .infer_schema(open_csv(input, options.skip_lines)?, None)
                    .map_err(bad)?;
                // A shard with no data rows has nothing to infer from; it is
                // skipped when reading too.
                if rows == 0 || schema.fields().is_empty() {
                    continue;
                }
                let schema = csv_schema_with_csv_names(schema, &options)?;
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
    write_mosaic(out, overwrite, &schema, stats, |writer, rows| {
        for input in inputs {
            let layout = csv_input_layout(input, &options, scan_widths)?;
            // An empty file read in header mode yields an empty header: nothing
            // to read, and no header names to complain about.
            if layout.header.as_ref().is_some_and(|h| h.is_empty()) {
                continue;
            }
            let reader_schema = csv_reader_schema(&schema, &layout);
            let source_mapping = csv_output_mapping(&schema, &layout);
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
                let batch = align_csv_batch_to_schema(batch, &schema, &mapping)?;
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
                field.name()
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

fn csv_input_layout(
    path: &Path,
    options: &CsvConvertOptions,
    scan_widths: bool,
) -> std::io::Result<CsvInputLayout> {
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
    let mut records = reader.records();
    let header = if let Some(header) = &options.header {
        Some(parse_csv_header(header, options)?)
    } else if options.no_header {
        None
    } else {
        match records.next() {
            Some(record) => {
                let record =
                    record.map_err(|e| invalid_schema(format!("invalid CSV header: {e}")))?;
                let header: Vec<String> = record.iter().map(ToString::to_string).collect();
                validate_csv_header_names(&header)?;
                Some(header)
            }
            None => Some(Vec::new()),
        }
    };
    let mut columns = header.as_ref().map_or(0, Vec::len);
    if scan_widths {
        for record in records {
            let record = record.map_err(|e| invalid_schema(format!("invalid CSV record: {e}")))?;
            columns = columns.max(record.len());
        }
    }
    Ok(CsvInputLayout { header, columns })
}

fn csv_reader_schema(output_schema: &Schema, layout: &CsvInputLayout) -> Schema {
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
                    .and_then(|name| output_schema.index_of(name).ok())
            } else {
                (i < output_schema.fields().len()).then_some(i)
            };
            if let Some(source) = source {
                // Read as nullable: not-null enforcement happens when the batch
                // is re-attached to the output schema, where the error carries
                // the real column name rather than a positional one.
                output_schema.fields()[source]
                    .as_ref()
                    .clone()
                    .with_name(format!("field_{i}"))
                    .with_nullable(true)
            } else {
                Field::new(format!("field_{i}"), DataType::Utf8, true)
            }
        })
        .collect();
    Schema::new(fields)
}

fn csv_output_mapping(output_schema: &Schema, layout: &CsvInputLayout) -> Vec<Option<usize>> {
    if let Some(header) = &layout.header {
        let mut mapping = vec![None; output_schema.fields().len()];
        for (csv_index, name) in header.iter().enumerate() {
            if let Ok(field_index) = output_schema.index_of(name) {
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
) -> std::io::Result<RecordBatch> {
    let columns: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .zip(mapping)
        .map(|(field, index)| match index {
            Some(index) => batch.column(*index).clone(),
            None => new_null_array(field.data_type(), batch.num_rows()),
        })
        .collect();
    RecordBatch::try_new(Arc::new(schema.clone()), columns)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
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
                "duplicate CSV header field '{name}'"
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
    let fields: Vec<Field> = prev
        .fields()
        .iter()
        .zip(next.fields())
        .map(|(left, right)| merge_csv_inferred_field(left.as_ref(), right.as_ref(), input))
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
    let avro_fields = obj
        .get("fields")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_schema("Avro record schema must contain a fields array"))?;
    let mut fields = Vec::with_capacity(avro_fields.len());
    for field in avro_fields {
        let field_obj = field
            .as_object()
            .ok_or_else(|| invalid_schema("Avro field must be an object"))?;
        let name = field_obj
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_schema("Avro field must contain a string name"))?;
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
        return parse_avro_logical_type(name, logical_type, full_type.unwrap());
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
) -> Result<DataType, String> {
    match (physical_type, logical_type) {
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
        _ => Err(format!(
            "unsupported Avro logical type '{logical_type}' on '{physical_type}'"
        )),
    }
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
    fn csv_projection_remaps_output_columns() {
        let (projection, mapping) = csv_projection(&[Some(5), None, Some(1)]);
        assert_eq!(projection, [5, 1]);
        assert_eq!(mapping, [Some(0), None, Some(1)]);
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
}
