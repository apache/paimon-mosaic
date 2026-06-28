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

use arrow::array::{Array, RecordBatch};
use paimon_mosaic_core::values::Value;

/// Render a stats min/max [`Value`] to a short, human-readable string.
pub fn render_value(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::TinyInt(x) => x.to_string(),
        Value::SmallInt(x) => x.to_string(),
        Value::Integer(x) => x.to_string(),
        Value::BigInt(x) => x.to_string(),
        Value::Float(x) => x.to_string(),
        Value::Double(x) => x.to_string(),
        Value::Date(x) => format!("{} (epoch-day)", x),
        Value::Time(x) => format!("{} (ms)", x),
        Value::String(b) => safe(&String::from_utf8_lossy(b)),
        Value::Bytes(b) | Value::DecimalLarge(b) => format!("0x{}", hex(b)),
        Value::DecimalCompact(x) => x.to_string(),
        Value::TimestampMillis(x) => format!("{} (ms)", x),
        Value::TimestampMicros(x) => format!("{} (us)", x),
        Value::TimestampNanos {
            millis,
            nanos_of_milli,
        } => {
            format!("{}ms+{}ns", millis, nanos_of_milli)
        }
    }
}

/// Render a stats min/max [`Value`] for JSON: same as [`render_value`] but
/// without the human-only unit suffixes (`(epoch-day)`, `(ms)`, `ms+ns`), so
/// the value stays machine-parseable. Bytes/decimal keep hex; nanos collapse to
/// a single nanosecond count.
pub fn render_json(v: &Value) -> String {
    match v {
        Value::Date(x) | Value::Time(x) => x.to_string(),
        Value::TimestampMillis(x) | Value::TimestampMicros(x) => x.to_string(),
        Value::TimestampNanos {
            millis,
            nanos_of_milli,
        } => (*millis as i128 * 1_000_000 + *nanos_of_milli as i128).to_string(),
        _ => render_value(v),
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

/// Strip control chars so a crafted file can't inject ANSI escapes into the
/// inspector's terminal. Use on any file-derived string sent to text output;
/// JSON output is escaped by serde (`jsonout`)/the Arrow writer instead.
pub fn safe(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_control() { '\u{fffd}' } else { c })
        .collect()
}

/// Human-readable encoding name.
pub fn encoding_name(e: paimon_mosaic_core::reader::Encoding) -> String {
    use paimon_mosaic_core::reader::Encoding::*;
    match e {
        Plain => "plain".into(),
        Const => "const".into(),
        Dict => "dict".into(),
        AllNull => "all_null".into(),
        Other(c) => format!("enc{c}"),
        _ => "other".into(),
    }
}

/// Human-readable bucket kind name. Returns `String` to match [`encoding_name`]
/// (whose `enc{c}` fallback needs an allocation), so callers handle one type.
pub fn bucket_kind(k: paimon_mosaic_core::reader::BucketKind) -> String {
    use paimon_mosaic_core::reader::BucketKind::*;
    match k {
        Empty => "empty".into(),
        Monolithic => "monolithic".into(),
        Paged => "paged".into(),
        _ => "unknown".into(),
    }
}

/// Compression ratio suffix like `" (uncompressed 1024 B, 2.50x)"`, or empty
/// when the uncompressed size is unknown (paged buckets don't record it).
pub fn ratio(compressed: usize, uncompressed: usize) -> String {
    if uncompressed == 0 || compressed == 0 {
        return String::new();
    }
    format!(
        " (uncompressed {} B, {:.2}x)",
        uncompressed,
        uncompressed as f64 / compressed as f64
    )
}

/// Pretty-print a slice of record batches as an aligned ASCII table.
pub fn pretty_table(batches: &[RecordBatch], max_rows: usize) -> String {
    if batches.is_empty() {
        return String::new();
    }
    let schema = batches[0].schema();
    let headers: Vec<String> = schema.fields().iter().map(|f| safe(f.name())).collect();
    let ncols = headers.len();

    let mut rows: Vec<Vec<String>> = Vec::new();
    'outer: for batch in batches {
        // Downcast each column once per batch, not once per cell.
        let fmts: Vec<_> = (0..ncols)
            .map(|c| col_formatter(batch.column(c).as_ref()))
            .collect();
        for r in 0..batch.num_rows() {
            if rows.len() >= max_rows {
                break 'outer;
            }
            rows.push((0..ncols).map(|c| fmts[c](r)).collect());
        }
    }

    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for row in &rows {
        for (i, v) in row.iter().enumerate() {
            widths[i] = widths[i].max(v.chars().count());
        }
    }

    let sep = |out: &mut String| {
        out.push('+');
        for w in &widths {
            out.push_str(&"-".repeat(w + 2));
            out.push('+');
        }
        out.push('\n');
    };
    let line = |out: &mut String, cells: &[String]| {
        out.push('|');
        for (i, c) in cells.iter().enumerate() {
            out.push_str(&format!(" {:<w$} |", c, w = widths[i]));
        }
        out.push('\n');
    };

    let mut out = String::new();
    sep(&mut out);
    line(&mut out, &headers);
    sep(&mut out);
    for row in &rows {
        line(&mut out, row);
    }
    sep(&mut out);
    out
}

/// Render up to `max_rows` as newline-delimited JSON objects.
pub fn ndjson(batches: &[RecordBatch], max_rows: usize) -> std::io::Result<String> {
    use std::io;
    // Use Arrow's JSON writer so every type the reader supports renders as valid
    // JSON (NaN/Infinity become null); explicit nulls keep absent fields visible.
    if batches.is_empty() {
        return Ok(String::new());
    }
    let mut taken: Vec<RecordBatch> = Vec::new();
    let mut got = 0usize;
    for b in batches {
        if got >= max_rows {
            break;
        }
        let n = b.num_rows().min(max_rows - got);
        taken.push(b.slice(0, n));
        got += n;
    }
    let buf = Vec::new();
    let mut w = arrow::json::WriterBuilder::new()
        .with_explicit_nulls(true)
        .build::<_, arrow::json::writer::LineDelimited>(buf);
    for b in &taken {
        w.write(b).map_err(|e| io::Error::other(e.to_string()))?;
    }
    w.finish().map_err(|e| io::Error::other(e.to_string()))?;
    String::from_utf8(w.into_inner()).map_err(|e| io::Error::other(e.to_string()))
}

/// Build a per-column cell formatter, downcasting once instead of per cell.
/// Nulls render empty; control chars are stripped so a crafted file can't inject
/// ANSI into the terminal (JSON is escaped by the writer instead).
fn col_formatter(arr: &dyn Array) -> Box<dyn Fn(usize) -> String + '_> {
    use arrow::array::*;
    use arrow::datatypes::DataType::*;
    macro_rules! d {
        ($ty:ty) => {{
            let a = arr.as_any().downcast_ref::<$ty>().unwrap();
            Box::new(move |r| if a.is_null(r) { String::new() } else { a.value(r).to_string() })
        }};
    }
    match arr.data_type() {
        Boolean => d!(BooleanArray),
        Int8 => d!(Int8Array),
        Int16 => d!(Int16Array),
        Int32 => d!(Int32Array),
        Int64 => d!(Int64Array),
        Float32 => d!(Float32Array),
        Float64 => d!(Float64Array),
        Date32 => d!(Date32Array),
        Utf8 => {
            let a = arr.as_any().downcast_ref::<StringArray>().unwrap();
            Box::new(move |r| if a.is_null(r) { String::new() } else { safe(a.value(r)) })
        }
        // Text rendering for types cat doesn't format yet — show the type, not "?".
        other => {
            let t = format!("<{other:?}>");
            Box::new(move |r| if arr.is_null(r) { String::new() } else { t.clone() })
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn sample() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec![Some("ann"), None])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn render_value_types() {
        assert_eq!(render_value(&Value::Integer(5)), "5");
        assert_eq!(render_value(&Value::String(b"hi".to_vec())), "hi");
        assert_eq!(render_value(&Value::Null), "null");
    }

    #[test]
    fn render_json_drops_human_units() {
        // JSON gets the bare machine value; text keeps the unit suffix.
        assert_eq!(render_json(&Value::Date(18627)), "18627");
        assert_eq!(render_value(&Value::Date(18627)), "18627 (epoch-day)");
        assert_eq!(render_json(&Value::TimestampMillis(5)), "5");
        assert_eq!(render_json(&Value::Integer(5)), "5");
    }

    #[test]
    fn render_value_strips_control_chars() {
        let s = render_value(&Value::String(b"\x1b[31mred".to_vec()));
        assert!(!s.contains('\x1b'), "ANSI escape must not survive: {s:?}");
    }

    #[test]
    fn ndjson_renders_null_and_quotes() {
        let out = ndjson(&[sample()], 10).unwrap();
        assert_eq!(
            out,
            "{\"id\":1,\"name\":\"ann\"}\n{\"id\":2,\"name\":null}\n"
        );
    }

    #[test]
    fn pretty_table_truncates_and_aligns() {
        let t = pretty_table(&[sample()], 1);
        assert!(t.contains("| id "));
        assert!(t.contains("| 1  "));
        assert!(!t.contains("| 2 "));
    }

    #[test]
    fn pretty_table_strips_control_chars() {
        let schema = Schema::new(vec![Field::new("name", DataType::Utf8, false)]);
        let b = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(vec!["\x1b[31mred"]))],
        )
        .unwrap();
        let t = pretty_table(&[b], 1);
        assert!(
            !t.contains('\x1b'),
            "ANSI escape must not reach the terminal: {t:?}"
        );
    }
}
