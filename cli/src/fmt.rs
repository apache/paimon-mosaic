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
        Value::String(b) => String::from_utf8_lossy(b).into_owned(),
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

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
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

/// Human-readable bucket kind name.
pub fn bucket_kind(k: paimon_mosaic_core::reader::BucketKind) -> &'static str {
    use paimon_mosaic_core::reader::BucketKind::*;
    match k {
        Empty => "empty",
        Monolithic => "monolithic",
        Paged => "paged",
        _ => "unknown",
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

/// Escape a string as a JSON string literal (quotes included).
pub fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Pretty-print a slice of record batches as an aligned ASCII table.
pub fn pretty_table(batches: &[RecordBatch], max_rows: usize) -> String {
    if batches.is_empty() {
        return String::new();
    }
    let schema = batches[0].schema();
    let headers: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let ncols = headers.len();

    let mut rows: Vec<Vec<String>> = Vec::new();
    'outer: for batch in batches {
        for r in 0..batch.num_rows() {
            if rows.len() >= max_rows {
                break 'outer;
            }
            let mut row = Vec::with_capacity(ncols);
            for c in 0..ncols {
                row.push(cell(batch.column(c).as_ref(), r));
            }
            rows.push(row);
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

/// Render one Arrow cell to a string by downcasting on the column type.
fn cell(arr: &dyn Array, row: usize) -> String {
    use arrow::array::*;
    use arrow::datatypes::DataType::*;
    if arr.is_null(row) {
        return "".to_string();
    }
    macro_rules! d {
        ($ty:ty) => {
            arr.as_any()
                .downcast_ref::<$ty>()
                .unwrap()
                .value(row)
                .to_string()
        };
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
        Utf8 => arr
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(row)
            .to_string(),
        // Text rendering for types cat doesn't format yet — show the type, not "?".
        other => format!("<{other:?}>"),
    }
}

/// A single `column op value` filter. Ops: `=` `!=` `>` `>=` `<` `<=`.
pub struct Where {
    pub column: String,
    pub op: &'static str,
    pub value: String,
}

/// Parse one condition like `id>100` or `kind=a`. Longest operators first.
pub fn parse_where(s: &str) -> Result<Where, String> {
    for op in [">=", "<=", "!=", "=", ">", "<"] {
        if let Some(i) = s.find(op) {
            let column = s[..i].trim().to_string();
            let value = s[i + op.len()..].trim().to_string();
            if column.is_empty() || value.is_empty() {
                return Err(format!("bad --where: {s}"));
            }
            return Ok(Where { column, op, value });
        }
    }
    Err(format!("bad --where (need =, !=, >, >=, <, <=): {s}"))
}

/// Keep rows where the condition holds. Numeric columns compare numerically;
/// booleans by true/false; everything else as exact strings (only `=`/`!=`
/// meaningful). Nulls never match. Unsupported types error rather than drop.
pub fn apply_where(batch: &RecordBatch, w: &Where) -> Result<RecordBatch, String> {
    use arrow::datatypes::DataType::*;
    let col = batch
        .column_by_name(&w.column)
        .ok_or_else(|| format!("--where: column '{}' not found", w.column))?;
    let int = matches!(col.data_type(), Int8 | Int16 | Int32 | Int64 | Date32);
    let float = matches!(col.data_type(), Float32 | Float64);
    let boolean = matches!(col.data_type(), Boolean);
    // Integer columns compare in i128 (exact for full i64 range); float columns
    // in f64; everything else as exact strings. Ordering is numeric-only.
    if matches!(w.op, ">" | ">=" | "<" | "<=")
        && !((int && w.value.parse::<i128>().is_ok()) || (float && w.value.parse::<f64>().is_ok()))
    {
        return Err(format!(
            "--where: '{}' needs a numeric column and value (got '{}' {} '{}')",
            w.op, w.column, w.op, w.value
        ));
    }
    // Downcast the column once and compare per row directly, instead of
    // rendering each cell to a String. Integers use i128 (exact), floats f64.
    use arrow::array::*;
    let n = batch.num_rows();
    let row_ok = |r: usize| !col.is_null(r);
    // When the RHS can't parse for a numeric column, `=` matches nothing and
    // `!=` matches every non-null row (nulls never match, either way).
    let no_match =
        |keep_nonnull: bool| -> Vec<bool> { (0..n).map(|r| keep_nonnull && row_ok(r)).collect() };
    // Downcast the column once and return a per-row value accessor; avoids
    // re-downcasting inside the row loop.
    macro_rules! d {
        ($ty:ty, $v:ident, $r:ident => $body:expr) => {{
            let $v = col.as_any().downcast_ref::<$ty>().unwrap();
            Box::new(move |$r: usize| $body)
        }};
    }
    let mask: Vec<bool> = if int {
        let Ok(rhs) = w.value.parse::<i128>() else {
            return finish(batch, no_match(w.op == "!="));
        };
        let at: Box<dyn Fn(usize) -> i128 + '_> = match col.data_type() {
            Int8 => d!(Int8Array, v, r => v.value(r) as i128),
            Int16 => d!(Int16Array, v, r => v.value(r) as i128),
            Int32 => d!(Int32Array, v, r => v.value(r) as i128),
            Date32 => d!(Date32Array, v, r => v.value(r) as i128),
            _ => d!(Int64Array, v, r => v.value(r) as i128),
        };
        (0..n)
            .map(|r| row_ok(r) && cmp_op(w.op, &at(r), &rhs))
            .collect()
    } else if float {
        let Ok(rhs) = w.value.parse::<f64>() else {
            return finish(batch, no_match(w.op == "!="));
        };
        // Parse the RHS at the column's own precision so an f32 cell compares
        // against an f32-rounded value (stored 0.1f32 == "0.1", not the f64 0.1).
        let (rhs, at): (f64, Box<dyn Fn(usize) -> f64 + '_>) = match col.data_type() {
            Float32 => (
                w.value.parse::<f32>().unwrap_or(f32::NAN) as f64,
                d!(Float32Array, v, r => v.value(r) as f64),
            ),
            _ => (rhs, d!(Float64Array, v, r => v.value(r))),
        };
        (0..n)
            .map(|r| row_ok(r) && cmp_op(w.op, &at(r), &rhs))
            .collect()
    } else if boolean {
        let rhs = match w.value.as_str() {
            "true" => true,
            "false" => false,
            _ => {
                return Err(format!(
                    "--where: boolean column '{}' needs true/false (got '{}')",
                    w.column, w.value
                ))
            }
        };
        let a = col.as_any().downcast_ref::<BooleanArray>().unwrap();
        (0..n)
            .map(|r| {
                row_ok(r)
                    && match w.op {
                        "=" => a.value(r) == rhs,
                        "!=" => a.value(r) != rhs,
                        _ => false,
                    }
            })
            .collect()
    } else if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        (0..n)
            .map(|r| {
                row_ok(r)
                    && match w.op {
                        "=" => a.value(r) == w.value,
                        "!=" => a.value(r) != w.value,
                        _ => false,
                    }
            })
            .collect()
    } else {
        // Fail loudly rather than silently dropping every row for a type we
        // can't compare (e.g. nested/binary), instead of returning "(no rows)".
        return Err(format!(
            "--where: column '{}' has unsupported type {:?}",
            w.column,
            col.data_type()
        ));
    };
    finish(batch, mask)
}

/// Apply a boolean row mask to a batch, returning the filtered batch.
fn finish(batch: &RecordBatch, mask: Vec<bool>) -> Result<RecordBatch, String> {
    let m = arrow::array::BooleanArray::from(mask);
    arrow::compute::filter_record_batch(batch, &m).map_err(|e| e.to_string())
}

/// Apply a comparison operator to any ordered pair.
fn cmp_op<T: PartialOrd>(op: &str, a: &T, b: &T) -> bool {
    match op {
        "=" => a == b,
        "!=" => a != b,
        ">" => a > b,
        ">=" => a >= b,
        "<" => a < b,
        "<=" => a <= b,
        _ => false,
    }
}

/// Integer value of a stats [`Value`], or `None` if not integral. Used so large
/// i64 (e.g. Snowflake ids) compare exactly rather than via lossy f64.
fn to_i128(v: &Value) -> Option<i128> {
    use Value::*;
    match v {
        TinyInt(x) => Some(*x as i128),
        SmallInt(x) => Some(*x as i128),
        Integer(x) | Date(x) => Some(*x as i128),
        BigInt(x) => Some(*x as i128),
        _ => None,
    }
}

/// Float value of a stats [`Value`], or `None` for non-numeric types.
fn to_f64(v: &Value) -> Option<f64> {
    use Value::*;
    match v {
        Float(x) => Some(*x as f64),
        Double(x) => Some(*x),
        _ => None,
    }
}

/// True when a row group's `[min, max]` provably excludes the filter — safe to
/// skip. Numeric only and conservative: any missing/unparsable stat → keep.
pub fn stats_exclude(w: &Where, min: &Option<Value>, max: &Option<Value>) -> bool {
    let (min, max) = match (min.as_ref(), max.as_ref()) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };
    // Integer columns: compare exactly in i128. Float columns: f64. Excluded
    // when the value lies strictly outside [lo, hi] for the operator.
    if let (Some(lo), Some(hi), Ok(v)) = (to_i128(min), to_i128(max), w.value.parse::<i128>()) {
        return excl(w.op, lo, hi, v);
    }
    if let (Some(lo), Some(hi), Ok(v)) = (to_f64(min), to_f64(max), w.value.parse::<f64>()) {
        return excl(w.op, lo, hi, v);
    }
    false
}

fn excl<T: PartialOrd>(op: &str, lo: T, hi: T, v: T) -> bool {
    match op {
        ">" => hi <= v,
        ">=" => hi < v,
        "<" => lo >= v,
        "<=" => lo > v,
        "=" => v < lo || v > hi,
        _ => false,
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
    fn json_str_escapes() {
        assert_eq!(json_str("a\"b\n"), "\"a\\\"b\\n\"");
        assert_eq!(json_str("x"), "\"x\"");
    }

    #[test]
    fn render_value_types() {
        assert_eq!(render_value(&Value::Integer(5)), "5");
        assert_eq!(render_value(&Value::String(b"hi".to_vec())), "hi");
        assert_eq!(render_value(&Value::Null), "null");
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
}
