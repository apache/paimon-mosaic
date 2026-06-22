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

use arrow_array::{Array, RecordBatch};
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
        Value::TimestampNanos { millis, nanos_of_milli } => {
            format!("{}ms+{}ns", millis, nanos_of_milli)
        }
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

/// Human-readable encoding name for a `spec::ENCODING_*` id.
pub fn encoding_name(e: u8) -> &'static str {
    use paimon_mosaic_core::spec::*;
    match e {
        ENCODING_PLAIN => "plain",
        ENCODING_CONST => "const",
        ENCODING_DICT => "dict",
        ENCODING_ALL_NULL => "all_null",
        _ => "?",
    }
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
pub fn ndjson(batches: &[RecordBatch], max_rows: usize) -> String {
    let schema = batches[0].schema();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let mut out = String::new();
    let mut got = 0usize;
    'outer: for batch in batches {
        for r in 0..batch.num_rows() {
            if got >= max_rows {
                break 'outer;
            }
            out.push('{');
            for (c, name) in names.iter().enumerate() {
                if c > 0 {
                    out.push(',');
                }
                out.push_str(&json_str(name));
                out.push(':');
                out.push_str(&cell_json(batch.column(c).as_ref(), r));
            }
            out.push_str("}\n");
            got += 1;
        }
    }
    out
}

/// Render one Arrow cell as a JSON value (numbers bare, strings quoted, null).
fn cell_json(arr: &dyn Array, row: usize) -> String {
    use arrow_schema::DataType::*;
    if arr.is_null(row) {
        return "null".to_string();
    }
    match arr.data_type() {
        Utf8 | Date32 => json_str(&cell(arr, row)),
        _ => cell(arr, row),
    }
}

/// Render one Arrow cell to a string by downcasting on the column type.
fn cell(arr: &dyn Array, row: usize) -> String {
    use arrow_array::*;
    use arrow_schema::DataType::*;
    if arr.is_null(row) {
        return "".to_string();
    }
    macro_rules! d {
        ($ty:ty) => {
            arr.as_any().downcast_ref::<$ty>().unwrap().value(row).to_string()
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
        Utf8 => arr.as_any().downcast_ref::<StringArray>().unwrap().value(row).to_string(),
        _ => "?".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
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
        let out = ndjson(&[sample()], 10);
        assert_eq!(out, "{\"id\":1,\"name\":\"ann\"}\n{\"id\":2,\"name\":null}\n");
    }

    #[test]
    fn pretty_table_truncates_and_aligns() {
        let t = pretty_table(&[sample()], 1);
        assert!(t.contains("| id "));
        assert!(t.contains("| 1  "));
        assert!(!t.contains("| 2 "));
    }
}
