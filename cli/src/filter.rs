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

//! `--where` row filtering: parse, apply, and stats-based row-group skipping.

use arrow::array::RecordBatch;
use paimon_mosaic_core::values::Value;

/// Comparison operator for a `--where` filter.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

impl Op {
    /// The source token, for error messages.
    pub fn as_str(self) -> &'static str {
        match self {
            Op::Eq => "=",
            Op::Ne => "!=",
            Op::Gt => ">",
            Op::Ge => ">=",
            Op::Lt => "<",
            Op::Le => "<=",
        }
    }
    fn ordered(self) -> bool {
        matches!(self, Op::Gt | Op::Ge | Op::Lt | Op::Le)
    }
}

/// A single `column op value` filter. Ops: `=` `!=` `>` `>=` `<` `<=`.
pub struct Where {
    pub column: String,
    pub op: Op,
    pub value: String,
}

/// Parse one condition like `id>100` or `kind=a`. Longest operators first.
pub fn parse_where(s: &str) -> Result<Where, String> {
    for (tok, op) in [
        (">=", Op::Ge),
        ("<=", Op::Le),
        ("!=", Op::Ne),
        ("=", Op::Eq),
        (">", Op::Gt),
        ("<", Op::Lt),
    ] {
        if let Some(i) = s.find(tok) {
            let column = s[..i].trim().to_string();
            let value = s[i + tok.len()..].trim().to_string();
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
    if w.op.ordered()
        && !((int && parse_int_rhs(col.data_type(), &w.value).is_some())
            || (float && w.value.parse::<f64>().is_ok()))
    {
        return Err(format!(
            "--where: '{}' needs a numeric column and value (got '{}' {} '{}')",
            w.op.as_str(),
            w.column,
            w.op.as_str(),
            w.value
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
        let Some(rhs) = parse_int_rhs(col.data_type(), &w.value) else {
            return finish(batch, no_match(w.op == Op::Ne));
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
            return finish(batch, no_match(w.op == Op::Ne));
        };
        // Parse the RHS at the column's own precision so an f32 cell compares
        // against an f32-rounded value (stored 0.1f32 == "0.1", not the f64 0.1).
        let (rhs, at): (f64, Box<dyn Fn(usize) -> f64 + '_>) = match col.data_type() {
            Float32 => (
                rhs as f32 as f64,
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
                        Op::Eq => a.value(r) == rhs,
                        Op::Ne => a.value(r) != rhs,
                        _ => false,
                    }
            })
            .collect()
    } else if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        (0..n)
            .map(|r| {
                row_ok(r)
                    && match w.op {
                        Op::Eq => a.value(r) == w.value,
                        Op::Ne => a.value(r) != w.value,
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
fn cmp_op<T: PartialOrd>(op: Op, a: &T, b: &T) -> bool {
    match op {
        Op::Eq => a == b,
        Op::Ne => a != b,
        Op::Gt => a > b,
        Op::Ge => a >= b,
        Op::Lt => a < b,
        Op::Le => a <= b,
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

/// Parse an integer-like filter literal for the column type. `Date32` accepts
/// both epoch-day integers and ISO dates (`YYYY-MM-DD`) to match Arrow JSON.
fn parse_int_rhs(dt: &arrow::datatypes::DataType, value: &str) -> Option<i128> {
    if matches!(dt, arrow::datatypes::DataType::Date32) {
        parse_date32(value).map(i128::from)
    } else {
        value.parse::<i128>().ok()
    }
}

fn parse_date32(value: &str) -> Option<i32> {
    if let Ok(days) = value.parse::<i32>() {
        return Some(days);
    }
    let mut parts = value.split('-');
    let year = parts.next()?.parse::<i32>().ok()?;
    let month = parts.next()?.parse::<u32>().ok()?;
    let day = parts.next()?.parse::<u32>().ok()?;
    if parts.next().is_some() || !valid_ymd(year, month, day) {
        return None;
    }
    Some(days_from_civil(year, month, day))
}

fn valid_ymd(year: i32, month: u32, day: u32) -> bool {
    let mdays = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => return false,
    };
    (1..=mdays).contains(&day)
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i32 {
    let year = year - if month <= 2 { 1 } else { 0 };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month = month as i32;
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
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
    if let (Some(lo), Some(hi)) = (to_i128(min), to_i128(max)) {
        let v = match min {
            Value::Date(_) => parse_date32(&w.value).map(i128::from),
            _ => w.value.parse::<i128>().ok(),
        };
        if let Some(v) = v {
            return excl(w.op, lo, hi, v);
        }
    }
    if let (Some(lo), Some(hi)) = (to_f64(min), to_f64(max)) {
        // Round the RHS to the stat's own width so the bound matches apply_where:
        // an f32 group min/max compares against an f32-rounded value, not f64 0.1.
        let v = match min {
            Value::Float(_) => w.value.parse::<f32>().map(|x| x as f64),
            _ => w.value.parse::<f64>(),
        };
        if let Ok(v) = v {
            return excl(w.op, lo, hi, v);
        }
    }
    false
}

fn excl<T: PartialOrd>(op: Op, lo: T, hi: T, v: T) -> bool {
    match op {
        Op::Gt => hi <= v,
        Op::Ge => hi < v,
        Op::Lt => lo >= v,
        Op::Le => lo > v,
        Op::Eq => v < lo || v > hi,
        Op::Ne => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn where_f32_rhs_overflow_does_not_nan_match() {
        // RHS beyond f32 range saturates to +inf (not NaN), so id<1e40 keeps all.
        let schema = Schema::new(vec![Field::new("v", DataType::Float32, false)]);
        let b = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Float32Array::from(vec![1.0f32, 2.0]))],
        )
        .unwrap();
        let w = parse_where("v<1e40").unwrap();
        assert_eq!(apply_where(&b, &w).unwrap().num_rows(), 2);
    }

    #[test]
    fn where_date32_accepts_iso_literal() {
        let schema = Schema::new(vec![Field::new("d", DataType::Date32, false)]);
        let b = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(arrow::array::Date32Array::from(vec![
                18_262, 18_628,
            ]))],
        )
        .unwrap();
        let w = parse_where("d>=2021-01-01").unwrap();
        assert_eq!(apply_where(&b, &w).unwrap().num_rows(), 1);
    }
}
