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

use std::cmp::Ordering;

use arrow_array::*;
use arrow_schema::{DataType, TimeUnit};

use crate::types;
use crate::values::Value;
use crate::varint;

#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub column_index: usize,
    pub null_count: usize,
    pub min: Option<Value>,
    pub max: Option<Value>,
}

pub fn supports_stats(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Time32(_)
            | DataType::Decimal128(0..=18, _)
            | DataType::Timestamp(_, _)
            | DataType::Utf8
    ) || matches!(dt, DataType::Struct(fields) if types::is_timestamp_nanos_struct(fields))
}

pub fn compare_values(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Boolean(x), Value::Boolean(y)) => Some(x.cmp(y)),
        (Value::TinyInt(x), Value::TinyInt(y)) => Some(x.cmp(y)),
        (Value::SmallInt(x), Value::SmallInt(y)) => Some(x.cmp(y)),
        (Value::Integer(x), Value::Integer(y)) => Some(x.cmp(y)),
        (Value::BigInt(x), Value::BigInt(y)) => Some(x.cmp(y)),
        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y),
        (Value::Double(x), Value::Double(y)) => x.partial_cmp(y),
        (Value::Date(x), Value::Date(y)) => Some(x.cmp(y)),
        (Value::Time(x), Value::Time(y)) => Some(x.cmp(y)),
        (Value::DecimalCompact(x), Value::DecimalCompact(y)) => Some(x.cmp(y)),
        (Value::TimestampMillis(x), Value::TimestampMillis(y)) => Some(x.cmp(y)),
        (Value::TimestampMicros(x), Value::TimestampMicros(y)) => Some(x.cmp(y)),
        (
            Value::TimestampNanos {
                millis: m1,
                nanos_of_milli: n1,
            },
            Value::TimestampNanos {
                millis: m2,
                nanos_of_milli: n2,
            },
        ) => Some(m1.cmp(m2).then(n1.cmp(n2))),
        (Value::String(x), Value::String(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

struct ColTracker {
    column_index: usize,
    batch_col_index: usize,
    data_type: DataType,
    null_count: usize,
    min: Option<Value>,
    max: Option<Value>,
}

pub struct StatsCollector {
    trackers: Vec<ColTracker>,
}

impl StatsCollector {
    pub fn new(columns: &[(usize, usize, DataType)]) -> Self {
        let trackers = columns
            .iter()
            .map(|(idx, batch_idx, dt)| ColTracker {
                column_index: *idx,
                batch_col_index: *batch_idx,
                data_type: dt.clone(),
                null_count: 0,
                min: None,
                max: None,
            })
            .collect();
        StatsCollector { trackers }
    }

    pub fn is_empty(&self) -> bool {
        self.trackers.is_empty()
    }

    pub fn update(&mut self, row: &[Value]) {
        for tracker in &mut self.trackers {
            let value = &row[tracker.batch_col_index];
            if value.is_null() {
                tracker.null_count += 1;
                continue;
            }
            if !supports_stats(&tracker.data_type) {
                continue;
            }
            update_min_max(tracker, value.clone());
        }
    }

    pub fn update_batch(&mut self, batch: &RecordBatch) {
        for tracker in &mut self.trackers {
            let array = batch.column(tracker.batch_col_index);
            tracker.null_count += array.null_count();
            if !supports_stats(&tracker.data_type) {
                continue;
            }
            for row in 0..array.len() {
                if array.is_null(row) {
                    continue;
                }
                if let Some(val) = extract_value_for_stats(array.as_ref(), row, &tracker.data_type)
                {
                    update_min_max(tracker, val);
                }
            }
        }
    }

    pub fn update_expanded(&mut self, expanded: &[ArrayRef]) {
        for tracker in &mut self.trackers {
            let array = &expanded[tracker.column_index];
            tracker.null_count += array.null_count();
            if !supports_stats(&tracker.data_type) {
                continue;
            }
            for row in 0..array.len() {
                if array.is_null(row) {
                    continue;
                }
                if let Some(val) = extract_value_for_stats(array.as_ref(), row, &tracker.data_type)
                {
                    update_min_max(tracker, val);
                }
            }
        }
    }

    pub fn finish(&mut self) -> Vec<ColumnStats> {
        let stats = self
            .trackers
            .iter()
            .map(|t| ColumnStats {
                column_index: t.column_index,
                null_count: t.null_count,
                min: t.min.clone(),
                max: t.max.clone(),
            })
            .collect();
        self.reset();
        stats
    }

    pub fn reset(&mut self) {
        for tracker in &mut self.trackers {
            tracker.null_count = 0;
            tracker.min = None;
            tracker.max = None;
        }
    }
}

fn update_min_max(tracker: &mut ColTracker, value: Value) {
    let update_min = match &tracker.min {
        None => true,
        Some(cur) => compare_values(&value, cur) == Some(Ordering::Less),
    };
    if update_min {
        tracker.min = Some(value.clone());
    }
    let update_max = match &tracker.max {
        None => true,
        Some(cur) => compare_values(&value, cur) == Some(Ordering::Greater),
    };
    if update_max {
        tracker.max = Some(value);
    }
}

fn extract_value_for_stats(array: &dyn Array, row: usize, dt: &DataType) -> Option<Value> {
    match dt {
        DataType::Boolean => {
            let a = array.as_any().downcast_ref::<BooleanArray>()?;
            Some(Value::Boolean(a.value(row)))
        }
        DataType::Int8 => {
            let a = array.as_any().downcast_ref::<Int8Array>()?;
            Some(Value::TinyInt(a.value(row)))
        }
        DataType::Int16 => {
            let a = array.as_any().downcast_ref::<Int16Array>()?;
            Some(Value::SmallInt(a.value(row)))
        }
        DataType::Int32 => {
            let a = array.as_any().downcast_ref::<Int32Array>()?;
            Some(Value::Integer(a.value(row)))
        }
        DataType::Int64 => {
            let a = array.as_any().downcast_ref::<Int64Array>()?;
            Some(Value::BigInt(a.value(row)))
        }
        DataType::Float32 => {
            let a = array.as_any().downcast_ref::<Float32Array>()?;
            Some(Value::Float(a.value(row)))
        }
        DataType::Float64 => {
            let a = array.as_any().downcast_ref::<Float64Array>()?;
            Some(Value::Double(a.value(row)))
        }
        DataType::Date32 => {
            let a = array.as_any().downcast_ref::<Date32Array>()?;
            Some(Value::Date(a.value(row)))
        }
        DataType::Time32(_) => {
            let a = array.as_any().downcast_ref::<Time32MillisecondArray>()?;
            Some(Value::Time(a.value(row)))
        }
        DataType::Utf8 => {
            let a = array.as_any().downcast_ref::<StringArray>()?;
            Some(Value::String(a.value(row).as_bytes().to_vec()))
        }
        DataType::Decimal128(p, _) => {
            let a = array.as_any().downcast_ref::<Decimal128Array>()?;
            let val = a.value(row);
            if *p <= 18 {
                Some(Value::DecimalCompact(val as i64))
            } else {
                None
            }
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            let a = array.as_any().downcast_ref::<TimestampMillisecondArray>()?;
            Some(Value::TimestampMillis(a.value(row)))
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let a = array.as_any().downcast_ref::<TimestampMicrosecondArray>()?;
            Some(Value::TimestampMicros(a.value(row)))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            let a = array.as_any().downcast_ref::<TimestampNanosecondArray>()?;
            let (millis, nanos) = types::ns_to_millis_nanos(a.value(row));
            Some(Value::TimestampNanos {
                millis,
                nanos_of_milli: nanos,
            })
        }
        DataType::Struct(fields) if types::is_timestamp_nanos_struct(fields) => {
            let s = array.as_any().downcast_ref::<StructArray>()?;
            let millis = s
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()?
                .value(row);
            let nanos = s
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()?
                .value(row);
            Some(Value::TimestampNanos {
                millis,
                nanos_of_milli: nanos,
            })
        }
        _ => None,
    }
}

pub fn serialize_stats(
    stats: &[ColumnStats],
    schema_columns: &[crate::schema::ColumnMeta],
) -> Vec<u8> {
    let mut buf = Vec::new();
    varint::encode(&mut buf, stats.len() as u32);
    for stat in stats {
        varint::encode(&mut buf, stat.column_index as u32);
        varint::encode(&mut buf, stat.null_count as u32);
        if let (Some(min_val), Some(max_val)) = (&stat.min, &stat.max) {
            let dt = &schema_columns[stat.column_index].data_type;
            serialize_value(&mut buf, min_val, dt);
            serialize_value(&mut buf, max_val, dt);
        }
    }
    buf
}

pub fn deserialize_stats(
    data: &[u8],
    pos: &mut usize,
    schema_columns: &[crate::schema::ColumnMeta],
    num_rows: usize,
) -> std::io::Result<Vec<ColumnStats>> {
    let num_stats = varint::decode(data, pos)? as usize;
    let mut stats = Vec::with_capacity(num_stats);
    for _ in 0..num_stats {
        let column_index = varint::decode(data, pos)? as usize;
        if column_index >= schema_columns.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "stats: column_index out of range",
            ));
        }
        let null_count = varint::decode(data, pos)? as usize;
        if null_count > num_rows {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "stats: null_count exceeds num_rows",
            ));
        }
        let (min, max) = if null_count < num_rows {
            let dt = &schema_columns[column_index].data_type;
            let min_val = deserialize_value(data, pos, dt)?;
            let max_val = deserialize_value(data, pos, dt)?;
            (Some(min_val), Some(max_val))
        } else {
            (None, None)
        };
        stats.push(ColumnStats {
            column_index,
            null_count,
            min,
            max,
        });
    }
    Ok(stats)
}

fn serialize_value(buf: &mut Vec<u8>, value: &Value, dt: &DataType) {
    let w = types::fixed_width(dt);
    if w > 0 {
        crate::values::write_fixed(buf, value, w).expect("stats value type must match column type");
    } else {
        crate::values::write_variable(buf, value).expect("stats value type must match column type");
    }
}

fn deserialize_value(data: &[u8], pos: &mut usize, dt: &DataType) -> std::io::Result<Value> {
    let w = types::fixed_width(dt);
    if w > 0 {
        let end = *pos + w as usize;
        if end > data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "stats: not enough data for fixed value",
            ));
        }
        let value = read_fixed_value(data, *pos, dt, w);
        *pos += w as usize;
        Ok(value)
    } else {
        let (value, size) = read_variable_value(data, *pos, dt)?;
        *pos += size;
        Ok(value)
    }
}

fn read_fixed_value(buf: &[u8], pos: usize, dt: &DataType, width: i32) -> Value {
    match dt {
        DataType::Boolean => Value::Boolean(buf[pos] != 0),
        DataType::Int8 => Value::TinyInt(buf[pos] as i8),
        DataType::Int16 => Value::SmallInt(i16::from_be_bytes([buf[pos], buf[pos + 1]])),
        DataType::Int32 => Value::Integer(i32::from_be_bytes([
            buf[pos],
            buf[pos + 1],
            buf[pos + 2],
            buf[pos + 3],
        ])),
        DataType::Date32 => Value::Date(i32::from_be_bytes([
            buf[pos],
            buf[pos + 1],
            buf[pos + 2],
            buf[pos + 3],
        ])),
        DataType::Time32(_) => Value::Time(i32::from_be_bytes([
            buf[pos],
            buf[pos + 1],
            buf[pos + 2],
            buf[pos + 3],
        ])),
        DataType::Float32 => {
            let bits = u32::from_be_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
            Value::Float(f32::from_bits(bits))
        }
        DataType::Int64 => Value::BigInt(i64::from_be_bytes([
            buf[pos],
            buf[pos + 1],
            buf[pos + 2],
            buf[pos + 3],
            buf[pos + 4],
            buf[pos + 5],
            buf[pos + 6],
            buf[pos + 7],
        ])),
        DataType::Float64 => {
            let bits = u64::from_be_bytes([
                buf[pos],
                buf[pos + 1],
                buf[pos + 2],
                buf[pos + 3],
                buf[pos + 4],
                buf[pos + 5],
                buf[pos + 6],
                buf[pos + 7],
            ]);
            Value::Double(f64::from_bits(bits))
        }
        DataType::Decimal128(_, _) => Value::DecimalCompact(i64::from_be_bytes([
            buf[pos],
            buf[pos + 1],
            buf[pos + 2],
            buf[pos + 3],
            buf[pos + 4],
            buf[pos + 5],
            buf[pos + 6],
            buf[pos + 7],
        ])),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            Value::TimestampMillis(i64::from_be_bytes([
                buf[pos],
                buf[pos + 1],
                buf[pos + 2],
                buf[pos + 3],
                buf[pos + 4],
                buf[pos + 5],
                buf[pos + 6],
                buf[pos + 7],
            ]))
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            Value::TimestampMicros(i64::from_be_bytes([
                buf[pos],
                buf[pos + 1],
                buf[pos + 2],
                buf[pos + 3],
                buf[pos + 4],
                buf[pos + 5],
                buf[pos + 6],
                buf[pos + 7],
            ]))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            debug_assert_eq!(width, 12);
            let millis = i64::from_be_bytes([
                buf[pos],
                buf[pos + 1],
                buf[pos + 2],
                buf[pos + 3],
                buf[pos + 4],
                buf[pos + 5],
                buf[pos + 6],
                buf[pos + 7],
            ]);
            let nanos =
                i32::from_be_bytes([buf[pos + 8], buf[pos + 9], buf[pos + 10], buf[pos + 11]]);
            Value::TimestampNanos {
                millis,
                nanos_of_milli: nanos,
            }
        }
        DataType::Struct(fields) if types::is_timestamp_nanos_struct(fields) => {
            debug_assert_eq!(width, 12);
            let millis = i64::from_be_bytes([
                buf[pos],
                buf[pos + 1],
                buf[pos + 2],
                buf[pos + 3],
                buf[pos + 4],
                buf[pos + 5],
                buf[pos + 6],
                buf[pos + 7],
            ]);
            let nanos =
                i32::from_be_bytes([buf[pos + 8], buf[pos + 9], buf[pos + 10], buf[pos + 11]]);
            Value::TimestampNanos {
                millis,
                nanos_of_milli: nanos,
            }
        }
        _ => Value::Null,
    }
}

fn read_variable_value(buf: &[u8], pos: usize, dt: &DataType) -> std::io::Result<(Value, usize)> {
    let mut p = pos;
    let len = varint::decode(buf, &mut p)? as usize;
    let header_size = p - pos;
    if p + len > buf.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "stats: not enough data for variable value",
        ));
    }
    let bytes = buf[p..p + len].to_vec();
    let total_size = header_size + len;
    let value = match dt {
        DataType::Utf8 => Value::String(bytes),
        _ => Value::Null,
    };
    Ok((value, total_size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ColumnMeta;
    use arrow_schema::DataType;

    #[test]
    fn test_compare_values() {
        assert_eq!(
            compare_values(&Value::Integer(1), &Value::Integer(2)),
            Some(Ordering::Less)
        );
        assert_eq!(
            compare_values(&Value::Integer(2), &Value::Integer(2)),
            Some(Ordering::Equal)
        );
        assert_eq!(
            compare_values(&Value::Double(1.5), &Value::Double(2.5)),
            Some(Ordering::Less)
        );
        assert_eq!(
            compare_values(
                &Value::String(b"abc".to_vec()),
                &Value::String(b"abd".to_vec())
            ),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn test_stats_collector() {
        let columns = vec![
            (0usize, 0usize, DataType::Int32),
            (1usize, 1usize, DataType::Float64),
        ];
        let mut collector = StatsCollector::new(&columns);

        collector.update(&[Value::Integer(10), Value::Double(1.5)]);
        collector.update(&[Value::Integer(5), Value::Double(3.0)]);
        collector.update(&[Value::Integer(20), Value::Null]);
        collector.update(&[Value::Null, Value::Double(0.5)]);

        let stats = collector.finish();
        assert_eq!(stats.len(), 2);

        assert_eq!(stats[0].column_index, 0);
        assert_eq!(stats[0].null_count, 1);
        assert!(matches!(&stats[0].min, Some(Value::Integer(5))));
        assert!(matches!(&stats[0].max, Some(Value::Integer(20))));

        assert_eq!(stats[1].column_index, 1);
        assert_eq!(stats[1].null_count, 1);
        match &stats[1].min {
            Some(Value::Double(v)) => assert!((*v - 0.5).abs() < 1e-10),
            other => panic!("expected Double(0.5), got {:?}", other),
        }
        match &stats[1].max {
            Some(Value::Double(v)) => assert!((*v - 3.0).abs() < 1e-10),
            other => panic!("expected Double(3.0), got {:?}", other),
        }
    }

    #[test]
    fn test_serialize_deserialize_stats() {
        let schema_columns = vec![
            ColumnMeta {
                name: "a".to_string(),
                column_id: 0,
                data_type: DataType::Int32,
                nullable: true,
                bucket_id: 0,
            },
            ColumnMeta {
                name: "b".to_string(),
                column_id: 1,
                data_type: DataType::Utf8,
                nullable: true,
                bucket_id: 0,
            },
        ];

        let stats = vec![
            ColumnStats {
                column_index: 0,
                null_count: 3,
                min: Some(Value::Integer(5)),
                max: Some(Value::Integer(100)),
            },
            ColumnStats {
                column_index: 1,
                null_count: 0,
                min: Some(Value::String(b"abc".to_vec())),
                max: Some(Value::String(b"xyz".to_vec())),
            },
        ];

        let buf = serialize_stats(&stats, &schema_columns);
        let mut pos = 0;
        let num_rows = 10;
        let result = deserialize_stats(&buf, &mut pos, &schema_columns, num_rows).unwrap();
        assert_eq!(pos, buf.len());
        assert_eq!(result.len(), 2);

        assert_eq!(result[0].column_index, 0);
        assert_eq!(result[0].null_count, 3);
        assert!(matches!(&result[0].min, Some(Value::Integer(5))));
        assert!(matches!(&result[0].max, Some(Value::Integer(100))));

        assert_eq!(result[1].column_index, 1);
        assert_eq!(result[1].null_count, 0);
        assert!(matches!(&result[1].min, Some(Value::String(b)) if b == b"abc"));
        assert!(matches!(&result[1].max, Some(Value::String(b)) if b == b"xyz"));
    }

    #[test]
    fn test_serialize_empty_stats() {
        let schema_columns = vec![];
        let stats: Vec<ColumnStats> = vec![];
        let buf = serialize_stats(&stats, &schema_columns);
        assert_eq!(buf, vec![0]);
        let mut pos = 0;
        let result = deserialize_stats(&buf, &mut pos, &schema_columns, 0).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_serialize_all_null_stats() {
        let schema_columns = vec![ColumnMeta {
            name: "x".to_string(),
            column_id: 0,
            data_type: DataType::Int32,
            nullable: true,
            bucket_id: 0,
        }];
        let stats = vec![ColumnStats {
            column_index: 0,
            null_count: 100,
            min: None,
            max: None,
        }];
        let buf = serialize_stats(&stats, &schema_columns);
        let mut pos = 0;
        let result = deserialize_stats(&buf, &mut pos, &schema_columns, 100).unwrap();
        assert_eq!(result[0].null_count, 100);
        assert!(result[0].min.is_none());
        assert!(result[0].max.is_none());
    }
}
