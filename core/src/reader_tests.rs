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

use super::*;
use crate::writer::{MosaicWriter, OutputFile, WriterOptions};
use arrow_array::*;
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use std::sync::Arc;

fn columns_to_arrow_schema(columns: &[(String, DataType, bool)]) -> Schema {
    Schema::new(
        columns
            .iter()
            .map(|(name, dt, nullable)| Field::new(name, dt.clone(), *nullable))
            .collect::<Vec<_>>(),
    )
}

struct ByteArrayInputFile {
    data: Vec<u8>,
}

impl ByteArrayInputFile {
    fn new(data: Vec<u8>) -> Self {
        Self { data }
    }
}

impl InputFile for ByteArrayInputFile {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let start = offset as usize;
        let end = start + buf.len();
        if end > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "read past end",
            ));
        }
        buf.copy_from_slice(&self.data[start..end]);
        Ok(())
    }
}

struct MemOutputFile {
    buf: Vec<u8>,
}

impl MemOutputFile {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
}

impl OutputFile for MemOutputFile {
    fn write(&mut self, data: &[u8]) -> io::Result<()> {
        self.buf.extend_from_slice(data);
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
    fn pos(&self) -> u64 {
        self.buf.len() as u64
    }
}

fn values_to_batch(rows: &[Vec<Value>], columns: &[(String, DataType, bool)]) -> RecordBatch {
    let num_cols = columns.len();
    let mut fields = Vec::with_capacity(num_cols);
    let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(num_cols);

    for (c, (name, dt, nullable)) in columns.iter().enumerate() {
        fields.push(Field::new(name, dt.clone(), *nullable));
        arrays.push(build_array_from_values(rows, c, dt));
    }

    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn build_array_from_values(rows: &[Vec<Value>], col: usize, dt: &DataType) -> Arc<dyn Array> {
    match dt {
        DataType::Boolean => {
            let vals: Vec<Option<bool>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::Boolean(v) => Some(*v),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            Arc::new(BooleanArray::from(vals))
        }
        DataType::Int8 => {
            let vals: Vec<Option<i8>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::TinyInt(v) => Some(*v),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            Arc::new(Int8Array::from(vals))
        }
        DataType::Int16 => {
            let vals: Vec<Option<i16>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::SmallInt(v) => Some(*v),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            Arc::new(Int16Array::from(vals))
        }
        DataType::Int32 => {
            let vals: Vec<Option<i32>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::Integer(v) | Value::Date(v) | Value::Time(v) => Some(*v),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            Arc::new(Int32Array::from(vals))
        }
        DataType::Int64 => {
            let vals: Vec<Option<i64>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::BigInt(v) => Some(*v),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            Arc::new(Int64Array::from(vals))
        }
        DataType::Float32 => {
            let vals: Vec<Option<f32>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::Float(v) => Some(*v),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            Arc::new(Float32Array::from(vals))
        }
        DataType::Float64 => {
            let vals: Vec<Option<f64>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::Double(v) => Some(*v),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            Arc::new(Float64Array::from(vals))
        }
        DataType::Date32 => {
            let vals: Vec<Option<i32>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::Date(v) => Some(*v),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            Arc::new(Date32Array::from(vals))
        }
        DataType::Time32(_) => {
            let vals: Vec<Option<i32>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::Time(v) => Some(*v),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            Arc::new(Time32MillisecondArray::from(vals))
        }
        DataType::Utf8 => {
            let vals: Vec<Option<String>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::String(b) => Some(String::from_utf8(b.clone()).unwrap()),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            Arc::new(StringArray::from(vals))
        }
        DataType::Binary => {
            let vals: Vec<Option<Vec<u8>>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::Bytes(b) => Some(b.clone()),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            let refs: Vec<Option<&[u8]>> = vals.iter().map(|v| v.as_deref()).collect();
            Arc::new(BinaryArray::from(refs))
        }
        DataType::Decimal128(p, s) => {
            let vals: Vec<Option<i128>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::DecimalCompact(v) => Some(*v as i128),
                    Value::DecimalLarge(bytes) => Some(biginteger_bytes_to_i128(bytes)),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            Arc::new(
                Decimal128Array::from(vals)
                    .with_precision_and_scale(*p, *s)
                    .unwrap(),
            )
        }
        DataType::Timestamp(TimeUnit::Millisecond, tz) => {
            let vals: Vec<Option<i64>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::TimestampMillis(v) => Some(*v),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            let arr = TimestampMillisecondArray::from(vals);
            Arc::new(if let Some(tz) = tz {
                arr.with_timezone(tz.clone())
            } else {
                arr
            })
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            let vals: Vec<Option<i64>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::TimestampMicros(v) => Some(*v),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            let arr = TimestampMicrosecondArray::from(vals);
            Arc::new(if let Some(tz) = tz {
                arr.with_timezone(tz.clone())
            } else {
                arr
            })
        }
        DataType::Struct(struct_fields)
            if crate::types::is_timestamp_nanos_struct(struct_fields) =>
        {
            let millis_vals: Vec<Option<i64>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::TimestampNanos { millis, .. } => Some(*millis),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            let nanos_vals: Vec<Option<i32>> = rows
                .iter()
                .map(|row| match &row[col] {
                    Value::TimestampNanos { nanos_of_milli, .. } => Some(*nanos_of_milli),
                    Value::Null => None,
                    _ => panic!("type mismatch"),
                })
                .collect();
            let nulls: Option<arrow_buffer::NullBuffer> = {
                let any_null = rows.iter().any(|row| row[col].is_null());
                if any_null {
                    let bools: Vec<bool> = rows.iter().map(|row| !row[col].is_null()).collect();
                    Some(arrow_buffer::NullBuffer::from(bools))
                } else {
                    None
                }
            };
            Arc::new(StructArray::new(
                struct_fields.clone(),
                vec![
                    Arc::new(Int64Array::from(
                        millis_vals
                            .into_iter()
                            .map(|v| v.unwrap_or(0))
                            .collect::<Vec<_>>(),
                    )),
                    Arc::new(Int32Array::from(
                        nanos_vals
                            .into_iter()
                            .map(|v| v.unwrap_or(0))
                            .collect::<Vec<_>>(),
                    )),
                ],
                nulls,
            ))
        }
        _ => panic!("unsupported type in test helper: {:?}", dt),
    }
}

fn biginteger_bytes_to_i128(bytes: &[u8]) -> i128 {
    let negative = bytes[0] & 0x80 != 0;
    let pad = if negative { 0xFF } else { 0x00 };
    let mut buf = [pad; 16];
    let start = 16 - bytes.len();
    buf[start..].copy_from_slice(bytes);
    i128::from_be_bytes(buf)
}

fn write_values(
    writer: &mut MosaicWriter<MemOutputFile>,
    columns: &[(String, DataType, bool)],
    rows: &[Vec<Value>],
) {
    if rows.is_empty() {
        return;
    }
    for row in rows {
        let batch = values_to_batch(std::slice::from_ref(row), columns);
        writer.write_batch(&batch).unwrap();
    }
}

fn batch_col_bool<'a>(batch: &'a RecordBatch, name: &str) -> &'a BooleanArray {
    let idx = batch.schema().index_of(name).unwrap();
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap()
}

fn batch_col_i8<'a>(batch: &'a RecordBatch, name: &str) -> &'a Int8Array {
    let idx = batch.schema().index_of(name).unwrap();
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Int8Array>()
        .unwrap()
}

fn batch_col_i16<'a>(batch: &'a RecordBatch, name: &str) -> &'a Int16Array {
    let idx = batch.schema().index_of(name).unwrap();
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Int16Array>()
        .unwrap()
}

fn batch_col_i32<'a>(batch: &'a RecordBatch, name: &str) -> &'a Int32Array {
    let idx = batch.schema().index_of(name).unwrap();
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
}

fn batch_col_i64<'a>(batch: &'a RecordBatch, name: &str) -> &'a Int64Array {
    let idx = batch.schema().index_of(name).unwrap();
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
}

fn batch_col_f32<'a>(batch: &'a RecordBatch, name: &str) -> &'a Float32Array {
    let idx = batch.schema().index_of(name).unwrap();
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
}

fn batch_col_f64<'a>(batch: &'a RecordBatch, name: &str) -> &'a Float64Array {
    let idx = batch.schema().index_of(name).unwrap();
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
}

fn batch_col_string<'a>(batch: &'a RecordBatch, name: &str) -> &'a StringArray {
    let idx = batch.schema().index_of(name).unwrap();
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
}

fn batch_col_binary<'a>(batch: &'a RecordBatch, name: &str) -> &'a BinaryArray {
    let idx = batch.schema().index_of(name).unwrap();
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap()
}

#[test]
fn test_roundtrip_basic() {
    let columns = vec![
        ("age".to_string(), DataType::Int32, true),
        ("name".to_string(), DataType::Utf8, true),
        ("score".to_string(), DataType::Float64, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            num_buckets: 2,
            ..Default::default()
        },
    )
    .unwrap();

    let rows_to_write: Vec<Vec<Value>> = (0..100)
        .map(|i| {
            vec![
                Value::Integer(20 + (i % 50)),
                Value::String(format!("user_{}", i).into_bytes()),
                Value::Double(i as f64 * 1.5),
            ]
        })
        .collect();

    write_values(&mut writer, &columns, &rows_to_write);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();
    assert_eq!(reader.schema().columns.len(), 3);
    assert_eq!(reader.num_row_groups(), 1);

    let mut rg = reader.row_group_reader(0).unwrap();
    assert_eq!(rg.num_rows(), 100);

    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 100);
    assert_eq!(batch.num_columns(), 3);

    let ages = batch_col_i32(&batch, "age");
    let names = batch_col_string(&batch, "name");
    let scores = batch_col_f64(&batch, "score");

    for i in 0..100usize {
        assert_eq!(ages.value(i), 20 + (i as i32 % 50));
        assert_eq!(names.value(i), format!("user_{}", i));
        assert!((scores.value(i) - i as f64 * 1.5).abs() < 1e-10);
    }
}

#[test]
fn test_roundtrip_with_nulls() {
    let columns = vec![
        ("id".to_string(), DataType::Int32, true),
        ("name".to_string(), DataType::Utf8, true),
        ("value".to_string(), DataType::Float64, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            num_buckets: 2,
            ..Default::default()
        },
    )
    .unwrap();

    let rows = vec![
        vec![
            Value::Integer(1),
            Value::String(b"hello".to_vec()),
            Value::Double(1.0),
        ],
        vec![Value::Integer(2), Value::Null, Value::Double(2.0)],
        vec![
            Value::Integer(3),
            Value::String(b"world".to_vec()),
            Value::Null,
        ],
        vec![Value::Integer(4), Value::Null, Value::Null],
    ];
    write_values(&mut writer, &columns, &rows);

    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 4);

    let ids = batch_col_i32(&batch, "id");
    let names = batch_col_string(&batch, "name");
    let vals = batch_col_f64(&batch, "value");

    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
    assert_eq!(ids.value(2), 3);
    assert_eq!(ids.value(3), 4);

    assert!(!names.is_null(0));
    assert_eq!(names.value(0), "hello");
    assert!(names.is_null(1));
    assert!(!names.is_null(2));
    assert_eq!(names.value(2), "world");
    assert!(names.is_null(3));

    assert!(!vals.is_null(0));
    assert!((vals.value(0) - 1.0).abs() < 1e-10);
    assert!(!vals.is_null(1));
    assert!((vals.value(1) - 2.0).abs() < 1e-10);
    assert!(vals.is_null(2));
    assert!(vals.is_null(3));
}

#[test]
fn test_roundtrip_with_zstd() {
    let columns = vec![
        ("a".to_string(), DataType::Int64, true),
        ("b".to_string(), DataType::Int64, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_ZSTD,
            zstd_level: 3,
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();

    let rows: Vec<Vec<Value>> = (0..1000)
        .map(|i| vec![Value::BigInt(i), Value::BigInt(i * 2)])
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 1000);

    let col_a = batch_col_i64(&batch, "a");
    let col_b = batch_col_i64(&batch, "b");

    for i in 0..1000usize {
        assert_eq!(col_a.value(i), i as i64);
        assert_eq!(col_b.value(i), i as i64 * 2);
    }
}

#[test]
fn test_roundtrip_all_types() {
    let columns = vec![
        ("f_boolean".to_string(), DataType::Boolean, true),
        ("f_tinyint".to_string(), DataType::Int8, true),
        ("f_smallint".to_string(), DataType::Int16, true),
        ("f_int".to_string(), DataType::Int32, true),
        ("f_bigint".to_string(), DataType::Int64, true),
        ("f_float".to_string(), DataType::Float32, true),
        ("f_double".to_string(), DataType::Float64, true),
        ("f_string".to_string(), DataType::Utf8, true),
        ("f_bytes".to_string(), DataType::Binary, true),
        (
            "f_decimal_compact".to_string(),
            DataType::Decimal128(10, 2),
            true,
        ),
        ("f_date".to_string(), DataType::Date32, true),
        (
            "f_timestamp".to_string(),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ),
        (
            "f_timestamp_high".to_string(),
            DataType::Struct(
                vec![
                    arrow_schema::Field::new("millis", DataType::Int64, false),
                    arrow_schema::Field::new("nanos_of_milli", DataType::Int32, false),
                ]
                .into(),
            ),
            true,
        ),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_ZSTD,
            num_buckets: DEFAULT_NUM_BUCKETS,
            ..Default::default()
        },
    )
    .unwrap();

    let rows = vec![vec![
        Value::Boolean(true),
        Value::TinyInt(42),
        Value::SmallInt(1234),
        Value::Integer(999999),
        Value::BigInt(123456789012345),
        Value::Float(1.25),
        Value::Double(9.876543210),
        Value::String("hello world".as_bytes().to_vec()),
        Value::Bytes(vec![1, 2, 3, 4, 5]),
        Value::DecimalCompact(1234567),
        Value::Date(19000),
        Value::TimestampMillis(1700000000000),
        Value::TimestampNanos {
            millis: 1700000000000,
            nanos_of_milli: 123456,
        },
    ]];
    write_values(&mut writer, &columns, &rows);

    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    let col = |name: &str| -> usize {
        reader
            .schema()
            .columns
            .iter()
            .position(|c| c.name == name)
            .unwrap()
    };

    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 1);

    assert!(batch_col_bool(&batch, "f_boolean").value(0));
    assert_eq!(batch_col_i8(&batch, "f_tinyint").value(0), 42);
    assert_eq!(batch_col_i16(&batch, "f_smallint").value(0), 1234);
    assert_eq!(batch_col_i32(&batch, "f_int").value(0), 999999);
    assert_eq!(
        batch_col_i64(&batch, "f_bigint").value(0),
        123456789012345i64
    );
    assert!((batch_col_f32(&batch, "f_float").value(0) - 1.25).abs() < 0.001);
    assert!((batch_col_f64(&batch, "f_double").value(0) - 9.876543210).abs() < 1e-9);
    assert_eq!(batch_col_string(&batch, "f_string").value(0), "hello world");
    assert_eq!(
        batch_col_binary(&batch, "f_bytes").value(0),
        &[1u8, 2, 3, 4, 5]
    );

    let c_dec = batch
        .column(col("f_decimal_compact"))
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    assert_eq!(c_dec.value(0), 1234567i128);

    let c_date = batch
        .column(col("f_date"))
        .as_any()
        .downcast_ref::<Date32Array>()
        .unwrap();
    assert_eq!(c_date.value(0), 19000);

    let c_ts = batch
        .column(col("f_timestamp"))
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .unwrap();
    assert_eq!(c_ts.value(0), 1700000000000i64);

    let c_ts_high = batch
        .column(col("f_timestamp_high"))
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let millis_arr = c_ts_high
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let nanos_arr = c_ts_high
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(millis_arr.value(0), 1700000000000i64);
    assert_eq!(nanos_arr.value(0), 123456);
}

fn write_and_read(
    columns: Vec<(String, DataType, bool)>,
    rows: &[Vec<Value>],
) -> (MosaicReader<ByteArrayInputFile>, Vec<u8>) {
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_ZSTD,
            num_buckets: DEFAULT_NUM_BUCKETS,
            ..Default::default()
        },
    )
    .unwrap();
    write_values(&mut writer, &columns, rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data.clone()), len).unwrap();
    (reader, data)
}

/// Mirrors Paimon's testSchemaEvolutionTypeWidening.
/// Writes with narrow types (INT, FLOAT, TINYINT) and verifies the reader
/// produces the exact values that a higher layer can widen to BIGINT/DOUBLE/INT.
#[test]
fn test_evolution_type_widening_readback() {
    let columns = vec![
        ("id".to_string(), DataType::Int32, true),
        ("name".to_string(), DataType::Utf8, true),
        ("score".to_string(), DataType::Float32, true),
        ("amount".to_string(), DataType::Int8, true),
    ];
    let rows: Vec<Vec<Value>> = (0..100)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::String(format!("name_{}", i).into_bytes()),
                Value::Float(i as f32 * 0.5),
                Value::TinyInt((i % 127) as i8),
            ]
        })
        .collect();

    let (reader, _) = write_and_read(columns, &rows);

    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 100);

    let ids = batch_col_i32(&batch, "id");
    let names = batch_col_string(&batch, "name");
    let scores = batch_col_f32(&batch, "score");
    let amounts = batch_col_i8(&batch, "amount");

    for i in 0..100usize {
        assert_eq!(ids.value(i) as i64, i as i64);
        assert_eq!(names.value(i), format!("name_{}", i));
        assert!((scores.value(i) as f64 - i as f64 * 0.5).abs() < 1e-5);
        assert_eq!(amounts.value(i) as i32, i as i32 % 127);
    }
}

/// Mirrors Paimon's testSchemaEvolutionDroppedColumn.
/// Writes 3 columns, then reads back and accesses only column "c" by name,
/// ignoring "a" and "b". This is the pattern used by the adapter when the read
/// schema has fewer columns than the write schema.
#[test]
fn test_evolution_dropped_column_readback() {
    let columns = vec![
        ("a".to_string(), DataType::Int32, true),
        ("b".to_string(), DataType::Utf8, true),
        ("c".to_string(), DataType::Float64, true),
    ];
    let rows: Vec<Vec<Value>> = (0..50)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::String(format!("v{}", i).into_bytes()),
                Value::Double(i as f64 * 1.1),
            ]
        })
        .collect();

    let (reader, _) = write_and_read(columns, &rows);

    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 50);

    let cc = batch_col_f64(&batch, "c");
    for i in 0..50usize {
        assert!((cc.value(i) - i as f64 * 1.1).abs() < 1e-10);
    }
}

/// Mirrors Paimon's testSchemaEvolutionConstAndDictCast.
/// Writes data that triggers const encoding (all-same value) and dict encoding
/// (few distinct values), then reads back and verifies all values are correct.
/// The adapter layer would then widen these to BIGINT/INT/DOUBLE.
#[test]
fn test_evolution_const_and_dict_encoding_readback() {
    let columns = vec![
        ("const_col".to_string(), DataType::Int32, true),
        ("dict_col".to_string(), DataType::Int16, true),
        ("plain_col".to_string(), DataType::Float32, true),
    ];
    let rows: Vec<Vec<Value>> = (0..200)
        .map(|i| {
            vec![
                Value::Integer(42),
                Value::SmallInt((i % 5) as i16),
                Value::Float(i as f32),
            ]
        })
        .collect();

    let (reader, _) = write_and_read(columns, &rows);

    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 200);

    let const_col = batch_col_i32(&batch, "const_col");
    let dict_col = batch_col_i16(&batch, "dict_col");
    let plain_col = batch_col_f32(&batch, "plain_col");

    for i in 0..200usize {
        assert_eq!(const_col.value(i), 42);
        assert_eq!(dict_col.value(i), (i % 5) as i16);
        assert!((plain_col.value(i) - i as f32).abs() < 1e-5);
    }
}

#[test]
fn test_stats_roundtrip() {
    let columns = vec![
        ("id".to_string(), DataType::Int32, true),
        ("name".to_string(), DataType::Utf8, true),
        ("score".to_string(), DataType::Float64, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            stats_columns: vec![0, 2],
            num_buckets: 2,
            ..Default::default()
        },
    )
    .unwrap();

    let rows: Vec<Vec<Value>> = (0..100)
        .map(|i| {
            vec![
                Value::Integer(i * 2),
                Value::String(format!("row_{}", i).into_bytes()),
                Value::Double(i as f64 * 0.5),
            ]
        })
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();
    let stats = reader.row_group_stats(0).unwrap();
    assert_eq!(stats.len(), 2);

    // Column 0 (id): min=0, max=198
    assert_eq!(stats[0].column_index, 0);
    assert_eq!(stats[0].null_count, 0);
    assert!(matches!(&stats[0].min, Some(Value::Integer(0))));
    assert!(matches!(&stats[0].max, Some(Value::Integer(198))));

    // Column 2 (score): min=0.0, max=49.5
    assert_eq!(stats[1].column_index, 2);
    assert_eq!(stats[1].null_count, 0);
    match &stats[1].min {
        Some(Value::Double(v)) => assert!(v.abs() < 1e-10),
        other => panic!("expected Double(0.0), got {:?}", other),
    }
    match &stats[1].max {
        Some(Value::Double(v)) => assert!((*v - 49.5).abs() < 1e-10),
        other => panic!("expected Double(49.5), got {:?}", other),
    }
}

#[test]
fn test_stats_with_nulls() {
    let columns = vec![
        ("a".to_string(), DataType::Int32, true),
        ("b".to_string(), DataType::Int64, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            stats_columns: vec![0, 1],
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();

    let rows = vec![
        vec![Value::Integer(10), Value::Null],
        vec![Value::Null, Value::Null],
        vec![Value::Integer(5), Value::BigInt(100)],
        vec![Value::Integer(20), Value::BigInt(50)],
    ];
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();
    let stats = reader.row_group_stats(0).unwrap();
    assert_eq!(stats.len(), 2);

    assert_eq!(stats[0].column_index, 0);
    assert_eq!(stats[0].null_count, 1);
    assert!(matches!(&stats[0].min, Some(Value::Integer(5))));
    assert!(matches!(&stats[0].max, Some(Value::Integer(20))));

    assert_eq!(stats[1].column_index, 1);
    assert_eq!(stats[1].null_count, 2);
    assert!(matches!(&stats[1].min, Some(Value::BigInt(50))));
    assert!(matches!(&stats[1].max, Some(Value::BigInt(100))));
}

#[test]
fn test_stats_all_null_column() {
    let columns = vec![("x".to_string(), DataType::Int32, true)];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            stats_columns: vec![0],
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();

    let rows: Vec<Vec<Value>> = (0..10).map(|_| vec![Value::Null]).collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();
    let stats = reader.row_group_stats(0).unwrap();
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].null_count, 10);
    assert!(stats[0].min.is_none());
    assert!(stats[0].max.is_none());
}

#[test]
fn test_no_stats_minimal_overhead() {
    let columns = vec![
        ("a".to_string(), DataType::Int32, true),
        ("b".to_string(), DataType::Int32, true),
    ];

    let rows: Vec<Vec<Value>> = (0..10)
        .map(|i| vec![Value::Integer(i), Value::Integer(i * 2)])
        .collect();

    // Write without stats
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            stats_columns: vec![],
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let no_stats_size = writer.output().buf.len();

    // Write with stats on column 0
    let out2 = MemOutputFile::new();
    let mut writer2 = MosaicWriter::new(
        out2,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            stats_columns: vec![0],
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();
    write_values(&mut writer2, &columns, &rows);
    writer2.close().unwrap();
    let with_stats_size = writer2.output().buf.len();

    // No-stats file should have exactly 1 byte overhead per row group (varint 0)
    // With stats: varint(1) + varint(col_idx=0) + varint(null_count=0) + has_non_null(1) + min(4) + max(4) = ~12 bytes
    let overhead = with_stats_size - no_stats_size;
    assert!(
        overhead < 20,
        "stats overhead too large: {} bytes",
        overhead
    );
    assert!(
        overhead >= 10,
        "stats overhead suspiciously small: {} bytes",
        overhead
    );

    // Verify no-stats reader still works
    let no_stats_data = writer.output().buf.clone();
    let no_stats_len = no_stats_data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(no_stats_data), no_stats_len).unwrap();
    let stats = reader.row_group_stats(0).unwrap();
    assert!(stats.is_empty());
}

#[test]
fn test_stats_string_column() {
    let columns = vec![("s".to_string(), DataType::Utf8, true)];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            stats_columns: vec![0],
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();

    let rows = vec![
        vec![Value::String(b"banana".to_vec())],
        vec![Value::String(b"apple".to_vec())],
        vec![Value::String(b"cherry".to_vec())],
    ];
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();
    let stats = reader.row_group_stats(0).unwrap();
    assert_eq!(stats.len(), 1);
    assert!(matches!(&stats[0].min, Some(Value::String(b)) if b == b"apple"));
    assert!(matches!(&stats[0].max, Some(Value::String(b)) if b == b"cherry"));
}

#[test]
fn test_cursor_api_basic() {
    let columns = vec![
        ("id".to_string(), DataType::Int32, true),
        ("val".to_string(), DataType::Utf8, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            num_buckets: 2,
            ..Default::default()
        },
    )
    .unwrap();

    let rows: Vec<Vec<Value>> = (0..50)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::String(format!("v{}", i).into_bytes()),
            ]
        })
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 50);

    let ids = batch_col_i32(&batch, "id");
    let vals = batch_col_string(&batch, "val");

    for i in 0..50usize {
        assert_eq!(ids.value(i), i as i32);
        assert_eq!(vals.value(i), format!("v{}", i));
    }
}

#[test]
fn test_cursor_api_overwrites_previous_row() {
    let columns = vec![("x".to_string(), DataType::Int32, true)];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let rows = vec![vec![Value::Integer(100)], vec![Value::Integer(200)]];
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 2);

    let xs = batch_col_i32(&batch, "x");
    assert_eq!(xs.value(0), 100);
    assert_eq!(xs.value(1), 200);
}

#[test]
fn test_projection_subset() {
    let columns = vec![
        ("a".to_string(), DataType::Int32, true),
        ("b".to_string(), DataType::Utf8, true),
        ("c".to_string(), DataType::Float64, true),
        ("d".to_string(), DataType::Int64, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            num_buckets: 4,
            ..Default::default()
        },
    )
    .unwrap();

    let rows: Vec<Vec<Value>> = (0..30)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::String(format!("s{}", i).into_bytes()),
                Value::Double(i as f64),
                Value::BigInt(i as i64 * 100),
            ]
        })
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    let col_a = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "a")
        .unwrap();
    let col_c = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "c")
        .unwrap();

    let mut rg = reader
        .row_group_reader_projected(0, &[col_a, col_c])
        .unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 30);
    assert_eq!(batch.num_columns(), 2);

    let ca = batch_col_i32(&batch, "a");
    let cc = batch_col_f64(&batch, "c");
    for i in 0..30usize {
        assert_eq!(ca.value(i), i as i32);
        assert!((cc.value(i) - i as f64).abs() < 1e-10);
    }
}

#[test]
fn test_projection_single_column() {
    let columns = vec![
        ("x".to_string(), DataType::Int32, true),
        ("y".to_string(), DataType::Int32, true),
        ("z".to_string(), DataType::Int32, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            num_buckets: 3,
            ..Default::default()
        },
    )
    .unwrap();

    let rows: Vec<Vec<Value>> = (0..20)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::Integer(i * 10),
                Value::Integer(i * 100),
            ]
        })
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();
    let col_y = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "y")
        .unwrap();

    let mut rg = reader.row_group_reader_projected(0, &[col_y]).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 20);
    assert_eq!(batch.num_columns(), 1);

    let cy = batch_col_i32(&batch, "y");
    for i in 0..20usize {
        assert_eq!(cy.value(i), i as i32 * 10);
    }
}

#[test]
fn test_multiple_row_groups() {
    let columns = vec![
        ("id".to_string(), DataType::Int32, true),
        ("data".to_string(), DataType::Int64, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            row_group_max_size: 200,
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();

    let total_rows = 500;
    let rows: Vec<Vec<Value>> = (0..total_rows)
        .map(|i| vec![Value::Integer(i), Value::BigInt(i as i64 * 3)])
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    assert!(reader.num_row_groups() > 1, "expected multiple row groups");

    let mut offset = 0usize;
    for rg_idx in 0..reader.num_row_groups() {
        let mut rg = reader.row_group_reader(rg_idx).unwrap();
        let batch = rg.read_columns().unwrap();
        let ids = batch_col_i32(&batch, "id");
        let data = batch_col_i64(&batch, "data");
        for i in 0..batch.num_rows() {
            assert_eq!(ids.value(i), (offset + i) as i32);
            assert_eq!(data.value(i), (offset + i) as i64 * 3);
        }
        offset += batch.num_rows();
    }
    assert_eq!(offset, total_rows as usize);
}

#[test]
fn test_multiple_row_groups_with_projection() {
    let columns = vec![
        ("a".to_string(), DataType::Int32, true),
        ("b".to_string(), DataType::Utf8, true),
        ("c".to_string(), DataType::Float64, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            row_group_max_size: 100,
            num_buckets: 2,
            ..Default::default()
        },
    )
    .unwrap();

    let total_rows = 200;
    let rows: Vec<Vec<Value>> = (0..total_rows)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::String(format!("r{}", i).into_bytes()),
                Value::Double(i as f64),
            ]
        })
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    assert!(reader.num_row_groups() > 1);

    let col_c = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "c")
        .unwrap();
    let mut count = 0usize;
    for rg_idx in 0..reader.num_row_groups() {
        let mut rg = reader.row_group_reader_projected(rg_idx, &[col_c]).unwrap();
        let batch = rg.read_columns().unwrap();
        let cc = batch_col_f64(&batch, "c");
        for i in 0..batch.num_rows() {
            assert!((cc.value(i) - (count + i) as f64).abs() < 1e-10);
        }
        count += batch.num_rows();
    }
    assert_eq!(count, total_rows as usize);
}

#[test]
fn test_single_row() {
    let columns = vec![
        ("k".to_string(), DataType::Int32, true),
        ("v".to_string(), DataType::Utf8, true),
    ];
    let (reader, _) = write_and_read(
        columns,
        &[vec![Value::Integer(42), Value::String(b"only".to_vec())]],
    );
    let mut rg = reader.row_group_reader(0).unwrap();
    assert_eq!(rg.num_rows(), 1);
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch_col_i32(&batch, "k").value(0), 42);
    assert_eq!(batch_col_string(&batch, "v").value(0), "only");
}

#[test]
fn test_zero_rows() {
    let columns = vec![("a".to_string(), DataType::Int32, true)];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();
    assert_eq!(reader.num_row_groups(), 0);
}

#[test]
fn test_many_columns_few_buckets() {
    let columns: Vec<(String, DataType, bool)> = (0..50)
        .map(|i| (format!("col_{:03}", i), DataType::Int32, true))
        .collect();
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            num_buckets: 3,
            ..Default::default()
        },
    )
    .unwrap();

    let rows: Vec<Vec<Value>> = (0..20)
        .map(|r| (0..50).map(|c| Value::Integer(r * 50 + c)).collect())
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();
    assert_eq!(reader.schema().columns.len(), 50);

    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 20);
    assert_eq!(batch.num_columns(), 50);

    for r in 0..20usize {
        for c in 0..50usize {
            let col = batch
                .column(c)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            assert_eq!(col.value(r), (r as i32) * 50 + c as i32);
        }
    }
}

#[test]
fn test_single_bucket() {
    let columns = vec![
        ("x".to_string(), DataType::Int32, true),
        ("y".to_string(), DataType::Float64, true),
        ("z".to_string(), DataType::Utf8, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let rows: Vec<Vec<Value>> = (0..30)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::Double(i as f64),
                Value::String(format!("s{}", i).into_bytes()),
            ]
        })
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 30);

    let xs = batch_col_i32(&batch, "x");
    for i in 0..30usize {
        assert_eq!(xs.value(i), i as i32);
    }
}

#[test]
fn test_const_encoding_with_nulls() {
    let columns = vec![("v".to_string(), DataType::Int32, true)];
    let mut rows = Vec::new();
    for i in 0..20 {
        if i % 3 == 0 {
            rows.push(vec![Value::Null]);
        } else {
            rows.push(vec![Value::Integer(77)]);
        }
    }
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 20);

    let vs = batch_col_i32(&batch, "v");
    for i in 0..20usize {
        if i % 3 == 0 {
            assert!(vs.is_null(i));
        } else {
            assert_eq!(vs.value(i), 77);
        }
    }
}

#[test]
fn test_dict_encoding_with_nulls() {
    let columns = vec![("v".to_string(), DataType::Int32, true)];
    let mut rows = Vec::new();
    for i in 0..100 {
        if i % 5 == 0 {
            rows.push(vec![Value::Null]);
        } else {
            rows.push(vec![Value::Integer(i % 4)]);
        }
    }
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 100);

    let vs = batch_col_i32(&batch, "v");
    for i in 0..100usize {
        if i % 5 == 0 {
            assert!(vs.is_null(i));
        } else {
            assert_eq!(vs.value(i), i as i32 % 4);
        }
    }
}

#[test]
fn test_const_string_encoding() {
    let columns = vec![("s".to_string(), DataType::Utf8, true)];
    let rows: Vec<Vec<Value>> = (0..50)
        .map(|_| vec![Value::String(b"constant_value".to_vec())])
        .collect();
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 50);

    let ss = batch_col_string(&batch, "s");
    for i in 0..50usize {
        assert_eq!(ss.value(i), "constant_value");
    }
}

#[test]
fn test_dict_string_encoding() {
    let dict_values = ["apple", "banana", "cherry"];
    let columns = vec![("fruit".to_string(), DataType::Utf8, true)];
    let rows: Vec<Vec<Value>> = (0..90)
        .map(|i| vec![Value::String(dict_values[i % 3].as_bytes().to_vec())])
        .collect();
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 90);

    let fruits = batch_col_string(&batch, "fruit");
    for i in 0..90usize {
        assert_eq!(fruits.value(i), dict_values[i % 3]);
    }
}

#[test]
fn test_dict_string_encoding_with_nulls() {
    let dict_values = ["foo", "bar"];
    let columns = vec![("s".to_string(), DataType::Utf8, true)];
    let mut rows = Vec::new();
    for i in 0..60 {
        if i % 4 == 0 {
            rows.push(vec![Value::Null]);
        } else {
            rows.push(vec![Value::String(dict_values[i % 2].as_bytes().to_vec())]);
        }
    }
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 60);

    let ss = batch_col_string(&batch, "s");
    for i in 0..60usize {
        if i % 4 == 0 {
            assert!(ss.is_null(i));
        } else {
            assert_eq!(ss.value(i), dict_values[i % 2]);
        }
    }
}

#[test]
fn test_varchar_char_roundtrip() {
    let columns = vec![
        ("vc".to_string(), DataType::Utf8, true),
        ("ch".to_string(), DataType::Utf8, true),
    ];
    let rows = vec![
        vec![
            Value::String(b"hello".to_vec()),
            Value::String(b"world12345".to_vec()),
        ],
        vec![Value::String(b"test".to_vec()), Value::Null],
        vec![Value::Null, Value::String(b"abcdefghij".to_vec())],
    ];
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 3);

    let vc = batch_col_string(&batch, "vc");
    let ch = batch_col_string(&batch, "ch");

    assert_eq!(vc.value(0), "hello");
    assert_eq!(ch.value(0), "world12345");
    assert_eq!(vc.value(1), "test");
    assert!(ch.is_null(1));
    assert!(vc.is_null(2));
    assert_eq!(ch.value(2), "abcdefghij");
}

#[test]
fn test_binary_varbinary_roundtrip() {
    let columns = vec![
        ("bin".to_string(), DataType::Binary, true),
        ("vbin".to_string(), DataType::Binary, true),
    ];
    let rows = vec![
        vec![
            Value::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]),
            Value::Bytes(vec![1, 2, 3]),
        ],
        vec![Value::Null, Value::Bytes(vec![0xFF; 100])],
    ];
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 2);

    let bin = batch_col_binary(&batch, "bin");
    let vbin = batch_col_binary(&batch, "vbin");

    assert_eq!(bin.value(0), &[0xDE, 0xAD, 0xBE, 0xEF]);
    assert_eq!(vbin.value(0), &[1, 2, 3]);
    assert!(bin.is_null(1));
    assert_eq!(vbin.value(1).len(), 100);
    assert!(vbin.value(1).iter().all(|&x| x == 0xFF));
}

#[test]
fn test_time_roundtrip() {
    let columns = vec![(
        "t".to_string(),
        DataType::Time32(TimeUnit::Millisecond),
        true,
    )];
    let rows = vec![
        vec![Value::Time(0)],
        vec![Value::Time(43200000)],
        vec![Value::Null],
        vec![Value::Time(86399999)],
    ];
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 4);

    let ts = batch
        .column(0)
        .as_any()
        .downcast_ref::<Time32MillisecondArray>()
        .unwrap();
    assert_eq!(ts.value(0), 0);
    assert_eq!(ts.value(1), 43200000);
    assert!(ts.is_null(2));
    assert_eq!(ts.value(3), 86399999);
}

#[test]
fn test_timestamp_ltz_roundtrip() {
    let columns = vec![
        (
            "ts_millis".to_string(),
            DataType::Timestamp(TimeUnit::Millisecond, Some("Asia/Shanghai".into())),
            true,
        ),
        (
            "ts_micros".to_string(),
            DataType::Timestamp(TimeUnit::Microsecond, Some("America/New_York".into())),
            true,
        ),
    ];
    let rows = vec![
        vec![
            Value::TimestampMillis(1700000000000),
            Value::TimestampMicros(1700000000000000),
        ],
        vec![Value::Null, Value::Null],
        vec![Value::TimestampMillis(0), Value::TimestampMicros(0)],
    ];
    let (reader, _) = write_and_read(columns, &rows);

    let schema = reader.schema();
    let dt0 = &schema.columns[0].data_type;
    let dt1 = &schema.columns[1].data_type;
    assert_eq!(
        *dt0,
        DataType::Timestamp(TimeUnit::Millisecond, Some("Asia/Shanghai".into()))
    );
    assert_eq!(
        *dt1,
        DataType::Timestamp(TimeUnit::Microsecond, Some("America/New_York".into()))
    );

    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 3);

    let ts0 = batch
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .unwrap();
    assert_eq!(ts0.value(0), 1700000000000i64);
    assert!(ts0.is_null(1));
    assert_eq!(ts0.value(2), 0);
    assert_eq!(ts0.timezone().unwrap(), "Asia/Shanghai");

    let ts1 = batch
        .column(1)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    assert_eq!(ts1.value(0), 1700000000000000i64);
    assert!(ts1.is_null(1));
    assert_eq!(ts1.value(2), 0);
    assert_eq!(ts1.timezone().unwrap(), "America/New_York");
}

#[test]
fn test_timestamp_micros_roundtrip() {
    let columns = vec![
        (
            "ts_millis".to_string(),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ),
        (
            "ts_micros".to_string(),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        (
            "ts_nanos".to_string(),
            DataType::Struct(
                vec![
                    arrow_schema::Field::new("millis", DataType::Int64, false),
                    arrow_schema::Field::new("nanos_of_milli", DataType::Int32, false),
                ]
                .into(),
            ),
            true,
        ),
    ];
    let rows = vec![
        vec![
            Value::TimestampMillis(1700000000000),
            Value::TimestampMicros(1_700_000_000_000_000),
            Value::TimestampNanos {
                millis: 1700000000000,
                nanos_of_milli: 123456,
            },
        ],
        vec![Value::Null, Value::Null, Value::Null],
        vec![
            Value::TimestampMillis(0),
            Value::TimestampMicros(0),
            Value::TimestampNanos {
                millis: 0,
                nanos_of_milli: 0,
            },
        ],
    ];
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 3);

    let ts_millis = batch
        .column(0)
        .as_any()
        .downcast_ref::<TimestampMillisecondArray>()
        .unwrap();
    let ts_micros = batch
        .column(1)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    let ts_nanos = batch
        .column(2)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let millis = ts_nanos
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let nanos = ts_nanos
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();

    assert_eq!(ts_millis.value(0), 1700000000000i64);
    assert_eq!(ts_micros.value(0), 1_700_000_000_000_000i64);
    assert_eq!(millis.value(0), 1700000000000i64);
    assert_eq!(nanos.value(0), 123456);

    assert!(ts_millis.is_null(1));
    assert!(ts_micros.is_null(1));
    assert!(ts_nanos.is_null(1));

    assert_eq!(ts_millis.value(2), 0);
    assert_eq!(ts_micros.value(2), 0);
}

#[test]
fn test_decimal_large_roundtrip() {
    let columns = vec![("d".to_string(), DataType::Decimal128(30, 5), true)];
    let large_bytes = vec![
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x02, 0x03, 0x04,
    ];
    let rows = vec![
        vec![Value::DecimalLarge(large_bytes.clone())],
        vec![Value::Null],
        vec![Value::DecimalLarge(vec![0xFF])],
    ];
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 3);

    let ds = batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    assert_eq!(ds.value(0), biginteger_bytes_to_i128(&large_bytes));
    assert!(ds.is_null(1));
    assert_eq!(ds.value(2), biginteger_bytes_to_i128(&[0xFF]));
}

#[test]
fn test_writer_close_idempotent() {
    let columns = vec![("a".to_string(), DataType::Int32, true)];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();
    write_values(&mut writer, &columns, &[vec![Value::Integer(1)]]);
    writer.close().unwrap();
    let size_after_first_close = writer.output().buf.len();

    writer.close().unwrap();
    let size_after_second_close = writer.output().buf.len();
    assert_eq!(size_after_first_close, size_after_second_close);
}

#[test]
fn test_all_null_rows() {
    let columns = vec![
        ("a".to_string(), DataType::Int32, true),
        ("b".to_string(), DataType::Utf8, true),
        ("c".to_string(), DataType::Float64, true),
    ];
    let rows: Vec<Vec<Value>> = (0..30)
        .map(|_| vec![Value::Null, Value::Null, Value::Null])
        .collect();
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 30);

    for c in 0..batch.num_columns() {
        for r in 0..30usize {
            assert!(batch.column(c).is_null(r));
        }
    }
}

#[test]
fn test_mixed_encodings_multi_column() {
    let columns = vec![
        ("all_null".to_string(), DataType::Int32, true),
        ("const_int".to_string(), DataType::Int64, true),
        ("dict_str".to_string(), DataType::Utf8, true),
        ("plain_dbl".to_string(), DataType::Float64, true),
    ];
    let dict_vals = ["alpha", "beta", "gamma"];
    let mut rows = Vec::new();
    for i in 0..100 {
        rows.push(vec![
            Value::Null,
            Value::BigInt(999),
            Value::String(dict_vals[i % 3].as_bytes().to_vec()),
            Value::Double(i as f64 * 0.1),
        ]);
    }
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 100);

    let col0 = batch_col_i32(&batch, "all_null");
    let col1 = batch_col_i64(&batch, "const_int");
    let col2 = batch_col_string(&batch, "dict_str");
    let col3 = batch_col_f64(&batch, "plain_dbl");

    for i in 0..100usize {
        assert!(col0.is_null(i));
        assert_eq!(col1.value(i), 999);
        assert_eq!(col2.value(i), dict_vals[i % 3]);
        assert!((col3.value(i) - i as f64 * 0.1).abs() < 1e-10);
    }
}

#[test]
fn test_cursor_api_with_projection() {
    let columns = vec![
        ("a".to_string(), DataType::Int32, true),
        ("b".to_string(), DataType::Utf8, true),
        ("c".to_string(), DataType::Int64, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            num_buckets: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let rows: Vec<Vec<Value>> = (0..40)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::String(format!("x{}", i).into_bytes()),
                Value::BigInt(i as i64 * 7),
            ]
        })
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    let col_b = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "b")
        .unwrap();

    let mut rg = reader.row_group_reader_projected(0, &[col_b]).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 40);

    let bs = batch_col_string(&batch, "b");
    for i in 0..40usize {
        assert_eq!(bs.value(i), format!("x{}", i));
    }
}

#[test]
fn test_boolean_all_true() {
    let columns = vec![("b".to_string(), DataType::Boolean, true)];
    let rows: Vec<Vec<Value>> = (0..20).map(|_| vec![Value::Boolean(true)]).collect();
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 20);

    let bs = batch_col_bool(&batch, "b");
    for i in 0..20usize {
        assert!(bs.value(i));
    }
}

#[test]
fn test_boolean_alternating() {
    let columns = vec![("b".to_string(), DataType::Boolean, true)];
    let rows: Vec<Vec<Value>> = (0..40).map(|i| vec![Value::Boolean(i % 2 == 0)]).collect();
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 40);

    let bs = batch_col_bool(&batch, "b");
    for i in 0..40usize {
        assert_eq!(bs.value(i), i % 2 == 0);
    }
}

#[test]
fn test_zstd_with_multiple_row_groups() {
    let columns = vec![
        ("id".to_string(), DataType::Int32, true),
        ("name".to_string(), DataType::Utf8, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_ZSTD,
            zstd_level: 1,
            row_group_max_size: 200,
            num_buckets: 2,
            ..Default::default()
        },
    )
    .unwrap();

    let total = 300;
    let rows: Vec<Vec<Value>> = (0..total)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::String(format!("name_{}", i).into_bytes()),
            ]
        })
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    assert!(reader.num_row_groups() > 1);

    let mut count = 0usize;
    for rg_idx in 0..reader.num_row_groups() {
        let mut rg = reader.row_group_reader(rg_idx).unwrap();
        let batch = rg.read_columns().unwrap();
        let ids = batch_col_i32(&batch, "id");
        let names = batch_col_string(&batch, "name");
        for i in 0..batch.num_rows() {
            assert_eq!(ids.value(i), (count + i) as i32);
            assert_eq!(names.value(i), format!("name_{}", count + i));
        }
        count += batch.num_rows();
    }
    assert_eq!(count, total as usize);
}

#[test]
fn test_stats_across_multiple_row_groups() {
    let columns = vec![("v".to_string(), DataType::Int32, true)];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            row_group_max_size: 100,
            stats_columns: vec![0],
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();

    let rows: Vec<Vec<Value>> = (0..500).map(|i| vec![Value::Integer(i)]).collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    assert!(reader.num_row_groups() > 1);

    let mut prev_max = i32::MIN;
    for rg_idx in 0..reader.num_row_groups() {
        let stats = reader.row_group_stats(rg_idx).unwrap();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].null_count, 0);
        if let (Some(Value::Integer(min)), Some(Value::Integer(max))) =
            (&stats[0].min, &stats[0].max)
        {
            assert!(*min >= prev_max || rg_idx == 0);
            assert!(*max >= *min);
            prev_max = *max;
        } else {
            panic!("expected Integer stats");
        }
    }
}

#[test]
fn test_schema_order_preserved_after_roundtrip() {
    let columns = vec![
        ("name".to_string(), DataType::Utf8, true),
        ("age".to_string(), DataType::Int32, true),
        ("score".to_string(), DataType::Float64, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            stats_columns: vec![1],
            num_buckets: 2,
            ..Default::default()
        },
    )
    .unwrap();

    let rows: Vec<Vec<Value>> = (0..50)
        .map(|i| {
            vec![
                Value::String(format!("user_{}", i).into_bytes()),
                Value::Integer(20 + i),
                Value::Double(i as f64 * 0.5),
            ]
        })
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    assert_eq!(reader.schema().columns[0].name, "name");
    assert_eq!(reader.schema().columns[1].name, "age");
    assert_eq!(reader.schema().columns[2].name, "score");

    let stats = reader.row_group_stats(0).unwrap();
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].column_index, 1);
    assert!(matches!(&stats[0].min, Some(Value::Integer(20))));
    assert!(matches!(&stats[0].max, Some(Value::Integer(69))));

    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch_col_string(&batch, "name").value(0), "user_0");
    assert_eq!(batch_col_i32(&batch, "age").value(0), 20);
}

#[test]
fn test_write_batch_non_nullable_rejects_nulls() {
    let columns = vec![
        ("a".to_string(), DataType::Int32, false),
        ("b".to_string(), DataType::Int32, false),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            num_buckets: 2,
            ..Default::default()
        },
    )
    .unwrap();

    let bad_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![None, Some(1)])),
            Arc::new(Int32Array::from(vec![Some(2), Some(3)])),
        ],
    )
    .unwrap();
    assert!(writer.write_batch(&bad_batch).is_err());

    let good_rows = vec![vec![Value::Integer(222), Value::Integer(333)]];
    write_values(&mut writer, &columns, &good_rows);

    writer.close().unwrap();
    let data = &writer.output().buf;

    let input = ByteArrayInputFile::new(data.clone());
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch_col_i32(&batch, "a").value(0), 222);
    assert_eq!(batch_col_i32(&batch, "b").value(0), 333);
}

fn write_and_read_paged(
    columns: Vec<(String, DataType, bool)>,
    rows: &[Vec<Value>],
) -> (MosaicReader<ByteArrayInputFile>, Vec<u8>) {
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_ZSTD,
            page_size_threshold: 1,
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();
    write_values(&mut writer, &columns, rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data.clone()), len).unwrap();
    (reader, data)
}

#[test]
fn test_paged_roundtrip_basic() {
    let columns = vec![
        ("id".to_string(), DataType::Int32, true),
        ("name".to_string(), DataType::Utf8, true),
        ("score".to_string(), DataType::Float64, true),
    ];
    let rows: Vec<Vec<Value>> = (0..200)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::String(format!("user_{}", i).into_bytes()),
                Value::Double(i as f64 * 1.5),
            ]
        })
        .collect();

    let (reader, _) = write_and_read_paged(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 200);

    let ids = batch_col_i32(&batch, "id");
    let names = batch_col_string(&batch, "name");
    let scores = batch_col_f64(&batch, "score");

    for i in 0..200usize {
        assert_eq!(ids.value(i), i as i32);
        assert_eq!(names.value(i), format!("user_{}", i));
        assert!((scores.value(i) - i as f64 * 1.5).abs() < 1e-10);
    }
}

#[test]
fn test_paged_projection() {
    let columns = vec![
        ("a".to_string(), DataType::Int32, true),
        ("b".to_string(), DataType::Utf8, true),
        ("c".to_string(), DataType::Float64, true),
        ("d".to_string(), DataType::Int64, true),
    ];
    let rows: Vec<Vec<Value>> = (0..100)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::String(format!("s{}", i).into_bytes()),
                Value::Double(i as f64),
                Value::BigInt(i as i64 * 100),
            ]
        })
        .collect();

    let (reader, _) = write_and_read_paged(columns, &rows);
    let col_a = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "a")
        .unwrap();
    let col_c = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "c")
        .unwrap();

    let mut rg = reader
        .row_group_reader_projected(0, &[col_a, col_c])
        .unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 100);
    assert_eq!(batch.num_columns(), 2);

    let ca = batch_col_i32(&batch, "a");
    let cc = batch_col_f64(&batch, "c");
    for i in 0..100usize {
        assert_eq!(ca.value(i), i as i32);
        assert!((cc.value(i) - i as f64).abs() < 1e-10);
    }
}

#[test]
fn test_paged_mixed_encodings() {
    let columns = vec![
        ("all_null".to_string(), DataType::Int32, true),
        ("const_int".to_string(), DataType::Int64, true),
        ("dict_str".to_string(), DataType::Utf8, true),
        ("plain_dbl".to_string(), DataType::Float64, true),
    ];
    let dict_vals = ["alpha", "beta", "gamma"];
    let rows: Vec<Vec<Value>> = (0..100)
        .map(|i| {
            vec![
                Value::Null,
                Value::BigInt(999),
                Value::String(dict_vals[i % 3].as_bytes().to_vec()),
                Value::Double(i as f64 * 0.1),
            ]
        })
        .collect();

    let (reader, _) = write_and_read_paged(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 100);

    let col0 = batch_col_i32(&batch, "all_null");
    let col1 = batch_col_i64(&batch, "const_int");
    let col2 = batch_col_string(&batch, "dict_str");
    let col3 = batch_col_f64(&batch, "plain_dbl");

    for i in 0..100usize {
        assert!(col0.is_null(i));
        assert_eq!(col1.value(i), 999);
        assert_eq!(col2.value(i), dict_vals[i % 3]);
        assert!((col3.value(i) - i as f64 * 0.1).abs() < 1e-10);
    }
}

#[test]
fn test_paged_const_with_nulls() {
    let columns = vec![("v".to_string(), DataType::Int32, true)];
    let rows: Vec<Vec<Value>> = (0..50)
        .map(|i| {
            if i % 3 == 0 {
                vec![Value::Null]
            } else {
                vec![Value::Integer(42)]
            }
        })
        .collect();

    let (reader, _) = write_and_read_paged(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 50);

    let vs = batch_col_i32(&batch, "v");
    for i in 0..50usize {
        if i % 3 == 0 {
            assert!(vs.is_null(i));
        } else {
            assert_eq!(vs.value(i), 42);
        }
    }
}

#[test]
fn test_paged_dict_with_nulls() {
    let columns = vec![("v".to_string(), DataType::Int32, true)];
    let rows: Vec<Vec<Value>> = (0..100)
        .map(|i| {
            if i % 5 == 0 {
                vec![Value::Null]
            } else {
                vec![Value::Integer(i % 3)]
            }
        })
        .collect();

    let (reader, _) = write_and_read_paged(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 100);

    let vs = batch_col_i32(&batch, "v");
    for i in 0..100usize {
        if i % 5 == 0 {
            assert!(vs.is_null(i));
        } else {
            assert_eq!(vs.value(i), i as i32 % 3);
        }
    }
}

#[test]
fn test_paged_multiple_row_groups() {
    let columns = vec![
        ("id".to_string(), DataType::Int32, true),
        ("data".to_string(), DataType::Int64, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_ZSTD,
            page_size_threshold: 1,
            row_group_max_size: 200,
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();

    let total_rows = 500;
    let rows: Vec<Vec<Value>> = (0..total_rows)
        .map(|i| vec![Value::Integer(i), Value::BigInt(i as i64 * 3)])
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    assert!(reader.num_row_groups() > 1);

    let mut offset = 0usize;
    for rg_idx in 0..reader.num_row_groups() {
        let mut rg = reader.row_group_reader(rg_idx).unwrap();
        let batch = rg.read_columns().unwrap();
        let ids = batch_col_i32(&batch, "id");
        let data = batch_col_i64(&batch, "data");
        for i in 0..batch.num_rows() {
            assert_eq!(ids.value(i), (offset + i) as i32);
            assert_eq!(data.value(i), (offset + i) as i64 * 3);
        }
        offset += batch.num_rows();
    }
    assert_eq!(offset, total_rows as usize);
}

#[test]
fn test_paged_string_dict_encoding() {
    let dict_values = ["apple", "banana", "cherry"];
    let columns = vec![("fruit".to_string(), DataType::Utf8, true)];
    let rows: Vec<Vec<Value>> = (0..90)
        .map(|i| vec![Value::String(dict_values[i % 3].as_bytes().to_vec())])
        .collect();
    let (reader, _) = write_and_read_paged(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 90);

    let fruits = batch_col_string(&batch, "fruit");
    for i in 0..90usize {
        assert_eq!(fruits.value(i), dict_values[i % 3]);
    }
}

#[test]
fn test_read_columns_basic() {
    let columns = vec![
        ("age".to_string(), DataType::Int32, true),
        ("name".to_string(), DataType::Utf8, true),
        ("score".to_string(), DataType::Float64, true),
    ];
    let rows: Vec<Vec<Value>> = (0..100)
        .map(|i| {
            vec![
                Value::Integer(20 + (i % 50)),
                Value::String(format!("user_{}", i).into_bytes()),
                Value::Double(i as f64 * 1.5),
            ]
        })
        .collect();
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();

    assert_eq!(batch.num_rows(), 100);
    assert_eq!(batch.num_columns(), 3);

    let ages = batch_col_i32(&batch, "age");
    let names = batch_col_string(&batch, "name");
    let scores = batch_col_f64(&batch, "score");

    assert_eq!(ages.null_count(), 0);

    for i in 0..100 {
        assert_eq!(ages.value(i), 20 + (i as i32 % 50));
        assert_eq!(names.value(i), format!("user_{}", i));
        assert!((scores.value(i) - i as f64 * 1.5).abs() < 1e-10);
    }
}

#[test]
fn test_read_columns_with_nulls() {
    let columns = vec![
        ("id".to_string(), DataType::Int32, true),
        ("name".to_string(), DataType::Utf8, true),
        ("value".to_string(), DataType::Float64, true),
    ];
    let rows = vec![
        vec![
            Value::Integer(1),
            Value::String(b"hello".to_vec()),
            Value::Double(1.0),
        ],
        vec![Value::Integer(2), Value::Null, Value::Double(2.0)],
        vec![
            Value::Integer(3),
            Value::String(b"world".to_vec()),
            Value::Null,
        ],
        vec![Value::Integer(4), Value::Null, Value::Null],
    ];
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();

    let ids = batch_col_i32(&batch, "id");
    let names = batch_col_string(&batch, "name");
    let values = batch_col_f64(&batch, "value");

    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
    assert_eq!(ids.value(2), 3);
    assert_eq!(ids.value(3), 4);

    assert!(!names.is_null(0));
    assert_eq!(names.value(0), "hello");
    assert!(names.is_null(1));
    assert!(!names.is_null(2));
    assert_eq!(names.value(2), "world");
    assert!(names.is_null(3));

    assert!(!values.is_null(0));
    assert!(!values.is_null(1));
    assert!(values.is_null(2));
    assert!(values.is_null(3));
}

#[test]
fn test_read_columns_with_projection() {
    let columns = vec![
        ("a".to_string(), DataType::Int32, true),
        ("b".to_string(), DataType::Utf8, true),
        ("c".to_string(), DataType::Float64, true),
        ("d".to_string(), DataType::Int64, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_ZSTD,
            num_buckets: 4,
            ..Default::default()
        },
    )
    .unwrap();
    let rows: Vec<Vec<Value>> = (0..30)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::String(format!("s{}", i).into_bytes()),
                Value::Double(i as f64),
                Value::BigInt(i as i64 * 100),
            ]
        })
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    let col_a = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "a")
        .unwrap();
    let col_c = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "c")
        .unwrap();

    let mut rg = reader
        .row_group_reader_projected(0, &[col_a, col_c])
        .unwrap();
    let batch = rg.read_columns().unwrap();

    assert_eq!(batch.num_columns(), 2);
    assert!(batch.schema().index_of("a").is_ok());
    assert!(batch.schema().index_of("c").is_ok());

    let ca = batch_col_i32(&batch, "a");
    let cc = batch_col_f64(&batch, "c");
    for i in 0..30 {
        assert_eq!(ca.value(i), i as i32);
        assert!((cc.value(i) - i as f64).abs() < 1e-10);
    }
}

#[test]
fn test_read_columns_paged() {
    let columns = vec![
        ("id".to_string(), DataType::Int32, true),
        ("name".to_string(), DataType::Utf8, true),
        ("score".to_string(), DataType::Float64, true),
    ];
    let rows: Vec<Vec<Value>> = (0..200)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::String(format!("user_{}", i).into_bytes()),
                Value::Double(i as f64 * 1.5),
            ]
        })
        .collect();
    let (reader, _) = write_and_read_paged(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();

    let ids = batch_col_i32(&batch, "id");
    let names = batch_col_string(&batch, "name");
    let scores = batch_col_f64(&batch, "score");

    for i in 0..200 {
        assert_eq!(ids.value(i), i as i32);
        assert_eq!(names.value(i), format!("user_{}", i));
        assert!((scores.value(i) - i as f64 * 1.5).abs() < 1e-10);
    }
}

#[test]
fn test_read_columns_mixed_encodings() {
    let columns = vec![
        ("all_null".to_string(), DataType::Int32, true),
        ("const_int".to_string(), DataType::Int64, true),
        ("dict_str".to_string(), DataType::Utf8, true),
        ("plain_dbl".to_string(), DataType::Float64, true),
    ];
    let dict_vals = ["alpha", "beta", "gamma"];
    let rows: Vec<Vec<Value>> = (0..100)
        .map(|i| {
            vec![
                Value::Null,
                Value::BigInt(999),
                Value::String(dict_vals[i % 3].as_bytes().to_vec()),
                Value::Double(i as f64 * 0.1),
            ]
        })
        .collect();
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();

    let col0 = batch_col_i32(&batch, "all_null");
    assert_eq!(col0.len(), 100);
    for i in 0..100 {
        assert!(col0.is_null(i));
    }

    let col1 = batch_col_i64(&batch, "const_int");
    for i in 0..100 {
        assert_eq!(col1.value(i), 999);
    }

    let col2 = batch_col_string(&batch, "dict_str");
    for i in 0..100 {
        assert_eq!(col2.value(i), dict_vals[i % 3]);
    }

    let col3 = batch_col_f64(&batch, "plain_dbl");
    for i in 0..100 {
        assert!((col3.value(i) - i as f64 * 0.1).abs() < 1e-10);
    }
}

#[test]
fn test_read_columns_binary_offsets() {
    let columns = vec![("s".to_string(), DataType::Utf8, true)];
    let rows: Vec<Vec<Value>> = vec![
        vec![Value::String(b"hello".to_vec())],
        vec![Value::String(b"world!".to_vec())],
        vec![Value::String(b"".to_vec())],
        vec![Value::String(b"test".to_vec())],
    ];
    let (reader, _) = write_and_read(columns, &rows);
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();

    let col = batch_col_string(&batch, "s");
    assert_eq!(col.value(0), "hello");
    assert_eq!(col.value(1), "world!");
    assert_eq!(col.value(2), "");
    assert_eq!(col.value(3), "test");
}

#[test]
fn test_read_columns_multiple_row_groups() {
    let columns = vec![
        ("id".to_string(), DataType::Int32, true),
        ("data".to_string(), DataType::Int64, true),
    ];
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_NONE,
            row_group_max_size: 200,
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();

    let total_rows = 500;
    let rows: Vec<Vec<Value>> = (0..total_rows)
        .map(|i| vec![Value::Integer(i), Value::BigInt(i as i64 * 3)])
        .collect();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    assert!(reader.num_row_groups() > 1);

    let mut all_id = Vec::new();
    let mut all_data = Vec::new();
    for rg_idx in 0..reader.num_row_groups() {
        let mut rg = reader.row_group_reader(rg_idx).unwrap();
        let batch = rg.read_columns().unwrap();

        let ids = batch_col_i32(&batch, "id");
        let data = batch_col_i64(&batch, "data");

        for i in 0..ids.len() {
            all_id.push(ids.value(i));
            all_data.push(data.value(i));
        }
    }

    assert_eq!(all_id.len(), total_rows as usize);
    assert_eq!(all_data.len(), total_rows as usize);
    for i in 0..total_rows as usize {
        assert_eq!(all_id[i], i as i32);
        assert_eq!(all_data[i], i as i64 * 3);
    }
}

// ====================== read_ranges coalescing tests ======================

#[test]
fn test_read_ranges_coalesces_adjacent() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingInputFile {
        data: Vec<u8>,
        read_count: AtomicUsize,
    }

    impl InputFile for CountingInputFile {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            self.read_count.fetch_add(1, Ordering::Relaxed);
            let start = offset as usize;
            buf.copy_from_slice(&self.data[start..start + buf.len()]);
            Ok(())
        }
    }

    let data = vec![0u8; 4096];
    let input = CountingInputFile {
        data,
        read_count: AtomicUsize::new(0),
    };

    // 4 adjacent ranges (gap = 0) should merge into 1 read_at call
    let ranges = vec![(0u64, 100usize), (100, 200), (300, 100), (400, 200)];
    let results = input.read_ranges(&ranges).unwrap();
    assert_eq!(results.len(), 4);
    assert_eq!(results[0].len(), 100);
    assert_eq!(results[1].len(), 200);
    assert_eq!(results[2].len(), 100);
    assert_eq!(results[3].len(), 200);
    assert_eq!(input.read_count.load(Ordering::Relaxed), 1);
}

#[test]
fn test_read_ranges_shared_reuses_coalesced_buffer() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingInputFile {
        data: Vec<u8>,
        read_count: AtomicUsize,
    }

    impl InputFile for CountingInputFile {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            self.read_count.fetch_add(1, Ordering::Relaxed);
            let start = offset as usize;
            buf.copy_from_slice(&self.data[start..start + buf.len()]);
            Ok(())
        }
    }

    let data: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
    let input = CountingInputFile {
        data: data.clone(),
        read_count: AtomicUsize::new(0),
    };

    let ranges = vec![(0u64, 64usize), (64, 64), (200, 32)];
    let results = input.read_ranges_shared(&ranges).unwrap();

    assert_eq!(input.read_count.load(Ordering::Relaxed), 1);
    assert_eq!(results[0].as_slice(), &data[0..64]);
    assert_eq!(results[1].as_slice(), &data[64..128]);
    assert_eq!(results[2].as_slice(), &data[200..232]);
    assert!(Arc::ptr_eq(&results[0].data, &results[1].data));
    assert!(Arc::ptr_eq(&results[0].data, &results[2].data));
}

#[test]
fn test_read_range_buffer_new_validates_bounds() {
    let data = Arc::new(vec![1, 2, 3, 4]);

    let buffer = ReadRangeBuffer::new(data.clone(), 1, 2).unwrap();
    assert_eq!(buffer.as_slice(), &[2, 3]);

    match ReadRangeBuffer::new(data, 3, 2) {
        Ok(_) => panic!("expected out-of-bounds range to fail"),
        Err(err) => assert_eq!(err.kind(), io::ErrorKind::InvalidInput),
    }
}

#[test]
fn test_read_ranges_splits_large_gap() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingInputFile {
        data: Vec<u8>,
        read_count: AtomicUsize,
    }

    impl InputFile for CountingInputFile {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            self.read_count.fetch_add(1, Ordering::Relaxed);
            let start = offset as usize;
            buf.copy_from_slice(&self.data[start..start + buf.len()]);
            Ok(())
        }
    }

    // Gap > 1MB between two ranges — should NOT merge
    let size = 2 * 1024 * 1024 + 200;
    let data = vec![42u8; size];
    let input = CountingInputFile {
        data,
        read_count: AtomicUsize::new(0),
    };

    let gap = COALESCE_GAP as usize + 1;
    let ranges = vec![(0u64, 100usize), ((100 + gap) as u64, 100)];
    let results = input.read_ranges(&ranges).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], vec![42u8; 100]);
    assert_eq!(results[1], vec![42u8; 100]);
    assert_eq!(input.read_count.load(Ordering::Relaxed), 2);
}

#[test]
fn test_read_ranges_coalesces_small_gap() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingInputFile {
        data: Vec<u8>,
        read_count: AtomicUsize,
    }

    impl InputFile for CountingInputFile {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            self.read_count.fetch_add(1, Ordering::Relaxed);
            let start = offset as usize;
            buf.copy_from_slice(&self.data[start..start + buf.len()]);
            Ok(())
        }
    }

    // Gap = 512KB (< 1MB) — should merge into 1 read
    let gap: usize = 512 * 1024;
    let total = 100 + gap + 100;
    let data: Vec<u8> = (0..total).map(|i| (i % 256) as u8).collect();
    let input = CountingInputFile {
        data: data.clone(),
        read_count: AtomicUsize::new(0),
    };

    let ranges = vec![(0u64, 100usize), ((100 + gap) as u64, 100)];
    let results = input.read_ranges(&ranges).unwrap();
    assert_eq!(results[0], &data[0..100]);
    assert_eq!(results[1], &data[100 + gap..100 + gap + 100]);
    assert_eq!(input.read_count.load(Ordering::Relaxed), 1);
}

#[test]
fn test_read_ranges_out_of_order() {
    let data: Vec<u8> = (0..200).map(|i| i as u8).collect();
    let input = ByteArrayInputFile::new(data.clone());

    // Ranges not in file order — results should still match original order
    let ranges = vec![(150u64, 50usize), (0, 50), (50, 50)];
    let results = input.read_ranges(&ranges).unwrap();
    assert_eq!(results[0], &data[150..200]);
    assert_eq!(results[1], &data[0..50]);
    assert_eq!(results[2], &data[50..100]);
}

#[test]
fn test_read_ranges_empty() {
    let input = ByteArrayInputFile::new(vec![0u8; 100]);
    let results = input.read_ranges(&[]).unwrap();
    assert!(results.is_empty());
}

// ====================== Paged bucket multi-bucket tests ======================

#[test]
fn test_paged_multi_bucket_projection() {
    let columns = vec![
        ("a".to_string(), DataType::Int32, true),
        ("b".to_string(), DataType::Int64, true),
        ("c".to_string(), DataType::Float64, true),
        ("d".to_string(), DataType::Utf8, true),
        ("e".to_string(), DataType::Int32, true),
        ("f".to_string(), DataType::Int64, true),
    ];
    let rows: Vec<Vec<Value>> = (0..200)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::BigInt(i as i64 * 2),
                Value::Double(i as f64 * 0.5),
                Value::String(format!("v{}", i).into_bytes()),
                Value::Integer(i * 10),
                Value::BigInt(i as i64 * 100),
            ]
        })
        .collect();

    // Use 3 buckets to spread columns across multiple paged buckets
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_ZSTD,
            page_size_threshold: 1,
            num_buckets: 3,
            ..Default::default()
        },
    )
    .unwrap();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    // Project only "a" and "f" — should hit different buckets
    let col_a = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "a")
        .unwrap();
    let col_f = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "f")
        .unwrap();

    let mut rg = reader
        .row_group_reader_projected(0, &[col_a, col_f])
        .unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 200);
    assert_eq!(batch.num_columns(), 2);

    let ca = batch_col_i32(&batch, "a");
    let cf = batch_col_i64(&batch, "f");
    for i in 0..200usize {
        assert_eq!(ca.value(i), i as i32);
        assert_eq!(cf.value(i), i as i64 * 100);
    }
}

#[test]
fn test_paged_wide_table_sparse_projection() {
    // 20 columns, project only 2 — tests that skipped columns aren't read
    let num_cols = 20;
    let columns: Vec<(String, DataType, bool)> = (0..num_cols)
        .map(|i| (format!("col_{}", i), DataType::Int64, true))
        .collect();
    let rows: Vec<Vec<Value>> = (0..100)
        .map(|r| {
            (0..num_cols)
                .map(|c| Value::BigInt((r * num_cols + c) as i64))
                .collect()
        })
        .collect();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_ZSTD,
            page_size_threshold: 1,
            num_buckets: 4,
            ..Default::default()
        },
    )
    .unwrap();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    // Project col_3 and col_17
    let idx3 = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "col_3")
        .unwrap();
    let idx17 = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "col_17")
        .unwrap();

    let mut rg = reader
        .row_group_reader_projected(0, &[idx3, idx17])
        .unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 100);
    assert_eq!(batch.num_columns(), 2);

    let c3 = batch_col_i64(&batch, "col_3");
    let c17 = batch_col_i64(&batch, "col_17");
    for r in 0..100usize {
        assert_eq!(c3.value(r), (r * num_cols + 3) as i64);
        assert_eq!(c17.value(r), (r * num_cols + 17) as i64);
    }
}

#[test]
fn test_mixed_paged_and_monolithic_buckets() {
    // Create a scenario where some buckets are paged and others are monolithic.
    // Use page_size_threshold high enough that small buckets stay monolithic.
    let columns = vec![
        ("tiny".to_string(), DataType::Int32, true),
        ("big_a".to_string(), DataType::Int64, true),
        ("big_b".to_string(), DataType::Float64, true),
        ("big_c".to_string(), DataType::Utf8, true),
    ];
    let rows: Vec<Vec<Value>> = (0..500)
        .map(|i| {
            vec![
                Value::Integer(i),
                Value::BigInt(i as i64 * 7),
                Value::Double(i as f64 / 3.0),
                Value::String(format!("str_{:05}", i).into_bytes()),
            ]
        })
        .collect();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_ZSTD,
            page_size_threshold: 4096,
            num_buckets: 4,
            ..Default::default()
        },
    )
    .unwrap();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    // Read all columns
    let mut rg = reader.row_group_reader(0).unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 500);

    let col_tiny = batch_col_i32(&batch, "tiny");
    let col_a = batch_col_i64(&batch, "big_a");
    let col_b = batch_col_f64(&batch, "big_b");
    let col_c = batch_col_string(&batch, "big_c");
    for i in 0..500usize {
        assert_eq!(col_tiny.value(i), i as i32);
        assert_eq!(col_a.value(i), i as i64 * 7);
        assert!((col_b.value(i) - i as f64 / 3.0).abs() < 1e-10);
        assert_eq!(col_c.value(i), format!("str_{:05}", i));
    }

    // Project only "tiny" and "big_c"
    let idx_tiny = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "tiny")
        .unwrap();
    let idx_c = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "big_c")
        .unwrap();
    let mut rg2 = reader
        .row_group_reader_projected(0, &[idx_tiny, idx_c])
        .unwrap();
    let batch2 = rg2.read_columns().unwrap();
    assert_eq!(batch2.num_rows(), 500);
    assert_eq!(batch2.num_columns(), 2);

    let col_tiny2 = batch_col_i32(&batch2, "tiny");
    let col_c2 = batch_col_string(&batch2, "big_c");
    for i in 0..500usize {
        assert_eq!(col_tiny2.value(i), i as i32);
        assert_eq!(col_c2.value(i), format!("str_{:05}", i));
    }
}

#[test]
fn test_paged_all_null_columns_in_projection() {
    let columns = vec![
        ("real".to_string(), DataType::Int32, true),
        ("null_col".to_string(), DataType::Int64, true),
        ("also_real".to_string(), DataType::Float64, true),
    ];
    let rows: Vec<Vec<Value>> = (0..100)
        .map(|i| vec![Value::Integer(i), Value::Null, Value::Double(i as f64)])
        .collect();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_ZSTD,
            page_size_threshold: 1,
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile::new(data), len).unwrap();

    // Project including the ALL_NULL column
    let idx_null = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "null_col")
        .unwrap();
    let idx_real = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "also_real")
        .unwrap();
    let mut rg = reader
        .row_group_reader_projected(0, &[idx_null, idx_real])
        .unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 100);

    let null_col = batch_col_i64(&batch, "null_col");
    let real_col = batch_col_f64(&batch, "also_real");
    for i in 0..100usize {
        assert!(null_col.is_null(i));
        assert!((real_col.value(i) - i as f64).abs() < 1e-10);
    }
}

#[test]
fn test_paged_adjacent_columns_coalesced_read() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingInputFile {
        data: Vec<u8>,
        read_count: AtomicUsize,
    }

    impl InputFile for CountingInputFile {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            self.read_count.fetch_add(1, Ordering::Relaxed);
            let start = offset as usize;
            let end = start + buf.len();
            if end > self.data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "read past end",
                ));
            }
            buf.copy_from_slice(&self.data[start..end]);
            Ok(())
        }
    }

    // 10 columns in 1 bucket, project col 2,3,4 (adjacent)
    let columns: Vec<(String, DataType, bool)> = (0..10)
        .map(|i| (format!("c{}", i), DataType::Int64, true))
        .collect();
    let rows: Vec<Vec<Value>> = (0..100)
        .map(|r| {
            (0..10)
                .map(|c| Value::BigInt((r * 10 + c) as i64))
                .collect()
        })
        .collect();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions {
            compression: COMPRESSION_ZSTD,
            page_size_threshold: 1,
            num_buckets: 1,
            ..Default::default()
        },
    )
    .unwrap();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let data = writer.output().buf.clone();
    let len = data.len() as u64;

    let input = CountingInputFile {
        data: data.clone(),
        read_count: AtomicUsize::new(0),
    };
    let reader = MosaicReader::new(input, len).unwrap();
    let open_reads = reader.input().read_count.load(Ordering::Relaxed);

    let idx2 = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "c2")
        .unwrap();
    let idx3 = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "c3")
        .unwrap();
    let idx4 = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "c4")
        .unwrap();

    let mut rg = reader
        .row_group_reader_projected(0, &[idx2, idx3, idx4])
        .unwrap();
    let batch = rg.read_columns().unwrap();
    assert_eq!(batch.num_rows(), 100);
    assert_eq!(batch.num_columns(), 3);

    let c2 = batch_col_i64(&batch, "c2");
    let c3 = batch_col_i64(&batch, "c3");
    let c4 = batch_col_i64(&batch, "c4");
    for r in 0..100usize {
        assert_eq!(c2.value(r), (r * 10 + 2) as i64);
        assert_eq!(c3.value(r), (r * 10 + 3) as i64);
        assert_eq!(c4.value(r), (r * 10 + 4) as i64);
    }

    // After open: 1 tail-prefetch IO covers footer + schema + index
    // projected read should do: round1 (directory) + round2 (3 adjacent slots coalesced)
    // = 2 read_ranges calls, each coalesced into 1 read_at
    let projection_reads = reader.input().read_count.load(Ordering::Relaxed) - open_reads;
    assert_eq!(
        projection_reads, 2,
        "expected 2 read_at calls (dir + coalesced slots)"
    );
}

#[test]
fn test_file_open_single_io() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingInputFile {
        data: Vec<u8>,
        read_count: AtomicUsize,
    }

    impl InputFile for CountingInputFile {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            self.read_count.fetch_add(1, Ordering::Relaxed);
            let start = offset as usize;
            let end = start + buf.len();
            if end > self.data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "read past end",
                ));
            }
            buf.copy_from_slice(&self.data[start..end]);
            Ok(())
        }
    }

    // Write a file with modest metadata (fits within 256KB tail)
    let columns: Vec<(String, DataType, bool)> = vec![
        ("a".into(), DataType::Int64, false),
        ("b".into(), DataType::Int64, false),
        ("c".into(), DataType::Utf8, true),
    ];
    let rows: Vec<Vec<Value>> = (0..10)
        .map(|r| {
            vec![
                Value::BigInt(r as i64),
                Value::BigInt(r as i64 * 2),
                Value::String(format!("row{}", r).into_bytes()),
            ]
        })
        .collect();

    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(
        out,
        &columns_to_arrow_schema(&columns),
        WriterOptions::default(),
    )
    .unwrap();
    write_values(&mut writer, &columns, &rows);
    writer.close().unwrap();
    let file_data = writer.output().buf.clone();
    let file_len = file_data.len() as u64;

    let input = CountingInputFile {
        data: file_data,
        read_count: AtomicUsize::new(0),
    };

    // Open the file — should require only 1 read_at (tail prefetch covers all metadata)
    let reader = MosaicReader::new(input, file_len).unwrap();
    assert_eq!(
        reader.input().read_count.load(Ordering::Relaxed),
        1,
        "file open should use only 1 IO when metadata fits in tail prefetch"
    );
}
