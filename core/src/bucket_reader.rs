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

use std::io;
use std::sync::Arc;

use arrow_array::*;
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, TimeUnit};

use crate::spec::*;
use crate::types;
use crate::values::Value;
use crate::varint;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DataVariant {
    Boolean,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Binary,
    TimestampNanos,
}

fn data_variant_for_type(dt: &DataType) -> DataVariant {
    match dt {
        DataType::Boolean => DataVariant::Boolean,
        DataType::Int8 => DataVariant::Int8,
        DataType::Int16 => DataVariant::Int16,
        DataType::Int32 | DataType::Date32 | DataType::Time32(_) => DataVariant::Int32,
        DataType::Float32 => DataVariant::Float32,
        DataType::Int64 => DataVariant::Int64,
        DataType::Float64 => DataVariant::Float64,
        DataType::Decimal128(p, _) => {
            if *p <= 18 {
                DataVariant::Int64
            } else {
                DataVariant::Binary
            }
        }
        DataType::Timestamp(_, _) => DataVariant::Int64,
        DataType::Struct(fields) if types::is_timestamp_nanos_struct(fields) => {
            DataVariant::TimestampNanos
        }
        _ => DataVariant::Binary,
    }
}

#[derive(Debug, Clone)]
enum RawColumnData {
    Boolean(Vec<u8>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Binary {
        offsets: Vec<u32>,
        data: Vec<u8>,
    },
    TimestampNanos {
        millis: Vec<i64>,
        nanos_of_milli: Vec<i32>,
    },
}

fn empty_raw_data_for_type(dt: &DataType) -> RawColumnData {
    match dt {
        DataType::Boolean => RawColumnData::Boolean(Vec::new()),
        DataType::Int8 => RawColumnData::Int8(Vec::new()),
        DataType::Int16 => RawColumnData::Int16(Vec::new()),
        DataType::Int32 | DataType::Date32 | DataType::Time32(_) => {
            RawColumnData::Int32(Vec::new())
        }
        DataType::Float32 => RawColumnData::Float32(Vec::new()),
        DataType::Int64 => RawColumnData::Int64(Vec::new()),
        DataType::Float64 => RawColumnData::Float64(Vec::new()),
        DataType::Decimal128(p, _) => {
            if *p <= 18 {
                RawColumnData::Int64(Vec::new())
            } else {
                RawColumnData::Binary {
                    offsets: vec![0],
                    data: Vec::new(),
                }
            }
        }
        DataType::Timestamp(_, _) => RawColumnData::Int64(Vec::new()),
        DataType::Struct(fields) if types::is_timestamp_nanos_struct(fields) => {
            RawColumnData::TimestampNanos {
                millis: Vec::new(),
                nanos_of_milli: Vec::new(),
            }
        }
        _ => RawColumnData::Binary {
            offsets: vec![0],
            data: Vec::new(),
        },
    }
}

fn invert_bitmap(bitmap: &[u8]) -> Vec<u8> {
    bitmap.iter().map(|b| !b).collect()
}

fn make_null_buffer(bitmap: Option<Vec<u8>>, num_rows: usize) -> Option<NullBuffer> {
    bitmap.map(|bm| NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(bm), 0, num_rows)))
}

fn scatter_fixed<T: Default + Copy>(
    values: Vec<T>,
    bitmap: &Option<Vec<u8>>,
    num_rows: usize,
) -> Vec<T> {
    let bm = match bitmap {
        None => return values,
        Some(bm) => bm,
    };
    let mut out = vec![T::default(); num_rows];
    let mut src = 0;
    for i in 0..num_rows {
        if (bm[i / 8] & (1 << (i % 8))) != 0 {
            out[i] = values[src];
            src += 1;
        }
    }
    out
}

fn scatter_binary_offsets(
    offsets: Vec<u32>,
    data: Vec<u8>,
    bitmap: &Option<Vec<u8>>,
    num_rows: usize,
) -> (Vec<i32>, Vec<u8>) {
    let bm = match bitmap {
        None => {
            let i32_offsets: Vec<i32> = offsets.into_iter().map(|o| o as i32).collect();
            return (i32_offsets, data);
        }
        Some(bm) => bm,
    };
    let mut out_offsets = Vec::with_capacity(num_rows + 1);
    let mut out_data = Vec::with_capacity(data.len());
    out_offsets.push(0i32);
    let mut src = 0usize;
    for i in 0..num_rows {
        if (bm[i / 8] & (1 << (i % 8))) != 0 {
            let start = offsets[src] as usize;
            let end = offsets[src + 1] as usize;
            assert!(
                start <= end && end <= data.len(),
                "binary offset out of bounds: start={}, end={}, data_len={}",
                start,
                end,
                data.len()
            );
            out_data.extend_from_slice(&data[start..end]);
            src += 1;
        }
        out_offsets.push(out_data.len() as i32);
    }
    (out_offsets, out_data)
}

fn build_all_null_array(dt: &DataType, num_rows: usize) -> ArrayRef {
    arrow_array::new_null_array(dt, num_rows)
}

fn build_array(
    data: RawColumnData,
    dt: &DataType,
    null_bitmap: Option<Vec<u8>>,
    num_rows: usize,
) -> ArrayRef {
    let null_buf = make_null_buffer(null_bitmap.clone(), num_rows);

    match data {
        RawColumnData::Boolean(values) => {
            let bool_buf = BooleanBuffer::new(Buffer::from_vec(values), 0, num_rows);
            Arc::new(BooleanArray::new(bool_buf, null_buf))
        }
        RawColumnData::Int8(values) => {
            let scattered = scatter_fixed(values, &null_bitmap, num_rows);
            Arc::new(Int8Array::new(ScalarBuffer::from(scattered), null_buf))
        }
        RawColumnData::Int16(values) => {
            let scattered = scatter_fixed(values, &null_bitmap, num_rows);
            Arc::new(Int16Array::new(ScalarBuffer::from(scattered), null_buf))
        }
        RawColumnData::Int32(values) => {
            let scattered = scatter_fixed(values, &null_bitmap, num_rows);
            match dt {
                DataType::Date32 => {
                    Arc::new(Date32Array::new(ScalarBuffer::from(scattered), null_buf))
                }
                DataType::Time32(_) => Arc::new(Time32MillisecondArray::new(
                    ScalarBuffer::from(scattered),
                    null_buf,
                )),
                _ => Arc::new(Int32Array::new(ScalarBuffer::from(scattered), null_buf)),
            }
        }
        RawColumnData::Int64(values) => {
            let scattered = scatter_fixed(values, &null_bitmap, num_rows);
            match dt {
                DataType::Decimal128(p, s) => {
                    let i128_values: Vec<i128> = scattered.iter().map(|&v| v as i128).collect();
                    Arc::new(
                        Decimal128Array::new(ScalarBuffer::from(i128_values), null_buf)
                            .with_precision_and_scale(*p, *s)
                            .unwrap(),
                    )
                }
                DataType::Timestamp(TimeUnit::Millisecond, tz) => {
                    let arr =
                        TimestampMillisecondArray::new(ScalarBuffer::from(scattered), null_buf);
                    Arc::new(if let Some(tz) = tz {
                        arr.with_timezone(tz.clone())
                    } else {
                        arr
                    })
                }
                DataType::Timestamp(TimeUnit::Microsecond, tz) => {
                    let arr =
                        TimestampMicrosecondArray::new(ScalarBuffer::from(scattered), null_buf);
                    Arc::new(if let Some(tz) = tz {
                        arr.with_timezone(tz.clone())
                    } else {
                        arr
                    })
                }
                _ => Arc::new(Int64Array::new(ScalarBuffer::from(scattered), null_buf)),
            }
        }
        RawColumnData::Float32(values) => {
            let scattered = scatter_fixed(values, &null_bitmap, num_rows);
            Arc::new(Float32Array::new(ScalarBuffer::from(scattered), null_buf))
        }
        RawColumnData::Float64(values) => {
            let scattered = scatter_fixed(values, &null_bitmap, num_rows);
            Arc::new(Float64Array::new(ScalarBuffer::from(scattered), null_buf))
        }
        RawColumnData::Binary { offsets, data } => {
            let (i32_offsets, out_data) =
                scatter_binary_offsets(offsets, data, &null_bitmap, num_rows);
            let offset_buf = OffsetBuffer::new(ScalarBuffer::from(i32_offsets));
            match dt {
                DataType::Utf8 => Arc::new(StringArray::new(
                    offset_buf,
                    Buffer::from_vec(out_data),
                    null_buf,
                )),
                DataType::Decimal128(p, s) => {
                    let bin = BinaryArray::new(offset_buf, Buffer::from_vec(out_data), null_buf);
                    let i128_values: Vec<i128> = (0..num_rows)
                        .map(|i| {
                            if bin.is_null(i) {
                                0i128
                            } else {
                                let bytes = bin.value(i);
                                let negative = !bytes.is_empty() && bytes[0] & 0x80 != 0;
                                let pad = if negative { 0xFF } else { 0x00 };
                                let mut buf = [pad; 16];
                                let start = 16usize.saturating_sub(bytes.len());
                                buf[start..].copy_from_slice(bytes);
                                i128::from_be_bytes(buf)
                            }
                        })
                        .collect();
                    let null_buf2 = bin.nulls().cloned();
                    Arc::new(
                        Decimal128Array::new(ScalarBuffer::from(i128_values), null_buf2)
                            .with_precision_and_scale(*p, *s)
                            .unwrap(),
                    )
                }
                _ => Arc::new(BinaryArray::new(
                    offset_buf,
                    Buffer::from_vec(out_data),
                    null_buf,
                )),
            }
        }
        RawColumnData::TimestampNanos {
            millis,
            nanos_of_milli,
        } => {
            let millis_scattered = scatter_fixed(millis, &null_bitmap, num_rows);
            let nanos_scattered = scatter_fixed(nanos_of_milli, &null_bitmap, num_rows);
            let millis_array = Arc::new(Int64Array::from(millis_scattered)) as ArrayRef;
            let nanos_array = Arc::new(Int32Array::from(nanos_scattered)) as ArrayRef;
            let fields = vec![
                Field::new("millis", DataType::Int64, false),
                Field::new("nanos_of_milli", DataType::Int32, false),
            ];
            Arc::new(StructArray::new(
                fields.into(),
                vec![millis_array, nanos_array],
                null_buf,
            ))
        }
    }
}

pub struct BucketReader {
    data: Vec<u8>,
    num_columns: usize,
    num_rows: usize,
    col_types: Vec<DataType>,

    encodings: Vec<u8>,
    has_nulls: Vec<bool>,
    null_bitmaps: Vec<Vec<u8>>,
    const_values: Vec<Value>,
    dict_values: Vec<Vec<Value>>,
    dict_bit_widths: Vec<usize>,
    data_cursors: Vec<usize>,
}

impl BucketReader {
    pub fn new(col_types: Vec<DataType>, data: Vec<u8>, num_rows: usize) -> io::Result<Self> {
        let num_columns = col_types.len();
        let mut reader = BucketReader {
            data,
            num_columns,
            num_rows,
            col_types,
            encodings: vec![0; num_columns],
            has_nulls: vec![false; num_columns],
            null_bitmaps: Vec::new(),
            const_values: Vec::new(),
            dict_values: Vec::new(),
            dict_bit_widths: vec![0; num_columns],
            data_cursors: vec![0; num_columns],
        };
        reader.init()?;
        Ok(reader)
    }

    fn check_bounds(&self, pos: usize, need: usize) -> io::Result<()> {
        if pos
            .checked_add(need)
            .is_none_or(|end| end > self.data.len())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bucket data truncated",
            ));
        }
        Ok(())
    }

    fn init(&mut self) -> io::Result<()> {
        self.null_bitmaps = vec![Vec::new(); self.num_columns];
        self.const_values = vec![Value::Null; self.num_columns];
        self.dict_values = vec![Vec::new(); self.num_columns];

        let mut pos = 0;

        // 1. Encoding flags (2 bits per column)
        let encoding_flags_bytes = (self.num_columns * 2).div_ceil(8);
        self.check_bounds(pos, encoding_flags_bytes)?;
        for i in 0..self.num_columns {
            let byte_idx = (i * 2) / 8;
            let bit_idx = (i * 2) % 8;
            self.encodings[i] = (self.data[pos + byte_idx] >> bit_idx) & 0x03;
        }
        pos += encoding_flags_bytes;

        // 2. Has-nulls flags (1 bit per column)
        let has_nulls_bytes = self.num_columns.div_ceil(8);
        self.check_bounds(pos, has_nulls_bytes)?;
        for i in 0..self.num_columns {
            self.has_nulls[i] = (self.data[pos + i / 8] & (1 << (i % 8))) != 0;
        }
        pos += has_nulls_bytes;

        // 3. CONST metadata
        for i in 0..self.num_columns {
            if self.encodings[i] == ENCODING_CONST {
                let (value, size) = self.read_value_at(&self.col_types[i], pos)?;
                self.const_values[i] = value;
                pos += size;
            }
        }

        // 4. DICT metadata
        for i in 0..self.num_columns {
            if self.encodings[i] == ENCODING_DICT {
                let num_entries = varint::decode(&self.data, &mut pos)? as usize;
                self.dict_bit_widths[i] = bit_width(num_entries);
                let mut entries = Vec::with_capacity(num_entries);
                for _ in 0..num_entries {
                    let (value, size) = self.read_value_at(&self.col_types[i], pos)?;
                    entries.push(value);
                    pos += size;
                }
                self.dict_values[i] = entries;
            }
        }

        // 5. Null bitmaps
        let null_bitmap_bytes = self.num_rows.div_ceil(8);
        for i in 0..self.num_columns {
            if self.has_nulls[i] && self.encodings[i] != ENCODING_ALL_NULL {
                self.check_bounds(pos, null_bitmap_bytes)?;
                self.null_bitmaps[i] = self.data[pos..pos + null_bitmap_bytes].to_vec();
                pos += null_bitmap_bytes;
            }
        }

        // 6. Record column data start offsets, skip past data
        for i in 0..self.num_columns {
            self.data_cursors[i] = pos;
            if self.encodings[i] == ENCODING_PLAIN {
                let w = types::fixed_width(&self.col_types[i]);
                let non_null_count = self.count_non_null(i);
                if w > 0 {
                    let size = non_null_count * w as usize;
                    self.check_bounds(pos, size)?;
                    pos += size;
                } else {
                    for _ in 0..non_null_count {
                        let len = varint::decode(&self.data, &mut pos)? as usize;
                        self.check_bounds(pos, len)?;
                        pos += len;
                    }
                }
            } else if self.encodings[i] == ENCODING_DICT {
                let non_null_count = self.count_non_null(i);
                let size = (non_null_count * self.dict_bit_widths[i]).div_ceil(8);
                self.check_bounds(pos, size)?;
                pos += size;
            }
        }
        Ok(())
    }

    fn read_value_at(&self, dt: &DataType, pos: usize) -> io::Result<(Value, usize)> {
        let w = types::fixed_width(dt);
        if w > 0 {
            self.check_bounds(pos, w as usize)?;
            Ok((read_typed_value(dt, &self.data, pos, w), w as usize))
        } else {
            read_variable_value(dt, &self.data, pos)
        }
    }

    fn count_non_null(&self, col: usize) -> usize {
        if !self.has_nulls[col] {
            return self.num_rows;
        }
        if self.encodings[col] == ENCODING_ALL_NULL {
            return 0;
        }
        let bitmap = &self.null_bitmaps[col];
        let full_bytes = self.num_rows / 8;
        let mut null_count = 0usize;
        for byte in bitmap.iter().take(full_bytes) {
            null_count += (*byte as u32).count_ones() as usize;
        }
        let remaining = self.num_rows % 8;
        if remaining > 0 {
            let mask = (1u8 << remaining) - 1;
            null_count += (bitmap[full_bytes] & mask).count_ones() as usize;
        }
        self.num_rows - null_count
    }

    pub fn read_all_columns(&self) -> io::Result<Vec<ArrayRef>> {
        let num_rows = self.num_rows;
        let mut result = Vec::with_capacity(self.num_columns);

        for i in 0..self.num_columns {
            let variant = data_variant_for_type(&self.col_types[i]);

            if self.encodings[i] == ENCODING_ALL_NULL {
                result.push(build_all_null_array(&self.col_types[i], num_rows));
                continue;
            }

            let has_nulls = self.has_nulls[i];
            let null_bitmap = if has_nulls {
                Some(invert_bitmap(&self.null_bitmaps[i]))
            } else {
                None
            };

            let data = match self.encodings[i] {
                ENCODING_CONST => read_all_const(
                    &self.const_values[i],
                    num_rows,
                    has_nulls,
                    &self.null_bitmaps[i],
                    variant,
                )?,
                ENCODING_DICT => read_all_dict(
                    &self.data,
                    self.data_cursors[i],
                    &self.dict_values[i],
                    self.dict_bit_widths[i],
                    num_rows,
                    has_nulls,
                    &self.null_bitmaps[i],
                    variant,
                )?,
                ENCODING_PLAIN => read_all_plain(
                    &self.data,
                    self.data_cursors[i],
                    &self.col_types[i],
                    num_rows,
                    has_nulls,
                    &self.null_bitmaps[i],
                    variant,
                )?,
                _ => empty_raw_data_for_type(&self.col_types[i]),
            };

            result.push(build_array(data, &self.col_types[i], null_bitmap, num_rows));
        }

        Ok(result)
    }
}

pub struct ColumnPageReader {
    col_type: DataType,
    encoding: u8,
    has_nulls: bool,
    const_value: Value,
    dict_values: Vec<Value>,
    dict_bit_width: usize,
    null_bitmap: Vec<u8>,
    data: Vec<u8>,
    data_cursor: usize,
    num_rows: usize,
}

impl ColumnPageReader {
    pub fn new(
        col_type: DataType,
        encoding: u8,
        has_nulls: bool,
        const_value: Value,
        page_data: Vec<u8>,
        num_rows: usize,
    ) -> io::Result<Self> {
        let mut reader = ColumnPageReader {
            col_type,
            encoding,
            has_nulls,
            const_value,
            dict_values: Vec::new(),
            dict_bit_width: 0,
            null_bitmap: Vec::new(),
            data: page_data,
            data_cursor: 0,
            num_rows,
        };
        reader.init_page()?;
        Ok(reader)
    }

    fn init_page(&mut self) -> io::Result<()> {
        let null_bitmap_bytes = self.num_rows.div_ceil(8);
        let mut pos = 0;

        match self.encoding {
            ENCODING_ALL_NULL => {}
            ENCODING_CONST if self.has_nulls => {
                if pos + null_bitmap_bytes > self.data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "column page truncated: null bitmap",
                    ));
                }
                self.null_bitmap = self.data[pos..pos + null_bitmap_bytes].to_vec();
            }
            ENCODING_CONST => {}
            ENCODING_DICT => {
                let num_entries = varint::decode(&self.data, &mut pos)? as usize;
                self.dict_bit_width = bit_width(num_entries);
                let mut entries = Vec::with_capacity(num_entries);
                for _ in 0..num_entries {
                    let (value, size) = self.read_value_at(pos)?;
                    entries.push(value);
                    pos += size;
                }
                self.dict_values = entries;
                if self.has_nulls {
                    if pos + null_bitmap_bytes > self.data.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "column page truncated: null bitmap",
                        ));
                    }
                    self.null_bitmap = self.data[pos..pos + null_bitmap_bytes].to_vec();
                    pos += null_bitmap_bytes;
                }
                self.data_cursor = pos;
                let non_null_count = self.count_non_null();
                let packed_bytes = (non_null_count * self.dict_bit_width).div_ceil(8);
                if pos
                    .checked_add(packed_bytes)
                    .is_none_or(|end| end > self.data.len())
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "column page truncated: dict bit-packed data",
                    ));
                }
            }
            ENCODING_PLAIN => {
                if self.has_nulls {
                    if pos + null_bitmap_bytes > self.data.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "column page truncated: null bitmap",
                        ));
                    }
                    self.null_bitmap = self.data[pos..pos + null_bitmap_bytes].to_vec();
                    pos += null_bitmap_bytes;
                }
                self.data_cursor = pos;
                let non_null_count = self.count_non_null();
                let w = types::fixed_width(&self.col_type);
                if w > 0 {
                    let size = non_null_count * w as usize;
                    if pos
                        .checked_add(size)
                        .is_none_or(|end| end > self.data.len())
                    {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "column page truncated: plain fixed-width data",
                        ));
                    }
                } else {
                    let mut scan = pos;
                    for _ in 0..non_null_count {
                        let len = varint::decode(&self.data, &mut scan)? as usize;
                        if scan
                            .checked_add(len)
                            .is_none_or(|end| end > self.data.len())
                        {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "column page truncated: plain variable-width data",
                            ));
                        }
                        scan += len;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn count_non_null(&self) -> usize {
        if !self.has_nulls {
            return self.num_rows;
        }
        if self.encoding == ENCODING_ALL_NULL {
            return 0;
        }
        let bitmap = &self.null_bitmap;
        let full_bytes = self.num_rows / 8;
        let mut null_count = 0usize;
        for byte in bitmap.iter().take(full_bytes) {
            null_count += (*byte as u32).count_ones() as usize;
        }
        let remaining = self.num_rows % 8;
        if remaining > 0 {
            let mask = (1u8 << remaining) - 1;
            null_count += (bitmap[full_bytes] & mask).count_ones() as usize;
        }
        self.num_rows - null_count
    }

    fn read_value_at(&self, pos: usize) -> io::Result<(Value, usize)> {
        let w = types::fixed_width(&self.col_type);
        if w > 0 {
            if pos + w as usize > self.data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "column page truncated",
                ));
            }
            Ok((
                read_typed_value(&self.col_type, &self.data, pos, w),
                w as usize,
            ))
        } else {
            read_variable_value(&self.col_type, &self.data, pos)
        }
    }

    pub fn read_all(&self) -> io::Result<ArrayRef> {
        let num_rows = self.num_rows;
        let variant = data_variant_for_type(&self.col_type);

        if self.encoding == ENCODING_ALL_NULL {
            return Ok(build_all_null_array(&self.col_type, num_rows));
        }

        let has_nulls = self.has_nulls;
        let null_bitmap = if has_nulls {
            Some(invert_bitmap(&self.null_bitmap))
        } else {
            None
        };

        let data = match self.encoding {
            ENCODING_CONST => read_all_const(
                &self.const_value,
                num_rows,
                has_nulls,
                &self.null_bitmap,
                variant,
            )?,
            ENCODING_DICT => read_all_dict(
                &self.data,
                self.data_cursor,
                &self.dict_values,
                self.dict_bit_width,
                num_rows,
                has_nulls,
                &self.null_bitmap,
                variant,
            )?,
            ENCODING_PLAIN => read_all_plain(
                &self.data,
                self.data_cursor,
                &self.col_type,
                num_rows,
                has_nulls,
                &self.null_bitmap,
                variant,
            )?,
            _ => empty_raw_data_for_type(&self.col_type),
        };

        Ok(build_array(data, &self.col_type, null_bitmap, num_rows))
    }
}

pub(crate) fn read_typed_value(dt: &DataType, buf: &[u8], pos: usize, width: i32) -> Value {
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
        DataType::Int64 => Value::BigInt(read_i64(buf, pos)),
        DataType::Float64 => Value::Double(f64::from_bits(read_u64(buf, pos))),
        DataType::Decimal128(_, _) => Value::DecimalCompact(read_i64(buf, pos)),
        DataType::Timestamp(TimeUnit::Millisecond, _) => Value::TimestampMillis(read_i64(buf, pos)),
        DataType::Timestamp(TimeUnit::Microsecond, _) => Value::TimestampMicros(read_i64(buf, pos)),
        DataType::Struct(fields) if types::is_timestamp_nanos_struct(fields) => {
            debug_assert_eq!(width, 12);
            let millis = read_i64(buf, pos);
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

pub(crate) fn read_variable_value(
    dt: &DataType,
    buf: &[u8],
    pos: usize,
) -> io::Result<(Value, usize)> {
    let mut p = pos;
    let len = varint::decode(buf, &mut p).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated varint in variable-length value",
        )
    })? as usize;
    let header_size = p - pos;
    if p + len > buf.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "buffer truncated in variable-length value",
        ));
    }
    let bytes = buf[p..p + len].to_vec();
    let total_size = header_size + len;

    let value = match dt {
        DataType::Utf8 => Value::String(bytes),
        DataType::Binary => Value::Bytes(bytes),
        DataType::Decimal128(_, _) => Value::DecimalLarge(bytes),
        _ => Value::Null,
    };
    Ok((value, total_size))
}

// ======================== Columnar batch decode helpers ========================

fn read_all_const(
    const_value: &Value,
    num_rows: usize,
    has_nulls: bool,
    null_bitmap: &[u8],
    variant: DataVariant,
) -> io::Result<RawColumnData> {
    let non_null_count = if has_nulls {
        count_non_null(null_bitmap, num_rows)
    } else {
        num_rows
    };

    match variant {
        DataVariant::Boolean => {
            let b = match const_value {
                Value::Boolean(v) => *v,
                _ => false,
            };
            let mut buf = vec![0u8; num_rows.div_ceil(8)];
            if b {
                for row in 0..num_rows {
                    if !has_nulls || !is_null(null_bitmap, row) {
                        buf[row / 8] |= 1 << (row % 8);
                    }
                }
            }
            Ok(RawColumnData::Boolean(buf))
        }
        DataVariant::Int8 => {
            let v = match const_value {
                Value::TinyInt(x) => *x,
                _ => 0,
            };
            let mut out = vec![0i8; non_null_count];
            out.fill(v);
            Ok(RawColumnData::Int8(out))
        }
        DataVariant::Int16 => {
            let v = match const_value {
                Value::SmallInt(x) => *x,
                _ => 0,
            };
            let mut out = vec![0i16; non_null_count];
            out.fill(v);
            Ok(RawColumnData::Int16(out))
        }
        DataVariant::Int32 => {
            let v = match const_value {
                Value::Integer(x) | Value::Date(x) | Value::Time(x) => *x,
                _ => 0,
            };
            let mut out = vec![0i32; non_null_count];
            out.fill(v);
            Ok(RawColumnData::Int32(out))
        }
        DataVariant::Int64 => {
            let v = match const_value {
                Value::BigInt(x)
                | Value::DecimalCompact(x)
                | Value::TimestampMillis(x)
                | Value::TimestampMicros(x) => *x,
                _ => 0,
            };
            let mut out = vec![0i64; non_null_count];
            out.fill(v);
            Ok(RawColumnData::Int64(out))
        }
        DataVariant::Float32 => {
            let v = match const_value {
                Value::Float(x) => *x,
                _ => 0.0,
            };
            let mut out = vec![0.0f32; non_null_count];
            out.fill(v);
            Ok(RawColumnData::Float32(out))
        }
        DataVariant::Float64 => {
            let v = match const_value {
                Value::Double(x) => *x,
                _ => 0.0,
            };
            let mut out = vec![0.0f64; non_null_count];
            out.fill(v);
            Ok(RawColumnData::Float64(out))
        }
        DataVariant::Binary => {
            let bytes = match const_value {
                Value::String(b) | Value::Bytes(b) | Value::DecimalLarge(b) => b.as_slice(),
                _ => &[],
            };
            let mut offsets = Vec::with_capacity(non_null_count + 1);
            let mut data = Vec::with_capacity(non_null_count * bytes.len());
            offsets.push(0u32);
            for _ in 0..non_null_count {
                data.extend_from_slice(bytes);
                offsets.push(data.len() as u32);
            }
            Ok(RawColumnData::Binary { offsets, data })
        }
        DataVariant::TimestampNanos => {
            let (m, n) = match const_value {
                Value::TimestampNanos {
                    millis,
                    nanos_of_milli,
                } => (*millis, *nanos_of_milli),
                _ => (0, 0),
            };
            let mut millis_out = vec![0i64; non_null_count];
            let mut nanos_out = vec![0i32; non_null_count];
            millis_out.fill(m);
            nanos_out.fill(n);
            Ok(RawColumnData::TimestampNanos {
                millis: millis_out,
                nanos_of_milli: nanos_out,
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn read_all_dict(
    data: &[u8],
    data_cursor: usize,
    dict_values: &[Value],
    dict_bit_width: usize,
    num_rows: usize,
    has_nulls: bool,
    null_bitmap: &[u8],
    variant: DataVariant,
) -> io::Result<RawColumnData> {
    let non_null_count = if has_nulls {
        count_non_null(null_bitmap, num_rows)
    } else {
        num_rows
    };

    let mut indices = Vec::with_capacity(non_null_count);
    let mut bit_offset = 0;
    for row in 0..num_rows {
        if has_nulls && is_null(null_bitmap, row) {
            continue;
        }
        let idx = read_bit_packed(data, data_cursor, bit_offset, dict_bit_width);
        bit_offset += dict_bit_width;
        if idx >= dict_values.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "corrupt dict index",
            ));
        }
        indices.push(idx);
    }

    match variant {
        DataVariant::Boolean => {
            let mut buf = vec![0u8; num_rows.div_ceil(8)];
            let mut vi = 0;
            for row in 0..num_rows {
                if has_nulls && is_null(null_bitmap, row) {
                    continue;
                }
                if let Value::Boolean(true) = &dict_values[indices[vi]] {
                    buf[row / 8] |= 1 << (row % 8);
                }
                vi += 1;
            }
            Ok(RawColumnData::Boolean(buf))
        }
        DataVariant::Int8 => {
            let out: Vec<i8> = indices
                .iter()
                .map(|&i| match &dict_values[i] {
                    Value::TinyInt(x) => *x,
                    _ => 0,
                })
                .collect();
            Ok(RawColumnData::Int8(out))
        }
        DataVariant::Int16 => {
            let out: Vec<i16> = indices
                .iter()
                .map(|&i| match &dict_values[i] {
                    Value::SmallInt(x) => *x,
                    _ => 0,
                })
                .collect();
            Ok(RawColumnData::Int16(out))
        }
        DataVariant::Int32 => {
            let out: Vec<i32> = indices
                .iter()
                .map(|&i| match &dict_values[i] {
                    Value::Integer(x) | Value::Date(x) | Value::Time(x) => *x,
                    _ => 0,
                })
                .collect();
            Ok(RawColumnData::Int32(out))
        }
        DataVariant::Int64 => {
            let out: Vec<i64> = indices
                .iter()
                .map(|&i| match &dict_values[i] {
                    Value::BigInt(x)
                    | Value::DecimalCompact(x)
                    | Value::TimestampMillis(x)
                    | Value::TimestampMicros(x) => *x,
                    _ => 0,
                })
                .collect();
            Ok(RawColumnData::Int64(out))
        }
        DataVariant::Float32 => {
            let out: Vec<f32> = indices
                .iter()
                .map(|&i| match &dict_values[i] {
                    Value::Float(x) => *x,
                    _ => 0.0,
                })
                .collect();
            Ok(RawColumnData::Float32(out))
        }
        DataVariant::Float64 => {
            let out: Vec<f64> = indices
                .iter()
                .map(|&i| match &dict_values[i] {
                    Value::Double(x) => *x,
                    _ => 0.0,
                })
                .collect();
            Ok(RawColumnData::Float64(out))
        }
        DataVariant::Binary => {
            let mut offsets = Vec::with_capacity(non_null_count + 1);
            let mut out_data = Vec::new();
            offsets.push(0u32);
            for &idx in &indices {
                let bytes = match &dict_values[idx] {
                    Value::String(b) | Value::Bytes(b) | Value::DecimalLarge(b) => b.as_slice(),
                    _ => &[],
                };
                out_data.extend_from_slice(bytes);
                offsets.push(out_data.len() as u32);
            }
            Ok(RawColumnData::Binary {
                offsets,
                data: out_data,
            })
        }
        DataVariant::TimestampNanos => {
            let mut millis_out = Vec::with_capacity(non_null_count);
            let mut nanos_out = Vec::with_capacity(non_null_count);
            for &idx in &indices {
                match &dict_values[idx] {
                    Value::TimestampNanos {
                        millis,
                        nanos_of_milli,
                    } => {
                        millis_out.push(*millis);
                        nanos_out.push(*nanos_of_milli);
                    }
                    _ => {
                        millis_out.push(0);
                        nanos_out.push(0);
                    }
                }
            }
            Ok(RawColumnData::TimestampNanos {
                millis: millis_out,
                nanos_of_milli: nanos_out,
            })
        }
    }
}

fn read_all_plain(
    data: &[u8],
    data_cursor: usize,
    col_type: &DataType,
    num_rows: usize,
    has_nulls: bool,
    null_bitmap: &[u8],
    variant: DataVariant,
) -> io::Result<RawColumnData> {
    let non_null_count = if has_nulls {
        count_non_null(null_bitmap, num_rows)
    } else {
        num_rows
    };
    let w = types::fixed_width(col_type);

    match variant {
        DataVariant::Boolean => {
            let mut buf = vec![0u8; num_rows.div_ceil(8)];
            let mut cursor = data_cursor;
            for row in 0..num_rows {
                if has_nulls && is_null(null_bitmap, row) {
                    continue;
                }
                if data[cursor] != 0 {
                    buf[row / 8] |= 1 << (row % 8);
                }
                cursor += 1;
            }
            Ok(RawColumnData::Boolean(buf))
        }
        DataVariant::Int8 => {
            let out: Vec<i8> = data[data_cursor..data_cursor + non_null_count]
                .iter()
                .map(|&b| b as i8)
                .collect();
            Ok(RawColumnData::Int8(out))
        }
        DataVariant::Int16 => {
            let mut out = Vec::with_capacity(non_null_count);
            let mut cursor = data_cursor;
            for _ in 0..non_null_count {
                out.push(i16::from_be_bytes([data[cursor], data[cursor + 1]]));
                cursor += 2;
            }
            Ok(RawColumnData::Int16(out))
        }
        DataVariant::Int32 => {
            let mut out = Vec::with_capacity(non_null_count);
            let mut cursor = data_cursor;
            for _ in 0..non_null_count {
                out.push(i32::from_be_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                ]));
                cursor += 4;
            }
            Ok(RawColumnData::Int32(out))
        }
        DataVariant::Int64 => {
            let mut out = Vec::with_capacity(non_null_count);
            let mut cursor = data_cursor;
            for _ in 0..non_null_count {
                out.push(read_i64(data, cursor));
                cursor += 8;
            }
            Ok(RawColumnData::Int64(out))
        }
        DataVariant::Float32 => {
            let mut out = Vec::with_capacity(non_null_count);
            let mut cursor = data_cursor;
            for _ in 0..non_null_count {
                let bits = u32::from_be_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                ]);
                out.push(f32::from_bits(bits));
                cursor += 4;
            }
            Ok(RawColumnData::Float32(out))
        }
        DataVariant::Float64 => {
            let mut out = Vec::with_capacity(non_null_count);
            let mut cursor = data_cursor;
            for _ in 0..non_null_count {
                let bits = read_u64(data, cursor);
                out.push(f64::from_bits(bits));
                cursor += 8;
            }
            Ok(RawColumnData::Float64(out))
        }
        DataVariant::Binary => {
            let mut offsets = Vec::with_capacity(non_null_count + 1);
            let mut out_data = Vec::new();
            offsets.push(0u32);
            let mut cursor = data_cursor;
            for _ in 0..non_null_count {
                let len = varint::decode(data, &mut cursor)? as usize;
                out_data.extend_from_slice(&data[cursor..cursor + len]);
                cursor += len;
                offsets.push(out_data.len() as u32);
            }
            Ok(RawColumnData::Binary {
                offsets,
                data: out_data,
            })
        }
        DataVariant::TimestampNanos => {
            debug_assert_eq!(w, 12);
            let mut millis_out = Vec::with_capacity(non_null_count);
            let mut nanos_out = Vec::with_capacity(non_null_count);
            let mut cursor = data_cursor;
            for _ in 0..non_null_count {
                millis_out.push(read_i64(data, cursor));
                nanos_out.push(i32::from_be_bytes([
                    data[cursor + 8],
                    data[cursor + 9],
                    data[cursor + 10],
                    data[cursor + 11],
                ]));
                cursor += 12;
            }
            Ok(RawColumnData::TimestampNanos {
                millis: millis_out,
                nanos_of_milli: nanos_out,
            })
        }
    }
}

fn count_non_null(null_bitmap: &[u8], num_rows: usize) -> usize {
    let full_bytes = num_rows / 8;
    let mut null_count = 0usize;
    for byte in null_bitmap.iter().take(full_bytes) {
        null_count += (*byte as u32).count_ones() as usize;
    }
    let remaining = num_rows % 8;
    if remaining > 0 {
        let mask = (1u8 << remaining) - 1;
        null_count += (null_bitmap[full_bytes] & mask).count_ones() as usize;
    }
    num_rows - null_count
}

fn is_null(null_bitmap: &[u8], row: usize) -> bool {
    (null_bitmap[row / 8] & (1 << (row % 8))) != 0
}

fn read_bit_packed(buf: &[u8], byte_base: usize, bit_offset: usize, bit_width: usize) -> usize {
    let mut value = 0;
    for b in 0..bit_width {
        let global_bit = bit_offset + b;
        if (buf[byte_base + global_bit / 8] & (1 << (global_bit % 8))) != 0 {
            value |= 1 << b;
        }
    }
    value
}

fn bit_width(num_entries: usize) -> usize {
    if num_entries <= 1 {
        return 0;
    }
    usize::BITS as usize - (num_entries - 1).leading_zeros() as usize
}

fn read_i64(buf: &[u8], pos: usize) -> i64 {
    i64::from_be_bytes([
        buf[pos],
        buf[pos + 1],
        buf[pos + 2],
        buf[pos + 3],
        buf[pos + 4],
        buf[pos + 5],
        buf[pos + 6],
        buf[pos + 7],
    ])
}

fn read_u64(buf: &[u8], pos: usize) -> u64 {
    u64::from_be_bytes([
        buf[pos],
        buf[pos + 1],
        buf[pos + 2],
        buf[pos + 3],
        buf[pos + 4],
        buf[pos + 5],
        buf[pos + 6],
        buf[pos + 7],
    ])
}
