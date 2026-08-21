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
use std::mem::size_of;
use std::sync::Arc;

use arrow_array::*;
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, TimeUnit};

use crate::reader::Encoding;
use crate::spec::*;
use crate::types;
use crate::values::Value;
use crate::varint;

/// Borrowed view of one encoded scalar column.
///
/// The view borrows buffers owned by a [`crate::reader::RowGroupReader`] and must be consumed
/// synchronously during [`crate::reader::RowGroupReader::visit_encoded_columns`]. It exposes the
/// physical encoding without first materializing an Arrow array.
#[derive(Clone, Copy)]
pub struct EncodedColumn<'a> {
    data_type: &'a DataType,
    encoding: u8,
    has_nulls: bool,
    null_bitmap: &'a [u8],
    const_value: &'a Value,
    dict_values: &'a [Value],
    dict_bit_width: usize,
    data: &'a [u8],
    data_cursor: usize,
    num_rows: usize,
}

impl<'a> EncodedColumn<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        data_type: &'a DataType,
        encoding: u8,
        has_nulls: bool,
        null_bitmap: &'a [u8],
        const_value: &'a Value,
        dict_values: &'a [Value],
        dict_bit_width: usize,
        data: &'a [u8],
        data_cursor: usize,
        num_rows: usize,
    ) -> Self {
        Self {
            data_type,
            encoding,
            has_nulls,
            null_bitmap,
            const_value,
            dict_values,
            dict_bit_width,
            data,
            data_cursor,
            num_rows,
        }
    }

    pub fn data_type(&self) -> &DataType {
        self.data_type
    }

    /// Returns the physical encoding used by this column.
    pub fn encoding(&self) -> Encoding {
        Encoding::from_code(self.encoding)
    }

    /// Returns the logical row count.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Returns whether at least one row is null.
    pub fn has_nulls(&self) -> bool {
        self.has_nulls || self.encoding == ENCODING_ALL_NULL
    }

    /// Returns the physical null bitmap when one is present.
    ///
    /// A set bit means that the corresponding row is null.
    pub fn null_bitmap(&self) -> Option<&'a [u8]> {
        self.has_nulls.then_some(self.null_bitmap)
    }

    /// Returns whether `row` is null.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.num_rows()`.
    pub fn is_null(&self, row: usize) -> bool {
        assert!(row < self.num_rows, "encoded column row out of bounds");
        self.encoding == ENCODING_ALL_NULL || (self.has_nulls && is_null(self.null_bitmap, row))
    }

    /// Returns the single encoded value for a CONST column.
    pub fn constant(&self) -> io::Result<Option<EncodedValueRef<'a>>> {
        if self.encoding != ENCODING_CONST {
            return Ok(None);
        }
        encoded_value_ref(self.data_type, self.const_value).map(Some)
    }

    /// Iterates values in logical row order without materializing an Arrow array.
    ///
    /// Null rows are returned as [`EncodedValueRef::Null`]. Dictionary indexes and PLAIN values
    /// are decoded lazily as the iterator advances.
    pub fn values(&self) -> EncodedColumnValues<'a> {
        EncodedColumnValues {
            column: *self,
            row: 0,
            data_cursor: self.data_cursor,
            bit_offset: 0,
            done: false,
        }
    }
}

/// Borrowed scalar value returned by [`EncodedColumnValues`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum EncodedValueRef<'a> {
    Null,
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Utf8(&'a [u8]),
    Binary(&'a [u8]),
    DecimalCompact(i64),
    DecimalLarge(&'a [u8]),
    Date32(i32),
    Time32(i32),
    TimestampMillis(i64),
    TimestampMicros(i64),
    TimestampNanos { millis: i64, nanos_of_milli: i32 },
}

/// Iterator over an [`EncodedColumn`] in logical row order.
pub struct EncodedColumnValues<'a> {
    column: EncodedColumn<'a>,
    row: usize,
    data_cursor: usize,
    bit_offset: usize,
    done: bool,
}

impl<'a> Iterator for EncodedColumnValues<'a> {
    type Item = io::Result<EncodedValueRef<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.row >= self.column.num_rows {
            return None;
        }

        let row = self.row;
        self.row += 1;
        if self.column.is_null(row) {
            return Some(Ok(EncodedValueRef::Null));
        }

        let value = match self.column.encoding {
            ENCODING_CONST => encoded_value_ref(self.column.data_type, self.column.const_value),
            ENCODING_DICT => read_bit_packed_checked(
                self.column.data,
                self.column.data_cursor,
                self.bit_offset,
                self.column.dict_bit_width,
            )
            .and_then(|index| {
                self.bit_offset += self.column.dict_bit_width;
                self.column
                    .dict_values
                    .get(index)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "corrupt dict index"))
                    .and_then(|value| encoded_value_ref(self.column.data_type, value))
            }),
            ENCODING_PLAIN => {
                let result =
                    read_encoded_value(self.column.data_type, self.column.data, self.data_cursor);
                if let Ok((_, size)) = result {
                    self.data_cursor += size;
                }
                result.map(|(value, _)| value)
            }
            ENCODING_ALL_NULL => Ok(EncodedValueRef::Null),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported encoding {}", self.column.encoding),
            )),
        };
        if value.is_err() {
            self.done = true;
        }
        Some(value)
    }
}

fn encoded_value_ref<'a>(
    data_type: &DataType,
    value: &'a Value,
) -> io::Result<EncodedValueRef<'a>> {
    let value = match value {
        Value::Null => EncodedValueRef::Null,
        Value::Boolean(value) => EncodedValueRef::Boolean(*value),
        Value::TinyInt(value) => EncodedValueRef::Int8(*value),
        Value::SmallInt(value) => EncodedValueRef::Int16(*value),
        Value::Integer(value) => EncodedValueRef::Int32(*value),
        Value::BigInt(value) => EncodedValueRef::Int64(*value),
        Value::Float(value) => EncodedValueRef::Float32(*value),
        Value::Double(value) => EncodedValueRef::Float64(*value),
        Value::Date(value) => EncodedValueRef::Date32(*value),
        Value::Time(value) => EncodedValueRef::Time32(*value),
        Value::String(value) => EncodedValueRef::Utf8(value),
        Value::Bytes(value) => EncodedValueRef::Binary(value),
        Value::DecimalCompact(value) => EncodedValueRef::DecimalCompact(*value),
        Value::DecimalLarge(value) => EncodedValueRef::DecimalLarge(value),
        Value::TimestampMillis(value) => EncodedValueRef::TimestampMillis(*value),
        Value::TimestampMicros(value) => EncodedValueRef::TimestampMicros(*value),
        Value::TimestampNanos {
            millis,
            nanos_of_milli,
        } => EncodedValueRef::TimestampNanos {
            millis: *millis,
            nanos_of_milli: *nanos_of_milli,
        },
    };
    validate_encoded_value(data_type, value)
}

fn validate_encoded_value<'a>(
    data_type: &DataType,
    value: EncodedValueRef<'a>,
) -> io::Result<EncodedValueRef<'a>> {
    if let EncodedValueRef::TimestampNanos {
        millis,
        nanos_of_milli,
    } = value
    {
        if types::is_timestamp_nanos(data_type) {
            types::millis_nanos_to_ns(millis, nanos_of_milli)?;
        }
    }
    Ok(value)
}

fn read_encoded_value<'a>(
    data_type: &DataType,
    data: &'a [u8],
    position: usize,
) -> io::Result<(EncodedValueRef<'a>, usize)> {
    let width = types::fixed_width(data_type);
    if width <= 0 {
        let mut payload = position;
        let length = varint::decode(data, &mut payload).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated varint in variable-length value",
            )
        })? as usize;
        let end = payload
            .checked_add(length)
            .filter(|end| *end <= data.len())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "buffer truncated in variable-length value",
                )
            })?;
        let value = match data_type {
            DataType::Utf8 => EncodedValueRef::Utf8(&data[payload..end]),
            DataType::Binary => EncodedValueRef::Binary(&data[payload..end]),
            DataType::Decimal128(_, _) => EncodedValueRef::DecimalLarge(&data[payload..end]),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported encoded data type: {data_type:?}"),
                ));
            }
        };
        return Ok((value, end - position));
    }

    let width = width as usize;
    let end = position
        .checked_add(width)
        .filter(|end| *end <= data.len())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "column data truncated"))?;
    let bytes = &data[position..end];
    let value = match data_type {
        DataType::Boolean => EncodedValueRef::Boolean(bytes[0] != 0),
        DataType::Int8 => EncodedValueRef::Int8(bytes[0] as i8),
        DataType::Int16 => EncodedValueRef::Int16(i16::from_be_bytes([bytes[0], bytes[1]])),
        DataType::Int32 => {
            EncodedValueRef::Int32(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        DataType::Date32 => {
            EncodedValueRef::Date32(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        DataType::Time32(_) => {
            EncodedValueRef::Time32(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        DataType::Float32 => EncodedValueRef::Float32(f32::from_bits(u32::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ]))),
        DataType::Int64 => EncodedValueRef::Int64(read_i64(data, position)),
        DataType::Float64 => EncodedValueRef::Float64(f64::from_bits(read_u64(data, position))),
        DataType::Decimal128(_, _) => EncodedValueRef::DecimalCompact(read_i64(data, position)),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            EncodedValueRef::TimestampMillis(read_i64(data, position))
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            EncodedValueRef::TimestampMicros(read_i64(data, position))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) | DataType::Struct(_)
            if types::is_timestamp_nanos(data_type) =>
        {
            EncodedValueRef::TimestampNanos {
                millis: read_i64(data, position),
                nanos_of_milli: i32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            }
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported encoded data type: {data_type:?}"),
            ));
        }
    };
    validate_encoded_value(data_type, value).map(|value| (value, width))
}

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
        dt if types::is_timestamp_nanos(dt) => DataVariant::TimestampNanos,
        DataType::Timestamp(_, _) => DataVariant::Int64,
        _ => DataVariant::Binary,
    }
}

fn const_values_are_row_aligned(variant: DataVariant) -> bool {
    match variant {
        DataVariant::Boolean
        | DataVariant::Int8
        | DataVariant::Int16
        | DataVariant::Int32
        | DataVariant::Int64
        | DataVariant::Float32
        | DataVariant::Float64
        | DataVariant::TimestampNanos => true,
        DataVariant::Binary => false,
    }
}

const CONST_FILL_ALL_MIN_NON_NULL_DENOMINATOR: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstFillStrategy {
    All,
    NonNullOnly,
    BulkFillIfAllPagesTouched,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstOutputLayout {
    FixedWidth { bytes_per_row: usize },
    BitPackedBoolean,
}

impl ConstOutputLayout {
    fn row_range_for_bytes(
        self,
        start_byte: usize,
        end_byte: usize,
        num_rows: usize,
    ) -> (usize, usize) {
        debug_assert!(start_byte < end_byte);
        match self {
            Self::FixedWidth { bytes_per_row } => {
                debug_assert!(bytes_per_row > 0);
                (
                    start_byte / bytes_per_row,
                    end_byte.div_ceil(bytes_per_row).min(num_rows),
                )
            }
            Self::BitPackedBoolean => (
                start_byte.saturating_mul(8).min(num_rows),
                end_byte.saturating_mul(8).min(num_rows),
            ),
        }
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
        dt if types::is_timestamp_nanos(dt) => RawColumnData::TimestampNanos {
            millis: Vec::new(),
            nanos_of_milli: Vec::new(),
        },
        DataType::Timestamp(_, _) => RawColumnData::Int64(Vec::new()),
        _ => RawColumnData::Binary {
            offsets: vec![0],
            data: Vec::new(),
        },
    }
}

fn invert_bitmap(bitmap: &[u8]) -> Vec<u8> {
    bitmap.iter().map(|b| !b).collect()
}

fn for_each_non_null(null_bitmap: &[u8], num_rows: usize, mut f: impl FnMut(usize)) {
    for (byte_index, &nulls) in null_bitmap.iter().enumerate() {
        let row_base = byte_index * 8;
        if row_base >= num_rows {
            break;
        }

        let rows_in_byte = (num_rows - row_base).min(8);
        let row_mask = if rows_in_byte == 8 {
            u8::MAX
        } else {
            (1u8 << rows_in_byte) - 1
        };
        let mut non_nulls = !nulls & row_mask;
        while non_nulls != 0 {
            let bit = non_nulls.trailing_zeros() as usize;
            f(row_base + bit);
            non_nulls &= non_nulls - 1;
        }
    }
}

fn row_range_has_non_null(null_bitmap: &[u8], start_row: usize, end_row: usize) -> bool {
    if start_row >= end_row {
        return false;
    }

    let first_byte = start_row / 8;
    let last_byte = (end_row - 1) / 8;
    let first_bit = start_row % 8;
    let end_bit = end_row % 8;

    if first_byte == last_byte {
        let end_mask = if end_bit == 0 {
            u8::MAX
        } else {
            (1u8 << end_bit) - 1
        };
        let row_mask = end_mask & (u8::MAX << first_bit);
        return (!null_bitmap[first_byte] & row_mask) != 0;
    }

    let mut full_start = first_byte;
    if first_bit != 0 {
        if (!null_bitmap[first_byte] & (u8::MAX << first_bit)) != 0 {
            return true;
        }
        full_start += 1;
    }

    let full_end = if end_bit == 0 {
        last_byte + 1
    } else {
        last_byte
    };
    if null_bitmap[full_start..full_end]
        .iter()
        .any(|&nulls| nulls != u8::MAX)
    {
        return true;
    }

    end_bit != 0 && (!null_bitmap[last_byte] & ((1u8 << end_bit) - 1)) != 0
}

fn for_each_non_null_row_run(null_bitmap: &[u8], num_rows: usize, mut f: impl FnMut(usize, usize)) {
    let mut run_start = None;
    let mut run_end = 0;
    for_each_non_null(null_bitmap, num_rows, |row| match run_start {
        None => {
            run_start = Some(row);
            run_end = row + 1;
        }
        Some(_) if row == run_end => run_end += 1,
        Some(_) => {
            if let Some(start) = run_start.replace(row) {
                f(start, run_end);
            }
            run_end = row + 1;
        }
    });
    if let Some(start) = run_start {
        f(start, run_end);
    }
}

fn system_page_size() -> Option<usize> {
    #[cfg(unix)]
    {
        static PAGE_SIZE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        *PAGE_SIZE.get_or_init(|| {
            // SAFETY: sysconf with _SC_PAGESIZE has no pointer arguments or caller-owned state.
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            (page_size > 0).then_some(page_size as usize)
        })
    }

    #[cfg(not(unix))]
    {
        None
    }
}

fn all_output_pages_touched(
    null_bitmap: &[u8],
    num_rows: usize,
    output_addr: usize,
    output_len_bytes: usize,
    page_size: usize,
    layout: ConstOutputLayout,
) -> bool {
    if output_len_bytes == 0 || page_size == 0 {
        return false;
    }
    let Some(output_end) = output_addr.checked_add(output_len_bytes) else {
        return false;
    };

    let mut page_start = output_addr - output_addr % page_size;
    loop {
        let Some(next_page) = page_start.checked_add(page_size) else {
            return false;
        };
        let start_byte = page_start.max(output_addr) - output_addr;
        let end_byte = next_page.min(output_end) - output_addr;
        let (start_row, end_row) = layout.row_range_for_bytes(start_byte, end_byte, num_rows);
        if !row_range_has_non_null(null_bitmap, start_row, end_row) {
            return false;
        }

        if next_page >= output_end {
            return true;
        }
        page_start = next_page;
    }
}

fn const_fill_strategy(
    has_nulls: bool,
    non_null_count: usize,
    num_rows: usize,
) -> ConstFillStrategy {
    if !has_nulls || non_null_count == num_rows {
        ConstFillStrategy::All
    } else if non_null_count < num_rows.div_ceil(CONST_FILL_ALL_MIN_NON_NULL_DENOMINATOR) {
        ConstFillStrategy::NonNullOnly
    } else {
        ConstFillStrategy::BulkFillIfAllPagesTouched
    }
}

trait ConstMaterializeValue: Default + Copy {
    fn has_default_bit_pattern(self) -> bool;
}

macro_rules! impl_const_materialize_integer {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ConstMaterializeValue for $ty {
                fn has_default_bit_pattern(self) -> bool {
                    self == 0
                }
            }
        )+
    };
}

impl_const_materialize_integer!(i8, i16, i32, i64);

impl ConstMaterializeValue for f32 {
    fn has_default_bit_pattern(self) -> bool {
        self.to_bits() == 0
    }
}

impl ConstMaterializeValue for f64 {
    fn has_default_bit_pattern(self) -> bool {
        self.to_bits() == 0
    }
}

fn materialize_fixed_const<T: ConstMaterializeValue>(
    value: T,
    num_rows: usize,
    strategy: ConstFillStrategy,
    null_bitmap: &[u8],
) -> Vec<T> {
    if strategy == ConstFillStrategy::All {
        return vec![value; num_rows];
    }

    let mut out = vec![T::default(); num_rows];
    if value.has_default_bit_pattern() {
        return out;
    }

    if strategy == ConstFillStrategy::BulkFillIfAllPagesTouched {
        if let Some(page_size) = system_page_size() {
            let output_len_bytes = out.len() * size_of::<T>();
            if all_output_pages_touched(
                null_bitmap,
                num_rows,
                out.as_ptr() as usize,
                output_len_bytes,
                page_size,
                ConstOutputLayout::FixedWidth {
                    bytes_per_row: size_of::<T>(),
                },
            ) {
                out.fill(value);
                return out;
            }
        }
    }

    match strategy {
        ConstFillStrategy::All => unreachable!(),
        ConstFillStrategy::NonNullOnly => {
            for_each_non_null(null_bitmap, num_rows, |row| out[row] = value);
        }
        ConstFillStrategy::BulkFillIfAllPagesTouched => {
            for_each_non_null_row_run(null_bitmap, num_rows, |start, end| {
                out[start..end].fill(value);
            });
        }
    }
    out
}

fn materialize_boolean_const(
    value: bool,
    num_rows: usize,
    strategy: ConstFillStrategy,
    null_bitmap: &[u8],
) -> Vec<u8> {
    let mut out = vec![0u8; num_rows.div_ceil(8)];
    if !value {
        return out;
    }

    match strategy {
        ConstFillStrategy::All => out.fill(u8::MAX),
        ConstFillStrategy::NonNullOnly => {
            for_each_non_null(null_bitmap, num_rows, |row| {
                out[row / 8] |= 1 << (row % 8);
            });
        }
        ConstFillStrategy::BulkFillIfAllPagesTouched => {
            if system_page_size().is_some_and(|page_size| {
                all_output_pages_touched(
                    null_bitmap,
                    num_rows,
                    out.as_ptr() as usize,
                    out.len(),
                    page_size,
                    ConstOutputLayout::BitPackedBoolean,
                )
            }) {
                out.fill(u8::MAX);
            } else {
                for_each_non_null(null_bitmap, num_rows, |row| {
                    out[row / 8] |= 1 << (row % 8);
                });
            }
        }
    }

    if num_rows & 7 != 0 {
        let last = out.len() - 1;
        out[last] &= (1u8 << (num_rows % 8)) - 1;
    }
    out
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
    values_are_row_aligned: bool,
) -> io::Result<ArrayRef> {
    debug_assert!(
        !values_are_row_aligned
            || match &data {
                RawColumnData::Boolean(values) => values.len() == num_rows.div_ceil(8),
                RawColumnData::Int8(values) => values.len() == num_rows,
                RawColumnData::Int16(values) => values.len() == num_rows,
                RawColumnData::Int32(values) => values.len() == num_rows,
                RawColumnData::Int64(values) => values.len() == num_rows,
                RawColumnData::Float32(values) => values.len() == num_rows,
                RawColumnData::Float64(values) => values.len() == num_rows,
                RawColumnData::Binary { .. } => false,
                RawColumnData::TimestampNanos {
                    millis,
                    nanos_of_milli,
                } => millis.len() == num_rows && nanos_of_milli.len() == num_rows,
            },
        "row-aligned CONST data must match the row cardinality"
    );

    let null_buf = make_null_buffer(null_bitmap.clone(), num_rows);
    let no_scatter = None;
    let scatter_bitmap = if values_are_row_aligned {
        &no_scatter
    } else {
        &null_bitmap
    };

    Ok(match data {
        RawColumnData::Boolean(values) => {
            let bool_buf = BooleanBuffer::new(Buffer::from_vec(values), 0, num_rows);
            Arc::new(BooleanArray::new(bool_buf, null_buf))
        }
        RawColumnData::Int8(values) => {
            let scattered = scatter_fixed(values, scatter_bitmap, num_rows);
            Arc::new(Int8Array::new(ScalarBuffer::from(scattered), null_buf))
        }
        RawColumnData::Int16(values) => {
            let scattered = scatter_fixed(values, scatter_bitmap, num_rows);
            Arc::new(Int16Array::new(ScalarBuffer::from(scattered), null_buf))
        }
        RawColumnData::Int32(values) => {
            let scattered = scatter_fixed(values, scatter_bitmap, num_rows);
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
            let scattered = scatter_fixed(values, scatter_bitmap, num_rows);
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
            let scattered = scatter_fixed(values, scatter_bitmap, num_rows);
            Arc::new(Float32Array::new(ScalarBuffer::from(scattered), null_buf))
        }
        RawColumnData::Float64(values) => {
            let scattered = scatter_fixed(values, scatter_bitmap, num_rows);
            Arc::new(Float64Array::new(ScalarBuffer::from(scattered), null_buf))
        }
        RawColumnData::Binary { offsets, data } => {
            let (i32_offsets, out_data) =
                scatter_binary_offsets(offsets, data, scatter_bitmap, num_rows);
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
            let millis_scattered = scatter_fixed(millis, scatter_bitmap, num_rows);
            let nanos_scattered = scatter_fixed(nanos_of_milli, scatter_bitmap, num_rows);

            match dt {
                DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
                    let values = match (values_are_row_aligned, null_buf.as_ref()) {
                        (true, Some(nulls)) => millis_scattered
                            .into_iter()
                            .zip(nanos_scattered)
                            .enumerate()
                            .map(|(row, (millis, nanos))| {
                                if nulls.is_null(row) {
                                    Ok(0)
                                } else {
                                    types::millis_nanos_to_ns(millis, nanos)
                                }
                            })
                            .collect::<io::Result<Vec<_>>>()?,
                        _ => millis_scattered
                            .into_iter()
                            .zip(nanos_scattered)
                            .map(|(millis, nanos)| types::millis_nanos_to_ns(millis, nanos))
                            .collect::<io::Result<Vec<_>>>()?,
                    };
                    let arr = TimestampNanosecondArray::new(ScalarBuffer::from(values), null_buf);
                    Arc::new(if let Some(tz) = tz {
                        arr.with_timezone(tz.clone())
                    } else {
                        arr
                    })
                }
                DataType::Struct(fields) if types::is_timestamp_nanos_struct(fields) => {
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
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("timestamp nanos data for non-nanos type: {:?}", dt),
                    ));
                }
            }
        }
    })
}

pub fn reassemble_list_columns_pub(
    arrays: &mut [ArrayRef],
    children: &[ChildColumnMeta],
    logical_type_refs: &[&DataType],
    num_primary: usize,
    num_rows: usize,
) {
    let logical_types: Vec<DataType> = logical_type_refs.iter().map(|t| (*t).clone()).collect();
    reassemble_list_columns(arrays, children, &logical_types, num_primary, num_rows);
}

fn reassemble_list_columns(
    arrays: &mut [ArrayRef],
    children: &[ChildColumnMeta],
    logical_types: &[DataType],
    _num_primary: usize,
    num_rows: usize,
) {
    let mut processed = vec![false; children.len()];

    // Process innermost children first (highest physical index → lowest)
    for idx in (0..children.len()).rev() {
        if processed[idx] {
            continue;
        }
        let child = &children[idx];
        match child.role {
            ChildColumnRole::MapValue => {
                let key_idx = children.iter().position(|candidate| {
                    candidate.parent_logical_col == child.parent_logical_col
                        && candidate.length_physical_index == child.length_physical_index
                        && candidate.role == ChildColumnRole::MapKey
                });
                let Some(key_idx) = key_idx else {
                    continue;
                };
                if processed[key_idx] {
                    continue;
                }

                let key_child = &children[key_idx];
                let lengths_idx = child.length_physical_index;
                let lengths = arrays[lengths_idx].clone();
                let keys = arrays[key_child.physical_index].clone();
                let values = arrays[child.physical_index].clone();
                let lengths_rows = lengths.len();

                let container_dt = if lengths_idx < logical_types.len() {
                    &logical_types[lengths_idx]
                } else if let Some(length_child) =
                    children.iter().find(|c| c.physical_index == lengths_idx)
                {
                    length_child.element_field.data_type()
                } else {
                    key_child.element_field.data_type()
                };

                if let DataType::Map(entries_field, sorted) = container_dt {
                    arrays[lengths_idx] = reassemble_map_array(
                        lengths,
                        keys,
                        values,
                        entries_field,
                        *sorted,
                        lengths_rows,
                    );
                } else {
                    let entries_field = Arc::new(Field::new(
                        "entries",
                        DataType::Struct(arrow_schema::Fields::from(vec![
                            key_child.element_field.as_ref().clone(),
                            child.element_field.as_ref().clone(),
                        ])),
                        false,
                    ));
                    arrays[lengths_idx] = reassemble_map_array(
                        lengths,
                        keys,
                        values,
                        &entries_field,
                        false,
                        lengths_rows,
                    );
                }
                processed[idx] = true;
                processed[key_idx] = true;
            }
            ChildColumnRole::MapKey => {}
            ChildColumnRole::ListElement => {
                let lengths_idx = child.length_physical_index;
                let lengths = arrays[lengths_idx].clone();
                let values = arrays[child.physical_index].clone();
                let lengths_rows = lengths.len();
                arrays[lengths_idx] = reassemble_list_array(
                    lengths,
                    values,
                    child.element_field.clone(),
                    lengths_rows,
                );
                processed[idx] = true;
            }
        }
    }

    // Handle ALL_NULL list/map columns
    for (i, lt) in logical_types.iter().enumerate() {
        if (matches!(lt, DataType::List(_)) || matches!(lt, DataType::Map(_, _)))
            && !children.iter().any(|c| c.parent_logical_col == i)
        {
            arrays[i] = arrow_array::new_null_array(lt, num_rows);
        }
    }
}

fn reassemble_map_array(
    lengths: ArrayRef,
    keys: ArrayRef,
    values: ArrayRef,
    entries_field: &Arc<Field>,
    sorted: bool,
    num_rows: usize,
) -> ArrayRef {
    let lengths_arr = lengths
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("map lengths must be Int32Array");

    let mut offsets: Vec<i32> = Vec::with_capacity(num_rows + 1);
    offsets.push(0);
    for i in 0..num_rows {
        let len = if lengths_arr.is_null(i) {
            0
        } else {
            lengths_arr.value(i)
        };
        offsets.push(offsets.last().unwrap() + len);
    }

    let null_buf = lengths_arr.nulls().cloned();
    let entries = StructArray::new(
        match entries_field.data_type() {
            DataType::Struct(fields) => fields.clone(),
            _ => unreachable!(),
        },
        vec![keys, values],
        None,
    );

    Arc::new(MapArray::new(
        entries_field.clone(),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        entries,
        null_buf,
        sorted,
    ))
}

fn reassemble_list_array(
    lengths: ArrayRef,
    values: ArrayRef,
    element_field: Arc<Field>,
    num_rows: usize,
) -> ArrayRef {
    let lengths_arr = lengths
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("list lengths must be Int32Array");

    let mut offsets: Vec<i32> = Vec::with_capacity(num_rows + 1);
    offsets.push(0);
    for i in 0..num_rows {
        let len = if lengths_arr.is_null(i) {
            0
        } else {
            lengths_arr.value(i)
        };
        offsets.push(offsets.last().unwrap() + len);
    }

    let null_buf = lengths_arr.nulls().cloned();
    let offset_buf = OffsetBuffer::new(ScalarBuffer::from(offsets));
    Arc::new(ListArray::new(element_field, offset_buf, values, null_buf))
}

use crate::bucket_writer::{expand_col_types, ChildColumnMeta, ChildColumnRole};

pub struct BucketReader {
    data: Vec<u8>,
    num_primary: usize,
    total_columns: usize,
    num_rows: usize,
    col_types: Vec<DataType>,

    encodings: Vec<u8>,
    has_nulls: Vec<bool>,
    null_bitmaps: Vec<Vec<u8>>,
    const_values: Vec<Value>,
    dict_values: Vec<Vec<Value>>,
    dict_bit_widths: Vec<usize>,
    data_cursors: Vec<usize>,

    logical_types: Vec<DataType>,
    children: Vec<ChildColumnMeta>,
    child_num_rows: Vec<usize>,
}

impl BucketReader {
    pub(crate) fn encoded_column(&self, column: usize) -> io::Result<EncodedColumn<'_>> {
        if column >= self.total_columns {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "column index {} out of range (num_columns={})",
                    column, self.total_columns
                ),
            ));
        }
        Ok(EncodedColumn::new(
            &self.col_types[column],
            self.encodings[column],
            self.has_nulls[column],
            &self.null_bitmaps[column],
            &self.const_values[column],
            &self.dict_values[column],
            self.dict_bit_widths[column],
            &self.data,
            self.data_cursors[column],
            self.col_num_rows(column),
        ))
    }

    fn col_num_rows(&self, col: usize) -> usize {
        if col < self.num_primary {
            self.num_rows
        } else {
            let child_idx = col - self.num_primary;
            if child_idx < self.child_num_rows.len() {
                self.child_num_rows[child_idx]
            } else {
                0
            }
        }
    }

    pub fn new(col_types: Vec<DataType>, data: Vec<u8>, num_rows: usize) -> io::Result<Self> {
        let logical_types = col_types.clone();
        let num_primary = col_types.len();
        let col_refs: Vec<&DataType> = col_types.iter().collect();
        let (physical_types, children) = expand_col_types(&col_refs);
        let total_columns = physical_types.len();

        let mut reader = BucketReader {
            data,
            num_primary,
            total_columns,
            num_rows,
            col_types: physical_types,
            encodings: vec![0; total_columns],
            has_nulls: vec![false; total_columns],
            null_bitmaps: Vec::new(),
            const_values: Vec::new(),
            dict_values: Vec::new(),
            dict_bit_widths: vec![0; total_columns],
            data_cursors: vec![0; total_columns],
            logical_types,
            children,
            child_num_rows: Vec::new(),
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
        self.null_bitmaps = vec![Vec::new(); self.total_columns];
        self.const_values = vec![Value::Null; self.total_columns];
        self.dict_values = vec![Vec::new(); self.total_columns];

        if self.num_rows == 0 || self.data.is_empty() {
            return Ok(());
        }

        let mut pos = 0;

        // Header: only present when ARRAY columns exist (backward compatible with v1)
        let has_children = !self.children.is_empty();
        if has_children {
            let _num_primary = varint::decode(&self.data, &mut pos)? as usize;
            let num_children = varint::decode(&self.data, &mut pos)? as usize;
            self.child_num_rows = Vec::with_capacity(num_children);
            for _ in 0..num_children {
                self.child_num_rows
                    .push(varint::decode(&self.data, &mut pos)? as usize);
            }
        }

        // 1. Encoding flags (2 bits per column)
        let encoding_flags_bytes = (self.total_columns * 2).div_ceil(8);
        self.check_bounds(pos, encoding_flags_bytes)?;
        for i in 0..self.total_columns {
            let byte_idx = (i * 2) / 8;
            let bit_idx = (i * 2) % 8;
            self.encodings[i] = (self.data[pos + byte_idx] >> bit_idx) & 0x03;
        }
        pos += encoding_flags_bytes;

        // 2. Has-nulls flags (1 bit per column)
        let has_nulls_bytes = self.total_columns.div_ceil(8);
        self.check_bounds(pos, has_nulls_bytes)?;
        for i in 0..self.total_columns {
            self.has_nulls[i] = (self.data[pos + i / 8] & (1 << (i % 8))) != 0;
        }
        pos += has_nulls_bytes;

        // 3. CONST metadata
        for i in 0..self.total_columns {
            if self.encodings[i] == ENCODING_CONST {
                let (value, size) = self.read_value_at(&self.col_types[i], pos)?;
                self.const_values[i] = value;
                pos += size;
            }
        }

        // 4. DICT metadata
        for i in 0..self.total_columns {
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

        // 5. Null bitmaps (per-column row count)
        for i in 0..self.total_columns {
            if self.has_nulls[i] && self.encodings[i] != ENCODING_ALL_NULL {
                let null_bitmap_bytes = self.col_num_rows(i).div_ceil(8);
                self.check_bounds(pos, null_bitmap_bytes)?;
                self.null_bitmaps[i] = self.data[pos..pos + null_bitmap_bytes].to_vec();
                pos += null_bitmap_bytes;
            }
        }

        // 6. Record column data start offsets, skip past data
        for i in 0..self.total_columns {
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
        let col_rows = self.col_num_rows(col);
        if !self.has_nulls[col] {
            return col_rows;
        }
        if self.encodings[col] == ENCODING_ALL_NULL {
            return 0;
        }
        let bitmap = &self.null_bitmaps[col];
        let full_bytes = col_rows / 8;
        let mut null_count = 0usize;
        for byte in bitmap.iter().take(full_bytes) {
            null_count += (*byte as u32).count_ones() as usize;
        }
        let remaining = col_rows % 8;
        if remaining > 0 {
            let mask = (1u8 << remaining) - 1;
            null_count += (bitmap[full_bytes] & mask).count_ones() as usize;
        }
        col_rows - null_count
    }

    /// Per-column encoding ids (in this bucket's column order). See `spec::ENCODING_*`.
    pub fn encodings(&self) -> &[u8] {
        &self.encodings
    }

    /// Dictionary entries for one column (empty if it is not dict-encoded).
    pub fn dict_values(&self, col: usize) -> &[Value] {
        &self.dict_values[col]
    }

    pub fn read_all_columns(&self) -> io::Result<Vec<ArrayRef>> {
        // Read all N+C physical columns
        let mut all_arrays: Vec<ArrayRef> = Vec::with_capacity(self.total_columns);

        for i in 0..self.total_columns {
            let col_rows = self.col_num_rows(i);
            let variant = data_variant_for_type(&self.col_types[i]);

            if self.encodings[i] == ENCODING_ALL_NULL {
                all_arrays.push(build_all_null_array(&self.col_types[i], col_rows));
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
                    col_rows,
                    has_nulls,
                    &self.null_bitmaps[i],
                    variant,
                )?,
                ENCODING_DICT => read_all_dict(
                    &self.data,
                    self.data_cursors[i],
                    &self.dict_values[i],
                    self.dict_bit_widths[i],
                    col_rows,
                    has_nulls,
                    &self.null_bitmaps[i],
                    variant,
                )?,
                ENCODING_PLAIN => read_all_plain(
                    &self.data,
                    self.data_cursors[i],
                    &self.col_types[i],
                    col_rows,
                    has_nulls,
                    &self.null_bitmaps[i],
                    variant,
                )?,
                _ => empty_raw_data_for_type(&self.col_types[i]),
            };

            all_arrays.push(build_array(
                data,
                &self.col_types[i],
                null_bitmap,
                col_rows,
                self.encodings[i] == ENCODING_CONST && const_values_are_row_aligned(variant),
            )?);
        }

        reassemble_list_columns(
            &mut all_arrays,
            &self.children,
            &self.logical_types,
            self.num_primary,
            self.num_rows,
        );

        // Return only the primary (logical) columns
        Ok(all_arrays.into_iter().take(self.num_primary).collect())
    }
}

pub struct ColumnPageReader {
    pub(crate) col_type: DataType,
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
    pub(crate) fn encoded_column(&self) -> EncodedColumn<'_> {
        EncodedColumn::new(
            &self.col_type,
            self.encoding,
            self.has_nulls,
            &self.null_bitmap,
            &self.const_value,
            &self.dict_values,
            self.dict_bit_width,
            &self.data,
            self.data_cursor,
            self.num_rows,
        )
    }

    pub fn new(
        col_type: DataType,
        encoding: u8,
        has_nulls: bool,
        const_value: Value,
        page_data: Vec<u8>,
        num_rows: usize,
    ) -> io::Result<Self> {
        Self::new_with_page_data_start(
            col_type,
            encoding,
            has_nulls,
            const_value,
            page_data,
            0,
            num_rows,
        )
    }

    pub(crate) fn new_with_page_data_start(
        col_type: DataType,
        encoding: u8,
        has_nulls: bool,
        const_value: Value,
        data: Vec<u8>,
        page_data_start: usize,
        num_rows: usize,
    ) -> io::Result<Self> {
        if !matches!(
            encoding,
            ENCODING_PLAIN | ENCODING_CONST | ENCODING_DICT | ENCODING_ALL_NULL
        ) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("column page: unsupported encoding {}", encoding),
            ));
        }
        if page_data_start > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "column page data start out of bounds",
            ));
        }

        let mut reader = ColumnPageReader {
            col_type,
            encoding,
            has_nulls,
            const_value,
            dict_values: Vec::new(),
            dict_bit_width: 0,
            null_bitmap: Vec::new(),
            data,
            data_cursor: page_data_start,
            num_rows,
        };
        reader.init_page()?;
        Ok(reader)
    }

    fn init_page(&mut self) -> io::Result<()> {
        let null_bitmap_bytes = self.num_rows.div_ceil(8);
        let mut pos = self.data_cursor;

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

    /// Encoding id of this column page. See `spec::ENCODING_*`.
    pub fn encoding(&self) -> u8 {
        self.encoding
    }

    /// Dictionary entries for a dict-encoded page (empty for other encodings).
    pub fn dict_values(&self) -> &[Value] {
        &self.dict_values
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

        build_array(
            data,
            &self.col_type,
            null_bitmap,
            num_rows,
            self.encoding == ENCODING_CONST && const_values_are_row_aligned(variant),
        )
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
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            debug_assert_eq!(width, 12);
            let millis = read_i64(buf, pos);
            let nanos =
                i32::from_be_bytes([buf[pos + 8], buf[pos + 9], buf[pos + 10], buf[pos + 11]]);
            Value::TimestampNanos {
                millis,
                nanos_of_milli: nanos,
            }
        }
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
    // Fixed-width CONST arrays are always materialized at row cardinality, so build_array can
    // wrap them without another scatter buffer. Sparse columns write only valid positions. At
    // ordinary densities, bulk-fill the whole buffer only when every physical page covered by the
    // actual allocation already contains valid rows. Otherwise fill only contiguous non-null row
    // runs so null-only pages remain untouched.
    let fill_strategy = const_fill_strategy(has_nulls, non_null_count, num_rows);

    match variant {
        DataVariant::Boolean => {
            let b = match const_value {
                Value::Boolean(v) => *v,
                _ => false,
            };
            Ok(RawColumnData::Boolean(materialize_boolean_const(
                b,
                num_rows,
                fill_strategy,
                null_bitmap,
            )))
        }
        DataVariant::Int8 => {
            let v = match const_value {
                Value::TinyInt(x) => *x,
                _ => 0,
            };
            Ok(RawColumnData::Int8(materialize_fixed_const(
                v,
                num_rows,
                fill_strategy,
                null_bitmap,
            )))
        }
        DataVariant::Int16 => {
            let v = match const_value {
                Value::SmallInt(x) => *x,
                _ => 0,
            };
            Ok(RawColumnData::Int16(materialize_fixed_const(
                v,
                num_rows,
                fill_strategy,
                null_bitmap,
            )))
        }
        DataVariant::Int32 => {
            let v = match const_value {
                Value::Integer(x) | Value::Date(x) | Value::Time(x) => *x,
                _ => 0,
            };
            Ok(RawColumnData::Int32(materialize_fixed_const(
                v,
                num_rows,
                fill_strategy,
                null_bitmap,
            )))
        }
        DataVariant::Int64 => {
            let v = match const_value {
                Value::BigInt(x)
                | Value::DecimalCompact(x)
                | Value::TimestampMillis(x)
                | Value::TimestampMicros(x) => *x,
                _ => 0,
            };
            Ok(RawColumnData::Int64(materialize_fixed_const(
                v,
                num_rows,
                fill_strategy,
                null_bitmap,
            )))
        }
        DataVariant::Float32 => {
            let v = match const_value {
                Value::Float(x) => *x,
                _ => 0.0,
            };
            Ok(RawColumnData::Float32(materialize_fixed_const(
                v,
                num_rows,
                fill_strategy,
                null_bitmap,
            )))
        }
        DataVariant::Float64 => {
            let v = match const_value {
                Value::Double(x) => *x,
                _ => 0.0,
            };
            Ok(RawColumnData::Float64(materialize_fixed_const(
                v,
                num_rows,
                fill_strategy,
                null_bitmap,
            )))
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
            Ok(RawColumnData::TimestampNanos {
                millis: materialize_fixed_const(m, num_rows, fill_strategy, null_bitmap),
                nanos_of_milli: materialize_fixed_const(n, num_rows, fill_strategy, null_bitmap),
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

fn read_bit_packed_checked(
    buf: &[u8],
    byte_base: usize,
    bit_offset: usize,
    bit_width: usize,
) -> io::Result<usize> {
    let bit_end = bit_offset
        .checked_add(bit_width)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "dict index offset overflow"))?;
    let byte_end = byte_base
        .checked_add(bit_end.div_ceil(8))
        .filter(|end| *end <= buf.len())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated dict indexes"))?;
    let _ = byte_end;
    Ok(read_bit_packed(buf, byte_base, bit_offset, bit_width))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn null_bitmap(num_rows: usize, non_null_rows: &[usize]) -> Vec<u8> {
        let mut bitmap = vec![u8::MAX; num_rows.div_ceil(8)];
        for &row in non_null_rows {
            bitmap[row / 8] &= !(1 << (row % 8));
        }
        bitmap
    }

    fn test_page_size() -> usize {
        system_page_size().unwrap_or(4096)
    }

    #[test]
    fn test_sparse_fixed_const_writes_only_non_null_slots() {
        let num_rows = 32;
        let bitmap = null_bitmap(num_rows, &[0, 31]);
        let data = read_all_const(
            &Value::BigInt(42),
            num_rows,
            true,
            &bitmap,
            DataVariant::Int64,
        )
        .unwrap();

        let RawColumnData::Int64(values) = data else {
            panic!("expected Int64 CONST data");
        };
        assert_eq!(values.len(), num_rows);
        assert_eq!(values[0], 42);
        assert_eq!(values[31], 42);
        assert!(values[1..31].iter().all(|&value| value == 0));
    }

    #[test]
    fn test_dense_fixed_const_values_are_row_aligned() {
        let num_rows = 32;
        let non_null_rows = (0..24).collect::<Vec<_>>();
        let bitmap = null_bitmap(num_rows, &non_null_rows);
        let data = read_all_const(
            &Value::BigInt(42),
            num_rows,
            true,
            &bitmap,
            DataVariant::Int64,
        )
        .unwrap();

        let RawColumnData::Int64(values) = data else {
            panic!("expected Int64 CONST data");
        };
        assert_eq!(values.len(), num_rows);
        for row in non_null_rows {
            assert_eq!(values[row], 42);
        }
    }

    #[test]
    fn test_interleaved_const_chunks_write_only_non_null_runs() {
        let rows_per_chunk = test_page_size() / size_of::<i64>();
        let num_rows = rows_per_chunk * 4;
        let non_null_rows = [0, 2]
            .into_iter()
            .flat_map(|chunk| {
                let start = chunk * rows_per_chunk;
                start..start + rows_per_chunk / 4
            })
            .collect::<Vec<_>>();
        assert_eq!(
            non_null_rows.len(),
            num_rows.div_ceil(CONST_FILL_ALL_MIN_NON_NULL_DENOMINATOR)
        );
        let bitmap = null_bitmap(num_rows, &non_null_rows);
        let data = read_all_const(
            &Value::BigInt(42),
            num_rows,
            true,
            &bitmap,
            DataVariant::Int64,
        )
        .unwrap();

        let RawColumnData::Int64(values) = data else {
            panic!("expected Int64 CONST data");
        };
        for (row, value) in values.into_iter().enumerate() {
            let chunk = row / rows_per_chunk;
            let row_in_chunk = row % rows_per_chunk;
            let expected = chunk & 1 == 0 && row_in_chunk < rows_per_chunk / 4;
            assert_eq!(value, if expected { 42 } else { 0 }, "row {row}");
        }
    }

    #[test]
    fn test_clustered_const_at_density_cutoff_fills_only_non_null_runs() {
        let rows_per_chunk = test_page_size() / size_of::<i64>();
        let num_rows = rows_per_chunk * CONST_FILL_ALL_MIN_NON_NULL_DENOMINATOR;
        let non_null_rows = (0..rows_per_chunk).collect::<Vec<_>>();
        let bitmap = null_bitmap(num_rows, &non_null_rows);
        let data = read_all_const(
            &Value::BigInt(42),
            num_rows,
            true,
            &bitmap,
            DataVariant::Int64,
        )
        .unwrap();

        let RawColumnData::Int64(values) = data else {
            panic!("expected Int64 CONST data");
        };
        assert!(values[..rows_per_chunk].iter().all(|&value| value == 42));
        assert!(values[rows_per_chunk..].iter().all(|&value| value == 0));
    }

    #[test]
    fn test_distributed_const_at_density_cutoff_preserves_valid_values() {
        let rows_per_chunk = test_page_size() / size_of::<i64>();
        let num_rows = rows_per_chunk * CONST_FILL_ALL_MIN_NON_NULL_DENOMINATOR;
        let non_null_rows = (0..num_rows)
            .step_by(CONST_FILL_ALL_MIN_NON_NULL_DENOMINATOR)
            .collect::<Vec<_>>();
        let bitmap = null_bitmap(num_rows, &non_null_rows);
        let data = read_all_const(
            &Value::BigInt(42),
            num_rows,
            true,
            &bitmap,
            DataVariant::Int64,
        )
        .unwrap();

        let RawColumnData::Int64(values) = data else {
            panic!("expected Int64 CONST data");
        };
        assert_eq!(values.len(), num_rows);
        for row in non_null_rows {
            assert_eq!(values[row], 42);
        }
    }

    #[test]
    fn test_output_page_coverage_distinguishes_distributed_and_clustered_values() {
        let page_size = 4096;
        let rows_per_page = page_size / size_of::<i64>();
        let num_rows = rows_per_page * CONST_FILL_ALL_MIN_NON_NULL_DENOMINATOR;
        let output_addr = 0;
        let output_len_bytes = num_rows * size_of::<i64>();
        let layout = ConstOutputLayout::FixedWidth {
            bytes_per_row: size_of::<i64>(),
        };

        let distributed_rows = (0..num_rows)
            .step_by(CONST_FILL_ALL_MIN_NON_NULL_DENOMINATOR)
            .collect::<Vec<_>>();
        let distributed_bitmap = null_bitmap(num_rows, &distributed_rows);
        assert!(all_output_pages_touched(
            &distributed_bitmap,
            num_rows,
            output_addr,
            output_len_bytes,
            page_size,
            layout,
        ));

        let clustered_rows = (0..rows_per_page).collect::<Vec<_>>();
        let clustered_bitmap = null_bitmap(num_rows, &clustered_rows);
        assert!(!all_output_pages_touched(
            &clustered_bitmap,
            num_rows,
            output_addr,
            output_len_bytes,
            page_size,
            layout,
        ));
    }

    #[test]
    fn test_page_coverage_accounts_for_allocation_offset() {
        let page_size = 4096;
        let rows_per_chunk = page_size / size_of::<i64>();
        let num_rows = rows_per_chunk * CONST_FILL_ALL_MIN_NON_NULL_DENOMINATOR;
        let non_null_rows = (0..CONST_FILL_ALL_MIN_NON_NULL_DENOMINATOR / 2)
            .flat_map(|pair| {
                let even_chunk_start = pair * 2 * rows_per_chunk;
                let odd_chunk_end = even_chunk_start + 2 * rows_per_chunk;
                (even_chunk_start..even_chunk_start + 127).chain(std::iter::once(odd_chunk_end - 1))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            non_null_rows.len(),
            num_rows.div_ceil(CONST_FILL_ALL_MIN_NON_NULL_DENOMINATOR)
        );
        let bitmap = null_bitmap(num_rows, &non_null_rows);
        for chunk_start in (0..num_rows).step_by(rows_per_chunk) {
            assert!(row_range_has_non_null(
                &bitmap,
                chunk_start,
                (chunk_start + rows_per_chunk).min(num_rows)
            ));
        }

        let layout = ConstOutputLayout::FixedWidth {
            bytes_per_row: size_of::<i64>(),
        };
        let output_len_bytes = num_rows * size_of::<i64>();
        assert!(all_output_pages_touched(
            &bitmap,
            num_rows,
            0,
            output_len_bytes,
            page_size,
            layout,
        ));
        assert!(!all_output_pages_touched(
            &bitmap,
            num_rows,
            16,
            output_len_bytes,
            page_size,
            layout,
        ));
    }

    #[test]
    fn test_page_coverage_ignores_trailing_bitmap_bits() {
        let page_size = 4096;
        let rows_per_page = page_size / size_of::<i64>();
        let num_rows = rows_per_page + 1;
        let non_null_rows = (0..num_rows.div_ceil(8)).collect::<Vec<_>>();
        let mut bitmap = null_bitmap(num_rows, &non_null_rows);
        bitmap[num_rows / 8] = 0b0000_0001;

        assert!(!all_output_pages_touched(
            &bitmap,
            num_rows,
            0,
            num_rows * size_of::<i64>(),
            page_size,
            ConstOutputLayout::FixedWidth {
                bytes_per_row: size_of::<i64>(),
            },
        ));
    }

    #[test]
    fn test_boolean_fallback_respects_non_null_and_bitmap_boundaries() {
        let rows_per_chunk = test_page_size() * 8;
        let num_rows = rows_per_chunk * 2 + 1;
        let non_null_rows = (0..num_rows.div_ceil(8)).collect::<Vec<_>>();
        let mut bitmap = null_bitmap(num_rows, &non_null_rows);
        bitmap[num_rows / 8] = 0b0000_0001;
        let data = read_all_const(
            &Value::Boolean(true),
            num_rows,
            true,
            &bitmap,
            DataVariant::Boolean,
        )
        .unwrap();

        let RawColumnData::Boolean(values) = data else {
            panic!("expected Boolean CONST data");
        };
        for row in 0..num_rows {
            let value = values[row / 8] & (1 << (row % 8)) != 0;
            assert_eq!(value, row < non_null_rows.len(), "row {row}");
        }
    }

    #[test]
    fn test_timestamp_nanos_uses_each_buffer_element_width() {
        let page_size = test_page_size();
        let millis_rows_per_chunk = page_size / size_of::<i64>();
        let nanos_rows_per_chunk = page_size / size_of::<i32>();
        let num_rows = millis_rows_per_chunk * CONST_FILL_ALL_MIN_NON_NULL_DENOMINATOR;
        let non_null_rows = (millis_rows_per_chunk..nanos_rows_per_chunk).collect::<Vec<_>>();
        let bitmap = null_bitmap(num_rows, &non_null_rows);
        let data = read_all_const(
            &Value::TimestampNanos {
                millis: 42,
                nanos_of_milli: 7,
            },
            num_rows,
            true,
            &bitmap,
            DataVariant::TimestampNanos,
        )
        .unwrap();

        let RawColumnData::TimestampNanos {
            millis,
            nanos_of_milli,
        } = data
        else {
            panic!("expected TimestampNanos CONST data");
        };
        assert!(millis[..millis_rows_per_chunk]
            .iter()
            .all(|&value| value == 0));
        assert!(millis[millis_rows_per_chunk..nanos_rows_per_chunk]
            .iter()
            .all(|&value| value == 42));
        assert!(millis[nanos_rows_per_chunk..]
            .iter()
            .all(|&value| value == 0));
        assert!(nanos_of_milli[..millis_rows_per_chunk]
            .iter()
            .all(|&value| value == 0));
        assert!(nanos_of_milli[millis_rows_per_chunk..nanos_rows_per_chunk]
            .iter()
            .all(|&value| value == 7));
        assert!(nanos_of_milli[nanos_rows_per_chunk..]
            .iter()
            .all(|&value| value == 0));
    }

    #[test]
    fn test_timestamp_nanos_conversion_ignores_null_hidden_values() {
        let (min_millis, min_nanos) = types::ns_to_millis_nanos(i64::MIN);
        let (max_millis, max_nanos) = types::ns_to_millis_nanos(i64::MAX);
        let data = RawColumnData::TimestampNanos {
            millis: vec![min_millis, min_millis, max_millis, max_millis],
            nanos_of_milli: vec![min_nanos, 0, max_nanos, 999_999],
        };
        let array = build_array(
            data,
            &DataType::Timestamp(TimeUnit::Nanosecond, None),
            Some(vec![0b0000_0101]),
            4,
            true,
        )
        .unwrap();
        let values = array
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap();

        assert_eq!(values.value(0), i64::MIN);
        assert!(values.is_null(1));
        assert_eq!(values.value(2), i64::MAX);
        assert!(values.is_null(3));
    }

    #[test]
    fn test_fixed_const_preserves_default_and_negative_zero_values() {
        let num_rows = test_page_size();
        let non_null_rows =
            (0..num_rows.div_ceil(CONST_FILL_ALL_MIN_NON_NULL_DENOMINATOR)).collect::<Vec<_>>();
        let bitmap = null_bitmap(num_rows, &non_null_rows);

        let zero_data = read_all_const(
            &Value::SmallInt(0),
            num_rows,
            true,
            &bitmap,
            DataVariant::Int16,
        )
        .unwrap();
        let RawColumnData::Int16(zero_values) = zero_data else {
            panic!("expected Int16 CONST data");
        };
        assert!(zero_values.iter().all(|&value| value == 0));

        let negative_zero_data = read_all_const(
            &Value::Float(-0.0),
            num_rows,
            true,
            &bitmap,
            DataVariant::Float32,
        )
        .unwrap();
        let RawColumnData::Float32(negative_zero_values) = negative_zero_data else {
            panic!("expected Float32 CONST data");
        };
        assert!(negative_zero_values[..non_null_rows.len()]
            .iter()
            .all(|value| value.to_bits() == (-0.0f32).to_bits()));
        assert!(negative_zero_values[non_null_rows.len()..]
            .iter()
            .all(|value| value.to_bits() == 0.0f32.to_bits()));
    }
}

#[cfg(test)]
mod encoded_column_tests {
    use super::*;

    #[test]
    fn column_page_rejects_unknown_encoding_without_panicking() {
        let result = std::panic::catch_unwind(|| -> io::Result<()> {
            let page =
                ColumnPageReader::new(DataType::Int32, 0xff, true, Value::Null, Vec::new(), 1)?;
            page.read_all()?;
            Ok(())
        });

        let err = result
            .expect("unknown column page encoding must not panic")
            .expect_err("unknown column page encoding must be rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("unsupported encoding 255"));
    }

    #[test]
    fn rejects_invalid_timestamp_nanos_for_all_encodings() {
        let data_type = DataType::Timestamp(TimeUnit::Nanosecond, None);
        let invalid = Value::TimestampNanos {
            millis: 0,
            nanos_of_milli: 1_000_000,
        };
        let placeholder = Value::Null;

        let constant = EncodedColumn::new(
            &data_type,
            ENCODING_CONST,
            false,
            &[],
            &invalid,
            &[],
            0,
            &[],
            0,
            1,
        );
        assert_eq!(
            constant.constant().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            constant.values().next().unwrap().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );

        let dictionary = EncodedColumn::new(
            &data_type,
            ENCODING_DICT,
            false,
            &[],
            &placeholder,
            std::slice::from_ref(&invalid),
            0,
            &[],
            0,
            1,
        );
        assert_eq!(
            dictionary.values().next().unwrap().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );

        let mut plain_data = 0i64.to_be_bytes().to_vec();
        plain_data.extend_from_slice(&1_000_000i32.to_be_bytes());
        let plain = EncodedColumn::new(
            &data_type,
            ENCODING_PLAIN,
            false,
            &[],
            &placeholder,
            &[],
            0,
            &plain_data,
            0,
            1,
        );
        assert_eq!(
            plain.values().next().unwrap().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn reports_corrupt_dict_indexes_without_panicking() {
        let data_type = DataType::Int32;
        let placeholder = Value::Null;
        let dict_values = [Value::Integer(7)];

        let truncated = EncodedColumn::new(
            &data_type,
            ENCODING_DICT,
            false,
            &[],
            &placeholder,
            &dict_values,
            1,
            &[],
            0,
            1,
        );
        assert_eq!(
            truncated.values().next().unwrap().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );

        let out_of_range = EncodedColumn::new(
            &data_type,
            ENCODING_DICT,
            false,
            &[],
            &placeholder,
            &dict_values,
            1,
            &[1],
            0,
            1,
        );
        assert_eq!(
            out_of_range.values().next().unwrap().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn reports_truncated_plain_values_without_panicking() {
        let placeholder = Value::Null;

        let fixed = EncodedColumn::new(
            &DataType::Int64,
            ENCODING_PLAIN,
            false,
            &[],
            &placeholder,
            &[],
            0,
            &[0; 7],
            0,
            1,
        );
        assert_eq!(
            fixed.values().next().unwrap().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );

        let variable = EncodedColumn::new(
            &DataType::Utf8,
            ENCODING_PLAIN,
            false,
            &[],
            &placeholder,
            &[],
            0,
            &[3, b'a', b'b'],
            0,
            1,
        );
        assert_eq!(
            variable.values().next().unwrap().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }
}
