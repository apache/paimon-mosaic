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

use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::Arc;

use arrow_array::*;
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field};

use crate::integer_buffer::IntegerBuffer;
use crate::spec::*;
use crate::types;
use crate::values;
use crate::varint;

pub(crate) struct BucketEncodingPlan {
    pub(crate) encodings: Vec<u8>,
    pub(crate) has_nulls: Vec<bool>,
    // The original finish_paged payload sizes, excluding const metadata and
    // the two-byte page prefix. Keep these for the existing layout heuristic.
    pub(crate) page_sizes: Vec<Option<usize>>,
}

pub struct PagedBucketOutput {
    pub encodings: Vec<u8>,
    pub has_nulls: Vec<bool>,
    pub const_data: Vec<Vec<u8>>,
    pub column_pages: Vec<Option<Vec<u8>>>,
    pub num_primary: usize,
    pub children: Vec<ChildColumnMeta>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChildColumnRole {
    ListElement,
    MapKey,
    MapValue,
}

#[derive(Clone, Debug)]
pub struct ChildColumnMeta {
    pub parent_logical_col: usize,
    pub physical_index: usize,
    pub length_physical_index: usize,
    pub role: ChildColumnRole,
    pub element_field: Arc<Field>,
    pub num_elements: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DictTracking {
    Pending,
    Active,
    Disabled,
}

pub struct BucketWriter {
    num_primary: usize,
    total_columns: usize,
    fixed_widths: Vec<i32>,

    null_bitmaps: Vec<Vec<u8>>,
    value_buffers: Vec<Vec<u8>>,
    integer_buffers: Vec<Option<Box<IntegerBuffer>>>,
    non_null_counts: Vec<usize>,

    const_tracking: Vec<bool>,
    first_value_len: Vec<usize>,

    long_dict_maps: Vec<Option<HashMap<u64, usize>>>,
    byte_dict_maps: Vec<Option<HashMap<Vec<u8>, usize>>>,
    dict_tracking: Vec<DictTracking>,
    dict_total_bytes: Vec<usize>,
    max_dict_total_bytes: usize,
    max_dict_entries: usize,

    num_rows: usize,
    children: Vec<ChildColumnMeta>,
}

impl BucketWriter {
    pub fn new(
        col_types: &[&DataType],
        max_dict_total_bytes: usize,
        max_dict_entries: usize,
    ) -> Self {
        let num_primary = col_types.len();
        let (physical_types, children) = expand_col_types(col_types);
        let total_columns = physical_types.len();
        let fixed_widths: Vec<i32> = physical_types.iter().map(types::fixed_width).collect();
        let integer_buffers: Vec<_> = physical_types
            .iter()
            .map(|dt| match dt {
                DataType::Int16 => Some(Box::new(IntegerBuffer::new(2))),
                DataType::Int32 => Some(Box::new(IntegerBuffer::new(4))),
                DataType::Int64 => Some(Box::new(IntegerBuffer::new(8))),
                _ => None,
            })
            .collect();
        BucketWriter {
            num_primary,
            total_columns,
            fixed_widths,
            null_bitmaps: vec![vec![0u8; 128]; total_columns],
            value_buffers: vec![Vec::new(); total_columns],
            integer_buffers,
            non_null_counts: vec![0; total_columns],
            const_tracking: vec![true; total_columns],
            first_value_len: vec![0; total_columns],
            long_dict_maps: (0..total_columns).map(|_| None).collect(),
            byte_dict_maps: (0..total_columns).map(|_| None).collect(),
            dict_tracking: vec![DictTracking::Pending; total_columns],
            dict_total_bytes: vec![0; total_columns],
            max_dict_total_bytes,
            max_dict_entries,
            num_rows: 0,
            children,
        }
    }

    pub fn num_primary(&self) -> usize {
        self.num_primary
    }

    pub fn children(&self) -> &[ChildColumnMeta] {
        &self.children
    }

    fn find_child_index(
        &self,
        parent_logical_col: usize,
        length_physical_index: usize,
        role: ChildColumnRole,
    ) -> io::Result<usize> {
        self.children
            .iter()
            .position(|c| {
                c.parent_logical_col == parent_logical_col
                    && c.length_physical_index == length_physical_index
                    && c.role == role
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "missing complex child column: parent={}, length={}, role={:?}",
                        parent_logical_col, length_physical_index, role
                    ),
                )
            })
    }

    fn col_num_rows(&self, col: usize) -> usize {
        if col < self.num_primary {
            self.num_rows
        } else {
            self.children
                .iter()
                .find(|c| c.physical_index == col)
                .map_or(0, |c| c.num_elements)
        }
    }

    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    pub fn estimated_raw_size(&self) -> usize {
        if self.num_rows == 0 {
            return 0;
        }
        let (encodings, has_nulls) = self.compute_encodings();
        self.compute_out_size(&encodings, &has_nulls)
    }

    pub fn write_columns(
        &mut self,
        arrays: &[&dyn Array],
        data_types: &[&DataType],
    ) -> io::Result<usize> {
        debug_assert_eq!(arrays.len(), self.num_primary);
        let num_new_rows = arrays[0].len();
        if num_new_rows == 0 {
            return Ok(0);
        }
        let start_row = self.num_rows;
        let mut total_size = 0;

        // Split List/Map arrays into lengths + child values
        struct ColSplit {
            col: usize,
            lengths: Int32Array,
        }
        let mut splits: Vec<ColSplit> = Vec::new();
        let mut pending: Vec<(usize, ArrayRef)> = Vec::new();
        let mut seen_cols = Vec::new();
        for child in &self.children {
            let col = child.parent_logical_col;
            if seen_cols.contains(&col) {
                continue;
            }
            seen_cols.push(col);

            match data_types[col] {
                DataType::List(_) => {
                    let list_array = arrays[col]
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidInput, "expected ListArray")
                        })?;
                    let lengths = extract_list_lengths(list_array);
                    let values = flatten_list_values(list_array);
                    splits.push(ColSplit { col, lengths });
                    let child_idx =
                        self.find_child_index(col, col, ChildColumnRole::ListElement)?;
                    pending.push((child_idx, values));
                }
                DataType::Map(_, _) => {
                    let map_array =
                        arrays[col]
                            .as_any()
                            .downcast_ref::<MapArray>()
                            .ok_or_else(|| {
                                io::Error::new(io::ErrorKind::InvalidInput, "expected MapArray")
                            })?;
                    let lengths = extract_map_lengths(map_array);
                    let (keys, values) = flatten_map_entries(map_array);
                    splits.push(ColSplit { col, lengths });
                    let key_idx = self.find_child_index(col, col, ChildColumnRole::MapKey)?;
                    let value_idx = self.find_child_index(col, col, ChildColumnRole::MapValue)?;
                    pending.push((key_idx, keys));
                    pending.push((value_idx, values));
                }
                _ => {}
            }
        }

        // Write primary columns (lengths for ARRAY/MAP cols, regular data for others)
        let int32_dt = DataType::Int32;
        for i in 0..self.num_primary {
            if let Some(split) = splits.iter().find(|s| s.col == i) {
                total_size += self.append_array_column(i, &split.lengths, &int32_dt, start_row)?;
            } else {
                total_size += self.append_array_column(i, arrays[i], data_types[i], start_row)?;
            }
        }

        // Write child columns. Nested children are located through explicit layout metadata.
        while let Some((child_idx, values)) = pending.pop() {
            if child_idx >= self.children.len() || values.is_empty() {
                continue;
            }
            let phys_idx = self.children[child_idx].physical_index;
            let child_start = self.children[child_idx].num_elements;

            match values.data_type() {
                DataType::List(_) => {
                    let inner_list =
                        values.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidInput, "expected ListArray")
                        })?;
                    let inner_lengths = extract_list_lengths(inner_list);
                    let inner_values = flatten_list_values(inner_list);
                    total_size +=
                        self.append_array_column(phys_idx, &inner_lengths, &int32_dt, child_start)?;
                    self.children[child_idx].num_elements += inner_lengths.len();
                    let nested_idx = self.find_child_index(
                        self.children[child_idx].parent_logical_col,
                        phys_idx,
                        ChildColumnRole::ListElement,
                    )?;
                    pending.push((nested_idx, inner_values));
                }
                DataType::Map(_, _) => {
                    let inner_map =
                        values.as_any().downcast_ref::<MapArray>().ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidInput, "expected MapArray")
                        })?;
                    let inner_lengths = extract_map_lengths(inner_map);
                    let (inner_keys, inner_values) = flatten_map_entries(inner_map);
                    total_size +=
                        self.append_array_column(phys_idx, &inner_lengths, &int32_dt, child_start)?;
                    self.children[child_idx].num_elements += inner_lengths.len();
                    let parent = self.children[child_idx].parent_logical_col;
                    let key_idx =
                        self.find_child_index(parent, phys_idx, ChildColumnRole::MapKey)?;
                    let value_idx =
                        self.find_child_index(parent, phys_idx, ChildColumnRole::MapValue)?;
                    pending.push((key_idx, inner_keys));
                    pending.push((value_idx, inner_values));
                }
                _ => {
                    let elem_dt = self.children[child_idx].element_field.data_type().clone();
                    total_size +=
                        self.append_array_column(phys_idx, values.as_ref(), &elem_dt, child_start)?;
                    self.children[child_idx].num_elements += values.len();
                }
            }
        }

        self.num_rows += num_new_rows;
        total_size += num_new_rows * self.num_primary.div_ceil(8);
        Ok(total_size)
    }

    fn append_array_column(
        &mut self,
        col: usize,
        array: &dyn Array,
        dt: &DataType,
        start_row: usize,
    ) -> io::Result<usize> {
        let num_new_rows = array.len();
        let needed_bytes = (start_row + num_new_rows).div_ceil(8);
        if self.null_bitmaps[col].len() < needed_bytes {
            self.null_bitmaps[col].resize(needed_bytes.next_power_of_two(), 0);
        }

        let typed = downcast_array(array, dt)?;
        let null_count = array.null_count();

        if null_count == num_new_rows {
            let nulls = array
                .nulls()
                .expect("an all-null array must have a physical null buffer");
            append_null_bitmap(&mut self.null_bitmaps[col], start_row, nulls);
            return Ok(0);
        }

        if self.integer_buffers[col].is_some() {
            return match &typed {
                TypedArrayRef::Int16(a) => self.append_integer_column(col, a, start_row),
                TypedArrayRef::Int32(a) => self.append_integer_column(col, a, start_row),
                TypedArrayRef::Int64(a) => self.append_integer_column(col, a, start_row),
                _ => unreachable!("integer buffer requires an integer array"),
            };
        }

        if null_count == 0 {
            if let Some(col_size) =
                self.append_no_null_batch(col, &typed, start_row, num_new_rows)?
            {
                return Ok(col_size);
            }
        }

        if let TypedArrayRef::TimestampMillis(array) = &typed {
            return Ok(self.append_nullable_timestamp_millis_batch(
                col,
                array,
                start_row,
                num_new_rows,
            ));
        }

        let mut col_size = 0usize;
        for row in 0..num_new_rows {
            let abs_row = start_row + row;
            if array.is_null(row) {
                self.null_bitmaps[col][abs_row / 8] |= 1 << (abs_row % 8);
            } else {
                self.non_null_counts[col] += 1;
                let before = self.value_buffers[col].len();
                write_typed_value(&mut self.value_buffers[col], &typed, row)?;
                let written = self.value_buffers[col].len() - before;
                col_size += written;

                self.track_encoding_value(col, before, written);
            }
        }
        Ok(col_size)
    }

    fn append_integer_column<T: arrow_array::types::ArrowPrimitiveType>(
        &mut self,
        col: usize,
        array: &PrimitiveArray<T>,
        start_row: usize,
    ) -> io::Result<usize>
    where
        T::Native: Into<i64>,
    {
        if array.null_count() == 0 {
            return Ok(self.append_integer_values(col, array.values()));
        }
        let nulls = array.nulls().expect("nullable integer batch");
        append_null_bitmap(&mut self.null_bitmaps[col], start_row, nulls);
        let mut scratch = [0i64; 4096];
        let mut used = 0;
        let mut size = 0;
        for row in nulls.valid_indices() {
            scratch[used] = array.value(row).into();
            used += 1;
            if used == scratch.len() {
                size += self.append_integer_values(col, &scratch);
                used = 0;
            }
        }
        size += self.append_integer_values(col, &scratch[..used]);
        Ok(size)
    }

    fn append_integer_values<T: Copy + Into<i64>>(&mut self, col: usize, values: &[T]) -> usize {
        if values.is_empty() {
            return 0;
        }
        let width = self.fixed_widths[col] as usize;
        let mask = u64::MAX >> (64 - width * 8);
        let mut dict_start = 0;
        if self.const_tracking[col] {
            let first = self.integer_buffers[col]
                .as_ref()
                .unwrap()
                .keys()
                .next()
                .unwrap_or(values[0].into() as u64 & mask);
            self.first_value_len[col] = width;
            if let Some(first_different) = values
                .iter()
                .position(|&v| (v.into() as u64 & mask) != first)
            {
                // Preserve lazy dictionary activation: seed the constant prefix
                // once, then track values starting at the first different key.
                debug_assert_eq!(self.dict_tracking[col], DictTracking::Pending);
                self.const_tracking[col] = false;
                let dict = self.long_dict_maps[col].get_or_insert_with(HashMap::new);
                dict.clear();
                dict.insert(first, 0);
                self.dict_tracking[col] = DictTracking::Active;
                dict_start = first_different;
            } else {
                dict_start = values.len();
            }
        }
        if self.dict_tracking[col] == DictTracking::Active {
            let dict = self.long_dict_maps[col].as_mut().unwrap();
            if dict.len() <= self.max_dict_entries {
                for &value in &values[dict_start..] {
                    let key = value.into() as u64 & mask;
                    let next = dict.len();
                    dict.entry(key).or_insert(next);
                    if dict.len() > self.max_dict_entries {
                        break;
                    }
                }
            }
            if dict.len() > self.max_dict_entries {
                self.long_dict_maps[col] = None;
                self.dict_tracking[col] = DictTracking::Disabled;
            }
        }
        self.integer_buffers[col]
            .as_mut()
            .unwrap()
            .extend_from_slice(values);
        self.non_null_counts[col] += values.len();
        // Row-group boundaries and dictionary decisions use the logical width,
        // never the compact allocation size.
        values.len() * width
    }

    fn logical_value_size(&self, col: usize) -> usize {
        self.integer_buffers[col]
            .as_ref()
            .map_or(self.value_buffers[col].len(), |b| b.logical_len())
    }

    fn append_no_null_batch(
        &mut self,
        col: usize,
        typed: &TypedArrayRef,
        start_row: usize,
        num_rows: usize,
    ) -> io::Result<Option<usize>> {
        let buf = &mut self.value_buffers[col];
        let before_all = buf.len();

        match typed {
            TypedArrayRef::Boolean(a) => {
                buf.reserve(num_rows);
                for i in 0..num_rows {
                    buf.push(if a.value(i) { 1 } else { 0 });
                }
            }
            TypedArrayRef::Int8(a) => {
                let vals = a.values();
                buf.reserve(num_rows);
                for &v in vals.iter() {
                    buf.push(v as u8);
                }
            }
            TypedArrayRef::Int16(a) => {
                let vals = a.values();
                buf.reserve(num_rows * 2);
                for &v in vals.iter() {
                    buf.extend_from_slice(&v.to_be_bytes());
                }
            }
            TypedArrayRef::Int32(a) => {
                let vals = a.values();
                buf.reserve(num_rows * 4);
                for &v in vals.iter() {
                    buf.extend_from_slice(&v.to_be_bytes());
                }
            }
            TypedArrayRef::Int64(a) => {
                let vals = a.values();
                buf.reserve(num_rows * 8);
                for &v in vals.iter() {
                    buf.extend_from_slice(&v.to_be_bytes());
                }
            }
            TypedArrayRef::Float32(a) => {
                let vals = a.values();
                buf.reserve(num_rows * 4);
                for &v in vals.iter() {
                    buf.extend_from_slice(&v.to_bits().to_be_bytes());
                }
            }
            TypedArrayRef::Float64(a) => {
                let vals = a.values();
                buf.reserve(num_rows * 8);
                for &v in vals.iter() {
                    buf.extend_from_slice(&v.to_bits().to_be_bytes());
                }
            }
            TypedArrayRef::Date32(a) => {
                let vals = a.values();
                buf.reserve(num_rows * 4);
                for &v in vals.iter() {
                    buf.extend_from_slice(&v.to_be_bytes());
                }
            }
            TypedArrayRef::Time32(a) => {
                let vals = a.values();
                buf.reserve(num_rows * 4);
                for &v in vals.iter() {
                    buf.extend_from_slice(&v.to_be_bytes());
                }
            }
            TypedArrayRef::Decimal128Compact(a) => {
                let vals = a.values();
                buf.reserve(num_rows * 8);
                for &v in vals.iter() {
                    buf.extend_from_slice(&(v as i64).to_be_bytes());
                }
            }
            TypedArrayRef::TimestampMillis(a) => {
                let vals = a.values();
                buf.reserve(num_rows * 8);
                for &v in vals.iter() {
                    buf.extend_from_slice(&v.to_be_bytes());
                }
            }
            TypedArrayRef::TimestampMicros(a) => {
                let vals = a.values();
                buf.reserve(num_rows * 8);
                for &v in vals.iter() {
                    buf.extend_from_slice(&v.to_be_bytes());
                }
            }
            TypedArrayRef::TimestampNanos(a) => {
                let vals = a.values();
                buf.reserve(num_rows * 12);
                for &v in vals.iter() {
                    write_timestamp_nanos_value(buf, v);
                }
            }
            TypedArrayRef::LegacyTimestampNanos { millis, nanos } => {
                let m_vals = millis.values();
                let n_vals = nanos.values();
                buf.reserve(num_rows * 12);
                for i in 0..num_rows {
                    write_legacy_timestamp_nanos_value(buf, m_vals[i], n_vals[i])?;
                }
            }
            _ => return Ok(None),
        }

        let col_size = buf.len() - before_all;
        let fw = self.fixed_widths[col] as usize;
        if !self.const_tracking[col] && self.dict_tracking[col] == DictTracking::Disabled {
            self.non_null_counts[col] += num_rows;
        } else {
            for i in 0..num_rows {
                self.non_null_counts[col] += 1;
                self.track_encoding_value(col, before_all + i * fw, fw);
                if !self.const_tracking[col] && self.dict_tracking[col] == DictTracking::Disabled {
                    self.non_null_counts[col] += num_rows - i - 1;
                    break;
                }
            }
        }

        // null_bitmap stays all-zero (no nulls), start_row offsets are fine
        let _ = start_row;

        Ok(Some(col_size))
    }

    fn append_nullable_timestamp_millis_batch(
        &mut self,
        col: usize,
        array: &TimestampMillisecondArray,
        start_row: usize,
        num_rows: usize,
    ) -> usize {
        let nulls = array
            .nulls()
            .expect("nullable timestamp batch must have a physical null buffer");
        debug_assert!(nulls.null_count() > 0);
        debug_assert!(nulls.null_count() < num_rows);

        append_null_bitmap(&mut self.null_bitmaps[col], start_row, nulls);

        let valid_count = num_rows - nulls.null_count();
        self.value_buffers[col].reserve(valid_count * 8);
        let values = array.values();
        let mut track_encoding =
            self.const_tracking[col] || self.dict_tracking[col] != DictTracking::Disabled;

        for row in nulls.valid_indices() {
            let before = self.value_buffers[col].len();
            self.value_buffers[col].extend_from_slice(&values[row].to_be_bytes());
            self.non_null_counts[col] += 1;

            if track_encoding {
                self.track_encoding_value(col, before, 8);
                track_encoding =
                    self.const_tracking[col] || self.dict_tracking[col] != DictTracking::Disabled;
            }
        }

        valid_count * 8
    }

    fn track_encoding_value(&mut self, col: usize, value_start: usize, value_len: usize) {
        if self.non_null_counts[col] == 1 {
            self.first_value_len[col] = value_len;
            return;
        }

        if self.const_tracking[col] {
            if value_len == self.first_value_len[col]
                && equals_first_value(&self.value_buffers[col], value_start, value_len)
            {
                return;
            }

            self.const_tracking[col] = false;
            self.activate_dict_tracking(col, value_start, value_len);
        } else {
            self.track_dict_value(col, value_start, value_len);
        }
    }

    fn activate_dict_tracking(&mut self, col: usize, value_start: usize, value_len: usize) {
        debug_assert_eq!(self.dict_tracking[col], DictTracking::Pending);
        self.dict_total_bytes[col] = 0;
        if uses_long_dict(self.fixed_widths[col]) {
            self.long_dict_maps[col]
                .get_or_insert_with(HashMap::new)
                .clear();
        } else {
            self.byte_dict_maps[col]
                .get_or_insert_with(HashMap::new)
                .clear();
        }
        self.dict_tracking[col] = DictTracking::Active;

        self.track_dict_value(col, 0, self.first_value_len[col]);
        if self.dict_tracking[col] == DictTracking::Active {
            self.track_dict_value(col, value_start, value_len);
        }
    }

    fn track_dict_value(&mut self, col: usize, value_start: usize, value_len: usize) {
        if self.dict_tracking[col] != DictTracking::Active {
            return;
        }

        let disable = if let Some(ref mut dict) = self.long_dict_maps[col] {
            let key = values::extract_fixed_key(
                &self.value_buffers[col],
                value_start,
                self.fixed_widths[col],
            );
            let len = dict.len();
            dict.entry(key).or_insert(len);
            dict.len() > self.max_dict_entries
        } else if let Some(ref mut dict) = self.byte_dict_maps[col] {
            let value = &self.value_buffers[col][value_start..value_start + value_len];
            if !dict.contains_key(value) {
                let len = dict.len();
                dict.insert(value.to_vec(), len);
                self.dict_total_bytes[col] += value_len;
            }
            dict.len() > self.max_dict_entries
                || self.dict_total_bytes[col] > self.max_dict_total_bytes
        } else {
            unreachable!("active dictionary tracking must have a dictionary map")
        };

        if disable {
            self.dict_tracking[col] = DictTracking::Disabled;
            self.long_dict_maps[col] = None;
            self.byte_dict_maps[col] = None;
        }
    }

    fn compute_encodings(&self) -> (Vec<u8>, Vec<bool>) {
        let mut encodings = vec![0u8; self.total_columns];
        let mut has_nulls = vec![false; self.total_columns];
        for i in 0..self.total_columns {
            let col_rows = self.col_num_rows(i);
            if self.non_null_counts[i] == 0 {
                encodings[i] = ENCODING_ALL_NULL;
            } else if self.const_tracking[i] {
                encodings[i] = ENCODING_CONST;
                has_nulls[i] = self.non_null_counts[i] < col_rows;
            } else {
                let dict_size = self.get_dict_size(i);
                if dict_size >= 2
                    && dict_size <= self.max_dict_entries
                    && self.dict_encoded_size(i) < self.logical_value_size(i)
                {
                    encodings[i] = ENCODING_DICT;
                } else {
                    encodings[i] = ENCODING_PLAIN;
                }
                has_nulls[i] = self.non_null_counts[i] < col_rows;
            }
        }
        (encodings, has_nulls)
    }

    /// Encode into a caller-owned sink without materializing the whole bucket.
    pub(crate) fn write_encoded(&self, out: &mut impl Write) -> io::Result<()> {
        if self.num_rows == 0 {
            return Ok(());
        }
        let plan = self.encoding_plan();
        if !self.children.is_empty() {
            write_varint(out, self.num_primary as u32)?;
            write_varint(out, self.children.len() as u32)?;
            for child in &self.children {
                write_varint(out, child.num_elements as u32)?;
            }
        }
        let mut flags = vec![0; (self.total_columns * 2).div_ceil(8)];
        for (i, &encoding) in plan.encodings.iter().enumerate() {
            flags[i / 4] |= encoding << ((i % 4) * 2);
        }
        out.write_all(&flags)?;
        flags.clear();
        flags.resize(self.total_columns.div_ceil(8), 0);
        for (i, &has_nulls) in plan.has_nulls.iter().enumerate() {
            if has_nulls {
                flags[i / 8] |= 1 << (i % 8);
            }
        }
        out.write_all(&flags)?;
        for (col, &encoding) in plan.encodings.iter().enumerate() {
            if encoding == ENCODING_CONST {
                self.write_const(col, out)?;
            }
        }
        for (col, &encoding) in plan.encodings.iter().enumerate() {
            if encoding == ENCODING_DICT {
                self.write_dictionary(col, out)?;
            }
        }
        for (col, &has_nulls) in plan.has_nulls.iter().enumerate() {
            if has_nulls && plan.encodings[col] != ENCODING_ALL_NULL {
                self.write_nulls(col, out)?;
            }
        }
        for (col, &encoding) in plan.encodings.iter().enumerate() {
            self.write_values(col, encoding, out)?;
        }
        Ok(())
    }

    pub fn finish(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_encoded(&mut out)
            .expect("writing into a Vec cannot fail");
        out
    }

    pub(crate) fn has_integer_buffers(&self) -> bool {
        self.integer_buffers.iter().any(Option::is_some)
    }

    #[cfg(test)]
    pub(crate) fn use_wide_integer_buffers(&mut self) {
        assert_eq!(self.num_rows, 0);
        for buffer in &mut self.integer_buffers {
            *buffer = None;
        }
    }

    pub(crate) fn encoding_plan(&self) -> BucketEncodingPlan {
        let (encodings, has_nulls) = self.compute_encodings();
        let page_sizes = (0..self.total_columns)
            .map(|col| {
                let nulls = if has_nulls[col] {
                    self.col_num_rows(col).div_ceil(8)
                } else {
                    0
                };
                match encodings[col] {
                    ENCODING_ALL_NULL => None,
                    ENCODING_CONST => has_nulls[col].then_some(nulls),
                    ENCODING_DICT => Some(nulls + self.dict_encoded_size(col)),
                    ENCODING_PLAIN => Some(nulls + self.logical_value_size(col)),
                    _ => unreachable!(),
                }
            })
            .collect();
        BucketEncodingPlan {
            encodings,
            has_nulls,
            page_sizes,
        }
    }

    pub(crate) fn page_size(&self, plan: &BucketEncodingPlan, col: usize) -> usize {
        if plan.encodings[col] == ENCODING_ALL_NULL {
            return 0;
        }
        2 + plan.page_sizes[col].unwrap_or(0)
            + if plan.encodings[col] == ENCODING_CONST {
                self.first_value_len[col]
            } else {
                0
            }
    }

    pub(crate) fn write_page(
        &self,
        plan: &BucketEncodingPlan,
        col: usize,
        out: &mut impl Write,
    ) -> io::Result<()> {
        let encoding = plan.encodings[col];
        if encoding == ENCODING_ALL_NULL {
            return Ok(());
        }
        out.write_all(&[encoding, u8::from(plan.has_nulls[col])])?;
        if encoding == ENCODING_CONST {
            self.write_const(col, out)?;
        }
        self.write_page_data(plan, col, out)
    }

    fn write_page_data(
        &self,
        plan: &BucketEncodingPlan,
        col: usize,
        out: &mut impl Write,
    ) -> io::Result<()> {
        let encoding = plan.encodings[col];
        if encoding == ENCODING_DICT {
            self.write_dictionary(col, out)?;
        }
        if plan.has_nulls[col] {
            self.write_nulls(col, out)?;
        }
        self.write_values(col, encoding, out)
    }

    pub fn finish_paged(&self) -> PagedBucketOutput {
        if self.num_rows == 0 {
            return PagedBucketOutput {
                encodings: Vec::new(),
                has_nulls: Vec::new(),
                const_data: Vec::new(),
                column_pages: Vec::new(),
                num_primary: self.num_primary,
                children: self.children.clone(),
            };
        }
        let plan = self.encoding_plan();
        let mut const_data = vec![Vec::new(); self.total_columns];
        let mut column_pages = vec![None; self.total_columns];
        for col in 0..self.total_columns {
            if plan.encodings[col] == ENCODING_CONST {
                self.write_const(col, &mut const_data[col])
                    .expect("Vec write");
            }
            if let Some(size) = plan.page_sizes[col] {
                let mut page = Vec::with_capacity(size);
                self.write_page_data(&plan, col, &mut page)
                    .expect("Vec write");
                column_pages[col] = Some(page);
            }
        }
        PagedBucketOutput {
            encodings: plan.encodings,
            has_nulls: plan.has_nulls,
            const_data,
            column_pages,
            num_primary: self.num_primary,
            children: self.children.clone(),
        }
    }

    fn write_const(&self, col: usize, out: &mut impl Write) -> io::Result<()> {
        if let Some(buffer) = &self.integer_buffers[col] {
            let key = buffer.keys().next().expect("non-null constant");
            out.write_all(&key.to_be_bytes()[8 - self.fixed_widths[col] as usize..])
        } else {
            out.write_all(&self.value_buffers[col][..self.first_value_len[col]])
        }
    }

    fn write_dictionary(&self, col: usize, out: &mut impl Write) -> io::Result<()> {
        if let Some(dict) = &self.long_dict_maps[col] {
            write_varint(out, dict.len() as u32)?;
            let mut keys = vec![0; dict.len()];
            for (&key, &index) in dict {
                keys[index] = key;
            }
            for key in keys {
                out.write_all(&key.to_be_bytes()[8 - self.fixed_widths[col] as usize..])?;
            }
        } else if let Some(dict) = &self.byte_dict_maps[col] {
            write_varint(out, dict.len() as u32)?;
            let mut keys: Vec<_> = dict.iter().collect();
            keys.sort_unstable_by_key(|(_, index)| **index);
            for (key, _) in keys {
                out.write_all(key)?;
            }
        }
        Ok(())
    }

    fn write_nulls(&self, col: usize, out: &mut impl Write) -> io::Result<()> {
        out.write_all(&self.null_bitmaps[col][..self.col_num_rows(col).div_ceil(8)])
    }

    fn write_values(&self, col: usize, encoding: u8, out: &mut impl Write) -> io::Result<()> {
        match encoding {
            ENCODING_PLAIN => match &self.integer_buffers[col] {
                Some(buffer) => buffer.write_to(out),
                None => out.write_all(&self.value_buffers[col]),
            },
            ENCODING_DICT => self.write_dict_indices(col, out),
            _ => Ok(()),
        }
    }

    fn write_dict_indices(&self, col: usize, out: &mut impl Write) -> io::Result<()> {
        let width = self.fixed_widths[col];
        let bits = bit_width(self.get_dict_size(col));
        let mut integer_keys = self.integer_buffers[col].as_ref().map(|b| b.keys());
        let mut pos = 0;
        let mut scratch = [0; 16 * 1024];
        let mut used = 0;
        let mut pending = 0u128;
        let mut pending_bits = 0;
        // Value buffers contain only non-null values, in row order.
        for _ in 0..self.non_null_counts[col] {
            let index = if let Some(dict) = &self.long_dict_maps[col] {
                let key = if let Some(keys) = &mut integer_keys {
                    keys.next().expect("non-null integer key")
                } else {
                    let key = values::extract_fixed_key(&self.value_buffers[col], pos, width);
                    pos += width as usize;
                    key
                };
                dict[&key]
            } else {
                let len = if width > 0 {
                    width as usize
                } else {
                    let len = varint::decode(&self.value_buffers[col], &mut pos.clone())
                        .expect("internal varint in value buffer");
                    varint::encoded_size(len) + len as usize
                };
                let key = &self.value_buffers[col][pos..pos + len];
                pos += len;
                self.byte_dict_maps[col].as_ref().unwrap()[key]
            };
            pending |= (index as u128) << pending_bits;
            pending_bits += bits;
            while pending_bits >= 8 {
                scratch[used] = pending as u8;
                used += 1;
                pending >>= 8;
                pending_bits -= 8;
                if used == scratch.len() {
                    out.write_all(&scratch)?;
                    used = 0;
                }
            }
        }
        if pending_bits > 0 {
            scratch[used] = pending as u8;
            used += 1;
        }
        out.write_all(&scratch[..used])
    }

    pub fn reset(&mut self) {
        for i in 0..self.total_columns {
            self.null_bitmaps[i].fill(0);
            self.value_buffers[i].clear();
            if let Some(buffer) = &mut self.integer_buffers[i] {
                buffer.clear();
            }
            self.non_null_counts[i] = 0;
            self.const_tracking[i] = true;
            self.first_value_len[i] = 0;
            self.dict_tracking[i] = DictTracking::Pending;
            self.dict_total_bytes[i] = 0;
            if uses_long_dict(self.fixed_widths[i]) {
                if let Some(ref mut dict) = self.long_dict_maps[i] {
                    dict.clear();
                }
            } else if let Some(ref mut dict) = self.byte_dict_maps[i] {
                dict.clear();
            }
        }
        self.num_rows = 0;
        for child in &mut self.children {
            child.num_elements = 0;
        }
    }

    fn get_dict_size(&self, col: usize) -> usize {
        if self.dict_tracking[col] != DictTracking::Active {
            return 0;
        }
        if let Some(ref dict) = self.long_dict_maps[col] {
            return dict.len();
        }
        if let Some(ref dict) = self.byte_dict_maps[col] {
            return dict.len();
        }
        0
    }

    fn dict_encoded_size(&self, col: usize) -> usize {
        if self.dict_tracking[col] != DictTracking::Active {
            return usize::MAX;
        }
        let (num_entries, entry_bytes) = if let Some(ref dict) = self.long_dict_maps[col] {
            (dict.len(), dict.len() * self.fixed_widths[col] as usize)
        } else if let Some(ref dict) = self.byte_dict_maps[col] {
            let bytes: usize = dict.keys().map(|k| k.len()).sum();
            (dict.len(), bytes)
        } else {
            return usize::MAX;
        };
        let index_bytes = (self.non_null_counts[col] * bit_width(num_entries)).div_ceil(8);
        varint::encoded_size(num_entries as u32) + entry_bytes + index_bytes
    }

    fn compute_out_size(&self, encodings: &[u8], has_nulls: &[bool]) -> usize {
        // Header: varint(num_primary) + varint(num_children) + varint per child
        let mut size = 0;
        if !self.children.is_empty() {
            size += varint::encoded_size(self.num_primary as u32)
                + varint::encoded_size(self.children.len() as u32);
            for child in &self.children {
                size += varint::encoded_size(child.num_elements as u32);
            }
        }

        size += (self.total_columns * 2).div_ceil(8) + self.total_columns.div_ceil(8);

        for i in 0..self.total_columns {
            if encodings[i] == ENCODING_ALL_NULL {
                continue;
            }
            if has_nulls[i] {
                size += self.col_num_rows(i).div_ceil(8);
            }
            match encodings[i] {
                ENCODING_CONST => {
                    size += self.first_value_len[i];
                }
                ENCODING_DICT => {
                    if let Some(ref dict) = self.long_dict_maps[i] {
                        let n = dict.len();
                        size += varint::encoded_size(n as u32) + n * self.fixed_widths[i] as usize;
                        size += (self.non_null_counts[i] * bit_width(n)).div_ceil(8);
                    } else if let Some(ref dict) = self.byte_dict_maps[i] {
                        let n = dict.len();
                        size += varint::encoded_size(n as u32);
                        size += dict.keys().map(|k| k.len()).sum::<usize>();
                        size += (self.non_null_counts[i] * bit_width(n)).div_ceil(8);
                    }
                }
                ENCODING_PLAIN => {
                    size += self.logical_value_size(i);
                }
                _ => {}
            }
        }
        size
    }
}

fn append_null_bitmap(bitmap: &mut [u8], start_row: usize, nulls: &NullBuffer) {
    debug_assert!(start_row + nulls.len() <= bitmap.len() * 8);

    let chunks = nulls.inner().bit_chunks();
    for (chunk, valid_bits) in chunks.iter().enumerate() {
        or_null_bits(bitmap, start_row + chunk * 64, !valid_bits, 64);
    }

    let remainder_len = nulls.len() % 64;
    if remainder_len != 0 {
        let remainder_mask = (1u64 << remainder_len) - 1;
        let null_bits = !chunks.remainder_bits() & remainder_mask;
        or_null_bits(
            bitmap,
            start_row + nulls.len() - remainder_len,
            null_bits,
            remainder_len,
        );
    }
}

#[inline]
fn or_null_bits(bitmap: &mut [u8], mut bit_offset: usize, mut bits: u64, mut len: usize) {
    while len != 0 {
        let byte_offset = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;
        let take = len.min(8 - bit_in_byte);
        let mask = ((1u16 << take) - 1) as u8;
        bitmap[byte_offset] |= ((bits as u8) & mask) << bit_in_byte;
        bits >>= take;
        bit_offset += take;
        len -= take;
    }
}

enum TypedArrayRef<'a> {
    Boolean(&'a BooleanArray),
    Int8(&'a Int8Array),
    Int16(&'a Int16Array),
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    Float32(&'a Float32Array),
    Float64(&'a Float64Array),
    Date32(&'a Date32Array),
    Time32(&'a Time32MillisecondArray),
    Utf8(&'a StringArray),
    Binary(&'a BinaryArray),
    Decimal128Compact(&'a Decimal128Array),
    Decimal128Large(&'a Decimal128Array),
    TimestampMillis(&'a TimestampMillisecondArray),
    TimestampMicros(&'a TimestampMicrosecondArray),
    TimestampNanos(&'a TimestampNanosecondArray),
    LegacyTimestampNanos {
        millis: &'a Int64Array,
        nanos: &'a Int32Array,
    },
}

fn cast_err(dt: &DataType) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("array downcast failed for DataType: {:?}", dt),
    )
}

fn downcast_array<'a>(array: &'a dyn Array, dt: &DataType) -> io::Result<TypedArrayRef<'a>> {
    let any = array.as_any();
    match dt {
        DataType::Boolean => Ok(TypedArrayRef::Boolean(
            any.downcast_ref::<BooleanArray>()
                .ok_or_else(|| cast_err(dt))?,
        )),
        DataType::Int8 => Ok(TypedArrayRef::Int8(
            any.downcast_ref::<Int8Array>()
                .ok_or_else(|| cast_err(dt))?,
        )),
        DataType::Int16 => Ok(TypedArrayRef::Int16(
            any.downcast_ref::<Int16Array>()
                .ok_or_else(|| cast_err(dt))?,
        )),
        DataType::Int32 => Ok(TypedArrayRef::Int32(
            any.downcast_ref::<Int32Array>()
                .ok_or_else(|| cast_err(dt))?,
        )),
        DataType::Int64 => Ok(TypedArrayRef::Int64(
            any.downcast_ref::<Int64Array>()
                .ok_or_else(|| cast_err(dt))?,
        )),
        DataType::Float32 => Ok(TypedArrayRef::Float32(
            any.downcast_ref::<Float32Array>()
                .ok_or_else(|| cast_err(dt))?,
        )),
        DataType::Float64 => Ok(TypedArrayRef::Float64(
            any.downcast_ref::<Float64Array>()
                .ok_or_else(|| cast_err(dt))?,
        )),
        DataType::Date32 => Ok(TypedArrayRef::Date32(
            any.downcast_ref::<Date32Array>()
                .ok_or_else(|| cast_err(dt))?,
        )),
        DataType::Time32(_) => Ok(TypedArrayRef::Time32(
            any.downcast_ref::<Time32MillisecondArray>()
                .ok_or_else(|| cast_err(dt))?,
        )),
        DataType::Utf8 => Ok(TypedArrayRef::Utf8(
            any.downcast_ref::<StringArray>()
                .ok_or_else(|| cast_err(dt))?,
        )),
        DataType::Binary => Ok(TypedArrayRef::Binary(
            any.downcast_ref::<BinaryArray>()
                .ok_or_else(|| cast_err(dt))?,
        )),
        DataType::Decimal128(p, _) if *p <= 18 => Ok(TypedArrayRef::Decimal128Compact(
            any.downcast_ref::<Decimal128Array>()
                .ok_or_else(|| cast_err(dt))?,
        )),
        DataType::Decimal128(_, _) => Ok(TypedArrayRef::Decimal128Large(
            any.downcast_ref::<Decimal128Array>()
                .ok_or_else(|| cast_err(dt))?,
        )),
        DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => {
            Ok(TypedArrayRef::TimestampMillis(
                any.downcast_ref::<TimestampMillisecondArray>()
                    .ok_or_else(|| cast_err(dt))?,
            ))
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
            Ok(TypedArrayRef::TimestampMicros(
                any.downcast_ref::<TimestampMicrosecondArray>()
                    .ok_or_else(|| cast_err(dt))?,
            ))
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => {
            Ok(TypedArrayRef::TimestampNanos(
                any.downcast_ref::<TimestampNanosecondArray>()
                    .ok_or_else(|| cast_err(dt))?,
            ))
        }
        DataType::Struct(fields) if types::is_timestamp_nanos_struct(fields) => {
            let s = any
                .downcast_ref::<StructArray>()
                .ok_or_else(|| cast_err(dt))?;
            let ts_dt = DataType::Int64;
            let ns_dt = DataType::Int32;
            let millis = s
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| cast_err(&ts_dt))?;
            let nanos = s
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| cast_err(&ns_dt))?;
            validate_legacy_timestamp_nanos(s, millis, nanos)?;
            Ok(TypedArrayRef::LegacyTimestampNanos { millis, nanos })
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported DataType: {:?}", dt),
        )),
    }
}

#[inline]
fn write_typed_value(buf: &mut Vec<u8>, typed: &TypedArrayRef, row: usize) -> io::Result<()> {
    match typed {
        TypedArrayRef::Boolean(a) => buf.push(if a.value(row) { 1 } else { 0 }),
        TypedArrayRef::Int8(a) => buf.push(a.value(row) as u8),
        TypedArrayRef::Int16(a) => buf.extend_from_slice(&a.value(row).to_be_bytes()),
        TypedArrayRef::Int32(a) => buf.extend_from_slice(&a.value(row).to_be_bytes()),
        TypedArrayRef::Int64(a) => buf.extend_from_slice(&a.value(row).to_be_bytes()),
        TypedArrayRef::Float32(a) => buf.extend_from_slice(&a.value(row).to_bits().to_be_bytes()),
        TypedArrayRef::Float64(a) => buf.extend_from_slice(&a.value(row).to_bits().to_be_bytes()),
        TypedArrayRef::Date32(a) => buf.extend_from_slice(&a.value(row).to_be_bytes()),
        TypedArrayRef::Time32(a) => buf.extend_from_slice(&a.value(row).to_be_bytes()),
        TypedArrayRef::Utf8(a) => {
            let bytes = a.value(row).as_bytes();
            varint::encode(buf, bytes.len() as u32);
            buf.extend_from_slice(bytes);
        }
        TypedArrayRef::Binary(a) => {
            let bytes = a.value(row);
            varint::encode(buf, bytes.len() as u32);
            buf.extend_from_slice(bytes);
        }
        TypedArrayRef::Decimal128Compact(a) => {
            buf.extend_from_slice(&(a.value(row) as i64).to_be_bytes())
        }
        TypedArrayRef::Decimal128Large(a) => {
            let bytes = i128_to_biginteger_bytes(a.value(row));
            varint::encode(buf, bytes.len() as u32);
            buf.extend_from_slice(&bytes);
        }
        TypedArrayRef::TimestampMillis(a) => buf.extend_from_slice(&a.value(row).to_be_bytes()),
        TypedArrayRef::TimestampMicros(a) => buf.extend_from_slice(&a.value(row).to_be_bytes()),
        TypedArrayRef::TimestampNanos(a) => write_timestamp_nanos_value(buf, a.value(row)),
        TypedArrayRef::LegacyTimestampNanos { millis, nanos } => {
            write_legacy_timestamp_nanos_value(buf, millis.value(row), nanos.value(row))?;
        }
    }
    Ok(())
}

fn validate_legacy_timestamp_nanos(
    parent: &StructArray,
    millis: &Int64Array,
    nanos: &Int32Array,
) -> io::Result<()> {
    for row in 0..parent.len() {
        if parent.is_null(row) {
            continue;
        }
        if millis.is_null(row) || nanos.is_null(row) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "legacy timestamp nanos has null child for a non-null parent row",
            ));
        }
        validate_timestamp_nanos_pair(millis.value(row), nanos.value(row))?;
    }
    Ok(())
}

#[inline]
fn write_timestamp_nanos_value(buf: &mut Vec<u8>, ns: i64) {
    let (millis, nanos) = types::ns_to_millis_nanos(ns);
    buf.extend_from_slice(&millis.to_be_bytes());
    buf.extend_from_slice(&nanos.to_be_bytes());
}

#[inline]
fn write_legacy_timestamp_nanos_value(
    buf: &mut Vec<u8>,
    millis: i64,
    nanos: i32,
) -> io::Result<()> {
    validate_timestamp_nanos_pair(millis, nanos)?;
    buf.extend_from_slice(&millis.to_be_bytes());
    buf.extend_from_slice(&nanos.to_be_bytes());
    Ok(())
}

fn validate_timestamp_nanos_pair(millis: i64, nanos: i32) -> io::Result<()> {
    types::millis_nanos_to_ns(millis, nanos)
        .map(|_| ())
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err.to_string()))
}

fn i128_to_biginteger_bytes(val: i128) -> Vec<u8> {
    let bytes = val.to_be_bytes();
    let negative = val < 0;
    let pad = if negative { 0xFF } else { 0x00 };
    let mut start = 0;
    while start < 15 {
        if bytes[start] != pad {
            break;
        }
        if (bytes[start + 1] & 0x80 != 0) != negative {
            break;
        }
        start += 1;
    }
    bytes[start..].to_vec()
}

fn extract_map_lengths(map_array: &MapArray) -> Int32Array {
    let offsets = map_array.value_offsets();
    let num_rows = map_array.len();
    let mut lengths = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        if map_array.is_null(i) {
            lengths.push(0);
        } else {
            lengths.push(offsets[i + 1] - offsets[i]);
        }
    }
    let null_buf = map_array.nulls().cloned();
    Int32Array::new(ScalarBuffer::from(lengths), null_buf)
}

fn flatten_map_entries(map_array: &MapArray) -> (ArrayRef, ArrayRef) {
    let offsets = map_array.value_offsets();
    let num_rows = map_array.len();
    let keys = map_array.keys();
    let values = map_array.values();

    if map_array.null_count() == 0 {
        let start = offsets[0] as usize;
        let end = offsets[num_rows] as usize;
        return (
            keys.slice(start, end - start),
            values.slice(start, end - start),
        );
    }

    let mut indices: Vec<u32> = Vec::new();
    for i in 0..num_rows {
        if !map_array.is_null(i) {
            let start = offsets[i] as u32;
            let end = offsets[i + 1] as u32;
            for idx in start..end {
                indices.push(idx);
            }
        }
    }

    if indices.is_empty() {
        return (keys.slice(0, 0), values.slice(0, 0));
    }

    let idx_array = UInt32Array::from(indices);
    (
        take_array(keys.as_ref(), &idx_array),
        take_array(values.as_ref(), &idx_array),
    )
}

fn extract_list_lengths(list_array: &ListArray) -> Int32Array {
    let offsets = list_array.value_offsets();
    let num_rows = list_array.len();
    let mut lengths = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        if list_array.is_null(i) {
            lengths.push(0);
        } else {
            lengths.push(offsets[i + 1] - offsets[i]);
        }
    }
    let null_buf = list_array.nulls().cloned();
    Int32Array::new(ScalarBuffer::from(lengths), null_buf)
}

fn flatten_list_values(list_array: &ListArray) -> ArrayRef {
    let offsets = list_array.value_offsets();
    let values = list_array.values();
    let num_rows = list_array.len();

    if list_array.null_count() == 0 {
        let start = offsets[0] as usize;
        let end = offsets[num_rows] as usize;
        return values.slice(start, end - start);
    }

    // Skip child values for null rows — collect only non-null row ranges
    let mut indices: Vec<u32> = Vec::new();
    for i in 0..num_rows {
        if !list_array.is_null(i) {
            let start = offsets[i] as u32;
            let end = offsets[i + 1] as u32;
            for idx in start..end {
                indices.push(idx);
            }
        }
    }

    if indices.is_empty() {
        return values.slice(0, 0);
    }

    let idx_array = UInt32Array::from(indices);
    take_array(values.as_ref(), &idx_array)
}

fn take_array(array: &dyn Array, indices: &UInt32Array) -> ArrayRef {
    use arrow_array::builder::*;
    macro_rules! take_prim {
        ($arr_ty:ty, $bld_ty:ty) => {{
            let src = array.as_any().downcast_ref::<$arr_ty>().unwrap();
            let mut b = <$bld_ty>::with_capacity(indices.len());
            for i in 0..indices.len() {
                let idx = indices.value(i) as usize;
                if src.is_null(idx) {
                    b.append_null();
                } else {
                    b.append_value(src.value(idx));
                }
            }
            Arc::new(b.finish()) as ArrayRef
        }};
    }
    match array.data_type() {
        DataType::Boolean => take_prim!(BooleanArray, BooleanBuilder),
        DataType::Int8 => take_prim!(Int8Array, Int8Builder),
        DataType::Int16 => take_prim!(Int16Array, Int16Builder),
        DataType::Int32 => take_prim!(Int32Array, Int32Builder),
        DataType::Int64 => take_prim!(Int64Array, Int64Builder),
        DataType::Float32 => take_prim!(Float32Array, Float32Builder),
        DataType::Float64 => take_prim!(Float64Array, Float64Builder),
        DataType::Date32 => take_prim!(Date32Array, Date32Builder),
        DataType::Time32(_) => take_prim!(Time32MillisecondArray, Time32MillisecondBuilder),
        DataType::Decimal128(p, s) => {
            let src = array.as_any().downcast_ref::<Decimal128Array>().unwrap();
            let mut b = Decimal128Builder::new()
                .with_precision_and_scale(*p, *s)
                .unwrap();
            for i in 0..indices.len() {
                let idx = indices.value(i) as usize;
                if src.is_null(idx) {
                    b.append_null();
                } else {
                    b.append_value(src.value(idx));
                }
            }
            Arc::new(b.finish()) as ArrayRef
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, tz) => {
            let src = array
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .unwrap();
            let mut b = TimestampMillisecondBuilder::new();
            for i in 0..indices.len() {
                let idx = indices.value(i) as usize;
                if src.is_null(idx) {
                    b.append_null();
                } else {
                    b.append_value(src.value(idx));
                }
            }
            let arr = b.finish();
            Arc::new(if let Some(tz) = tz {
                arr.with_timezone(tz.clone())
            } else {
                arr
            })
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, tz) => {
            let src = array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let mut b = TimestampMicrosecondBuilder::new();
            for i in 0..indices.len() {
                let idx = indices.value(i) as usize;
                if src.is_null(idx) {
                    b.append_null();
                } else {
                    b.append_value(src.value(idx));
                }
            }
            let arr = b.finish();
            Arc::new(if let Some(tz) = tz {
                arr.with_timezone(tz.clone())
            } else {
                arr
            })
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, tz) => {
            let src = array
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap();
            let mut b = TimestampNanosecondBuilder::new();
            for i in 0..indices.len() {
                let idx = indices.value(i) as usize;
                if src.is_null(idx) {
                    b.append_null();
                } else {
                    b.append_value(src.value(idx));
                }
            }
            let arr = b.finish();
            Arc::new(if let Some(tz) = tz {
                arr.with_timezone(tz.clone())
            } else {
                arr
            })
        }
        DataType::Utf8 => {
            let src = array.as_any().downcast_ref::<StringArray>().unwrap();
            let mut b = StringBuilder::new();
            for i in 0..indices.len() {
                let idx = indices.value(i) as usize;
                if src.is_null(idx) {
                    b.append_null();
                } else {
                    b.append_value(src.value(idx));
                }
            }
            Arc::new(b.finish())
        }
        DataType::Binary => {
            let src = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            let mut b = BinaryBuilder::new();
            for i in 0..indices.len() {
                let idx = indices.value(i) as usize;
                if src.is_null(idx) {
                    b.append_null();
                } else {
                    b.append_value(src.value(idx));
                }
            }
            Arc::new(b.finish())
        }
        DataType::List(_) => {
            // For nested arrays, rebuild by collecting slices
            let src = array.as_any().downcast_ref::<ListArray>().unwrap();
            let mut offsets_builder = vec![0i32];
            let mut child_indices: Vec<u32> = Vec::new();
            for i in 0..indices.len() {
                let idx = indices.value(i) as usize;
                let start = src.value_offsets()[idx] as u32;
                let end = src.value_offsets()[idx + 1] as u32;
                for ci in start..end {
                    child_indices.push(ci);
                }
                offsets_builder.push(child_indices.len() as i32);
            }
            let child_idx_arr = UInt32Array::from(child_indices);
            let new_values = take_array(src.values().as_ref(), &child_idx_arr);
            let field = match array.data_type() {
                DataType::List(f) => f.clone(),
                _ => unreachable!(),
            };
            let null_buf = if !indices.is_empty() {
                let mut bm = vec![0u8; indices.len().div_ceil(8)];
                for i in 0..indices.len() {
                    let idx = indices.value(i) as usize;
                    if !src.is_null(idx) {
                        bm[i / 8] |= 1 << (i % 8);
                    }
                }
                if bm.iter().all(|&b| b == 0xFF) || indices.is_empty() {
                    None
                } else {
                    Some(NullBuffer::new(BooleanBuffer::new(
                        Buffer::from_vec(bm),
                        0,
                        indices.len(),
                    )))
                }
            } else {
                None
            };
            Arc::new(ListArray::new(
                field,
                OffsetBuffer::new(ScalarBuffer::from(offsets_builder)),
                new_values,
                null_buf,
            ))
        }
        DataType::Map(entries_field, sorted) => {
            let src = array.as_any().downcast_ref::<MapArray>().unwrap();
            let mut offsets_builder = vec![0i32];
            let mut child_indices: Vec<u32> = Vec::new();
            for i in 0..indices.len() {
                let idx = indices.value(i) as usize;
                let start = src.value_offsets()[idx] as u32;
                let end = src.value_offsets()[idx + 1] as u32;
                for ci in start..end {
                    child_indices.push(ci);
                }
                offsets_builder.push(child_indices.len() as i32);
            }
            let child_idx_arr = UInt32Array::from(child_indices);
            let new_keys = take_array(src.keys().as_ref(), &child_idx_arr);
            let new_values = take_array(src.values().as_ref(), &child_idx_arr);
            let null_buf = if !indices.is_empty() {
                let mut bm = vec![0u8; indices.len().div_ceil(8)];
                for i in 0..indices.len() {
                    let idx = indices.value(i) as usize;
                    if !src.is_null(idx) {
                        bm[i / 8] |= 1 << (i % 8);
                    }
                }
                if bm.iter().all(|&b| b == 0xFF) || indices.is_empty() {
                    None
                } else {
                    Some(NullBuffer::new(BooleanBuffer::new(
                        Buffer::from_vec(bm),
                        0,
                        indices.len(),
                    )))
                }
            } else {
                None
            };
            let entries_struct = StructArray::new(
                match entries_field.data_type() {
                    DataType::Struct(fields) => fields.clone(),
                    _ => unreachable!(),
                },
                vec![new_keys, new_values],
                None,
            );
            Arc::new(MapArray::new(
                entries_field.clone(),
                OffsetBuffer::new(ScalarBuffer::from(offsets_builder)),
                entries_struct,
                null_buf,
                *sorted,
            ))
        }
        other => panic!("take_array: unsupported DataType {:?}", other),
    }
}

pub(crate) fn expand_col_types(col_types: &[&DataType]) -> (Vec<DataType>, Vec<ChildColumnMeta>) {
    let mut physical_types: Vec<DataType> = col_types
        .iter()
        .map(|t| {
            if matches!(t, DataType::List(_) | DataType::Map(_, _)) {
                DataType::Int32
            } else {
                (*t).clone()
            }
        })
        .collect();
    let mut children = Vec::new();

    for (i, t) in col_types.iter().enumerate() {
        expand_container(i, i, t, &mut physical_types, &mut children);
    }
    (physical_types, children)
}

fn expand_container(
    parent_logical: usize,
    length_physical_index: usize,
    dt: &DataType,
    physical_types: &mut Vec<DataType>,
    children: &mut Vec<ChildColumnMeta>,
) {
    match dt {
        DataType::List(element_field) => {
            expand_element(
                parent_logical,
                length_physical_index,
                ChildColumnRole::ListElement,
                element_field,
                physical_types,
                children,
            );
        }
        DataType::Map(entries_field, _) => {
            if let DataType::Struct(fields) = entries_field.data_type() {
                expand_element(
                    parent_logical,
                    length_physical_index,
                    ChildColumnRole::MapKey,
                    &fields[0],
                    physical_types,
                    children,
                );
                expand_element(
                    parent_logical,
                    length_physical_index,
                    ChildColumnRole::MapValue,
                    &fields[1],
                    physical_types,
                    children,
                );
            }
        }
        _ => {}
    }
}

fn expand_element(
    parent_logical: usize,
    length_physical_index: usize,
    role: ChildColumnRole,
    element_field: &Arc<Field>,
    physical_types: &mut Vec<DataType>,
    children: &mut Vec<ChildColumnMeta>,
) {
    let elem_dt = element_field.data_type();
    let child_phys_idx = physical_types.len();

    match elem_dt {
        DataType::List(_) | DataType::Map(_, _) => {
            // Complex element: this child stores lengths (INT32), recurse for deeper levels
            physical_types.push(DataType::Int32);
            children.push(ChildColumnMeta {
                parent_logical_col: parent_logical,
                physical_index: child_phys_idx,
                length_physical_index,
                role,
                element_field: element_field.clone(),
                num_elements: 0,
            });
            expand_container(
                parent_logical,
                child_phys_idx,
                elem_dt,
                physical_types,
                children,
            );
        }
        _ => {
            // Primitive element: direct leaf column
            physical_types.push(elem_dt.clone());
            children.push(ChildColumnMeta {
                parent_logical_col: parent_logical,
                physical_index: child_phys_idx,
                length_physical_index,
                role,
                element_field: element_field.clone(),
                num_elements: 0,
            });
        }
    }
}

fn uses_long_dict(fixed_width: i32) -> bool {
    fixed_width > 0 && fixed_width <= 8
}

fn bit_width(num_entries: usize) -> usize {
    if num_entries <= 1 {
        return 0;
    }
    usize::BITS as usize - (num_entries - 1).leading_zeros() as usize
}

fn equals_first_value(buf: &[u8], offset: usize, len: usize) -> bool {
    buf[..len] == buf[offset..offset + len]
}

fn write_varint(out: &mut impl Write, mut value: u32) -> io::Result<()> {
    let mut bytes = [0; 5];
    let mut len = 0;
    while value >= 128 {
        bytes[len] = (value as u8) | 0x80;
        value >>= 7;
        len += 1;
    }
    bytes[len] = value as u8;
    out.write_all(&bytes[..len + 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header_size(_data: &[u8]) -> usize {
        // No header for non-ARRAY buckets (v1 compatible)
        0
    }

    #[test]
    fn test_expand_col_types_records_explicit_child_layout() {
        let map_type = DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(arrow_schema::Fields::from(vec![
                    Field::new("keys", DataType::Int32, false),
                    Field::new(
                        "values",
                        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                        true,
                    ),
                ])),
                false,
            )),
            false,
        );
        let list_type = DataType::List(Arc::new(Field::new("item", map_type, true)));
        let col_refs = vec![&list_type];

        let (physical_types, children) = expand_col_types(&col_refs);

        assert_eq!(
            physical_types,
            vec![
                DataType::Int32,
                DataType::Int32,
                DataType::Int32,
                DataType::Int32,
                DataType::Utf8,
            ]
        );
        assert_eq!(children.len(), 4);
        assert_eq!(children[0].role, ChildColumnRole::ListElement);
        assert_eq!(children[0].physical_index, 1);
        assert_eq!(children[0].length_physical_index, 0);
        assert_eq!(children[1].role, ChildColumnRole::MapKey);
        assert_eq!(children[1].physical_index, 2);
        assert_eq!(children[1].length_physical_index, 1);
        assert_eq!(children[2].role, ChildColumnRole::MapValue);
        assert_eq!(children[2].physical_index, 3);
        assert_eq!(children[2].length_physical_index, 1);
        assert_eq!(children[3].role, ChildColumnRole::ListElement);
        assert_eq!(children[3].physical_index, 4);
        assert_eq!(children[3].length_physical_index, 3);
    }

    #[test]
    fn test_all_null_encoding() {
        let types = [DataType::Int32];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let arr = Int32Array::new_null(10);
        writer.write_columns(&[&arr], &[&DataType::Int32]).unwrap();

        let data = writer.finish();
        assert!(!data.is_empty());
        let h = header_size(&data);
        assert_eq!(data[h] & 0x03, ENCODING_ALL_NULL);
    }

    #[test]
    fn test_const_encoding() {
        let types = [DataType::Int32];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let arr = Int32Array::from(vec![42; 10]);
        writer.write_columns(&[&arr], &[&DataType::Int32]).unwrap();

        let data = writer.finish();
        let h = header_size(&data);
        assert_eq!(data[h] & 0x03, ENCODING_CONST);
    }

    #[test]
    fn test_dict_encoding() {
        let types = [DataType::Int32];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let vals: Vec<i32> = (0..100).map(|i| i % 3).collect();
        let arr = Int32Array::from(vals);
        writer.write_columns(&[&arr], &[&DataType::Int32]).unwrap();

        let data = writer.finish();
        let h = header_size(&data);
        assert_eq!(data[h] & 0x03, ENCODING_DICT);
    }

    #[test]
    fn test_plain_encoding() {
        let types = [DataType::Int32];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let vals: Vec<i32> = (0..1000).collect();
        let arr = Int32Array::from(vals);
        writer.write_columns(&[&arr], &[&DataType::Int32]).unwrap();

        let data = writer.finish();
        let h = header_size(&data);
        assert_eq!(data[h] & 0x03, ENCODING_PLAIN);
    }

    #[test]
    fn test_const_string_encoding() {
        let types = [DataType::Utf8];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let arr = StringArray::from(vec!["same"; 50]);
        writer.write_columns(&[&arr], &[&DataType::Utf8]).unwrap();

        let data = writer.finish();
        let h = header_size(&data);
        assert_eq!(data[h] & 0x03, ENCODING_CONST);
    }

    #[test]
    fn test_all_null_wide_columns_do_not_preallocate_value_payloads() {
        let types = vec![&DataType::Float64; 10_000];
        let mut writer = BucketWriter::new(&types, 32768, 255);
        let nulls = Float64Array::new_null(65);
        let arrays = vec![&nulls as &dyn Array; types.len()];
        for _ in 0..2 {
            writer.write_columns(&arrays, &types).unwrap();
            let allocated: usize = writer.value_buffers.iter().map(Vec::capacity).sum();
            assert!(allocated <= 1024, "all-null payload capacity: {allocated}");
            writer.reset();
        }
    }

    fn compact_integer_test_array(dt: &DataType, values: &[Option<i64>]) -> ArrayRef {
        match dt {
            DataType::Int16 => Arc::new(Int16Array::from(
                values
                    .iter()
                    .map(|v| v.map(|v| v as i16))
                    .collect::<Vec<_>>(),
            )),
            DataType::Int32 => Arc::new(Int32Array::from(
                values
                    .iter()
                    .map(|v| v.map(|v| v as i32))
                    .collect::<Vec<_>>(),
            )),
            DataType::Int64 => Arc::new(Int64Array::from(values.to_vec())),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_compact_integer_lazy_dictionary_after_nulls_promotion_and_reset() {
        for dt in [DataType::Int16, DataType::Int32, DataType::Int64] {
            let mut compact = BucketWriter::new(&[&dt], 32768, 2);
            let mut wide = BucketWriter::new(&[&dt], 32768, 2);
            wide.use_wide_integer_buffers();
            let append =
                |compact: &mut BucketWriter, wide: &mut BucketWriter, values: &[Option<i64>]| {
                    // Slice the source and append at non-byte-aligned destination offsets.
                    let mut padded = vec![Some(99)];
                    padded.extend_from_slice(values);
                    let array = compact_integer_test_array(&dt, &padded).slice(1, values.len());
                    assert_eq!(
                        compact.write_columns(&[array.as_ref()], &[&dt]).unwrap(),
                        wide.write_columns(&[array.as_ref()], &[&dt]).unwrap()
                    );
                    assert_eq!(compact.estimated_raw_size(), wide.estimated_raw_size());
                    assert_eq!(compact.finish(), wide.finish());
                    assert_eq!(
                        compact.finish_paged().column_pages,
                        wide.finish_paged().column_pages
                    );
                };
            append(&mut compact, &mut wide, &[None; 65]);
            append(&mut compact, &mut wide, &vec![Some(255); 4095]);
            assert_eq!(compact.dict_tracking[0], DictTracking::Pending);
            assert!(compact.long_dict_maps[0].is_none());
            append(
                &mut compact,
                &mut wide,
                &[None, Some(255), Some(-1), None, Some(-1)],
            );
            assert_eq!(compact.dict_tracking[0], DictTracking::Active);
            assert_eq!(compact.long_dict_maps[0].as_ref().unwrap().len(), 2);
            assert_eq!(compact.compute_encodings().0, vec![ENCODING_DICT]);

            compact.reset();
            wide.reset();
            append(&mut compact, &mut wide, &vec![Some(7); 100]);
            assert_eq!(compact.dict_tracking[0], DictTracking::Pending);
            assert!(compact.long_dict_maps[0].as_ref().unwrap().is_empty());
            append(&mut compact, &mut wide, &[Some(8), None, Some(7)]);
            let dict = compact.long_dict_maps[0].as_ref().unwrap();
            assert_eq!(dict.len(), 2);
            assert_eq!(dict[&7], 0);
            assert_eq!(dict[&8], 1);
        }
    }

    #[test]
    fn test_compact_integer_dictionary_limit_stays_disabled_until_reset() {
        for limit in [0, 1, 2] {
            let dt = DataType::Int64;
            let mut compact = BucketWriter::new(&[&dt], 32768, limit);
            let prefix = Int64Array::from(vec![-1; 100]);
            compact.write_columns(&[&prefix], &[&dt]).unwrap();
            assert!(compact.long_dict_maps[0].is_none());
            let varied = Int64Array::from(vec![Some(255), None, Some(256), Some(-1)]);
            compact.write_columns(&[&varied], &[&dt]).unwrap();
            assert_eq!(compact.dict_tracking[0], DictTracking::Disabled);
            assert!(compact.long_dict_maps[0].is_none());
            compact.write_columns(&[&prefix], &[&dt]).unwrap();
            assert_eq!(compact.dict_tracking[0], DictTracking::Disabled);
            assert_eq!(compact.compute_encodings().0, vec![ENCODING_PLAIN]);
            compact.reset();
            compact.write_columns(&[&prefix], &[&dt]).unwrap();
            assert_eq!(compact.dict_tracking[0], DictTracking::Pending);
            assert_eq!(compact.compute_encodings().0, vec![ENCODING_CONST]);
        }
    }

    #[test]
    fn test_const_columns_do_not_build_dictionaries() {
        let types = [DataType::Int32, DataType::Utf8];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let ints = Int32Array::from(vec![42; 100]);
        let strings = StringArray::from(vec!["same"; 100]);
        writer
            .write_columns(&[&ints, &strings], &[&types[0], &types[1]])
            .unwrap();

        assert!(writer.long_dict_maps[0].is_none());
        assert!(writer.byte_dict_maps[1].is_none());
        assert_eq!(writer.dict_total_bytes[1], 0);
    }

    #[test]
    fn test_dictionary_tracking_starts_on_first_non_const_value() {
        let types = [DataType::Int32, DataType::Utf8];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let const_ints = Int32Array::from(vec![42; 50]);
        let const_strings = StringArray::from(vec!["same"; 50]);
        writer
            .write_columns(&[&const_ints, &const_strings], &[&types[0], &types[1]])
            .unwrap();

        let varied_ints = Int32Array::from(vec![42, 7, 42, 7]);
        let varied_strings = StringArray::from(vec!["same", "other", "same", "other"]);
        writer
            .write_columns(&[&varied_ints, &varied_strings], &[&types[0], &types[1]])
            .unwrap();

        assert_eq!(writer.dict_tracking, vec![DictTracking::Active; 2]);
        assert_eq!(writer.long_dict_maps[0].as_ref().unwrap().len(), 2);
        assert_eq!(writer.byte_dict_maps[1].as_ref().unwrap().len(), 2);

        let data = writer.finish();
        assert_eq!(data[0] & 0x03, ENCODING_DICT);
        assert_eq!((data[0] >> 2) & 0x03, ENCODING_DICT);
    }

    #[test]
    fn test_dictionary_tracking_disables_after_entry_limit() {
        let types = [DataType::Int32, DataType::Utf8];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let ints = Int32Array::from_iter_values((0..255).chain(0..255));
        let string_values: Vec<String> = (0..255)
            .chain(0..255)
            .map(|i| format!("value-{i:03}"))
            .collect();
        let strings = StringArray::from_iter_values(string_values.iter().map(String::as_str));
        writer
            .write_columns(&[&ints, &strings], &[&types[0], &types[1]])
            .unwrap();

        assert_eq!(writer.dict_tracking, vec![DictTracking::Active; 2]);
        assert_eq!(writer.long_dict_maps[0].as_ref().unwrap().len(), 255);
        assert_eq!(writer.byte_dict_maps[1].as_ref().unwrap().len(), 255);
        let data = writer.finish();
        assert_eq!(data[0] & 0x03, ENCODING_DICT);
        assert_eq!((data[0] >> 2) & 0x03, ENCODING_DICT);

        let next_int = Int32Array::from(vec![255]);
        let next_string = StringArray::from(vec!["value-255"]);
        writer
            .write_columns(&[&next_int, &next_string], &[&types[0], &types[1]])
            .unwrap();

        assert_eq!(writer.dict_tracking, vec![DictTracking::Disabled; 2]);
        assert!(writer.long_dict_maps.iter().all(Option::is_none));
        assert!(writer.byte_dict_maps.iter().all(Option::is_none));

        let later_ints = Int32Array::from(vec![0, 256]);
        let later_strings = StringArray::from(vec!["value-000", "value-256"]);
        writer
            .write_columns(&[&later_ints, &later_strings], &[&types[0], &types[1]])
            .unwrap();

        assert_eq!(writer.dict_tracking, vec![DictTracking::Disabled; 2]);
        assert!(writer.long_dict_maps.iter().all(Option::is_none));
        assert!(writer.byte_dict_maps.iter().all(Option::is_none));
        let data = writer.finish();
        assert_eq!(data[0] & 0x03, ENCODING_PLAIN);
        assert_eq!((data[0] >> 2) & 0x03, ENCODING_PLAIN);
    }

    #[test]
    fn test_dictionary_tracking_disables_after_byte_limit() {
        let types = [DataType::Utf8];
        let type_refs: Vec<&DataType> = types.iter().collect();
        // Encoded dictionary keys include the one-byte string length prefix:
        // "a" is 2 bytes and "bb" is 3 bytes, exactly filling this budget.
        let mut writer = BucketWriter::new(&type_refs, 5, 255);

        // Repeat enough rows so DICT is smaller than PLAIN and the encoding assertion is stable.
        let mut within_budget: Vec<Option<&str>> = (0..64)
            .map(|i| Some(if i % 2 == 0 { "a" } else { "bb" }))
            .collect();
        within_budget.push(None);
        let within_budget = StringArray::from(within_budget);
        writer
            .write_columns(&[&within_budget], &[&DataType::Utf8])
            .unwrap();

        assert_eq!(writer.dict_tracking[0], DictTracking::Active);
        assert_eq!(writer.byte_dict_maps[0].as_ref().unwrap().len(), 2);
        assert_eq!(writer.dict_total_bytes[0], 5);
        let data = writer.finish();
        assert_eq!(data[0] & 0x03, ENCODING_DICT);

        let beyond_budget = StringArray::from(vec![Some("a"), Some("ccc"), None]);
        writer
            .write_columns(&[&beyond_budget], &[&DataType::Utf8])
            .unwrap();

        assert_eq!(writer.dict_tracking[0], DictTracking::Disabled);
        assert!(writer.byte_dict_maps[0].is_none());
        let data = writer.finish();
        assert_eq!(data[0] & 0x03, ENCODING_PLAIN);
    }

    #[test]
    fn test_append_null_bitmap_handles_source_and_destination_offsets() {
        let validity = BooleanBuffer::from(vec![
            true, false, true, false, false, true, true, false, true, false, true,
        ]);
        let nulls = NullBuffer::new(validity).slice(2, 7);
        let mut bitmap = vec![0b0000_0101, 0, 0];

        append_null_bitmap(&mut bitmap, 3, &nulls);

        assert_eq!(bitmap, vec![0b0011_0101, 0b0000_0001, 0]);
    }

    #[test]
    fn test_append_null_bitmap_matches_scalar_copy_across_chunk_boundaries() {
        for source_offset in 0..8 {
            for destination_offset in 0..16 {
                for len in [1, 7, 8, 9, 63, 64, 65, 127, 128, 129] {
                    let validity: Vec<bool> = (0..source_offset + len)
                        .map(|i| i % 3 != 0 && i % 11 != 0)
                        .collect();
                    let nulls = NullBuffer::new(BooleanBuffer::from(validity.clone()))
                        .slice(source_offset, len);
                    let mut actual = vec![0u8; (destination_offset + len).div_ceil(8) + 1];
                    let mut expected = actual.clone();

                    for row in 0..len {
                        if !validity[source_offset + row] {
                            let bit = destination_offset + row;
                            expected[bit / 8] |= 1 << (bit % 8);
                        }
                    }
                    append_null_bitmap(&mut actual, destination_offset, &nulls);

                    assert_eq!(
                        actual, expected,
                        "source_offset={source_offset}, destination_offset={destination_offset}, len={len}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_nullable_timestamp_millis_batch_preserves_slices_and_append_offsets() {
        let data_type = DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, None);
        let mut writer = BucketWriter::new(&[&data_type], 32768, 255);

        let source = TimestampMillisecondArray::from(vec![
            Some(99),
            Some(10),
            None,
            Some(10),
            Some(20),
            None,
        ]);
        let first = source.slice(1, 4);
        let first_size = writer.write_columns(&[&first], &[&data_type]).unwrap();
        assert_eq!(first_size, 28);

        let second = TimestampMillisecondArray::from(vec![None, Some(20), None]);
        let second_size = writer.write_columns(&[&second], &[&data_type]).unwrap();
        assert_eq!(second_size, 11);

        assert_eq!(writer.num_rows, 7);
        assert_eq!(writer.non_null_counts[0], 4);
        assert_eq!(writer.null_bitmaps[0][0], 0b0101_0010);
        assert_eq!(
            writer.value_buffers[0],
            [10_i64, 10, 20, 20]
                .into_iter()
                .flat_map(i64::to_be_bytes)
                .collect::<Vec<_>>()
        );
        assert!(!writer.const_tracking[0]);
        assert_eq!(writer.dict_tracking[0], DictTracking::Active);
        assert_eq!(writer.long_dict_maps[0].as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_dict_string_encoding() {
        let types = [DataType::Utf8];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let vals: Vec<&str> = (0..60).map(|i| ["aa", "bb", "cc"][i % 3]).collect();
        let arr = StringArray::from(vals);
        writer.write_columns(&[&arr], &[&DataType::Utf8]).unwrap();

        let data = writer.finish();
        let h = header_size(&data);
        assert_eq!(data[h] & 0x03, ENCODING_DICT);
    }

    #[test]
    fn test_const_with_nulls() {
        let types = [DataType::Int32];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let vals: Vec<Option<i32>> = (0..20)
            .map(|i| if i % 3 == 0 { None } else { Some(42) })
            .collect();
        let arr = Int32Array::from(vals);
        writer.write_columns(&[&arr], &[&DataType::Int32]).unwrap();

        let data = writer.finish();
        let h = header_size(&data);
        assert_eq!(data[h] & 0x03, ENCODING_CONST);
    }

    #[test]
    fn test_dict_with_nulls() {
        let types = [DataType::Int32];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let vals: Vec<Option<i32>> = (0..100)
            .map(|i| if i % 5 == 0 { None } else { Some(i % 3) })
            .collect();
        let arr = Int32Array::from(vals);
        writer.write_columns(&[&arr], &[&DataType::Int32]).unwrap();

        let data = writer.finish();
        let h = header_size(&data);
        assert_eq!(data[h] & 0x03, ENCODING_DICT);
    }

    #[test]
    fn test_timestamp_nanos_byte_dict_after_no_null_batch() {
        let types = [DataType::Timestamp(
            arrow_schema::TimeUnit::Nanosecond,
            None,
        )];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let first = TimestampNanosecondArray::from(vec![Some(1), None, Some(2)]);
        writer.write_columns(&[&first], &[&types[0]]).unwrap();

        let second_values: Vec<i64> = (0..120).map(|i| 3 + (i % 3) as i64).collect();
        let second = TimestampNanosecondArray::from(second_values);
        writer.write_columns(&[&second], &[&types[0]]).unwrap();

        let data = writer.finish();
        let h = header_size(&data);
        assert_eq!(data[h] & 0x03, ENCODING_DICT);
    }

    #[test]
    fn test_multi_column_mixed_encodings() {
        let types = [DataType::Int32, DataType::Utf8, DataType::Int64];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let col0 = Int32Array::new_null(100);
        let col1 = StringArray::from(vec!["same"; 100]);
        let col2_vals: Vec<i64> = (0..100).map(|i| i % 4).collect();
        let col2 = Int64Array::from(col2_vals);

        writer
            .write_columns(
                &[&col0, &col1, &col2],
                &[&DataType::Int32, &DataType::Utf8, &DataType::Int64],
            )
            .unwrap();

        let data = writer.finish();
        let h = header_size(&data);
        assert_eq!(data[h] & 0x03, ENCODING_ALL_NULL);
        assert_eq!((data[h] >> 2) & 0x03, ENCODING_CONST);
        assert_eq!((data[h] >> 4) & 0x03, ENCODING_DICT);
    }

    #[test]
    fn test_reset_and_reuse() {
        let types = [DataType::Int32];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let arr1 = Int32Array::from(vec![42; 10]);
        writer.write_columns(&[&arr1], &[&DataType::Int32]).unwrap();
        let data1 = writer.finish();
        assert_eq!(data1[0] & 0x03, ENCODING_CONST);

        writer.reset();
        assert!(writer.is_empty());

        let vals: Vec<i32> = (0..1000).collect();
        let arr2 = Int32Array::from(vals);
        writer.write_columns(&[&arr2], &[&DataType::Int32]).unwrap();
        let data2 = writer.finish();
        let h2 = header_size(&data2);
        assert_eq!(data2[h2] & 0x03, ENCODING_PLAIN);
    }
}
