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
use std::io;
use std::sync::Arc;

use arrow_array::*;
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field};

use crate::spec::*;
use crate::types;
use crate::values;
use crate::varint;

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

pub struct BucketWriter {
    num_primary: usize,
    total_columns: usize,
    fixed_widths: Vec<i32>,

    null_bitmaps: Vec<Vec<u8>>,
    value_buffers: Vec<Vec<u8>>,
    non_null_counts: Vec<usize>,

    const_tracking: Vec<bool>,
    first_value_len: Vec<usize>,

    long_dict_maps: Vec<Option<HashMap<u64, usize>>>,
    byte_dict_maps: Vec<Option<HashMap<Vec<u8>, usize>>>,
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

        let mut long_dict_maps = Vec::with_capacity(total_columns);
        let mut byte_dict_maps = Vec::with_capacity(total_columns);
        for fw in &fixed_widths {
            if uses_long_dict(*fw) {
                long_dict_maps.push(Some(HashMap::new()));
                byte_dict_maps.push(None);
            } else {
                long_dict_maps.push(None);
                byte_dict_maps.push(Some(HashMap::new()));
            }
        }

        BucketWriter {
            num_primary,
            total_columns,
            fixed_widths,
            null_bitmaps: vec![vec![0u8; 128]; total_columns],
            value_buffers: vec![Vec::with_capacity(1024); total_columns],
            non_null_counts: vec![0; total_columns],
            const_tracking: vec![true; total_columns],
            first_value_len: vec![0; total_columns],
            long_dict_maps,
            byte_dict_maps,
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

        if array.null_count() == 0 {
            if let Some(col_size) =
                self.append_no_null_batch(col, &typed, start_row, num_new_rows)?
            {
                return Ok(col_size);
            }
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

                if self.const_tracking[col] {
                    if self.non_null_counts[col] == 1 {
                        self.first_value_len[col] = written;
                    } else if written != self.first_value_len[col]
                        || !equals_first_value(&self.value_buffers[col], before, written)
                    {
                        self.const_tracking[col] = false;
                    }
                }

                if let Some(ref mut dict) = self.long_dict_maps[col] {
                    let key = values::extract_fixed_key(
                        &self.value_buffers[col],
                        before,
                        self.fixed_widths[col],
                    );
                    let len = dict.len();
                    dict.entry(key).or_insert(len);
                    if dict.len() > self.max_dict_entries {
                        self.long_dict_maps[col] = None;
                    }
                } else if let Some(ref mut dict) = self.byte_dict_maps[col] {
                    let slice = &self.value_buffers[col][before..before + written];
                    if !dict.contains_key(slice) {
                        let len = dict.len();
                        dict.insert(slice.to_vec(), len);
                        self.dict_total_bytes[col] += written;
                    }
                    if dict.len() > self.max_dict_entries
                        || self.dict_total_bytes[col] > self.max_dict_total_bytes
                    {
                        self.byte_dict_maps[col] = None;
                    }
                }
            }
        }
        Ok(col_size)
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
        let prev_non_null = self.non_null_counts[col];
        self.non_null_counts[col] += num_rows;

        if self.const_tracking[col] {
            if prev_non_null == 0 {
                self.first_value_len[col] = fw;
                for i in 1..num_rows {
                    if !equals_first_value(buf, before_all + i * fw, fw) {
                        self.const_tracking[col] = false;
                        break;
                    }
                }
            } else {
                for i in 0..num_rows {
                    if !equals_first_value(buf, before_all + i * fw, fw) {
                        self.const_tracking[col] = false;
                        break;
                    }
                }
            }
        }

        if let Some(ref mut dict) = self.long_dict_maps[col] {
            for i in 0..num_rows {
                let key =
                    values::extract_fixed_key(buf, before_all + i * fw, self.fixed_widths[col]);
                let len = dict.len();
                dict.entry(key).or_insert(len);
                if dict.len() > self.max_dict_entries {
                    self.long_dict_maps[col] = None;
                    break;
                }
            }
        } else if let Some(ref mut dict) = self.byte_dict_maps[col] {
            for i in 0..num_rows {
                let start = before_all + i * fw;
                let slice = &buf[start..start + fw];
                if !dict.contains_key(slice) {
                    let len = dict.len();
                    dict.insert(slice.to_vec(), len);
                    self.dict_total_bytes[col] += fw;
                }
                if dict.len() > self.max_dict_entries
                    || self.dict_total_bytes[col] > self.max_dict_total_bytes
                {
                    self.byte_dict_maps[col] = None;
                    break;
                }
            }
        }

        // null_bitmap stays all-zero (no nulls), start_row offsets are fine
        let _ = start_row;

        Ok(Some(col_size))
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
                    && self.dict_encoded_size(i) < self.value_buffers[i].len()
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

    #[allow(clippy::needless_range_loop)]
    pub fn finish(&self) -> Vec<u8> {
        if self.num_rows == 0 {
            return Vec::new();
        }

        let (encodings, has_nulls) = self.compute_encodings();

        let mut out = Vec::new();

        // Header: only written when ARRAY columns exist (backward compatible with v1)
        if !self.children.is_empty() {
            varint::encode(&mut out, self.num_primary as u32);
            varint::encode(&mut out, self.children.len() as u32);
            for child in &self.children {
                varint::encode(&mut out, child.num_elements as u32);
            }
        }

        // Encoding flags: 2 bits per column
        let encoding_flags_bytes = (self.total_columns * 2).div_ceil(8);
        let ef_start = out.len();
        out.resize(ef_start + encoding_flags_bytes, 0);
        for i in 0..self.total_columns {
            let byte_idx = (i * 2) / 8;
            let bit_idx = (i * 2) % 8;
            out[ef_start + byte_idx] |= encodings[i] << bit_idx;
        }

        // Has-nulls flags: 1 bit per column
        let has_nulls_bytes = self.total_columns.div_ceil(8);
        let hn_start = out.len();
        out.resize(hn_start + has_nulls_bytes, 0);
        for i in 0..self.total_columns {
            if has_nulls[i] {
                out[hn_start + i / 8] |= 1 << (i % 8);
            }
        }

        // CONST metadata
        for i in 0..self.total_columns {
            if encodings[i] == ENCODING_CONST {
                let len = self.first_value_len[i];
                out.extend_from_slice(&self.value_buffers[i][..len]);
            }
        }

        // Dict metadata
        for i in 0..self.total_columns {
            if encodings[i] == ENCODING_DICT {
                if let Some(ref dict) = self.long_dict_maps[i] {
                    let num_entries = dict.len();
                    varint::encode(&mut out, num_entries as u32);
                    let w = self.fixed_widths[i];
                    let mut keys = vec![0u64; num_entries];
                    for (&key, &idx) in dict {
                        keys[idx] = key;
                    }
                    for key in &keys {
                        write_fixed_key_to_vec(&mut out, *key, w);
                    }
                } else if let Some(ref dict) = self.byte_dict_maps[i] {
                    let num_entries = dict.len();
                    varint::encode(&mut out, num_entries as u32);
                    let mut keys: Vec<(&Vec<u8>, &usize)> = dict.iter().collect();
                    keys.sort_by_key(|&(_, idx)| *idx);
                    for (key, _) in keys {
                        out.extend_from_slice(key);
                    }
                }
            }
        }

        // Null bitmaps (per-column row count)
        for i in 0..self.total_columns {
            if has_nulls[i] && encodings[i] != ENCODING_ALL_NULL {
                let nbytes = self.col_num_rows(i).div_ceil(8);
                out.extend_from_slice(&self.null_bitmaps[i][..nbytes]);
            }
        }

        // Column data
        for i in 0..self.total_columns {
            if encodings[i] == ENCODING_PLAIN {
                out.extend_from_slice(&self.value_buffers[i]);
            } else if encodings[i] == ENCODING_DICT {
                let bw = bit_width(self.get_dict_size(i));
                let packed_bytes = (self.non_null_counts[i] * bw).div_ceil(8);
                let data_start = out.len();
                out.resize(data_start + packed_bytes, 0);
                self.write_dict_bit_packed(i, &mut out, data_start);
            }
        }

        out
    }

    #[allow(clippy::needless_range_loop)]
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

        let (encodings, has_nulls) = self.compute_encodings();

        let mut const_data = vec![Vec::new(); self.total_columns];
        let mut column_pages: Vec<Option<Vec<u8>>> = vec![None; self.total_columns];

        for i in 0..self.total_columns {
            let col_rows = self.col_num_rows(i);
            let null_bitmap_bytes = col_rows.div_ceil(8);

            match encodings[i] {
                ENCODING_ALL_NULL => {}
                ENCODING_CONST => {
                    let len = self.first_value_len[i];
                    const_data[i] = self.value_buffers[i][..len].to_vec();
                    if has_nulls[i] {
                        column_pages[i] = Some(self.null_bitmaps[i][..null_bitmap_bytes].to_vec());
                    }
                }
                ENCODING_DICT => {
                    let mut page = Vec::new();
                    if let Some(ref dict) = self.long_dict_maps[i] {
                        let num_entries = dict.len();
                        varint::encode(&mut page, num_entries as u32);
                        let w = self.fixed_widths[i];
                        let mut keys = vec![0u64; num_entries];
                        for (&key, &idx) in dict {
                            keys[idx] = key;
                        }
                        for key in &keys {
                            write_fixed_key_to_vec(&mut page, *key, w);
                        }
                    } else if let Some(ref dict) = self.byte_dict_maps[i] {
                        let num_entries = dict.len();
                        varint::encode(&mut page, num_entries as u32);
                        let mut keys: Vec<(&Vec<u8>, &usize)> = dict.iter().collect();
                        keys.sort_by_key(|&(_, idx)| *idx);
                        for (key, _) in keys {
                            page.extend_from_slice(key);
                        }
                    }
                    if has_nulls[i] {
                        page.extend_from_slice(&self.null_bitmaps[i][..null_bitmap_bytes]);
                    }
                    let dict_size = self.get_dict_size(i);
                    let bw = bit_width(dict_size);
                    let packed_bytes = (self.non_null_counts[i] * bw).div_ceil(8);
                    let data_start = page.len();
                    page.resize(data_start + packed_bytes, 0);
                    self.write_dict_bit_packed(i, &mut page, data_start);
                    column_pages[i] = Some(page);
                }
                ENCODING_PLAIN => {
                    let mut page = Vec::new();
                    if has_nulls[i] {
                        page.extend_from_slice(&self.null_bitmaps[i][..null_bitmap_bytes]);
                    }
                    page.extend_from_slice(&self.value_buffers[i]);
                    column_pages[i] = Some(page);
                }
                _ => {}
            }
        }

        PagedBucketOutput {
            encodings,
            has_nulls,
            const_data,
            column_pages,
            num_primary: self.num_primary,
            children: self.children.clone(),
        }
    }

    pub fn reset(&mut self) {
        for i in 0..self.total_columns {
            for b in &mut self.null_bitmaps[i] {
                *b = 0;
            }
            self.value_buffers[i].clear();
            self.non_null_counts[i] = 0;
            self.const_tracking[i] = true;
            self.first_value_len[i] = 0;
            self.dict_total_bytes[i] = 0;
            if uses_long_dict(self.fixed_widths[i]) {
                if let Some(ref mut dict) = self.long_dict_maps[i] {
                    dict.clear();
                } else {
                    self.long_dict_maps[i] = Some(HashMap::new());
                }
            } else if let Some(ref mut dict) = self.byte_dict_maps[i] {
                dict.clear();
            } else {
                self.byte_dict_maps[i] = Some(HashMap::new());
            }
        }
        self.num_rows = 0;
        for child in &mut self.children {
            child.num_elements = 0;
        }
    }

    fn get_dict_size(&self, col: usize) -> usize {
        if let Some(ref dict) = self.long_dict_maps[col] {
            return dict.len();
        }
        if let Some(ref dict) = self.byte_dict_maps[col] {
            return dict.len();
        }
        0
    }

    fn write_dict_bit_packed(&self, col: usize, buf: &mut [u8], data_start: usize) -> usize {
        let dict_size = self.get_dict_size(col);
        let bw = bit_width(dict_size);
        let w = self.fixed_widths[col];
        let mut bit_offset = 0usize;
        let mut val_pos = 0usize;

        for r in 0..self.col_num_rows(col) {
            let is_null = (self.null_bitmaps[col][r / 8] & (1 << (r % 8))) != 0;
            if !is_null {
                let idx = if let Some(ref dict) = self.long_dict_maps[col] {
                    let key = values::extract_fixed_key(&self.value_buffers[col], val_pos, w);
                    val_pos += w as usize;
                    *dict.get(&key).unwrap()
                } else if let Some(ref dict) = self.byte_dict_maps[col] {
                    let value_len = if w > 0 {
                        w as usize
                    } else {
                        let var_len =
                            varint::decode(&self.value_buffers[col], &mut val_pos.clone())
                                .expect("internal varint in value buffer");
                        varint::encoded_size(var_len) + var_len as usize
                    };
                    let slice = &self.value_buffers[col][val_pos..val_pos + value_len];
                    val_pos += value_len;
                    *dict.get(slice).unwrap()
                } else {
                    unreachable!()
                };
                write_bit_packed(buf, data_start, bit_offset, idx, bw);
                bit_offset += bw;
            }
        }
        bit_offset
    }

    fn dict_encoded_size(&self, col: usize) -> usize {
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
                    size += self.value_buffers[i].len();
                }
                _ => {}
            }
        }
        size
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

fn write_bit_packed(buf: &mut [u8], byte_base: usize, bit_offset: usize, value: usize, bw: usize) {
    if bw == 0 {
        return;
    }
    let start_byte = byte_base + bit_offset / 8;
    let bit_shift = bit_offset % 8;
    let mut bits = (value as u64) << bit_shift;
    let total_bits = bit_shift + bw;
    let num_bytes = total_bits.div_ceil(8);
    for i in 0..num_bytes {
        buf[start_byte + i] |= (bits & 0xFF) as u8;
        bits >>= 8;
    }
}

fn equals_first_value(buf: &[u8], offset: usize, len: usize) -> bool {
    buf[..len] == buf[offset..offset + len]
}

fn write_fixed_key_to_vec(buf: &mut Vec<u8>, key: u64, width: i32) {
    match width {
        1 => buf.push(key as u8),
        2 => buf.extend_from_slice(&(key as u16).to_be_bytes()),
        4 => buf.extend_from_slice(&(key as u32).to_be_bytes()),
        8 => buf.extend_from_slice(&key.to_be_bytes()),
        _ => {}
    }
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
