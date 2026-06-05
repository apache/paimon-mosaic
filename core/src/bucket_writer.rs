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
use arrow_buffer::ScalarBuffer;
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
}

struct ListColumnState {
    physical_col: usize,
    element_field: Arc<Field>,
    values_writer: BucketWriter,
    total_elements: usize,
}

pub struct BucketWriter {
    num_columns: usize,
    fixed_widths: Vec<i32>,

    null_bitmaps: Vec<Vec<u8>>,
    value_buffers: Vec<Vec<u8>>,
    non_null_counts: Vec<usize>,

    // CONST tracking
    const_tracking: Vec<bool>,
    first_value_len: Vec<usize>,

    // Dict tracking: fixed-width <=8 uses u64 keys, variable-width uses byte keys
    long_dict_maps: Vec<Option<HashMap<u64, usize>>>,
    byte_dict_maps: Vec<Option<HashMap<Vec<u8>, usize>>>,
    dict_total_bytes: Vec<usize>,
    max_dict_total_bytes: usize,
    max_dict_entries: usize,

    num_rows: usize,

    list_columns: Vec<ListColumnState>,
}

impl BucketWriter {
    pub fn new(
        col_types: &[&DataType],
        max_dict_total_bytes: usize,
        max_dict_entries: usize,
    ) -> Self {
        let num_columns = col_types.len();

        let physical_types: Vec<DataType> = col_types
            .iter()
            .map(|t| {
                if matches!(t, DataType::List(_)) {
                    DataType::Int32
                } else {
                    (*t).clone()
                }
            })
            .collect();
        let fixed_widths: Vec<i32> = physical_types.iter().map(types::fixed_width).collect();

        let mut list_columns = Vec::new();
        for (i, t) in col_types.iter().enumerate() {
            if let DataType::List(element_field) = t {
                let elem_dt = element_field.data_type();
                let values_writer =
                    BucketWriter::new(&[elem_dt], max_dict_total_bytes, max_dict_entries);
                list_columns.push(ListColumnState {
                    physical_col: i,
                    element_field: element_field.clone(),
                    values_writer,
                    total_elements: 0,
                });
            }
        }

        let mut long_dict_maps = Vec::with_capacity(num_columns);
        let mut byte_dict_maps = Vec::with_capacity(num_columns);

        for fw in fixed_widths.iter().take(num_columns) {
            if uses_long_dict(*fw) {
                long_dict_maps.push(Some(HashMap::new()));
                byte_dict_maps.push(None);
            } else {
                long_dict_maps.push(None);
                byte_dict_maps.push(Some(HashMap::new()));
            }
        }

        BucketWriter {
            num_columns,
            fixed_widths,
            null_bitmaps: vec![vec![0u8; 128]; num_columns],
            value_buffers: vec![Vec::with_capacity(1024); num_columns],
            non_null_counts: vec![0; num_columns],
            const_tracking: vec![true; num_columns],
            first_value_len: vec![0; num_columns],
            long_dict_maps,
            byte_dict_maps,
            dict_total_bytes: vec![0; num_columns],
            max_dict_total_bytes,
            max_dict_entries,
            num_rows: 0,
            list_columns,
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
        let mut size = self.compute_out_size(&encodings, &has_nulls);
        for lc in &self.list_columns {
            size += lc.values_writer.estimated_raw_size();
        }
        size
    }

    pub fn write_columns(
        &mut self,
        arrays: &[&dyn Array],
        data_types: &[&DataType],
    ) -> io::Result<usize> {
        debug_assert_eq!(arrays.len(), self.num_columns);
        let num_new_rows = arrays[0].len();
        if num_new_rows == 0 {
            return Ok(0);
        }
        let start_row = self.num_rows;
        let mut total_size = 0;

        let mut list_writes: Vec<(usize, Int32Array, ArrayRef)> = Vec::new();
        for lc in &self.list_columns {
            let col = lc.physical_col;
            let list_array = arrays[col]
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "expected ListArray"))?;
            let lengths = extract_list_lengths(list_array);
            let values = flatten_list_values(list_array);
            list_writes.push((col, lengths, values));
        }

        let int32_dt = DataType::Int32;
        for i in 0..self.num_columns {
            if let Some(lw) = list_writes.iter().find(|(col, _, _)| *col == i) {
                total_size += self.append_array_column(i, &lw.1, &int32_dt, start_row)?;
            } else {
                total_size += self.append_array_column(i, arrays[i], data_types[i], start_row)?;
            }
        }

        for (lc, (_, _, values)) in self.list_columns.iter_mut().zip(list_writes) {
            let elem_dt = lc.element_field.data_type();
            lc.values_writer
                .write_columns(&[values.as_ref()], &[elem_dt])?;
            lc.total_elements += values.len();
        }

        self.num_rows += num_new_rows;
        total_size += num_new_rows * self.num_columns.div_ceil(8);
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
        let mut encodings = vec![0u8; self.num_columns];
        let mut has_nulls = vec![false; self.num_columns];
        for i in 0..self.num_columns {
            if self.non_null_counts[i] == 0 {
                encodings[i] = ENCODING_ALL_NULL;
            } else if self.const_tracking[i] {
                encodings[i] = ENCODING_CONST;
                has_nulls[i] = self.non_null_counts[i] < self.num_rows;
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
                has_nulls[i] = self.non_null_counts[i] < self.num_rows;
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

        let out_size = self.compute_out_size(&encodings, &has_nulls);
        let mut out = vec![0u8; out_size];
        let mut pos = 0;

        // Encoding flags: 2 bits per column
        let encoding_flags_bytes = (self.num_columns * 2).div_ceil(8);
        for i in 0..self.num_columns {
            let byte_idx = (i * 2) / 8;
            let bit_idx = (i * 2) % 8;
            out[pos + byte_idx] |= encodings[i] << bit_idx;
        }
        pos += encoding_flags_bytes;

        // Has-nulls flags: 1 bit per column
        let has_nulls_bytes = self.num_columns.div_ceil(8);
        for i in 0..self.num_columns {
            if has_nulls[i] {
                out[pos + i / 8] |= 1 << (i % 8);
            }
        }
        pos += has_nulls_bytes;

        // CONST metadata
        for i in 0..self.num_columns {
            if encodings[i] == ENCODING_CONST {
                let len = self.first_value_len[i];
                out[pos..pos + len].copy_from_slice(&self.value_buffers[i][..len]);
                pos += len;
            }
        }

        // Dict metadata
        for i in 0..self.num_columns {
            if encodings[i] == ENCODING_DICT {
                if let Some(ref dict) = self.long_dict_maps[i] {
                    let num_entries = dict.len();
                    pos = varint::encode_to_slice(&mut out, pos, num_entries as u32);
                    let w = self.fixed_widths[i];
                    let mut keys = vec![0u64; num_entries];
                    for (&key, &idx) in dict {
                        keys[idx] = key;
                    }
                    for key in &keys {
                        write_fixed_key_to_slice(&mut out, &mut pos, *key, w);
                    }
                } else if let Some(ref dict) = self.byte_dict_maps[i] {
                    let num_entries = dict.len();
                    pos = varint::encode_to_slice(&mut out, pos, num_entries as u32);
                    let mut keys: Vec<(&Vec<u8>, &usize)> = dict.iter().collect();
                    keys.sort_by_key(|&(_, idx)| *idx);
                    for (key, _) in keys {
                        out[pos..pos + key.len()].copy_from_slice(key);
                        pos += key.len();
                    }
                }
            }
        }

        // Null bitmaps
        let null_bitmap_bytes = self.num_rows.div_ceil(8);
        for i in 0..self.num_columns {
            if has_nulls[i] && encodings[i] != ENCODING_ALL_NULL {
                out[pos..pos + null_bitmap_bytes]
                    .copy_from_slice(&self.null_bitmaps[i][..null_bitmap_bytes]);
                pos += null_bitmap_bytes;
            }
        }

        // Column data
        for i in 0..self.num_columns {
            if encodings[i] == ENCODING_PLAIN {
                let len = self.value_buffers[i].len();
                out[pos..pos + len].copy_from_slice(&self.value_buffers[i]);
                pos += len;
            } else if encodings[i] == ENCODING_DICT {
                let data_start = pos;
                let bit_offset = self.write_dict_bit_packed(i, &mut out, data_start);
                pos += bit_offset.div_ceil(8);
            }
        }

        debug_assert_eq!(pos, out.len());

        if !self.list_columns.is_empty() {
            varint::encode(&mut out, self.list_columns.len() as u32);
            for lc in &self.list_columns {
                varint::encode(&mut out, lc.total_elements as u32);
                let sub_blob = lc.values_writer.finish();
                varint::encode(&mut out, sub_blob.len() as u32);
                out.extend_from_slice(&sub_blob);
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
            };
        }

        let (mut encodings, mut has_nulls) = self.compute_encodings();
        let null_bitmap_bytes = self.num_rows.div_ceil(8);

        let mut const_data = vec![Vec::new(); self.num_columns];
        let mut column_pages: Vec<Option<Vec<u8>>> = vec![None; self.num_columns];

        for i in 0..self.num_columns {
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
                    // Dict table
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
                    // Null bitmap
                    if has_nulls[i] {
                        page.extend_from_slice(&self.null_bitmaps[i][..null_bitmap_bytes]);
                    }
                    // Bit-packed indices
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

        for lc in &self.list_columns {
            let col = lc.physical_col;

            let mut composite = Vec::new();
            varint::encode(&mut composite, lc.total_elements as u32);

            // Build complete lengths page_content: encoding(1B) + flags(1B) + [const_value] + page_data
            let lengths_encoding = encodings[col];
            let lengths_has_nulls = has_nulls[col];
            let mut lengths_content = vec![lengths_encoding, if lengths_has_nulls { 1 } else { 0 }];
            if lengths_encoding == ENCODING_CONST {
                lengths_content.extend_from_slice(&const_data[col]);
            }
            if let Some(ref page_data) = column_pages[col] {
                lengths_content.extend_from_slice(page_data);
            }
            varint::encode(&mut composite, lengths_content.len() as u32);
            composite.extend_from_slice(&lengths_content);

            // Build complete values page_content: encoding(1B) + flags(1B) + [const_value] + page_data
            let values_paged = lc.values_writer.finish_paged();
            let v_encoding = values_paged
                .encodings
                .first()
                .copied()
                .unwrap_or(ENCODING_ALL_NULL);
            let v_has_nulls = values_paged.has_nulls.first().copied().unwrap_or(false);
            let v_const = values_paged.const_data.first().cloned().unwrap_or_default();
            let v_page = values_paged
                .column_pages
                .first()
                .and_then(|p| p.clone())
                .unwrap_or_default();

            let mut values_content = vec![v_encoding, if v_has_nulls { 1 } else { 0 }];
            if v_encoding == ENCODING_CONST {
                values_content.extend_from_slice(&v_const);
            }
            values_content.extend_from_slice(&v_page);

            varint::encode(&mut composite, lc.total_elements as u32);
            varint::encode(&mut composite, values_content.len() as u32);
            composite.extend_from_slice(&values_content);

            // Override: use ENCODING_PLAIN so write_paged_bucket includes this slot.
            // The 2-byte [PLAIN, 0x00] prefix added by write_paged_bucket will be
            // skipped by parse_list_column_slot on the read side.
            encodings[col] = ENCODING_PLAIN;
            has_nulls[col] = false;
            const_data[col] = Vec::new();
            column_pages[col] = Some(composite);
        }

        PagedBucketOutput {
            encodings,
            has_nulls,
            const_data,
            column_pages,
        }
    }

    pub fn reset(&mut self) {
        for i in 0..self.num_columns {
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
        for lc in &mut self.list_columns {
            lc.values_writer.reset();
            lc.total_elements = 0;
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

        for r in 0..self.num_rows {
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
        let null_bitmap_bytes = self.num_rows.div_ceil(8);
        let mut size = (self.num_columns * 2).div_ceil(8) + self.num_columns.div_ceil(8);

        for i in 0..self.num_columns {
            if encodings[i] == ENCODING_ALL_NULL {
                continue;
            }
            if has_nulls[i] {
                size += null_bitmap_bytes;
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

fn extract_list_lengths(list_array: &ListArray) -> Int32Array {
    let offsets = list_array.value_offsets();
    let num_rows = list_array.len();
    let mut lengths = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        lengths.push(offsets[i + 1] - offsets[i]);
    }
    let null_buf = list_array.nulls().cloned();
    Int32Array::new(ScalarBuffer::from(lengths), null_buf)
}

fn flatten_list_values(list_array: &ListArray) -> ArrayRef {
    let offsets = list_array.value_offsets();
    let start = offsets[0] as usize;
    let end = offsets[list_array.len()] as usize;
    list_array.values().slice(start, end - start)
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

fn write_fixed_key_to_slice(buf: &mut [u8], pos: &mut usize, key: u64, width: i32) {
    match width {
        1 => {
            buf[*pos] = key as u8;
            *pos += 1;
        }
        2 => {
            let bytes = (key as u16).to_be_bytes();
            buf[*pos..*pos + 2].copy_from_slice(&bytes);
            *pos += 2;
        }
        4 => {
            let bytes = (key as u32).to_be_bytes();
            buf[*pos..*pos + 4].copy_from_slice(&bytes);
            *pos += 4;
        }
        8 => {
            let bytes = key.to_be_bytes();
            buf[*pos..*pos + 8].copy_from_slice(&bytes);
            *pos += 8;
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_null_encoding() {
        let types = [DataType::Int32];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let arr = Int32Array::new_null(10);
        writer.write_columns(&[&arr], &[&DataType::Int32]).unwrap();

        let data = writer.finish();
        assert!(!data.is_empty());
        assert_eq!(data[0] & 0x03, ENCODING_ALL_NULL);
    }

    #[test]
    fn test_const_encoding() {
        let types = [DataType::Int32];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let arr = Int32Array::from(vec![42; 10]);
        writer.write_columns(&[&arr], &[&DataType::Int32]).unwrap();

        let data = writer.finish();
        assert_eq!(data[0] & 0x03, ENCODING_CONST);
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
        assert_eq!(data[0] & 0x03, ENCODING_DICT);
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
        assert_eq!(data[0] & 0x03, ENCODING_PLAIN);
    }

    #[test]
    fn test_const_string_encoding() {
        let types = [DataType::Utf8];
        let type_refs: Vec<&DataType> = types.iter().collect();
        let mut writer = BucketWriter::new(&type_refs, 32768, 255);

        let arr = StringArray::from(vec!["same"; 50]);
        writer.write_columns(&[&arr], &[&DataType::Utf8]).unwrap();

        let data = writer.finish();
        assert_eq!(data[0] & 0x03, ENCODING_CONST);
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
        assert_eq!(data[0] & 0x03, ENCODING_DICT);
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
        assert_eq!(data[0] & 0x03, ENCODING_CONST);
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
        assert_eq!(data[0] & 0x03, ENCODING_DICT);
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
        assert_eq!(data[0] & 0x03, ENCODING_DICT);
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
        assert_eq!(data[0] & 0x03, ENCODING_ALL_NULL);
        assert_eq!((data[0] >> 2) & 0x03, ENCODING_CONST);
        assert_eq!((data[0] >> 4) & 0x03, ENCODING_DICT);
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
        assert_eq!(data2[0] & 0x03, ENCODING_PLAIN);
    }
}
