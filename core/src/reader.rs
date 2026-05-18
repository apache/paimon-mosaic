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

use arrow_array::ArrayRef;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};

use crate::bucket_reader::{read_typed_value, read_variable_value, BucketReader, ColumnPageReader};
use crate::schema::MosaicSchema;
use crate::spec::*;
use crate::stats::{self, ColumnStats};
use crate::types;
use crate::values::Value;
use crate::varint;

const COALESCE_GAP: u64 = 1024 * 1024;
const COALESCE_MAX_RANGE: u64 = 32 * 1024 * 1024;

#[derive(Clone)]
pub struct ReadRangeBuffer {
    data: Arc<Vec<u8>>,
    start: usize,
    len: usize,
}

impl ReadRangeBuffer {
    pub fn new(data: Arc<Vec<u8>>, start: usize, len: usize) -> io::Result<Self> {
        if start.checked_add(len).is_none_or(|end| end > data.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read range buffer bounds exceed backing data",
            ));
        }

        Ok(Self { data, start, len })
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data[self.start..self.start + self.len]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A random-access file abstraction for reading Mosaic files.
///
/// The `Sync` bound is required because the reader may call `read_at` from
/// multiple threads in parallel (e.g. when coalescing IO ranges).
/// Implementations must ensure that concurrent `read_at` calls are safe.
pub trait InputFile: Sync {
    /// Read `buf.len()` bytes starting at `offset`.
    ///
    /// # Thread safety
    /// This method must be safe to call concurrently from multiple threads.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    fn read_ranges_shared(&self, ranges: &[(u64, usize)]) -> io::Result<Vec<ReadRangeBuffer>> {
        let (merged, fetched) = read_merged_ranges(self, ranges)?;
        let fetched: Vec<Arc<Vec<u8>>> = fetched.into_iter().map(Arc::new).collect();

        // Distribute views back to original order
        let mut results: Vec<Option<ReadRangeBuffer>> = Vec::with_capacity(ranges.len());
        results.resize_with(ranges.len(), || None);
        for (mi, mr) in merged.iter().enumerate() {
            let data = fetched[mi].clone();
            for &idx in &mr.members {
                let (offset, len) = ranges[idx];
                let rel_start = (offset - mr.start) as usize;
                results[idx] = Some(ReadRangeBuffer::new(data.clone(), rel_start, len)?);
            }
        }

        results
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing range buffer"))
    }

    fn read_ranges(&self, ranges: &[(u64, usize)]) -> io::Result<Vec<Vec<u8>>> {
        let (merged, fetched) = read_merged_ranges(self, ranges)?;

        // Distribute slices back to original order
        let mut results: Vec<Vec<u8>> = Vec::with_capacity(ranges.len());
        results.resize_with(ranges.len(), Vec::new);
        for (mi, mr) in merged.iter().enumerate() {
            let buf = &fetched[mi];
            for &idx in &mr.members {
                let (offset, len) = ranges[idx];
                let rel_start = (offset - mr.start) as usize;
                results[idx] = buf[rel_start..rel_start + len].to_vec();
            }
        }

        Ok(results)
    }
}

struct MergedRange {
    start: u64,
    end: u64,
    members: Vec<usize>,
}

fn read_merged_ranges<I: InputFile + ?Sized>(
    input: &I,
    ranges: &[(u64, usize)],
) -> io::Result<(Vec<MergedRange>, Vec<Vec<u8>>)> {
    if ranges.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut indices: Vec<usize> = (0..ranges.len()).collect();
    indices.sort_unstable_by_key(|&i| ranges[i].0);

    let mut merged: Vec<MergedRange> = Vec::new();
    for &idx in &indices {
        let (offset, len) = ranges[idx];
        let range_end = offset + len as u64;

        let should_merge = if let Some(last) = merged.last() {
            offset >= last.start
                && offset.saturating_sub(last.end) <= COALESCE_GAP
                && (range_end - last.start) <= COALESCE_MAX_RANGE
        } else {
            false
        };

        if should_merge {
            let last = merged.last_mut().unwrap();
            last.end = last.end.max(range_end);
            last.members.push(idx);
        } else {
            merged.push(MergedRange {
                start: offset,
                end: range_end,
                members: vec![idx],
            });
        }
    }

    let fetched: Vec<io::Result<Vec<u8>>> = std::thread::scope(|s| {
        let handles: Vec<_> = merged
            .iter()
            .map(|mr| {
                s.spawn(|| {
                    let len = (mr.end - mr.start) as usize;
                    let mut buf = vec![0u8; len];
                    input.read_at(mr.start, &mut buf)?;
                    Ok(buf)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let fetched = fetched.into_iter().collect::<io::Result<Vec<_>>>()?;

    Ok((merged, fetched))
}

pub struct RowGroupMeta {
    pub num_rows: usize,
    pub bucket_offsets: Vec<u64>,
    pub bucket_layouts: Vec<BucketLayout>,
    pub stats: Vec<ColumnStats>,
}

pub trait ReaderAccess {
    fn schema(&self) -> &MosaicSchema;
    fn num_row_groups(&self) -> usize;
    fn row_group_reader(&self, rg_index: usize) -> io::Result<RowGroupReader>;
    fn row_group_reader_projected(
        &self,
        rg_index: usize,
        columns: &[usize],
    ) -> io::Result<RowGroupReader>;
    fn row_group_stats(&self, rg_index: usize) -> io::Result<&[ColumnStats]>;
}

pub struct MosaicReader<I: InputFile> {
    input: I,
    schema: MosaicSchema,
    row_group_metas: Vec<RowGroupMeta>,
    compression: u8,
    num_buckets: usize,
}

fn read_range(input: &dyn InputFile, offset: u64, len: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    input.read_at(offset, &mut buf)?;
    Ok(buf)
}

const TAIL_PREFETCH_SIZE: u64 = 64 * 1024;

impl<I: InputFile> MosaicReader<I> {
    pub fn new(input: I, file_len: u64) -> io::Result<Self> {
        if (file_len as usize) < FOOTER_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        // Read a tail chunk that likely covers all metadata in one IO
        let tail_size = file_len.min(TAIL_PREFETCH_SIZE) as usize;
        let tail_offset = file_len - tail_size as u64;
        let tail = read_range(&input, tail_offset, tail_size)?;

        let footer = &tail[tail_size - FOOTER_SIZE..];

        if footer[28] != MAGIC[0]
            || footer[29] != MAGIC[1]
            || footer[30] != MAGIC[2]
            || footer[31] != MAGIC[3]
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad magic bytes",
            ));
        }

        let version = footer[25];
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version: {}", version),
            ));
        }

        let index_offset = u64::from_be_bytes(footer[0..8].try_into().unwrap());
        let schema_block_offset = u64::from_be_bytes(footer[8..16].try_into().unwrap());
        let num_buckets = u32::from_be_bytes(footer[16..20].try_into().unwrap()) as usize;
        let num_row_groups = u32::from_be_bytes(footer[20..24].try_into().unwrap()) as usize;
        let compression = footer[24];

        let schema_data_start = schema_block_offset.checked_add(4).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "corrupted footer offsets")
        })?;
        let footer_start = file_len - FOOTER_SIZE as u64;
        if !(schema_data_start <= index_offset
            && index_offset <= footer_start
            && footer_start <= file_len)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "corrupted footer offsets",
            ));
        }

        // All metadata starts at schema_block_offset. Check if our tail covers it.
        let meta_buf = if schema_block_offset >= tail_offset {
            // Tail covers all metadata — zero additional IO
            let local_start = (schema_block_offset - tail_offset) as usize;
            let local_end = tail_size - FOOTER_SIZE;
            tail[local_start..local_end].to_vec()
        } else {
            // Metadata is larger than our tail prefetch — one more IO
            let meta_len = (footer_start - schema_block_offset) as usize;
            read_range(&input, schema_block_offset, meta_len)?
        };

        // Parse schema block from meta_buf
        let schema_uncompressed_size =
            u32::from_be_bytes(meta_buf[0..4].try_into().unwrap()) as usize;
        let schema_compressed_len = (index_offset - schema_block_offset - 4) as usize;
        let schema_compressed = &meta_buf[4..4 + schema_compressed_len];

        let schema_raw = match compression {
            COMPRESSION_NONE => schema_compressed.to_vec(),
            COMPRESSION_ZSTD => zstd::bulk::decompress(schema_compressed, schema_uncompressed_size)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported compression: {}", compression),
                ))
            }
        };

        let schema = MosaicSchema::deserialize(&schema_raw)?;

        if schema.num_buckets != num_buckets {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "footer num_buckets does not match schema",
            ));
        }

        if num_buckets == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "num_buckets must be > 0",
            ));
        }

        // Parse row group index from meta_buf
        let index_local_start = (index_offset - schema_block_offset) as usize;
        let index_data = &meta_buf[index_local_start..];
        let mut pos = 0usize;
        let mut row_group_metas = Vec::with_capacity(num_row_groups);

        for _ in 0..num_row_groups {
            let num_rows = varint::decode(index_data, &mut pos)? as usize;
            let non_empty = varint::decode(index_data, &mut pos)? as usize;

            if non_empty > num_buckets {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "non_empty count exceeds num_buckets",
                ));
            }

            let mut bucket_offsets = vec![0u64; num_buckets];
            let mut bucket_layouts = vec![BucketLayout::Empty; num_buckets];
            let mut seen_buckets = vec![false; num_buckets];

            for _ in 0..non_empty {
                let bucket_id = varint::decode(index_data, &mut pos)? as usize;
                if bucket_id >= num_buckets || pos + 8 > index_data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "corrupted row group index",
                    ));
                }
                if seen_buckets[bucket_id] {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "duplicate bucket_id in row group index",
                    ));
                }
                seen_buckets[bucket_id] = true;
                bucket_offsets[bucket_id] =
                    u64::from_be_bytes(index_data[pos..pos + 8].try_into().unwrap());
                pos += 8;
                let compressed_size = varint::decode(index_data, &mut pos)? as usize;
                let bulk_decompress_size = varint::decode(index_data, &mut pos)? as usize;
                bucket_layouts[bucket_id] =
                    BucketLayout::decode(compressed_size, bulk_decompress_size)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

                let end = bucket_offsets[bucket_id]
                    .checked_add(compressed_size as u64)
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "bucket offset overflow")
                    })?;
                if end > schema_block_offset {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "bucket data extends past schema block",
                    ));
                }
            }

            let rg_stats =
                stats::deserialize_stats(index_data, &mut pos, &schema.columns, num_rows)?;

            row_group_metas.push(RowGroupMeta {
                num_rows,
                bucket_offsets,
                bucket_layouts,
                stats: rg_stats,
            });
        }

        if pos != index_data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "trailing bytes in row group index",
            ));
        }

        Ok(MosaicReader {
            input,
            schema,
            row_group_metas,
            compression,
            num_buckets,
        })
    }

    pub fn input(&self) -> &I {
        &self.input
    }

    fn parse_column_slot(
        slot_data: &[u8],
        col_type: &DataType,
        num_rows: usize,
    ) -> io::Result<ColumnPageReader> {
        let mut spos = 0usize;
        let uncompressed_size = varint::decode(slot_data, &mut spos)? as usize;
        let compressed_data = &slot_data[spos..];
        let page_content = zstd::bulk::decompress(compressed_data, uncompressed_size)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        if page_content.len() < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "paged bucket: page_content too short",
            ));
        }
        let encoding = page_content[0];
        let flags = page_content[1];
        let has_nulls = (flags & 1) != 0;
        let mut ppos = 2usize;

        let mut const_value = Value::Null;
        if encoding == ENCODING_CONST {
            let w = types::fixed_width(col_type);
            if w > 0 {
                if ppos + w as usize > page_content.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "paged bucket: page_content truncated at const value",
                    ));
                }
                const_value = read_typed_value(col_type, &page_content, ppos, w);
                ppos += w as usize;
            } else {
                let (value, size) = read_variable_value(col_type, &page_content, ppos)?;
                const_value = value;
                ppos += size;
            }
        }

        ColumnPageReader::new_with_page_data_start(
            col_type.clone(),
            encoding,
            has_nulls,
            const_value,
            page_content,
            ppos,
            num_rows,
        )
    }
}

impl<I: InputFile> ReaderAccess for MosaicReader<I> {
    fn schema(&self) -> &MosaicSchema {
        &self.schema
    }

    fn num_row_groups(&self) -> usize {
        self.row_group_metas.len()
    }

    fn row_group_stats(&self, rg_index: usize) -> io::Result<&[ColumnStats]> {
        if rg_index >= self.row_group_metas.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "row group index {} out of range (num_row_groups={})",
                    rg_index,
                    self.row_group_metas.len()
                ),
            ));
        }
        Ok(&self.row_group_metas[rg_index].stats)
    }

    fn row_group_reader(&self, rg_index: usize) -> io::Result<RowGroupReader> {
        let all_columns: Vec<usize> = (0..self.schema.columns.len()).collect();
        self.row_group_reader_projected(rg_index, &all_columns)
    }

    #[allow(clippy::needless_range_loop)]
    fn row_group_reader_projected(
        &self,
        rg_index: usize,
        columns: &[usize],
    ) -> io::Result<RowGroupReader> {
        if rg_index >= self.row_group_metas.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "row group index out of range",
            ));
        }

        let meta = &self.row_group_metas[rg_index];
        let num_cols = self.schema.columns.len();

        let mut projected = vec![false; num_cols];
        for &c in columns {
            if c >= num_cols {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "projected column index {} out of range (num_columns={})",
                        c, num_cols
                    ),
                ));
            }
            projected[c] = true;
        }

        let mut needed_buckets = vec![false; self.num_buckets];
        let mut all_projected_in_bucket = vec![false; self.num_buckets];
        for b in 0..self.num_buckets {
            let mut any = false;
            let mut all = true;
            for &gi in &self.schema.bucket_to_global[b] {
                if projected[gi] {
                    any = true;
                } else {
                    all = false;
                }
            }
            needed_buckets[b] = any;
            all_projected_in_bucket[b] = any && all;
        }

        // Classify buckets and collect Round 1 ranges:
        // - Monolithic buckets: read entire compressed blob
        // - Paged buckets with all columns projected: read entire bucket (skip round 2)
        // - Paged buckets with partial projection: read directory only (round 2 fetches slots)
        let mut bucket_kinds = Vec::with_capacity(self.num_buckets);
        let mut r1_ranges: Vec<(u64, usize)> = Vec::new();
        let mut r1_bucket_ids: Vec<usize> = Vec::new();

        for b in 0..self.num_buckets {
            let layout = if needed_buckets[b] {
                meta.bucket_layouts[b]
            } else {
                BucketLayout::Empty
            };
            match layout {
                BucketLayout::Empty => {
                    bucket_kinds.push(BucketLayout::Empty);
                }
                BucketLayout::Paged { total_size } => {
                    if self.compression != COMPRESSION_ZSTD {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "paged bucket requires ZSTD compression",
                        ));
                    }
                    let dir_size = self.schema.bucket_to_global[b].len() * 4;
                    if dir_size > total_size {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "paged bucket {}: directory size {} exceeds total size {}",
                                b, dir_size, total_size
                            ),
                        ));
                    }
                    if all_projected_in_bucket[b] {
                        r1_ranges.push((meta.bucket_offsets[b], total_size));
                    } else {
                        r1_ranges.push((meta.bucket_offsets[b], dir_size));
                    }
                    r1_bucket_ids.push(b);
                    bucket_kinds.push(layout);
                }
                BucketLayout::Monolithic {
                    compressed_size, ..
                } => {
                    r1_ranges.push((meta.bucket_offsets[b], compressed_size));
                    r1_bucket_ids.push(b);
                    bucket_kinds.push(layout);
                }
            }
        }

        // Round 1: batch read all directories + monolithic blobs
        let r1_buffers = self.input.read_ranges_shared(&r1_ranges)?;

        // Process Round 1 results, build Round 2 ranges for paged bucket slots
        let mut bucket_states: Vec<Option<BucketState>> =
            (0..self.num_buckets).map(|_| None).collect();
        let mut r2_ranges: Vec<(u64, usize)> = Vec::new();
        // Track which paged bucket each merged range group belongs to,
        // and which columns within that bucket
        struct PagedSlotInfo {
            bucket_id: usize,
            col_idx: usize,
        }
        let mut r2_group_infos: Vec<Vec<PagedSlotInfo>> = Vec::new();

        // Per-bucket directory parse results (slot_sizes, slot_file_offsets) for paged buckets
        let mut paged_dir_info: Vec<Option<(Vec<usize>, Vec<u64>)>> = vec![None; self.num_buckets];

        for (ri, &b) in r1_bucket_ids.iter().enumerate() {
            let buf = r1_buffers[ri].as_slice();
            match bucket_kinds[b] {
                BucketLayout::Monolithic {
                    uncompressed_size, ..
                } => {
                    let global_indices = &self.schema.bucket_to_global[b];
                    let bucket_data = match self.compression {
                        COMPRESSION_NONE => buf.to_vec(),
                        COMPRESSION_ZSTD => zstd::bulk::decompress(buf, uncompressed_size)
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
                        _ => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "unsupported compression",
                            ))
                        }
                    };
                    let col_types: Vec<DataType> = global_indices
                        .iter()
                        .map(|&gi| self.schema.columns[gi].data_type.clone())
                        .collect();
                    let reader =
                        Box::new(BucketReader::new(col_types, bucket_data, meta.num_rows)?);
                    bucket_states[b] = Some(BucketState::Monolithic { reader });
                }
                BucketLayout::Paged { total_size } => {
                    let global_indices = &self.schema.bucket_to_global[b];
                    let num_columns = global_indices.len();

                    // Parse directory
                    let mut slot_sizes = Vec::with_capacity(num_columns);
                    for i in 0..num_columns {
                        let off = i * 4;
                        let size =
                            u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
                        slot_sizes.push(size);
                    }

                    // Validate: directory + slots must exactly equal total_size
                    let dir_size = num_columns * 4;
                    let slot_total: usize = slot_sizes.iter().sum();
                    if dir_size + slot_total != total_size {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "paged bucket {}: directory ({}) + slots ({}) != total size ({})",
                                b, dir_size, slot_total, total_size
                            ),
                        ));
                    }

                    if all_projected_in_bucket[b] {
                        // All columns projected — we already read the full bucket in round 1,
                        // parse all slots directly without a second read_ranges call.
                        let mut column_readers: Vec<Option<ColumnPageReader>> =
                            Vec::with_capacity(num_columns);
                        let mut data_offset = dir_size;
                        for i in 0..num_columns {
                            let gi = global_indices[i];
                            let col_type = self.schema.columns[gi].data_type.clone();

                            if slot_sizes[i] == 0 {
                                column_readers.push(Some(ColumnPageReader::new(
                                    col_type,
                                    ENCODING_ALL_NULL,
                                    false,
                                    Value::Null,
                                    Vec::new(),
                                    meta.num_rows,
                                )?));
                            } else {
                                let slot_data = &buf[data_offset..data_offset + slot_sizes[i]];
                                let column_reader =
                                    Self::parse_column_slot(slot_data, &col_type, meta.num_rows)?;
                                column_readers.push(Some(column_reader));
                            }
                            data_offset += slot_sizes[i];
                        }
                        bucket_states[b] = Some(BucketState::Paged { column_readers });
                    } else {
                        // Partial projection — only directory was read in round 1,
                        // collect ranges for round 2.
                        let bucket_offset = meta.bucket_offsets[b];
                        let mut slot_file_offsets = Vec::with_capacity(num_columns);
                        let mut foff = bucket_offset + dir_size as u64;
                        for &size in &slot_sizes {
                            slot_file_offsets.push(foff);
                            foff += size as u64;
                        }

                        let mut projected_cols: Vec<usize> = Vec::new();
                        for i in 0..num_columns {
                            let gi = global_indices[i];
                            if projected[gi] && slot_sizes[i] > 0 {
                                projected_cols.push(i);
                            }
                        }

                        for &col_idx in &projected_cols {
                            let col_offset = slot_file_offsets[col_idx];
                            let col_size = slot_sizes[col_idx];

                            if let Some(last_range) = r2_ranges.last_mut() {
                                let last_end = last_range.0 + last_range.1 as u64;
                                if col_offset == last_end {
                                    last_range.1 += col_size;
                                    r2_group_infos.last_mut().unwrap().push(PagedSlotInfo {
                                        bucket_id: b,
                                        col_idx,
                                    });
                                    continue;
                                }
                            }
                            r2_ranges.push((col_offset, col_size));
                            r2_group_infos.push(vec![PagedSlotInfo {
                                bucket_id: b,
                                col_idx,
                            }]);
                        }

                        paged_dir_info[b] = Some((slot_sizes, slot_file_offsets));
                    }
                }
                BucketLayout::Empty => {}
            }
        }

        // Round 2: batch read all paged column slots
        if !r2_ranges.is_empty() {
            let r2_buffers = self.input.read_ranges_shared(&r2_ranges)?;

            struct SlotLocation {
                group_idx: usize,
                start: usize,
                len: usize,
            }

            let mut slot_locations: Vec<Vec<Option<SlotLocation>>> =
                Vec::with_capacity(self.num_buckets);
            for b in 0..self.num_buckets {
                let n = self.schema.bucket_to_global[b].len();
                slot_locations.push((0..n).map(|_| None).collect());
            }

            for (group_idx, group) in r2_group_infos.iter().enumerate() {
                let buf = r2_buffers[group_idx].as_slice();
                let group_base = r2_ranges[group_idx].0;
                for info in group {
                    let (_, ref slot_file_offsets) =
                        paged_dir_info[info.bucket_id].as_ref().unwrap();
                    let (ref slot_sizes, _) = paged_dir_info[info.bucket_id].as_ref().unwrap();
                    let rel_start = (slot_file_offsets[info.col_idx] - group_base) as usize;
                    let slot_len = slot_sizes[info.col_idx];
                    let Some(rel_end) = rel_start.checked_add(slot_len) else {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "paged bucket: slot range overflows read buffer",
                        ));
                    };
                    if rel_end > buf.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "paged bucket: slot range exceeds read buffer",
                        ));
                    }
                    slot_locations[info.bucket_id][info.col_idx] = Some(SlotLocation {
                        group_idx,
                        start: rel_start,
                        len: slot_len,
                    });
                }
            }

            // Build ColumnPageReaders for each paged bucket
            for b in 0..self.num_buckets {
                if !matches!(bucket_kinds[b], BucketLayout::Paged { .. }) {
                    continue;
                }
                let global_indices = &self.schema.bucket_to_global[b];
                let num_columns = global_indices.len();
                let (ref slot_sizes, _) = paged_dir_info[b].as_ref().unwrap();

                let mut column_readers: Vec<Option<ColumnPageReader>> =
                    Vec::with_capacity(num_columns);
                for i in 0..num_columns {
                    let gi = global_indices[i];
                    if !projected[gi] {
                        column_readers.push(None);
                        continue;
                    }

                    let col_type = self.schema.columns[gi].data_type.clone();

                    if slot_sizes[i] == 0 {
                        column_readers.push(Some(ColumnPageReader::new(
                            col_type,
                            ENCODING_ALL_NULL,
                            false,
                            Value::Null,
                            Vec::new(),
                            meta.num_rows,
                        )?));
                        continue;
                    }

                    let location = slot_locations[b][i].as_ref().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "paged bucket: missing projected slot data",
                        )
                    })?;
                    let group_buffer = r2_buffers[location.group_idx].as_slice();
                    let slot_data = &group_buffer[location.start..location.start + location.len];
                    let column_reader =
                        Self::parse_column_slot(slot_data, &col_type, meta.num_rows)?;
                    column_readers.push(Some(column_reader));
                }
                bucket_states[b] = Some(BucketState::Paged { column_readers });
            }
        }

        Ok(RowGroupReader::new(
            bucket_states,
            self.schema.bucket_to_global.clone(),
            self.schema.clone(),
            num_cols,
            meta.num_rows,
        ))
    }
}

enum BucketState {
    Monolithic {
        reader: Box<BucketReader>,
    },
    Paged {
        column_readers: Vec<Option<ColumnPageReader>>,
    },
}

pub struct RowGroupReader {
    bucket_states: Vec<Option<BucketState>>,
    bucket_to_global: Vec<Vec<usize>>,
    active_buckets: Vec<usize>,
    schema: MosaicSchema,
    num_rows: usize,
    num_columns: usize,
}

impl RowGroupReader {
    fn new(
        bucket_states: Vec<Option<BucketState>>,
        bucket_to_global: Vec<Vec<usize>>,
        schema: MosaicSchema,
        num_columns: usize,
        num_rows: usize,
    ) -> Self {
        let active_buckets: Vec<usize> = bucket_states
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if s.is_some() { Some(i) } else { None })
            .collect();
        RowGroupReader {
            bucket_states,
            bucket_to_global,
            active_buckets,
            schema,
            num_rows,
            num_columns,
        }
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn read_columns(&mut self) -> io::Result<RecordBatch> {
        let num_cols = self.num_columns;
        let mut arrays: Vec<Option<ArrayRef>> = vec![None; num_cols];

        for &bucket_id in &self.active_buckets {
            let global_indices = &self.bucket_to_global[bucket_id];
            let state = self.bucket_states[bucket_id].as_ref().unwrap();

            match state {
                BucketState::Paged { column_readers } => {
                    for (local_idx, &global_idx) in global_indices.iter().enumerate() {
                        if let Some(ref cr) = column_readers[local_idx] {
                            arrays[global_idx] = Some(cr.read_all()?);
                        }
                    }
                }
                BucketState::Monolithic { reader, .. } => {
                    let columns = reader.read_all_columns()?;
                    for (local_idx, &global_idx) in global_indices.iter().enumerate() {
                        if local_idx < columns.len() {
                            arrays[global_idx] = Some(columns[local_idx].clone());
                        }
                    }
                }
            }
        }

        let mut fields = Vec::new();
        let mut batch_arrays = Vec::new();
        for (i, arr_opt) in arrays.into_iter().enumerate() {
            if let Some(arr) = arr_opt {
                let col_meta = &self.schema.columns[i];
                fields.push(Field::new(
                    &col_meta.name,
                    col_meta.data_type.clone(),
                    col_meta.nullable,
                ));
                batch_arrays.push(arr);
            }
        }

        let arrow_schema = std::sync::Arc::new(Schema::new(fields));
        RecordBatch::try_new(arrow_schema, batch_arrays)
            .map_err(|e| io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
#[path = "reader_tests.rs"]
mod tests;
