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

use arrow_array::*;
use arrow_schema::Schema;

use crate::bucket_writer::{BucketWriter, PagedBucketOutput};
use crate::schema::MosaicSchema;
use crate::spec::*;
use crate::varint;

fn to_u32(val: usize, field: &str) -> io::Result<u32> {
    u32::try_from(val).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} ({}) exceeds u32::MAX", field, val),
        )
    })
}

pub trait OutputFile {
    fn write(&mut self, data: &[u8]) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()>;
    fn pos(&self) -> u64;
}

pub struct WriterOptions {
    pub compression: u8,
    pub zstd_level: i32,
    pub num_buckets: usize,
    pub row_group_max_size: u64,
    pub max_dict_total_bytes: usize,
    pub max_dict_entries: usize,
    pub page_size_threshold: usize,
}

impl Default for WriterOptions {
    fn default() -> Self {
        Self {
            compression: COMPRESSION_ZSTD,
            zstd_level: DEFAULT_ZSTD_LEVEL,
            num_buckets: DEFAULT_NUM_BUCKETS,
            row_group_max_size: DEFAULT_ROW_GROUP_MAX_SIZE,
            max_dict_total_bytes: DEFAULT_DICT_MAX_TOTAL_BYTES,
            max_dict_entries: DEFAULT_DICT_MAX_ENTRIES,
            page_size_threshold: DEFAULT_PAGE_SIZE_THRESHOLD,
        }
    }
}

struct RowGroupMeta {
    num_rows: usize,
    bucket_offsets: Vec<u64>,
    bucket_layouts: Vec<BucketLayout>,
}

pub struct MosaicWriter<S: OutputFile> {
    out: S,
    schema: MosaicSchema,
    bucket_writers: Vec<Option<BucketWriter>>,
    active_buckets: Vec<usize>,
    num_buckets: usize,
    compression: u8,
    zstd_level: i32,
    row_group_max_size: u64,
    page_size_threshold: usize,

    row_group_metas: Vec<RowGroupMeta>,
    current_row_group_rows: usize,
    current_buffered_size: u64,
    compression_ratio: f64,
    total_uncompressed: u64,
    total_compressed: u64,
    closed: bool,
}

impl<S: OutputFile> MosaicWriter<S> {
    pub fn new(out: S, schema: &Schema, options: WriterOptions) -> io::Result<Self> {
        let mosaic_schema = MosaicSchema::from_arrow(schema, options.num_buckets)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        Ok(Self::from_mosaic_schema(out, mosaic_schema, options))
    }

    pub fn from_mosaic_schema(out: S, schema: MosaicSchema, options: WriterOptions) -> Self {
        let num_buckets = schema.num_buckets;
        let mut bucket_writers = Vec::with_capacity(num_buckets);

        for b in 0..num_buckets {
            let global_indices = &schema.bucket_to_global[b];
            if global_indices.is_empty() {
                bucket_writers.push(None);
            } else {
                let col_types: Vec<&arrow_schema::DataType> = global_indices
                    .iter()
                    .map(|&gi| &schema.columns[gi].data_type)
                    .collect();
                bucket_writers.push(Some(BucketWriter::new(
                    &col_types,
                    options.max_dict_total_bytes,
                    options.max_dict_entries,
                )));
            }
        }

        let active_buckets: Vec<usize> = bucket_writers
            .iter()
            .enumerate()
            .filter_map(|(i, bw)| if bw.is_some() { Some(i) } else { None })
            .collect();

        let compression_ratio = if options.compression == COMPRESSION_NONE {
            1.0
        } else {
            0.3
        };

        MosaicWriter {
            out,
            schema,
            bucket_writers,
            active_buckets,
            num_buckets,
            compression: options.compression,
            zstd_level: options.zstd_level,
            row_group_max_size: options.row_group_max_size,
            page_size_threshold: options.page_size_threshold,
            row_group_metas: Vec::new(),
            current_row_group_rows: 0,
            current_buffered_size: 0,
            compression_ratio,
            total_uncompressed: 0,
            total_compressed: 0,
            closed: false,
        }
    }

    pub fn schema(&self) -> &MosaicSchema {
        &self.schema
    }

    pub fn output(&self) -> &S {
        &self.out
    }

    pub fn output_mut(&mut self) -> &mut S {
        &mut self.out
    }

    pub fn estimated_file_size(&self) -> u64 {
        let written = self.out.pos();
        let buffered_estimate = (self.current_buffered_size as f64 * self.compression_ratio) as u64;
        written + buffered_estimate + 1024
    }

    pub fn write_batch(&mut self, batch: &RecordBatch) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "writer is already closed",
            ));
        }
        let num_cols = self.schema.columns.len();
        if batch.num_columns() != num_cols {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "column count mismatch: schema has {} but batch has {}",
                    num_cols,
                    batch.num_columns()
                ),
            ));
        }

        for (i, col) in self.schema.columns.iter().enumerate() {
            if !col.nullable && batch.column(i).null_count() > 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "non-nullable column '{}' has {} nulls in batch",
                        col.name,
                        batch.column(i).null_count()
                    ),
                ));
            }
        }

        let mut size = 0u64;
        for &b in &self.active_buckets {
            let global_indices = &self.schema.bucket_to_global[b];
            let arrays: Vec<&dyn Array> = global_indices
                .iter()
                .map(|&gi| batch.column(gi).as_ref())
                .collect();
            let data_types: Vec<&arrow_schema::DataType> = global_indices
                .iter()
                .map(|&gi| &self.schema.columns[gi].data_type)
                .collect();
            let bw = self.bucket_writers[b].as_mut().unwrap();
            size += bw.write_columns(&arrays, &data_types)? as u64;
        }

        self.current_row_group_rows += batch.num_rows();
        self.current_buffered_size += size;

        if self.current_buffered_size >= self.row_group_max_size {
            self.flush_row_group()?;
        }
        Ok(())
    }

    fn flush_row_group(&mut self) -> io::Result<()> {
        if self.current_row_group_rows == 0 {
            return Ok(());
        }

        let mut bucket_offsets = vec![0u64; self.num_buckets];
        let mut bucket_layouts = vec![BucketLayout::Empty; self.num_buckets];

        let num_active = self.active_buckets.len();
        let mut actual_uncompressed_sizes = vec![0usize; self.num_buckets];
        for ai in 0..num_active {
            let b = self.active_buckets[ai];
            let bw = self.bucket_writers[b].as_ref().unwrap();
            if bw.is_empty() {
                continue;
            }
            let est_size = bw.estimated_raw_size();
            let try_paged =
                self.compression == COMPRESSION_ZSTD && est_size >= self.page_size_threshold;

            let paged_output = if try_paged {
                let paged = bw.finish_paged();
                let num_pages = paged.column_pages.iter().filter(|p| p.is_some()).count();
                let total: usize = paged
                    .column_pages
                    .iter()
                    .filter_map(|p| p.as_ref())
                    .map(|p| p.len())
                    .sum();
                let avg_ok = total
                    .checked_div(num_pages)
                    .is_some_and(|avg| avg >= self.page_size_threshold);
                if avg_ok {
                    Some(paged)
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(paged) = paged_output {
                let paged_raw_size: usize = paged
                    .column_pages
                    .iter()
                    .filter_map(|p| p.as_ref())
                    .map(|p| p.len())
                    .sum();
                let total_size = self.write_paged_bucket(&paged)?;
                bucket_layouts[b] = BucketLayout::Paged { total_size };
                actual_uncompressed_sizes[b] = paged_raw_size;
                bucket_offsets[b] = self.out.pos() - total_size as u64;
            } else {
                let raw = self.bucket_writers[b].as_ref().unwrap().finish();
                let comp_size = self.write_compressed(&raw)?;
                bucket_layouts[b] = BucketLayout::Monolithic {
                    compressed_size: comp_size,
                    uncompressed_size: raw.len(),
                };
                actual_uncompressed_sizes[b] = raw.len();
                bucket_offsets[b] = self.out.pos() - comp_size as u64;
            }
        }

        let rg_uncompressed: u64 = actual_uncompressed_sizes.iter().map(|&s| s as u64).sum();
        let rg_compressed: u64 = bucket_layouts
            .iter()
            .map(|l| {
                let (cs, _) = l.encode();
                cs as u64
            })
            .sum();
        self.total_uncompressed += rg_uncompressed;
        self.total_compressed += rg_compressed;
        if self.total_uncompressed > 0 {
            self.compression_ratio = self.total_compressed as f64 / self.total_uncompressed as f64;
        }

        for ai in 0..num_active {
            let b = self.active_buckets[ai];
            self.bucket_writers[b].as_mut().unwrap().reset();
        }

        self.row_group_metas.push(RowGroupMeta {
            num_rows: self.current_row_group_rows,
            bucket_offsets,
            bucket_layouts,
        });

        self.current_row_group_rows = 0;
        self.current_buffered_size = 0;
        Ok(())
    }

    fn write_compressed(&mut self, raw: &[u8]) -> io::Result<usize> {
        match self.compression {
            COMPRESSION_NONE => {
                self.out.write(raw)?;
                Ok(raw.len())
            }
            COMPRESSION_ZSTD => {
                let compressed =
                    zstd::bulk::compress(raw, self.zstd_level).map_err(io::Error::other)?;
                self.out.write(&compressed)?;
                Ok(compressed.len())
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Unsupported compression: {}", self.compression),
            )),
        }
    }

    fn write_paged_bucket(&mut self, paged: &PagedBucketOutput) -> io::Result<usize> {
        let num_columns = paged.encodings.len();

        // Build and compress each column's page_content independently.
        // page_content = [encoding(1B) | flags(1B) | meta | data]
        // On-disk slot = [uncompressed_size (varint) | zstd(page_content)]
        // ALL_NULL columns have no slot (directory entry = 0).
        let mut column_slots: Vec<Vec<u8>> = Vec::with_capacity(num_columns);
        for i in 0..num_columns {
            if paged.encodings[i] == ENCODING_ALL_NULL {
                column_slots.push(Vec::new());
                continue;
            }

            // Build page_content
            let mut page_content = Vec::new();
            page_content.push(paged.encodings[i]);
            let flags: u8 = if paged.has_nulls[i] { 1 } else { 0 };
            page_content.push(flags);

            // Meta + data depend on encoding
            match paged.encodings[i] {
                ENCODING_CONST => {
                    page_content.extend_from_slice(&paged.const_data[i]);
                    if let Some(ref page_data) = paged.column_pages[i] {
                        page_content.extend_from_slice(page_data);
                    }
                }
                _ => {
                    if let Some(ref page_data) = paged.column_pages[i] {
                        page_content.extend_from_slice(page_data);
                    }
                }
            }

            // Compress and build on-disk slot: uncompressed_size varint + compressed data
            let uncompressed_size = page_content.len();
            let compressed =
                zstd::bulk::compress(&page_content, self.zstd_level).map_err(io::Error::other)?;
            let mut slot = Vec::new();
            varint::encode(
                &mut slot,
                to_u32(uncompressed_size, "page uncompressed size")?,
            );
            slot.extend_from_slice(&compressed);
            column_slots.push(slot);
        }

        // Write fixed-length directory: num_columns * 4 bytes (u32 LE per column = slot size)
        let dir_size = num_columns * 4;
        let mut total_size = dir_size;
        for slot in &column_slots {
            let slot_size = to_u32(slot.len(), "paged slot size")?;
            self.out.write(&slot_size.to_le_bytes())?;
        }

        // Write column slots sequentially
        for slot in &column_slots {
            if !slot.is_empty() {
                self.out.write(slot)?;
                total_size += slot.len();
            }
        }

        Ok(total_size)
    }

    pub fn close(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;

        self.flush_row_group()?;

        // Write schema block
        let schema_raw = self.schema.serialize();
        let schema_block_offset = self.out.pos();

        let uncomp_size = to_u32(schema_raw.len(), "schema uncompressed size")?;
        self.out.write(&uncomp_size.to_be_bytes())?;

        match self.compression {
            COMPRESSION_NONE => {
                self.out.write(&schema_raw)?;
            }
            COMPRESSION_ZSTD => {
                let compressed =
                    zstd::bulk::compress(&schema_raw, self.zstd_level).map_err(io::Error::other)?;
                self.out.write(&compressed)?;
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Unsupported compression",
                ));
            }
        }

        // Write row group index (varint encoded, only non-empty buckets)
        let index_offset = self.out.pos();
        let num_row_groups = self.row_group_metas.len();

        let mut index_buf = Vec::with_capacity(num_row_groups * (5 + self.num_buckets * 25));
        for meta in &self.row_group_metas {
            varint::encode(&mut index_buf, to_u32(meta.num_rows, "row group num_rows")?);
            let non_empty = meta
                .bucket_layouts
                .iter()
                .filter(|l| !matches!(l, BucketLayout::Empty))
                .count();
            varint::encode(&mut index_buf, to_u32(non_empty, "non_empty bucket count")?);
            for b in 0..self.num_buckets {
                let (compressed_size, bulk_decompress_size) = meta.bucket_layouts[b].encode();
                if compressed_size > 0 {
                    varint::encode(&mut index_buf, to_u32(b, "bucket index")?);
                    index_buf.extend_from_slice(&meta.bucket_offsets[b].to_be_bytes());
                    varint::encode(
                        &mut index_buf,
                        to_u32(compressed_size, "bucket compressed_size")?,
                    );
                    varint::encode(
                        &mut index_buf,
                        to_u32(bulk_decompress_size, "bucket bulk_decompress_size")?,
                    );
                }
            }
        }
        self.out.write(&index_buf)?;

        // Write footer (32 bytes, big-endian)
        let mut footer = [0u8; FOOTER_SIZE];
        footer[0..8].copy_from_slice(&index_offset.to_be_bytes());
        footer[8..16].copy_from_slice(&schema_block_offset.to_be_bytes());
        footer[16..20].copy_from_slice(&to_u32(self.num_buckets, "num_buckets")?.to_be_bytes());
        footer[20..24].copy_from_slice(&to_u32(num_row_groups, "num_row_groups")?.to_be_bytes());
        footer[24] = self.compression;
        footer[25] = VERSION;
        footer[26..28].copy_from_slice(&[0, 0]);
        footer[28..32].copy_from_slice(&MAGIC);

        self.out.write(&footer)?;
        self.out.flush()?;
        Ok(())
    }
}

impl<S: OutputFile> Drop for MosaicWriter<S> {
    fn drop(&mut self) {
        if !self.closed {
            if let Err(e) = self.close() {
                eprintln!("MosaicWriter::drop: close failed: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

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

    #[test]
    fn test_write_simple_file() {
        let arrow_schema = Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int32, true),
            Field::new("score", DataType::Float64, true),
        ]);
        let out = MemOutputFile::new();
        let mut writer = MosaicWriter::new(
            out,
            &arrow_schema,
            WriterOptions {
                num_buckets: 2,
                compression: COMPRESSION_NONE,
                ..Default::default()
            },
        )
        .unwrap();

        let names: Vec<String> = (0..100).map(|i| format!("user_{}", i)).collect();
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let ages: Vec<i32> = (0..100).map(|i| 20 + (i % 50)).collect();
        let scores: Vec<f64> = (0..100).map(|i| i as f64 * 1.5).collect();

        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![
                Arc::new(StringArray::from(name_refs)),
                Arc::new(Int32Array::from(ages)),
                Arc::new(Float64Array::from(scores)),
            ],
        )
        .unwrap();
        writer.write_batch(&batch).unwrap();

        writer.close().unwrap();
        let data = &writer.out.buf;

        assert!(data.len() >= FOOTER_SIZE);
        let magic = &data[data.len() - 4..];
        assert_eq!(magic, &MAGIC);
        assert_eq!(data[data.len() - 7], VERSION);
    }

    #[test]
    fn test_write_with_zstd() {
        let arrow_schema = Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
        ]);
        let out = MemOutputFile::new();
        let mut writer = MosaicWriter::new(
            out,
            &arrow_schema,
            WriterOptions {
                num_buckets: 1,
                compression: COMPRESSION_ZSTD,
                zstd_level: 3,
                ..Default::default()
            },
        )
        .unwrap();

        let a_vals: Vec<i64> = (0..1000).collect();
        let b_vals: Vec<i64> = (0..1000).map(|i| i * 2).collect();
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![
                Arc::new(Int64Array::from(a_vals)),
                Arc::new(Int64Array::from(b_vals)),
            ],
        )
        .unwrap();
        writer.write_batch(&batch).unwrap();

        writer.close().unwrap();
        let magic = &writer.out.buf[writer.out.buf.len() - 4..];
        assert_eq!(magic, &MAGIC);
    }

    #[test]
    fn test_estimated_file_size() {
        let arrow_schema = Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Utf8, true),
        ]);
        let out = MemOutputFile::new();
        let mut writer = MosaicWriter::new(
            out,
            &arrow_schema,
            WriterOptions {
                num_buckets: 1,
                compression: COMPRESSION_ZSTD,
                zstd_level: 3,
                row_group_max_size: 4096,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(writer.estimated_file_size(), 1024);

        let arrow_schema = Arc::new(arrow_schema);

        let a_vals: Vec<i64> = (0..10).collect();
        let b_vals: Vec<String> = (0..10).map(|i| format!("val_{}", i)).collect();
        let b_refs: Vec<&str> = b_vals.iter().map(|s| s.as_str()).collect();
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int64Array::from(a_vals)),
                Arc::new(StringArray::from(b_refs)),
            ],
        )
        .unwrap();
        writer.write_batch(&batch).unwrap();
        let est_before_flush = writer.estimated_file_size();
        assert!(est_before_flush > 1024);

        let a_vals: Vec<i64> = (10..500).collect();
        let b_vals: Vec<String> = (10..500).map(|i| format!("val_{}", i)).collect();
        let b_refs: Vec<&str> = b_vals.iter().map(|s| s.as_str()).collect();
        let batch = RecordBatch::try_new(
            arrow_schema,
            vec![
                Arc::new(Int64Array::from(a_vals)),
                Arc::new(StringArray::from(b_refs)),
            ],
        )
        .unwrap();
        writer.write_batch(&batch).unwrap();
        assert!(!writer.row_group_metas.is_empty());
        assert!(writer.compression_ratio < 1.0);

        let est_after_flush = writer.estimated_file_size();
        assert!(est_after_flush > 0);

        writer.close().unwrap();
        let actual = writer.out.buf.len() as u64;
        assert!(actual > 0);
    }

    #[test]
    fn test_non_nullable_rejects_null() {
        let arrow_schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]);
        let out = MemOutputFile::new();
        let mut writer = MosaicWriter::new(out, &arrow_schema, WriterOptions::default()).unwrap();

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, true),
                Field::new("name", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![None, Some(1)])),
                Arc::new(StringArray::from(vec![Some("hello"), Some("world")])),
            ],
        )
        .unwrap();
        let result = writer.write_batch(&batch);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidInput);

        let batch2 = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("name", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec![None, Some("world")])),
            ],
        )
        .unwrap();
        let result = writer.write_batch(&batch2);
        assert!(result.is_ok());
    }
}
