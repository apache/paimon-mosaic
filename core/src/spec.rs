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

pub const MAGIC: [u8; 4] = [b'M', b'O', b'S', b'A'];
pub const VERSION: u8 = 2;
pub const VERSION_MIN_SUPPORTED: u8 = 1;
pub const FOOTER_SIZE: usize = 32;

pub const BLOOM_DEFAULT_FPP: f64 = 0.01;

pub const COMPRESSION_NONE: u8 = 0;
pub const COMPRESSION_ZSTD: u8 = 1;

pub const ENCODING_PLAIN: u8 = 0;
pub const ENCODING_CONST: u8 = 1;
pub const ENCODING_DICT: u8 = 2;
pub const ENCODING_ALL_NULL: u8 = 3;

pub const DEFAULT_NUM_BUCKETS: usize = 100;
pub const DEFAULT_ROW_GROUP_MAX_SIZE: u64 = 256 * 1024 * 1024;
pub const DEFAULT_ZSTD_LEVEL: i32 = 1;
pub const DEFAULT_DICT_MAX_TOTAL_BYTES: usize = 32 * 1024;
pub const DEFAULT_DICT_MAX_ENTRIES: usize = 255;
pub const DEFAULT_PAGE_SIZE_THRESHOLD: usize = 32 * 1024;

// ======================== Bucket Layout Sentinel ========================
//
// Each bucket in the row group index is described by (compressed_size, bulk_decompress_size):
//
//   compressed_size == 0                            → Empty bucket. No data on disk; skip.
//   compressed_size > 0 && bulk_decompress_size > 0 → Monolithic bucket. The on-disk blob
//                                                     is a single compressed block;
//                                                     bulk_decompress_size is the decompressed size.
//   compressed_size > 0 && bulk_decompress_size == 0 → Paged bucket. The on-disk content is
//                                                     [directory (num_cols × u32le slot sizes)]
//                                                     followed by per-column compressed slots.
//
// This encoding is unambiguous: a non-empty monolithic bucket always has
// bulk_decompress_size > 0 (decompressed payload cannot be zero bytes).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketLayout {
    Empty,
    Monolithic {
        compressed_size: usize,
        uncompressed_size: usize,
    },
    Paged {
        total_size: usize,
    },
}

impl BucketLayout {
    pub fn decode(
        compressed_size: usize,
        bulk_decompress_size: usize,
    ) -> Result<Self, &'static str> {
        match (compressed_size, bulk_decompress_size) {
            (0, 0) => Ok(BucketLayout::Empty),
            (0, _) => {
                Err("invalid bucket layout: compressed_size == 0 but bulk_decompress_size != 0")
            }
            (cs, 0) => Ok(BucketLayout::Paged { total_size: cs }),
            (cs, us) => Ok(BucketLayout::Monolithic {
                compressed_size: cs,
                uncompressed_size: us,
            }),
        }
    }

    pub fn encode(&self) -> (usize, usize) {
        match *self {
            BucketLayout::Empty => (0, 0),
            BucketLayout::Monolithic {
                compressed_size,
                uncompressed_size,
            } => (compressed_size, uncompressed_size),
            BucketLayout::Paged { total_size } => (total_size, 0),
        }
    }
}

pub fn assign_bucket(sorted_position: usize, num_columns: usize, num_buckets: usize) -> usize {
    sorted_position * num_buckets / num_columns
}
