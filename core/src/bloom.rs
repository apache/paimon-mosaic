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
use std::hash::Hasher;
use std::io;

use arrow_array::*;
use arrow_schema::DataType;

use crate::values::Value;
use crate::varint;
use twox_hash::XxHash64;

const SBBF_SALT: [u32; 8] = [
    0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d, 0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31,
];

pub const BLOOM_HEADER_VERSION: u8 = 1;
pub const BLOOM_ALGORITHM_SBBF: u8 = 1;
pub const BLOOM_HASH_XXHASH64: u8 = 1;
pub const BLOOM_COMPRESSION_NONE: u8 = 0;

pub const BLOOM_BLOCK_BYTES: usize = 32;
pub const BLOOM_BLOCK_WORDS: usize = 8;

pub const BLOOM_MIN_BLOCKS: usize = 1;
// Hard cap to keep one filter under ~512 MiB even on hostile configs.
pub const BLOOM_MAX_BLOCKS: usize = 1 << 24;

#[derive(Debug, Clone)]
pub struct BloomFilterConfig {
    pub column_name: String,
    pub ndv: u64,
    pub fpp: f64,
}

#[derive(Debug, Clone)]
pub struct SplitBlockBloomFilter {
    blocks: Vec<[u32; BLOOM_BLOCK_WORDS]>,
}

impl SplitBlockBloomFilter {
    pub fn with_num_blocks(num_blocks: usize) -> Self {
        let n = num_blocks
            .clamp(BLOOM_MIN_BLOCKS, BLOOM_MAX_BLOCKS)
            .next_power_of_two();
        Self {
            blocks: vec![[0u32; BLOOM_BLOCK_WORDS]; n],
        }
    }

    pub fn with_capacity(ndv: u64, fpp: f64) -> Self {
        Self::with_num_blocks(num_blocks_for(ndv, fpp))
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn bitset_num_bytes(&self) -> usize {
        self.blocks.len() * BLOOM_BLOCK_BYTES
    }

    #[inline]
    pub fn insert_hash(&mut self, h: u64) {
        let idx = block_index(h, self.blocks.len());
        let block = &mut self.blocks[idx];
        let mask = block_mask(h as u32);
        for i in 0..BLOOM_BLOCK_WORDS {
            block[i] |= mask[i];
        }
    }

    #[inline]
    pub fn contains_hash(&self, h: u64) -> bool {
        let idx = block_index(h, self.blocks.len());
        let block = &self.blocks[idx];
        let mask = block_mask(h as u32);
        for i in 0..BLOOM_BLOCK_WORDS {
            if (block[i] & mask[i]) != mask[i] {
                return false;
            }
        }
        true
    }

    pub fn write_to(&self, buf: &mut Vec<u8>) {
        buf.push(BLOOM_HEADER_VERSION);
        buf.push(BLOOM_ALGORITHM_SBBF);
        buf.push(BLOOM_HASH_XXHASH64);
        buf.push(BLOOM_COMPRESSION_NONE);
        let num_bytes = self.bitset_num_bytes() as u32;
        varint::encode(buf, num_bytes);
        for block in &self.blocks {
            for word in block {
                buf.extend_from_slice(&word.to_le_bytes());
            }
        }
    }

    pub fn read_from(data: &[u8]) -> io::Result<Self> {
        if data.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "bloom: header truncated",
            ));
        }
        let header_version = data[0];
        let algorithm = data[1];
        let hash = data[2];
        let compression = data[3];
        if header_version != BLOOM_HEADER_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bloom: unsupported header version {}", header_version),
            ));
        }
        if algorithm != BLOOM_ALGORITHM_SBBF {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bloom: unsupported algorithm {}", algorithm),
            ));
        }
        if hash != BLOOM_HASH_XXHASH64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bloom: unsupported hash {}", hash),
            ));
        }
        if compression != BLOOM_COMPRESSION_NONE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bloom: unsupported compression {}", compression),
            ));
        }
        let mut pos = 4;
        let num_bytes = varint::decode(data, &mut pos)? as usize;
        if !num_bytes.is_multiple_of(BLOOM_BLOCK_BYTES) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "bloom: numBytes {} not a multiple of block size {}",
                    num_bytes, BLOOM_BLOCK_BYTES
                ),
            ));
        }
        let num_blocks = num_bytes / BLOOM_BLOCK_BYTES;
        if num_blocks == 0 || !num_blocks.is_power_of_two() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "bloom: numBlocks {} not a positive power of two",
                    num_blocks
                ),
            ));
        }
        if num_blocks > BLOOM_MAX_BLOCKS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "bloom: numBlocks {} exceeds cap {}",
                    num_blocks, BLOOM_MAX_BLOCKS
                ),
            ));
        }
        if pos + num_bytes > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "bloom: bitset truncated",
            ));
        }
        let mut blocks = Vec::with_capacity(num_blocks);
        for b in 0..num_blocks {
            let mut block = [0u32; BLOOM_BLOCK_WORDS];
            for (i, word) in block.iter_mut().enumerate() {
                let off = pos + b * BLOOM_BLOCK_BYTES + i * 4;
                *word = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            }
            blocks.push(block);
        }
        Ok(Self { blocks })
    }
}

#[inline]
fn block_index(h: u64, num_blocks: usize) -> usize {
    let top = h >> 32;
    let z = num_blocks as u64;
    ((top.wrapping_mul(z)) >> 32) as usize
}

#[inline]
fn block_mask(x: u32) -> [u32; BLOOM_BLOCK_WORDS] {
    let mut mask = [0u32; BLOOM_BLOCK_WORDS];
    for i in 0..BLOOM_BLOCK_WORDS {
        let y = x.wrapping_mul(SBBF_SALT[i]);
        let bit = y >> 27;
        mask[i] = 1u32 << bit;
    }
    mask
}

fn bits_per_insert_for_fpp(fpp: f64) -> f64 {
    if fpp >= 0.10 {
        6.0
    } else if fpp >= 0.01 {
        10.5
    } else if fpp >= 0.001 {
        16.9
    } else if fpp >= 0.0001 {
        26.4
    } else {
        41.0
    }
}

pub fn num_blocks_for(ndv: u64, fpp: f64) -> usize {
    let bits_per_insert = bits_per_insert_for_fpp(fpp);
    let ndv_for_calc = ndv.max(1);
    let total_bits = ((ndv_for_calc as f64) * bits_per_insert).ceil() as u64;
    let blocks = total_bits.div_ceil(256) as usize;
    blocks.max(BLOOM_MIN_BLOCKS).next_power_of_two()
}

pub fn hash_value(value: &Value) -> u64 {
    let mut h = XxHash64::with_seed(0);
    write_canonical_bytes(value, &mut h);
    h.finish()
}

fn write_canonical_bytes(value: &Value, h: &mut XxHash64) {
    match value {
        Value::Boolean(b) => h.write(&[*b as u8]),
        Value::TinyInt(x) => h.write(&x.to_le_bytes()),
        Value::SmallInt(x) => h.write(&x.to_le_bytes()),
        Value::Integer(x) => h.write(&x.to_le_bytes()),
        Value::BigInt(x) => h.write(&x.to_le_bytes()),
        Value::Float(x) => h.write(&x.to_le_bytes()),
        Value::Double(x) => h.write(&x.to_le_bytes()),
        Value::Date(x) => h.write(&x.to_le_bytes()),
        Value::String(bytes) => h.write(bytes),
        _ => {}
    }
}

pub fn supports_bloom(dt: &DataType) -> bool {
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
            | DataType::Utf8
    )
}

struct ColEntry {
    column_index: usize,
    batch_col_index: usize,
    data_type: DataType,
    ndv: u64,
    fpp: f64,
    filter: SplitBlockBloomFilter,
}

pub type ResolvedBloomColumn = (usize, usize, DataType, u64, f64);

pub struct BloomFilterCollector {
    entries: Vec<ColEntry>,
}

impl BloomFilterCollector {
    pub fn new(columns: &[ResolvedBloomColumn]) -> Self {
        let entries = columns
            .iter()
            .map(|(idx, batch_idx, dt, ndv, fpp)| ColEntry {
                column_index: *idx,
                batch_col_index: *batch_idx,
                data_type: dt.clone(),
                ndv: *ndv,
                fpp: *fpp,
                filter: SplitBlockBloomFilter::with_capacity(*ndv, *fpp),
            })
            .collect();
        Self { entries }
    }

    pub fn update_batch(&mut self, batch: &RecordBatch) {
        for entry in &mut self.entries {
            let array = batch.column(entry.batch_col_index).as_ref();
            insert_array_hashes(array, &entry.data_type, &mut entry.filter);
        }
    }

    pub fn finish(&mut self) -> Vec<(usize, SplitBlockBloomFilter)> {
        let mut out = Vec::with_capacity(self.entries.len());
        for entry in &mut self.entries {
            let fresh = SplitBlockBloomFilter::with_capacity(entry.ndv, entry.fpp);
            let prev = std::mem::replace(&mut entry.filter, fresh);
            out.push((entry.column_index, prev));
        }
        out
    }
}

fn insert_array_hashes(array: &dyn Array, dt: &DataType, filter: &mut SplitBlockBloomFilter) {
    let len = array.len();
    for row in 0..len {
        if array.is_null(row) {
            continue;
        }
        if let Some(v) = extract_value(array, row, dt) {
            filter.insert_hash(hash_value(&v));
        }
    }
}

fn extract_value(array: &dyn Array, row: usize, dt: &DataType) -> Option<Value> {
    match dt {
        DataType::Boolean => array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .map(|a| Value::Boolean(a.value(row))),
        DataType::Int8 => array
            .as_any()
            .downcast_ref::<Int8Array>()
            .map(|a| Value::TinyInt(a.value(row))),
        DataType::Int16 => array
            .as_any()
            .downcast_ref::<Int16Array>()
            .map(|a| Value::SmallInt(a.value(row))),
        DataType::Int32 => array
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|a| Value::Integer(a.value(row))),
        DataType::Int64 => array
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| Value::BigInt(a.value(row))),
        DataType::Float32 => array
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|a| Value::Float(a.value(row))),
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|a| Value::Double(a.value(row))),
        DataType::Date32 => array
            .as_any()
            .downcast_ref::<Date32Array>()
            .map(|a| Value::Date(a.value(row))),
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|a| Value::String(a.value(row).as_bytes().to_vec())),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct BloomEntryMeta {
    pub column_index: usize,
    pub offset: u64,
    pub total_bytes: usize,
}

pub fn serialize_index_tail(blooms: &[BloomEntryMeta]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 + blooms.len() * 12);
    varint::encode(&mut buf, blooms.len() as u32);
    for b in blooms {
        varint::encode(&mut buf, b.column_index as u32);
        buf.extend_from_slice(&b.offset.to_be_bytes());
        varint::encode(&mut buf, b.total_bytes as u32);
    }
    buf
}

pub fn deserialize_index_tail(data: &[u8], pos: &mut usize) -> io::Result<Vec<BloomEntryMeta>> {
    let num = varint::decode(data, pos)? as usize;
    let mut out = Vec::with_capacity(num);
    for _ in 0..num {
        let column_index = varint::decode(data, pos)? as usize;
        if *pos + 8 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "bloom index tail: offset truncated",
            ));
        }
        let offset = u64::from_be_bytes(data[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        let total_bytes = varint::decode(data, pos)? as usize;
        out.push(BloomEntryMeta {
            column_index,
            offset,
            total_bytes,
        });
    }
    Ok(out)
}

// Convenience helper: validate user-supplied configs against the schema,
// returning the per-collector tuples used by BloomFilterCollector::new.
// Bubbles up clear errors for missing names or unsupported types.
pub fn resolve_configs(
    configs: &[BloomFilterConfig],
    schema_columns: &[crate::schema::ColumnMeta],
    batch_col_map: &[usize],
) -> io::Result<Vec<ResolvedBloomColumn>> {
    let mut name_to_idx: HashMap<&str, usize> = HashMap::with_capacity(schema_columns.len());
    for (i, c) in schema_columns.iter().enumerate() {
        name_to_idx.insert(c.name.as_str(), i);
    }
    let mut seen: HashMap<usize, ()> = HashMap::new();
    let mut out = Vec::with_capacity(configs.len());
    for c in configs {
        let idx = *name_to_idx.get(c.column_name.as_str()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "bloom_filter_columns: column '{}' not found in schema",
                    c.column_name
                ),
            )
        })?;
        if seen.insert(idx, ()).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "bloom_filter_columns: duplicate entry for column '{}'",
                    c.column_name
                ),
            ));
        }
        let dt = &schema_columns[idx].data_type;
        if !supports_bloom(dt) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "bloom_filter_columns: column '{}' has unsupported type {:?} for bloom filter",
                    c.column_name, dt
                ),
            ));
        }
        if !(0.0 < c.fpp && c.fpp < 1.0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "bloom_filter_columns: column '{}' fpp must be in (0, 1), got {}",
                    c.column_name, c.fpp
                ),
            ));
        }
        out.push((idx, batch_col_map[idx], dt.clone(), c.ndv, c.fpp));
    }
    out.sort_by_key(|(idx, _, _, _, _)| *idx);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_mask_sets_eight_bits() {
        let m = block_mask(0xdeadbeef);
        let total_bits: u32 = m.iter().map(|w| w.count_ones()).sum();
        assert_eq!(total_bits, 8);
    }

    #[test]
    fn insert_then_contains() {
        let mut f = SplitBlockBloomFilter::with_num_blocks(64);
        for i in 0..1000u64 {
            f.insert_hash(hash_value(&Value::BigInt(i as i64)));
        }
        for i in 0..1000u64 {
            assert!(f.contains_hash(hash_value(&Value::BigInt(i as i64))));
        }
    }

    #[test]
    fn fpp_within_expectation() {
        let n = 10_000u64;
        let target_fpp = 0.01;
        let mut f = SplitBlockBloomFilter::with_capacity(n, target_fpp);
        for i in 0..n {
            f.insert_hash(hash_value(&Value::BigInt(i as i64)));
        }
        let probe = 200_000u64;
        let mut fp = 0u64;
        for i in n..(n + probe) {
            if f.contains_hash(hash_value(&Value::BigInt(i as i64))) {
                fp += 1;
            }
        }
        let observed = fp as f64 / probe as f64;
        assert!(
            observed < target_fpp * 3.0,
            "observed fpp {} exceeded 3x target {}",
            observed,
            target_fpp
        );
    }

    #[test]
    fn write_read_roundtrip() {
        let mut f = SplitBlockBloomFilter::with_num_blocks(8);
        for i in 0..500u64 {
            f.insert_hash(hash_value(&Value::BigInt(i as i64)));
        }
        let mut buf = Vec::new();
        f.write_to(&mut buf);
        let g = SplitBlockBloomFilter::read_from(&buf).unwrap();
        assert_eq!(g.num_blocks(), f.num_blocks());
        for i in 0..500u64 {
            assert!(g.contains_hash(hash_value(&Value::BigInt(i as i64))));
        }
    }

    #[test]
    fn read_rejects_bad_discriminants() {
        let mut buf = vec![
            BLOOM_HEADER_VERSION,
            99,
            BLOOM_HASH_XXHASH64,
            BLOOM_COMPRESSION_NONE,
        ];
        varint::encode(&mut buf, 32);
        buf.extend(vec![0u8; 32]);
        let err = SplitBlockBloomFilter::read_from(&buf).unwrap_err();
        assert!(format!("{}", err).contains("unsupported algorithm"));
    }

    #[test]
    fn num_blocks_for_known_sizes() {
        assert!(num_blocks_for(0, 0.01) >= 1);
        assert!(num_blocks_for(1_000_000, 0.01) > num_blocks_for(1_000, 0.01));
        assert!(num_blocks_for(1_000, 0.0001) > num_blocks_for(1_000, 0.01));
    }

    #[test]
    fn index_tail_roundtrip() {
        let blooms = vec![
            BloomEntryMeta {
                column_index: 1,
                offset: 1024,
                total_bytes: 200,
            },
            BloomEntryMeta {
                column_index: 4,
                offset: 1224,
                total_bytes: 500,
            },
        ];
        let buf = serialize_index_tail(&blooms);
        let mut pos = 0;
        let got = deserialize_index_tail(&buf, &mut pos).unwrap();
        assert_eq!(pos, buf.len());
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].column_index, 1);
        assert_eq!(got[0].offset, 1024);
        assert_eq!(got[0].total_bytes, 200);
        assert_eq!(got[1].column_index, 4);
    }
}
