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

use std::io::{self, Write};

// A late outlier may widen this block, never the entire buffered column.
const BLOCK_VALUES: usize = 4096;
const OUTPUT_BYTES: usize = 16 * 1024;

#[derive(Default)]
struct Block {
    bytes: Vec<u8>,
    width: usize,
}

/// In-memory representation only. Keys retain the original integer's big-endian
/// bit pattern; negative values use the logical width, positive values may use
/// 1/2/4 bytes. The file schema, dictionary keys and encoded bytes never change.
pub(crate) struct IntegerBuffer {
    blocks: Vec<Block>,
    logical_width: usize,
    len: usize,
}

impl IntegerBuffer {
    pub(crate) fn new(logical_width: usize) -> Self {
        debug_assert!(matches!(logical_width, 2 | 4 | 8));
        Self {
            blocks: Vec::new(),
            logical_width,
            len: 0,
        }
    }

    #[cfg(test)]
    fn push(&mut self, value: i64) -> u64 {
        self.extend_from_slice(&[value]);
        (value as u64) & (u64::MAX >> (64 - self.logical_width * 8))
    }

    pub(crate) fn extend_from_slice<T: Copy + Into<i64>>(&mut self, mut values: &[T]) {
        let mask = u64::MAX >> (64 - self.logical_width * 8);
        while !values.is_empty() {
            let count = values.len().min(BLOCK_VALUES - self.len % BLOCK_VALUES);
            let input = &values[..count];
            // A block-sized range check is vectorizable, unlike a width branch
            // on every value in the copy loop. Negative keys require full width.
            let bits = input.iter().fold(0u64, |bits, &v| bits | v.into() as u64) & mask;
            let width = if bits <= u8::MAX as u64 {
                1
            } else if bits <= u16::MAX as u64 {
                2
            } else if bits <= u32::MAX as u64 {
                4
            } else {
                8
            };
            if self.len.is_multiple_of(BLOCK_VALUES) {
                self.blocks.push(Block {
                    bytes: Vec::new(),
                    width,
                });
            }
            let block = self.blocks.last_mut().unwrap();
            if width > block.width {
                let old_width = block.width;
                let old_count = block.bytes.len() / old_width;
                block.bytes.resize(old_count * width, 0);
                for i in (0..old_count).rev() {
                    let key = read_key(&block.bytes[i * old_width..(i + 1) * old_width]);
                    block.bytes[i * width..(i + 1) * width]
                        .copy_from_slice(&key.to_be_bytes()[8 - width..]);
                }
                block.width = width;
            }
            let offset = block.bytes.len();
            block.bytes.resize(offset + count * block.width, 0);
            let output = &mut block.bytes[offset..];
            match block.width {
                1 => {
                    for (out, &v) in output.iter_mut().zip(input) {
                        *out = v.into() as u8;
                    }
                }
                2 => {
                    for (out, &v) in output.chunks_exact_mut(2).zip(input) {
                        out.copy_from_slice(&(v.into() as u16).to_be_bytes());
                    }
                }
                4 => {
                    for (out, &v) in output.chunks_exact_mut(4).zip(input) {
                        out.copy_from_slice(&(v.into() as u32).to_be_bytes());
                    }
                }
                8 => {
                    for (out, &v) in output.chunks_exact_mut(8).zip(input) {
                        out.copy_from_slice(&v.into().to_be_bytes());
                    }
                }
                _ => unreachable!(),
            }
            self.len += count;
            values = &values[count..];
        }
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = u64> + '_ {
        self.blocks
            .iter()
            .flat_map(|block| block.bytes.chunks_exact(block.width).map(read_key))
    }

    pub(crate) fn logical_len(&self) -> usize {
        self.len * self.logical_width
    }

    pub(crate) fn write_to(&self, out: &mut impl Write) -> io::Result<()> {
        let mut scratch = [0; OUTPUT_BYTES];
        for block in &self.blocks {
            match (block.width, self.logical_width) {
                (src, dst) if src == dst => {
                    for chunk in block.bytes.chunks(OUTPUT_BYTES) {
                        out.write_all(chunk)?;
                    }
                }
                (1, 2) => widen::<1, 2>(&block.bytes, &mut scratch, out)?,
                (1, 4) => widen::<1, 4>(&block.bytes, &mut scratch, out)?,
                (1, 8) => widen::<1, 8>(&block.bytes, &mut scratch, out)?,
                (2, 4) => widen::<2, 4>(&block.bytes, &mut scratch, out)?,
                (2, 8) => widen::<2, 8>(&block.bytes, &mut scratch, out)?,
                (4, 8) => widen::<4, 8>(&block.bytes, &mut scratch, out)?,
                _ => unreachable!(),
            }
        }
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        // Do not retain a wide allocation after an outlier or a large row group.
        self.blocks.clear();
        self.len = 0;
    }

    #[cfg(test)]
    pub(crate) fn allocated_bytes(&self) -> usize {
        self.blocks.capacity() * std::mem::size_of::<Block>()
            + self
                .blocks
                .iter()
                .map(|b| b.bytes.capacity())
                .sum::<usize>()
    }
}

fn read_key(bytes: &[u8]) -> u64 {
    match bytes.len() {
        1 => bytes[0] as u64,
        2 => u16::from_be_bytes(bytes.try_into().unwrap()) as u64,
        4 => u32::from_be_bytes(bytes.try_into().unwrap()) as u64,
        8 => u64::from_be_bytes(bytes.try_into().unwrap()),
        _ => unreachable!(),
    }
}

fn widen<const SRC: usize, const DST: usize>(
    bytes: &[u8],
    scratch: &mut [u8; OUTPUT_BYTES],
    out: &mut impl Write,
) -> io::Result<()> {
    for input in bytes.chunks(OUTPUT_BYTES / DST * SRC) {
        let output = &mut scratch[..input.len() / SRC * DST];
        for (src, dst) in input.chunks_exact(SRC).zip(output.chunks_exact_mut(DST)) {
            // Clear the prefix: the previous block may have used a wider
            // compact representation in the same reusable scratch space.
            dst[..DST - SRC].fill(0);
            dst[DST - SRC..].copy_from_slice(src);
        }
        out.write_all(output)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsigned_ranges_negative_values_and_late_outliers() {
        for width in [2, 4, 8] {
            let max = i64::MAX >> (64 - width * 8);
            let min = -max - 1;
            let mut buffer = IntegerBuffer::new(width);
            let mut expected = Vec::new();
            for value in (0..BLOCK_VALUES * 3)
                .map(|i| (i % 256) as i64)
                .chain([max, min, -1, 0, 127, 128, 255, 256])
            {
                buffer.push(value);
                expected.extend_from_slice(&value.to_be_bytes()[8 - width..]);
            }
            assert_eq!(buffer.blocks[0].width, 1);
            assert_eq!(buffer.blocks[1].width, 1);
            assert_eq!(buffer.blocks.last().unwrap().width, width);
            let mut actual = Vec::new();
            buffer.write_to(&mut actual).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(buffer.logical_len(), expected.len());
        }
    }

    #[test]
    fn promotions_within_a_block_preserve_previous_values() {
        let values = [0, 255, 256, 65535, 65536, u32::MAX as i64, i64::MAX, -1];
        let mut buffer = IntegerBuffer::new(8);
        for value in values {
            buffer.push(value);
        }
        assert_eq!(buffer.keys().collect::<Vec<_>>(), values.map(|v| v as u64));
        buffer.clear();
        buffer.push(255);
        assert_eq!(buffer.blocks[0].width, 1);
        assert_eq!(buffer.keys().collect::<Vec<_>>(), vec![255]);
    }

    #[test]
    fn uint16_values_use_half_the_int32_payload_plus_bounded_metadata() {
        let mut buffer = IntegerBuffer::new(4);
        for i in 0..1_000_000 {
            buffer.push((i % 65536) as i64);
        }
        assert_eq!(buffer.logical_len(), 4_000_000);
        assert!(buffer.allocated_bytes() < 2_100_000);
    }

    #[test]
    fn widening_output_is_bounded_and_propagates_sink_errors() {
        struct Sink {
            bytes: usize,
            fail_after: usize,
        }
        impl Write for Sink {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                assert!(bytes.len() <= OUTPUT_BYTES);
                if self.bytes >= self.fail_after {
                    return Err(io::Error::other("sentinel"));
                }
                self.bytes += bytes.len();
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut buffer = IntegerBuffer::new(8);
        for i in 0..100_003 {
            buffer.push(i);
        }
        let mut sink = Sink {
            bytes: 0,
            fail_after: usize::MAX,
        };
        buffer.write_to(&mut sink).unwrap();
        assert_eq!(sink.bytes, 800_024);
        let mut sink = Sink {
            bytes: 0,
            fail_after: OUTPUT_BYTES,
        };
        assert_eq!(
            buffer.write_to(&mut sink).unwrap_err().to_string(),
            "sentinel"
        );
    }
}
