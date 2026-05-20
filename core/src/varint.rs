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

pub fn encode(buf: &mut Vec<u8>, mut value: u32) {
    loop {
        if (value & !0x7F) == 0 {
            buf.push(value as u8);
            return;
        }
        buf.push(((value & 0x7F) | 0x80) as u8);
        value >>= 7;
    }
}

pub fn encode_to_slice(buf: &mut [u8], pos: usize, mut value: u32) -> usize {
    let mut p = pos;
    loop {
        if (value & !0x7F) == 0 {
            buf[p] = value as u8;
            p += 1;
            return p;
        }
        buf[p] = ((value & 0x7F) | 0x80) as u8;
        p += 1;
        value >>= 7;
    }
}

pub fn encoded_size(mut value: u32) -> usize {
    let mut size = 1;
    while (value & !0x7F) != 0 {
        size += 1;
        value >>= 7;
    }
    size
}

pub fn decode(buf: &[u8], pos: &mut usize) -> Result<u32, std::io::Error> {
    let mut value: u32 = 0;
    let mut shift = 0;
    loop {
        if *pos >= buf.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "varint: unexpected end of data",
            ));
        }
        let b = buf[*pos] as u32;
        *pos += 1;
        value |= (b & 0x7F) << shift;
        if (b & 0x80) == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 35 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "varint: overflow (>5 bytes for u32)",
            ));
        }
    }
}

pub fn encode_u64(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        if (value & !0x7F) == 0 {
            buf.push(value as u8);
            return;
        }
        buf.push(((value & 0x7F) | 0x80) as u8);
        value >>= 7;
    }
}

pub fn decode_u64(buf: &[u8], pos: &mut usize) -> Result<u64, std::io::Error> {
    let mut value: u64 = 0;
    let mut shift = 0;
    loop {
        if *pos >= buf.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "varint: unexpected end of data",
            ));
        }
        let b = buf[*pos] as u64;
        *pos += 1;
        value |= (b & 0x7F) << shift;
        if (b & 0x80) == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 70 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "varint: overflow (>10 bytes for u64)",
            ));
        }
    }
}

pub fn encode_zigzag(buf: &mut Vec<u8>, value: i64) {
    let encoded = ((value << 1) ^ (value >> 63)) as u64;
    encode_u64(buf, encoded);
}

pub fn decode_zigzag(buf: &[u8], pos: &mut usize) -> Result<i64, std::io::Error> {
    let encoded = decode_u64(buf, pos)?;
    Ok(((encoded >> 1) as i64) ^ -((encoded & 1) as i64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        for &v in &[0u32, 1, 127, 128, 255, 16383, 16384, 2097151, u32::MAX] {
            let mut buf = Vec::new();
            encode(&mut buf, v);
            assert_eq!(buf.len(), encoded_size(v));
            let mut pos = 0;
            assert_eq!(decode(&buf, &mut pos).unwrap(), v);
            assert_eq!(pos, buf.len());
        }
    }

    #[test]
    fn test_u64_roundtrip() {
        for &v in &[
            0u64,
            1,
            127,
            128,
            255,
            16383,
            16384,
            u32::MAX as u64,
            u64::MAX,
        ] {
            let mut buf = Vec::new();
            encode_u64(&mut buf, v);
            let mut pos = 0;
            assert_eq!(decode_u64(&buf, &mut pos).unwrap(), v);
            assert_eq!(pos, buf.len());
        }
    }

    #[test]
    fn test_zigzag_roundtrip() {
        for &v in &[
            0i64,
            1,
            -1,
            2,
            -2,
            127,
            -128,
            10000,
            -10000,
            i64::MAX,
            i64::MIN,
        ] {
            let mut buf = Vec::new();
            encode_zigzag(&mut buf, v);
            let mut pos = 0;
            assert_eq!(decode_zigzag(&buf, &mut pos).unwrap(), v);
            assert_eq!(pos, buf.len());
        }
    }

    #[test]
    fn test_decode_eof() {
        let mut pos = 0;
        assert!(decode(&[], &mut pos).is_err());
        let truncated = vec![0x80];
        pos = 0;
        assert!(decode(&truncated, &mut pos).is_err());
    }
}
