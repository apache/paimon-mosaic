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

use std::collections::HashSet;
use std::io;

use arrow_schema::{DataType, Field, Schema};

use crate::bpe;
use crate::spec;
use crate::types;
use crate::varint;

const NAME_ENCODING_FRONT_CODE: u8 = 0;
const NAME_ENCODING_BPE: u8 = 1;
const SCHEMA_LAYOUT_V2: u8 = 2;
type BpeRules = Vec<[u8; 2]>;

#[derive(Debug, Clone)]
pub struct ColumnMeta {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub bucket_id: usize,
}

#[derive(Debug, Clone)]
pub struct MosaicSchema {
    pub num_buckets: usize,
    pub columns: Vec<ColumnMeta>,
    /// bucket_to_global[bucket_id] = [global_col_indices...] in name-sorted order
    pub bucket_to_global: Vec<Vec<usize>>,
    /// Column indices used as default output order when no projection is set.
    pub original_order: Vec<usize>,
}

impl MosaicSchema {
    pub fn validate(columns: &[(String, DataType, bool)]) -> Result<(), String> {
        let mut seen = HashSet::new();
        for (name, data_type, _nullable) in columns {
            if !seen.insert(name.as_str()) {
                return Err(format!("duplicate column name: {}", name));
            }
            types::validate_data_type(data_type)?;
        }
        Ok(())
    }

    pub fn from_arrow(schema: &Schema, num_buckets: usize) -> Result<Self, String> {
        let columns: Vec<(String, DataType, bool)> = schema
            .fields()
            .iter()
            .map(|f| (f.name().clone(), f.data_type().clone(), f.is_nullable()))
            .collect();
        Self::validate(&columns)?;
        Ok(Self::new(columns, num_buckets))
    }

    pub fn new(columns: Vec<(String, DataType, bool)>, num_buckets: usize) -> Self {
        let num_columns = columns.len();
        let actual_buckets = num_buckets.min(num_columns).max(1);

        let mut sorted_indices: Vec<usize> = (0..num_columns).collect();
        sorted_indices.sort_by(|&a, &b| columns[a].0.cmp(&columns[b].0));

        let mut bucket_to_global = vec![Vec::new(); actual_buckets];
        let mut cols: Vec<ColumnMeta> = Vec::with_capacity(num_columns);

        for (sorted_pos, &input_idx) in sorted_indices.iter().enumerate() {
            let bucket_id = spec::assign_bucket(sorted_pos, num_columns, actual_buckets);
            cols.push(ColumnMeta {
                name: columns[input_idx].0.clone(),
                data_type: columns[input_idx].1.clone(),
                nullable: columns[input_idx].2,
                bucket_id,
            });
            bucket_to_global[bucket_id].push(sorted_pos);
        }

        let mut original_order = vec![0usize; num_columns];
        for (sorted_pos, &input_idx) in sorted_indices.iter().enumerate() {
            original_order[input_idx] = sorted_pos;
        }

        MosaicSchema {
            num_buckets: actual_buckets,
            columns: cols,
            bucket_to_global,
            original_order,
        }
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        if has_current_layout_marker(data)? {
            Self::deserialize_current(data)
        } else {
            Self::deserialize_legacy(data)
        }
    }

    pub(crate) fn deserialize_for_version(version: u8, data: &[u8]) -> io::Result<Self> {
        match version {
            spec::LEGACY_VERSION => Self::deserialize_legacy(data),
            spec::VERSION => Self::deserialize_current(data),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version: {}", version),
            )),
        }
    }

    fn deserialize_current(data: &[u8]) -> io::Result<Self> {
        let mut pos = 0;
        let (num_columns, num_buckets, bpe_rules) = read_header(data, &mut pos, true)?;

        let mut columns = Vec::with_capacity(num_columns);
        let mut bucket_to_global = vec![Vec::new(); num_buckets];
        let mut name_decoder = NameDecoder::new(bpe_rules);
        let mut seen_names = std::collections::HashSet::with_capacity(num_columns);

        for sorted_pos in 0..num_columns {
            let name = name_decoder.read(data, &mut pos)?;
            validate_column_name(&name, &mut seen_names)?;

            let field = types::deserialize_field(&name, data, &mut pos)?;

            let bucket_id = spec::assign_bucket(sorted_pos, num_columns, num_buckets);
            columns.push(ColumnMeta {
                name,
                data_type: field.data_type().clone(),
                nullable: field.is_nullable(),
                bucket_id,
            });
            bucket_to_global[bucket_id].push(sorted_pos);
        }

        let original_order = read_original_order(data, &mut pos, num_columns)?;
        if pos != data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "schema: trailing bytes after original_order",
            ));
        }
        validate_original_order(&original_order, num_columns)?;

        Ok(MosaicSchema {
            num_buckets,
            columns,
            bucket_to_global,
            original_order,
        })
    }

    fn deserialize_legacy(data: &[u8]) -> io::Result<Self> {
        struct LegacyColumn {
            logical_index: usize,
            sorted_pos: usize,
            name: String,
            data_type: DataType,
            nullable: bool,
        }

        let mut pos = 0;
        let (num_columns, num_buckets, bpe_rules) = read_header(data, &mut pos, false)?;

        let mut entries = Vec::with_capacity(num_columns);
        let mut bucket_to_global = vec![Vec::new(); num_buckets];
        let mut name_decoder = NameDecoder::new(bpe_rules);
        let mut seen_names = std::collections::HashSet::with_capacity(num_columns);
        let mut seen_indices = vec![false; num_columns];

        for sorted_pos in 0..num_columns {
            let name = name_decoder.read(data, &mut pos)?;
            validate_column_name(&name, &mut seen_names)?;

            let logical_index = varint::decode(data, &mut pos)? as usize;
            if logical_index >= num_columns {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "schema: logical_index out of range",
                ));
            }
            if seen_indices[logical_index] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "schema: duplicate logical_index",
                ));
            }
            seen_indices[logical_index] = true;

            let field = types::deserialize_field(&name, data, &mut pos)?;

            let bucket_id = spec::assign_bucket(sorted_pos, num_columns, num_buckets);
            entries.push(LegacyColumn {
                logical_index,
                sorted_pos,
                name,
                data_type: field.data_type().clone(),
                nullable: field.is_nullable(),
            });
            bucket_to_global[bucket_id].push(logical_index);
        }

        if pos != data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "schema: trailing bytes in legacy schema",
            ));
        }
        if seen_indices.iter().any(|seen| !seen) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "schema: missing logical_index",
            ));
        }

        let mut columns = Vec::with_capacity(num_columns);
        columns.resize_with(num_columns, || ColumnMeta {
            name: String::new(),
            data_type: DataType::Int32,
            nullable: false,
            bucket_id: 0,
        });
        for entry in entries {
            columns[entry.logical_index] = ColumnMeta {
                name: entry.name,
                data_type: entry.data_type,
                nullable: entry.nullable,
                bucket_id: spec::assign_bucket(entry.sorted_pos, num_columns, num_buckets),
            };
        }
        let original_order = (0..num_columns).collect();

        Ok(MosaicSchema {
            num_buckets,
            columns,
            bucket_to_global,
            original_order,
        })
    }

    pub fn serialize(&self) -> Vec<u8> {
        let num_columns = self.columns.len();

        let raw_names: Vec<Vec<u8>> = self
            .columns
            .iter()
            .map(|c| c.name.as_bytes().to_vec())
            .collect();
        let raw_refs: Vec<&[u8]> = raw_names.iter().map(|v| v.as_slice()).collect();

        let plain_size = front_coded_size(&raw_refs);

        let mut bpe_rules = Vec::new();
        let mut bpe_names = Vec::new();
        let mut bpe_size = usize::MAX;

        if bpe::is_ascii_only(&raw_refs) {
            let rules = bpe::build_vocabulary(&raw_refs);
            if !rules.is_empty() {
                let names: Vec<Vec<u8>> = raw_refs
                    .iter()
                    .map(|name| bpe::encode(name, &rules))
                    .collect();
                let name_refs: Vec<&[u8]> = names.iter().map(|v| v.as_slice()).collect();
                bpe_size = 1 + rules.len() * 2 + front_coded_size(&name_refs);
                bpe_rules = rules;
                bpe_names = names;
            }
        }

        let mut buf = Vec::new();
        varint::encode(&mut buf, num_columns as u32);
        varint::encode(&mut buf, self.num_buckets as u32);
        buf.push(SCHEMA_LAYOUT_V2);

        if bpe_size < plain_size {
            buf.push(NAME_ENCODING_BPE);
            varint::encode(&mut buf, bpe_rules.len() as u32);
            for rule in &bpe_rules {
                buf.push(rule[0]);
                buf.push(rule[1]);
            }
            let bpe_refs: Vec<&[u8]> = bpe_names.iter().map(|v| v.as_slice()).collect();
            write_front_coded(&mut buf, &bpe_refs, &self.columns);
        } else {
            buf.push(NAME_ENCODING_FRONT_CODE);
            write_front_coded(&mut buf, &raw_refs, &self.columns);
        }

        // Append original column order as delta + zigzag encoded permutation
        let mut prev = 0i64;
        for &pos in &self.original_order {
            let delta = pos as i64 - prev;
            varint::encode_zigzag(&mut buf, delta);
            prev = pos as i64;
        }

        buf
    }
}

fn has_current_layout_marker(data: &[u8]) -> io::Result<bool> {
    let mut pos = 0;
    let _num_columns = varint::decode(data, &mut pos)?;
    let _num_buckets = varint::decode(data, &mut pos)?;
    if pos >= data.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "schema: missing layout or name encoding byte",
        ));
    }
    Ok(data[pos] == SCHEMA_LAYOUT_V2)
}

fn read_header(
    data: &[u8],
    pos: &mut usize,
    expect_current_layout_marker: bool,
) -> io::Result<(usize, usize, Option<BpeRules>)> {
    let num_columns = varint::decode(data, pos)? as usize;
    let num_buckets = varint::decode(data, pos)? as usize;

    if num_buckets == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "schema: num_buckets must be > 0",
        ));
    }

    if expect_current_layout_marker {
        if *pos >= data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "schema: missing layout marker",
            ));
        }
        if data[*pos] != SCHEMA_LAYOUT_V2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "schema: invalid layout marker",
            ));
        }
        *pos += 1;
    }

    if *pos >= data.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "schema: missing name encoding byte",
        ));
    }
    let name_encoding = data[*pos];
    *pos += 1;

    let bpe_rules = match name_encoding {
        NAME_ENCODING_FRONT_CODE => None,
        NAME_ENCODING_BPE => {
            let num_rules = varint::decode(data, pos)? as usize;
            let mut rules = Vec::with_capacity(num_rules);
            for _ in 0..num_rules {
                if *pos + 2 > data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "schema: truncated BPE rules",
                    ));
                }
                let left = data[*pos];
                let right = data[*pos + 1];
                *pos += 2;
                rules.push([left, right]);
            }
            Some(rules)
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("schema: invalid name encoding: {}", name_encoding),
            ));
        }
    };

    Ok((num_columns, num_buckets, bpe_rules))
}

struct NameDecoder {
    bpe_rules: Option<BpeRules>,
    prev_encoded: Vec<u8>,
}

impl NameDecoder {
    fn new(bpe_rules: Option<BpeRules>) -> Self {
        Self {
            bpe_rules,
            prev_encoded: Vec::new(),
        }
    }

    fn read(&mut self, data: &[u8], pos: &mut usize) -> io::Result<String> {
        let shared = varint::decode(data, pos)? as usize;
        let suffix_len = varint::decode(data, pos)? as usize;

        if shared > self.prev_encoded.len() || *pos + suffix_len > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "schema: corrupted column name encoding",
            ));
        }
        let mut encoded = Vec::with_capacity(shared + suffix_len);
        encoded.extend_from_slice(&self.prev_encoded[..shared]);
        encoded.extend_from_slice(&data[*pos..*pos + suffix_len]);
        *pos += suffix_len;
        self.prev_encoded = encoded.clone();

        let name_bytes = match &self.bpe_rules {
            Some(rules) => bpe::decode(&encoded, rules),
            None => encoded,
        };
        String::from_utf8(name_bytes).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "schema: invalid UTF-8 column name",
            )
        })
    }
}

fn validate_column_name(
    name: &str,
    seen_names: &mut std::collections::HashSet<String>,
) -> io::Result<()> {
    if name.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "schema: empty column name",
        ));
    }
    if !seen_names.insert(name.to_string()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("schema: duplicate column name '{}'", name),
        ));
    }
    Ok(())
}

fn read_original_order(data: &[u8], pos: &mut usize, num_columns: usize) -> io::Result<Vec<usize>> {
    let mut order = Vec::with_capacity(num_columns);
    let mut prev = 0i64;
    for _ in 0..num_columns {
        let delta = varint::decode_zigzag(data, pos)?;
        prev += delta;
        if prev < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "original_order is not a valid permutation",
            ));
        }
        order.push(prev as usize);
    }
    Ok(order)
}

fn validate_original_order(original_order: &[usize], num_columns: usize) -> io::Result<()> {
    if original_order.len() != num_columns {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "original_order length mismatch",
        ));
    }
    let mut seen = vec![false; num_columns];
    for &idx in original_order {
        if idx >= num_columns || seen[idx] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "original_order is not a valid permutation",
            ));
        }
        seen[idx] = true;
    }
    Ok(())
}

fn write_front_coded(buf: &mut Vec<u8>, name_bytes: &[&[u8]], columns: &[ColumnMeta]) {
    let mut prev: &[u8] = &[];
    for (i, col) in columns.iter().enumerate() {
        let cur = name_bytes[i];
        let shared = common_prefix_length(prev, cur);
        varint::encode(buf, shared as u32);
        varint::encode(buf, (cur.len() - shared) as u32);
        buf.extend_from_slice(&cur[shared..]);
        prev = cur;

        let field = Field::new(&col.name, col.data_type.clone(), col.nullable);
        types::serialize_field(&field, buf);
    }
}

fn front_coded_size(names: &[&[u8]]) -> usize {
    let mut size = 0;
    let mut prev: &[u8] = &[];
    for name in names {
        let shared = common_prefix_length(prev, name);
        let suffix_len = name.len() - shared;
        size += varint::encoded_size(shared as u32)
            + varint::encoded_size(suffix_len as u32)
            + suffix_len;
        prev = name;
    }
    size
}

fn common_prefix_length(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serialize_legacy_schema(
        columns: Vec<(String, DataType, bool)>,
        num_buckets: usize,
    ) -> Vec<u8> {
        let mut sorted_indices: Vec<usize> = (0..columns.len()).collect();
        sorted_indices.sort_by(|&a, &b| columns[a].0.cmp(&columns[b].0));

        let mut buf = Vec::new();
        varint::encode(&mut buf, columns.len() as u32);
        varint::encode(&mut buf, num_buckets as u32);
        buf.push(NAME_ENCODING_FRONT_CODE);

        let mut prev: &[u8] = &[];
        for &global_idx in &sorted_indices {
            let name = columns[global_idx].0.as_bytes();
            let shared = common_prefix_length(prev, name);
            varint::encode(&mut buf, shared as u32);
            varint::encode(&mut buf, (name.len() - shared) as u32);
            buf.extend_from_slice(&name[shared..]);
            prev = name;

            varint::encode(&mut buf, global_idx as u32);
            let field = Field::new(
                &columns[global_idx].0,
                columns[global_idx].1.clone(),
                columns[global_idx].2,
            );
            types::serialize_field(&field, &mut buf);
        }

        buf
    }

    #[test]
    fn test_bucket_assignment() {
        let columns: Vec<(String, DataType, bool)> = (0..1000)
            .map(|i| (format!("col_{:04}", i), DataType::Int32, true))
            .collect();
        let schema = MosaicSchema::new(columns, 10);
        assert_eq!(schema.num_buckets, 10);
        let total: usize = schema.bucket_to_global.iter().map(|b| b.len()).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_serialize() {
        let columns = vec![
            ("b_col".to_string(), DataType::Int32, true),
            ("a_col".to_string(), DataType::Float64, false),
        ];
        let schema = MosaicSchema::new(columns, 2);
        let data = schema.serialize();
        assert!(!data.is_empty());
    }

    #[test]
    fn test_deserializes_legacy_v1_schema_layout() {
        let data = serialize_legacy_schema(vec![("value".to_string(), DataType::Int32, false)], 1);

        let restored = MosaicSchema::deserialize(&data).unwrap();

        assert_eq!(restored.columns.len(), 1);
        assert_eq!(restored.columns[0].name, "value");
        assert_eq!(restored.columns[0].data_type, DataType::Int32);
        assert!(!restored.columns[0].nullable);
        assert_eq!(restored.original_order, vec![0]);
    }

    #[test]
    fn test_deserializes_legacy_v1_schema_order() {
        let data = serialize_legacy_schema(
            vec![
                ("name".to_string(), DataType::Utf8, true),
                ("age".to_string(), DataType::Int32, false),
                ("score".to_string(), DataType::Float64, true),
            ],
            2,
        );

        let restored = MosaicSchema::deserialize_for_version(spec::LEGACY_VERSION, &data).unwrap();

        assert_eq!(restored.columns.len(), 3);
        assert_eq!(restored.columns[0].name, "name");
        assert_eq!(restored.columns[1].name, "age");
        assert_eq!(restored.columns[2].name, "score");
        assert_eq!(restored.columns[1].data_type, DataType::Int32);
        assert!(!restored.columns[1].nullable);
        assert_eq!(restored.original_order, vec![0, 1, 2]);
        assert_eq!(restored.bucket_to_global, vec![vec![1, 0], vec![2]]);
    }

    #[test]
    fn test_serialize_deserialize_sorted_order() {
        let columns = vec![
            ("name".to_string(), DataType::Utf8, true),
            ("age".to_string(), DataType::Int32, true),
            ("score".to_string(), DataType::Float64, true),
        ];
        let schema = MosaicSchema::new(columns, 2);
        assert_eq!(schema.columns[0].name, "age");
        assert_eq!(schema.columns[1].name, "name");
        assert_eq!(schema.columns[2].name, "score");
        // original_order: "name"(orig 0)→sorted 1, "age"(orig 1)→sorted 0, "score"(orig 2)→sorted 2
        assert_eq!(schema.original_order, vec![1, 0, 2]);

        let data = schema.serialize();
        let restored = MosaicSchema::deserialize(&data).unwrap();

        assert_eq!(restored.columns.len(), 3);
        assert_eq!(restored.columns[0].name, "age");
        assert_eq!(restored.columns[1].name, "name");
        assert_eq!(restored.columns[2].name, "score");
        assert_eq!(restored.num_buckets, schema.num_buckets);
        assert_eq!(restored.original_order, vec![1, 0, 2]);

        for i in 0..3 {
            assert_eq!(restored.columns[i].bucket_id, schema.columns[i].bucket_id);
        }
        assert_eq!(restored.bucket_to_global, schema.bucket_to_global);
    }

    #[test]
    fn test_original_order_identity() {
        let columns = vec![
            ("a".to_string(), DataType::Int32, false),
            ("b".to_string(), DataType::Utf8, true),
            ("c".to_string(), DataType::Float64, true),
        ];
        let schema = MosaicSchema::new(columns, 1);
        assert_eq!(schema.original_order, vec![0, 1, 2]);

        let data = schema.serialize();
        let restored = MosaicSchema::deserialize(&data).unwrap();
        assert_eq!(restored.original_order, vec![0, 1, 2]);
    }

    #[test]
    fn test_original_order_duplicate_rejected() {
        let columns = vec![
            ("name".to_string(), DataType::Utf8, true),
            ("age".to_string(), DataType::Int32, false),
            ("score".to_string(), DataType::Float64, true),
        ];
        let mut schema = MosaicSchema::new(columns, 1);
        // Corrupt: duplicate index
        schema.original_order = vec![1, 1, 2];
        let data = schema.serialize();
        let err = MosaicSchema::deserialize(&data).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_original_order_out_of_range_rejected() {
        let columns = vec![
            ("name".to_string(), DataType::Utf8, true),
            ("age".to_string(), DataType::Int32, false),
            ("score".to_string(), DataType::Float64, true),
        ];
        let mut schema = MosaicSchema::new(columns, 1);
        // Corrupt: index out of range
        schema.original_order = vec![0, 1, 5];
        let data = schema.serialize();
        let err = MosaicSchema::deserialize(&data).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
