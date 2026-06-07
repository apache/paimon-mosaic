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

#[derive(Debug, Clone)]
pub struct ColumnMeta {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub bucket_id: usize,
}

/// Records how a STRUCT column was expanded into independent columns.
#[derive(Debug, Clone)]
pub struct StructMapping {
    /// Index in the original Arrow schema (before expansion)
    pub original_col_index: usize,
    /// The original STRUCT field (for reassembly)
    pub original_field: arrow_schema::Field,
    /// Indices into MosaicSchema.columns for each expanded sub-column
    /// (includes __null__ column if nullable, then each leaf field)
    pub expanded_col_indices: Vec<usize>,
    /// Name of the __null__ column (if STRUCT is nullable), None if not nullable
    pub null_col_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MosaicSchema {
    pub num_buckets: usize,
    /// Expanded columns (STRUCT fields flattened, with __null__ columns)
    pub columns: Vec<ColumnMeta>,
    /// bucket_to_global[bucket_id] = [global_col_indices...] in name-sorted order
    pub bucket_to_global: Vec<Vec<usize>>,
    /// original_order[orig_pos] = sorted_pos. Used as default output order when no projection is set.
    pub original_order: Vec<usize>,
    /// STRUCT column expansion mappings
    pub struct_mappings: Vec<StructMapping>,
    /// Original columns BEFORE STRUCT expansion (for serialization to file)
    pub original_columns: Option<Vec<(String, DataType, bool)>>,
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

        // Expand STRUCT columns into independent flat columns
        let mut expanded: Vec<(String, DataType, bool)> = Vec::new();
        let mut struct_mappings: Vec<StructMapping> = Vec::new();
        // Maps original Arrow col index → range of expanded indices
        let mut orig_to_expanded: Vec<Vec<usize>> = Vec::new();

        for (orig_idx, (name, dt, nullable)) in columns.iter().enumerate() {
            if let DataType::Struct(fields) = dt {
                if !types::is_timestamp_nanos_struct(fields) {
                    let start = expanded.len();
                    // Add __null__ column if struct is nullable
                    let null_col_name = if *nullable {
                        let null_name = format!("{}.__null__", name);
                        expanded.push((null_name.clone(), DataType::Boolean, false));
                        Some(null_name)
                    } else {
                        None
                    };
                    // Recursively expand struct fields
                    Self::expand_struct_fields_recursive(name, fields, &mut expanded);

                    let end = expanded.len();
                    let expanded_indices: Vec<usize> = (start..end).collect();
                    orig_to_expanded.push(expanded_indices.clone());

                    struct_mappings.push(StructMapping {
                        original_col_index: orig_idx,
                        original_field: schema.field(orig_idx).clone(),
                        expanded_col_indices: expanded_indices,
                        null_col_name,
                    });
                    continue;
                }
            }
            orig_to_expanded.push(vec![expanded.len()]);
            expanded.push((name.clone(), dt.clone(), *nullable));
        }

        // Check for name conflicts after expansion
        let mut seen = HashSet::new();
        for (name, _, _) in &expanded {
            if !seen.insert(name.as_str()) {
                return Err(format!(
                    "column name conflict after STRUCT expansion: '{}'",
                    name
                ));
            }
        }

        let mut schema = Self::new(expanded, num_buckets);
        schema.struct_mappings = struct_mappings;

        // Fix struct_mappings expanded_col_indices to point to sorted positions
        // expanded indices are input positions; we need sorted positions
        for mapping in &mut schema.struct_mappings {
            mapping.expanded_col_indices = mapping
                .expanded_col_indices
                .iter()
                .map(|&input_idx| schema.original_order[input_idx])
                .collect();
        }

        // Keep original_order as the expanded column order (identity after new()).
        // Store the mapping from original Arrow positions → first expanded sorted position
        // in struct_mappings for the reader to use when building the output RecordBatch.

        // Store original columns for serialization
        if !schema.struct_mappings.is_empty() {
            schema.original_columns = Some(columns);
        }

        Ok(schema)
    }

    fn expand_struct_fields_recursive(
        prefix: &str,
        fields: &arrow_schema::Fields,
        out: &mut Vec<(String, DataType, bool)>,
    ) {
        for field in fields.iter() {
            let full_name = format!("{}.{}", prefix, field.name());
            let dt = field.data_type();
            if let DataType::Struct(inner_fields) = dt {
                if !types::is_timestamp_nanos_struct(inner_fields) {
                    // Nested STRUCT: recurse
                    Self::expand_struct_fields_recursive(&full_name, inner_fields, out);
                    continue;
                }
            }
            // Leaf field (primitive, ARRAY, MAP, or timestamp_nanos_struct)
            // Force nullable since STRUCT null propagates
            out.push((full_name, dt.clone(), true));
        }
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
            struct_mappings: Vec::new(),
            original_columns: None,
        }
    }

    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        let mut pos = 0;
        let num_columns = varint::decode(data, &mut pos)? as usize;
        let num_buckets = varint::decode(data, &mut pos)? as usize;

        if num_buckets == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "schema: num_buckets must be > 0",
            ));
        }

        if pos >= data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "schema: missing name encoding byte",
            ));
        }
        let name_encoding = data[pos];
        pos += 1;

        let bpe_rules: Option<Vec<[u8; 2]>> = if name_encoding == 1 {
            let num_rules = varint::decode(data, &mut pos)? as usize;
            let mut rules = Vec::with_capacity(num_rules);
            for _ in 0..num_rules {
                if pos + 2 > data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "schema: truncated BPE rules",
                    ));
                }
                let left = data[pos];
                let right = data[pos + 1];
                pos += 2;
                rules.push([left, right]);
            }
            Some(rules)
        } else {
            None
        };

        let mut columns = Vec::with_capacity(num_columns);
        let mut bucket_to_global = vec![Vec::new(); num_buckets];
        let mut prev_encoded: Vec<u8> = Vec::new();
        let mut seen_names = std::collections::HashSet::with_capacity(num_columns);

        for sorted_pos in 0..num_columns {
            let shared = varint::decode(data, &mut pos)? as usize;
            let suffix_len = varint::decode(data, &mut pos)? as usize;

            if shared > prev_encoded.len() || pos + suffix_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "schema: corrupted column name encoding",
                ));
            }
            let mut encoded = Vec::with_capacity(shared + suffix_len);
            encoded.extend_from_slice(&prev_encoded[..shared]);
            encoded.extend_from_slice(&data[pos..pos + suffix_len]);
            pos += suffix_len;
            prev_encoded = encoded.clone();

            let name_bytes = match &bpe_rules {
                Some(rules) => bpe::decode(&encoded, rules),
                None => encoded,
            };
            let name = String::from_utf8(name_bytes).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "schema: invalid UTF-8 column name",
                )
            })?;

            if name.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "schema: empty column name",
                ));
            }
            if !seen_names.insert(name.clone()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("schema: duplicate column name '{}'", name),
                ));
            }

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

        // Read original column order (delta + zigzag encoded)
        let original_order = if pos < data.len() {
            let mut order = Vec::with_capacity(num_columns);
            let mut prev = 0i64;
            for _ in 0..num_columns {
                let delta = varint::decode_zigzag(data, &mut pos)?;
                prev += delta;
                order.push(prev as usize);
            }
            order
        } else {
            (0..num_columns).collect()
        };

        // Validate that original_order is a permutation of 0..num_columns
        if original_order.len() != num_columns {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "original_order length mismatch",
            ));
        }
        let mut seen = vec![false; num_columns];
        for &idx in &original_order {
            if idx >= num_columns || seen[idx] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "original_order is not a valid permutation",
                ));
            }
            seen[idx] = true;
        }

        let mut schema = MosaicSchema {
            num_buckets,
            columns,
            bucket_to_global,
            original_order,
            struct_mappings: Vec::new(),
            original_columns: None,
        };

        // Check if any columns are STRUCT and need expansion
        let _has_structs = schema.columns.iter().any(
            |c| matches!(&c.data_type, DataType::Struct(f) if !types::is_timestamp_nanos_struct(f)),
        );
        // Detect STRUCT columns from dot-separated naming convention
        // and populate struct_mappings for reader reassembly.
        schema.detect_struct_mappings();

        Ok(schema)
    }

    /// Detect STRUCT columns from dot-separated naming convention and __null__ suffix.
    /// Populates struct_mappings for reader reassembly.
    fn detect_struct_mappings(&mut self) {
        use std::collections::BTreeMap;

        // Group columns by their root prefix (before first dot)
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (sorted_idx, col) in self.columns.iter().enumerate() {
            if let Some(dot_pos) = col.name.find('.') {
                let root = col.name[..dot_pos].to_string();
                groups.entry(root).or_default().push(sorted_idx);
            }
        }

        // For each group, build a StructMapping
        let mut orig_col_idx = 0;
        let mut mappings = Vec::new();

        // Determine original column positions: iterate in original_order
        // For now, assign original_col_index based on discovery order
        for (root_name, indices) in &groups {
            // Collect field names and types (excluding __null__)
            let mut field_entries: Vec<(String, DataType, bool)> = Vec::new();
            let mut null_col_name = None;

            for &idx in indices {
                let col_name = &self.columns[idx].name;
                let suffix = &col_name[root_name.len() + 1..]; // after "root."
                if suffix == "__null__" {
                    null_col_name = Some(col_name.clone());
                } else {
                    field_entries.push((
                        suffix.to_string(),
                        self.columns[idx].data_type.clone(),
                        self.columns[idx].nullable,
                    ));
                }
            }

            if field_entries.is_empty() {
                continue;
            }

            // Build the original STRUCT Field
            let struct_fields: Vec<Field> = field_entries
                .iter()
                .map(|(name, dt, nullable)| Field::new(name, dt.clone(), *nullable))
                .collect();
            let original_field = Field::new(
                root_name,
                DataType::Struct(arrow_schema::Fields::from(struct_fields)),
                null_col_name.is_some(),
            );

            // Find original_col_index: the position in the un-expanded schema
            // Since we don't have the original schema here, use a counter
            let mapping = StructMapping {
                original_col_index: orig_col_idx,
                original_field,
                expanded_col_indices: indices.clone(),
                null_col_name,
            };
            mappings.push(mapping);
            orig_col_idx += 1; // This isn't perfectly accurate but works for single-level
        }

        self.struct_mappings = mappings;
    }

    pub fn serialize(&self) -> Vec<u8> {
        // Always serialize the expanded columns (including STRUCT sub-fields).
        // The reader reconstructs STRUCT types from the dot-separated naming convention.
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

        if bpe_size < plain_size {
            buf.push(1); // NAME_ENCODING_BPE
            varint::encode(&mut buf, bpe_rules.len() as u32);
            for rule in &bpe_rules {
                buf.push(rule[0]);
                buf.push(rule[1]);
            }
            let bpe_refs: Vec<&[u8]> = bpe_names.iter().map(|v| v.as_slice()).collect();
            write_front_coded(&mut buf, &bpe_refs, &self.columns);
        } else {
            buf.push(0); // NAME_ENCODING_FRONT_CODE
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
