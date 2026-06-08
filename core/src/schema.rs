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
    pub column_id: u32,
    pub data_type: DataType,
    pub nullable: bool,
    pub bucket_id: usize,
}

/// Records how a STRUCT column was expanded into independent columns.
#[derive(Debug, Clone)]
pub struct StructMapping {
    /// DFS column ID of the STRUCT node itself
    pub struct_id: u32,
    /// Index in the original Arrow schema (before expansion)
    pub original_col_index: usize,
    /// The original STRUCT field (for reassembly)
    pub original_field: arrow_schema::Field,
    /// Indices into MosaicSchema.columns for each expanded sub-column
    pub expanded_col_indices: Vec<usize>,
    /// Sorted index of the __null__ column (if STRUCT is nullable)
    pub null_col_sorted_idx: Option<usize>,
    /// DFS column IDs of each leaf field (parallel with expanded_col_indices, excluding __null__)
    pub field_ids: Vec<u32>,
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

        // Expand STRUCT columns into independent flat columns with DFS column IDs
        let mut expanded: Vec<(String, u32, DataType, bool)> = Vec::new();
        let mut struct_mappings: Vec<StructMapping> = Vec::new();
        let mut id_counter: u32 = 0;

        for (orig_idx, (name, dt, nullable)) in columns.iter().enumerate() {
            if let DataType::Struct(fields) = dt {
                if !types::is_timestamp_nanos_struct(fields) {
                    let struct_id = id_counter;
                    id_counter += 1;
                    let start = expanded.len();
                    let has_null_col = *nullable;
                    if has_null_col {
                        let null_name = format!("__null__({})", struct_id);
                        expanded.push((null_name, struct_id, DataType::Boolean, false));
                    }
                    let field_start = expanded.len();
                    Self::expand_struct_fields_recursive(fields, &mut id_counter, &mut expanded);
                    let end = expanded.len();

                    let expanded_indices: Vec<usize> = (start..end).collect();
                    let field_ids: Vec<u32> = expanded[field_start..end]
                        .iter()
                        .filter(|(n, _, _, _)| !n.starts_with("__null__("))
                        .map(|(_, id, _, _)| *id)
                        .collect();

                    struct_mappings.push(StructMapping {
                        struct_id,
                        original_col_index: orig_idx,
                        original_field: schema.field(orig_idx).clone(),
                        expanded_col_indices: expanded_indices,
                        null_col_sorted_idx: None, // filled after sorting
                        field_ids,
                    });
                    continue;
                }
            }
            let col_id = id_counter;
            id_counter += 1;
            expanded.push((name.clone(), col_id, dt.clone(), *nullable));
        }

        let mut schema = Self::new_with_ids(expanded, num_buckets);
        schema.struct_mappings = struct_mappings;

        // Fix struct_mappings expanded_col_indices and null_col_sorted_idx to sorted positions
        for mapping in &mut schema.struct_mappings {
            mapping.expanded_col_indices = mapping
                .expanded_col_indices
                .iter()
                .map(|&input_idx| schema.original_order[input_idx])
                .collect();
            // Find null_col_sorted_idx by struct_id
            let null_name = format!("__null__({})", mapping.struct_id);
            mapping.null_col_sorted_idx = schema.columns.iter().position(|c| c.name == null_name);
        }

        if !schema.struct_mappings.is_empty() {
            schema.original_columns = Some(columns);
        }

        Ok(schema)
    }

    fn expand_struct_fields_recursive(
        fields: &arrow_schema::Fields,
        id_counter: &mut u32,
        out: &mut Vec<(String, u32, DataType, bool)>,
    ) {
        for field in fields.iter() {
            let field_id = *id_counter;
            *id_counter += 1;
            let dt = field.data_type();
            if let DataType::Struct(inner_fields) = dt {
                if !types::is_timestamp_nanos_struct(inner_fields) {
                    if field.is_nullable() {
                        let null_name = format!("__null__({})", field_id);
                        out.push((null_name, field_id, DataType::Boolean, false));
                    }
                    Self::expand_struct_fields_recursive(inner_fields, id_counter, out);
                    continue;
                }
            }
            out.push((field.name().clone(), field_id, dt.clone(), true));
        }
    }

    /// Resolve column names to expanded column indices.
    /// Returns `(output_order, read_indices)`:
    /// - `output_order`: user-requested indices in request order (for output column ordering)
    /// - `read_indices`: output_order + auto-added `__null__` ancestors (for physical reads)
    ///
    /// Resolution strategy (like ORC/Parquet):
    /// 1. Exact match against original top-level column names (including STRUCT names
    ///    and names containing `.`)
    /// 2. If no match, split on `.` and resolve as a path in the schema tree
    pub fn resolve_projection(
        &self,
        column_names: &[&str],
    ) -> io::Result<(Vec<usize>, Vec<usize>)> {
        let orig = self.original_columns.as_deref();
        let mut output = Vec::with_capacity(column_names.len());

        for name in column_names {
            // Step 1: exact match against original top-level column names
            let exact = orig.is_some_and(|cols| cols.iter().any(|(n, _, _)| n == *name));
            if exact {
                // Check if it's a STRUCT (expand all sub-columns)
                if let Some(mapping) = self
                    .struct_mappings
                    .iter()
                    .find(|m| m.original_field.name() == *name)
                {
                    output.extend_from_slice(&mapping.expanded_col_indices);
                } else {
                    // Non-STRUCT top-level column: find by column_id
                    if let Some(idx) = self.columns.iter().position(|c| c.name == *name) {
                        output.push(idx);
                    }
                }
                continue;
            }

            // For non-STRUCT schemas, try direct column name match
            if self.struct_mappings.is_empty() {
                if let Some(idx) = self.columns.iter().position(|c| c.name == *name) {
                    output.push(idx);
                    continue;
                }
            }

            // Step 2: split on '.' and resolve as path in schema tree
            let parts: Vec<&str> = name.split('.').collect();
            let mut resolved = false;
            if parts.len() >= 2 {
                if let Some(mapping) = self
                    .struct_mappings
                    .iter()
                    .find(|m| m.original_field.name() == parts[0])
                {
                    let local_ids =
                        Self::resolve_struct_path(mapping.original_field.data_type(), &parts[1..]);
                    let base = mapping.struct_id + 1;
                    if !local_ids.is_empty() {
                        let target_ids: Vec<u32> = local_ids.iter().map(|&id| base + id).collect();
                        for &tid in &target_ids {
                            if let Some(idx) = self.columns.iter().position(|c| c.column_id == tid)
                            {
                                output.push(idx);
                            }
                        }
                        // Also include __null__ columns for intermediate STRUCTs
                        for &tid in &target_ids {
                            for m in &self.struct_mappings {
                                if m.field_ids.contains(&tid) || m.struct_id == tid {
                                    if let Some(null_idx) = m.null_col_sorted_idx {
                                        if !output.contains(&null_idx) {
                                            output.push(null_idx);
                                        }
                                    }
                                }
                            }
                        }
                        resolved = true;
                    }
                }
            }

            if !resolved {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("column '{}' not found in schema", name),
                ));
            }
        }

        // Auto-add ancestor __null__ columns for any projected STRUCT leaves
        let mut read = output.clone();
        for mapping in &self.struct_mappings {
            let has_leaf = mapping
                .expanded_col_indices
                .iter()
                .any(|idx| read.contains(idx) && mapping.null_col_sorted_idx != Some(*idx));
            if has_leaf {
                if let Some(null_idx) = mapping.null_col_sorted_idx {
                    if !read.contains(&null_idx) {
                        read.push(null_idx);
                    }
                }
            }
        }

        Ok((output, read))
    }

    /// Resolve a dotted path within a STRUCT type tree, returning matching column IDs.
    /// For leaf fields, returns that field's ID. For intermediate STRUCTs, returns all
    /// descendant leaf IDs plus nested __null__ IDs.
    fn resolve_struct_path(dt: &DataType, path: &[&str]) -> Vec<u32> {
        if let DataType::Struct(fields) = dt {
            if path.is_empty() {
                return Self::collect_all_leaf_ids(dt, &mut 0);
            }
            let target = path[0];
            let mut id = 0u32;
            for field in fields.iter() {
                let field_id = id;
                id += 1;
                if field.name() == target {
                    if path.len() == 1 {
                        if let DataType::Struct(inner) = field.data_type() {
                            if !types::is_timestamp_nanos_struct(inner) {
                                return Self::collect_all_leaf_ids(field.data_type(), &mut 0)
                                    .into_iter()
                                    .map(|offset| field_id + 1 + offset)
                                    .collect();
                            }
                        }
                        return vec![field_id];
                    }
                    if let DataType::Struct(inner) = field.data_type() {
                        if !types::is_timestamp_nanos_struct(inner) {
                            return Self::resolve_struct_path(field.data_type(), &path[1..])
                                .into_iter()
                                .map(|offset| field_id + 1 + offset)
                                .collect();
                        }
                    }
                    return vec![];
                }
                // Skip past this field's subtree
                if let DataType::Struct(inner) = field.data_type() {
                    if !types::is_timestamp_nanos_struct(inner) {
                        id += Self::count_tree_nodes(field.data_type());
                    }
                }
            }
        }
        vec![]
    }

    fn collect_all_leaf_ids(dt: &DataType, counter: &mut u32) -> Vec<u32> {
        let mut ids = Vec::new();
        if let DataType::Struct(fields) = dt {
            for field in fields.iter() {
                let field_id = *counter;
                *counter += 1;
                if let DataType::Struct(inner) = field.data_type() {
                    if !types::is_timestamp_nanos_struct(inner) {
                        ids.extend(Self::collect_all_leaf_ids(field.data_type(), counter));
                        continue;
                    }
                }
                ids.push(field_id);
            }
        }
        ids
    }

    pub fn count_tree_nodes(dt: &DataType) -> u32 {
        let mut count = 0;
        if let DataType::Struct(fields) = dt {
            for field in fields.iter() {
                count += 1;
                if let DataType::Struct(inner) = field.data_type() {
                    if !types::is_timestamp_nanos_struct(inner) {
                        count += Self::count_tree_nodes(field.data_type());
                    }
                }
            }
        }
        count
    }

    pub fn new(columns: Vec<(String, DataType, bool)>, num_buckets: usize) -> Self {
        let with_ids: Vec<(String, u32, DataType, bool)> = columns
            .into_iter()
            .enumerate()
            .map(|(i, (n, dt, nullable))| (n, i as u32, dt, nullable))
            .collect();
        Self::new_with_ids(with_ids, num_buckets)
    }

    fn new_with_ids(columns: Vec<(String, u32, DataType, bool)>, num_buckets: usize) -> Self {
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
                column_id: columns[input_idx].1,
                data_type: columns[input_idx].2.clone(),
                nullable: columns[input_idx].3,
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
                column_id: sorted_pos as u32,
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

        // Check if any deserialized columns are STRUCT types that need expansion
        let has_structs = columns.iter().any(
            |c| matches!(&c.data_type, DataType::Struct(f) if !types::is_timestamp_nanos_struct(f)),
        );

        if !has_structs {
            let schema = MosaicSchema {
                num_buckets,
                columns,
                bucket_to_global,
                original_order,
                struct_mappings: Vec::new(),
                original_columns: None,
            };
            return Ok(schema);
        }

        // Reconstruct original column order from sorted columns + original_order
        let mut input_columns: Vec<(String, DataType, bool)> =
            vec![(String::new(), DataType::Boolean, false); num_columns];
        for (input_idx, &sorted_pos) in original_order.iter().enumerate() {
            let col = &columns[sorted_pos];
            input_columns[input_idx] = (col.name.clone(), col.data_type.clone(), col.nullable);
        }

        // Expand STRUCTs using the same logic as from_arrow (with DFS IDs)
        let mut expanded: Vec<(String, u32, DataType, bool)> = Vec::new();
        let mut struct_mappings: Vec<StructMapping> = Vec::new();
        let mut id_counter: u32 = 0;

        for (orig_idx, (name, dt, nullable)) in input_columns.iter().enumerate() {
            if let DataType::Struct(fields) = dt {
                if !types::is_timestamp_nanos_struct(fields) {
                    let struct_id = id_counter;
                    id_counter += 1;
                    let start = expanded.len();
                    let has_null_col = *nullable;
                    if has_null_col {
                        let null_name = format!("__null__({})", struct_id);
                        expanded.push((null_name, struct_id, DataType::Boolean, false));
                    }
                    let field_start = expanded.len();
                    Self::expand_struct_fields_recursive(fields, &mut id_counter, &mut expanded);
                    let end = expanded.len();

                    let expanded_indices: Vec<usize> = (start..end).collect();
                    let field_ids: Vec<u32> = expanded[field_start..end]
                        .iter()
                        .filter(|(n, _, _, _)| !n.starts_with("__null__("))
                        .map(|(_, id, _, _)| *id)
                        .collect();

                    let original_field = Field::new(name, dt.clone(), *nullable);
                    struct_mappings.push(StructMapping {
                        struct_id,
                        original_col_index: orig_idx,
                        original_field,
                        expanded_col_indices: expanded_indices,
                        null_col_sorted_idx: None,
                        field_ids,
                    });
                    continue;
                }
            }
            let col_id = id_counter;
            id_counter += 1;
            expanded.push((name.clone(), col_id, dt.clone(), *nullable));
        }

        let mut schema = Self::new_with_ids(expanded, num_buckets);
        schema.struct_mappings = struct_mappings;

        for mapping in &mut schema.struct_mappings {
            mapping.expanded_col_indices = mapping
                .expanded_col_indices
                .iter()
                .map(|&input_idx| schema.original_order[input_idx])
                .collect();
            let null_name = format!("__null__({})", mapping.struct_id);
            mapping.null_col_sorted_idx = schema.columns.iter().position(|c| c.name == null_name);
        }

        schema.original_columns = Some(input_columns);
        Ok(schema)
    }

    pub fn serialize(&self) -> Vec<u8> {
        // When original_columns exists (schema has STRUCTs), serialize the original
        // pre-expansion columns with STRUCT type byte 20. The reader will expand
        // STRUCTs on deserialize using the type metadata — no dot-heuristic needed.
        let serialize_cols: Vec<ColumnMeta> = if let Some(ref orig) = self.original_columns {
            let mut sorted_indices: Vec<usize> = (0..orig.len()).collect();
            sorted_indices.sort_by(|&a, &b| orig[a].0.cmp(&orig[b].0));
            sorted_indices
                .iter()
                .map(|&i| {
                    let bucket_id = spec::assign_bucket(
                        sorted_indices.iter().position(|&x| x == i).unwrap(),
                        orig.len(),
                        self.num_buckets.min(orig.len()).max(1),
                    );
                    ColumnMeta {
                        name: orig[i].0.clone(),
                        column_id: i as u32,
                        data_type: orig[i].1.clone(),
                        nullable: orig[i].2,
                        bucket_id,
                    }
                })
                .collect()
        } else {
            self.columns.clone()
        };

        let num_columns = serialize_cols.len();

        let raw_names: Vec<Vec<u8>> = serialize_cols
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
            write_front_coded(&mut buf, &bpe_refs, &serialize_cols);
        } else {
            buf.push(0); // NAME_ENCODING_FRONT_CODE
            write_front_coded(&mut buf, &raw_refs, &serialize_cols);
        }

        // Append original column order as delta + zigzag encoded permutation
        // When original_columns exists, write the original column order (pre-expansion)
        let order = if let Some(ref orig) = self.original_columns {
            let mut sorted_indices: Vec<usize> = (0..orig.len()).collect();
            sorted_indices.sort_by(|&a, &b| orig[a].0.cmp(&orig[b].0));
            let mut order = vec![0usize; orig.len()];
            for (sorted_pos, &input_idx) in sorted_indices.iter().enumerate() {
                order[input_idx] = sorted_pos;
            }
            order
        } else {
            self.original_order.clone()
        };

        let mut prev = 0i64;
        for &pos in &order {
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
