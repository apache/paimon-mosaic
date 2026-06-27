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

//! Serializable models for each command's `--json` output, so the wire shape
//! lives in one typed place instead of hand-rolled `format!`/`json_str` strings.

use serde::Serialize;

#[derive(Serialize)]
pub struct Schema {
    pub columns: usize,
    pub buckets: usize,
    pub fields: Vec<SchemaField>,
}
#[derive(Serialize)]
pub struct SchemaField {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub nullable: bool,
    pub bucket: u32,
}

#[derive(Serialize)]
pub struct Meta {
    pub rows: usize,
    pub columns: usize,
    pub buckets: usize,
    pub row_groups: Vec<MetaRg>,
}
#[derive(Serialize)]
pub struct MetaRg {
    pub rows: usize,
    pub stats: Vec<Stat>,
}
#[derive(Serialize)]
pub struct Stat {
    pub column: String,
    pub nulls: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<String>,
}

#[derive(Serialize)]
pub struct Pages {
    pub row_groups: Vec<Vec<Page>>,
}
#[derive(Serialize)]
pub struct Page {
    pub column: String,
    pub bucket: usize,
    pub encoding: String,
    pub slot_size: usize,
}

#[derive(Serialize)]
pub struct Count {
    pub rows: usize,
}

#[derive(Serialize)]
pub struct Footer {
    pub magic: String,
    pub version: u32,
    pub buckets: usize,
    pub row_groups: usize,
    pub compression: String,
}

#[derive(Serialize)]
pub struct ColumnSize {
    pub columns: Vec<ColumnBytes>,
    pub total_bytes: usize,
}
#[derive(Serialize)]
pub struct ColumnBytes {
    pub column: String,
    pub bytes: usize,
    pub approximate: bool,
}

#[derive(Serialize)]
pub struct Dictionary {
    pub column: String,
    pub row_groups: Vec<Option<Vec<String>>>,
}

#[derive(Serialize)]
pub struct Buckets {
    pub row_groups: Vec<Vec<Bucket>>,
}
#[derive(Serialize)]
pub struct Bucket {
    pub bucket: usize,
    pub kind: String,
    pub size: usize,
    pub uncompressed: usize,
    pub columns: Vec<String>,
}

/// Serialize a model to a compact JSON line. Infallible for these plain structs.
pub fn line<T: Serialize>(v: &T) -> String {
    serde_json::to_string(v).expect("model serializes")
}
