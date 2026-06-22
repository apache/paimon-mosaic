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

//! End-to-end tests: drive the `mosaic` binary against a fixture file and
//! assert stdout. Zero external dev-deps — uses CARGO_BIN_EXE and std only.

use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use paimon_mosaic_core::writer::{MosaicWriter, OutputFile, WriterOptions};

struct FileOut {
    f: File,
    pos: u64,
}
impl OutputFile for FileOut {
    fn write(&mut self, d: &[u8]) -> std::io::Result<()> {
        self.f.write_all(d)?;
        self.pos += d.len() as u64;
        Ok(())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.f.flush()
    }
    fn pos(&self) -> u64 {
        self.pos
    }
}

/// Write a small fixture and return its path under the test temp dir.
fn fixture(name: &str) -> String {
    let path = format!("{}/mosaic_e2e_{}.mosaic", std::env::temp_dir().display(), name);
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("kind", DataType::Utf8, true),
        Field::new("flag", DataType::Int32, true),
    ]);
    let out = FileOut { f: File::create(&path).unwrap(), pos: 0 };
    let opts = WriterOptions {
        num_buckets: 3,
        page_size_threshold: 1,
        stats_columns: vec!["id".into()],
        ..Default::default()
    };
    let mut w = MosaicWriter::new(out, &schema, opts).unwrap();
    let n = 200;
    let ids: Vec<i32> = (0..n).collect();
    let kinds: Vec<&str> = (0..n).map(|i| ["a", "b", "c"][(i % 3) as usize]).collect();
    let flags = vec![7; n as usize];
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(kinds)),
            Arc::new(Int32Array::from(flags)),
        ],
    )
    .unwrap();
    w.write_batch(&batch).unwrap();
    w.close().unwrap();
    path
}

fn run(args: &[&str]) -> (String, String, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_mosaic")).args(args).output().unwrap();
    (
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
        out.status.success(),
    )
}

#[test]
fn schema_lists_columns() {
    let f = fixture("schema");
    let (out, _, ok) = run(&["schema", &f]);
    assert!(ok);
    assert!(out.contains("3 columns, 3 buckets"));
    assert!(out.contains("id: Int32 not null"));
    assert!(out.contains("kind: Utf8"));
}

#[test]
fn meta_shows_stats() {
    let f = fixture("meta");
    let (out, _, ok) = run(&["meta", &f]);
    assert!(ok);
    assert!(out.contains("200 rows"));
    assert!(out.contains("id: nulls=0 min=0 max=199"));
}

#[test]
fn pages_shows_encodings() {
    let f = fixture("pages");
    let (out, _, ok) = run(&["pages", &f]);
    assert!(ok);
    assert!(out.contains("flag: bucket 0 encoding=const"));
    assert!(out.contains("kind: bucket 2 encoding=dict"));
}

#[test]
fn cat_truncates_and_projects() {
    let f = fixture("cat");
    let (out, _, ok) = run(&["cat", &f, "-n", "2"]);
    assert!(ok);
    assert!(out.contains("| id | kind | flag |"));
    assert_eq!(out.matches('\n').count(), 6); // 3 borders + header + 2 rows
    let (proj, _, _) = run(&["cat", &f, "-c", "kind,id", "-n", "1"]);
    assert!(proj.contains("| kind | id |"));
}

#[test]
fn cat_json_is_ndjson() {
    let f = fixture("json");
    let (out, _, ok) = run(&["cat", &f, "-n", "2", "--json"]);
    assert!(ok);
    assert_eq!(out, "{\"id\":0,\"kind\":\"a\",\"flag\":7}\n{\"id\":1,\"kind\":\"b\",\"flag\":7}\n");
}

#[test]
fn missing_file_fails() {
    let (_, err, ok) = run(&["schema", "/no/such/file.mosaic"]);
    assert!(!ok);
    assert!(err.contains("error:"));
}

#[test]
fn footer_shows_format() {
    let f = fixture("footer");
    let (out, _, ok) = run(&["footer", &f]);
    assert!(ok);
    assert!(out.contains("magic=MOSA"));
    assert!(out.contains("buckets=3"));
    assert!(out.contains("compression=zstd"));
}

#[test]
fn dictionary_dumps_entries() {
    let f = fixture("dict");
    let (out, _, ok) = run(&["dictionary", &f, "kind"]);
    assert!(ok);
    assert!(out.contains("3 entries"));
    assert!(out.contains("a") && out.contains("b") && out.contains("c"));
}

#[test]
fn column_size_sums_bytes() {
    let f = fixture("size");
    let (out, _, ok) = run(&["column-size", &f]);
    assert!(ok);
    assert!(out.contains("id:") && out.contains("kind:"));
    assert!(out.contains("flag: 0 B")); // const column has no slot
}

#[test]
fn buckets_show_layout() {
    let f = fixture("buckets");
    let (out, _, ok) = run(&["buckets", &f]);
    assert!(ok);
    assert!(out.contains("row group 0:"));
    assert!(out.contains("[flag]") && out.contains("[id]") && out.contains("[kind]"));
    assert!(out.contains("monolithic") || out.contains("paged"));
}
