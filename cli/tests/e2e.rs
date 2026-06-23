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

use arrow::array::{Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
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
    fixture_threshold(name, 1)
}

/// Like `fixture` but with an explicit `page_size_threshold`; threshold 1 forces
/// paged buckets, the default (32 KiB) keeps small files monolithic.
fn fixture_threshold(name: &str, threshold: usize) -> String {
    let path = format!("{}/mosaic_e2e_{}.mosaic", std::env::temp_dir().display(), name);
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("kind", DataType::Utf8, true),
        Field::new("flag", DataType::Int32, true),
    ]);
    let out = FileOut { f: File::create(&path).unwrap(), pos: 0 };
    let opts = WriterOptions {
        num_buckets: 3,
        page_size_threshold: threshold,
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
fn head_is_alias_of_cat() {
    let f = fixture("headalias");
    let (out, _, ok) = run(&["head", &f, "-n", "1"]);
    assert!(ok && out.contains("| id | kind | flag |"));
}

#[test]
fn count_reports_total() {
    let f = fixture("count");
    let (out, _, ok) = run(&["count", &f]);
    assert!(ok && out.trim() == "200");
    let (j, _, _) = run(&["count", &f, "--json"]);
    assert!(j.contains("\"rows\":200"));
}

#[test]
fn cat_all_overrides_limit() {
    let f = fixture("all");
    let (out, _, ok) = run(&["cat", &f, "--all", "--json"]);
    assert!(ok);
    assert_eq!(out.lines().count(), 200); // every row, not the -n default
}

#[test]
fn cat_where_filters_rows() {
    let f = fixture("where");
    let (num, _, ok) = run(&["cat", &f, "--all", "--where", "id>197", "--json"]);
    assert!(ok && num.lines().count() == 2); // 198, 199
    let (str_eq, _, _) = run(&["cat", &f, "--all", "--where", "kind=b", "--json"]);
    assert!(str_eq.lines().count() > 0 && str_eq.lines().all(|l| l.contains("\"kind\":\"b\"")));
    let (none, _, _) = run(&["cat", &f, "--where", "id>9999"]);
    assert!(none.contains("(no rows)"));
    let (_, _, bad) = run(&["cat", &f, "--where", "nope??"]);
    assert!(!bad); // unparseable filter fails
    let (_, _, str_ord) = run(&["cat", &f, "--where", "kind>5"]);
    assert!(!str_ord); // ordering on a string column errors, not silent drop
    // Filtering a column dropped by -c works and doesn't leak into output.
    let (hid, _, ok) = run(&["cat", &f, "-c", "kind", "--where", "id>197", "--json"]);
    assert!(ok && hid.lines().count() == 2 && !hid.contains("\"id\""), "{hid}");
}

#[test]
fn convert_csv_then_inspect() {
    let csv = format!("{}/mosaic_e2e_in.csv", std::env::temp_dir().display());
    std::fs::write(&csv, "id,kind,score\n1,a,10.5\n2,b,20\n3,a,30.5\n").unwrap();
    let out = format!("{}/mosaic_e2e_conv.mosaic", std::env::temp_dir().display());
    let (msg, _, ok) = run(&["convert", &csv, "-o", &out, "--stats", "id"]);
    assert!(ok && msg.contains("3 rows"));
    let (c, _, _) = run(&["count", &out]);
    assert_eq!(c.trim(), "3");
    let (s, _, _) = run(&["schema", &out]);
    assert!(s.contains("id:") && s.contains("score:")); // inferred schema
}

#[test]
fn convert_json_then_inspect() {
    let js = format!("{}/mosaic_e2e_in.ndjson", std::env::temp_dir().display());
    std::fs::write(&js, "{\"id\":1,\"kind\":\"a\"}\n{\"id\":2,\"kind\":\"b\"}\n").unwrap();
    let out = format!("{}/mosaic_e2e_jconv.mosaic", std::env::temp_dir().display());
    let (msg, _, ok) = run(&["convert", &js, "-o", &out]);
    assert!(ok && msg.contains("2 rows"), "{msg}");
    let (j, _, _) = run(&["cat", &out, "--all", "--json"]);
    assert_eq!(j.lines().count(), 2);
    assert!(j.contains("\"kind\":\"a\""));
}

#[test]
fn where_pushdown_keeps_correct_rows() {
    // stats on id let id>100 skip the row group; boundaries must not drop matches.
    let csv = format!("{}/mosaic_e2e_pd.csv", std::env::temp_dir().display());
    std::fs::write(&csv, "id,kind\n1,a\n2,b\n3,a\n").unwrap();
    let out = format!("{}/mosaic_e2e_pd.mosaic", std::env::temp_dir().display());
    run(&["convert", &csv, "-o", &out, "--stats", "id"]);
    let (none, _, _) = run(&["cat", &out, "--all", "--where", "id>100"]);
    assert!(none.contains("(no rows)"));
    let (keep, _, _) = run(&["cat", &out, "--all", "--where", "id>=3", "--json"]);
    assert_eq!(keep.lines().count(), 1); // boundary kept, not skipped
}

#[test]
fn bigint_where_is_exact() {
    // Snowflake-scale ids differ below f64 precision; equality must be exact.
    let csv = format!("{}/mosaic_e2e_sf.csv", std::env::temp_dir().display());
    std::fs::write(&csv, "id\n1700000000000000001\n1700000000000000003\n").unwrap();
    let out = format!("{}/mosaic_e2e_sf.mosaic", std::env::temp_dir().display());
    run(&["convert", &csv, "-o", &out, "--stats", "id"]);
    let (j, _, ok) = run(&["cat", &out, "--all", "--where", "id=1700000000000000003", "--json"]);
    assert!(ok && j.lines().count() == 1 && j.contains("003"), "{j}");
}

#[test]
fn date_column_pushdown_keeps_match() {
    // Date stats are epoch-day ints; pushdown must read them numerically (no
    // string suffix) so the filter still finds the matching row.
    let csv = format!("{}/mosaic_e2e_date.csv", std::env::temp_dir().display());
    std::fs::write(&csv, "d\n2020-01-01\n2021-01-01\n").unwrap();
    let out = format!("{}/mosaic_e2e_date.mosaic", std::env::temp_dir().display());
    run(&["convert", &csv, "-o", &out, "--stats", "d"]);
    let (j, _, ok) = run(&["cat", &out, "--all", "--where", "d>18627", "--json"]);
    assert!(ok && j.lines().count() == 1 && j.contains("2021-01-01"), "{j}");
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
    let (j, _, ok) = run(&["footer", &f, "--json"]);
    assert!(ok);
    assert!(j.contains("\"magic\":\"MOSA\"") && j.contains("\"compression\":\"zstd\""));
}

#[test]
fn dictionary_dumps_entries() {
    let f = fixture("dict");
    let (out, _, ok) = run(&["dictionary", &f, "-c", "kind"]);
    assert!(ok);
    assert!(out.contains("3 entries"));
    assert!(out.contains("a") && out.contains("b") && out.contains("c"));
    let (j, _, ok) = run(&["dictionary", &f, "-c", "kind", "--json"]);
    assert!(ok);
    assert_eq!(j, "{\"column\":\"kind\",\"row_groups\":[[\"a\",\"b\",\"c\"]]}\n");
}

#[test]
fn column_size_sums_bytes() {
    let f = fixture("size");
    let (out, _, ok) = run(&["column-size", &f]);
    assert!(ok);
    assert!(out.contains("id:") && out.contains("kind:"));
    // Every column attributes its on-disk bucket bytes (even the const flag bucket).
    assert!(out.contains("flag: 15 B") && !out.contains(": 0 B"));
    // Paged buckets lack uncompressed sizes, so no (misleading) total ratio.
    assert!(!out.contains("uncompressed"), "paged total must omit ratio: {out}");
}

#[test]
fn column_size_nonzero_on_monolithic() {
    // Default threshold keeps small files monolithic; bytes must still attribute
    // (regression: monolithic buckets previously reported 0 B everywhere).
    let f = fixture_threshold("size_mono", 32 * 1024);
    let (b, _, _) = run(&["buckets", &f]);
    assert!(b.contains("monolithic"), "default file should be monolithic: {b}");
    let (out, _, ok) = run(&["column-size", &f]);
    assert!(ok);
    assert!(out.contains("id: ") && !out.contains("id: 0 B"), "id must be non-zero: {out}");
    assert!(out.contains("kind: ") && !out.contains("kind: 0 B"), "kind must be non-zero: {out}");
    // Single-column buckets are exact, so nothing is flagged approximate.
    assert!(out.contains("total:") && !out.contains("approx"), "single-col exact: {out}");
}

#[test]
fn buckets_show_layout() {
    let f = fixture("buckets");
    let (out, _, ok) = run(&["buckets", &f]);
    assert!(ok);
    assert!(out.contains("row group 0:"));
    assert!(out.contains("[flag]") && out.contains("[id]") && out.contains("[kind]"));
    assert!(out.contains("monolithic") || out.contains("paged"));
    let (j, _, ok) = run(&["buckets", &f, "--json"]);
    assert!(ok);
    assert!(j.contains("\"bucket\":0") && j.contains("\"columns\":") && j.contains("\"uncompressed\":"));
    // const flag bucket is monolithic, so its uncompressed size + ratio show.
    assert!(out.contains("uncompressed") && out.contains("x)"), "ratio: {out}");
}
