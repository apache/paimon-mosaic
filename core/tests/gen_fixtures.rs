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

//! Binary compatibility tests. Verifies that the current code produces
//! byte-identical output to committed golden files, catching unintended
//! format changes.

use std::io;
use std::sync::Arc;

use arrow_array::builder::*;
use arrow_array::*;
use arrow_schema::{DataType, Field, Schema};
use paimon_mosaic_core::reader::{InputFile, MosaicReader, ReaderAccess};
use paimon_mosaic_core::writer::{MosaicWriter, OutputFile, WriterOptions};

struct MemOutputFile {
    pub buf: Vec<u8>,
}
impl MemOutputFile {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }
}
impl OutputFile for MemOutputFile {
    fn write(&mut self, data: &[u8]) -> io::Result<()> {
        self.buf.extend_from_slice(data);
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
    fn pos(&self) -> u64 {
        self.buf.len() as u64
    }
}

struct ByteArrayInputFile {
    data: Vec<u8>,
}
impl InputFile for ByteArrayInputFile {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let s = offset as usize;
        buf.copy_from_slice(&self.data[s..s + buf.len()]);
        Ok(())
    }
}

fn golden_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/testdata")
        .join(name)
}

/// Generate the deterministic non-ARRAY file.
/// Schema: id(INT32 NOT NULL), name(UTF8), score(FLOAT64)
/// Data: 5 rows with nulls
/// Options: num_buckets=1, compression=none
fn gen_no_array() -> Vec<u8> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("score", DataType::Float64, true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                Some("alice"),
                None,
                Some("charlie"),
                Some("dave"),
                Some("eve"),
            ])),
            Arc::new(Float64Array::from(vec![
                Some(95.5),
                Some(87.0),
                None,
                Some(72.5),
                Some(100.0),
            ])),
        ],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let opts = WriterOptions {
        num_buckets: 1,
        compression: 0,
        ..WriterOptions::default()
    };
    let mut writer = MosaicWriter::new(out, &schema, opts).unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();
    writer.output().buf.clone()
}

/// Generate the deterministic ARRAY file.
/// Schema: id(INT32 NOT NULL), tags(ARRAY<INT32>)
/// Data: 4 rows — [10,20,30], null, [40,50], []
/// Options: num_buckets=1, compression=none
fn gen_with_array() -> Vec<u8> {
    let element_field = Arc::new(Field::new("item", DataType::Int32, true));
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("tags", DataType::List(element_field.clone()), true),
    ]);

    let ids = Int32Array::from(vec![1, 2, 3, 4]);
    let mut list_builder = ListBuilder::new(Int32Builder::new());
    list_builder.values().append_value(10);
    list_builder.values().append_value(20);
    list_builder.values().append_value(30);
    list_builder.append(true);
    list_builder.append(false);
    list_builder.values().append_value(40);
    list_builder.values().append_value(50);
    list_builder.append(true);
    list_builder.append(true);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(ids), Arc::new(list_builder.finish())],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let opts = WriterOptions {
        num_buckets: 1,
        compression: 0,
        ..WriterOptions::default()
    };
    let mut writer = MosaicWriter::new(out, &schema, opts).unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();
    writer.output().buf.clone()
}

#[test]
fn test_v1_no_array_binary_compatible() {
    let generated = gen_no_array();
    let golden = std::fs::read(golden_path("v1_no_array.mosaic"))
        .expect("golden file missing — run with MOSAIC_REGEN_FIXTURES=1 to regenerate");
    assert_eq!(
        generated, golden,
        "non-ARRAY file differs from golden — format may have changed unintentionally"
    );
}

#[test]
fn test_v1_with_array_binary_stable() {
    let generated = gen_with_array();
    let golden = std::fs::read(golden_path("v1_with_array.mosaic"))
        .expect("golden file missing — run with MOSAIC_REGEN_FIXTURES=1 to regenerate");
    assert_eq!(
        generated, golden,
        "ARRAY file differs from golden — format may have changed unintentionally"
    );
}

#[test]
fn test_v1_no_array_golden_readable() {
    let data = std::fs::read(golden_path("v1_no_array.mosaic")).unwrap();
    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();
    let mut rg = reader.row_group_reader(0).unwrap();
    let rb = rg.read_columns().unwrap();
    assert_eq!(rb.num_rows(), 5);
    assert_eq!(rb.num_columns(), 3);
}

#[test]
fn test_v1_with_array_golden_readable() {
    let data = std::fs::read(golden_path("v1_with_array.mosaic")).unwrap();
    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();
    let mut rg = reader.row_group_reader(0).unwrap();
    let rb = rg.read_columns().unwrap();
    assert_eq!(rb.num_rows(), 4);

    let tags = rb.column(1).as_any().downcast_ref::<ListArray>().unwrap();
    assert!(!tags.is_null(0));
    assert!(tags.is_null(1));
    assert!(!tags.is_null(2));
    assert!(!tags.is_null(3));

    let r0 = tags
        .value(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .clone();
    assert_eq!(r0.len(), 3);
    assert_eq!(r0.value(0), 10);

    let r3 = tags
        .value(3)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .clone();
    assert_eq!(r3.len(), 0);
}

/// Generate the deterministic MAP file.
/// Schema: id(INT32 NOT NULL), props(MAP<INT32, UTF8>)
/// Data: 3 rows — {1:"a", 2:"b"}, null, {}
/// Options: num_buckets=1, compression=none
fn gen_with_map() -> Vec<u8> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "props",
            DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(arrow_schema::Fields::from(vec![
                        Field::new("keys", DataType::Int32, false),
                        Field::new("values", DataType::Utf8, true),
                    ])),
                    false,
                )),
                false,
            ),
            true,
        ),
    ]);

    let ids = Int32Array::from(vec![1, 2, 3]);
    let key_builder = Int32Builder::new();
    let value_builder = StringBuilder::new();
    let mut map_builder = MapBuilder::new(None, key_builder, value_builder);
    // row 0: {1: "a", 2: "b"}
    map_builder.keys().append_value(1);
    map_builder.values().append_value("a");
    map_builder.keys().append_value(2);
    map_builder.values().append_value("b");
    map_builder.append(true).unwrap();
    // row 1: null
    map_builder.append(false).unwrap();
    // row 2: {} (empty)
    map_builder.append(true).unwrap();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(ids), Arc::new(map_builder.finish())],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let opts = WriterOptions {
        num_buckets: 1,
        compression: 0,
        ..WriterOptions::default()
    };
    let mut writer = MosaicWriter::new(out, &schema, opts).unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();
    writer.output().buf.clone()
}

#[test]
fn test_v1_with_map_binary_stable() {
    let generated = gen_with_map();
    let golden = std::fs::read(golden_path("v1_with_map.mosaic"))
        .expect("golden file missing — regenerate with gen_with_map()");
    assert_eq!(
        generated, golden,
        "MAP file differs from golden — format may have changed unintentionally"
    );
}

#[test]
fn test_v1_with_map_golden_readable() {
    let data = std::fs::read(golden_path("v1_with_map.mosaic")).unwrap();
    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();
    let mut rg = reader.row_group_reader(0).unwrap();
    let rb = rg.read_columns().unwrap();
    assert_eq!(rb.num_rows(), 3);

    let map_col = rb.column(1).as_any().downcast_ref::<MapArray>().unwrap();
    assert!(!map_col.is_null(0));
    assert!(map_col.is_null(1));
    assert!(!map_col.is_null(2));

    // Row 0: 2 entries
    assert_eq!(map_col.value_offsets()[1] - map_col.value_offsets()[0], 2);
    // Row 2: empty
    assert_eq!(map_col.value_offsets()[3] - map_col.value_offsets()[2], 0);
}
