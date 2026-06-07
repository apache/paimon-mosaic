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

#![allow(
    clippy::cloned_ref_to_slice_refs,
    clippy::unnecessary_cast,
    clippy::field_reassign_with_default
)]

use std::io;
use std::sync::Arc;

use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};

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
        let start = offset as usize;
        let end = start + buf.len();
        if end > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "read past end",
            ));
        }
        buf.copy_from_slice(&self.data[start..end]);
        Ok(())
    }
}

fn roundtrip(schema: &Schema, batches: &[RecordBatch]) -> Vec<RecordBatch> {
    roundtrip_with_options(schema, batches, WriterOptions::default())
}

fn roundtrip_with_options(
    schema: &Schema,
    batches: &[RecordBatch],
    options: WriterOptions,
) -> Vec<RecordBatch> {
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(out, schema, options).unwrap();
    for batch in batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();

    let data = writer.output().buf.clone();
    let file_len = data.len() as u64;
    let input = ByteArrayInputFile { data };
    let reader = MosaicReader::new(input, file_len).unwrap();

    let mut result = Vec::new();
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader(rg).unwrap();
        result.push(rg_reader.read_columns().unwrap());
    }
    result
}

#[test]
fn test_array_int32_basic() {
    let element_field = Arc::new(Field::new("item", DataType::Int32, true));
    let schema = Schema::new(vec![Field::new(
        "arr",
        DataType::List(element_field.clone()),
        true,
    )]);

    let mut builder = ListBuilder::new(Int32Builder::new());
    builder.values().append_value(1);
    builder.values().append_value(2);
    builder.values().append_value(3);
    builder.append(true);

    builder.values().append_value(4);
    builder.values().append_value(5);
    builder.append(true);

    builder.append(true); // empty array

    builder.append(false); // null

    let array = builder.finish();
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array)]).unwrap();

    let result = roundtrip(&schema, &[batch.clone()]);
    assert_eq!(result.len(), 1);
    let result_batch = &result[0];
    assert_eq!(result_batch.num_rows(), 4);

    let result_col = result_batch
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();

    assert!(!result_col.is_null(0));
    assert!(!result_col.is_null(1));
    assert!(!result_col.is_null(2));
    assert!(result_col.is_null(3));

    let row0 = result_col.value(0);
    let row0_ints = row0.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(row0_ints.len(), 3);
    assert_eq!(row0_ints.value(0), 1);
    assert_eq!(row0_ints.value(1), 2);
    assert_eq!(row0_ints.value(2), 3);

    let row1 = result_col.value(1);
    let row1_ints = row1.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(row1_ints.len(), 2);
    assert_eq!(row1_ints.value(0), 4);
    assert_eq!(row1_ints.value(1), 5);

    let row2 = result_col.value(2);
    let row2_ints = row2.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(row2_ints.len(), 0);
}

#[test]
fn test_array_with_null_elements() {
    let element_field = Arc::new(Field::new("item", DataType::Int64, true));
    let schema = Schema::new(vec![Field::new(
        "arr",
        DataType::List(element_field.clone()),
        true,
    )]);

    let mut builder = ListBuilder::new(Int64Builder::new());
    builder.values().append_value(100);
    builder.values().append_null();
    builder.values().append_value(300);
    builder.append(true);

    builder.values().append_null();
    builder.values().append_null();
    builder.append(true);

    builder.values().append_value(999);
    builder.append(true);

    let array = builder.finish();
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array)]).unwrap();

    let result = roundtrip(&schema, &[batch]);
    let result_col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();

    let row0 = result_col.value(0);
    let row0_arr = row0.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(row0_arr.len(), 3);
    assert_eq!(row0_arr.value(0), 100);
    assert!(row0_arr.is_null(1));
    assert_eq!(row0_arr.value(2), 300);

    let row1 = result_col.value(1);
    let row1_arr = row1.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(row1_arr.len(), 2);
    assert!(row1_arr.is_null(0));
    assert!(row1_arr.is_null(1));

    let row2 = result_col.value(2);
    let row2_arr = row2.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(row2_arr.len(), 1);
    assert_eq!(row2_arr.value(0), 999);
}

#[test]
fn test_array_string_elements() {
    let element_field = Arc::new(Field::new("item", DataType::Utf8, true));
    let schema = Schema::new(vec![Field::new(
        "arr",
        DataType::List(element_field.clone()),
        true,
    )]);

    let mut builder = ListBuilder::new(StringBuilder::new());
    builder.values().append_value("hello");
    builder.values().append_value("world");
    builder.append(true);

    builder.values().append_null();
    builder.values().append_value("foo");
    builder.append(true);

    builder.append(true); // empty

    let array = builder.finish();
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array)]).unwrap();

    let result = roundtrip(&schema, &[batch]);
    let result_col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();

    let row0 = result_col.value(0);
    let row0_arr = row0.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(row0_arr.len(), 2);
    assert_eq!(row0_arr.value(0), "hello");
    assert_eq!(row0_arr.value(1), "world");

    let row1 = result_col.value(1);
    let row1_arr = row1.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(row1_arr.len(), 2);
    assert!(row1_arr.is_null(0));
    assert_eq!(row1_arr.value(1), "foo");

    let row2 = result_col.value(2);
    let row2_arr = row2.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(row2_arr.len(), 0);
}

#[test]
fn test_array_nested_array() {
    let inner_field = Arc::new(Field::new("item", DataType::Int32, true));
    let outer_field = Arc::new(Field::new(
        "item",
        DataType::List(inner_field.clone()),
        true,
    ));
    let schema = Schema::new(vec![Field::new(
        "nested",
        DataType::List(outer_field.clone()),
        true,
    )]);

    let inner_builder = ListBuilder::new(Int32Builder::new());
    let mut outer_builder = ListBuilder::new(inner_builder);

    // Row 0: [[1, 2], [3]]
    outer_builder.values().values().append_value(1);
    outer_builder.values().values().append_value(2);
    outer_builder.values().append(true);
    outer_builder.values().values().append_value(3);
    outer_builder.values().append(true);
    outer_builder.append(true);

    // Row 1: [[4]]
    outer_builder.values().values().append_value(4);
    outer_builder.values().append(true);
    outer_builder.append(true);

    // Row 2: null
    outer_builder.append(false);

    let array = outer_builder.finish();
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array)]).unwrap();

    let result = roundtrip(&schema, &[batch]);
    let result_col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();

    assert!(!result_col.is_null(0));
    assert!(!result_col.is_null(1));
    assert!(result_col.is_null(2));

    let row0 = result_col.value(0);
    let row0_outer = row0.as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(row0_outer.len(), 2);

    let inner0 = row0_outer.value(0);
    let inner0_arr = inner0.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(inner0_arr.len(), 2);
    assert_eq!(inner0_arr.value(0), 1);
    assert_eq!(inner0_arr.value(1), 2);

    let inner1 = row0_outer.value(1);
    let inner1_arr = inner1.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(inner1_arr.len(), 1);
    assert_eq!(inner1_arr.value(0), 3);

    let row1 = result_col.value(1);
    let row1_outer = row1.as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(row1_outer.len(), 1);
    let inner2 = row1_outer.value(0);
    let inner2_arr = inner2.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(inner2_arr.value(0), 4);
}

#[test]
fn test_array_all_null() {
    let element_field = Arc::new(Field::new("item", DataType::Int32, true));
    let schema = Schema::new(vec![Field::new(
        "arr",
        DataType::List(element_field.clone()),
        true,
    )]);

    let mut builder = ListBuilder::new(Int32Builder::new());
    builder.append(false);
    builder.append(false);
    builder.append(false);
    let array = builder.finish();
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array)]).unwrap();

    let result = roundtrip(&schema, &[batch]);
    let result_col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(result_col.len(), 3);
    assert!(result_col.is_null(0));
    assert!(result_col.is_null(1));
    assert!(result_col.is_null(2));
}

#[test]
fn test_array_with_other_columns() {
    let element_field = Arc::new(Field::new("item", DataType::Int32, true));
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("tags", DataType::List(element_field.clone()), true),
        Field::new("name", DataType::Utf8, true),
    ]);

    let ids = Int64Array::from(vec![1, 2, 3]);

    let mut list_builder = ListBuilder::new(Int32Builder::new());
    list_builder.values().append_value(10);
    list_builder.values().append_value(20);
    list_builder.append(true);
    list_builder.append(false); // null
    list_builder.values().append_value(30);
    list_builder.append(true);
    let tags = list_builder.finish();

    let names = StringArray::from(vec![Some("alice"), None, Some("charlie")]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(ids), Arc::new(tags), Arc::new(names)],
    )
    .unwrap();

    let result = roundtrip(&schema, &[batch]);
    let rb = &result[0];
    assert_eq!(rb.num_rows(), 3);

    let result_ids = rb.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(result_ids.value(0), 1);
    assert_eq!(result_ids.value(1), 2);
    assert_eq!(result_ids.value(2), 3);

    let result_tags = rb.column(1).as_any().downcast_ref::<ListArray>().unwrap();
    assert!(!result_tags.is_null(0));
    assert!(result_tags.is_null(1));
    assert!(!result_tags.is_null(2));

    let row0 = result_tags.value(0);
    let row0_arr = row0.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(row0_arr.len(), 2);
    assert_eq!(row0_arr.value(0), 10);
    assert_eq!(row0_arr.value(1), 20);

    let result_names = rb.column(2).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(result_names.value(0), "alice");
    assert!(result_names.is_null(1));
    assert_eq!(result_names.value(2), "charlie");
}

#[test]
fn test_array_large_batch() {
    let element_field = Arc::new(Field::new("item", DataType::Int32, true));
    let schema = Schema::new(vec![Field::new(
        "arr",
        DataType::List(element_field.clone()),
        true,
    )]);

    let mut builder = ListBuilder::new(Int32Builder::new());
    for i in 0..1000 {
        if i % 10 == 0 {
            builder.append(false); // null every 10th row
        } else {
            let num_elements = (i % 5) + 1;
            for j in 0..num_elements {
                if j == 2 && i % 3 == 0 {
                    builder.values().append_null();
                } else {
                    builder.values().append_value((i * 10 + j) as i32);
                }
            }
            builder.append(true);
        }
    }
    let array = builder.finish();
    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array.clone())]).unwrap();

    let result = roundtrip(&schema, &[batch]);
    let result_col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();

    assert_eq!(result_col.len(), 1000);

    for i in 0..1000 {
        if i % 10 == 0 {
            assert!(result_col.is_null(i), "row {} should be null", i);
        } else {
            assert!(!result_col.is_null(i), "row {} should not be null", i);
            let expected = array.value(i);
            let actual = result_col.value(i);
            assert_eq!(&expected, &actual, "mismatch at row {}", i);
        }
    }
}

#[test]
fn test_array_paged_layout() {
    let element_field = Arc::new(Field::new("item", DataType::Int32, true));
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("arr", DataType::List(element_field.clone()), true),
    ]);

    let mut list_builder = ListBuilder::new(Int32Builder::new());
    let mut ids = Vec::new();
    for i in 0..200 {
        ids.push(i as i64);
        if i % 5 == 0 {
            list_builder.append(false);
        } else {
            let n = (i % 4) + 1;
            for j in 0..n {
                if j == 1 && i % 3 == 0 {
                    list_builder.values().append_null();
                } else {
                    list_builder.values().append_value((i * 10 + j) as i32);
                }
            }
            list_builder.append(true);
        }
    }

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(ids.clone())),
            Arc::new(list_builder.finish()),
        ],
    )
    .unwrap();

    let mut opts = WriterOptions::default();
    opts.page_size_threshold = 1;

    let result = roundtrip_with_options(&schema, &[batch.clone()], opts);
    let rb = &result[0];
    assert_eq!(rb.num_rows(), 200);

    let result_ids = rb.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let result_arr = rb.column(1).as_any().downcast_ref::<ListArray>().unwrap();

    for i in 0..200 {
        assert_eq!(result_ids.value(i), i as i64);
        if i % 5 == 0 {
            assert!(result_arr.is_null(i), "row {} should be null", i);
        } else {
            assert!(!result_arr.is_null(i), "row {} should not be null", i);
        }
    }
}

#[test]
fn test_array_null_row_preserves_child_offsets() {
    let element_field = Arc::new(Field::new("item", DataType::Int32, true));
    let schema = Schema::new(vec![Field::new(
        "arr",
        DataType::List(element_field.clone()),
        true,
    )]);

    // Manually construct: row 0 = [1, 2], row 1 = null (but owns child slots 99, 100), row 2 = [5]
    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 2, 4, 5]));
    let values = Arc::new(Int32Array::from(vec![1, 2, 99, 100, 5])) as ArrayRef;
    let nulls = Some(NullBuffer::new(BooleanBuffer::new(
        Buffer::from(vec![0b0000_0101]),
        0,
        3,
    )));
    let array = ListArray::new(element_field, offsets, values, nulls);
    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array.clone())]).unwrap();

    let result = roundtrip(&schema, &[batch]);
    let result_col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();

    assert_eq!(result_col.len(), 3);
    assert!(!result_col.is_null(0));
    assert!(result_col.is_null(1));
    assert!(!result_col.is_null(2));

    let row0 = result_col.value(0);
    let row0_arr = row0.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(row0_arr.len(), 2);
    assert_eq!(row0_arr.value(0), 1);
    assert_eq!(row0_arr.value(1), 2);

    let row2 = result_col.value(2);
    let row2_arr = row2.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(row2_arr.len(), 1);
    assert_eq!(row2_arr.value(0), 5);
}

#[test]
fn test_project_array_from_paged_bucket() {
    let element_field = Arc::new(Field::new("item", DataType::Int32, true));
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("arr", DataType::List(element_field.clone()), true),
    ]);

    let ids = Int64Array::from(vec![1, 2, 3]);
    let mut list_builder = ListBuilder::new(Int32Builder::new());
    list_builder.values().append_value(10);
    list_builder.values().append_value(20);
    list_builder.append(true);
    list_builder.values().append_value(30);
    list_builder.append(true);
    list_builder.values().append_value(40);
    list_builder.append(true);
    let arr = list_builder.finish();
    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(ids), Arc::new(arr)]).unwrap();

    let out = MemOutputFile::new();
    let mut options = WriterOptions::default();
    options.num_buckets = 1;
    options.page_size_threshold = 1;
    let mut writer = MosaicWriter::new(out, &schema, options).unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();

    let data = writer.output().buf.clone();
    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();

    // Project only the "arr" column
    let sorted_arr_idx = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "arr")
        .unwrap();
    let mut rg_reader = reader
        .row_group_reader_projected(0, &[sorted_arr_idx])
        .unwrap();
    let projected = rg_reader.read_columns().unwrap();
    assert_eq!(projected.num_columns(), 1);

    let result_arr = projected
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(result_arr.len(), 3);

    let r0 = result_arr.value(0);
    let r0a = r0.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(r0a.len(), 2);
    assert_eq!(r0a.value(0), 10);
    assert_eq!(r0a.value(1), 20);
}

#[test]
fn test_array_child_dict_encoding() {
    let element_field = Arc::new(Field::new("item", DataType::Int32, true));
    let schema = Schema::new(vec![Field::new(
        "arr",
        DataType::List(element_field.clone()),
        true,
    )]);

    let mut builder = ListBuilder::new(Int32Builder::new());
    for _ in 0..10 {
        for j in 0..20 {
            builder.values().append_value((j % 3) as i32);
        }
        builder.append(true);
    }
    let array = builder.finish();
    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(array.clone())]).unwrap();

    let result = roundtrip(&schema, &[batch]);
    let result_col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(result_col.len(), 10);
    for i in 0..10 {
        let expected = array.value(i);
        let actual = result_col.value(i);
        assert_eq!(&expected, &actual, "mismatch at row {}", i);
    }
}

#[test]
fn test_multiple_array_columns_in_bucket() {
    let elem_i32 = Arc::new(Field::new("item", DataType::Int32, true));
    let elem_i64 = Arc::new(Field::new("item", DataType::Int64, true));
    let schema = Schema::new(vec![
        Field::new("arr_a", DataType::List(elem_i32.clone()), true),
        Field::new("arr_b", DataType::List(elem_i64.clone()), true),
    ]);

    let mut builder_a = ListBuilder::new(Int32Builder::new());
    builder_a.values().append_value(1);
    builder_a.values().append_value(2);
    builder_a.append(true);
    builder_a.append(false); // null
    builder_a.values().append_value(3);
    builder_a.append(true);

    let mut builder_b = ListBuilder::new(Int64Builder::new());
    builder_b.values().append_value(100);
    builder_b.append(true);
    builder_b.values().append_value(200);
    builder_b.values().append_value(300);
    builder_b.append(true);
    builder_b.append(true); // empty

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(builder_a.finish()), Arc::new(builder_b.finish())],
    )
    .unwrap();

    let mut opts = WriterOptions::default();
    opts.num_buckets = 1;
    let result = roundtrip_with_options(&schema, &[batch], opts);
    let rb = &result[0];

    let col_a = rb.column(0).as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(col_a.len(), 3);
    assert!(!col_a.is_null(0));
    assert!(col_a.is_null(1));
    assert!(!col_a.is_null(2));
    let a0 = col_a
        .value(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .clone();
    assert_eq!(a0.len(), 2);
    assert_eq!(a0.value(0), 1);
    assert_eq!(a0.value(1), 2);
    let a2 = col_a
        .value(2)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .clone();
    assert_eq!(a2.value(0), 3);

    let col_b = rb.column(1).as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(col_b.len(), 3);
    let b0 = col_b
        .value(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .clone();
    assert_eq!(b0.value(0), 100);
    let b1 = col_b
        .value(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .clone();
    assert_eq!(b1.len(), 2);
    assert_eq!(b1.value(0), 200);
    assert_eq!(b1.value(1), 300);
    let b2 = col_b
        .value(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .clone();
    assert_eq!(b2.len(), 0);
}

#[test]
fn test_array_date32_elements() {
    let element_field = Arc::new(Field::new("item", DataType::Date32, true));
    let schema = Schema::new(vec![Field::new(
        "arr",
        DataType::List(element_field.clone()),
        true,
    )]);

    let mut builder = ListBuilder::new(Date32Builder::new());
    builder.values().append_value(18000);
    builder.values().append_value(19000);
    builder.append(true);
    builder.append(false); // null row with potential child slots
    builder.values().append_value(20000);
    builder.append(true);

    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(builder.finish())]).unwrap();
    let result = roundtrip(&schema, &[batch]);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(col.len(), 3);
    assert!(col.is_null(1));
    let r0 = col
        .value(0)
        .as_any()
        .downcast_ref::<Date32Array>()
        .unwrap()
        .clone();
    assert_eq!(r0.value(0), 18000);
    assert_eq!(r0.value(1), 19000);
    let r2 = col
        .value(2)
        .as_any()
        .downcast_ref::<Date32Array>()
        .unwrap()
        .clone();
    assert_eq!(r2.value(0), 20000);
}

#[test]
fn test_project_one_array_from_multi_array_paged() {
    let elem_i32 = Arc::new(Field::new("item", DataType::Int32, true));
    let elem_i64 = Arc::new(Field::new("item", DataType::Int64, true));
    let schema = Schema::new(vec![
        Field::new("arr_a", DataType::List(elem_i32.clone()), true),
        Field::new("arr_b", DataType::List(elem_i64.clone()), true),
    ]);

    let mut builder_a = ListBuilder::new(Int32Builder::new());
    builder_a.values().append_value(1);
    builder_a.values().append_value(2);
    builder_a.append(true);
    builder_a.values().append_value(3);
    builder_a.append(true);

    let mut builder_b = ListBuilder::new(Int64Builder::new());
    builder_b.values().append_value(100);
    builder_b.append(true);
    builder_b.values().append_value(200);
    builder_b.values().append_value(300);
    builder_b.append(true);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(builder_a.finish()), Arc::new(builder_b.finish())],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let mut options = WriterOptions::default();
    options.num_buckets = 1;
    options.page_size_threshold = 1;
    let mut writer = MosaicWriter::new(out, &schema, options).unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();

    let data = writer.output().buf.clone();
    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();

    // Project only arr_a
    let arr_a_idx = reader
        .schema()
        .columns
        .iter()
        .position(|c| c.name == "arr_a")
        .unwrap();
    let mut rg = reader.row_group_reader_projected(0, &[arr_a_idx]).unwrap();
    let projected = rg.read_columns().unwrap();
    assert_eq!(projected.num_columns(), 1);
    let col = projected
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(col.len(), 2);
    let r0 = col
        .value(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .clone();
    assert_eq!(r0.len(), 2);
    assert_eq!(r0.value(0), 1);
    assert_eq!(r0.value(1), 2);
}

// ======================== MAP Tests ========================

#[test]
fn test_map_int_string_basic() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "map",
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

    let ids = Int32Array::from(vec![1, 2, 3, 4]);

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

    // row 2: {3: null}
    map_builder.keys().append_value(3);
    map_builder.values().append_null();
    map_builder.append(true).unwrap();

    // row 3: {} (empty)
    map_builder.append(true).unwrap();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(ids), Arc::new(map_builder.finish())],
    )
    .unwrap();

    let result = roundtrip(&schema, &[batch]);
    let rb = &result[0];
    assert_eq!(rb.num_rows(), 4);

    let map_col = rb.column(1).as_any().downcast_ref::<MapArray>().unwrap();
    assert_eq!(map_col.len(), 4);
    assert!(!map_col.is_null(0));
    assert!(map_col.is_null(1));
    assert!(!map_col.is_null(2));
    assert!(!map_col.is_null(3));

    // row 0: {1: "a", 2: "b"}
    let keys0 = map_col
        .keys()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let vals0 = map_col
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(map_col.value_offsets()[0], 0);
    assert_eq!(map_col.value_offsets()[1], 2);
    assert_eq!(keys0.value(0), 1);
    assert_eq!(keys0.value(1), 2);
    assert_eq!(vals0.value(0), "a");
    assert_eq!(vals0.value(1), "b");

    // row 2: {3: null}
    assert_eq!(map_col.value_offsets()[3] - map_col.value_offsets()[2], 1);

    // row 3: empty
    assert_eq!(map_col.value_offsets()[4] - map_col.value_offsets()[3], 0);
}

#[test]
fn test_map_all_null() {
    let entries_field = Field::new(
        "entries",
        DataType::Struct(arrow_schema::Fields::from(vec![
            Field::new("keys", DataType::Utf8, false),
            Field::new("values", DataType::Int64, true),
        ])),
        false,
    );
    let schema = Schema::new(vec![Field::new(
        "m",
        DataType::Map(Arc::new(entries_field), false),
        true,
    )]);

    let key_builder = StringBuilder::new();
    let value_builder = Int64Builder::new();
    let mut map_builder = MapBuilder::new(None, key_builder, value_builder);
    map_builder.append(false).unwrap();
    map_builder.append(false).unwrap();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(map_builder.finish())],
    )
    .unwrap();

    let result = roundtrip(&schema, &[batch]);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<MapArray>()
        .unwrap();
    assert_eq!(col.len(), 2);
    assert!(col.is_null(0));
    assert!(col.is_null(1));
}

#[test]
fn test_map_with_other_columns() {
    let entries_field = Field::new(
        "entries",
        DataType::Struct(arrow_schema::Fields::from(vec![
            Field::new("keys", DataType::Utf8, false),
            Field::new("values", DataType::Float64, true),
        ])),
        false,
    );
    let element_field = Arc::new(Field::new("item", DataType::Int32, true));
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("tags", DataType::List(element_field.clone()), true),
        Field::new("props", DataType::Map(Arc::new(entries_field), false), true),
    ]);

    let ids = Int64Array::from(vec![1, 2]);

    let mut list_builder = ListBuilder::new(Int32Builder::new());
    list_builder.values().append_value(10);
    list_builder.append(true);
    list_builder.append(false);

    let key_builder = StringBuilder::new();
    let value_builder = Float64Builder::new();
    let mut map_builder = MapBuilder::new(None, key_builder, value_builder);
    map_builder.keys().append_value("x");
    map_builder.values().append_value(1.5);
    map_builder.append(true).unwrap();
    map_builder.keys().append_value("y");
    map_builder.values().append_value(2.5);
    map_builder.keys().append_value("z");
    map_builder.values().append_value(3.5);
    map_builder.append(true).unwrap();

    let mut opts = WriterOptions::default();
    opts.num_buckets = 1;
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(ids),
            Arc::new(list_builder.finish()),
            Arc::new(map_builder.finish()),
        ],
    )
    .unwrap();

    let result = roundtrip_with_options(&schema, &[batch], opts);
    let rb = &result[0];
    assert_eq!(rb.num_rows(), 2);

    let result_ids = rb.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(result_ids.value(0), 1);

    let result_tags = rb.column(1).as_any().downcast_ref::<ListArray>().unwrap();
    assert!(!result_tags.is_null(0));
    assert!(result_tags.is_null(1));

    let result_props = rb.column(2).as_any().downcast_ref::<MapArray>().unwrap();
    assert_eq!(result_props.len(), 2);
    assert_eq!(
        result_props.value_offsets()[1] - result_props.value_offsets()[0],
        1
    );
    assert_eq!(
        result_props.value_offsets()[2] - result_props.value_offsets()[1],
        2
    );
}

// ======================== Nested ARRAY/MAP Tests ========================

#[test]
fn test_array_of_map() {
    // ARRAY<MAP<INT32, UTF8>>
    let map_type = DataType::Map(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(arrow_schema::Fields::from(vec![
                Field::new("keys", DataType::Int32, false),
                Field::new("values", DataType::Utf8, true),
            ])),
            false,
        )),
        false,
    );
    let schema = Schema::new(vec![Field::new(
        "col",
        DataType::List(Arc::new(Field::new("item", map_type.clone(), true))),
        true,
    )]);

    // Build: row 0 = [{1:"a"}, {2:"b", 3:"c"}], row 1 = null
    let key_builder = Int32Builder::new();
    let val_builder = StringBuilder::new();
    let map_builder = MapBuilder::new(None, key_builder, val_builder);
    let mut list_builder = ListBuilder::new(map_builder);

    // row 0: [{1:"a"}, {2:"b", 3:"c"}]
    list_builder.values().keys().append_value(1);
    list_builder.values().values().append_value("a");
    list_builder.values().append(true).unwrap();
    list_builder.values().keys().append_value(2);
    list_builder.values().values().append_value("b");
    list_builder.values().keys().append_value(3);
    list_builder.values().values().append_value("c");
    list_builder.values().append(true).unwrap();
    list_builder.append(true);

    // row 1: null
    list_builder.append(false);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(list_builder.finish())],
    )
    .unwrap();

    let result = roundtrip(&schema, &[batch]);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(col.len(), 2);
    assert!(!col.is_null(0));
    assert!(col.is_null(1));

    let row0 = col.value(0);
    let maps = row0.as_any().downcast_ref::<MapArray>().unwrap();
    assert_eq!(maps.len(), 2);
    assert_eq!(maps.value_length(0), 1); // {1:"a"}
    assert_eq!(maps.value_length(1), 2); // {2:"b", 3:"c"}
}

#[test]
fn test_map_with_array_value() {
    // MAP<UTF8, ARRAY<INT32>>
    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
    let schema = Schema::new(vec![Field::new(
        "col",
        DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(arrow_schema::Fields::from(vec![
                    Field::new("keys", DataType::Utf8, false),
                    Field::new("values", list_type.clone(), true),
                ])),
                false,
            )),
            false,
        ),
        true,
    )]);

    // Build: row 0 = {"x": [1,2], "y": [3]}, row 1 = {}
    let key_builder = StringBuilder::new();
    let val_builder = ListBuilder::new(Int32Builder::new());
    let mut map_builder = MapBuilder::new(None, key_builder, val_builder);

    // row 0
    map_builder.keys().append_value("x");
    map_builder.values().values().append_value(1);
    map_builder.values().values().append_value(2);
    map_builder.values().append(true);
    map_builder.keys().append_value("y");
    map_builder.values().values().append_value(3);
    map_builder.values().append(true);
    map_builder.append(true).unwrap();

    // row 1: empty
    map_builder.append(true).unwrap();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(map_builder.finish())],
    )
    .unwrap();

    let result = roundtrip(&schema, &[batch]);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<MapArray>()
        .unwrap();
    assert_eq!(col.len(), 2);
    assert_eq!(col.value_length(0), 2); // 2 entries
    assert_eq!(col.value_length(1), 0); // empty

    let keys = col.keys().as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(keys.value(0), "x");
    assert_eq!(keys.value(1), "y");

    let vals = col.values().as_any().downcast_ref::<ListArray>().unwrap();
    let v0 = vals
        .value(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .clone();
    assert_eq!(v0.len(), 2);
    assert_eq!(v0.value(0), 1);
    assert_eq!(v0.value(1), 2);
    let v1 = vals
        .value(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .clone();
    assert_eq!(v1.len(), 1);
    assert_eq!(v1.value(0), 3);
}

// ======================== MAP Schema Validation Tests ========================

#[test]
fn test_map_custom_field_names_roundtrip() {
    let schema = Schema::new(vec![Field::new(
        "m",
        DataType::Map(
            Arc::new(Field::new(
                "my_entries",
                DataType::Struct(arrow_schema::Fields::from(vec![
                    Field::new("k_custom", DataType::Int32, false),
                    Field::new("v_custom", DataType::Utf8, true),
                ])),
                false,
            )),
            false,
        ),
        true,
    )]);

    let key_builder = Int32Builder::new();
    let value_builder = StringBuilder::new();
    let field_names = arrow_array::builder::MapFieldNames {
        entry: "my_entries".to_string(),
        key: "k_custom".to_string(),
        value: "v_custom".to_string(),
    };
    let mut map_builder = MapBuilder::new(Some(field_names), key_builder, value_builder);
    map_builder.keys().append_value(1);
    map_builder.values().append_value("a");
    map_builder.append(true).unwrap();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(map_builder.finish())],
    )
    .unwrap();

    let result = roundtrip(&schema, &[batch]);
    let map_type = result[0].schema().field(0).data_type().clone();
    match map_type {
        DataType::Map(entries, sorted) => {
            assert!(!sorted);
            assert_eq!(entries.name(), "my_entries");
            if let DataType::Struct(fields) = entries.data_type() {
                assert_eq!(fields[0].name(), "k_custom");
                assert_eq!(fields[1].name(), "v_custom");
            } else {
                panic!("entries should be struct");
            }
        }
        other => panic!("expected map, got {:?}", other),
    }
}

#[test]
fn test_sorted_map_rejected() {
    let schema = Schema::new(vec![Field::new(
        "m",
        DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(arrow_schema::Fields::from(vec![
                    Field::new("keys", DataType::Int32, false),
                    Field::new("values", DataType::Utf8, true),
                ])),
                false,
            )),
            true, // sorted = true
        ),
        true,
    )]);

    let out = MemOutputFile::new();
    match MosaicWriter::new(out, &schema, WriterOptions::default()) {
        Ok(_) => panic!("sorted MAP should be rejected"),
        Err(e) => assert!(
            e.to_string().contains("sorted"),
            "error should mention sorted: {}",
            e
        ),
    }
}

#[test]
fn test_complex_map_key_rejected() {
    let key_type = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
    let schema = Schema::new(vec![Field::new(
        "m",
        DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(arrow_schema::Fields::from(vec![
                    Field::new("keys", key_type, false),
                    Field::new("values", DataType::Utf8, true),
                ])),
                false,
            )),
            false,
        ),
        true,
    )]);

    let out = MemOutputFile::new();
    match MosaicWriter::new(out, &schema, WriterOptions::default()) {
        Ok(_) => panic!("complex MAP key should be rejected"),
        Err(e) => assert!(
            e.to_string().contains("MAP key"),
            "error should mention MAP key: {}",
            e
        ),
    }
}

// ======================== STRUCT Tests ========================

#[test]
fn test_struct_basic() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "info",
            DataType::Struct(arrow_schema::Fields::from(vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("age", DataType::Int32, true),
            ])),
            true,
        ),
    ]);

    let ids = Int32Array::from(vec![1, 2, 3]);

    let names = StringArray::from(vec![Some("alice"), Some("bob"), None]);
    let ages = Int32Array::from(vec![Some(30), None, Some(25)]);
    let info = StructArray::from(vec![
        (
            Arc::new(Field::new("name", DataType::Utf8, true)),
            Arc::new(names) as ArrayRef,
        ),
        (
            Arc::new(Field::new("age", DataType::Int32, true)),
            Arc::new(ages) as ArrayRef,
        ),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(ids), Arc::new(info)],
    )
    .unwrap();

    let result = roundtrip(&schema, &[batch]);
    let rb = &result[0];
    assert_eq!(rb.num_rows(), 3);
    assert_eq!(rb.num_columns(), 2);

    let result_ids = rb.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(result_ids.value(0), 1);
    assert_eq!(result_ids.value(1), 2);
    assert_eq!(result_ids.value(2), 3);

    let result_info = rb.column(1).as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(result_info.len(), 3);

    // Fields may be in alphabetical order after roundtrip (age, name)
    let age_col = result_info
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(age_col.value(0), 30);
    assert!(age_col.is_null(1));
    assert_eq!(age_col.value(2), 25);

    let name_col = result_info
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(name_col.value(0), "alice");
    assert_eq!(name_col.value(1), "bob");
    assert!(name_col.is_null(2));
}

#[test]
fn test_struct_nullable() {
    let schema = Schema::new(vec![Field::new(
        "info",
        DataType::Struct(arrow_schema::Fields::from(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Float64, true),
        ])),
        true,
    )]);

    let xs = Int32Array::from(vec![Some(1), Some(2), Some(3)]);
    let ys = Float64Array::from(vec![Some(1.5), Some(2.5), Some(3.5)]);
    let struct_null_buf = arrow_buffer::NullBuffer::new(arrow_buffer::BooleanBuffer::new(
        arrow_buffer::Buffer::from(vec![0b0000_0101]), // row 0 valid, row 1 null, row 2 valid
        0,
        3,
    ));
    let info = StructArray::new(
        arrow_schema::Fields::from(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Float64, true),
        ]),
        vec![Arc::new(xs) as ArrayRef, Arc::new(ys) as ArrayRef],
        Some(struct_null_buf),
    );

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(info)]).unwrap();

    let result = roundtrip(&schema, &[batch]);
    let rb = &result[0];
    assert_eq!(rb.num_rows(), 3);
    assert_eq!(rb.num_columns(), 1);

    let result_info = rb.column(0).as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!result_info.is_null(0));
    assert!(result_info.is_null(1));
    assert!(!result_info.is_null(2));

    let x_col = result_info
        .column_by_name("x")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(x_col.value(0), 1);
    assert_eq!(x_col.value(2), 3);
}

#[test]
fn test_struct_with_array_field() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "data",
            DataType::Struct(arrow_schema::Fields::from(vec![
                Field::new("label", DataType::Utf8, true),
                Field::new(
                    "values",
                    DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                    true,
                ),
            ])),
            true,
        ),
    ]);

    let ids = Int32Array::from(vec![1, 2]);

    let labels = StringArray::from(vec![Some("a"), Some("b")]);
    let mut list_builder = ListBuilder::new(Int32Builder::new());
    list_builder.values().append_value(10);
    list_builder.values().append_value(20);
    list_builder.append(true);
    list_builder.values().append_value(30);
    list_builder.append(true);
    let values_arr = list_builder.finish();

    let data_struct = StructArray::from(vec![
        (
            Arc::new(Field::new("label", DataType::Utf8, true)),
            Arc::new(labels) as ArrayRef,
        ),
        (
            Arc::new(Field::new(
                "values",
                DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
                true,
            )),
            Arc::new(values_arr) as ArrayRef,
        ),
    ]);

    let mut opts = WriterOptions::default();
    opts.num_buckets = 1;
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(ids), Arc::new(data_struct)],
    )
    .unwrap();

    let result = roundtrip_with_options(&schema, &[batch], opts);
    let rb = &result[0];
    assert_eq!(rb.num_rows(), 2);
    assert_eq!(rb.num_columns(), 2);

    let result_data = rb.column(1).as_any().downcast_ref::<StructArray>().unwrap();
    let result_label = result_data
        .column_by_name("label")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(result_label.value(0), "a");
    assert_eq!(result_label.value(1), "b");

    let result_values = result_data
        .column_by_name("values")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(result_values.len(), 2);
    let v0 = result_values
        .value(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .clone();
    assert_eq!(v0.len(), 2);
    assert_eq!(v0.value(0), 10);
    assert_eq!(v0.value(1), 20);
}

#[test]
fn test_array_of_struct() {
    // ARRAY<STRUCT<a INT, b UTF8>>
    let struct_type = DataType::Struct(arrow_schema::Fields::from(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8, true),
    ]));
    let schema = Schema::new(vec![Field::new(
        "col",
        DataType::List(Arc::new(Field::new("item", struct_type.clone(), true))),
        true,
    )]);

    // Row 0: [{a:1, b:"x"}, {a:2, b:"y"}]
    // Row 1: null
    // Row 2: [{a:3, b:null}]
    let fields = arrow_schema::Fields::from(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8, true),
    ]);
    let struct_builder = StructBuilder::from_fields(fields, 4);
    let mut list_builder = ListBuilder::new(struct_builder);

    // Row 0
    list_builder
        .values()
        .field_builder::<Int32Builder>(0)
        .unwrap()
        .append_value(1);
    list_builder
        .values()
        .field_builder::<StringBuilder>(1)
        .unwrap()
        .append_value("x");
    list_builder.values().append(true);
    list_builder
        .values()
        .field_builder::<Int32Builder>(0)
        .unwrap()
        .append_value(2);
    list_builder
        .values()
        .field_builder::<StringBuilder>(1)
        .unwrap()
        .append_value("y");
    list_builder.values().append(true);
    list_builder.append(true);

    // Row 1: null
    list_builder.append(false);

    // Row 2: [{a:3, b:null}]
    list_builder
        .values()
        .field_builder::<Int32Builder>(0)
        .unwrap()
        .append_value(3);
    list_builder
        .values()
        .field_builder::<StringBuilder>(1)
        .unwrap()
        .append_null();
    list_builder.values().append(true);
    list_builder.append(true);

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(list_builder.finish())],
    )
    .unwrap();

    let result = roundtrip(&schema, &[batch]);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(col.len(), 3);
    assert!(!col.is_null(0));
    assert!(col.is_null(1));
    assert!(!col.is_null(2));

    // Row 0: [{a:1, b:"x"}, {a:2, b:"y"}]
    let row0 = col.value(0);
    let structs0 = row0.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(structs0.len(), 2);
    let a0 = structs0
        .column_by_name("a")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(a0.value(0), 1);
    assert_eq!(a0.value(1), 2);
    let b0 = structs0
        .column_by_name("b")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(b0.value(0), "x");
    assert_eq!(b0.value(1), "y");

    // Row 2: [{a:3, b:null}]
    let row2 = col.value(2);
    let structs2 = row2.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(structs2.len(), 1);
    let a2 = structs2
        .column_by_name("a")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(a2.value(0), 3);
    let b2 = structs2
        .column_by_name("b")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(b2.is_null(0));
}

#[test]
fn test_map_of_struct_value() {
    // MAP<UTF8, STRUCT<x INT, y UTF8>>
    let struct_type = DataType::Struct(arrow_schema::Fields::from(vec![
        Field::new("x", DataType::Int32, true),
        Field::new("y", DataType::Utf8, true),
    ]));
    let map_field = Field::new(
        "entries",
        DataType::Struct(arrow_schema::Fields::from(vec![
            Field::new("keys", DataType::Utf8, false),
            Field::new("values", struct_type.clone(), true),
        ])),
        false,
    );
    let schema = Schema::new(vec![Field::new(
        "col",
        DataType::Map(Arc::new(map_field), false),
        true,
    )]);

    // Row 0: {"k1": {x:10, y:"hello"}}
    // Row 1: {"k2": {x:20, y:null}, "k3": {x:30, y:"world"}}
    let key_builder = StringBuilder::new();
    let value_fields = arrow_schema::Fields::from(vec![
        Field::new("x", DataType::Int32, true),
        Field::new("y", DataType::Utf8, true),
    ]);
    let value_builder = StructBuilder::from_fields(value_fields, 3);
    let mut map_builder = MapBuilder::new(None, key_builder, value_builder);

    // Row 0
    map_builder.keys().append_value("k1");
    map_builder
        .values()
        .field_builder::<Int32Builder>(0)
        .unwrap()
        .append_value(10);
    map_builder
        .values()
        .field_builder::<StringBuilder>(1)
        .unwrap()
        .append_value("hello");
    map_builder.values().append(true);
    map_builder.append(true).unwrap();

    // Row 1
    map_builder.keys().append_value("k2");
    map_builder
        .values()
        .field_builder::<Int32Builder>(0)
        .unwrap()
        .append_value(20);
    map_builder
        .values()
        .field_builder::<StringBuilder>(1)
        .unwrap()
        .append_null();
    map_builder.values().append(true);
    map_builder.keys().append_value("k3");
    map_builder
        .values()
        .field_builder::<Int32Builder>(0)
        .unwrap()
        .append_value(30);
    map_builder
        .values()
        .field_builder::<StringBuilder>(1)
        .unwrap()
        .append_value("world");
    map_builder.values().append(true);
    map_builder.append(true).unwrap();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(map_builder.finish())],
    )
    .unwrap();

    let result = roundtrip(&schema, &[batch]);
    let col = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<MapArray>()
        .unwrap();
    assert_eq!(col.len(), 2);

    // Row 0: {"k1": {x:10, y:"hello"}}
    let row0 = col.value(0);
    let entries0 = row0.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(entries0.len(), 1);
    let keys0 = entries0
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(keys0.value(0), "k1");
    let vals0 = entries0
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let x0 = vals0
        .column_by_name("x")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(x0.value(0), 10);
    let y0 = vals0
        .column_by_name("y")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(y0.value(0), "hello");

    // Row 1: 2 entries
    let row1 = col.value(1);
    let entries1 = row1.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(entries1.len(), 2);
    let vals1 = entries1
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let x1 = vals1
        .column_by_name("x")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(x1.value(0), 20);
    assert_eq!(x1.value(1), 30);
    let y1 = vals1
        .column_by_name("y")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(y1.is_null(0));
    assert_eq!(y1.value(1), "world");
}

fn roundtrip_projected(
    schema: &Schema,
    batches: &[RecordBatch],
    projection: &[&str],
) -> Vec<RecordBatch> {
    let out = MemOutputFile::new();
    let mut writer = MosaicWriter::new(out, schema, WriterOptions::default()).unwrap();
    for batch in batches {
        writer.write_batch(batch).unwrap();
    }
    writer.close().unwrap();

    let data = writer.output().buf.clone();
    let file_len = data.len() as u64;
    let input = ByteArrayInputFile { data };
    let mut reader = MosaicReader::new(input, file_len).unwrap();
    reader.project(projection).unwrap();

    let mut result = Vec::new();
    for rg in 0..reader.num_row_groups() {
        let mut rg_reader = reader.row_group_reader(rg).unwrap();
        result.push(rg_reader.read_columns().unwrap());
    }
    result
}

#[test]
fn test_dot_in_column_name_rejected() {
    use paimon_mosaic_core::schema::MosaicSchema;
    let result = MosaicSchema::validate(&[("user.name".to_string(), DataType::Utf8, false)]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("must not contain '.'"));
}

#[test]
fn test_null_suffix_in_column_name_rejected() {
    use paimon_mosaic_core::schema::MosaicSchema;
    let result = MosaicSchema::validate(&[("info__null__".to_string(), DataType::Utf8, false)]);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("__null__"));
}

#[test]
fn test_struct_nested_roundtrip() {
    // STRUCT<name UTF8, addr STRUCT<city UTF8, zip INT>>
    let addr_type = DataType::Struct(arrow_schema::Fields::from(vec![
        Field::new("city", DataType::Utf8, true),
        Field::new("zip", DataType::Int32, true),
    ]));
    let info_type = DataType::Struct(arrow_schema::Fields::from(vec![
        Field::new("name", DataType::Utf8, true),
        Field::new("addr", addr_type.clone(), true),
    ]));
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("info", info_type.clone(), true),
    ]);

    let ids = Int32Array::from(vec![1, 2]);

    let cities = StringArray::from(vec![Some("NYC"), Some("LA")]);
    let zips = Int32Array::from(vec![Some(10001), Some(90001)]);
    let addr = StructArray::new(
        arrow_schema::Fields::from(vec![
            Field::new("city", DataType::Utf8, true),
            Field::new("zip", DataType::Int32, true),
        ]),
        vec![Arc::new(cities) as ArrayRef, Arc::new(zips) as ArrayRef],
        None,
    );

    let names = StringArray::from(vec![Some("alice"), Some("bob")]);
    let info = StructArray::new(
        arrow_schema::Fields::from(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("addr", addr_type.clone(), true),
        ]),
        vec![Arc::new(names) as ArrayRef, Arc::new(addr) as ArrayRef],
        None,
    );

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(ids) as ArrayRef, Arc::new(info) as ArrayRef],
    )
    .unwrap();

    let result = roundtrip(&schema, &[batch]);
    let rb = &result[0];
    assert_eq!(rb.num_columns(), 2);

    let info_out = rb
        .column_by_name("info")
        .unwrap()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();

    let name_out = info_out
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(name_out.value(0), "alice");
    assert_eq!(name_out.value(1), "bob");

    let addr_out = info_out
        .column_by_name("addr")
        .unwrap()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let city_out = addr_out
        .column_by_name("city")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(city_out.value(0), "NYC");
    assert_eq!(city_out.value(1), "LA");
    let zip_out = addr_out
        .column_by_name("zip")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(zip_out.value(0), 10001);
    assert_eq!(zip_out.value(1), 90001);
}

#[test]
fn test_struct_leaf_projection() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "info",
            DataType::Struct(arrow_schema::Fields::from(vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("age", DataType::Int32, true),
            ])),
            true,
        ),
    ]);

    let ids = Int32Array::from(vec![1, 2]);
    let names = StringArray::from(vec![Some("alice"), Some("bob")]);
    let ages = Int32Array::from(vec![Some(30), Some(25)]);
    let info = StructArray::new(
        arrow_schema::Fields::from(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int32, true),
        ]),
        vec![Arc::new(names) as ArrayRef, Arc::new(ages) as ArrayRef],
        None,
    );
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(ids) as ArrayRef, Arc::new(info) as ArrayRef],
    )
    .unwrap();

    // Project only info.name (leaf projection)
    let result = roundtrip_projected(&schema, &[batch.clone()], &["info.name"]);
    let rb = &result[0];
    assert_eq!(rb.num_columns(), 1);
    let info_out = rb.column(0).as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(info_out.num_columns(), 1);
    let name_out = info_out
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(name_out.value(0), "alice");
    assert_eq!(name_out.value(1), "bob");
}

#[test]
fn test_struct_whole_column_projection() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "info",
            DataType::Struct(arrow_schema::Fields::from(vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("age", DataType::Int32, true),
            ])),
            true,
        ),
    ]);

    let ids = Int32Array::from(vec![1, 2]);
    let names = StringArray::from(vec![Some("alice"), Some("bob")]);
    let ages = Int32Array::from(vec![Some(30), Some(25)]);
    let null_buf = NullBuffer::new(BooleanBuffer::new(Buffer::from(vec![0b0000_0011]), 0, 2));
    let info = StructArray::new(
        arrow_schema::Fields::from(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int32, true),
        ]),
        vec![Arc::new(names) as ArrayRef, Arc::new(ages) as ArrayRef],
        Some(null_buf),
    );
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(ids) as ArrayRef, Arc::new(info) as ArrayRef],
    )
    .unwrap();

    // Project using original STRUCT name "info"
    let result = roundtrip_projected(&schema, &[batch], &["info"]);
    let rb = &result[0];
    assert_eq!(rb.num_columns(), 1);
    let info_out = rb.column(0).as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(info_out.num_columns(), 2);
    let name_out = info_out
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(name_out.value(0), "alice");
    assert_eq!(name_out.value(1), "bob");
    let age_out = info_out
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(age_out.value(0), 30);
    assert_eq!(age_out.value(1), 25);
}

#[test]
fn test_struct_nested_nullable_null_semantics() {
    // Verify that addr=null is distinguishable from addr={city:null, zip:null}
    let addr_type = DataType::Struct(arrow_schema::Fields::from(vec![
        Field::new("city", DataType::Utf8, true),
        Field::new("zip", DataType::Int32, true),
    ]));
    let schema = Schema::new(vec![Field::new(
        "info",
        DataType::Struct(arrow_schema::Fields::from(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("addr", addr_type.clone(), true),
        ])),
        true,
    )]);

    // Row 0: {name: "a", addr: {city: null, zip: null}} — addr is non-null, fields are null
    // Row 1: {name: "b", addr: null}                     — addr itself is null
    // Row 2: null                                        — entire info is null
    let cities = StringArray::from(vec![None::<&str>, Some(""), None]);
    let zips = Int32Array::from(vec![None::<i32>, Some(0), None]);
    let addr_null = NullBuffer::new(BooleanBuffer::new(
        Buffer::from(vec![0b0000_0001]), // row 0 valid, row 1 null, row 2 null
        0,
        3,
    ));
    let addr = StructArray::new(
        arrow_schema::Fields::from(vec![
            Field::new("city", DataType::Utf8, true),
            Field::new("zip", DataType::Int32, true),
        ]),
        vec![Arc::new(cities) as ArrayRef, Arc::new(zips) as ArrayRef],
        Some(addr_null),
    );

    let names = StringArray::from(vec![Some("a"), Some("b"), None::<&str>]);
    let info_null = NullBuffer::new(BooleanBuffer::new(
        Buffer::from(vec![0b0000_0011]), // rows 0,1 valid, row 2 null
        0,
        3,
    ));
    let info = StructArray::new(
        arrow_schema::Fields::from(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("addr", addr_type.clone(), true),
        ]),
        vec![Arc::new(names) as ArrayRef, Arc::new(addr) as ArrayRef],
        Some(info_null),
    );

    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(info)]).unwrap();

    let result = roundtrip(&schema, &[batch]);
    let rb = &result[0];
    let info_out = rb.column(0).as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!info_out.is_null(0));
    assert!(!info_out.is_null(1));
    assert!(info_out.is_null(2)); // entire info is null

    let addr_out = info_out
        .column_by_name("addr")
        .unwrap()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    assert!(!addr_out.is_null(0)); // addr non-null, but fields are null
    assert!(addr_out.is_null(1)); // addr itself is null
    assert!(addr_out.is_null(2)); // propagated from info null

    // Row 0: addr is non-null but city/zip are null
    let city_out = addr_out
        .column_by_name("city")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(city_out.is_null(0));
}

#[test]
fn test_struct_field_order_preserved() {
    // Verify that struct field order is preserved after roundtrip (by index, not by name)
    let schema = Schema::new(vec![Field::new(
        "info",
        DataType::Struct(arrow_schema::Fields::from(vec![
            Field::new("zebra", DataType::Int32, true),
            Field::new("alpha", DataType::Utf8, true),
        ])),
        false,
    )]);

    let zebras = Int32Array::from(vec![Some(1), Some(2)]);
    let alphas = StringArray::from(vec![Some("a"), Some("b")]);
    let info = StructArray::new(
        arrow_schema::Fields::from(vec![
            Field::new("zebra", DataType::Int32, true),
            Field::new("alpha", DataType::Utf8, true),
        ]),
        vec![Arc::new(zebras) as ArrayRef, Arc::new(alphas) as ArrayRef],
        None,
    );
    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(info) as ArrayRef]).unwrap();

    let result = roundtrip(&schema, &[batch]);
    let info_out = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    // Fields should be in original order: zebra first, alpha second
    assert_eq!(info_out.fields().first().unwrap().name(), "zebra");
    assert_eq!(info_out.fields().get(1).unwrap().name(), "alpha");
    let z = info_out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(z.value(0), 1);
}

#[test]
fn test_struct_leaf_projection_preserves_parent_null() {
    let schema = Schema::new(vec![Field::new(
        "info",
        DataType::Struct(arrow_schema::Fields::from(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int32, true),
        ])),
        true,
    )]);

    // Row 0: {name: "alice", age: 30}
    // Row 1: null (entire info is null)
    let names = StringArray::from(vec![Some("alice"), None::<&str>]);
    let ages = Int32Array::from(vec![Some(30), None]);
    let null_buf = NullBuffer::new(BooleanBuffer::new(Buffer::from(vec![0b0000_0001]), 0, 2));
    let info = StructArray::new(
        arrow_schema::Fields::from(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int32, true),
        ]),
        vec![Arc::new(names) as ArrayRef, Arc::new(ages) as ArrayRef],
        Some(null_buf),
    );
    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(info) as ArrayRef]).unwrap();

    // Leaf projection: only info.name — must still preserve info null at row 1
    let result = roundtrip_projected(&schema, &[batch], &["info.name"]);
    let rb = &result[0];
    assert_eq!(rb.num_columns(), 1);
    let info_out = rb.column(0).as_any().downcast_ref::<StructArray>().unwrap();
    assert!(!info_out.is_null(0));
    assert!(info_out.is_null(1)); // parent null must be preserved
}

#[test]
fn test_struct_mixed_projection_order() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "info",
            DataType::Struct(arrow_schema::Fields::from(vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("age", DataType::Int32, true),
            ])),
            true,
        ),
    ]);

    let ids = Int32Array::from(vec![1, 2]);
    let names = StringArray::from(vec![Some("alice"), Some("bob")]);
    let ages = Int32Array::from(vec![Some(30), Some(25)]);
    let info = StructArray::new(
        arrow_schema::Fields::from(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int32, true),
        ]),
        vec![Arc::new(names) as ArrayRef, Arc::new(ages) as ArrayRef],
        None,
    );
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(ids) as ArrayRef, Arc::new(info) as ArrayRef],
    )
    .unwrap();

    // Project in non-alphabetical order: info.age first, then id, then info.name
    let result = roundtrip_projected(&schema, &[batch], &["info.age", "id", "info.name"]);
    let rb = &result[0];
    // Output should be: info (with age+name), id — info appears first because info.age was first
    assert_eq!(rb.num_columns(), 2);
    assert_eq!(rb.schema().field(0).name(), "info");
    assert_eq!(rb.schema().field(1).name(), "id");

    let info_out = rb.column(0).as_any().downcast_ref::<StructArray>().unwrap();
    let age_out = info_out
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(age_out.value(0), 30);

    let id_out = rb.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(id_out.value(0), 1);
    assert_eq!(id_out.value(1), 2);
}

#[test]
fn test_struct_leaf_stats() {
    use paimon_mosaic_core::values::Value;

    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "info",
            DataType::Struct(arrow_schema::Fields::from(vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("age", DataType::Int32, true),
            ])),
            true,
        ),
    ]);

    let ids = Int32Array::from(vec![1, 2, 3]);
    let names = StringArray::from(vec![Some("alice"), Some("bob"), None::<&str>]);
    let ages = Int32Array::from(vec![Some(30), Some(25), Some(40)]);
    let null_buf = NullBuffer::new(BooleanBuffer::new(Buffer::from(vec![0b0000_0011]), 0, 3));
    let info = StructArray::new(
        arrow_schema::Fields::from(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int32, true),
        ]),
        vec![Arc::new(names) as ArrayRef, Arc::new(ages) as ArrayRef],
        Some(null_buf),
    );
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(ids) as ArrayRef, Arc::new(info) as ArrayRef],
    )
    .unwrap();

    let out = MemOutputFile::new();
    let mut opts = WriterOptions::default();
    opts.stats_columns = vec!["info.age".to_string()];
    let mut writer = MosaicWriter::new(out, &schema, opts).unwrap();
    writer.write_batch(&batch).unwrap();
    writer.close().unwrap();

    let data = writer.output().buf.clone();
    let input = ByteArrayInputFile { data: data.clone() };
    let reader = MosaicReader::new(input, data.len() as u64).unwrap();

    let stats = reader.row_group_stats(0).unwrap();
    assert!(!stats.is_empty());
    let age_stats = &stats[0];
    // age values after STRUCT null propagation: 30, 25, null (row 2 info is null)
    assert_eq!(age_stats.null_count, 1);
    match &age_stats.min {
        Some(Value::Integer(v)) => assert_eq!(*v, 25),
        other => panic!("expected Some(Integer(25)), got {:?}", other),
    }
    match &age_stats.max {
        Some(Value::Integer(v)) => assert_eq!(*v, 30),
        other => panic!("expected Some(Integer(30)), got {:?}", other),
    }
}
