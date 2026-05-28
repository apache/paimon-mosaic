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

use std::io;
use std::sync::Arc;

use arrow_array::*;
use arrow_schema::{DataType, Field, Schema};

use paimon_mosaic_core::bloom::{hash_value, BloomFilterConfig};
use paimon_mosaic_core::reader::{InputFile, MosaicReader, ReaderAccess};
use paimon_mosaic_core::values::Value;
use paimon_mosaic_core::writer::{MosaicWriter, OutputFile, WriterOptions};

struct MemOutputFile {
    buf: Vec<u8>,
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
        let e = s + buf.len();
        buf.copy_from_slice(&self.data[s..e]);
        Ok(())
    }
}

fn write_file(
    schema: Arc<Schema>,
    batches: Vec<RecordBatch>,
    bloom_columns: Vec<BloomFilterConfig>,
    num_buckets: usize,
) -> Vec<u8> {
    let out = MemOutputFile { buf: Vec::new() };
    let mut writer = MosaicWriter::new(
        out,
        schema.as_ref(),
        WriterOptions {
            num_buckets,
            bloom_filter_columns: bloom_columns,
            ..Default::default()
        },
    )
    .unwrap();
    for b in batches {
        writer.write_batch(&b).unwrap();
    }
    writer.close().unwrap();
    writer.output().buf.clone()
}

#[test]
fn bloom_present_values_match() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("payload", DataType::Int32, true),
    ]));
    let n = 5000i64;
    let id_array: Int64Array = (0..n).collect();
    let payload_array: Int32Array = (0..n).map(|i| Some(i as i32)).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(id_array), Arc::new(payload_array)],
    )
    .unwrap();
    let bytes = write_file(
        schema.clone(),
        vec![batch],
        vec![BloomFilterConfig {
            column_name: "id".to_string(),
            ndv: n as u64,
            fpp: 0.01,
        }],
        4,
    );
    let len = bytes.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile { data: bytes }, len).expect("reader open");

    assert_eq!(reader.num_row_groups(), 1);
    let metas = reader.row_group_bloom_meta(0).unwrap();
    assert_eq!(metas.len(), 1, "exactly one bloom for column id");
    assert_eq!(metas[0].column_index, 0);
    assert!(metas[0].total_bytes > 0);

    let filter = reader
        .bloom_filter(0, 0)
        .expect("bloom fetch ok")
        .expect("filter present");

    for i in 0..n {
        let h = hash_value(&Value::BigInt(i));
        assert!(filter.contains_hash(h), "present value {} missing", i);
    }

    let probe = 10_000i64;
    let mut fp = 0u64;
    for i in n..(n + probe) {
        let h = hash_value(&Value::BigInt(i));
        if filter.contains_hash(h) {
            fp += 1;
        }
    }
    let observed = fp as f64 / probe as f64;
    assert!(
        observed < 0.05,
        "observed fpp {} too high (target was 0.01)",
        observed
    );
}

#[test]
fn no_bloom_when_not_configured() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let id_array: Int64Array = (0..100i64).collect();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(id_array)]).unwrap();
    let bytes = write_file(schema.clone(), vec![batch], Vec::new(), 1);
    let len = bytes.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile { data: bytes }, len).unwrap();

    let metas = reader.row_group_bloom_meta(0).unwrap();
    assert!(metas.is_empty());
    assert!(reader.bloom_filter(0, 0).unwrap().is_none());
}

#[test]
fn bloom_string_column_skips_absent_value() {
    let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
    let names: Vec<&str> = vec!["alice", "bob", "carol", "dave", "eve"];
    let arr = StringArray::from(names.clone());
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();
    let bytes = write_file(
        schema.clone(),
        vec![batch],
        vec![BloomFilterConfig {
            column_name: "name".to_string(),
            ndv: 1024,
            fpp: 0.001,
        }],
        1,
    );
    let len = bytes.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile { data: bytes }, len).unwrap();
    let filter = reader.bloom_filter(0, 0).unwrap().unwrap();

    for n in &names {
        let h = hash_value(&Value::String(n.as_bytes().to_vec()));
        assert!(filter.contains_hash(h), "present name {} missing", n);
    }
    let absent = "zachary";
    let h = hash_value(&Value::String(absent.as_bytes().to_vec()));
    assert!(
        !filter.contains_hash(h),
        "filter false-positive on small set"
    );
}

#[test]
fn bloom_per_row_group_independent() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

    let batches: Vec<RecordBatch> = vec![
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from((0..1000i64).collect::<Vec<_>>()))],
        )
        .unwrap(),
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(
                (1_000_000..1_001_000i64).collect::<Vec<_>>(),
            ))],
        )
        .unwrap(),
    ];

    let mut out = MemOutputFile { buf: Vec::new() };
    let mut writer = MosaicWriter::new(
        out,
        schema.as_ref(),
        WriterOptions {
            num_buckets: 1,
            row_group_max_size: 16 * 1024,
            bloom_filter_columns: vec![BloomFilterConfig {
                column_name: "id".to_string(),
                ndv: 4000,
                fpp: 0.01,
            }],
            ..Default::default()
        },
    )
    .unwrap();
    for b in &batches {
        writer.write_batch(b).unwrap();
        // Force separate row groups by manually flushing buffer through tiny RG size.
    }
    writer.close().unwrap();
    let bytes = writer.output().buf.clone();
    drop(writer);
    out = MemOutputFile { buf: Vec::new() };
    drop(out);

    let len = bytes.len() as u64;
    let reader = MosaicReader::new(ByteArrayInputFile { data: bytes }, len).unwrap();
    assert!(reader.num_row_groups() >= 1);

    let f0 = reader.bloom_filter(0, 0).unwrap().unwrap();
    for i in 0..1000i64 {
        assert!(f0.contains_hash(hash_value(&Value::BigInt(i))));
    }
    if reader.num_row_groups() >= 2 {
        let f1 = reader.bloom_filter(1, 0).unwrap().unwrap();
        for i in 1_000_000..1_001_000i64 {
            assert!(f1.contains_hash(hash_value(&Value::BigInt(i))));
        }
        let h = hash_value(&Value::BigInt(0));
        assert!(!f1.contains_hash(h));
    }
}
