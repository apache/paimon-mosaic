// Generates a tiny mosaic file for CLI verification.
use std::fs::File;
use std::io::Write;
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

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/sample.mosaic".into());
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("age", DataType::Int32, true),
    ]);
    let out = FileOut { f: File::create(&path).unwrap(), pos: 0 };
    let opts = WriterOptions {
        num_buckets: 2,
        stats_columns: vec!["id".into(), "name".into(), "age".into()],
        ..Default::default()
    };
    let mut w = MosaicWriter::new(out, &schema, opts).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![Some("alice"), Some("bob"), None, Some("dan"), Some("eve")])),
            Arc::new(Int32Array::from(vec![Some(30), Some(25), Some(40), None, Some(28)])),
        ],
    )
    .unwrap();
    w.write_batch(&batch).unwrap();
    w.close().unwrap();
    println!("wrote {path}");
}
