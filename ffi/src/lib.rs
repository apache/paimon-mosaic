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

#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::CString;
use std::io;
use std::os::raw::{c_char, c_int};
use std::panic::{self, AssertUnwindSafe};
use std::ptr;

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::{RecordBatch, StructArray};
use arrow_schema::{Field, Schema};

use mosaic_core::reader::{InputFile, MosaicReader, ReaderAccess};
use mosaic_core::spec::*;
use mosaic_core::writer::{MosaicWriter, OutputFile, WriterOptions};

type MinMaxPair = (Option<Vec<u8>>, Option<Vec<u8>>);
type MinMaxCache = Vec<Vec<MinMaxPair>>;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

fn panic_message(e: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<String>() {
        format!("native panic: {}", s)
    } else if let Some(s) = e.downcast_ref::<&str>() {
        format!("native panic: {}", s)
    } else {
        "native panic: unknown".to_string()
    }
}

// ======================== OutputFile ========================

#[repr(C)]
pub struct MosaicOutputFile {
    pub ctx: *mut std::ffi::c_void,
    pub write_fn: Option<unsafe extern "C" fn(*mut std::ffi::c_void, *const u8, usize) -> i32>,
    pub flush_fn: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> i32>,
    pub get_pos_fn: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> i64>,
}

struct FfiOutputFile {
    raw: MosaicOutputFile,
    pos: u64,
}

impl OutputFile for FfiOutputFile {
    fn write(&mut self, data: &[u8]) -> io::Result<()> {
        if let Some(write_fn) = self.raw.write_fn {
            let result = unsafe { write_fn(self.raw.ctx, data.as_ptr(), data.len()) };
            if result != 0 {
                return Err(io::Error::other("write callback failed"));
            }
            self.pos += data.len() as u64;
            Ok(())
        } else {
            Err(io::Error::other("write_fn is null"))
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(flush_fn) = self.raw.flush_fn {
            let result = unsafe { flush_fn(self.raw.ctx) };
            if result != 0 {
                return Err(io::Error::other("flush callback failed"));
            }
        }
        Ok(())
    }

    fn pos(&self) -> u64 {
        if let Some(get_pos_fn) = self.raw.get_pos_fn {
            let p = unsafe { get_pos_fn(self.raw.ctx) };
            if p < 0 {
                return self.pos;
            }
            p as u64
        } else {
            self.pos
        }
    }
}

// ======================== Writer Options ========================

#[repr(C)]
pub struct MosaicWriterOptions {
    pub compression: u8,
    pub zstd_level: c_int,
    pub num_buckets: u32,
    pub row_group_max_size: u64,
    pub max_dict_total_bytes: u32,
    pub max_dict_entries: u32,
    pub stats_columns: *const *const c_char,
    pub num_stats_columns: u32,
    pub page_size_threshold: u32,
}

/// Returns default writer options.
#[no_mangle]
pub extern "C" fn mosaic_writer_options_default() -> MosaicWriterOptions {
    MosaicWriterOptions {
        compression: COMPRESSION_ZSTD,
        zstd_level: DEFAULT_ZSTD_LEVEL as c_int,
        num_buckets: DEFAULT_NUM_BUCKETS as u32,
        row_group_max_size: DEFAULT_ROW_GROUP_MAX_SIZE,
        max_dict_total_bytes: DEFAULT_DICT_MAX_TOTAL_BYTES as u32,
        max_dict_entries: DEFAULT_DICT_MAX_ENTRIES as u32,
        stats_columns: ptr::null(),
        num_stats_columns: 0,
        page_size_threshold: DEFAULT_PAGE_SIZE_THRESHOLD as u32,
    }
}

// ======================== Writer ========================

pub struct MosaicWriterHandle {
    inner: MosaicWriter<FfiOutputFile>,
    stat_name_cache: Option<Vec<Vec<CString>>>,
    stat_value_cache: Option<MinMaxCache>,
}

/// Open a writer. The `ffi_schema` is consumed: ownership transfers to the callee
/// and the caller's struct is zeroed to prevent double-release.
#[no_mangle]
pub unsafe extern "C" fn mosaic_writer_open(
    stream: MosaicOutputFile,
    ffi_schema: *mut FFI_ArrowSchema,
    options: MosaicWriterOptions,
) -> *mut MosaicWriterHandle {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if ffi_schema.is_null() {
            set_error("null schema".into());
            return ptr::null_mut();
        }
        let ffi_owned = ptr::read(ffi_schema);
        ptr::write_bytes(ffi_schema, 0, 1);
        let arrow_schema = match Schema::try_from(&ffi_owned) {
            Ok(s) => s,
            Err(e) => {
                set_error(format!("Arrow schema import failed: {}", e));
                return ptr::null_mut();
            }
        };
        let ffi_stream = FfiOutputFile {
            raw: stream,
            pos: 0,
        };
        let stats_cols = if options.stats_columns.is_null() || options.num_stats_columns == 0 {
            Vec::new()
        } else {
            let ptrs = std::slice::from_raw_parts(
                options.stats_columns,
                options.num_stats_columns as usize,
            );
            let mut names = Vec::with_capacity(ptrs.len());
            for &p in ptrs {
                if p.is_null() {
                    set_error("stats_columns contains null pointer".into());
                    return ptr::null_mut();
                }
                let cstr = std::ffi::CStr::from_ptr(p);
                match cstr.to_str() {
                    Ok(s) => names.push(s.to_owned()),
                    Err(_) => {
                        set_error("stats_columns contains invalid UTF-8".into());
                        return ptr::null_mut();
                    }
                }
            }
            names
        };
        let num_buckets = if options.num_buckets == 0 {
            DEFAULT_NUM_BUCKETS
        } else {
            options.num_buckets as usize
        };
        let opts = WriterOptions {
            compression: options.compression,
            zstd_level: options.zstd_level,
            num_buckets,
            row_group_max_size: options.row_group_max_size,
            max_dict_total_bytes: options.max_dict_total_bytes as usize,
            max_dict_entries: options.max_dict_entries as usize,
            stats_columns: stats_cols,
            page_size_threshold: options.page_size_threshold as usize,
        };
        match MosaicWriter::new(ffi_stream, &arrow_schema, opts) {
            Ok(writer) => Box::into_raw(Box::new(MosaicWriterHandle {
                inner: writer,
                stat_name_cache: None,
                stat_value_cache: None,
            })),
            Err(e) => {
                set_error(format!("writer open failed: {}", e));
                ptr::null_mut()
            }
        }
    }));
    match result {
        Ok(val) => val,
        Err(e) => {
            set_error(panic_message(&e));
            ptr::null_mut()
        }
    }
}

/// Close the writer (flush all data and write footer).
#[no_mangle]
pub unsafe extern "C" fn mosaic_writer_close(handle: *mut MosaicWriterHandle) -> c_int {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            return -1;
        }
        let h = &mut *handle;
        match h.inner.close() {
            Ok(()) => 0,
            Err(e) => {
                set_error(format!("close failed: {}", e));
                -1
            }
        }
    }));
    match result {
        Ok(val) => val,
        Err(e) => {
            set_error(panic_message(&e));
            -1
        }
    }
}

/// Free the writer handle.
#[no_mangle]
pub unsafe extern "C" fn mosaic_writer_free(handle: *mut MosaicWriterHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Get estimated file size for rolling decisions.
/// Returns 0 on success, -1 on error. Writes result to `out`.
#[no_mangle]
pub unsafe extern "C" fn mosaic_writer_estimated_file_size(
    handle: *const MosaicWriterHandle,
    out: *mut i64,
) -> c_int {
    if handle.is_null() || out.is_null() {
        set_error("null pointer".into());
        return -1;
    }
    *out = (&*handle).inner.estimated_file_size() as i64;
    0
}

// ======================== Writer Stats ========================

/// Get the number of row groups in a closed writer.
#[no_mangle]
pub unsafe extern "C" fn mosaic_writer_num_row_groups(
    handle: *const MosaicWriterHandle,
    out: *mut u32,
) -> c_int {
    if handle.is_null() || out.is_null() {
        set_error("null pointer".into());
        return -1;
    }
    *out = (&*handle).inner.num_row_groups() as u32;
    0
}

/// Get number of stats entries for a writer row group.
#[no_mangle]
pub unsafe extern "C" fn mosaic_writer_row_group_num_stats(
    handle: *const MosaicWriterHandle,
    rg_index: u32,
    out: *mut u32,
) -> c_int {
    if handle.is_null() || out.is_null() {
        set_error("null pointer".into());
        return -1;
    }
    let h = &*handle;
    let rg = rg_index as usize;
    if rg >= h.inner.num_row_groups() {
        set_error("rg_index out of range".into());
        return -1;
    }
    *out = h.inner.row_group_stats(rg).len() as u32;
    0
}

/// Batch-fetch all stats for a writer row group.
/// Caller must pre-allocate arrays with at least `num_stats` elements.
/// `names` is filled with pointers to NUL-terminated strings (valid until writer is freed).
/// `null_counts` is filled with null counts.
/// `min_ptrs`/`min_lens` and `max_ptrs`/`max_lens` are filled with pointers to byte data
/// (valid until writer is freed). A null pointer with len 0 means no min/max (all-null column).
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn mosaic_writer_row_group_stats(
    handle: *mut MosaicWriterHandle,
    rg_index: u32,
    names: *mut *const c_char,
    null_counts: *mut u64,
    min_ptrs: *mut *const u8,
    min_lens: *mut usize,
    max_ptrs: *mut *const u8,
    max_lens: *mut usize,
) -> c_int {
    if handle.is_null() {
        set_error("null pointer".into());
        return -1;
    }
    let h = &mut *handle;
    let rg = rg_index as usize;
    if rg >= h.inner.num_row_groups() {
        set_error("rg_index out of range".into());
        return -1;
    }
    let stats = h.inner.row_group_stats(rg);
    let schema = h.inner.schema();

    // Ensure name CStrings are cached
    if h.stat_name_cache.is_none() {
        let num_rg = h.inner.num_row_groups();
        let mut cache = Vec::with_capacity(num_rg);
        for r in 0..num_rg {
            let s = h.inner.row_group_stats(r);
            let names_vec: Vec<CString> = s
                .iter()
                .map(|st| {
                    CString::new(schema.columns[st.column_index].name.as_str()).unwrap_or_default()
                })
                .collect();
            cache.push(names_vec);
        }
        h.stat_name_cache = Some(cache);
    }
    // Ensure value bytes are cached
    if h.stat_value_cache.is_none() {
        let num_rg = h.inner.num_row_groups();
        let mut cache = Vec::with_capacity(num_rg);
        for r in 0..num_rg {
            let s = h.inner.row_group_stats(r);
            let vals: Vec<MinMaxPair> = s
                .iter()
                .map(|st| {
                    (
                        st.min.as_ref().map(|v| v.to_be_bytes()),
                        st.max.as_ref().map(|v| v.to_be_bytes()),
                    )
                })
                .collect();
            cache.push(vals);
        }
        h.stat_value_cache = Some(cache);
    }

    let name_cache = &h.stat_name_cache.as_ref().unwrap()[rg];
    let value_cache = &h.stat_value_cache.as_ref().unwrap()[rg];

    for (i, st) in stats.iter().enumerate() {
        if !names.is_null() {
            *names.add(i) = name_cache[i].as_ptr();
        }
        if !null_counts.is_null() {
            *null_counts.add(i) = st.null_count as u64;
        }
        if !min_ptrs.is_null() && !min_lens.is_null() {
            match &value_cache[i].0 {
                Some(b) => {
                    *min_ptrs.add(i) = b.as_ptr();
                    *min_lens.add(i) = b.len();
                }
                None => {
                    *min_ptrs.add(i) = ptr::null();
                    *min_lens.add(i) = 0;
                }
            }
        }
        if !max_ptrs.is_null() && !max_lens.is_null() {
            match &value_cache[i].1 {
                Some(b) => {
                    *max_ptrs.add(i) = b.as_ptr();
                    *max_lens.add(i) = b.len();
                }
                None => {
                    *max_ptrs.add(i) = ptr::null();
                    *max_lens.add(i) = 0;
                }
            }
        }
    }
    0
}

/// Write an Arrow RecordBatch to the writer via the Arrow C Data Interface.
/// The caller provides ArrowArray and ArrowSchema pointers that represent the batch.
/// Ownership of both structs transfers to the callee; the caller's structs are zeroed.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn mosaic_writer_write_batch(
    handle: *mut MosaicWriterHandle,
    ffi_array: *mut FFI_ArrowArray,
    ffi_schema: *mut FFI_ArrowSchema,
) -> c_int {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || ffi_array.is_null() || ffi_schema.is_null() {
            set_error("null pointer".into());
            return -1;
        }
        let h = &mut *handle;
        let arr_owned = ptr::read(ffi_array);
        let schema_owned = ptr::read(ffi_schema);
        ptr::write_bytes(ffi_array, 0, 1);
        ptr::write_bytes(ffi_schema, 0, 1);
        let arr_data = match arrow_array::ffi::from_ffi(arr_owned, &schema_owned) {
            Ok(d) => d,
            Err(e) => {
                set_error(format!("Arrow import failed: {}", e));
                return -1;
            }
        };
        let struct_array = StructArray::from(arr_data);
        let batch = RecordBatch::from(struct_array);
        match h.inner.write_batch(&batch) {
            Ok(()) => 0,
            Err(e) => {
                set_error(format!("write_batch failed: {}", e));
                -1
            }
        }
    }));
    result.unwrap_or_else(|e| {
        set_error(panic_message(&e));
        -1
    })
}

// ======================== Reader ========================

/// Input file for reading Mosaic files.
///
/// `read_at_fn` must be thread-safe: the reader may invoke it concurrently
/// from multiple threads to perform parallel IO.
#[repr(C)]
pub struct MosaicInputFile {
    pub ctx: *mut std::ffi::c_void,
    pub read_at_fn: Option<unsafe extern "C" fn(*mut std::ffi::c_void, u64, *mut u8, usize) -> i32>,
    pub length_fn: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> u64>,
}

struct FfiInputFile {
    raw: MosaicInputFile,
}

unsafe impl Send for FfiInputFile {}
unsafe impl Sync for FfiInputFile {}

impl InputFile for FfiInputFile {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        if let Some(read_at_fn) = self.raw.read_at_fn {
            let result = unsafe { read_at_fn(self.raw.ctx, offset, buf.as_mut_ptr(), buf.len()) };
            if result != 0 {
                return Err(io::Error::other("read_at callback failed"));
            }
            Ok(())
        } else {
            Err(io::Error::other("read_at_fn is null"))
        }
    }
}

pub struct MosaicReaderHandle {
    reader: MosaicReader<FfiInputFile>,
    stat_name_cache: Option<Vec<Vec<CString>>>,
    stat_value_cache: Option<MinMaxCache>,
}

pub struct MosaicRowGroupReaderHandle {
    inner: mosaic_core::reader::RowGroupReader,
}

/// Open a reader from an InputFile callback.
#[no_mangle]
pub unsafe extern "C" fn mosaic_reader_open(
    input_file: MosaicInputFile,
) -> *mut MosaicReaderHandle {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let file_len = if let Some(length_fn) = input_file.length_fn {
            unsafe { length_fn(input_file.ctx) }
        } else {
            0
        };
        let ffi_input = FfiInputFile { raw: input_file };
        match MosaicReader::new(ffi_input, file_len) {
            Ok(reader) => Box::into_raw(Box::new(MosaicReaderHandle {
                reader,
                stat_name_cache: None,
                stat_value_cache: None,
            })),
            Err(e) => {
                set_error(format!("open failed: {}", e));
                ptr::null_mut()
            }
        }
    }));
    match result {
        Ok(val) => val,
        Err(e) => {
            set_error(panic_message(&e));
            ptr::null_mut()
        }
    }
}

/// Free a reader.
#[no_mangle]
pub unsafe extern "C" fn mosaic_reader_free(handle: *mut MosaicReaderHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Get number of row groups.
/// Returns 0 on success, -1 on error. Writes result to `out`.
#[no_mangle]
pub unsafe extern "C" fn mosaic_reader_num_row_groups(
    handle: *const MosaicReaderHandle,
    out: *mut u32,
) -> c_int {
    if handle.is_null() || out.is_null() {
        set_error("null pointer".into());
        return -1;
    }
    *out = (*handle).reader.num_row_groups() as u32;
    0
}

/// Export the reader's schema via the Arrow C Data Interface.
/// Writes into caller-provided ArrowSchema struct.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn mosaic_reader_export_schema(
    handle: *const MosaicReaderHandle,
    out_schema: *mut FFI_ArrowSchema,
) -> c_int {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || out_schema.is_null() {
            set_error("null pointer".into());
            return -1;
        }
        let h = &*handle;
        let schema = h.reader.schema();
        let fields: Vec<Field> = schema
            .original_order
            .iter()
            .map(|&i| {
                let c = &schema.columns[i];
                Field::new(&c.name, c.data_type.clone(), c.nullable)
            })
            .collect();
        let arrow_schema = Schema::new(fields);
        match FFI_ArrowSchema::try_from(&arrow_schema) {
            Ok(ffi_schema) => {
                ptr::write(out_schema, ffi_schema);
                0
            }
            Err(e) => {
                set_error(format!("Arrow schema export failed: {}", e));
                -1
            }
        }
    }));
    match result {
        Ok(val) => val,
        Err(e) => {
            set_error(panic_message(&e));
            -1
        }
    }
}

/// Open a row group reader.
#[no_mangle]
pub unsafe extern "C" fn mosaic_reader_open_row_group(
    handle: *mut MosaicReaderHandle,
    rg_index: u32,
) -> *mut MosaicRowGroupReaderHandle {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            set_error("null reader handle".into());
            return ptr::null_mut();
        }
        let h = &*handle;
        match h.reader.row_group_reader(rg_index as usize) {
            Ok(rg) => Box::into_raw(Box::new(MosaicRowGroupReaderHandle { inner: rg })),
            Err(e) => {
                set_error(format!("open row group failed: {}", e));
                ptr::null_mut()
            }
        }
    }));
    match result {
        Ok(val) => val,
        Err(e) => {
            set_error(panic_message(&e));
            ptr::null_mut()
        }
    }
}

/// Set projection on the reader. Subsequent row group reads will only
/// decompress buckets containing the specified columns.
#[no_mangle]
pub unsafe extern "C" fn mosaic_reader_set_projection(
    handle: *mut MosaicReaderHandle,
    columns: *const *const c_char,
    num_columns: u32,
) -> c_int {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            set_error("null reader handle".into());
            return -1;
        }
        let h = &mut *handle;
        if num_columns == 0 || columns.is_null() {
            match h.reader.project(&[]) {
                Ok(()) => return 0,
                Err(e) => {
                    set_error(e.to_string());
                    return -1;
                }
            }
        }
        let ptrs = std::slice::from_raw_parts(columns, num_columns as usize);
        let mut names = Vec::with_capacity(num_columns as usize);
        for &p in ptrs {
            if p.is_null() {
                set_error("columns contains null pointer".into());
                return -1;
            }
            match std::ffi::CStr::from_ptr(p).to_str() {
                Ok(s) => names.push(s),
                Err(_) => {
                    set_error("columns contains invalid UTF-8".into());
                    return -1;
                }
            }
        }
        let col_refs: Vec<&str> = names.to_vec();
        match h.reader.project(&col_refs) {
            Ok(()) => 0,
            Err(e) => {
                set_error(e.to_string());
                -1
            }
        }
    }));
    match result {
        Ok(val) => val,
        Err(e) => {
            set_error(panic_message(&e));
            -1
        }
    }
}

/// Free a row group reader.
#[no_mangle]
pub unsafe extern "C" fn mosaic_row_group_reader_free(handle: *mut MosaicRowGroupReaderHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Get number of rows in the row group.
/// Returns 0 on success, -1 on error. Writes result to `out`.
#[no_mangle]
pub unsafe extern "C" fn mosaic_row_group_reader_num_rows(
    handle: *const MosaicRowGroupReaderHandle,
    out: *mut u32,
) -> c_int {
    if handle.is_null() || out.is_null() {
        set_error("null pointer".into());
        return -1;
    }
    *out = (*handle).inner.num_rows() as u32;
    0
}

// ======================== Row Group Stats ========================

/// Get number of rows in a row group.
/// Returns 0 on success, -1 on error. Writes result to `out`.
#[no_mangle]
pub unsafe extern "C" fn mosaic_reader_row_group_num_rows(
    handle: *const MosaicReaderHandle,
    rg_index: u32,
    out: *mut u32,
) -> c_int {
    if handle.is_null() || out.is_null() {
        set_error("null pointer".into());
        return -1;
    }
    let h = &*handle;
    match h.reader.row_group_num_rows(rg_index as usize) {
        Ok(n) => {
            *out = n as u32;
            0
        }
        Err(e) => {
            set_error(e.to_string());
            -1
        }
    }
}

/// Get number of stats entries for a row group.
/// Returns 0 on success, -1 on error. Writes result to `out`.
#[no_mangle]
pub unsafe extern "C" fn mosaic_reader_row_group_num_stats(
    handle: *const MosaicReaderHandle,
    rg_index: u32,
    out: *mut u32,
) -> c_int {
    if handle.is_null() || out.is_null() {
        set_error("null pointer".into());
        return -1;
    }
    let h = &*handle;
    let stats = match h.reader.row_group_stats(rg_index as usize) {
        Ok(s) => s,
        Err(e) => {
            set_error(e.to_string());
            return -1;
        }
    };
    *out = stats.len() as u32;
    0
}

/// Batch-fetch all stats for a reader row group.
/// Caller must pre-allocate arrays with at least `num_stats` elements.
/// `names` is filled with pointers to NUL-terminated strings (valid until reader is freed).
/// `null_counts` is filled with null counts.
/// `min_ptrs`/`min_lens` and `max_ptrs`/`max_lens` are filled with pointers to byte data
/// (valid until reader is freed). A null pointer with len 0 means no min/max (all-null column).
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn mosaic_reader_row_group_stats(
    handle: *mut MosaicReaderHandle,
    rg_index: u32,
    names: *mut *const c_char,
    null_counts: *mut u64,
    min_ptrs: *mut *const u8,
    min_lens: *mut usize,
    max_ptrs: *mut *const u8,
    max_lens: *mut usize,
) -> c_int {
    if handle.is_null() {
        set_error("null pointer".into());
        return -1;
    }
    let h = &mut *handle;
    let rg = rg_index as usize;
    let stats = match h.reader.row_group_stats(rg) {
        Ok(s) => s,
        Err(e) => {
            set_error(e.to_string());
            return -1;
        }
    };
    let schema = h.reader.schema();

    // Ensure caches are built
    if h.stat_name_cache.is_none() {
        let num_rg = h.reader.num_row_groups();
        let mut name_cache = Vec::with_capacity(num_rg);
        let mut value_cache = Vec::with_capacity(num_rg);
        for r in 0..num_rg {
            let s = h.reader.row_group_stats(r).unwrap_or(&[]);
            let names_vec: Vec<CString> = s
                .iter()
                .map(|st| {
                    CString::new(schema.columns[st.column_index].name.as_str()).unwrap_or_default()
                })
                .collect();
            let vals: Vec<MinMaxPair> = s
                .iter()
                .map(|st| {
                    (
                        st.min.as_ref().map(|v| v.to_be_bytes()),
                        st.max.as_ref().map(|v| v.to_be_bytes()),
                    )
                })
                .collect();
            name_cache.push(names_vec);
            value_cache.push(vals);
        }
        h.stat_name_cache = Some(name_cache);
        h.stat_value_cache = Some(value_cache);
    }

    let name_cache = &h.stat_name_cache.as_ref().unwrap()[rg];
    let value_cache = &h.stat_value_cache.as_ref().unwrap()[rg];

    for (i, st) in stats.iter().enumerate() {
        if !names.is_null() {
            *names.add(i) = name_cache[i].as_ptr();
        }
        if !null_counts.is_null() {
            *null_counts.add(i) = st.null_count as u64;
        }
        if !min_ptrs.is_null() && !min_lens.is_null() {
            match &value_cache[i].0 {
                Some(b) => {
                    *min_ptrs.add(i) = b.as_ptr();
                    *min_lens.add(i) = b.len();
                }
                None => {
                    *min_ptrs.add(i) = ptr::null();
                    *min_lens.add(i) = 0;
                }
            }
        }
        if !max_ptrs.is_null() && !max_lens.is_null() {
            match &value_cache[i].1 {
                Some(b) => {
                    *max_ptrs.add(i) = b.as_ptr();
                    *max_lens.add(i) = b.len();
                }
                None => {
                    *max_ptrs.add(i) = ptr::null();
                    *max_lens.add(i) = 0;
                }
            }
        }
    }
    0
}

// ======================== Record Batch (Arrow C Data Interface) ========================

pub struct MosaicRecordBatchHandle {
    batch: RecordBatch,
}

/// Read the entire row group as an Arrow RecordBatch.
/// Returns a record batch handle, or null on error.
#[no_mangle]
pub unsafe extern "C" fn mosaic_row_group_reader_read_columns(
    handle: *mut MosaicRowGroupReaderHandle,
) -> *mut MosaicRecordBatchHandle {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            set_error("null handle".into());
            return ptr::null_mut();
        }
        let h = &mut *handle;
        match h.inner.read_columns() {
            Ok(batch) => Box::into_raw(Box::new(MosaicRecordBatchHandle { batch })),
            Err(e) => {
                set_error(format!("read_columns failed: {}", e));
                ptr::null_mut()
            }
        }
    }));
    match result {
        Ok(val) => val,
        Err(e) => {
            set_error(panic_message(&e));
            ptr::null_mut()
        }
    }
}

/// Get number of rows in the record batch.
/// Returns 0 on success, -1 on error. Writes result to `out`.
#[no_mangle]
pub unsafe extern "C" fn mosaic_record_batch_num_rows(
    handle: *const MosaicRecordBatchHandle,
    out: *mut u32,
) -> c_int {
    if handle.is_null() || out.is_null() {
        set_error("null pointer".into());
        return -1;
    }
    *out = (*handle).batch.num_rows() as u32;
    0
}

/// Get number of columns in the record batch.
/// Returns 0 on success, -1 on error. Writes result to `out`.
#[no_mangle]
pub unsafe extern "C" fn mosaic_record_batch_num_columns(
    handle: *const MosaicRecordBatchHandle,
    out: *mut u32,
) -> c_int {
    if handle.is_null() || out.is_null() {
        set_error("null pointer".into());
        return -1;
    }
    *out = (*handle).batch.num_columns() as u32;
    0
}

/// Export the record batch via the Arrow C Data Interface.
/// Writes into caller-provided ArrowArray and ArrowSchema structs.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn mosaic_record_batch_export(
    handle: *const MosaicRecordBatchHandle,
    out_array: *mut FFI_ArrowArray,
    out_schema: *mut FFI_ArrowSchema,
) -> c_int {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || out_array.is_null() || out_schema.is_null() {
            set_error("null pointer".into());
            return -1;
        }
        let h = &*handle;
        let struct_array = StructArray::from(h.batch.clone());
        match arrow_array::ffi::to_ffi(&struct_array.into()) {
            Ok((ffi_array, ffi_schema)) => {
                ptr::write(out_array, ffi_array);
                ptr::write(out_schema, ffi_schema);
                0
            }
            Err(e) => {
                set_error(format!("Arrow export failed: {}", e));
                -1
            }
        }
    }));
    match result {
        Ok(val) => val,
        Err(e) => {
            set_error(panic_message(&e));
            -1
        }
    }
}

/// Free a record batch handle.
#[no_mangle]
pub unsafe extern "C" fn mosaic_record_batch_free(handle: *mut MosaicRecordBatchHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

// ======================== Error ========================

/// Get the last error message. Returns a NUL-terminated pointer to a thread-local string.
/// The pointer is valid until the next FFI call on the same thread.
#[no_mangle]
pub extern "C" fn mosaic_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        let borrow = e.borrow();
        match borrow.as_ref() {
            Some(cs) => cs.as_ptr(),
            None => ptr::null(),
        }
    })
}
