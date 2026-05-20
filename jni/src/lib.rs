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
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::Arc;

use jni::objects::{
    GlobalRef, JByteArray, JClass, JMethodID, JObject, JObjectArray, JString, JValue,
};
use jni::sys::{jint, jlong, jlongArray};
use jni::JNIEnv;
use jni::JavaVM;

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::{RecordBatch, StructArray};
use arrow_schema::Schema;

use mosaic_core::reader::{InputFile, MosaicReader, ReaderAccess, RowGroupReader};
use mosaic_core::spec::*;
use mosaic_core::writer::{MosaicWriter, OutputFile, WriterOptions};

fn panic_message(e: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<String>() {
        format!("native panic: {}", s)
    } else if let Some(s) = e.downcast_ref::<&str>() {
        format!("native panic: {}", s)
    } else {
        "native panic: unknown".to_string()
    }
}

struct JniOutputFile {
    jvm: Arc<JavaVM>,
    stream_ref: GlobalRef,
    write_mid: JMethodID,
    flush_mid: JMethodID,
    pos: u64,
    cached_array: Option<GlobalRef>,
    cached_array_len: usize,
}

unsafe impl Send for JniOutputFile {}

impl OutputFile for JniOutputFile {
    fn write(&mut self, data: &[u8]) -> io::Result<()> {
        let mut env = self
            .jvm
            .attach_current_thread()
            .map_err(|e| io::Error::other(e.to_string()))?;

        let len = data.len() as i32;

        let need_new = match &self.cached_array {
            Some(_) => data.len() > self.cached_array_len,
            None => true,
        };

        if need_new {
            let byte_array = env
                .new_byte_array(len)
                .map_err(|e| io::Error::other(e.to_string()))?;
            let global = env
                .new_global_ref(&byte_array)
                .map_err(|e| io::Error::other(e.to_string()))?;
            self.cached_array = Some(global);
            self.cached_array_len = data.len();
        }

        let raw = self.cached_array.as_ref().unwrap().as_raw();
        let byte_array = unsafe { JByteArray::from_raw(raw) };

        env.set_byte_array_region(&byte_array, 0, bytemuck_cast(data))
            .map_err(|e| io::Error::other(e.to_string()))?;

        unsafe {
            env.call_method_unchecked(
                &self.stream_ref,
                self.write_mid,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Void),
                &[
                    jni::sys::jvalue { l: raw },
                    jni::sys::jvalue { i: 0 },
                    jni::sys::jvalue { i: len },
                ],
            )
            .map_err(|e| io::Error::other(e.to_string()))?;
        }
        #[allow(clippy::forget_non_drop)]
        std::mem::forget(byte_array);
        self.pos += data.len() as u64;
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut env = self
            .jvm
            .attach_current_thread()
            .map_err(|e| io::Error::other(e.to_string()))?;
        unsafe {
            env.call_method_unchecked(
                &self.stream_ref,
                self.flush_mid,
                jni::signature::ReturnType::Primitive(jni::signature::Primitive::Void),
                &[],
            )
            .map_err(|e| io::Error::other(e.to_string()))?;
        }
        Ok(())
    }

    fn pos(&self) -> u64 {
        self.pos
    }
}

// ======================== JniInputFile ========================

struct JniInputFile {
    jvm: Arc<JavaVM>,
    input_file_ref: GlobalRef,
}

unsafe impl Send for JniInputFile {}
unsafe impl Sync for JniInputFile {}

impl InputFile for JniInputFile {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let mut env = self
            .jvm
            .attach_current_thread()
            .map_err(|e| io::Error::other(e.to_string()))?;

        let java_buf = env
            .new_byte_array(buf.len() as i32)
            .map_err(|e| io::Error::other(e.to_string()))?;

        env.call_method(
            &self.input_file_ref,
            "readFully",
            "(J[BII)V",
            &[
                JValue::Long(offset as jlong),
                JValue::Object(&java_buf),
                JValue::Int(0),
                JValue::Int(buf.len() as jint),
            ],
        )
        .map_err(|e| io::Error::other(e.to_string()))?;

        let i8_buf: &mut [i8] =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut i8, buf.len()) };
        env.get_byte_array_region(&java_buf, 0, i8_buf)
            .map_err(|e| io::Error::other(e.to_string()))?;

        Ok(())
    }
}

struct ReaderHandle {
    reader: Box<dyn ReaderAccess>,
    _input_file_ref: Option<GlobalRef>,
}

fn bytemuck_cast(data: &[u8]) -> &[i8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i8, data.len()) }
}

fn throw(env: &mut JNIEnv, msg: &str) {
    let _ = env.throw_new("java/lang/RuntimeException", msg);
}

struct WriterHandle {
    inner: MosaicWriter<JniOutputFile>,
    _stream_ref: GlobalRef,
}

// ======================== Writer ========================

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeWriterOpen(
    mut env: JNIEnv,
    _class: JClass,
    stream: JObject,
    arrow_schema_addr: jlong,
    num_buckets: jint,
    compression: jint,
    zstd_level: jint,
    row_group_max_size: jlong,
    max_dict_total_bytes: jint,
    max_dict_entries: jint,
    stats_columns: JObjectArray<'_>,
    page_size_threshold: jint,
) -> jlong {
    let raw_env = env.get_raw();
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if arrow_schema_addr == 0 {
            throw(&mut env, "null Arrow schema address");
            return 0;
        }

        let ffi_schema =
            unsafe { FFI_ArrowSchema::from_raw(arrow_schema_addr as *mut FFI_ArrowSchema) };
        let arrow_schema = match Schema::try_from(&ffi_schema) {
            Ok(s) => s,
            Err(e) => {
                throw(&mut env, &format!("Arrow schema import failed: {}", e));
                return 0;
            }
        };

        let stream_global = match env.new_global_ref(&stream) {
            Ok(g) => g,
            Err(e) => {
                throw(&mut env, &format!("failed to create global ref: {}", e));
                return 0;
            }
        };

        let write_mid = match env.get_method_id("java/io/OutputStream", "write", "([BII)V") {
            Ok(m) => m,
            Err(e) => {
                throw(&mut env, &format!("cannot find OutputStream.write: {}", e));
                return 0;
            }
        };
        let flush_mid = match env.get_method_id("java/io/OutputStream", "flush", "()V") {
            Ok(m) => m,
            Err(e) => {
                throw(&mut env, &format!("cannot find OutputStream.flush: {}", e));
                return 0;
            }
        };

        let jvm = match env.get_java_vm() {
            Ok(vm) => Arc::new(vm),
            Err(e) => {
                throw(&mut env, &format!("cannot get JavaVM: {}", e));
                return 0;
            }
        };

        let jni_stream = JniOutputFile {
            jvm,
            stream_ref: stream_global.clone(),
            write_mid,
            flush_mid,
            pos: 0,
            cached_array: None,
            cached_array_len: 0,
        };

        let stats_cols: Vec<String> = match env.get_array_length(&stats_columns) {
            Ok(len) if len > 0 => {
                let mut names = Vec::with_capacity(len as usize);
                for i in 0..len {
                    let obj = match env.get_object_array_element(&stats_columns, i) {
                        Ok(o) => o,
                        Err(_) => {
                            throw(&mut env, "failed to read stats_columns element");
                            return 0;
                        }
                    };
                    let jstr = JString::from(obj);
                    let s: String = match env.get_string(&jstr) {
                        Ok(s) => s.into(),
                        Err(_) => {
                            throw(
                                &mut env,
                                "failed to convert stats_columns element to string",
                            );
                            return 0;
                        }
                    };
                    names.push(s);
                }
                names
            }
            _ => Vec::new(),
        };

        let buckets = if num_buckets <= 0 {
            DEFAULT_NUM_BUCKETS
        } else {
            num_buckets as usize
        };

        let opts = WriterOptions {
            compression: compression as u8,
            zstd_level,
            num_buckets: buckets,
            row_group_max_size: row_group_max_size as u64,
            max_dict_total_bytes: max_dict_total_bytes as usize,
            max_dict_entries: max_dict_entries as usize,
            stats_columns: stats_cols,
            page_size_threshold: page_size_threshold as usize,
        };

        let writer = match MosaicWriter::new(jni_stream, &arrow_schema, opts) {
            Ok(w) => w,
            Err(e) => {
                throw(&mut env, &format!("writer open failed: {}", e));
                return 0;
            }
        };
        let handle = Box::new(WriterHandle {
            inner: writer,
            _stream_ref: stream_global,
        });
        Box::into_raw(handle) as jlong
    }));
    match result {
        Ok(val) => val,
        Err(e) => {
            let mut env = unsafe { JNIEnv::from_raw(raw_env).unwrap() };
            throw(&mut env, &panic_message(&e));
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeWriterClose(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    let raw_env = env.get_raw();
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if handle == 0 {
            return;
        }
        let writer = unsafe { &mut *(handle as *mut WriterHandle) };
        if let Err(e) = writer.inner.close() {
            throw(&mut env, &format!("close failed: {}", e));
        }
    }));
    if let Err(e) = result {
        let mut env = unsafe { JNIEnv::from_raw(raw_env).unwrap() };
        throw(&mut env, &panic_message(&e));
    }
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeWriterFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut WriterHandle)) };
    }
}

// ======================== Writer.estimatedSize ========================

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeWriterEstimatedSize(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jlong {
    if handle == 0 {
        return 0;
    }
    let writer = unsafe { &*(handle as *const WriterHandle) };
    writer.inner.estimated_file_size() as jlong
}

// ======================== Writer Stats ========================

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeWriterNumRowGroups(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jint {
    if handle == 0 {
        return 0;
    }
    let writer = unsafe { &*(handle as *const WriterHandle) };
    writer.inner.num_row_groups() as jint
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeWriterRowGroupStatNames<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    handle: jlong,
    rg_index: jint,
) -> JObjectArray<'local> {
    let null = JObjectArray::default();
    if handle == 0 {
        return null;
    }
    let writer = unsafe { &*(handle as *const WriterHandle) };
    let rg = rg_index as usize;
    if rg >= writer.inner.num_row_groups() {
        return null;
    }
    let stats = writer.inner.row_group_stats(rg);
    let schema = writer.inner.schema();
    let arr = match env.new_object_array(stats.len() as i32, "java/lang/String", &JObject::null()) {
        Ok(a) => a,
        Err(_) => return null,
    };
    for (i, st) in stats.iter().enumerate() {
        let name = &schema.columns[st.column_index].name;
        if let Ok(s) = env.new_string(name) {
            let _ = env.set_object_array_element(&arr, i as i32, &s);
        }
    }
    arr
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeWriterRowGroupStatNullCounts(
    env: JNIEnv,
    _class: JClass,
    handle: jlong,
    rg_index: jint,
) -> jlongArray {
    if handle == 0 {
        return std::ptr::null_mut();
    }
    let writer = unsafe { &*(handle as *const WriterHandle) };
    let rg = rg_index as usize;
    if rg >= writer.inner.num_row_groups() {
        return std::ptr::null_mut();
    }
    let stats = writer.inner.row_group_stats(rg);
    let counts: Vec<jlong> = stats.iter().map(|s| s.null_count as jlong).collect();
    match env.new_long_array(counts.len() as i32) {
        Ok(arr) => {
            let _ = env.set_long_array_region(&arr, 0, &counts);
            arr.into_raw()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeWriterRowGroupStatMins<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    handle: jlong,
    rg_index: jint,
) -> JObjectArray<'local> {
    let null = JObjectArray::default();
    if handle == 0 {
        return null;
    }
    let writer = unsafe { &*(handle as *const WriterHandle) };
    let rg = rg_index as usize;
    if rg >= writer.inner.num_row_groups() {
        return null;
    }
    let stats = writer.inner.row_group_stats(rg);
    let arr = match env.new_object_array(stats.len() as i32, "[B", &JObject::null()) {
        Ok(a) => a,
        Err(_) => return null,
    };
    for (i, st) in stats.iter().enumerate() {
        if let Some(v) = &st.min {
            let bytes = v.to_be_bytes();
            if let Ok(ba) = env.byte_array_from_slice(&bytes) {
                let _ = env.set_object_array_element(&arr, i as i32, &ba);
            }
        }
    }
    arr
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeWriterRowGroupStatMaxs<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    handle: jlong,
    rg_index: jint,
) -> JObjectArray<'local> {
    let null = JObjectArray::default();
    if handle == 0 {
        return null;
    }
    let writer = unsafe { &*(handle as *const WriterHandle) };
    let rg = rg_index as usize;
    if rg >= writer.inner.num_row_groups() {
        return null;
    }
    let stats = writer.inner.row_group_stats(rg);
    let arr = match env.new_object_array(stats.len() as i32, "[B", &JObject::null()) {
        Ok(a) => a,
        Err(_) => return null,
    };
    for (i, st) in stats.iter().enumerate() {
        if let Some(v) = &st.max {
            let bytes = v.to_be_bytes();
            if let Ok(ba) = env.byte_array_from_slice(&bytes) {
                let _ = env.set_object_array_element(&arr, i as i32, &ba);
            }
        }
    }
    arr
}

// ======================== Writer.writeBatch (Arrow C Data Interface) ========================

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeWriterWriteBatch(
    mut env: JNIEnv,
    _class: JClass,
    writer_handle: jlong,
    array_addr: jlong,
    schema_addr: jlong,
) {
    let raw_env = env.get_raw();
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if writer_handle == 0 {
            throw(&mut env, "null writer handle");
            return;
        }
        if array_addr == 0 || schema_addr == 0 {
            throw(&mut env, "null ArrowArray or ArrowSchema address");
            return;
        }
        let writer = unsafe { &mut *(writer_handle as *mut WriterHandle) };

        let ffi_array = array_addr as *mut FFI_ArrowArray;
        let ffi_schema = schema_addr as *mut FFI_ArrowSchema;

        let arr_owned = unsafe { FFI_ArrowArray::from_raw(ffi_array) };
        let schema_owned = unsafe { FFI_ArrowSchema::from_raw(ffi_schema) };
        let arr_data = match unsafe { arrow_array::ffi::from_ffi(arr_owned, &schema_owned) } {
            Ok(d) => d,
            Err(e) => {
                throw(&mut env, &format!("Arrow import failed: {}", e));
                return;
            }
        };

        let struct_array = StructArray::from(arr_data);
        let batch = RecordBatch::from(struct_array);
        if let Err(e) = writer.inner.write_batch(&batch) {
            throw(&mut env, &format!("write_batch failed: {}", e));
        }
    }));
    if let Err(e) = result {
        let mut env = unsafe { JNIEnv::from_raw(raw_env).unwrap() };
        throw(&mut env, &panic_message(&e));
    }
}

// ======================== Reader ========================

struct RowGroupReaderHandle {
    inner: RowGroupReader,
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeReaderOpen(
    mut env: JNIEnv,
    _class: JClass,
    input_file: JObject,
    file_length: jlong,
) -> jlong {
    let raw_env = env.get_raw();
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let global = match env.new_global_ref(&input_file) {
            Ok(g) => g,
            Err(e) => {
                throw(&mut env, &format!("failed to create global ref: {}", e));
                return 0;
            }
        };

        let length = file_length as u64;

        let jvm = match env.get_java_vm() {
            Ok(vm) => Arc::new(vm),
            Err(e) => {
                throw(&mut env, &format!("cannot get JavaVM: {}", e));
                return 0;
            }
        };

        let input = JniInputFile {
            jvm,
            input_file_ref: global.clone(),
        };

        match MosaicReader::new(input, length) {
            Ok(reader) => {
                let rh = ReaderHandle {
                    reader: Box::new(reader),
                    _input_file_ref: Some(global),
                };
                Box::into_raw(Box::new(rh)) as jlong
            }
            Err(e) => {
                throw(&mut env, &format!("open failed: {}", e));
                0
            }
        }
    }));
    match result {
        Ok(val) => val,
        Err(e) => {
            let mut env = unsafe { JNIEnv::from_raw(raw_env).unwrap() };
            throw(&mut env, &panic_message(&e));
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeReaderFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut ReaderHandle)) };
    }
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeReaderExportSchema(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    schema_addr: jlong,
) -> jint {
    if handle == 0 || schema_addr == 0 {
        return -1;
    }
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let rh = unsafe { &*(handle as *const ReaderHandle) };
        let reader = &*rh.reader;
        let schema = reader.schema();
        let fields: Vec<arrow_schema::Field> = schema
            .columns
            .iter()
            .map(|c| arrow_schema::Field::new(&c.name, c.data_type.clone(), c.nullable))
            .collect();
        let arrow_schema = Schema::new(fields);
        match FFI_ArrowSchema::try_from(&arrow_schema) {
            Ok(ffi_schema) => {
                unsafe {
                    ptr::write(schema_addr as *mut FFI_ArrowSchema, ffi_schema);
                }
                0
            }
            Err(_) => -1,
        }
    }));
    result.unwrap_or(-1)
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeReaderNumRowGroups(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jint {
    if handle == 0 {
        return 0;
    }
    let rh = unsafe { &*(handle as *const ReaderHandle) };
    let reader = &*rh.reader;
    reader.num_row_groups() as jint
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeReaderOpenRowGroup(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    rg_index: jint,
) -> jlong {
    let raw_env = env.get_raw();
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if handle == 0 {
            throw(&mut env, "null reader handle");
            return 0;
        }
        let rh = unsafe { &*(handle as *const ReaderHandle) };
        match rh.reader.row_group_reader(rg_index as usize) {
            Ok(rg) => {
                let rg_handle = Box::new(RowGroupReaderHandle { inner: rg });
                Box::into_raw(rg_handle) as jlong
            }
            Err(e) => {
                throw(&mut env, &format!("open row group failed: {}", e));
                0
            }
        }
    }));
    match result {
        Ok(val) => val,
        Err(e) => {
            let mut env = unsafe { JNIEnv::from_raw(raw_env).unwrap() };
            throw(&mut env, &panic_message(&e));
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeReaderSetProjection(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    columns: JObjectArray,
) {
    let raw_env = env.get_raw();
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if handle == 0 {
            throw(&mut env, "null reader handle");
            return;
        }
        let rh = unsafe { &mut *(handle as *mut ReaderHandle) };
        let col_names: Vec<String> = match env.get_array_length(&columns) {
            Ok(len) if len > 0 => {
                let mut names = Vec::with_capacity(len as usize);
                for i in 0..len {
                    let obj = match env.get_object_array_element(&columns, i) {
                        Ok(o) => o,
                        Err(_) => {
                            throw(&mut env, "failed to read columns array element");
                            return;
                        }
                    };
                    let jstr = JString::from(obj);
                    let s: String = match env.get_string(&jstr) {
                        Ok(js) => js.into(),
                        Err(_) => {
                            throw(&mut env, "failed to convert column name to string");
                            return;
                        }
                    };
                    names.push(s);
                }
                names
            }
            _ => Vec::new(),
        };
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        if let Err(e) = rh.reader.project(&col_refs) {
            throw(&mut env, &format!("set projection failed: {}", e));
        }
    }));
    if let Err(e) = result {
        let mut env = unsafe { JNIEnv::from_raw(raw_env).unwrap() };
        throw(&mut env, &panic_message(&e));
    }
}

// ======================== RowGroupReader ========================

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeRowGroupReaderNumRows(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jint {
    if handle == 0 {
        return 0;
    }
    let rg = unsafe { &*(handle as *const RowGroupReaderHandle) };
    rg.inner.num_rows() as jint
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeRowGroupReaderFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle != 0 {
        unsafe { drop(Box::from_raw(handle as *mut RowGroupReaderHandle)) };
    }
}

// ======================== Row Group Num Rows ========================

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeReaderRowGroupNumRows(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    rg_index: jint,
) -> jint {
    if handle == 0 {
        return -1;
    }
    let rh = unsafe { &*(handle as *const ReaderHandle) };
    match rh.reader.row_group_num_rows(rg_index as usize) {
        Ok(n) => n as jint,
        Err(_) => -1,
    }
}

// ======================== Row Group Stats ========================

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeReaderRowGroupStatNames<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    handle: jlong,
    rg_index: jint,
) -> JObjectArray<'local> {
    let null = JObjectArray::default();
    if handle == 0 {
        return null;
    }
    let rh = unsafe { &*(handle as *const ReaderHandle) };
    let stats = match rh.reader.row_group_stats(rg_index as usize) {
        Ok(s) => s,
        Err(_) => return null,
    };
    let schema = rh.reader.schema();
    let arr = match env.new_object_array(stats.len() as i32, "java/lang/String", &JObject::null()) {
        Ok(a) => a,
        Err(_) => return null,
    };
    for (i, st) in stats.iter().enumerate() {
        let name = &schema.columns[st.column_index].name;
        if let Ok(s) = env.new_string(name) {
            let _ = env.set_object_array_element(&arr, i as i32, &s);
        }
    }
    arr
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeReaderRowGroupStatNullCounts(
    env: JNIEnv,
    _class: JClass,
    handle: jlong,
    rg_index: jint,
) -> jlongArray {
    if handle == 0 {
        return std::ptr::null_mut();
    }
    let rh = unsafe { &*(handle as *const ReaderHandle) };
    let stats = match rh.reader.row_group_stats(rg_index as usize) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let counts: Vec<jlong> = stats.iter().map(|s| s.null_count as jlong).collect();
    match env.new_long_array(counts.len() as i32) {
        Ok(arr) => {
            let _ = env.set_long_array_region(&arr, 0, &counts);
            arr.into_raw()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeReaderRowGroupStatMins<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    handle: jlong,
    rg_index: jint,
) -> JObjectArray<'local> {
    let null = JObjectArray::default();
    if handle == 0 {
        return null;
    }
    let rh = unsafe { &*(handle as *const ReaderHandle) };
    let stats = match rh.reader.row_group_stats(rg_index as usize) {
        Ok(s) => s,
        Err(_) => return null,
    };
    let arr = match env.new_object_array(stats.len() as i32, "[B", &JObject::null()) {
        Ok(a) => a,
        Err(_) => return null,
    };
    for (i, st) in stats.iter().enumerate() {
        if let Some(v) = &st.min {
            let bytes = v.to_be_bytes();
            if let Ok(ba) = env.byte_array_from_slice(&bytes) {
                let _ = env.set_object_array_element(&arr, i as i32, &ba);
            }
        }
    }
    arr
}

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeReaderRowGroupStatMaxs<
    'local,
>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    handle: jlong,
    rg_index: jint,
) -> JObjectArray<'local> {
    let null = JObjectArray::default();
    if handle == 0 {
        return null;
    }
    let rh = unsafe { &*(handle as *const ReaderHandle) };
    let stats = match rh.reader.row_group_stats(rg_index as usize) {
        Ok(s) => s,
        Err(_) => return null,
    };
    let arr = match env.new_object_array(stats.len() as i32, "[B", &JObject::null()) {
        Ok(a) => a,
        Err(_) => return null,
    };
    for (i, st) in stats.iter().enumerate() {
        if let Some(v) = &st.max {
            let bytes = v.to_be_bytes();
            if let Ok(ba) = env.byte_array_from_slice(&bytes) {
                let _ = env.set_object_array_element(&arr, i as i32, &ba);
            }
        }
    }
    arr
}

// ======================== Columnar Read (Arrow C Data Interface) ========================

#[no_mangle]
pub extern "system" fn Java_org_apache_paimon_mosaic_NativeLib_nativeRowGroupReaderReadColumns(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    array_addr: jlong,
    schema_addr: jlong,
) -> jint {
    let raw_env = env.get_raw();
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        if handle == 0 {
            throw(&mut env, "null handle");
            return -1;
        }
        if array_addr == 0 || schema_addr == 0 {
            throw(&mut env, "null ArrowArray or ArrowSchema address");
            return -1;
        }
        let rg = unsafe { &mut *(handle as *mut RowGroupReaderHandle) };
        let batch = match rg.inner.read_columns() {
            Ok(b) => b,
            Err(e) => {
                throw(&mut env, &format!("read_columns failed: {}", e));
                return -1;
            }
        };

        let struct_array = StructArray::from(batch);
        match arrow_array::ffi::to_ffi(&struct_array.into()) {
            Ok((ffi_array, ffi_schema)) => {
                unsafe {
                    ptr::write(array_addr as *mut FFI_ArrowArray, ffi_array);
                    ptr::write(schema_addr as *mut FFI_ArrowSchema, ffi_schema);
                }
                0
            }
            Err(e) => {
                throw(&mut env, &format!("Arrow export failed: {}", e));
                -1
            }
        }
    }));
    match result {
        Ok(val) => val,
        Err(e) => {
            let mut env = unsafe { JNIEnv::from_raw(raw_env).unwrap() };
            throw(&mut env, &panic_message(&e));
            -1
        }
    }
}
