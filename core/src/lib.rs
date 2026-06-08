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

use arrow_array::{Array, ArrayRef};

pub mod bpe;

/// Propagate STRUCT nulls to a child array: where struct is null, child becomes null.
pub(crate) fn propagate_struct_nulls(
    struct_arr: &arrow_array::StructArray,
    child: &ArrayRef,
) -> ArrayRef {
    use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
    let num_rows = struct_arr.len();
    let mut null_bm = vec![0xFFu8; num_rows.div_ceil(8)];

    for i in 0..num_rows {
        let valid = !child.is_null(i) && !struct_arr.is_null(i);
        if !valid {
            null_bm[i / 8] &= !(1 << (i % 8));
        }
    }

    let new_null_buf = NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(null_bm), 0, num_rows));
    arrow_array::make_array(
        child
            .to_data()
            .into_builder()
            .null_bit_buffer(Some(new_null_buf.into_inner().into_inner()))
            .build()
            .unwrap(),
    )
}
pub mod bucket_reader;
pub mod bucket_writer;
pub mod reader;
pub mod schema;
pub mod spec;
pub mod stats;
pub mod types;
pub mod values;
pub mod varint;
pub mod writer;
