<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Integer writer memory

The writer compacts `Int16`, `Int32`, and `Int64` value buffers automatically.
This includes integer children of lists/maps and their `Int32` length columns.
`Int8` already occupies one byte. Floating-point, date/time, timestamp, decimal,
string and binary caches retain their existing representation. Input Arrow
schemas, `WriterOptions`, the C/JNI APIs and on-disk column types are unchanged.

## Representation and range changes

Each integer block contains at most 4,096 non-null values. A block uses 1, 2, 4,
or 8 bytes per value, capped at the logical type width. A block-sized OR reduction
selects the required width before a batch copy. Positive values such as widened
UInt16 data therefore occupy two bytes even in an Int32 column. Negative values
retain their logical two's-complement width.

A later outlier widens only the current block, copying backwards in the same
vector. Completed blocks stay compact. Range checking and promotion never scan
or expand the whole buffered column. A new block can narrow again. Reset drops
block payload allocations; the allocator may retain freed pages in process RSS.

Dictionary keys, constant tracking, statistics, dictionary/plain selection,
row-group accounting and page-layout selection still use logical values and
logical byte sizes. Saving memory does not cause the same row group to accumulate
more rows. Existing large row-group settings remain effective. Constant columns
keep dictionary tracking pending; the first differing value activates the
dictionary with the constant prefix as its first key. Entry limits and reset
retain the same tracking lifecycle as non-integer columns.

## Encoding and compression

Integer values are expanded into a reusable 16 KiB scratch buffer at encoding
time. Nullable append gathers at most 4,096 i64 values (32 KiB). Dictionary indices
are also emitted incrementally. Buckets containing compact integer columns use
this sink-based encoder rather than assembling the full bucket/page first.
Uncompressed integer buckets use an additional 16 KiB output buffer to coalesce
small writes before invoking file or C/JNI output callbacks. Pending bytes are
drained explicitly at the end of the bucket; errors or unwinding discard them
without retrying a potentially partial sink write.

Encoded units up to 1 MiB use bounded staging and the original one-shot Zstd
compressor to avoid streaming overhead on small pages. Larger units stream to
one Zstd frame per original bucket/page, with the exact uncompressed size pledged
to Zstd. Scratch chunks do not create new pages, frames or row groups.

Paged files place slot lengths before their compressed data, so the writer still
retains compressed column slots until it can emit the directory. This change
does not bound all writer memory: row-group caches, null bitmaps, dictionaries,
Zstd context/window memory, compressed slots and the caller's Arrow input remain.
It requires neither seekable output callbacks nor temporary files. The public
`BucketWriter::finish`/`finish_paged` convenience methods still materialize their
returned data; `MosaicWriter` uses the incremental path for integer buckets.

Streaming Zstd output can differ bytewise from one-shot compression for some
inputs. Schema, decoded values, encoding choices and compression-unit boundaries
remain unchanged; file-size validation is part of the test matrix.

## Validation and reproduction

Unit tests cover width changes within/across blocks, signed extremes, nullable
and sliced Arrow arrays, ALL_NULL/CONST/DICT/PLAIN transitions, mixed columns,
logical row-group accounting, statistics, reset, bounded expansion, sink errors,
and Zstd block boundaries. A wide-buffer reference path exercises the previous
writer/compression behavior. The comparison matrix checks input-value round trips,
row-group/schema/statistics equality and no file-size increase in that corpus.

Run the synthetic example on the base and candidate revisions using independent
build directories and separate processes:

```sh
cargo test --locked --workspace
cargo clippy --all-targets --workspace -- -D warnings
cargo build --release -p paimon-mosaic-core --example integer_writer_memory
/usr/bin/time -v target/release/examples/integer_writer_memory 250000 20 paged unsigned output.mosaic
```

Use `monolithic` for the other bucket layout, `negative` for full-range signed
random data, and `outlier` for a late full-range value. Copy the same example to
the baseline revision before building. Compare output hashes, kernel peak RSS,
phase RSS/PSS snapshots, CPU time and append/close timings. Run each pair in both
orders; do not compile or run the two workloads concurrently.

ARM64 Linux results against official `main` at `c082354` (2026-10-12, Rust
1.97.1, Neoverse-N2; 5 repetitions per version/scenario, separate build directories,
alternating version order, 20 batches of 250,000 rows, three integer columns,
Zstd level 1, dictionary disabled). These are fresh measurements of this change
on that upstream baseline. Medians:

| Layout | Input | Peak RSS before → after (MiB) | Append total before → after (ms) | Close before → after (ms) |
| --- | --- | ---: | ---: | ---: |
| Paged | Widened unsigned | 232.3 → 79.9 | 47.4 → 14.1 | 271.4 → 231.7 |
| Paged | Full-range signed | 276.7 → 142.6 | 46.7 → 23.4 | 136.8 → 75.4 |
| Paged | Late outlier | 232.3 → 79.9 | 47.3 → 14.7 | 270.3 → 232.9 |
| Monolithic | Widened unsigned | 180.7 → 42.3 | 46.8 → 14.2 | 233.1 → 219.4 |
| Monolithic | Full-range signed | 210.1 → 75.7 | 47.2 → 23.4 | 90.4 → 50.9 |
| Monolithic | Late outlier | 180.7 → 42.3 | 46.6 → 13.8 | 231.8 → 218.6 |

All six scenarios produced byte-identical files across all baseline/candidate
runs. All 90 million integer values in the six candidate files were also decoded
and checked against the generated input. Native ARM64 core tests passed (183),
and the release FFI library built successfully. An additional review harness
performed 1,728 differential comparisons against the original `c082354` bucket
encoder, including nested List/Map columns, nulls, slices, dictionary limits and
reset; this harness is separate from the committed test suite.

Figures are medians of synthetic runs, not measured savings for a particular
Android application. Source Arrow arrays are still widened by the caller;
avoiding those arrays requires a separate input API/client integration change.
