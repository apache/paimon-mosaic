<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# mosaic CLI

A native command-line inspector for Mosaic files. It drives the read-only
`MosaicReader` API, so it needs no JVM and ships as a single native binary.

## Build & run

```bash
cargo run -p paimon-mosaic-cli -- <command> <file>   # from source
cargo install --path cli                             # install `mosaic`
mosaic <command> <file>
```

## Commands

| Command | Shows | Reads |
|---------|-------|-------|
| `schema` | column names, Arrow types, nullability, bucket | footer only |
| `meta`   | row groups, rows, per-column stats (null/min/max) | footer + index |
| `footer` | magic, version, buckets, compression | footer only |
| `buckets`| per-bucket layout and member columns | footer + index |
| `pages`  | per-column encoding + on-disk slot size | bucket data |
| `dictionary` | dictionary entries of a dict column (`-c`) | bucket data |
| `column-size` | on-disk bytes per column + total compression ratio | footer + index |
| `cat` / `head` | first N rows as a table | column data |
| `count` | total row count | footer + index |

Every command accepts `--json`. `cat`/`head` take `-n <N>`, `--all`, `-c a,b`
(projection) and `--where "col op val"` (one condition: `=`,`!=`,`>`,`>=`,`<`,`<=`);
`dictionary` takes `-c <col>`.

```text
$ mosaic schema data.mosaic
5 columns, 4 buckets
  id: Int32 not null [bucket 0]
  name: Utf8 [bucket 2]
  kind: Utf8 [bucket 1]

$ mosaic buckets data.mosaic
row group 0:
    bucket 0: monolithic 27B (uncompressed 59 B, 2.19x) [kind]
    bucket 1: paged 373B [flag, id]

$ mosaic column-size data.mosaic
  id: 349 B
  kind: 28 B
  total: 377 B (uncompressed 861 B, 2.28x)

$ mosaic pages data.mosaic
row group 0:
    flag: bucket 0 encoding=const slot=16B
    kind: bucket 1 encoding=dict slot=28B

$ mosaic cat data.mosaic -n 2 --json
{"id":0,"name":"user_0","kind":"a","score":0,"flag":7}
{"id":1,"name":"user_1","kind":"b","score":1.5,"flag":7}

$ mosaic cat data.mosaic --all --where "score>1" -c id,score
$ mosaic count data.mosaic
200
```

For C/C++ or Java callers, embed the format directly via the `ffi`
(`mosaic.h`) or `jni` crates rather than shelling out to this CLI.
