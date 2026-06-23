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
`MosaicReader` API, so it needs no JVM and ships as a single binary. For C/C++
or Java callers, embed the format via the `ffi` (`mosaic.h`) or `jni` crates
rather than shelling out to this tool.

## Install

```bash
cargo run -p paimon-mosaic-cli -- <command> <file>   # run from source
cargo install --path cli                             # install `mosaic`
mosaic <command> <file>
```

## Commands

Every command accepts `--json`.

| Command | Shows | Reads |
|---------|-------|-------|
| `schema` | column names, Arrow types, nullability, bucket | footer only |
| `meta` | row groups, rows, per-column stats (null/min/max) | footer + index |
| `footer` | magic, version, buckets, compression | footer only |
| `buckets` | per-bucket layout, member columns, ratio | footer + index |
| `pages` | per-column encoding + on-disk slot size | bucket data |
| `dictionary` | dictionary entries of a dict column | bucket data |
| `column-size` | bytes per column + total compression ratio | footer + index |
| `cat` / `head` | rows as a table | column data |
| `count` | total row count | footer + index |
| `convert` | import CSV or JSON into a new file | writes file |

## Inspect

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
```

## Query

`cat`/`head` take `-n <N>`, `--all`, `-c a,b` (projection) and
`--where "col op val"` (one condition: `=` `!=` `>` `>=` `<` `<=`).

```text
$ mosaic count data.mosaic
200

$ mosaic cat data.mosaic -n 2 --json
{"id":0,"name":"user_0","kind":"a","flag":7}
{"id":1,"name":"user_1","kind":"b","flag":7}

$ mosaic cat data.mosaic --all --where "id>100" -c id,kind
```

## Convert

Import CSV or JSON lines into a new Mosaic file; the schema is inferred.
`--stats id` builds min/max for those columns, which `cat --where` then uses to
skip row groups that cannot match.

```text
$ mosaic convert data.csv -o data.mosaic --stats id
wrote data.mosaic (200 rows, 5 columns)
```
