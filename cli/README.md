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

All inspection and query commands accept `--json`; `convert` and `convert-csv` write files.

| Command | Shows | Reads |
|---------|-------|-------|
| `schema` | column names, Arrow types, nullability, bucket | footer only |
| `meta` | row groups, rows, per-column stats (null/min/max) | footer + index |
| `footer` | magic, version, buckets, compression | footer only |
| `buckets` | per-bucket layout, member columns, ratio | footer + index |
| `pages` | per-column encoding + on-disk slot size | bucket data |
| `dictionary` | dictionary entries of a dict column | bucket data |
| `column-size` | bytes per column, exact for paged slots and approximate for shared monolithic buckets | footer + index + paged directories |
| `cat` | rows as a table (all rows by default; `-n` to limit) | column data |
| `head` | first N rows (default 10) | column data |
| `count` | total row count | footer + index |
| `convert` | import JSON into a new file | writes file |
| `convert-csv` | import CSV into a new file | writes file |

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
  total: 377 B

$ mosaic pages data.mosaic
row group 0:
    flag: bucket 0 encoding=const slot=16B
    kind: bucket 1 encoding=dict slot=28B
```

## Query

`cat` scans all rows by default (`-n` to limit);
`head` shows 10 rows by default. Both take `-c a,b` (projection),
`pages`/`column-size` take `-c` too, and `--where "col op val"` (one condition:
`=` `!=` `>` `>=` `<` `<=`; integers and floats compare exactly, so `=0.3`
only matches a stored 0.3; Date32 accepts epoch-day or `YYYY-MM-DD`).

```text
$ mosaic count data.mosaic
200

$ mosaic cat data.mosaic -n 2 --json
{"id":0,"name":"user_0","kind":"a","flag":7}
{"id":1,"name":"user_1","kind":"b","flag":7}

$ mosaic cat data.mosaic --where "id>100" -c id,kind
$ mosaic head data.mosaic --json
```

## Convert

Import a JSON data file (`.json`/`.ndjson`/`.jsonl`, one object per line) into
a new Mosaic file; the schema is inferred from the input unless `--schema` is
provided. A field with no non-null value cannot be inferred and is reported as
an error — pass `--schema` for such data.
An existing output is kept unless `--overwrite` is given.
`--schema` accepts the supported subset of an Avro record schema: primitive
fields except raw `bytes`, nullable unions with one non-null branch, arrays/maps, and `date`,
`time-millis`, `timestamp-*`, `local-timestamp-*`, `decimal`, and `uuid`
logical types. It is not a general Avro name resolver: nested records, enums,
named-type references, and non-decimal `fixed` fields are rejected.
Avro arrays and maps are converted recursively. Avro `timestamp-*` logical
types remain UTC instants, while `local-timestamp-*` remains timezone-free;
inputs with an explicit offset are rejected for local timestamps. Decimal
inputs must be exactly representable at the schema scale (extra trailing zero
digits are accepted). UUID values are validated during import but stored as
`Utf8`; the Avro logical-type marker is not preserved in Mosaic. Unknown logical
types are ignored and use their underlying Avro type.
Use `-c`/`--column` to project top-level fields; each occurrence
accepts a comma-separated list.
`--stats id` builds min/max for those columns, which `cat --where` then uses
to skip row groups that cannot match.
`convert` accepts JSON inputs only; use `convert-csv` for CSV inputs.
Each JSON root value is limited to 16 MiB and 100,000 structural units
(container boundaries, separators, and string starts) before serde or Arrow
materializes it. Schema-aware decimal normalization is subject to the same
16 MiB output limit.

```text
$ mosaic convert data.json -o data.mosaic
$ mosaic convert data.json -o data.mosaic --schema schema.avsc
$ mosaic convert data.json -o data.mosaic -c id,kind
$ mosaic convert data.json -o data.mosaic --stats id
```

## Convert CSV

Import CSV into a new Mosaic file, either with an Avro record schema file
given via `--schema`, or with a schema inferred from the CSV data. For multiple
inputs with headers, inferred fields are matched by name rather than position,
and compatible `Int64`/`Float64` fields are promoted to `Float64`. In every
inferred `Float64` field, bare integer literals must be exactly representable;
decimal and exponent forms use `Float64` rounding, but finite literals that
overflow to infinity are rejected. Compatible timestamp precisions are widened
to the finest observed unit; other field-name or type conflicts require
`--schema`. Empty and header-only files are skipped.
CSV cannot say what type an all-empty column is, so columns inferred as Arrow
`Null` fall back to nullable `Utf8` and their values remain null. Use
`--schema` when such a column should have another type. When a CSV schema is
inferred, `--require col` marks an inferred field as not null (repeat it for
multiple fields); combining `--require` with `--schema` is rejected.

With `--schema`, fields are matched to CSV columns by header name (by position
with `--no-header`); a schema field absent from the header is filled with
nulls, but the conversion fails if the field is required or if no schema field
matches the header at all.
Backslash escaping is disabled by default so literal values such as
`C:\temp\file` are preserved; pass `--escape '\'` only for CSV dialects that
use a separate escape character. `--delimiter`, `--quote`, and `--skip-lines`
control the CSV dialect; `--skip-lines N` drops N physical lines before CSV
parsing begins without decoding skipped bytes as UTF-8. Every logical CSV
record is limited to 65,535 columns and 64 MiB of decoded field payload before
csv or Arrow materializes it. Schema inference snapshots the guarded input, so
decode replays exactly the bytes used for inference even if the source path is
replaced or modified. Avro `bytes`, `array`, and `map` fields are rejected
because the CSV decoder supports scalar text fields only. `--header` and
`--no-header` are mutually exclusive. Inferred local timestamps reject
explicit offsets and direct users to `--schema` to select timestamp semantics;
second-precision values are stored with millisecond precision. Explicit-schema
local timestamps and decimals use the same offset and exact-scale rules as JSON
conversion.
`--stats id` builds min/max for those columns, which `cat --where` then uses to
skip row groups that cannot match.

```text
$ mosaic convert-csv data.csv -o data.mosaic

$ mosaic convert-csv data.csv -o data.mosaic --schema schema.avsc
$ mosaic convert-csv data.csv -o data.mosaic --require id --require ts
$ mosaic convert-csv data.csv -o data.mosaic --stats id
```
