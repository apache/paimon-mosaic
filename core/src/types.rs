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

use arrow_schema::{DataType, Field, Fields, TimeUnit};

use crate::varint;

pub const NANOS_PER_MILLI: i64 = 1_000_000;

pub fn fixed_width(dt: &DataType) -> i32 {
    match dt {
        DataType::Boolean | DataType::Int8 => 1,
        DataType::Int16 => 2,
        DataType::Int32 | DataType::Date32 | DataType::Time32(_) | DataType::Float32 => 4,
        DataType::Int64 | DataType::Float64 => 8,
        DataType::Decimal128(p, _) if *p <= 18 => 8,
        DataType::Timestamp(TimeUnit::Millisecond | TimeUnit::Microsecond, _) => 8,
        DataType::Timestamp(TimeUnit::Nanosecond, _) => 12,
        DataType::Struct(fields) if is_timestamp_nanos_struct(fields) => 12,
        _ => -1,
    }
}

pub fn is_timestamp_nanos(dt: &DataType) -> bool {
    matches!(dt, DataType::Timestamp(TimeUnit::Nanosecond, _))
        || matches!(dt, DataType::Struct(fields) if is_timestamp_nanos_struct(fields))
}

pub fn is_timestamp_nanos_struct(fields: &Fields) -> bool {
    fields.len() == 2
        && fields[0].name() == "millis"
        && *fields[0].data_type() == DataType::Int64
        && fields[1].name() == "nanos_of_milli"
        && *fields[1].data_type() == DataType::Int32
}

pub fn is_valid_nanos_of_milli(nanos: i32) -> bool {
    (0..NANOS_PER_MILLI as i32).contains(&nanos)
}

pub fn ns_to_millis_nanos(ns: i64) -> (i64, i32) {
    (
        ns.div_euclid(NANOS_PER_MILLI),
        ns.rem_euclid(NANOS_PER_MILLI) as i32,
    )
}

pub fn millis_nanos_to_ns(millis: i64, nanos: i32) -> io::Result<i64> {
    if !is_valid_nanos_of_milli(nanos) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid nanos_of_milli: {}", nanos),
        ));
    }
    let ns = millis as i128 * NANOS_PER_MILLI as i128 + nanos as i128;
    if ns < i64::MIN as i128 || ns > i64::MAX as i128 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "timestamp ns overflow",
        ));
    }
    Ok(ns as i64)
}

pub fn validate_data_type(dt: &DataType) -> Result<(), String> {
    match dt {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Float32
        | DataType::Float64
        | DataType::Date32
        | DataType::Utf8
        | DataType::Binary => Ok(()),
        DataType::Time32(TimeUnit::Millisecond) => Ok(()),
        DataType::Decimal128(p, _s) => {
            if *p == 0 || *p > 38 {
                Err(format!("DECIMAL precision must be 1..38, got {}", p))
            } else {
                Ok(())
            }
        }
        DataType::Timestamp(unit, _) => match unit {
            TimeUnit::Millisecond | TimeUnit::Microsecond | TimeUnit::Nanosecond => Ok(()),
            _ => Err(format!("unsupported Timestamp unit: {:?}", unit)),
        },
        DataType::Struct(fields) if is_timestamp_nanos_struct(fields) => Ok(()),
        DataType::List(field) => {
            let elem = field.data_type();
            if let DataType::Struct(fields) = elem {
                if is_timestamp_nanos_struct(fields) {
                    return Err("ARRAY<legacy timestamp nanos struct> is not supported".to_string());
                }
            }
            validate_data_type(elem)
        }
        DataType::Map(entries_field, sorted) => {
            if *sorted {
                return Err("sorted MAP is not supported".to_string());
            }
            if let DataType::Struct(fields) = entries_field.data_type() {
                if fields.len() != 2 {
                    return Err("MAP entries struct must have exactly 2 fields".to_string());
                }
                let key_dt = fields[0].data_type();
                if matches!(key_dt, DataType::List(_) | DataType::Map(_, _)) {
                    return Err("MAP key type cannot be ARRAY or MAP".to_string());
                }
                validate_data_type(key_dt)?;
                validate_data_type(fields[1].data_type())
            } else {
                Err("MAP entries field must be a Struct".to_string())
            }
        }
        _ => Err(format!("unsupported DataType: {:?}", dt)),
    }
}

pub fn data_type_to_type_byte(dt: &DataType) -> u8 {
    match dt {
        DataType::Boolean => 0,
        DataType::Int8 => 1,
        DataType::Int16 => 2,
        DataType::Int32 => 3,
        DataType::Int64 => 4,
        DataType::Float32 => 5,
        DataType::Float64 => 6,
        DataType::Date32 => 7,
        DataType::Utf8 => 10,
        DataType::Binary => 13,
        DataType::Decimal128(_, _) => 14,
        DataType::Time32(_) => 15,
        DataType::Timestamp(_, None) => 16,
        DataType::Timestamp(_, Some(_)) => 17,
        DataType::Struct(fields) if is_timestamp_nanos_struct(fields) => 16,
        DataType::List(_) => 18,
        DataType::Map(_, _) => 19,
        _ => panic!("unsupported DataType for serialization: {:?}", dt),
    }
}

pub fn precision_of(dt: &DataType) -> u32 {
    match dt {
        DataType::Decimal128(p, _) => *p as u32,
        DataType::Timestamp(TimeUnit::Millisecond, _) => 3,
        DataType::Timestamp(TimeUnit::Microsecond, _) => 6,
        DataType::Timestamp(TimeUnit::Nanosecond, _) => 9,
        DataType::Struct(fields) if is_timestamp_nanos_struct(fields) => 9,
        DataType::Time32(TimeUnit::Millisecond) => 3,
        _ => 0,
    }
}

pub fn scale_of(dt: &DataType) -> u32 {
    match dt {
        DataType::Decimal128(_, s) => *s as u32,
        _ => 0,
    }
}

pub fn serialize_field(field: &Field, buf: &mut Vec<u8>) {
    let dt = field.data_type();
    let type_byte = data_type_to_type_byte(dt);
    buf.push(type_byte);
    buf.push(if field.is_nullable() { 1 } else { 0 });
    match dt {
        DataType::Decimal128(p, s) => {
            varint::encode(buf, *p as u32);
            varint::encode(buf, *s as u32);
        }
        DataType::Time32(_) => {
            varint::encode(buf, precision_of(dt));
        }
        DataType::Timestamp(unit, tz) => {
            let p = match unit {
                TimeUnit::Millisecond => 3u32,
                TimeUnit::Microsecond => 6u32,
                TimeUnit::Nanosecond => 9u32,
                _ => 0,
            };
            varint::encode(buf, p);
            if let Some(tz) = tz {
                let tz_bytes = tz.as_bytes();
                varint::encode(buf, tz_bytes.len() as u32);
                buf.extend_from_slice(tz_bytes);
            }
        }
        DataType::Struct(fields) if is_timestamp_nanos_struct(fields) => {
            varint::encode(buf, 9u32);
        }
        DataType::List(element_field) => {
            let name_bytes = element_field.name().as_bytes();
            varint::encode(buf, name_bytes.len() as u32);
            buf.extend_from_slice(name_bytes);
            serialize_field(element_field, buf);
        }
        DataType::Map(entries_field, _sorted) => {
            // entries field name
            let entries_name = entries_field.name().as_bytes();
            varint::encode(buf, entries_name.len() as u32);
            buf.extend_from_slice(entries_name);
            if let DataType::Struct(fields) = entries_field.data_type() {
                // key field name + type
                let key_name = fields[0].name().as_bytes();
                varint::encode(buf, key_name.len() as u32);
                buf.extend_from_slice(key_name);
                serialize_field(&fields[0], buf);
                // value field name + type
                let val_name = fields[1].name().as_bytes();
                varint::encode(buf, val_name.len() as u32);
                buf.extend_from_slice(val_name);
                serialize_field(&fields[1], buf);
            }
        }
        _ => {}
    }
}

fn read_utf8_field_name(
    buf: &[u8],
    pos: &mut usize,
    len: usize,
    context: &str,
) -> Result<String, std::io::Error> {
    if *pos + len > buf.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("type: not enough bytes for {} field name", context),
        ));
    }
    let name = std::str::from_utf8(&buf[*pos..*pos + len])
        .map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("type: invalid UTF-8 in {} field name", context),
            )
        })?
        .to_string();
    *pos += len;
    Ok(name)
}

pub fn deserialize_field(name: &str, buf: &[u8], pos: &mut usize) -> Result<Field, std::io::Error> {
    if *pos + 1 >= buf.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "type: not enough bytes for type header",
        ));
    }
    let type_byte = buf[*pos];
    *pos += 1;
    let nullable = buf[*pos] != 0;
    *pos += 1;

    let dt = match type_byte {
        0 => DataType::Boolean,
        1 => DataType::Int8,
        2 => DataType::Int16,
        3 => DataType::Int32,
        4 => DataType::Int64,
        5 => DataType::Float32,
        6 => DataType::Float64,
        7 => DataType::Date32,
        8 | 9 => {
            // Char/VarChar — skip length, map to Utf8
            let _length = varint::decode(buf, pos)?;
            DataType::Utf8
        }
        10 => DataType::Utf8,
        11 | 12 => {
            // Binary/VarBinary — skip length, map to Binary
            let _length = varint::decode(buf, pos)?;
            DataType::Binary
        }
        13 => DataType::Binary,
        14 => {
            let precision = varint::decode(buf, pos)?;
            let scale = varint::decode(buf, pos)?;
            DataType::Decimal128(precision as u8, scale as i8)
        }
        15 => {
            let _precision = varint::decode(buf, pos)?;
            DataType::Time32(TimeUnit::Millisecond)
        }
        16 => {
            let precision = varint::decode(buf, pos)?;
            if precision <= 3 {
                DataType::Timestamp(TimeUnit::Millisecond, None)
            } else if precision <= 6 {
                DataType::Timestamp(TimeUnit::Microsecond, None)
            } else {
                DataType::Timestamp(TimeUnit::Nanosecond, None)
            }
        }
        17 => {
            let precision = varint::decode(buf, pos)?;
            let tz_len = varint::decode(buf, pos)? as usize;
            if *pos + tz_len > buf.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "type: not enough bytes for timezone string",
                ));
            }
            let tz_str = std::str::from_utf8(&buf[*pos..*pos + tz_len]).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "type: invalid UTF-8 timezone",
                )
            })?;
            *pos += tz_len;
            let tz: std::sync::Arc<str> = std::sync::Arc::from(tz_str);
            if precision <= 3 {
                DataType::Timestamp(TimeUnit::Millisecond, Some(tz))
            } else if precision <= 6 {
                DataType::Timestamp(TimeUnit::Microsecond, Some(tz))
            } else {
                DataType::Timestamp(TimeUnit::Nanosecond, Some(tz))
            }
        }
        18 => {
            let name_len = varint::decode(buf, pos)? as usize;
            if *pos + name_len > buf.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "type: not enough bytes for ARRAY element field name",
                ));
            }
            let element_name = std::str::from_utf8(&buf[*pos..*pos + name_len])
                .map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "type: invalid UTF-8 in ARRAY element field name",
                    )
                })?
                .to_string();
            *pos += name_len;
            let element_field = deserialize_field(&element_name, buf, pos)?;
            DataType::List(std::sync::Arc::new(element_field))
        }
        19 => {
            // entries field name
            let entries_name_len = varint::decode(buf, pos)? as usize;
            let entries_name = read_utf8_field_name(buf, pos, entries_name_len, "MAP entries")?;
            // key field name + type
            let key_name_len = varint::decode(buf, pos)? as usize;
            let key_name = read_utf8_field_name(buf, pos, key_name_len, "MAP key")?;
            let key_field = deserialize_field(&key_name, buf, pos)?;
            // value field name + type
            let val_name_len = varint::decode(buf, pos)? as usize;
            let val_name = read_utf8_field_name(buf, pos, val_name_len, "MAP value")?;
            let value_field = deserialize_field(&val_name, buf, pos)?;
            let entries_field = Field::new(
                &entries_name,
                DataType::Struct(Fields::from(vec![key_field, value_field])),
                false,
            );
            DataType::Map(std::sync::Arc::new(entries_field), false)
        }
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown type tag: {}", type_byte),
            ));
        }
    };

    Ok(Field::new(name, dt, nullable))
}
