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

use std::collections::HashMap;

const TOKEN_BASE: usize = 0x80;
const MAX_RULES: usize = 128;

pub fn is_ascii_only(names: &[&[u8]]) -> bool {
    names.iter().all(|name| name.iter().all(|&b| b & 0x80 == 0))
}

pub fn build_vocabulary(names: &[&[u8]]) -> Vec<[u8; 2]> {
    let mut tokens: Vec<Vec<u16>> = names
        .iter()
        .map(|name| name.iter().map(|&b| b as u16).collect())
        .collect();

    let mut rules = Vec::new();

    for _ in 0..MAX_RULES {
        let mut pair_counts: HashMap<u32, u32> = HashMap::new();
        for seq in &tokens {
            for w in seq.windows(2) {
                let pair = (w[0] as u32) << 16 | w[1] as u32;
                *pair_counts.entry(pair).or_default() += 1;
            }
        }

        let best = pair_counts.iter().max_by_key(|&(&k, &v)| (v, k));
        match best {
            Some((&pair, &count)) if count > 1 => {
                let left = (pair >> 16) as u16;
                let right = (pair & 0xFFFF) as u16;
                let new_token = (TOKEN_BASE + rules.len()) as u16;
                rules.push([left as u8, right as u8]);

                for seq in &mut tokens {
                    replace_pair(seq, left, right, new_token);
                }
            }
            _ => break,
        }
    }

    rules
}

fn replace_pair(seq: &mut Vec<u16>, left: u16, right: u16, new_token: u16) {
    let mut i = 0;
    let mut out = 0;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == left && seq[i + 1] == right {
            seq[out] = new_token;
            i += 2;
        } else {
            seq[out] = seq[i];
            i += 1;
        }
        out += 1;
    }
    seq.truncate(out);
}

pub fn encode(name: &[u8], rules: &[[u8; 2]]) -> Vec<u8> {
    let mut tokens: Vec<u16> = name.iter().map(|&b| b as u16).collect();

    for (r, rule) in rules.iter().enumerate() {
        let left = rule[0] as u16;
        let right = rule[1] as u16;
        let new_token = (TOKEN_BASE + r) as u16;
        replace_pair(&mut tokens, left, right, new_token);
    }

    tokens.iter().map(|&t| t as u8).collect()
}

pub fn decode(encoded: &[u8], rules: &[[u8; 2]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(encoded.len() * 2);
    for &b in encoded {
        expand(b as usize, rules, &mut out);
    }
    out
}

fn expand(token: usize, rules: &[[u8; 2]], out: &mut Vec<u8>) {
    if token < TOKEN_BASE {
        out.push(token as u8);
    } else {
        let idx = token - TOKEN_BASE;
        expand(rules[idx][0] as usize, rules, out);
        expand(rules[idx][1] as usize, rules, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_encode() {
        let names: Vec<&[u8]> = vec![
            b"engine_coolant_temp",
            b"engine_coolant_pressure",
            b"engine_oil_temp",
            b"engine_oil_pressure",
        ];
        let rules = build_vocabulary(&names);
        assert!(!rules.is_empty());

        for &name in &names {
            let encoded = encode(name, &rules);
            assert!(encoded.len() <= name.len());
        }
    }

    #[test]
    fn test_ascii_only() {
        assert!(is_ascii_only(&[b"hello", b"world"]));
        assert!(!is_ascii_only(&[b"hello", &[0x80, 0x81]]));
    }
}
