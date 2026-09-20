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

const TOKEN_BASE: u8 = 0x80;
const MAX_RULES: usize = 128;
const PAIR_TABLE_SIZE: usize = 1 << 16;
const DENSE_PAIR_THRESHOLD: usize = 4_096;

pub fn is_ascii_only(names: &[&[u8]]) -> bool {
    names.iter().all(|name| name.iter().all(|&b| b & 0x80 == 0))
}

pub fn build_vocabulary(names: &[&[u8]]) -> Vec<[u8; 2]> {
    build_vocabulary_and_encode(names).0
}

pub(crate) fn build_vocabulary_and_encode(names: &[&[u8]]) -> (Vec<[u8; 2]>, Vec<Vec<u8>>) {
    let mut tokens: Vec<Vec<u8>> = names.iter().map(|name| name.to_vec()).collect();

    let mut rules = Vec::new();
    let use_dense_counter = tokens
        .iter()
        .map(|seq| seq.len().saturating_sub(1))
        .sum::<usize>()
        >= DENSE_PAIR_THRESHOLD;
    let mut pair_counts = use_dense_counter.then(|| vec![0u32; PAIR_TABLE_SIZE]);
    let mut touched_pairs = Vec::new();

    for _ in 0..MAX_RULES {
        let best = if let Some(pair_counts) = &mut pair_counts {
            most_frequent_pair_dense(&tokens, pair_counts, &mut touched_pairs)
        } else {
            most_frequent_pair_hashmap(&tokens)
        };
        match best {
            Some((left, right, count)) if count > 1 => {
                let new_token = TOKEN_BASE + rules.len() as u8;
                rules.push([left, right]);

                for seq in &mut tokens {
                    replace_pair(seq, left, right, new_token);
                }
            }
            _ => break,
        }
    }

    (rules, tokens)
}

fn most_frequent_pair_dense(
    tokens: &[Vec<u8>],
    pair_counts: &mut [u32],
    touched_pairs: &mut Vec<u16>,
) -> Option<(u8, u8, u32)> {
    debug_assert_eq!(pair_counts.len(), PAIR_TABLE_SIZE);
    for pair in touched_pairs.drain(..) {
        pair_counts[pair as usize] = 0;
    }

    let mut best_pair = 0usize;
    let mut best_count = 0u32;
    for seq in tokens {
        for pair in seq.windows(2) {
            let pair = ((pair[0] as usize) << 8) | pair[1] as usize;
            if pair_counts[pair] == 0 {
                touched_pairs.push(pair as u16);
            }
            pair_counts[pair] += 1;
            let count = pair_counts[pair];
            if count > best_count || (count == best_count && pair > best_pair) {
                best_pair = pair;
                best_count = count;
            }
        }
    }

    (best_count != 0).then_some(((best_pair >> 8) as u8, (best_pair & 0xff) as u8, best_count))
}

fn most_frequent_pair_hashmap(tokens: &[Vec<u8>]) -> Option<(u8, u8, u32)> {
    let mut pair_counts: HashMap<u16, u32> = HashMap::new();
    for seq in tokens {
        for pair in seq.windows(2) {
            let pair = (pair[0] as u16) << 8 | pair[1] as u16;
            *pair_counts.entry(pair).or_default() += 1;
        }
    }

    pair_counts
        .into_iter()
        .max_by_key(|&(pair, count)| (count, pair))
        .map(|(pair, count)| ((pair >> 8) as u8, pair as u8, count))
}

fn replace_pair(seq: &mut Vec<u8>, left: u8, right: u8, new_token: u8) {
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
    let mut tokens = name.to_vec();

    for (r, rule) in rules.iter().enumerate() {
        let left = rule[0];
        let right = rule[1];
        let new_token = TOKEN_BASE + r as u8;
        replace_pair(&mut tokens, left, right, new_token);
    }

    tokens
}

pub fn decode(encoded: &[u8], rules: &[[u8; 2]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(encoded.len() * 2);
    for &b in encoded {
        expand(b as usize, rules, &mut out);
    }
    out
}

fn expand(token: usize, rules: &[[u8; 2]], out: &mut Vec<u8>) {
    if token < TOKEN_BASE as usize {
        out.push(token as u8);
    } else {
        let idx = token - TOKEN_BASE as usize;
        expand(rules[idx][0] as usize, rules, out);
        expand(rules[idx][1] as usize, rules, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEGACY_TOKEN_BASE: u8 = 0x80;
    const LEGACY_MAX_RULES: usize = 128;

    fn reference_replace_pair(seq: &mut Vec<u16>, left: u16, right: u16, new_token: u16) {
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

    fn reference_build_vocabulary(names: &[&[u8]]) -> Vec<[u8; 2]> {
        let mut tokens: Vec<Vec<u16>> = names
            .iter()
            .map(|name| name.iter().map(|&b| b as u16).collect())
            .collect();
        let mut rules = Vec::new();

        for _ in 0..LEGACY_MAX_RULES {
            let mut pair_counts: HashMap<u32, u32> = HashMap::new();
            for seq in &tokens {
                for pair in seq.windows(2) {
                    let pair = (pair[0] as u32) << 16 | pair[1] as u32;
                    *pair_counts.entry(pair).or_default() += 1;
                }
            }

            match pair_counts
                .iter()
                .max_by_key(|&(&pair, &count)| (count, pair))
            {
                Some((&pair, &count)) if count > 1 => {
                    let left = (pair >> 16) as u16;
                    let right = pair as u16;
                    let new_token = LEGACY_TOKEN_BASE as u16 + rules.len() as u16;
                    rules.push([left as u8, right as u8]);
                    for seq in &mut tokens {
                        reference_replace_pair(seq, left, right, new_token);
                    }
                }
                _ => break,
            }
        }

        rules
    }

    fn legacy_decode(encoded: &[u8], rules: &[[u8; 2]]) -> Vec<u8> {
        let mut out = Vec::with_capacity(encoded.len() * 2);
        for &token in encoded {
            legacy_expand(token as usize, rules, &mut out);
        }
        out
    }

    fn legacy_expand(token: usize, rules: &[[u8; 2]], out: &mut Vec<u8>) {
        if token < LEGACY_TOKEN_BASE as usize {
            out.push(token as u8);
        } else {
            let rule = rules
                .get(token - LEGACY_TOKEN_BASE as usize)
                .expect("encoded token must reference a frozen legacy rule");
            legacy_expand(rule[0] as usize, rules, out);
            legacy_expand(rule[1] as usize, rules, out);
        }
    }

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
    fn test_build_vocabulary_and_encode_matches_separate_encoding() {
        let mut owned_names = vec![b"aaaaa".to_vec()];
        owned_names.extend((0..128).map(|i| {
            format!(
                "vehicle_data_collection_powertrain_group_{:03}_signal_measurement_{i:05}_value",
                i % 17
            )
            .into_bytes()
        }));
        let names: Vec<&[u8]> = owned_names.iter().map(Vec::as_slice).collect();

        let (rules, encoded) = build_vocabulary_and_encode(&names);
        assert_eq!(rules, reference_build_vocabulary(&names));
        assert_eq!(
            encoded,
            names
                .iter()
                .map(|name| encode(name, &rules))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_full_legacy_token_space_round_trips() {
        let owned_names: Vec<Vec<u8>> = (0..4096)
            .map(|i| {
                format!("prefix_{:04}_{}_{}_{}_suffix", i % 257, i, i % 97, i % 31).into_bytes()
            })
            .collect();
        let names: Vec<&[u8]> = owned_names.iter().map(Vec::as_slice).collect();

        let (rules, encoded) = build_vocabulary_and_encode(&names);

        assert_eq!(rules.len(), LEGACY_MAX_RULES);
        assert_eq!(rules, reference_build_vocabulary(&names));
        assert!(encoded.iter().flatten().any(|&token| token == 0xff));
        for (original, encoded_name) in names.iter().zip(&encoded) {
            assert_eq!(legacy_decode(encoded_name, &rules), *original);
            assert_eq!(decode(encoded_name, &rules), *original);
        }
    }

    #[test]
    fn test_ascii_only() {
        assert!(is_ascii_only(&[b"hello", b"world"]));
        assert!(!is_ascii_only(&[b"hello", &[0x80, 0x81]]));
    }

    #[test]
    fn test_dense_pair_counter_reuses_storage_and_preserves_tie_breaking() {
        let mut counts = vec![0; 1 << 16];
        let mut touched = Vec::new();

        let first = vec![vec![1, 2, 1, 3]];
        assert_eq!(
            most_frequent_pair_dense(&first, &mut counts, &mut touched),
            Some((2, 1, 1))
        );

        let second = vec![vec![255, 254, 255, 254]];
        assert_eq!(
            most_frequent_pair_dense(&second, &mut counts, &mut touched),
            Some((255, 254, 2))
        );
    }

    #[test]
    fn test_dense_pair_counter_matches_hashmap_reference() {
        let mut state = 1u64;
        for _ in 0..32 {
            let mut names = Vec::new();
            for _ in 0..256 {
                let len = 4 + (state as usize % 28);
                let mut name = Vec::with_capacity(len);
                for _ in 0..len {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    name.push(b'a' + ((state >> 32) % 26) as u8);
                }
                names.push(name);
            }
            let refs: Vec<&[u8]> = names.iter().map(Vec::as_slice).collect();
            assert_eq!(build_vocabulary(&refs), reference_build_vocabulary(&refs));
        }
    }
}
