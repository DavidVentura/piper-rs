use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use unicode_normalization::UnicodeNormalization;

use crate::{PiperError, PiperResult};

pub fn load_tokens(tokens_path: &Path) -> PiperResult<HashMap<String, i64>> {
    let file = File::open(tokens_path).map_err(|e| {
        PiperError::FailedToLoadResource(format!(
            "Failed to open tokens `{}`: {}",
            tokens_path.display(),
            e
        ))
    })?;
    let reader = BufReader::new(file);
    let mut token_to_id = HashMap::new();

    for (line_number, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "Failed to read tokens `{}` line {}: {}",
                tokens_path.display(),
                line_number + 1,
                e
            ))
        })?;
        if line.trim().is_empty() {
            continue;
        }

        let Some(split_idx) = line.rfind(char::is_whitespace) else {
            return Err(PiperError::FailedToLoadResource(format!(
                "Invalid tokens `{}` line {}: missing token id",
                tokens_path.display(),
                line_number + 1
            )));
        };
        let (token, id_part) = line.split_at(split_idx);
        let token = token.to_string();
        let id = id_part.trim().parse::<i64>().map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "Invalid token id in `{}` line {}: {}",
                tokens_path.display(),
                line_number + 1,
                e
            ))
        })?;
        token_to_id.insert(token, id);
    }

    if token_to_id.is_empty() {
        return Err(PiperError::FailedToLoadResource(format!(
            "Tokens file `{}` did not contain any tokens",
            tokens_path.display()
        )));
    }

    Ok(token_to_id)
}

pub fn max_token_chars(token_to_id: &HashMap<String, i64>) -> usize {
    token_to_id
        .keys()
        .map(|token| token.chars().count())
        .max()
        .unwrap_or(1)
}

pub fn normalize_text(
    text: &str,
    token_to_id: &HashMap<String, i64>,
    max_token_chars: usize,
) -> String {
    let normalized: String = text.to_lowercase().nfc().collect();
    let mut out = String::new();
    let mut offset = 0;

    while offset < normalized.len() {
        let remaining = &normalized[offset..];
        let Some((token, matched_len)) =
            longest_token_match(remaining, token_to_id, max_token_chars)
        else {
            if let Some(ch) = remaining.chars().next() {
                offset += ch.len_utf8();
                continue;
            }
            break;
        };
        out.push_str(token);
        offset += matched_len;
    }

    out
}

pub fn tokenize_to_ids(
    text: &str,
    token_to_id: &HashMap<String, i64>,
    max_token_chars: usize,
) -> Vec<i64> {
    let normalized: String = text.to_lowercase().nfc().collect();
    let mut ids = Vec::new();
    let mut offset = 0;

    while offset < normalized.len() {
        let remaining = &normalized[offset..];
        let Some((token, matched_len)) =
            longest_token_match(remaining, token_to_id, max_token_chars)
        else {
            if let Some(ch) = remaining.chars().next() {
                offset += ch.len_utf8();
                continue;
            }
            break;
        };
        if let Some(&id) = token_to_id.get(token) {
            ids.push(id);
        }
        offset += matched_len;
    }

    ids
}

pub fn collect_dropped_tokens(
    text: &str,
    token_to_id: &HashMap<String, i64>,
    max_token_chars: usize,
) -> Vec<String> {
    let normalized: String = text.to_lowercase().nfc().collect();
    let mut dropped = Vec::new();
    let mut offset = 0;

    while offset < normalized.len() {
        let remaining = &normalized[offset..];
        let Some((_, matched_len)) = longest_token_match(remaining, token_to_id, max_token_chars)
        else {
            if let Some(ch) = remaining.chars().next() {
                dropped.push(ch.to_string());
                offset += ch.len_utf8();
                continue;
            }
            break;
        };
        offset += matched_len;
    }

    dropped
}

pub fn intersperse(ids: &[i64], blank_id: i64) -> Vec<i64> {
    let mut result = Vec::with_capacity(ids.len() * 2 + 1);
    result.push(blank_id);
    for &id in ids {
        result.push(id);
        result.push(blank_id);
    }
    result
}

fn longest_token_match<'a>(
    text: &'a str,
    token_to_id: &'a HashMap<String, i64>,
    max_token_chars: usize,
) -> Option<(&'a str, usize)> {
    let mut candidate_ends = Vec::new();
    for (idx, ch) in text.char_indices().take(max_token_chars) {
        candidate_ends.push(idx + ch.len_utf8());
    }

    for end in candidate_ends.into_iter().rev() {
        let candidate = &text[..end];
        if token_to_id.contains_key(candidate) {
            return Some((candidate, end));
        }
    }

    None
}
