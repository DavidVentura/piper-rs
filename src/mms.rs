use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use ndarray::Array1;
use ort::session::Session;
use ort::value::Tensor;
use unicode_normalization::UnicodeNormalization;

use crate::{build_session, Backend, PiperError, PiperResult};

const DEFAULT_NOISE_SCALE: f32 = 0.667;
const DEFAULT_NOISE_SCALE_W: f32 = 0.8;
const DEFAULT_LENGTH_SCALE: f32 = 1.0;
const DEFAULT_SAMPLE_RATE: u32 = 16_000;

pub struct MmsModel {
    session: Session,
    sample_rate: u32,
    token_to_id: HashMap<String, i64>,
    max_token_chars: usize,
}

impl MmsModel {
    pub fn new(model_path: &Path, tokens_path: &Path, backend: &Backend) -> PiperResult<Self> {
        let token_to_id = load_tokens(tokens_path)?;
        let max_token_chars = token_to_id
            .keys()
            .map(|token| token.chars().count())
            .max()
            .unwrap_or(1);
        let session = build_session(model_path, backend)?;
        let sample_rate = session
            .metadata()
            .ok()
            .and_then(|metadata| metadata.custom("sample_rate"))
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(DEFAULT_SAMPLE_RATE);

        Ok(Self {
            session,
            sample_rate,
            token_to_id,
            max_token_chars,
        })
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        speaker_id: Option<i64>,
        speed: Option<f32>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let normalized = self.phonemize(text)?;
        self.synthesize_phonemes(&normalized, speaker_id, speed)
    }

    pub fn synthesize_phonemes(
        &mut self,
        phonemes: &str,
        _speaker_id: Option<i64>,
        speed: Option<f32>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let ids = self.text_to_ids(phonemes);
        if ids.is_empty() {
            return Ok((Vec::new(), self.sample_rate));
        }

        let samples = infer(
            &mut self.session,
            &ids,
            speed.unwrap_or(1.0),
            DEFAULT_NOISE_SCALE,
            DEFAULT_NOISE_SCALE_W,
        )?;
        Ok((samples, self.sample_rate))
    }

    pub fn phonemize(&self, text: &str) -> PiperResult<String> {
        Ok(normalize_text(
            text,
            &self.token_to_id,
            self.max_token_chars,
        ))
    }

    pub fn voices(&self) -> Option<&HashMap<String, i64>> {
        None
    }

    fn text_to_ids(&self, text: &str) -> Vec<i64> {
        tokenize_to_ids(text, &self.token_to_id, self.max_token_chars)
    }
}

fn load_tokens(tokens_path: &Path) -> PiperResult<HashMap<String, i64>> {
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

fn normalize_text(
    text: &str,
    token_to_id: &HashMap<String, i64>,
    max_token_chars: usize,
) -> String {
    let normalized: String = text.nfc().collect();
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

fn tokenize_to_ids(
    text: &str,
    token_to_id: &HashMap<String, i64>,
    max_token_chars: usize,
) -> Vec<i64> {
    let normalized: String = text.nfc().collect();
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

fn infer(
    session: &mut Session,
    ids: &[i64],
    speed: f32,
    noise_scale: f32,
    noise_scale_w: f32,
) -> PiperResult<Vec<f32>> {
    let input = Array1::<i64>::from_iter(ids.iter().copied());
    let input_len = Array1::<i64>::from_iter([ids.len() as i64]);
    let noise = Array1::<f32>::from_iter([noise_scale]);
    let length = Array1::<f32>::from_iter([DEFAULT_LENGTH_SCALE / speed.max(0.1)]);
    let noise_w = Array1::<f32>::from_iter([noise_scale_w]);

    let input_t = Tensor::<i64>::from_array((
        [1, ids.len()],
        input.into_raw_vec_and_offset().0.into_boxed_slice(),
    ))
    .unwrap();
    let input_len_t = Tensor::<i64>::from_array((
        [1],
        input_len.into_raw_vec_and_offset().0.into_boxed_slice(),
    ))
    .unwrap();
    let noise_t =
        Tensor::<f32>::from_array(([1], noise.into_raw_vec_and_offset().0.into_boxed_slice()))
            .unwrap();
    let length_t =
        Tensor::<f32>::from_array(([1], length.into_raw_vec_and_offset().0.into_boxed_slice()))
            .unwrap();
    let noise_w_t =
        Tensor::<f32>::from_array(([1], noise_w.into_raw_vec_and_offset().0.into_boxed_slice()))
            .unwrap();

    let outputs = session
        .run(ort::inputs![
            input_t,
            input_len_t,
            noise_t,
            length_t,
            noise_w_t
        ])
        .map_err(|e| PiperError::InferenceError(format!("Inference failed: {}", e)))?;

    let (_, audio) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| PiperError::InferenceError(format!("Failed to extract output: {}", e)))?;

    Ok(audio.to_vec())
}
