use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use ndarray::Array1;
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;

use crate::vits_tokenize;
use crate::{build_session, Backend, PiperError, PiperResult};

const DEFAULT_NOISE_SCALE: f32 = 0.667;
const DEFAULT_NOISE_SCALE_W: f32 = 0.8;
const DEFAULT_LENGTH_SCALE: f32 = 1.0;

#[derive(Deserialize)]
struct DataConfig {
    sampling_rate: u32,
    #[serde(default)]
    add_blank: bool,
}

#[derive(Deserialize)]
struct RawConfig {
    data: DataConfig,
}

/// VITS model in the sherpa-onnx export format, using a lexicon for
/// character-to-phoneme lookup (e.g. Cantonese, Mandarin).
pub struct SherpaVitsModel {
    session: Session,
    sample_rate: u32,
    token_to_id: HashMap<String, i64>,
    blank_id: Option<i64>,
    lexicon: HashMap<String, Vec<String>>,
}

impl SherpaVitsModel {
    pub fn new(model_path: &Path, config_path: &Path, backend: &Backend) -> PiperResult<Self> {
        let file = File::open(config_path).map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "Failed to open config `{}`: {}",
                config_path.display(),
                e
            ))
        })?;
        let raw: RawConfig = serde_json::from_reader(file).map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "Failed to parse config `{}`: {}",
                config_path.display(),
                e
            ))
        })?;

        let config_dir = config_path.parent().ok_or_else(|| {
            PiperError::FailedToLoadResource(format!(
                "Config `{}` does not have a parent directory",
                config_path.display()
            ))
        })?;

        let token_to_id = vits_tokenize::load_tokens(&config_dir.join("tokens.txt"))?;
        let lexicon = load_lexicon(&config_dir.join("lexicon.txt"))?;
        let session = build_session(model_path, backend)?;

        let blank_id = if raw.data.add_blank {
            Some(*token_to_id.get("_").ok_or_else(|| {
                PiperError::FailedToLoadResource(
                    "add_blank is true but blank token `_` not found in tokens".to_string(),
                )
            })?)
        } else {
            None
        };

        Ok(Self {
            session,
            sample_rate: raw.data.sampling_rate,
            token_to_id,
            blank_id,
            lexicon,
        })
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        speaker_id: Option<i64>,
        speed: Option<f32>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let phonemes = self.phonemize(text)?;
        self.synthesize_phonemes(&phonemes, speaker_id, speed)
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

        let samples = infer(&mut self.session, &ids, speed.unwrap_or(1.0))?;
        Ok((samples, self.sample_rate))
    }

    /// Convert text to a space-separated phoneme string via lexicon lookup.
    pub fn phonemize(&self, text: &str) -> PiperResult<String> {
        Ok(lexicon_phonemize(text, &self.lexicon, &self.token_to_id))
    }

    pub fn token_ids(&self, text: &str) -> Vec<i64> {
        let phonemes = lexicon_phonemize(text, &self.lexicon, &self.token_to_id);
        self.text_to_ids(&phonemes)
    }

    pub fn voices(&self) -> Option<&HashMap<String, i64>> {
        None
    }

    fn text_to_ids(&self, phonemes: &str) -> Vec<i64> {
        let ids: Vec<i64> = phonemes
            .split_whitespace()
            .filter_map(|tok| self.token_to_id.get(tok).copied())
            .collect();
        match self.blank_id {
            Some(blank_id) => vits_tokenize::intersperse(&ids, blank_id),
            None => ids,
        }
    }
}

/// Map Chinese punctuation to ASCII equivalents that appear in the token vocabulary.
fn normalize_punctuation(ch: char) -> Option<char> {
    match ch {
        '，' | '、' => Some(','),
        '。' => Some('.'),
        '！' => Some('!'),
        '？' => Some('?'),
        '～' => Some('~'),
        '；' | '：' => Some(','),
        '—' | '─' => Some('─'),
        '\u{201c}' | '\u{201d}' | '「' | '」' | '『' | '』' => None,
        _ => None,
    }
}

/// Look up each character in the lexicon to produce a space-separated phoneme string.
/// Characters not in the lexicon are checked as punctuation tokens.
fn lexicon_phonemize(
    text: &str,
    lexicon: &HashMap<String, Vec<String>>,
    token_to_id: &HashMap<String, i64>,
) -> String {
    let mut phonemes = Vec::new();

    for ch in text.chars() {
        let key = ch.to_string();

        if let Some(phons) = lexicon.get(&key) {
            phonemes.extend(phons.iter().cloned());
            continue;
        }

        // Try normalized punctuation
        let punct = normalize_punctuation(ch).unwrap_or(ch);
        let punct_str = punct.to_string();
        if token_to_id.contains_key(&punct_str) {
            phonemes.push(punct_str);
        }
        // Unknown characters are silently dropped
    }

    phonemes.join(" ")
}

fn load_lexicon(path: &Path) -> PiperResult<HashMap<String, Vec<String>>> {
    let file = File::open(path).map_err(|e| {
        PiperError::FailedToLoadResource(format!(
            "Failed to open lexicon `{}`: {}",
            path.display(),
            e
        ))
    })?;
    let reader = BufReader::new(file);
    let mut lexicon = HashMap::new();

    for (line_number, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "Failed to read lexicon `{}` line {}: {}",
                path.display(),
                line_number + 1,
                e
            ))
        })?;

        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }

        // Format: "character phoneme1 phoneme2 ..."
        // The key is everything before the first space.
        let Some(space_idx) = trimmed.find(' ') else {
            continue; // No phonemes — skip
        };
        let key = trimmed[..space_idx].to_string();
        let phoneme_str = trimmed[space_idx + 1..].trim();
        if phoneme_str.is_empty() {
            continue; // Empty pronunciation
        }
        let phonemes: Vec<String> = phoneme_str.split_whitespace().map(String::from).collect();
        lexicon.insert(key, phonemes);
    }

    Ok(lexicon)
}

fn infer(session: &mut Session, ids: &[i64], speed: f32) -> PiperResult<Vec<f32>> {
    let input = Array1::<i64>::from_iter(ids.iter().copied());
    let input_len = Array1::<i64>::from_iter([ids.len() as i64]);
    let noise = Array1::<f32>::from_iter([DEFAULT_NOISE_SCALE]);
    let length = Array1::<f32>::from_iter([DEFAULT_LENGTH_SCALE / speed.max(0.1)]);
    let noise_w = Array1::<f32>::from_iter([DEFAULT_NOISE_SCALE_W]);

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
