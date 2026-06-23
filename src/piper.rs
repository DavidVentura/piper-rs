use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use mnn_sys::{ModuleEngine, NamedInput, TensorData};
use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

use crate::{espeak_phonemize, load_module, vits_tokenize, Backend, PiperError, PiperResult};

const BOS: char = '^';
const EOS: char = '$';
const PAD: char = '_';

#[derive(Deserialize)]
struct AudioConfig {
    sample_rate: u32,
}

#[derive(Deserialize)]
struct ESpeakConfig {
    voice: String,
}

#[derive(Deserialize, Clone)]
struct InferenceConfig {
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
}

#[derive(Deserialize)]
struct RawConfig {
    audio: AudioConfig,
    espeak: ESpeakConfig,
    inference: InferenceConfig,
    #[serde(default = "default_phoneme_type")]
    phoneme_type: String,
    num_speakers: u32,
    #[serde(default)]
    speaker_id_map: HashMap<String, i64>,
    phoneme_id_map: HashMap<String, Vec<i64>>,
}

pub struct PiperModel {
    engine: ModuleEngine,
    sample_rate: u32,
    espeak_voice: String,
    phoneme_type: String,
    inference: InferenceConfig,
    num_speakers: u32,
    speaker_id_map: HashMap<String, i64>,
    phoneme_id_map: HashMap<String, Vec<i64>>,
}

pub struct PiperFrontendDebug {
    pub phoneme_type: String,
    pub phonemes: String,
    pub tokens: Vec<String>,
    pub token_ids: Vec<i64>,
}

impl PiperModel {
    pub fn new(model_path: &Path, config_path: &Path, _backend: &Backend) -> PiperResult<Self> {
        let file = File::open(config_path).map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "Failed to open config `{}`: {}",
                config_path.display(),
                e
            ))
        })?;
        let raw: RawConfig = serde_json::from_reader(file).map_err(|e| {
            PiperError::FailedToLoadResource(format!("Failed to parse config: {}", e))
        })?;
        let engine = load_module(model_path, piper_input_names(raw.num_speakers), &["output"])?;
        Ok(Self {
            engine,
            sample_rate: raw.audio.sample_rate,
            espeak_voice: raw.espeak.voice,
            phoneme_type: raw.phoneme_type,
            inference: raw.inference,
            num_speakers: raw.num_speakers,
            speaker_id_map: raw.speaker_id_map,
            phoneme_id_map: raw.phoneme_id_map,
        })
    }

    /// Construct a PiperModel from a mimic3 model directory.
    /// Reads the mimic3 JSON config + tokens.txt and maps them to Piper internals.
    pub fn from_mimic3(
        model_path: &Path,
        config_path: &Path,
        _backend: &Backend,
    ) -> PiperResult<Self> {
        #[derive(Deserialize)]
        struct Mimic3Audio {
            sample_rate: u32,
        }
        #[derive(Deserialize)]
        struct Mimic3Model {
            n_speakers: u32,
        }
        #[derive(Deserialize)]
        struct Mimic3Config {
            audio: Mimic3Audio,
            model: Mimic3Model,
            text_language: String,
        }

        let file = File::open(config_path).map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "Failed to open config `{}`: {}",
                config_path.display(),
                e
            ))
        })?;
        let raw: Mimic3Config = serde_json::from_reader(file).map_err(|e| {
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

        // Convert tokens.txt (String → i64) to phoneme_id_map (String → Vec<i64>).
        let mut phoneme_id_map = HashMap::new();
        for (token, id) in &token_to_id {
            phoneme_id_map.insert(token.clone(), vec![*id]);
        }

        let engine = load_module(
            model_path,
            piper_input_names(raw.model.n_speakers),
            &["output"],
        )?;
        Ok(Self {
            engine,
            sample_rate: raw.audio.sample_rate,
            espeak_voice: raw.text_language,
            phoneme_type: default_phoneme_type(),
            inference: InferenceConfig {
                noise_scale: 0.667,
                length_scale: 1.0,
                noise_w: 0.8,
            },
            num_speakers: raw.model.n_speakers,
            speaker_id_map: HashMap::new(),
            phoneme_id_map,
        })
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        speaker_id: Option<i64>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let phonemes = self.phonemize(text)?;
        self.synthesize_phonemes(&phonemes, speaker_id)
    }

    pub fn synthesize_with_options(
        &mut self,
        text: &str,
        speaker_id: Option<i64>,
        length_scale: Option<f32>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let phonemes = self.phonemize(text)?;
        self.synthesize_phonemes_with_options(&phonemes, speaker_id, length_scale)
    }

    pub fn synthesize_phonemes(
        &mut self,
        phonemes: &str,
        speaker_id: Option<i64>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        self.synthesize_phonemes_with_options(phonemes, speaker_id, None)
    }

    pub fn synthesize_phonemes_with_options(
        &mut self,
        phonemes: &str,
        speaker_id: Option<i64>,
        length_scale: Option<f32>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let phonemes: String = phonemes.trim().nfd().collect();
        let samples = infer(
            &self.engine,
            &self.phoneme_id_map,
            self.num_speakers,
            &phonemes,
            self.inference.noise_scale,
            length_scale.unwrap_or(self.inference.length_scale),
            self.inference.noise_w,
            speaker_id.unwrap_or(0),
        )?;
        Ok((samples, self.sample_rate))
    }

    pub fn voices(&self) -> Option<&HashMap<String, i64>> {
        if self.speaker_id_map.is_empty() {
            None
        } else {
            Some(&self.speaker_id_map)
        }
    }

    pub fn phonemize(&self, text: &str) -> PiperResult<String> {
        match self.phoneme_type.as_str() {
            "espeak" => espeak_phonemize(text, &self.espeak_voice),
            "text" => Ok(text.to_string()),
            other => Err(PiperError::PhonemizationError(format!(
                "Unsupported Piper phoneme_type `{other}`"
            ))),
        }
    }

    pub fn debug_frontend(&self, text: &str) -> PiperResult<PiperFrontendDebug> {
        let phonemes = self.phonemize(text)?;
        self.debug_phoneme_string(&phonemes)
    }

    pub fn debug_phoneme_string(&self, phonemes: &str) -> PiperResult<PiperFrontendDebug> {
        let tokens = phoneme_string_to_tokens(&self.phoneme_id_map, &phonemes);
        let token_ids = tokens
            .iter()
            .filter_map(|token| self.phoneme_id_map.get(token).and_then(|ids| ids.first()))
            .copied()
            .collect();

        Ok(PiperFrontendDebug {
            phoneme_type: self.phoneme_type.clone(),
            phonemes: phonemes.to_string(),
            tokens,
            token_ids,
        })
    }
}

fn default_phoneme_type() -> String {
    "espeak".to_string()
}

fn phonemes_to_ids(phoneme_id_map: &HashMap<String, Vec<i64>>, phonemes: &str) -> Vec<i64> {
    let pad_id = *phoneme_id_map
        .get(&PAD.to_string())
        .and_then(|v| v.first())
        .unwrap_or(&0);
    let bos_id = *phoneme_id_map
        .get(&BOS.to_string())
        .and_then(|v| v.first())
        .unwrap_or(&0);
    let eos_id = *phoneme_id_map
        .get(&EOS.to_string())
        .and_then(|v| v.first())
        .unwrap_or(&0);

    let mut ids = Vec::with_capacity((phonemes.len() + 1) * 2);
    ids.push(bos_id);
    ids.push(pad_id);

    let tokens = phoneme_string_to_tokens(phoneme_id_map, phonemes);
    for token in tokens {
        if let Some(id) = phoneme_id_map.get(&token).and_then(|v| v.first()) {
            ids.push(*id);
            ids.push(pad_id);
        }
    }
    ids.push(eos_id);
    ids
}

fn phoneme_string_to_tokens(
    phoneme_id_map: &HashMap<String, Vec<i64>>,
    phonemes: &str,
) -> Vec<String> {
    let whitespace_split: Vec<&str> = phonemes.split_whitespace().collect();
    if !whitespace_split.is_empty()
        && whitespace_split
            .iter()
            .all(|token| phoneme_id_map.contains_key(*token))
    {
        return whitespace_split
            .into_iter()
            .map(ToOwned::to_owned)
            .collect();
    }

    let chars: Vec<char> = phonemes.chars().collect();
    let max_token_chars = phoneme_id_map
        .keys()
        .map(|token| token.chars().count())
        .max()
        .unwrap_or(1);
    let mut tokens = Vec::new();
    let mut idx = 0usize;

    while idx < chars.len() {
        let remaining = chars.len() - idx;
        let max_len = remaining.min(max_token_chars);
        let mut matched = None;

        for len in (1..=max_len).rev() {
            let candidate: String = chars[idx..idx + len].iter().collect();
            if phoneme_id_map.contains_key(&candidate) {
                matched = Some(candidate);
                break;
            }
        }

        if let Some(token) = matched {
            idx += token.chars().count();
            tokens.push(token);
        } else {
            idx += 1;
        }
    }

    tokens
}

fn piper_input_names(num_speakers: u32) -> &'static [&'static str] {
    if num_speakers > 1 {
        &["input", "input_lengths", "scales", "sid"]
    } else {
        &["input", "input_lengths", "scales"]
    }
}

fn infer(
    engine: &ModuleEngine,
    phoneme_id_map: &HashMap<String, Vec<i64>>,
    num_speakers: u32,
    phonemes: &str,
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
    speaker_id: i64,
) -> PiperResult<Vec<f32>> {
    let ids = phonemes_to_ids(phoneme_id_map, phonemes);
    let input_len = ids.len();
    let ids_i32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    let lengths = [input_len as i32];
    let scales = [noise_scale, length_scale, noise_w];
    let sid = [speaker_id as i32];
    let input_shape = [1usize, input_len];

    let mut inputs = vec![
        NamedInput {
            name: "input",
            data: TensorData::I32(&ids_i32),
            shape: &input_shape,
        },
        NamedInput {
            name: "input_lengths",
            data: TensorData::I32(&lengths),
            shape: &[1],
        },
        NamedInput {
            name: "scales",
            data: TensorData::F32(&scales),
            shape: &[3],
        },
    ];
    if num_speakers > 1 {
        inputs.push(NamedInput {
            name: "sid",
            data: TensorData::I32(&sid),
            shape: &[1],
        });
    }

    let outputs = engine
        .run_named_dynamic(&inputs, &["output"])
        .map_err(|e| PiperError::InferenceError(format!("Inference failed: {}", e)))?;

    let mut samples = outputs
        .into_iter()
        .next()
        .ok_or_else(|| PiperError::InferenceError("MNN returned no waveform output".to_string()))?
        .data;
    crate::normalize_audio(&mut samples);
    Ok(samples)
}
