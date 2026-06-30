use std::collections::HashMap;
use std::path::Path;

use mnn_sys::{ModuleEngine, NamedInput, TensorData};

use crate::number_spellout::{self, Lang};
use crate::vits_tokenize;
use crate::{load_module, Backend, PiperError, PiperResult};

const DEFAULT_NOISE_SCALE: f32 = 0.667;
const DEFAULT_NOISE_SCALE_W: f32 = 0.8;
const DEFAULT_LENGTH_SCALE: f32 = 1.0;
// Facebook MMS-TTS models are always 16 kHz with add_blank=1. The ONNX carried
// these as metadata; MNN drops metadata on conversion so they are fixed here.
const MMS_SAMPLE_RATE: u32 = 16_000;

pub struct MmsModel {
    engine: ModuleEngine,
    sample_rate: u32,
    token_to_id: HashMap<String, i64>,
    max_token_chars: usize,
    blank_id: Option<i64>,
    // MMS has no number frontend and no digits in its vocab; when the language
    // is one we have CLDR rules for, numbers are spelled out before tokenizing.
    language: Option<Lang>,
}

impl MmsModel {
    pub fn new(
        model_path: &Path,
        tokens_path: &Path,
        language_code: &str,
        _backend: &Backend,
    ) -> PiperResult<Self> {
        let token_to_id = vits_tokenize::load_tokens(tokens_path)?;
        let max_token_chars = vits_tokenize::max_token_chars(&token_to_id);
        let engine = load_module(
            model_path,
            &[
                "x",
                "x_length",
                "noise_scale",
                "length_scale",
                "noise_scale_w",
            ],
            &["y"],
        )?;

        Ok(Self {
            engine,
            sample_rate: MMS_SAMPLE_RATE,
            token_to_id,
            max_token_chars,
            blank_id: Some(0),
            language: Lang::from_code(language_code),
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
            &self.engine,
            &ids,
            speed.unwrap_or(1.0),
            DEFAULT_NOISE_SCALE,
            DEFAULT_NOISE_SCALE_W,
        )?;
        Ok((samples, self.sample_rate))
    }

    pub fn phonemize(&self, text: &str) -> PiperResult<String> {
        let spelled = match self.language {
            Some(lang) => number_spellout::normalize_numbers(lang, text),
            None => text.to_owned(),
        };
        Ok(vits_tokenize::normalize_text(
            &spelled,
            &self.token_to_id,
            self.max_token_chars,
        ))
    }

    pub fn token_ids(&self, text: &str) -> Vec<i64> {
        self.text_to_ids(text)
    }

    pub fn voices(&self) -> Option<&HashMap<String, i64>> {
        None
    }

    fn text_to_ids(&self, text: &str) -> Vec<i64> {
        let ids = vits_tokenize::tokenize_to_ids(text, &self.token_to_id, self.max_token_chars);
        match self.blank_id {
            Some(blank_id) => vits_tokenize::intersperse(&ids, blank_id),
            None => ids,
        }
    }
}

fn infer(
    engine: &ModuleEngine,
    ids: &[i64],
    speed: f32,
    noise_scale: f32,
    noise_scale_w: f32,
) -> PiperResult<Vec<f32>> {
    let ids_i32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    let input_len = [ids.len() as i32];
    let noise = [noise_scale];
    let length = [DEFAULT_LENGTH_SCALE / speed.max(0.1)];
    let noise_w = [noise_scale_w];
    let input_shape = [1usize, ids.len()];

    let inputs = [
        NamedInput {
            name: "x",
            data: TensorData::I32(&ids_i32),
            shape: &input_shape,
        },
        NamedInput {
            name: "x_length",
            data: TensorData::I32(&input_len),
            shape: &[1],
        },
        NamedInput {
            name: "noise_scale",
            data: TensorData::F32(&noise),
            shape: &[1],
        },
        NamedInput {
            name: "length_scale",
            data: TensorData::F32(&length),
            shape: &[1],
        },
        NamedInput {
            name: "noise_scale_w",
            data: TensorData::F32(&noise_w),
            shape: &[1],
        },
    ];

    let outputs = engine
        .run_named_dynamic(&inputs, &["y"])
        .map_err(|e| PiperError::InferenceError(format!("Inference failed: {}", e)))?;

    let mut samples = outputs
        .into_iter()
        .next()
        .ok_or_else(|| PiperError::InferenceError("MNN returned no waveform output".to_string()))?
        .data;
    crate::normalize_audio(&mut samples);
    Ok(samples)
}
