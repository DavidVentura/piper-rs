use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

use crate::{build_session, espeak_phonemize, vits_tokenize, Backend, PiperError, PiperResult};

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
    num_speakers: u32,
    #[serde(default)]
    speaker_id_map: HashMap<String, i64>,
    phoneme_id_map: HashMap<char, Vec<i64>>,
}

pub struct PiperModel {
    session: Session,
    sample_rate: u32,
    espeak_voice: String,
    inference: InferenceConfig,
    num_speakers: u32,
    speaker_id_map: HashMap<String, i64>,
    phoneme_id_map: HashMap<char, Vec<i64>>,
}

impl PiperModel {
    pub fn new(model_path: &Path, config_path: &Path, backend: &Backend) -> PiperResult<Self> {
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
        let session = build_session(model_path, backend)?;
        Ok(Self {
            session,
            sample_rate: raw.audio.sample_rate,
            espeak_voice: raw.espeak.voice,
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
        backend: &Backend,
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

        // Convert tokens.txt (String → i64) to phoneme_id_map (char → Vec<i64>).
        // mimic3 tokens are all single characters.
        let mut phoneme_id_map = HashMap::new();
        for (token, id) in &token_to_id {
            let mut chars = token.chars();
            if let (Some(ch), None) = (chars.next(), chars.next()) {
                phoneme_id_map.insert(ch, vec![*id]);
            }
        }

        let session = build_session(model_path, backend)?;
        Ok(Self {
            session,
            sample_rate: raw.audio.sample_rate,
            espeak_voice: raw.text_language,
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
        let phonemes = espeak_phonemize(text, &self.espeak_voice)?;
        self.synthesize_phonemes(&phonemes, speaker_id)
    }

    pub fn synthesize_with_options(
        &mut self,
        text: &str,
        speaker_id: Option<i64>,
        length_scale: Option<f32>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let phonemes = espeak_phonemize(text, &self.espeak_voice)?;
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
            &mut self.session,
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
        espeak_phonemize(text, &self.espeak_voice)
    }
}

fn phonemes_to_ids(phoneme_id_map: &HashMap<char, Vec<i64>>, phonemes: &str) -> Vec<i64> {
    let pad_id = *phoneme_id_map
        .get(&PAD)
        .and_then(|v| v.first())
        .unwrap_or(&0);
    let bos_id = *phoneme_id_map
        .get(&BOS)
        .and_then(|v| v.first())
        .unwrap_or(&0);
    let eos_id = *phoneme_id_map
        .get(&EOS)
        .and_then(|v| v.first())
        .unwrap_or(&0);

    let mut ids = Vec::with_capacity((phonemes.len() + 1) * 2);
    ids.push(bos_id);
    for ch in phonemes.chars() {
        if let Some(id) = phoneme_id_map.get(&ch).and_then(|v| v.first()) {
            ids.push(*id);
            ids.push(pad_id);
        }
    }
    ids.push(eos_id);
    ids
}

fn infer(
    session: &mut Session,
    phoneme_id_map: &HashMap<char, Vec<i64>>,
    num_speakers: u32,
    phonemes: &str,
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
    speaker_id: i64,
) -> PiperResult<Vec<f32>> {
    let ids = phonemes_to_ids(phoneme_id_map, phonemes);
    let input_len = ids.len();
    let input = Array2::<i64>::from_shape_vec((1, input_len), ids).unwrap();
    let input_lengths = Array1::<i64>::from_iter([input_len as i64]);
    let scales = Array1::<f32>::from_iter([noise_scale, length_scale, noise_w]);

    let input_t = Tensor::<i64>::from_array((
        [1, input_len],
        input.into_raw_vec_and_offset().0.into_boxed_slice(),
    ))
    .unwrap();
    let lengths_t = Tensor::<i64>::from_array((
        [1],
        input_lengths.into_raw_vec_and_offset().0.into_boxed_slice(),
    ))
    .unwrap();
    let scales_t =
        Tensor::<f32>::from_array(([3], scales.into_raw_vec_and_offset().0.into_boxed_slice()))
            .unwrap();

    let outputs = if num_speakers > 1 {
        let sid = Array1::<i64>::from_iter([speaker_id]);
        let sid_t =
            Tensor::<i64>::from_array(([1], sid.into_raw_vec_and_offset().0.into_boxed_slice()))
                .unwrap();
        session.run(ort::inputs![input_t, lengths_t, scales_t, sid_t])
    } else {
        session.run(ort::inputs![input_t, lengths_t, scales_t])
    }
    .map_err(|e| PiperError::InferenceError(format!("Inference failed: {}", e)))?;

    let (_, audio) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| PiperError::InferenceError(format!("Failed to extract output: {}", e)))?;

    Ok(audio.to_vec())
}
