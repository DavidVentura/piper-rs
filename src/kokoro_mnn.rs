use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mnn_sys::{InferenceConfig, MemoryMode, ModuleEngine, NamedInput, TensorData};
use ndarray::Array2;

use crate::{espeak_phonemize, PiperError, PiperResult};

pub struct KokoroMnnModel {
    engine: ModuleEngine,
    model_path: PathBuf,
    config: InferenceConfig,
    engine_needs_reset: bool,
    espeak_voice: String,
    vocab: HashMap<char, i64>,
    voices: HashMap<i64, Array2<f32>>,
    speaker_id_map: HashMap<String, i64>,
    #[cfg(feature = "japanese")]
    japanese_dict: Option<mucab::Dictionary<'static>>,
}

impl KokoroMnnModel {
    pub fn new(model_path: &Path, voices_path: &Path, espeak_voice: &str) -> PiperResult<Self> {
        Self::new_with_config(
            model_path,
            voices_path,
            espeak_voice,
            InferenceConfig::new()
                .with_threads(4)
                .with_memory(MemoryMode::Low),
        )
    }

    pub fn new_with_config(
        model_path: &Path,
        voices_path: &Path,
        espeak_voice: &str,
        config: InferenceConfig,
    ) -> PiperResult<Self> {
        let (voices, speaker_id_map) = crate::kokoro::load_voices(voices_path)?;
        let engine = load_engine(model_path, &config)?;

        Ok(Self {
            engine,
            model_path: model_path.to_path_buf(),
            config,
            engine_needs_reset: false,
            espeak_voice: espeak_voice.to_string(),
            vocab: crate::kokoro::kokoro_vocab(),
            voices,
            speaker_id_map,
            #[cfg(feature = "japanese")]
            japanese_dict: None,
        })
    }

    #[cfg(feature = "japanese")]
    pub fn load_japanese_dict(&mut self, path: &str) -> PiperResult<()> {
        let dict = mucab::Dictionary::load(path).map_err(|e| {
            PiperError::FailedToLoadResource(format!("Failed to load Japanese dictionary: {}", e))
        })?;
        self.japanese_dict = Some(dict);
        Ok(())
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
        speaker_id: Option<i64>,
        speed: Option<f32>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        if self.engine_needs_reset {
            let reload_started_at = Instant::now();
            self.engine = load_engine(&self.model_path, &self.config)?;
            log::info!(
                "kokoro_mnn.module_reload: took_ms={}",
                reload_started_at.elapsed().as_millis()
            );
            self.engine_needs_reset = false;
        }
        let samples = infer_mnn(
            &self.engine,
            &self.vocab,
            &self.voices,
            phonemes,
            speed.unwrap_or(1.0),
            speaker_id.unwrap_or(0),
        );
        self.engine_needs_reset = true;
        let samples = samples?;
        Ok((samples, 24000))
    }

    pub fn phonemize(&mut self, text: &str) -> PiperResult<String> {
        #[cfg(feature = "japanese")]
        if self.espeak_voice == "ja" {
            if let Some(ref mut dict) = self.japanese_dict {
                return crate::japanese::phonemize(text, dict);
            }

            return Err(PiperError::PhonemizationError(
                "Japanese Kokoro models require `load_japanese_dict(...)` before synthesis"
                    .to_string(),
            ));
        }

        #[cfg(not(feature = "japanese"))]
        if self.espeak_voice == "ja" {
            return Err(PiperError::PhonemizationError(
                "Japanese Kokoro models require the `japanese` feature".to_string(),
            ));
        }

        espeak_phonemize(text, &self.espeak_voice)
    }

    pub fn voices(&self) -> Option<&HashMap<String, i64>> {
        if self.speaker_id_map.is_empty() {
            None
        } else {
            Some(&self.speaker_id_map)
        }
    }
}

fn load_engine(model_path: &Path, config: &InferenceConfig) -> PiperResult<ModuleEngine> {
    ModuleEngine::from_file(
        model_path,
        &["input_ids", "style", "speed"],
        &["audio"],
        Some(config.clone()),
    )
    .map_err(|e| {
        PiperError::FailedToLoadResource(format!(
            "Failed to load MNN module `{}`: {}",
            model_path.display(),
            e
        ))
    })
}

fn infer_mnn(
    engine: &ModuleEngine,
    vocab: &HashMap<char, i64>,
    voices: &HashMap<i64, Array2<f32>>,
    phonemes: &str,
    speed: f32,
    speaker_id: i64,
) -> PiperResult<Vec<f32>> {
    let ids_i64 = crate::kokoro::phonemes_to_ids(vocab, phonemes);
    let token_count = ids_i64.len() - 2;
    let ids: Vec<i32> = ids_i64
        .into_iter()
        .map(|id| {
            i32::try_from(id).map_err(|_| {
                PiperError::InferenceError(format!("Kokoro token ID does not fit i32: {}", id))
            })
        })
        .collect::<PiperResult<_>>()?;

    let voice_styles = voices
        .get(&speaker_id)
        .ok_or_else(|| PiperError::InferenceError(format!("Unknown speaker ID: {}", speaker_id)))?;
    let style_idx = token_count.min(voice_styles.shape()[0] - 1);
    let style_row = voice_styles.row(style_idx).to_owned();
    let style = style_row.into_raw_vec_and_offset().0;
    let speed_buf = [speed];

    let outputs = engine
        .run_named_dynamic(
            &[
                NamedInput {
                    name: "input_ids",
                    data: TensorData::I32(&ids),
                    shape: &[1, ids.len()],
                },
                NamedInput {
                    name: "style",
                    data: TensorData::F32(&style),
                    shape: &[1, 256],
                },
                NamedInput {
                    name: "speed",
                    data: TensorData::F32(&speed_buf),
                    shape: &[1],
                },
            ],
            &["audio"],
        )
        .map_err(|e| PiperError::InferenceError(format!("MNN inference failed: {}", e)))?;

    let mut samples = outputs
        .into_iter()
        .next()
        .ok_or_else(|| PiperError::InferenceError("MNN returned no waveform output".to_string()))?
        .data;
    crate::normalize_audio(&mut samples);
    Ok(samples)
}
