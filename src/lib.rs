mod model;

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use espeak_rs::text_to_phonemes;
use ort::session::Session;
use serde_json;
use unicode_normalization::UnicodeNormalization;

use model::infer;
pub use model::ModelConfig;

fn normalize_phonemes(phonemes: &str) -> String {
    phonemes.trim().nfd().collect()
}

#[derive(Debug)]
pub enum PiperError {
    FailedToLoadResource(String),
    PhonemizationError(String),
    InferenceError(String),
}

impl std::fmt::Display for PiperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FailedToLoadResource(msg) => write!(f, "Failed to load resource: {}", msg),
            Self::PhonemizationError(msg) => write!(f, "Phonemization error: {}", msg),
            Self::InferenceError(msg) => write!(f, "Inference error: {}", msg),
        }
    }
}

impl std::error::Error for PiperError {}

pub type PiperResult<T> = Result<T, PiperError>;

pub struct Piper {
    config: ModelConfig,
    session: Session,
}

impl Piper {
    pub fn new(model_path: &Path, config_path: &Path) -> PiperResult<Self> {
        let file = File::open(config_path).map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "Failed to open config `{}`: {}",
                config_path.display(),
                e
            ))
        })?;
        let config: ModelConfig = serde_json::from_reader(file).map_err(|e| {
            PiperError::FailedToLoadResource(format!("Failed to parse config: {}", e))
        })?;
        let session = Session::builder()
            .map_err(|e| {
                PiperError::FailedToLoadResource(format!("Failed to create session builder: {}", e))
            })?
            .commit_from_file(model_path)
            .map_err(|e| {
                PiperError::FailedToLoadResource(format!(
                    "Failed to load model `{}`: {}",
                    model_path.display(),
                    e
                ))
            })?;
        Ok(Self { config, session })
    }

    pub fn from_session(session: Session, config: ModelConfig) -> Self {
        Self { session, config }
    }

    fn phonemize_text(&self, text: &str) -> PiperResult<String> {
        text_to_phonemes(text, &self.config.espeak.voice, None)
            .map(|phonemes| phonemes.join(" ").nfd().collect())
            .map_err(|e| PiperError::PhonemizationError(format!("{e}")))
    }

    /// Translate text into phoneme chunks grouped by sentence/clause.
    pub fn phonemize_sentences(&self, text: &str) -> PiperResult<Vec<String>> {
        text_to_phonemes(text, &self.config.espeak.voice, None)
            .map(|phonemes| {
                phonemes
                    .into_iter()
                    .map(|phoneme| phoneme.nfd().collect())
                    .collect()
            })
            .map_err(|e| PiperError::PhonemizationError(format!("{e}")))
    }

    /// Synthesize speech from text or phonemes.
    ///
    /// Returns `(samples, sample_rate)` where samples are f32 PCM audio.
    pub fn create(
        &mut self,
        text: &str,
        is_phonemes: bool,
        speaker_id: Option<i64>,
        length_scale: Option<f32>,
        noise_scale: Option<f32>,
        noise_w: Option<f32>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let phonemes = if is_phonemes {
            normalize_phonemes(text)
        } else {
            self.phonemize_text(text)?
        };

        let inf = &self.config.inference;
        let samples = infer(
            &mut self.session,
            &self.config,
            &phonemes,
            noise_scale.unwrap_or(inf.noise_scale),
            length_scale.unwrap_or(inf.length_scale),
            noise_w.unwrap_or(inf.noise_w),
            speaker_id.unwrap_or(0),
        )?;

        Ok((samples, self.config.audio.sample_rate))
    }

    /// Returns the speaker name→id map, or `None` for single-speaker models.
    pub fn voices(&self) -> Option<&HashMap<String, i64>> {
        if self.config.speaker_id_map.is_empty() {
            None
        } else {
            Some(&self.config.speaker_id_map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_phonemes;

    #[test]
    fn normalizes_raw_phonemes() {
        assert_eq!(normalize_phonemes("  e\u{301}  "), "e\u{301}");
    }
}
