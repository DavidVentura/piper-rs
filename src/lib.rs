mod model;

#[cfg(feature = "japanese")]
mod japanese;

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use espeak_rs::{text_to_phoneme_chunks, text_to_phonemes};
use ndarray::{Array3, Axis};
use ndarray_npy::NpzReader;
use ort::session::Session;
use unicode_normalization::UnicodeNormalization;

pub use espeak_rs::{BoundaryAfter, PhonemeChunk};
use model::{infer_kokoro, infer_piper, PiperRawConfig};
pub use model::{InferenceConfig, ModelConfig, ModelKind};

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
    #[cfg(feature = "japanese")]
    japanese_dict: Option<mucab::Dictionary<'static>>,
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
        let raw: PiperRawConfig = serde_json::from_reader(file).map_err(|e| {
            PiperError::FailedToLoadResource(format!("Failed to parse config: {}", e))
        })?;
        let config = ModelConfig::from_piper(raw);
        let session = Self::build_session(model_path)?;
        Ok(Self {
            config,
            session,
            #[cfg(feature = "japanese")]
            japanese_dict: None,
        })
    }

    pub fn new_kokoro(
        model_path: &Path,
        voices_path: &Path,
        espeak_voice: &str,
    ) -> PiperResult<Self> {
        let (voices, speaker_id_map) = load_kokoro_voices(voices_path)?;
        let config = ModelConfig::from_kokoro(voices, speaker_id_map, espeak_voice.to_string());
        let session = Self::build_session(model_path)?;
        Ok(Self {
            config,
            session,
            #[cfg(feature = "japanese")]
            japanese_dict: None,
        })
    }

    /// Load a mucab dictionary for Japanese phonemization.
    ///
    /// When loaded and espeak_voice is "ja", `create()` will use mucab
    /// for morphological analysis instead of espeak.
    #[cfg(feature = "japanese")]
    pub fn load_japanese_dict(&mut self, path: &str) -> PiperResult<()> {
        let dict = mucab::Dictionary::load(path).map_err(|e| {
            PiperError::FailedToLoadResource(format!("Failed to load Japanese dictionary: {}", e))
        })?;
        self.japanese_dict = Some(dict);
        Ok(())
    }

    fn build_session(model_path: &Path) -> PiperResult<Session> {
        Session::builder()
            .map_err(|e| {
                PiperError::FailedToLoadResource(format!(
                    "Failed to create session builder: {}",
                    e
                ))
            })?
            .with_intra_threads(2)
            .map_err(|e| {
                PiperError::FailedToLoadResource(format!(
                    "Failed to set ORT intra-op threads: {}",
                    e
                ))
            })?
            .with_inter_threads(2)
            .map_err(|e| {
                PiperError::FailedToLoadResource(format!(
                    "Failed to set ORT inter-op threads: {}",
                    e
                ))
            })?
            .with_intra_op_spinning(false)
            .map_err(|e| {
                PiperError::FailedToLoadResource(format!(
                    "Failed to disable ORT intra-op spinning: {}",
                    e
                ))
            })?
            .with_inter_op_spinning(false)
            .map_err(|e| {
                PiperError::FailedToLoadResource(format!(
                    "Failed to disable ORT inter-op spinning: {}",
                    e
                ))
            })?
            .commit_from_file(model_path)
            .map_err(|e| {
                PiperError::FailedToLoadResource(format!(
                    "Failed to load model `{}`: {}",
                    model_path.display(),
                    e
                ))
            })
    }

    pub fn from_session(session: Session, config: ModelConfig) -> Self {
        Self {
            session,
            config,
            #[cfg(feature = "japanese")]
            japanese_dict: None,
        }
    }

    pub fn phonemize_text_debug(&mut self, text: &str) -> PiperResult<String> {
        self.phonemize_text(text)
    }

    fn phonemize_text(&mut self, text: &str) -> PiperResult<String> {
        #[cfg(feature = "japanese")]
        if self.config.espeak_voice == "ja" {
            if let Some(ref mut dict) = self.japanese_dict {
                return japanese::phonemize(text, dict);
            }
        }

        text_to_phonemes(text, &self.config.espeak_voice, None)
            .map(|phonemes| phonemes.join(" ").nfd().collect())
            .map_err(|e| PiperError::PhonemizationError(format!("{e}")))
    }

    /// Translate text into phoneme chunks with explicit boundary metadata.
    pub fn phonemize_chunks(&self, text: &str) -> PiperResult<Vec<PhonemeChunk>> {
        text_to_phoneme_chunks(text, &self.config.espeak_voice, None)
            .map(|chunks| {
                chunks
                    .into_iter()
                    .map(|mut chunk| {
                        chunk.phonemes = chunk.phonemes.nfd().collect();
                        chunk
                    })
                    .collect()
            })
            .map_err(|e| PiperError::PhonemizationError(format!("{e}")))
    }

    /// Translate text into phoneme chunks grouped by sentence/clause.
    pub fn phonemize_sentences(&self, text: &str) -> PiperResult<Vec<String>> {
        self.phonemize_chunks(text)
            .map(|chunks| chunks.into_iter().map(|chunk| chunk.phonemes).collect())
    }

    /// Synthesize speech from text or phonemes.
    ///
    /// Returns `(samples, sample_rate)` where samples are f32 PCM audio.
    ///
    /// For Kokoro models, `length_scale` controls speech speed (default 1.0),
    /// and `noise_scale`/`noise_w` are ignored.
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

        let samples = match &self.config.kind {
            ModelKind::Piper {
                inference,
                num_speakers,
                phoneme_id_map,
            } => infer_piper(
                &mut self.session,
                phoneme_id_map,
                *num_speakers,
                &phonemes,
                noise_scale.unwrap_or(inference.noise_scale),
                length_scale.unwrap_or(inference.length_scale),
                noise_w.unwrap_or(inference.noise_w),
                speaker_id.unwrap_or(0),
            ),
            ModelKind::Kokoro { vocab, voices } => infer_kokoro(
                &mut self.session,
                vocab,
                voices,
                &phonemes,
                length_scale.unwrap_or(1.0),
                speaker_id.unwrap_or(0),
            ),
        }?;

        Ok((samples, self.config.sample_rate))
    }

    /// Returns the speaker name->id map, or `None` for single-speaker models.
    pub fn voices(&self) -> Option<&HashMap<String, i64>> {
        if self.config.speaker_id_map.is_empty() {
            None
        } else {
            Some(&self.config.speaker_id_map)
        }
    }
}

fn load_kokoro_voices(
    voices_path: &Path,
) -> PiperResult<(HashMap<i64, ndarray::Array2<f32>>, HashMap<String, i64>)> {
    let file = File::open(voices_path).map_err(|e| {
        PiperError::FailedToLoadResource(format!(
            "Failed to open voices `{}`: {}",
            voices_path.display(),
            e
        ))
    })?;
    let mut npz = NpzReader::new(BufReader::new(file)).map_err(|e| {
        PiperError::FailedToLoadResource(format!("Failed to read voices npz: {}", e))
    })?;

    let mut names: Vec<String> = npz.names().map_err(|e| {
        PiperError::FailedToLoadResource(format!("Failed to list voices: {}", e))
    })?;
    names.sort();

    let mut styles = HashMap::new();
    let mut speaker_id_map = HashMap::new();

    for (idx, npy_name) in names.iter().enumerate() {
        let voice_name = npy_name.trim_end_matches(".npy").to_string();
        let array: Array3<f32> = npz.by_name(npy_name).map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "Failed to read voice '{}': {}",
                voice_name, e
            ))
        })?;
        // [511, 1, 256] -> [511, 256]
        let squeezed = array.index_axis_move(Axis(1), 0);

        let id = idx as i64;
        speaker_id_map.insert(voice_name, id);
        styles.insert(id, squeezed);
    }

    Ok((styles, speaker_id_map))
}

#[cfg(test)]
mod tests {
    use super::normalize_phonemes;

    #[test]
    fn normalizes_raw_phonemes() {
        assert_eq!(normalize_phonemes("  e\u{301}  "), "e\u{301}");
    }
}
