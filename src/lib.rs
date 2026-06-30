mod coqui_vits;
mod cotovia_vits;
mod gl_g2p;
mod kokoro;
mod kokoro_mnn;
mod mms;
mod number_spellout;
mod piper;
mod sherpa_vits;
mod supertonic3_mnn;
mod vits_tokenize;

#[cfg(feature = "japanese")]
mod japanese;

pub use coqui_vits::CoquiVitsModel;
pub use cotovia_vits::CotoviaVitsModel;
pub use gl_g2p::galician_g2p;
pub use kokoro_mnn::KokoroMnnModel;
pub use mms::MmsModel;
pub use number_spellout::{normalize_numbers, Lang, Speller};
pub use piper::PiperModel;
pub use sherpa_vits::SherpaVitsModel;
pub use supertonic3_mnn::Supertonic3MnnModel;

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

pub use espeak_rs::{init as init_espeak, BoundaryAfter, PhonemeChunk};

pub enum Backend {
    Cpu,
    Xnnpack,
}

fn load_module(
    model_path: &std::path::Path,
    input_names: &[&str],
    output_names: &[&str],
) -> PiperResult<mnn_sys::ModuleEngine> {
    use mnn_sys::{InferenceConfig, MemoryMode, ModuleEngine};
    ModuleEngine::from_file(
        model_path,
        input_names,
        output_names,
        Some(
            InferenceConfig::new()
                .with_threads(2)
                .with_memory(MemoryMode::Low),
        ),
    )
    .map_err(|e| {
        PiperError::FailedToLoadResource(format!(
            "Failed to load MNN module `{}`: {}",
            model_path.display(),
            e
        ))
    })
}

/// Normalize audio samples to fill the [-1, 1] range, matching the original
/// piper C++ behavior which divides by peak amplitude before int16 conversion.
fn normalize_audio(samples: &mut [f32]) {
    let peak = samples.iter().fold(0.01f32, |m, s| m.max(s.abs()));
    let scale = 1.0 / peak;
    for s in samples.iter_mut() {
        *s *= scale;
    }
}

fn espeak_phonemize(text: &str, voice: &str) -> PiperResult<String> {
    use unicode_normalization::UnicodeNormalization;
    espeak_rs::text_to_phonemes(text, normalize_espeak_voice(voice), None)
        .map(|phonemes| phonemes.join(" ").nfd().collect())
        .map_err(|e| PiperError::PhonemizationError(format!("{e}")))
}

fn normalize_espeak_voice(voice: &str) -> &str {
    match voice {
        // The bucket catalog maps app language `zh` to the Mandarin espeak dictionary `cmn`.
        "zh" => "cmn",
        _ => voice,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn normalizes_raw_phonemes() {
        use unicode_normalization::UnicodeNormalization;
        let normalized: String = "  e\u{301}  ".trim().nfd().collect();
        assert_eq!(normalized, "e\u{301}");
    }
}
