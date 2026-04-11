mod kokoro;
mod mms;
mod piper;

#[cfg(feature = "japanese")]
mod japanese;

pub use kokoro::KokoroModel;
pub use mms::MmsModel;
pub use piper::PiperModel;

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

pub use espeak_rs::{BoundaryAfter, PhonemeChunk};

pub enum Backend {
    Cpu,
    Xnnpack,
}

fn build_session(
    model_path: &std::path::Path,
    backend: &Backend,
) -> PiperResult<ort::session::Session> {
    let builder = ort::session::Session::builder().map_err(|e| {
        PiperError::FailedToLoadResource(format!("Failed to create session builder: {}", e))
    })?;

    let builder = match backend {
        Backend::Cpu => builder,
        Backend::Xnnpack => {
            match builder.with_execution_providers([ort::ep::XNNPACK::default().build()]) {
                Ok(b) => {
                    eprintln!("ORT: XNNPACK execution provider registered");
                    b
                }
                Err(e) => {
                    eprintln!("ORT: XNNPACK not available ({}), falling back to CPU", e);
                    ort::session::Session::builder().map_err(|e| {
                        PiperError::FailedToLoadResource(format!(
                            "Failed to create session builder: {}",
                            e
                        ))
                    })?
                }
            }
        }
    };

    builder
        .with_intra_threads(2)
        .map_err(|e| {
            PiperError::FailedToLoadResource(format!("Failed to set ORT intra-op threads: {}", e))
        })?
        .with_inter_threads(2)
        .map_err(|e| {
            PiperError::FailedToLoadResource(format!("Failed to set ORT inter-op threads: {}", e))
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

fn espeak_phonemize(text: &str, voice: &str) -> PiperResult<String> {
    use unicode_normalization::UnicodeNormalization;
    espeak_rs::text_to_phonemes(text, voice, None)
        .map(|phonemes| phonemes.join(" ").nfd().collect())
        .map_err(|e| PiperError::PhonemizationError(format!("{e}")))
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
