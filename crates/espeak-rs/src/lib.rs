use std::env;
use std::ffi::{c_char, c_void, CStr, CString};
use std::mem;
use std::path::PathBuf;
use std::ptr;
use std::sync::OnceLock;

const PIPER_ESPEAKNG_DATA_DIRECTORY: &str = "PIPER_ESPEAKNG_DATA_DIRECTORY";
const ESPEAKNG_DATA_DIR_NAME: &str = "espeak-ng-data";

#[derive(Debug, Clone)]
pub struct ESpeakError(pub String);

impl std::error::Error for ESpeakError {}

impl std::fmt::Display for ESpeakError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "eSpeak-ng error: {}", self.0)
    }
}

pub type ESpeakResult<T> = Result<T, ESpeakError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryAfter {
    None,
    Sentence,
    Paragraph,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhonemeChunk {
    pub phonemes: String,
    pub boundary_after: BoundaryAfter,
}

static ESPEAK_INIT: OnceLock<ESpeakResult<()>> = OnceLock::new();

fn init_espeak() -> ESpeakResult<()> {
    let data_dir = locate_espeak_data();
    // Keep the CString alive until after Initialize returns.
    let path_cstr = data_dir
        .as_ref()
        .and_then(|p| CString::new(p.to_string_lossy().as_ref()).ok());
    let path_ptr = path_cstr.as_ref().map_or(ptr::null(), |c| c.as_ptr());

    let sample_rate = unsafe {
        espeak_rs_sys::espeak_Initialize(
            espeak_rs_sys::espeak_AUDIO_OUTPUT_AUDIO_OUTPUT_RETRIEVAL,
            0,
            path_ptr,
            espeak_rs_sys::espeakINITIALIZE_DONT_EXIT as i32,
        )
    };

    if sample_rate <= 0 {
        Err(ESpeakError(format!(
            "Failed to initialize eSpeak-ng (code {sample_rate}). \
            Try setting `{PIPER_ESPEAKNG_DATA_DIRECTORY}` to the directory containing `{ESPEAKNG_DATA_DIR_NAME}`."
        )))
    } else {
        Ok(())
    }
}

fn locate_espeak_data() -> Option<PathBuf> {
    // 1. Environment variable
    if let Ok(dir) = env::var(PIPER_ESPEAKNG_DATA_DIRECTORY) {
        let p = PathBuf::from(dir);
        if p.join(ESPEAKNG_DATA_DIR_NAME).exists() {
            return Some(p);
        }
    }
    // 2. Current working directory
    if let Ok(cwd) = env::current_dir() {
        if cwd.join(ESPEAKNG_DATA_DIR_NAME).exists() {
            return Some(cwd);
        }
    }
    // 3. Directory of the current executable
    if let Ok(exe) = env::current_exe() {
        if let Some(dir) = exe.parent() {
            if dir.join(ESPEAKNG_DATA_DIR_NAME).exists() {
                return Some(dir.to_path_buf());
            }
        }
    }
    None
}

/// Strip inline language-switch markers of the form `(xx)` that espeak inserts
/// when the text contains words from a different language than the active voice.
fn strip_lang_switches(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth: usize = 0;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

fn ends_sentence(text: &str) -> bool {
    matches!(text.trim_end().chars().last(), Some('.' | '?' | '!'))
}

fn split_into_paragraphs(text: &str) -> Vec<String> {
    let mut paragraphs = Vec::new();
    let mut current = String::new();
    let mut prev_line_ended_sentence = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !current.is_empty() {
                paragraphs.push(mem::take(&mut current));
            }
            prev_line_ended_sentence = false;
            continue;
        }

        if !current.is_empty() && prev_line_ended_sentence {
            paragraphs.push(mem::take(&mut current));
        } else if !current.is_empty() {
            current.push(' ');
        }

        current.push_str(trimmed);
        prev_line_ended_sentence = ends_sentence(trimmed);
    }

    if !current.is_empty() {
        paragraphs.push(current);
    }

    paragraphs
}

fn split_paragraph_into_sentenceish_segments(paragraph: &str) -> Vec<(String, bool)> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = paragraph.chars().collect();
    let mut index = 0;

    while index < chars.len() {
        let ch = chars[index];
        current.push(ch);
        index += 1;

        if matches!(ch, '.' | '?' | '!') {
            while index < chars.len() && matches!(chars[index], '.' | '?' | '!') {
                current.push(chars[index]);
                index += 1;
            }

            let is_sentence_boundary = index == chars.len() || chars[index].is_whitespace();
            if is_sentence_boundary {
                let segment = current.trim();
                if !segment.is_empty() {
                    segments.push((segment.to_owned(), true));
                }
                current.clear();
                while index < chars.len() && chars[index].is_whitespace() {
                    index += 1;
                }
            }
        }
    }

    let tail = current.trim();
    if !tail.is_empty() {
        segments.push((tail.to_owned(), false));
    }

    if segments.is_empty() && !paragraph.trim().is_empty() {
        segments.push((paragraph.trim().to_owned(), false));
    }

    segments
}

fn phonemize_text_segment(text: &str, phoneme_mode: i32) -> ESpeakResult<String> {
    let text_cstr =
        CString::new(text).map_err(|_| ESpeakError("Text contains a null byte".into()))?;
    let mut phonemes = String::new();

    // espeak advances this pointer clause by clause, setting it to null when done.
    let mut text_ptr: *const c_char = text_cstr.as_ptr();

    while !text_ptr.is_null() {
        let clause = unsafe {
            let res = espeak_rs_sys::espeak_TextToPhonemes(
                &mut text_ptr as *mut *const c_char as *mut *const c_void,
                espeak_rs_sys::espeakCHARS_UTF8 as i32,
                phoneme_mode,
            );
            if res.is_null() {
                continue;
            }
            CStr::from_ptr(res).to_string_lossy().into_owned()
        };

        let clause = strip_lang_switches(&clause);
        if clause.is_empty() {
            continue;
        }

        phonemes.push_str(&clause);
    }

    Ok(phonemes)
}

fn phonemize_paragraph(paragraph: &str, phoneme_mode: i32) -> ESpeakResult<Vec<PhonemeChunk>> {
    let mut chunks: Vec<PhonemeChunk> = Vec::new();

    for (segment, ended_with_sentence_punctuation) in
        split_paragraph_into_sentenceish_segments(paragraph)
    {
        let phonemes = phonemize_text_segment(&segment, phoneme_mode)?;
        if phonemes.is_empty() {
            continue;
        }

        chunks.push(PhonemeChunk {
            phonemes,
            boundary_after: if ended_with_sentence_punctuation {
                BoundaryAfter::Sentence
            } else {
                BoundaryAfter::None
            },
        });
    }

    if let Some(last_chunk) = chunks.last_mut() {
        last_chunk.boundary_after = BoundaryAfter::Paragraph;
    }

    Ok(chunks)
}

/// Convert `text` to IPA phoneme chunks using the given espeak-ng voice/language.
///
/// `espeak_TextToPhonemes` returns one clause at a time (advancing an internal
/// pointer through the input). Clauses that end a sentence are terminated by
/// `.`, `?`, or `!` in the phoneme output; sub-clauses (comma, semicolon, …)
/// end with the corresponding punctuation but do not break a sentence. Blank
/// lines split paragraphs, and a sentence-ending `.? !` followed by a newline
/// also starts a new paragraph.
///
/// This function accumulates sub-clauses and emits one `PhonemeChunk` per
/// sentence or paragraph tail.
///
/// Inline language-switch markers (`(en)`, `(ar)`, …) are always stripped.
pub fn text_to_phoneme_chunks(
    text: &str,
    language: &str,
    phoneme_separator: Option<char>,
) -> ESpeakResult<Vec<PhonemeChunk>> {
    // Ensure the library is initialised exactly once.
    ESPEAK_INIT
        .get_or_init(init_espeak)
        .as_ref()
        .map_err(|e| e.clone())?;

    let lang_cstr = CString::new(language)
        .map_err(|_| ESpeakError("Language name contains a null byte".into()))?;
    let set_voice = unsafe { espeak_rs_sys::espeak_SetVoiceByName(lang_cstr.as_ptr()) };
    if set_voice != espeak_rs_sys::espeak_ERROR_EE_OK {
        return Err(ESpeakError(format!("Failed to set voice: `{language}`")));
    }

    let phoneme_mode = match phoneme_separator {
        Some(c) => ((c as u32) << 8) | espeak_rs_sys::espeakINITIALIZE_PHONEME_IPA,
        None => espeak_rs_sys::espeakINITIALIZE_PHONEME_IPA,
    } as i32;

    let mut chunks = Vec::new();
    for paragraph in split_into_paragraphs(text) {
        chunks.extend(phonemize_paragraph(&paragraph, phoneme_mode)?);
    }

    Ok(chunks)
}

/// Compatibility wrapper that drops boundary metadata.
pub fn text_to_phonemes(
    text: &str,
    language: &str,
    phoneme_separator: Option<char>,
) -> ESpeakResult<Vec<String>> {
    text_to_phoneme_chunks(text, language, phoneme_separator).map(|chunks| {
        chunks
            .into_iter()
            .map(|chunk| chunk.phonemes)
            .collect::<Vec<_>>()
    })
}

// ==============================

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT_ALICE: &str =
        "Who are you? said the Caterpillar. Replied Alice , rather shyly, I hardly know, sir!";

    #[test]
    fn test_basic_en() -> ESpeakResult<()> {
        let phonemes = text_to_phonemes("test", "en-US", None)?.join("");
        assert_eq!(phonemes, "tˈɛst.");
        Ok(())
    }

    #[test]
    fn test_it_splits_sentences() -> ESpeakResult<()> {
        let phonemes = text_to_phonemes(TEXT_ALICE, "en-US", None)?;
        assert_eq!(phonemes.len(), 3);
        Ok(())
    }

    #[test]
    fn test_it_adds_phoneme_separator() -> ESpeakResult<()> {
        let phonemes = text_to_phonemes("test", "en-US", Some('_'))?.join("");
        assert_eq!(phonemes, "t_ˈɛ_s_t.");
        Ok(())
    }

    #[test]
    fn test_it_preserves_clause_breakers() -> ESpeakResult<()> {
        let phonemes = text_to_phonemes(TEXT_ALICE, "en-US", None)?.join("");
        for c in ['.', ',', '?', '!'] {
            assert!(phonemes.contains(c), "Clause breaker `{c}` not preserved");
        }
        Ok(())
    }

    #[test]
    fn test_arabic() -> ESpeakResult<()> {
        let phonemes = text_to_phonemes("مَرْحَبَاً بِكَ أَيُّهَا الْرَّجُلْ", "ar", None)?.join("");
        assert_eq!(phonemes, "mˈarħabˌaː bikˌa ʔaˈiːuhˌaː alrrˈadʒul.");
        Ok(())
    }

    #[test]
    fn test_lang_switch_markers_stripped() -> ESpeakResult<()> {
        // Mixed-language text: espeak inserts (en)/(ar) markers; we always strip them.
        let phonemes = text_to_phonemes("Hello معناها مرحباً", "ar", None)?.join("");
        assert!(!phonemes.contains("(en)"));
        assert!(!phonemes.contains("(ar)"));
        Ok(())
    }

    #[test]
    fn test_line_splitting() -> ESpeakResult<()> {
        let phonemes = text_to_phonemes("Hello\nThere\nAnd\nWelcome", "en-US", None)?;
        assert_eq!(phonemes.len(), 1);
        Ok(())
    }

    #[test]
    fn test_split_into_paragraphs_blank_lines() {
        assert_eq!(
            split_into_paragraphs("One line\n\nTwo line"),
            vec!["One line".to_owned(), "Two line".to_owned()]
        );
    }

    #[test]
    fn test_split_into_paragraphs_sentence_newline_marks_paragraph() {
        assert_eq!(
            split_into_paragraphs("One.\nTwo"),
            vec!["One.".to_owned(), "Two".to_owned()]
        );
    }

    #[test]
    fn test_split_into_paragraphs_single_newline_is_formatting() {
        assert_eq!(
            split_into_paragraphs("One\nstill one"),
            vec!["One still one".to_owned()]
        );
    }

    #[test]
    fn test_split_paragraph_into_sentenceish_segments() {
        assert_eq!(
            split_paragraph_into_sentenceish_segments(
                "this is a word. this is another word. this is yet a third word"
            ),
            vec![
                ("this is a word.".to_owned(), true),
                ("this is another word.".to_owned(), true),
                ("this is yet a third word".to_owned(), false),
            ]
        );
    }

    #[test]
    fn test_text_to_phoneme_chunks_upgrades_sentence_to_paragraph() -> ESpeakResult<()> {
        let chunks = text_to_phoneme_chunks("Hello.\nThere", "en-US", None)?;
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].boundary_after, BoundaryAfter::Paragraph);
        assert_eq!(chunks[1].boundary_after, BoundaryAfter::Paragraph);
        Ok(())
    }

    #[test]
    fn test_text_to_phoneme_chunks_marks_sentence_boundaries_in_paragraph() -> ESpeakResult<()> {
        let chunks = text_to_phoneme_chunks(
            "this is a word. this is another word. this is yet a third word",
            "en-US",
            None,
        )?;
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].boundary_after, BoundaryAfter::Sentence);
        assert_eq!(chunks[1].boundary_after, BoundaryAfter::Sentence);
        assert_eq!(chunks[2].boundary_after, BoundaryAfter::Paragraph);
        Ok(())
    }
}
