use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::Tensor;
use unicode_normalization::UnicodeNormalization;

use crate::gl_g2p::galician_g2p;
use crate::vits_tokenize;
use crate::{build_session, Backend, PiperError, PiperResult};

// Captured from the AhoTTS/cotovia frontend that produced the sabela voice: the
// model consumes `0 p1 0 p2 ... 0 3 0` — blank id 0 interspersed, words joined
// by id 10, sentence terminated by EOS id 3. The lexicon stores the bare per-word
// id runs; we re-join, terminate and re-intersperse here.
const BLANK_ID: i64 = 0;
const EOS_ID: i64 = 3; // `.`, `;`, `:`
const EOS_EXCLAM: i64 = 1; // `!`
const EOS_QUESTION: i64 = 6; // `?`
const WORD_SEP_ID: i64 = 10;
const COMMA_PAUSE_ID: i64 = 9; // `,` / `-`, written as `10 9 10` between words

const SAMPLE_RATE: u32 = 22050;
const NOISE_SCALE: f32 = 0.667;
const NOISE_SCALE_W: f32 = 0.8;
const LENGTH_SCALE: f32 = 1.0;

/// VITS voice whose grapheme-to-phoneme is precomputed offline into a
/// `word -> token ids` lexicon (cotovia g2p, baked). Out-of-vocabulary words are
/// dropped until the normalization / rule fallback lands.
pub struct CotoviaVitsModel {
    session: Session,
    lexicon: HashMap<String, Vec<i64>>,
    // abbreviation -> expansion (e.g. "sr." -> "señor"); cotovia's abr.txt,
    // shipped in the lexicon file as `@key\texpansion` lines.
    abbreviations: HashMap<String, String>,
}

impl CotoviaVitsModel {
    pub fn new(model_path: &Path, lexicon_path: &Path, backend: &Backend) -> PiperResult<Self> {
        let (lexicon, abbreviations) = load_lexicon(lexicon_path)?;
        let session = build_session(model_path, backend)?;
        Ok(Self {
            session,
            lexicon,
            abbreviations,
        })
    }

    /// Replace whole-token abbreviations (`Sr.` -> `señor`) before number
    /// normalization. The trailing `.` is part of the key, so it doesn't trip EOS.
    fn expand_abbreviations(&self, text: &str) -> String {
        if self.abbreviations.is_empty() {
            return text.to_string();
        }
        text.split_whitespace()
            .map(|tok| {
                self.abbreviations
                    .get(&tok.to_lowercase())
                    .map(String::as_str)
                    .unwrap_or(tok)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        _speaker_id: Option<i64>,
        speed: Option<f32>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let ids = self.text_to_ids(text);
        if ids.is_empty() {
            return Ok((Vec::new(), SAMPLE_RATE));
        }
        let samples = infer(&mut self.session, &ids, speed.unwrap_or(1.0))?;
        Ok((samples, SAMPLE_RATE))
    }

    pub fn phonemize(&self, text: &str) -> PiperResult<String> {
        let ids = self.lexicon_ids(text);
        Ok(ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(" "))
    }

    pub fn voices(&self) -> Option<&HashMap<String, i64>> {
        None
    }

    /// Words joined by the separator token, with `,`/`-` pauses. No EOS, no blanks.
    fn lexicon_ids(&self, text: &str) -> Vec<i64> {
        let expanded = self.expand_abbreviations(text);
        let normalized = crate::gl_g2p::normalize_numbers(&expanded);
        let mut seq = Vec::new();
        let mut need_sep = false;
        for tok in tokenize(&normalized) {
            match tok {
                Token::Word(word) => {
                    // Known words use cotovia's exact pronunciation; OOV words
                    // fall back to the Galician grapheme-to-phoneme rules.
                    let ids = match self.lexicon.get(&word) {
                        Some(ids) => ids.clone(),
                        None => galician_g2p(&word),
                    };
                    if ids.is_empty() {
                        continue;
                    }
                    if need_sep {
                        seq.push(WORD_SEP_ID);
                    }
                    seq.extend_from_slice(&ids);
                    need_sep = true;
                }
                // `10 9 10` between words: the comma here, the trailing 10 from
                // the next word's separator.
                Token::Pause => {
                    if need_sep {
                        seq.push(WORD_SEP_ID);
                        seq.push(COMMA_PAUSE_ID);
                    }
                }
            }
        }
        seq
    }

    fn text_to_ids(&self, text: &str) -> Vec<i64> {
        let mut seq = self.lexicon_ids(text);
        if seq.is_empty() {
            return seq;
        }
        seq.push(eos_token(text));
        vits_tokenize::intersperse(&seq, BLANK_ID)
    }
}

/// Final-punctuation EOS: `?`→6, `!`→1, otherwise (`.`/`;`/`:`/none)→3.
fn eos_token(text: &str) -> i64 {
    for ch in text.chars().rev() {
        if ch.is_whitespace() {
            continue;
        }
        return match ch {
            '?' => EOS_QUESTION,
            '!' => EOS_EXCLAM,
            _ => EOS_ID,
        };
    }
    EOS_ID
}

enum Token {
    Word(String),
    Pause,
}

/// Split into lowercased words (keeping the Galician geminate middle dot), with a
/// `Pause` marker for `,`/`-`/`;`/`:`. Text is NFC-normalized first so decomposed
/// accents compose into single letters instead of splitting words at the mark.
fn tokenize(text: &str) -> Vec<Token> {
    let mut toks = Vec::new();
    let mut current = String::new();
    let mut pause = false;
    for ch in text.nfc() {
        if ch.is_alphabetic() || ch == '·' {
            current.extend(ch.to_lowercase());
        } else {
            if !current.is_empty() {
                if pause {
                    toks.push(Token::Pause);
                    pause = false;
                }
                toks.push(Token::Word(std::mem::take(&mut current)));
            }
            if matches!(ch, ',' | ';' | ':' | '-' | '\u{2013}' | '\u{2014}') {
                pause = true;
            }
        }
    }
    if !current.is_empty() {
        if pause {
            toks.push(Token::Pause);
        }
        toks.push(Token::Word(current));
    }
    toks
}

type Lexicons = (HashMap<String, Vec<i64>>, HashMap<String, String>);

fn load_lexicon(path: &Path) -> PiperResult<Lexicons> {
    let file = File::open(path).map_err(|e| {
        PiperError::FailedToLoadResource(format!(
            "Failed to open lexicon `{}`: {}",
            path.display(),
            e
        ))
    })?;
    let inner: Box<dyn Read> = if path.extension().is_some_and(|ext| ext == "zst") {
        Box::new(zeekstd::Decoder::new(file).map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "Failed to open zstd lexicon `{}`: {:?}",
                path.display(),
                e
            ))
        })?)
    } else {
        Box::new(file)
    };
    let reader = BufReader::new(inner);
    let mut lexicon = HashMap::new();
    let mut abbreviations = HashMap::new();

    for line in reader.lines() {
        let line = line.map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "Failed to read lexicon `{}`: {}",
                path.display(),
                e
            ))
        })?;
        let Some((key, value)) = line.split_once('\t') else {
            continue;
        };
        // `@abbr\texpansion` lines are cotovia's abbreviation table.
        if let Some(abbr) = key.strip_prefix('@') {
            abbreviations.insert(
                abbr.nfc().collect::<String>().to_lowercase(),
                value.to_string(),
            );
            continue;
        }
        let ids: Vec<i64> = value
            .split_whitespace()
            .filter_map(|tok| tok.parse().ok())
            .collect();
        if !ids.is_empty() {
            lexicon.insert(key.nfc().collect::<String>().to_lowercase(), ids);
        }
    }

    Ok((lexicon, abbreviations))
}

fn infer(session: &mut Session, ids: &[i64], speed: f32) -> PiperResult<Vec<f32>> {
    let input = Array2::<i64>::from_shape_vec((1, ids.len()), ids.to_vec()).unwrap();
    let input_lengths = Array1::<i64>::from_iter([ids.len() as i64]);
    let scales =
        Array1::<f32>::from_iter([NOISE_SCALE, LENGTH_SCALE / speed.max(0.1), NOISE_SCALE_W]);

    let input_t = Tensor::<i64>::from_array((
        [1, ids.len()],
        input.into_raw_vec_and_offset().0.into_boxed_slice(),
    ))
    .unwrap();
    let lengths_t = Tensor::<i64>::from_array((
        [1],
        input_lengths.into_raw_vec_and_offset().0.into_boxed_slice(),
    ))
    .unwrap();
    // sabela declares `scales` as rank-2 [batch, 3], unlike the coqui exports.
    let scales_t = Tensor::<f32>::from_array((
        [1, 3],
        scales.into_raw_vec_and_offset().0.into_boxed_slice(),
    ))
    .unwrap();

    let outputs = session
        .run(ort::inputs! {
            "input" => input_t,
            "input_lengths" => lengths_t,
            "scales" => scales_t,
        })
        .map_err(|e| PiperError::InferenceError(format!("Inference failed: {}", e)))?;

    let (_, audio) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| PiperError::InferenceError(format!("Failed to extract output: {}", e)))?;

    let mut samples = audio.to_vec();
    crate::normalize_audio(&mut samples);
    Ok(samples)
}
