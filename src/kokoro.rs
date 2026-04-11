use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use ndarray::{Array2, Array3, Axis};
use ndarray_npy::NpzReader;
use ort::session::Session;
use ort::value::Tensor;

use crate::{build_session, espeak_phonemize, PiperError, PiperResult};

pub struct KokoroModel {
    session: Session,
    espeak_voice: String,
    vocab: HashMap<char, i64>,
    voices: HashMap<i64, Array2<f32>>,
    speaker_id_map: HashMap<String, i64>,
    #[cfg(feature = "japanese")]
    japanese_dict: Option<mucab::Dictionary<'static>>,
}

impl KokoroModel {
    pub fn new(
        model_path: &Path,
        voices_path: &Path,
        espeak_voice: &str,
    ) -> PiperResult<Self> {
        let (voices, speaker_id_map) = load_voices(voices_path)?;
        let session = build_session(model_path)?;
        Ok(Self {
            session,
            espeak_voice: espeak_voice.to_string(),
            vocab: kokoro_vocab(),
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
        let samples = infer(
            &mut self.session,
            &self.vocab,
            &self.voices,
            phonemes,
            speed.unwrap_or(1.0),
            speaker_id.unwrap_or(0),
        )?;
        Ok((samples, 24000))
    }

    pub fn phonemize(&mut self, text: &str) -> PiperResult<String> {
        #[cfg(feature = "japanese")]
        if self.espeak_voice == "ja" {
            if let Some(ref mut dict) = self.japanese_dict {
                return crate::japanese::phonemize(text, dict);
            }
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

fn load_voices(
    voices_path: &Path,
) -> PiperResult<(HashMap<i64, Array2<f32>>, HashMap<String, i64>)> {
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

fn phonemes_to_ids(vocab: &HashMap<char, i64>, phonemes: &str) -> Vec<i64> {
    let mut ids = Vec::with_capacity(phonemes.len() + 2);
    ids.push(0);
    for ch in phonemes.chars() {
        if let Some(&id) = vocab.get(&ch) {
            ids.push(id);
        }
    }
    ids.push(0);
    ids
}

fn infer(
    session: &mut Session,
    vocab: &HashMap<char, i64>,
    voices: &HashMap<i64, Array2<f32>>,
    phonemes: &str,
    speed: f32,
    speaker_id: i64,
) -> PiperResult<Vec<f32>> {
    let ids = phonemes_to_ids(vocab, phonemes);
    let token_count = ids.len() - 2;
    let input_len = ids.len();

    let voice_styles = voices.get(&speaker_id).ok_or_else(|| {
        PiperError::InferenceError(format!("Unknown speaker ID: {}", speaker_id))
    })?;
    let style_idx = token_count.min(voice_styles.shape()[0] - 1);
    let style_row = voice_styles.row(style_idx).to_owned();

    let input_t =
        Tensor::<i64>::from_array(([1, input_len], ids.into_boxed_slice())).unwrap();
    let style_t = Tensor::<f32>::from_array((
        [1, 256],
        style_row.into_raw_vec_and_offset().0.into_boxed_slice(),
    ))
    .unwrap();

    // v1.0 models name the first input "input_ids" and use i32 speed;
    // older models name it "tokens" and use f32 speed.
    let is_v1 = session.inputs()[0].name() == "input_ids";
    let outputs = if is_v1 {
        let speed_t =
            Tensor::<i32>::from_array(([1], vec![speed as i32].into_boxed_slice())).unwrap();
        session.run(ort::inputs![input_t, style_t, speed_t])
    } else {
        let speed_t =
            Tensor::<f32>::from_array(([1], vec![speed].into_boxed_slice())).unwrap();
        session.run(ort::inputs![input_t, style_t, speed_t])
    }
    .map_err(|e| PiperError::InferenceError(format!("Inference failed: {}", e)))?;

    let (_, audio) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| PiperError::InferenceError(format!("Failed to extract output: {}", e)))?;

    Ok(audio.to_vec())
}

/// Kokoro v1.0 phoneme vocabulary (from hexgrad/Kokoro-82M config.json)
fn kokoro_vocab() -> HashMap<char, i64> {
    [
        (';', 1), (':', 2), (',', 3), ('.', 4), ('!', 5), ('?', 6),
        ('\u{2014}', 9), ('\u{2026}', 10), ('"', 11), ('(', 12), (')', 13),
        ('\u{201c}', 14), ('\u{201d}', 15), (' ', 16), ('\u{0303}', 17),
        ('ʣ', 18), ('ʥ', 19), ('ʦ', 20), ('ʨ', 21), ('ᵝ', 22), ('\u{ab67}', 23),
        ('A', 24), ('I', 25), ('O', 31), ('Q', 33), ('S', 35), ('T', 36),
        ('W', 39), ('Y', 41), ('ᵊ', 42),
        ('a', 43), ('b', 44), ('c', 45), ('d', 46), ('e', 47), ('f', 48),
        ('h', 50), ('i', 51), ('j', 52), ('k', 53), ('l', 54), ('m', 55),
        ('n', 56), ('o', 57), ('p', 58), ('q', 59), ('r', 60), ('s', 61),
        ('t', 62), ('u', 63), ('v', 64), ('w', 65), ('x', 66), ('y', 67),
        ('z', 68),
        ('\u{0251}', 69), ('\u{0250}', 70), ('\u{0252}', 71), ('\u{00e6}', 72),
        ('\u{03b2}', 75), ('\u{0254}', 76), ('\u{0255}', 77), ('\u{00e7}', 78),
        ('\u{0256}', 80), ('\u{00f0}', 81), ('ʤ', 82), ('\u{0259}', 83),
        ('\u{025a}', 85), ('\u{025b}', 86), ('\u{025c}', 87),
        ('\u{025f}', 90), ('\u{0261}', 92), ('\u{0265}', 99),
        ('\u{0268}', 101), ('\u{026a}', 102), ('\u{029d}', 103),
        ('\u{026f}', 110), ('\u{0270}', 111), ('\u{014b}', 112),
        ('\u{0273}', 113), ('\u{0272}', 114), ('\u{0274}', 115),
        ('\u{00f8}', 116), ('\u{0278}', 118), ('\u{03b8}', 119),
        ('\u{0153}', 120), ('\u{0279}', 123), ('\u{027e}', 125),
        ('\u{027b}', 126), ('\u{0281}', 128), ('\u{027d}', 129),
        ('\u{0282}', 130), ('\u{0283}', 131), ('\u{0288}', 132),
        ('ʧ', 133), ('\u{028a}', 135), ('\u{028b}', 136),
        ('\u{028c}', 138), ('\u{0263}', 139), ('\u{0264}', 140),
        ('\u{03c7}', 142), ('\u{028e}', 143), ('\u{0292}', 147),
        ('\u{0294}', 148),
        ('\u{02c8}', 156), ('\u{02cc}', 157), ('\u{02d0}', 158),
        ('\u{02b0}', 162), ('\u{02b2}', 164),
        ('\u{2193}', 169), ('\u{2192}', 171), ('\u{2197}', 172), ('\u{2198}', 173),
        ('\u{1d7b}', 177),
    ]
    .into_iter()
    .collect()
}
