use std::collections::HashMap;

use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;

use crate::PiperError;
use crate::PiperResult;

pub const BOS: char = '^';
pub const EOS: char = '$';
pub const PAD: char = '_';

#[derive(Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
}

#[derive(Deserialize)]
pub struct ESpeakConfig {
    pub voice: String,
}

#[derive(Deserialize, Clone)]
pub struct InferenceConfig {
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_w: f32,
}

#[derive(Deserialize)]
pub(crate) struct PiperRawConfig {
    pub audio: AudioConfig,
    pub espeak: ESpeakConfig,
    pub inference: InferenceConfig,
    pub num_speakers: u32,
    #[serde(default)]
    pub speaker_id_map: HashMap<String, i64>,
    pub phoneme_id_map: HashMap<char, Vec<i64>>,
}

pub(crate) fn kokoro_vocab() -> HashMap<char, i64> {
    // Kokoro v1.0 phoneme vocabulary (from hexgrad/Kokoro-82M config.json)
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

pub struct ModelConfig {
    pub sample_rate: u32,
    pub espeak_voice: String,
    pub speaker_id_map: HashMap<String, i64>,
    pub kind: ModelKind,
}

pub enum ModelKind {
    Piper {
        inference: InferenceConfig,
        num_speakers: u32,
        phoneme_id_map: HashMap<char, Vec<i64>>,
    },
    Kokoro {
        vocab: HashMap<char, i64>,
        voices: HashMap<i64, Array2<f32>>,
    },
}

impl ModelConfig {
    pub(crate) fn from_piper(raw: PiperRawConfig) -> Self {
        Self {
            sample_rate: raw.audio.sample_rate,
            espeak_voice: raw.espeak.voice,
            speaker_id_map: raw.speaker_id_map,
            kind: ModelKind::Piper {
                inference: raw.inference,
                num_speakers: raw.num_speakers,
                phoneme_id_map: raw.phoneme_id_map,
            },
        }
    }

    pub(crate) fn from_kokoro(
        voices: HashMap<i64, Array2<f32>>,
        speaker_id_map: HashMap<String, i64>,
        espeak_voice: String,
    ) -> Self {
        Self {
            sample_rate: 24000,
            espeak_voice,
            speaker_id_map,
            kind: ModelKind::Kokoro {
                vocab: kokoro_vocab(),
                voices,
            },
        }
    }
}

fn piper_phonemes_to_ids(phoneme_id_map: &HashMap<char, Vec<i64>>, phonemes: &str) -> Vec<i64> {
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

fn kokoro_phonemes_to_ids(vocab: &HashMap<char, i64>, phonemes: &str) -> Vec<i64> {
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

pub(crate) fn infer_piper(
    session: &mut Session,
    phoneme_id_map: &HashMap<char, Vec<i64>>,
    num_speakers: u32,
    phonemes: &str,
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
    speaker_id: i64,
) -> PiperResult<Vec<f32>> {
    let ids = piper_phonemes_to_ids(phoneme_id_map, phonemes);
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

pub(crate) fn infer_kokoro(
    session: &mut Session,
    vocab: &HashMap<char, i64>,
    voices: &HashMap<i64, Array2<f32>>,
    phonemes: &str,
    speed: f32,
    speaker_id: i64,
) -> PiperResult<Vec<f32>> {
    let ids = kokoro_phonemes_to_ids(vocab, phonemes);
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
