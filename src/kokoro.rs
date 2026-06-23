use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use ndarray::{Array2, Array3, Axis};
use ndarray_npy::NpzReader;

use crate::{PiperError, PiperResult};

pub(crate) fn load_voices(
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

    let mut names: Vec<String> = npz
        .names()
        .map_err(|e| PiperError::FailedToLoadResource(format!("Failed to list voices: {}", e)))?;
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

pub(crate) fn phonemes_to_ids(vocab: &HashMap<char, i64>, phonemes: &str) -> Vec<i64> {
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

/// Kokoro v1.0 phoneme vocabulary (from hexgrad/Kokoro-82M config.json)
pub(crate) fn kokoro_vocab() -> HashMap<char, i64> {
    [
        (';', 1),
        (':', 2),
        (',', 3),
        ('.', 4),
        ('!', 5),
        ('?', 6),
        ('\u{2014}', 9),
        ('\u{2026}', 10),
        ('"', 11),
        ('(', 12),
        (')', 13),
        ('\u{201c}', 14),
        ('\u{201d}', 15),
        (' ', 16),
        ('\u{0303}', 17),
        ('ʣ', 18),
        ('ʥ', 19),
        ('ʦ', 20),
        ('ʨ', 21),
        ('ᵝ', 22),
        ('\u{ab67}', 23),
        ('A', 24),
        ('I', 25),
        ('O', 31),
        ('Q', 33),
        ('S', 35),
        ('T', 36),
        ('W', 39),
        ('Y', 41),
        ('ᵊ', 42),
        ('a', 43),
        ('b', 44),
        ('c', 45),
        ('d', 46),
        ('e', 47),
        ('f', 48),
        ('h', 50),
        ('i', 51),
        ('j', 52),
        ('k', 53),
        ('l', 54),
        ('m', 55),
        ('n', 56),
        ('o', 57),
        ('p', 58),
        ('q', 59),
        ('r', 60),
        ('s', 61),
        ('t', 62),
        ('u', 63),
        ('v', 64),
        ('w', 65),
        ('x', 66),
        ('y', 67),
        ('z', 68),
        ('\u{0251}', 69),
        ('\u{0250}', 70),
        ('\u{0252}', 71),
        ('\u{00e6}', 72),
        ('\u{03b2}', 75),
        ('\u{0254}', 76),
        ('\u{0255}', 77),
        ('\u{00e7}', 78),
        ('\u{0256}', 80),
        ('\u{00f0}', 81),
        ('ʤ', 82),
        ('\u{0259}', 83),
        ('\u{025a}', 85),
        ('\u{025b}', 86),
        ('\u{025c}', 87),
        ('\u{025f}', 90),
        ('\u{0261}', 92),
        ('\u{0265}', 99),
        ('\u{0268}', 101),
        ('\u{026a}', 102),
        ('\u{029d}', 103),
        ('\u{026f}', 110),
        ('\u{0270}', 111),
        ('\u{014b}', 112),
        ('\u{0273}', 113),
        ('\u{0272}', 114),
        ('\u{0274}', 115),
        ('\u{00f8}', 116),
        ('\u{0278}', 118),
        ('\u{03b8}', 119),
        ('\u{0153}', 120),
        ('\u{0279}', 123),
        ('\u{027e}', 125),
        ('\u{027b}', 126),
        ('\u{0281}', 128),
        ('\u{027d}', 129),
        ('\u{0282}', 130),
        ('\u{0283}', 131),
        ('\u{0288}', 132),
        ('ʧ', 133),
        ('\u{028a}', 135),
        ('\u{028b}', 136),
        ('\u{028c}', 138),
        ('\u{0263}', 139),
        ('\u{0264}', 140),
        ('\u{03c7}', 142),
        ('\u{028e}', 143),
        ('\u{0292}', 147),
        ('\u{0294}', 148),
        ('\u{02c8}', 156),
        ('\u{02cc}', 157),
        ('\u{02d0}', 158),
        ('\u{02b0}', 162),
        ('\u{02b2}', 164),
        ('\u{2193}', 169),
        ('\u{2192}', 171),
        ('\u{2197}', 172),
        ('\u{2198}', 173),
        ('\u{1d7b}', 177),
    ]
    .into_iter()
    .collect()
}
