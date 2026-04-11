// Katakana-to-IPA (M2P) table derived from:
//   https://github.com/hexgrad/misaki, specifically misaki/ja.py
//   Copyright (c) 2025 hexgrad, MIT License

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::PiperError;
use crate::PiperResult;

/// Japanese punctuation → ASCII equivalents for Kokoro vocab
const PUNCT_MAP: &[(char, char)] = &[
    ('、', ','),
    ('。', '.'),
    ('！', '!'),
    ('？', '?'),
    ('：', ':'),
    ('；', ';'),
    ('（', '('),
    ('）', ')'),
    ('「', '"'),
    ('」', '"'),
    ('『', '"'),
    ('』', '"'),
    ('【', '"'),
    ('】', '"'),
    ('〈', '"'),
    ('〉', '"'),
    ('《', '"'),
    ('》', '"'),
    ('«', '"'),
    ('»', '"'),
];

fn punct_map() -> &'static HashMap<char, char> {
    static MAP: OnceLock<HashMap<char, char>> = OnceLock::new();
    MAP.get_or_init(|| PUNCT_MAP.iter().copied().collect())
}

/// Katakana/hiragana mora → IPA phoneme table.
///
/// Based on misaki's M2P table (hexgrad/misaki ja.py), with non-vocab
/// characters replaced:
///   - ASCII 'g' → IPA 'ɡ' (U+0261, vocab id 92)
///   - ᶄ→kj, ᶃ→ɡj, ᶀ→bj, ᶁ→dj, ᶆ→mj, ᶈ→pj, ᶉ→rj, ƫ→tj
///
/// Two-char entries are matched before single-char (greedy).
fn m2p() -> &'static HashMap<&'static str, &'static str> {
    static M2P: OnceLock<HashMap<&str, &str>> = OnceLock::new();
    M2P.get_or_init(|| {
        [
            // --- Single katakana (88 entries) ---
            // Vowels
            ("ァ", "a"),
            ("ア", "a"),
            ("ィ", "i"),
            ("イ", "i"),
            ("ゥ", "u"),
            ("ウ", "u"),
            ("ェ", "e"),
            ("エ", "e"),
            ("ォ", "o"),
            ("オ", "o"),
            // Ka-row
            ("カ", "ka"),
            ("ガ", "ɡa"),
            ("キ", "ki"),
            ("ギ", "ɡi"),
            ("ク", "ku"),
            ("グ", "ɡu"),
            ("ケ", "ke"),
            ("ゲ", "ɡe"),
            ("コ", "ko"),
            ("ゴ", "ɡo"),
            // Sa-row
            ("サ", "sa"),
            ("ザ", "za"),
            ("シ", "ɕi"),
            ("ジ", "ʥi"),
            ("ス", "su"),
            ("ズ", "zu"),
            ("セ", "se"),
            ("ゼ", "ze"),
            ("ソ", "so"),
            ("ゾ", "zo"),
            // Ta-row
            ("タ", "ta"),
            ("ダ", "da"),
            ("チ", "ʨi"),
            ("ヂ", "ʥi"),
            ("ツ", "ʦu"),
            ("ヅ", "zu"),
            ("テ", "te"),
            ("デ", "de"),
            ("ト", "to"),
            ("ド", "do"),
            // Na-row
            ("ナ", "na"),
            ("ニ", "ni"),
            ("ヌ", "nu"),
            ("ネ", "ne"),
            ("ノ", "no"),
            // Ha-row
            ("ハ", "ha"),
            ("バ", "ba"),
            ("パ", "pa"),
            ("ヒ", "hi"),
            ("ビ", "bi"),
            ("ピ", "pi"),
            ("フ", "fu"),
            ("ブ", "bu"),
            ("プ", "pu"),
            ("ヘ", "he"),
            ("ベ", "be"),
            ("ペ", "pe"),
            ("ホ", "ho"),
            ("ボ", "bo"),
            ("ポ", "po"),
            // Ma-row
            ("マ", "ma"),
            ("ミ", "mi"),
            ("ム", "mu"),
            ("メ", "me"),
            ("モ", "mo"),
            // Ya-row
            ("ャ", "ja"),
            ("ヤ", "ja"),
            ("ュ", "ju"),
            ("ユ", "ju"),
            ("ョ", "jo"),
            ("ヨ", "jo"),
            // Ra-row
            ("ラ", "ra"),
            ("リ", "ri"),
            ("ル", "ru"),
            ("レ", "re"),
            ("ロ", "ro"),
            // Wa-row
            ("ヮ", "wa"),
            ("ワ", "wa"),
            ("ヰ", "i"),
            ("ヱ", "e"),
            ("ヲ", "o"),
            // Misc
            ("ヴ", "vu"),
            ("ヵ", "ka"),
            ("ヶ", "ke"),
            ("ヷ", "va"),
            ("ヸ", "vi"),
            ("ヹ", "ve"),
            ("ヺ", "vo"),
            // Special
            ("ッ", "ʔ"),
            ("ン", "ɴ"),
            ("ー", "ː"),
            // --- Two-char combinations (102 entries) ---
            ("イェ", "je"),
            ("ウィ", "wi"),
            ("ウゥ", "wu"),
            ("ウェ", "we"),
            ("ウォ", "wo"),
            // Ki-row palatalized
            ("キィ", "kji"),
            ("キェ", "kje"),
            ("キャ", "kja"),
            ("キュ", "kju"),
            ("キョ", "kjo"),
            // Gi-row palatalized
            ("ギィ", "ɡji"),
            ("ギェ", "ɡje"),
            ("ギャ", "ɡja"),
            ("ギュ", "ɡju"),
            ("ギョ", "ɡjo"),
            // Ku/Gu + small vowel (labialized)
            ("クァ", "kwa"),
            ("クィ", "kwi"),
            ("クゥ", "kwu"),
            ("クェ", "kwe"),
            ("クォ", "kwo"),
            ("クヮ", "kwa"),
            ("グァ", "ɡwa"),
            ("グィ", "ɡwi"),
            ("グゥ", "ɡwu"),
            ("グェ", "ɡwe"),
            ("グォ", "ɡwo"),
            ("グヮ", "ɡwa"),
            // Sh-row
            ("シェ", "ɕe"),
            ("シャ", "ɕa"),
            ("シュ", "ɕu"),
            ("ショ", "ɕo"),
            // J-row
            ("ジェ", "ʥe"),
            ("ジャ", "ʥa"),
            ("ジュ", "ʥu"),
            ("ジョ", "ʥo"),
            // S/Z + small i
            ("スィ", "si"),
            ("ズィ", "zi"),
            // Ch-row
            ("チェ", "ʨe"),
            ("チャ", "ʨa"),
            ("チュ", "ʨu"),
            ("チョ", "ʨo"),
            // Dj-row (ヂ combinations)
            ("ヂェ", "ʥe"),
            ("ヂャ", "ʥa"),
            ("ヂュ", "ʥu"),
            ("ヂョ", "ʥo"),
            // Ts-row
            ("ツァ", "ʦa"),
            ("ツィ", "ʦi"),
            ("ツェ", "ʦe"),
            ("ツォ", "ʦo"),
            // T + small vowels
            ("ティ", "ti"),
            ("テェ", "tje"),
            ("テャ", "tja"),
            ("テュ", "tju"),
            ("テョ", "tjo"),
            // D + small vowels
            ("ディ", "di"),
            ("デェ", "dje"),
            ("デャ", "dja"),
            ("デュ", "dju"),
            ("デョ", "djo"),
            ("トゥ", "tu"),
            ("ドゥ", "du"),
            // Ni-row palatalized
            ("ニィ", "nji"),
            ("ニェ", "nje"),
            ("ニャ", "nja"),
            ("ニュ", "nju"),
            ("ニョ", "njo"),
            // Hi-row palatalized
            ("ヒィ", "çi"),
            ("ヒェ", "çe"),
            ("ヒャ", "ça"),
            ("ヒュ", "çu"),
            ("ヒョ", "ço"),
            // Bi-row palatalized
            ("ビィ", "bji"),
            ("ビェ", "bje"),
            ("ビャ", "bja"),
            ("ビュ", "bju"),
            ("ビョ", "bjo"),
            // Pi-row palatalized
            ("ピィ", "pji"),
            ("ピェ", "pje"),
            ("ピャ", "pja"),
            ("ピュ", "pju"),
            ("ピョ", "pjo"),
            // F-row
            ("ファ", "fa"),
            ("フィ", "fi"),
            ("フェ", "fe"),
            ("フォ", "fo"),
            // Mi-row palatalized
            ("ミィ", "mji"),
            ("ミェ", "mje"),
            ("ミャ", "mja"),
            ("ミュ", "mju"),
            ("ミョ", "mjo"),
            // Ri-row palatalized
            ("リィ", "rji"),
            ("リェ", "rje"),
            ("リャ", "rja"),
            ("リュ", "rju"),
            ("リョ", "rjo"),
            // Vu-row
            ("ヴァ", "va"),
            ("ヴィ", "vi"),
            ("ヴェ", "ve"),
            ("ヴォ", "vo"),
            ("ヴャ", "bja"),
            ("ヴュ", "bju"),
            ("ヴョ", "bjo"),
        ]
        .into_iter()
        .collect()
    })
}

fn to_katakana(c: char) -> char {
    if ('\u{3041}'..='\u{3096}').contains(&c) {
        char::from_u32(c as u32 + 0x60).unwrap_or(c)
    } else {
        c
    }
}

fn reading_to_ipa(reading: &str) -> String {
    let table = m2p();
    let chars: Vec<char> = reading.chars().map(to_katakana).collect();
    let mut result = String::new();
    let mut i = 0;
    while i < chars.len() {
        if i + 1 < chars.len() {
            let two: String = [chars[i], chars[i + 1]].iter().collect();
            if let Some(ipa) = table.get(two.as_str()) {
                result.push_str(ipa);
                i += 2;
                continue;
            }
        }
        let one: String = chars[i].to_string();
        if let Some(ipa) = table.get(one.as_str()) {
            result.push_str(ipa);
        } else {
            let pmap = punct_map();
            if let Some(&ascii) = pmap.get(&chars[i]) {
                result.push(ascii);
            } else {
                result.push(chars[i]);
            }
        }
        i += 1;
    }
    result
}

pub fn phonemize(text: &str, dict: &mut mucab::Dictionary) -> PiperResult<String> {
    let words = mucab::analyze(text, dict);
    if words.is_empty() {
        return Err(PiperError::PhonemizationError(
            "No words produced from Japanese text".to_string(),
        ));
    }

    let mut parts: Vec<String> = Vec::with_capacity(words.len());
    for word in &words {
        let ipa = reading_to_ipa(&word.reading);
        if !ipa.is_empty() {
            parts.push(ipa);
        }
    }
    Ok(parts.join(" "))
}
