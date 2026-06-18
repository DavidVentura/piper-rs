//! Galician grapheme-to-phoneme fallback for words not in the cotovia lexicon.
//!
//! Emits the same sabela VITS token ids the cotovia frontend produces, recovered
//! by aligning the baked lexicon (see scripts/gl_sabela_lexicon). Galician
//! orthography is highly regular, so this covers OOV names/words intelligibly.
//! The one thing rules can't recover is the *lexical* open/close mid-vowel
//! contrast (e: 89 close / 115 open, o: 99 / 125) — for OOV we default to the
//! more common close form and accept occasional errors on rare words.

// vowels: stressed / unstressed, plus i/u glide forms
const A_STRESS: i64 = 85;
const A: i64 = 22;
const E_STRESS: i64 = 89; // close (default); open 115 is lexical, unrecoverable
const E: i64 = 26;
const I_STRESS: i64 = 93;
const I: i64 = 30;
const I_GLIDE: i64 = 31;
const O_STRESS: i64 = 99; // close (default); open 125 is lexical
const O: i64 = 36;
const U_STRESS: i64 = 105;
const U: i64 = 42;
const U_GLIDE: i64 = 44;

// consonants
const P: i64 = 37;
const T: i64 = 41;
const K: i64 = 32;
// stop [b d g] word-initially and after a nasal; fricative [β ð ɣ] elsewhere
const B_STOP: i64 = 23;
const B_FRIC: i64 = 49; // b and v share these
const D_STOP: i64 = 25;
const D_FRIC: i64 = 51;
const G_STOP: i64 = 28;
const G_FRIC: i64 = 54;
const THETA: i64 = 67; // c before e/i, and z
const SH: i64 = 66; // x
const VELAR_FRIC: i64 = 45; // j, g before e/i
const F: i64 = 27;
const S: i64 = 40;
const M: i64 = 34;
const N: i64 = 35;
const NG: i64 = 61; // final -n, n before a velar, and the nh digraph
const L: i64 = 33;
const LL: i64 = 73;
const R_TAP: i64 = 39;
const R_TRILL: i64 = 65; // word-initial r, after n/l/s, and rr
const ENYE: i64 = 57; // ñ
const CH: i64 = 50;

fn is_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'i' | 'o' | 'u' | 'á' | 'é' | 'í' | 'ó' | 'ú' | 'ü'
    )
}

fn is_accented(c: char) -> bool {
    matches!(c, 'á' | 'é' | 'í' | 'ó' | 'ú')
}

/// Index of the stressed vowel: a written accent wins; otherwise the
/// penultimate vowel if the word ends in a vowel, `n`, or `s`, else the last.
fn stressed_vowel_index(chars: &[char]) -> Option<usize> {
    let vowels: Vec<usize> = chars
        .iter()
        .enumerate()
        .filter(|(_, c)| is_vowel(**c))
        .map(|(i, _)| i)
        .collect();
    if vowels.is_empty() {
        return None;
    }
    if let Some(&i) = vowels.iter().find(|&&i| is_accented(chars[i])) {
        return Some(i);
    }
    let last = *chars.last().unwrap();
    let penult = is_vowel(last) || last == 'n' || last == 's';
    if penult && vowels.len() >= 2 {
        Some(vowels[vowels.len() - 2])
    } else {
        Some(*vowels.last().unwrap())
    }
}

fn vowel_id(c: char, stressed: bool, glide: bool) -> i64 {
    match c {
        'a' | 'á' => {
            if stressed {
                A_STRESS
            } else {
                A
            }
        }
        'e' | 'é' => {
            if stressed {
                E_STRESS
            } else {
                E
            }
        }
        'o' | 'ó' => {
            if stressed {
                O_STRESS
            } else {
                O
            }
        }
        'i' | 'í' => {
            if glide {
                I_GLIDE
            } else if stressed {
                I_STRESS
            } else {
                I
            }
        }
        'u' | 'ú' | 'ü' => {
            if glide {
                U_GLIDE
            } else if stressed {
                U_STRESS
            } else {
                U
            }
        }
        _ => A,
    }
}

fn is_front_vowel(c: char) -> bool {
    matches!(c, 'e' | 'i' | 'é' | 'í')
}

// Galician cardinals — validated to reproduce cotovia's own digit reading
// exactly (see scripts/gl_sabela_lexicon). Numbers are expanded to words before
// tokenization; the words then route through the lexicon/g2p like any text.
const UNITS: [&str; 20] = [
    "cero", "un", "dous", "tres", "catro", "cinco", "seis", "sete", "oito", "nove", "dez", "once",
    "doce", "trece", "catorce", "quince", "dezaseis", "dezasete", "dezaoito", "dezanove",
];

fn tens_word(t: u64) -> &'static str {
    match t {
        20 => "vinte",
        30 => "trinta",
        40 => "corenta",
        50 => "cincuenta",
        60 => "sesenta",
        70 => "setenta",
        80 => "oitenta",
        90 => "noventa",
        _ => "",
    }
}

fn hundreds_word(h: u64) -> &'static str {
    match h {
        200 => "douscentos",
        300 => "trescentos",
        400 => "catrocentos",
        500 => "cincocentos",
        600 => "seiscentos",
        700 => "setecentos",
        800 => "oitocentos",
        900 => "novecentos",
        _ => "",
    }
}

fn under_100(n: u64) -> String {
    if n < 20 {
        return UNITS[n as usize].to_string();
    }
    let (t, u) = ((n / 10) * 10, n % 10);
    if u == 0 {
        tens_word(t).to_string()
    } else {
        format!("{} e {}", tens_word(t), UNITS[u as usize])
    }
}

fn under_1000(n: u64) -> String {
    if n < 100 {
        return under_100(n);
    }
    let (h, r) = ((n / 100) * 100, n % 100);
    let head = if h == 100 {
        if r > 0 { "cento" } else { "cen" }.to_string()
    } else {
        hundreds_word(h).to_string()
    };
    if r == 0 {
        head
    } else {
        format!("{} {}", head, under_100(r))
    }
}

fn num_to_galician(n: u64) -> String {
    if n < 1000 {
        return under_1000(n);
    }
    if n < 1_000_000 {
        let (th, r) = (n / 1000, n % 1000);
        let head = if th == 1 {
            "mil".to_string()
        } else {
            format!("{} mil", under_1000(th))
        };
        return if r == 0 {
            head
        } else {
            format!("{} {}", head, under_1000(r))
        };
    }
    let (mill, r) = (n / 1_000_000, n % 1_000_000);
    let head = if mill == 1 {
        "un millón".to_string()
    } else {
        format!("{} millóns", under_1000(mill))
    };
    if r == 0 {
        head
    } else {
        format!("{} {}", head, num_to_galician(r))
    }
}

/// Replace runs of ASCII digits with their Galician cardinal words.
pub(crate) fn normalize_numbers(text: &str) -> String {
    let mut out = String::new();
    let mut digits = String::new();
    let flush = |out: &mut String, digits: &str| match digits.parse::<u64>() {
        Ok(n) => {
            out.push(' ');
            out.push_str(&num_to_galician(n));
            out.push(' ');
        }
        Err(_) => out.push_str(digits), // out of u64 range: leave the digits
    };
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            if !digits.is_empty() {
                flush(&mut out, &digits);
                digits.clear();
            }
            out.push(ch);
        }
    }
    if !digits.is_empty() {
        flush(&mut out, &digits);
    }
    out
}

pub fn galician_g2p(word: &str) -> Vec<i64> {
    let chars: Vec<char> = word.chars().collect();
    let stress_idx = stressed_vowel_index(&chars);
    let mut out = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        let prev = if i > 0 { Some(chars[i - 1]) } else { None };
        let next = chars.get(i + 1).copied();
        // stop allophone after pause or nasal (also after `l` for `d`)
        let after_pause_or_nasal = i == 0 || matches!(prev, Some('m') | Some('n'));

        // two-char graphemes
        if c == 'c' && next == Some('h') {
            out.push(CH);
            i += 2;
            continue;
        }
        if c == 'l' && next == Some('l') {
            out.push(LL);
            i += 2;
            continue;
        }
        if c == 'r' && next == Some('r') {
            out.push(R_TRILL);
            i += 2;
            continue;
        }
        if c == 'n' && next == Some('h') {
            out.push(NG);
            i += 2;
            continue;
        }
        // qu / gu before e,i: the u is silent
        if c == 'q' && next == Some('u') {
            out.push(K);
            i += 2;
            if matches!(chars.get(i), Some(&v) if is_front_vowel(v)) {
                // u already consumed; fall through to the vowel next iteration
            }
            continue;
        }
        if c == 'g'
            && next == Some('u')
            && matches!(chars.get(i + 2), Some(&v) if is_front_vowel(v))
        {
            out.push(if after_pause_or_nasal { G_STOP } else { G_FRIC });
            i += 2; // skip the silent u
            continue;
        }

        if is_vowel(c) {
            let stressed = stress_idx == Some(i);
            let glide = !stressed
                && matches!(c, 'i' | 'u' | 'ü')
                && (matches!(prev, Some(p) if is_vowel(p))
                    || matches!(next, Some(n) if is_vowel(n)));
            out.push(vowel_id(c, stressed, glide));
            i += 1;
            continue;
        }

        let id = match c {
            'p' => Some(P),
            't' => Some(T),
            'k' => Some(K),
            'f' => Some(F),
            's' => Some(S),
            'm' => Some(M),
            'l' => Some(L),
            'ñ' => Some(ENYE),
            'x' => Some(SH),
            'j' => Some(VELAR_FRIC),
            'z' => Some(THETA),
            'h' => None, // silent
            'c' => Some(if matches!(next, Some(n) if is_front_vowel(n)) {
                THETA
            } else {
                K
            }),
            'q' => Some(K),
            'b' | 'v' | 'w' => Some(if after_pause_or_nasal { B_STOP } else { B_FRIC }),
            'd' => Some(if after_pause_or_nasal || prev == Some('l') {
                D_STOP
            } else {
                D_FRIC
            }),
            'g' => Some(if matches!(next, Some(n) if is_front_vowel(n)) {
                VELAR_FRIC
            } else if after_pause_or_nasal {
                G_STOP
            } else {
                G_FRIC
            }),
            'n' => {
                // /n/ before a vowel and before t/d/v/y; /ŋ/ (assimilated) word-
                // finally and before any other consonant — derived from the lexicon.
                let onset =
                    matches!(next, Some(c) if is_vowel(c) || matches!(c, 't' | 'd' | 'v' | 'y'));
                Some(if onset { N } else { NG })
            }
            'r' => {
                let trill = i == 0 || matches!(prev, Some('n') | Some('l') | Some('s'));
                Some(if trill { R_TRILL } else { R_TAP })
            }
            'y' => {
                let intervocalic = matches!(prev, Some(p) if is_vowel(p))
                    && matches!(next, Some(n) if is_vowel(n));
                Some(if intervocalic {
                    LL
                } else if matches!(next, Some(n) if is_vowel(n)) {
                    I_GLIDE
                } else {
                    I
                })
            }
            _ => None, // unknown letter: drop
        };
        if let Some(id) = id {
            out.push(id);
        }
        i += 1;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_match_known_cotovia_readings() {
        for (n, w) in [
            (0u64, "cero"),
            (16, "dezaseis"),
            (21, "vinte e un"),
            (100, "cen"),
            (101, "cento un"),
            (123, "cento vinte e tres"),
            (200, "douscentos"),
            (1000, "mil"),
            (2024, "dous mil vinte e catro"),
            (1_000_000, "un millón"),
        ] {
            assert_eq!(num_to_galician(n), w, "n={n}");
        }
    }

    // Cross-check the Rust port against the cotovia-validated word list.
    // GL_NUM_TEST=~/gl_sabela_build/num_test.tsv cargo test --release numbers_match_validated -- --nocapture
    #[test]
    fn numbers_match_validated_list() {
        let Ok(path) = std::env::var("GL_NUM_TEST") else {
            return;
        };
        let text = std::fs::read_to_string(path).unwrap();
        let mut bad = 0;
        for line in text.lines() {
            let Some((n, words)) = line.split_once('\t') else {
                continue;
            };
            if num_to_galician(n.parse().unwrap()) != words {
                if bad < 10 {
                    eprintln!(
                        "  {n}: got={:?} want={words:?}",
                        num_to_galician(n.parse().unwrap())
                    );
                }
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{bad} mismatches vs validated list");
    }

    // Run with: GL_LEXICON=~/gl_sabela_build/gl_lexicon.txt \
    //   cargo test --release eval_against_lexicon -- --nocapture
    #[test]
    fn eval_against_lexicon() {
        let Ok(path) = std::env::var("GL_LEXICON") else {
            eprintln!("GL_LEXICON unset; skipping");
            return;
        };
        let text = std::fs::read_to_string(path).unwrap();
        let mut residual = String::new(); // entries the g2p cannot reproduce
        let (mut exact, mut total, mut tok_ok, mut tok_n) = (0usize, 0usize, 0usize, 0usize);
        let mut shown = 0;
        for line in text.lines() {
            let Some((word, ids_str)) = line.split_once('\t') else {
                continue;
            };
            let gold: Vec<i64> = ids_str
                .split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect();
            let got = galician_g2p(&word.to_lowercase());
            total += 1;
            if got == gold {
                exact += 1;
            } else {
                residual.push_str(line);
                residual.push('\n');
                if shown < 25 {
                    eprintln!("  {word}: gold={gold:?} got={got:?}");
                    shown += 1;
                }
            }
            // positional token accuracy on equal-length sequences
            if got.len() == gold.len() {
                tok_n += gold.len();
                tok_ok += got.iter().zip(&gold).filter(|(a, b)| a == b).count();
            }
        }
        eprintln!(
            "exact-word: {}/{} ({:.1}%) | token (eq-len): {}/{} ({:.1}%)",
            exact,
            total,
            100.0 * exact as f64 / total as f64,
            tok_ok,
            tok_n,
            100.0 * tok_ok as f64 / tok_n as f64
        );
        if let Ok(out) = std::env::var("RESIDUAL_OUT") {
            std::fs::write(&out, &residual).unwrap();
            eprintln!("residual: {} entries -> {}", total - exact, out);
        }
    }
}
