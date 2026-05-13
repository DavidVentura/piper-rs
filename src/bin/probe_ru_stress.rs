use std::path::Path;

use espeak_rs::{init, text_to_phonemes};

// Probe whether the vendored espeak-ng 1.52.0 honors an explicit Russian
// stress mark (combining acute, U+0301, or the ruaccent convention of '+'
// placed before the stressed vowel).
//
// For each Russian homograph we ask espeak to phonemize three variants:
//   - bare word
//   - '+' before the stressed vowel (ruaccent raw output)
//   - U+0301 after the stressed vowel
//
// If the three outputs differ, espeak is honoring the stress hint. If they
// are all identical, espeak's Russian frontend is ignoring the hint and
// preprocessing with ruaccent would be useless — the fix would have to go
// into the espeak translator itself.
fn main() {
    let dir = std::env::var_os("PIPER_ESPEAKNG_DATA_DIRECTORY")
        .expect("set PIPER_ESPEAKNG_DATA_DIRECTORY to a directory containing espeak-ng-data/");
    init(Path::new(&dir)).expect("failed to init espeak");

    // (bare, stress-position-from-start-in-chars, annotation for display)
    // Stress position is the char index of the vowel to mark.
    let cases: &[(&str, usize, usize, &str)] = &[
        // замо́к (lock) vs за́мок (castle) — same letters, stress on у vs а
        ("замок", 1, 3, "замок: а-stress vs о-stress"),
        // мука́ (torment) vs му́ка (flour)
        ("мука", 1, 3, "мука: у-stress vs а-stress"),
        // писа́ть (to write) vs пи́сать (to pee)
        ("писать", 1, 3, "писать: и-stress vs а-stress"),
        // реки́ (of the river, gen.sg) vs ре́ки (rivers, nom.pl)
        ("реки", 1, 3, "реки: е-stress vs и-stress"),
    ];

    println!("=== Russian stress mark probe ===\n");

    for (bare, pos_a, pos_b, label) in cases {
        println!("-- {label} --");
        let variants = [
            ("bare".to_owned(), (*bare).to_owned()),
            (
                format!("+stress@{}", pos_a),
                insert_plus_before(bare, *pos_a),
            ),
            (
                format!("+stress@{}", pos_b),
                insert_plus_before(bare, *pos_b),
            ),
            (
                format!("U+0301@{}", pos_a),
                insert_acute_after(bare, *pos_a),
            ),
            (
                format!("U+0301@{}", pos_b),
                insert_acute_after(bare, *pos_b),
            ),
        ];
        for (tag, input) in &variants {
            let phon = text_to_phonemes(input, "ru", None)
                .map(|v| v.join(""))
                .unwrap_or_else(|e| format!("<err: {e}>"));
            println!("  {tag:15} in={input:?} -> {phon:?}");
        }
        println!();
    }
}

fn insert_plus_before(word: &str, char_index: usize) -> String {
    let mut out = String::with_capacity(word.len() + 1);
    for (i, c) in word.chars().enumerate() {
        if i == char_index {
            out.push('+');
        }
        out.push(c);
    }
    out
}

fn insert_acute_after(word: &str, char_index: usize) -> String {
    let mut out = String::with_capacity(word.len() + 2);
    for (i, c) in word.chars().enumerate() {
        out.push(c);
        if i == char_index {
            out.push('\u{0301}');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn init_for_tests() {
        let dir = std::env::var_os("PIPER_ESPEAKNG_DATA_DIRECTORY")
            .expect("set PIPER_ESPEAKNG_DATA_DIRECTORY for these tests");
        init(Path::new(&dir)).unwrap();
    }

    // Each tuple: (input, expected-substring-in-phonemes). We only check that
    // primary stress (`ˈ`) lands before the right phoneme — the surrounding
    // vowel reductions are implementation detail of espeak's Russian phoneme
    // tables.
    fn assert_stress_like(input: &str, expected_substr: &str) {
        let got = text_to_phonemes(input, "ru", None).unwrap().join("");
        assert!(
            got.contains(expected_substr),
            "input {input:?} → {got:?} did not contain {expected_substr:?}"
        );
    }

    #[test]
    fn castle_vs_lock() {
        init_for_tests();
        // за́мок (castle) — stress on а
        assert_stress_like("за́мок", "ˈɑm");
        // замо́к (lock) — stress on о
        assert_stress_like("замо́к", "ˈok");
    }

    #[test]
    fn flour_vs_torment() {
        init_for_tests();
        assert_stress_like("му́ка", "mˈu");
        assert_stress_like("мука́", "ˈɑ");
    }

    #[test]
    fn write_vs_pee() {
        init_for_tests();
        assert_stress_like("пи́сать", "ˈis");
        assert_stress_like("писа́ть", "ˈɑtʲ");
    }
}
