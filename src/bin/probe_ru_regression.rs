use std::path::Path;

use espeak_rs::{init, text_to_phonemes};

// Regression check: sentences without U+0301 should phonemize the same as
// before the rules patch. We don't have a fixed "known-good" reference here,
// so just print — eyeball the output to confirm nothing is broken.
fn main() {
    let dir = std::env::var_os("PIPER_ESPEAKNG_DATA_DIRECTORY")
        .expect("set PIPER_ESPEAKNG_DATA_DIRECTORY");
    init(Path::new(&dir)).unwrap();

    for text in [
        "привет мир",
        "я люблю программирование",
        "москва — столица россии",
        "доброе утро",
        "как твои дела сегодня",
        "сегодня хорошая погода",
        "спасибо большое за помощь",
    ] {
        let p = text_to_phonemes(text, "ru", None)
            .map(|v| v.join(" / "))
            .unwrap_or_else(|e| format!("<err: {e}>"));
        println!("{text:50} -> {p}");
    }
}
