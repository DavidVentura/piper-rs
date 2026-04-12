// Test espeak-ng number and ordinal handling across languages.
//
// Requires espeak-ng-data with en_dict and es_dict.
// Usage: PIPER_ESPEAKNG_DATA_DIRECTORY=./espeak-ng-data cargo run --example test_numbers

fn main() {
    let data_dir = std::env::var("PIPER_ESPEAKNG_DATA_DIRECTORY").ok();

    let cases: &[(&str, &[&str])] = &[
        (
            "en",
            &[
                // Cardinals
                "1",
                "42",
                "1000",
                "1234567",
                // Ordinals
                "1st",
                "2nd",
                "3rd",
                "4th",
                "21st",
                "100th",
                // Numbers in context
                "I have 3 cats.",
                "There are 42 items in the list.",
                "The year 2025 was interesting.",
                // Edge cases
                "Call 911 for help.",
                "The temperature is -5 degrees.",
                "Pi is approximately 3.14159.",
            ],
        ),
        (
            "es",
            &[
                // Cardinals
                "1",
                "42",
                "1000",
                "1234567",
                // Ordinals (Spanish style)
                "1º",
                "2º",
                "3º",
                "4º",
                "21º",
                "100º",
                // Numbers in context
                "Tengo 3 gatos.",
                "Hay 42 elementos en la lista.",
                "El año 2025 fue interesante.",
                // Edge cases
                "Llama al 911 para ayuda.",
                "La temperatura es -5 grados.",
                "Pi es aproximadamente 3,14159.",
            ],
        ),
    ];

    for (lang, texts) in cases {
        let engine = match &data_dir {
            Some(dir) => espeak_ng::EspeakNg::with_data_dir(lang, std::path::Path::new(dir)),
            None => espeak_ng::EspeakNg::new(lang),
        };
        let engine = match engine {
            Ok(e) => e,
            Err(e) => {
                println!("[{lang}] SKIPPED: {e}  (missing {lang}_dict?)\n");
                continue;
            }
        };

        println!("=== {lang} ===");
        for text in *texts {
            match engine.text_to_phonemes(text) {
                Ok(ipa) => {
                    // Flag suspicious results: if digits survive into IPA output,
                    // the number wasn't translated
                    let has_digits = ipa.chars().any(|c| c.is_ascii_digit());
                    let flag = if has_digits {
                        " ⚠ DIGITS IN OUTPUT"
                    } else {
                        ""
                    };
                    println!("  {text:45} => {ipa}{flag}");
                }
                Err(e) => println!("  {text:45} => ERROR: {e}"),
            }
        }
        println!();
    }
}
