use espeak_rs::{init, text_to_phonemes};
use std::path::Path;
fn main() {
    init(Path::new(
        &std::env::var("PIPER_ESPEAKNG_DATA_DIRECTORY").unwrap(),
    ))
    .unwrap();
    for (label, text) in [
        ("bare воды", "воды"),
        ("в каплях воды (bare)", "в каплях воды"),
        ("во́ды (nom.pl)", "во\u{301}ды"),
        ("воды́ (gen.sg)", "воды\u{301}"),
        ("в каплях воды́", "в каплях воды\u{301}"),
    ] {
        let p = text_to_phonemes(text, "ru", None).unwrap().join("");
        println!("{:30} -> {}", label, p);
    }
}
