use std::path::Path;

use piper_rs::{Backend, PiperModel};

fn main() {
    let config_path = std::env::args()
        .nth(1)
        .expect("usage: debug_piper_phoneme_string <config.onnx.json> <phoneme-string>");
    let phonemes = std::env::args()
        .nth(2)
        .expect("usage: debug_piper_phoneme_string <config.onnx.json> <phoneme-string>");
    let onnx_path = config_path.replace(".onnx.json", ".onnx");

    let model = PiperModel::new(
        Path::new(&onnx_path),
        Path::new(&config_path),
        &Backend::Cpu,
    )
    .unwrap();
    let debug = model.debug_phoneme_string(&phonemes).unwrap();

    println!("phoneme_type: {}", debug.phoneme_type);
    println!("phonemes: {}", debug.phonemes);
    println!(
        "tokens ({}): {}",
        debug.tokens.len(),
        debug.tokens.join(" | ")
    );
    println!(
        "token_ids ({}): {}",
        debug.token_ids.len(),
        debug
            .token_ids
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    );
}
