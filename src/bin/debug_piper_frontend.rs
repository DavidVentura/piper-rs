use std::path::Path;

use piper_rs::{Backend, PiperModel};

fn main() {
    let mut args = std::env::args().skip(1);
    let config_path = args
        .next()
        .expect("usage: debug_piper_frontend <config.onnx.json> <text>");
    let text = args
        .next()
        .expect("usage: debug_piper_frontend <config.onnx.json> <text>");
    let onnx_path = config_path.replace(".onnx.json", ".onnx");

    let model = PiperModel::new(
        Path::new(&onnx_path),
        Path::new(&config_path),
        &Backend::Cpu,
    )
    .unwrap();
    let debug = model.debug_frontend(&text).unwrap();

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
