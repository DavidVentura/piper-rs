/*
ORT_DYLIB_PATH=./libonnxruntime.so cargo run --features japanese --example kokoro_phonemes \
    kokoro-v1.0.onnx voices-v1.0.bin mucab.bin
*/

use piper_rs::KokoroModel;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.onnx> <voices.bin> [mucab.bin]", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let voices_path = &args[2];
    let _mucab_path = args.get(3).map(|s| s.as_str());

    let samples = [
        ("en-us", "Hello! This is Kokoro, a text to speech model."),
        //("ja", "こんにちは世界。私はココロです。"),
        ("ja", "虹とは、水滴の反射、屈折、光の分散によって発生する気象現象であり、これにより空にさまざまな光が現れる。"),
        ("ko", "안녕하세요세계입니다.저는코코로입니다."),
    ];

    for (lang, text) in samples {
        let mut model = KokoroModel::new(
            Path::new(model_path),
            Path::new(voices_path),
            lang,
            &piper_rs::Backend::Cpu,
        )
        .unwrap();

        #[cfg(feature = "japanese")]
        if lang == "ja" {
            if let Some(dict_path) = _mucab_path {
                model.load_japanese_dict(dict_path).unwrap();
            }
        }

        let phonemes = model.phonemize(text).unwrap();
        println!("[{}] {}", lang, text);
        println!("  ipa: {}", phonemes);
        println!();
    }

    let model = KokoroModel::new(
        Path::new(model_path),
        Path::new(voices_path),
        "en-us",
        &piper_rs::Backend::Cpu,
    )
    .unwrap();
    if let Some(voices) = model.voices() {
        let mut speakers: Vec<_> = voices.iter().collect();
        speakers.sort_by_key(|(_, id)| *id);
        println!("Available voices ({}):", speakers.len());
        for (name, id) in &speakers {
            println!("  {}: {}", id, name);
        }
    }
}
