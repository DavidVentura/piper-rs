/*
ORT_DYLIB_PATH=./libonnxruntime.so cargo run --features japanese --example kokoro_wav -- \
    ~/Downloads/kokoro-v1.0.int8.onnx ~/Downloads/voices-v1.0.bin ~/git/mucab/out/mucab.bin
*/

use piper_rs::KokoroModel;
use std::io::Write;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <model.onnx> <voices.bin> <mucab.bin>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let voices_path = &args[2];
    let _mucab_path = &args[3];

    let samples = [
        (
            "en-us",
            "af_heart",
            "Hello! This is Kokoro, a text to speech model.",
        ),
        ("ja", "jf_alpha", "こんにちは世界。私はココロです。"),
        ("ko", "jf_alpha", "안녕하세요. 저는 코코로입니다."),
    ];

    for (lang, voice_name, text) in samples {
        let mut model =
            KokoroModel::new(Path::new(model_path), Path::new(voices_path), lang).unwrap();

        #[cfg(feature = "japanese")]
        if lang == "ja" {
            model.load_japanese_dict(_mucab_path).unwrap();
        }

        let voice_id = model
            .voices()
            .and_then(|v| v.get(voice_name).copied())
            .unwrap_or(0);

        let phonemes = model.phonemize(text).unwrap();
        println!("[{}] {} (voice: {}={})", lang, text, voice_name, voice_id);
        println!("  ipa: {}", phonemes);

        let (audio, sample_rate) = model.synthesize(text, Some(voice_id), None).unwrap();

        let filename = format!("kokoro_{}.wav", lang);
        let samples_i16: Vec<i16> = audio
            .iter()
            .map(|&s| (s * i16::MAX as f32) as i16)
            .collect();
        let mut file = std::fs::File::create(&filename).unwrap();
        write_wav(&mut file, &samples_i16, sample_rate, 1);
        println!(
            "  saved: {} ({:.1}s)\n",
            filename,
            audio.len() as f32 / sample_rate as f32
        );
    }
}

fn write_wav(w: &mut impl Write, samples: &[i16], sample_rate: u32, channels: u16) {
    let data_len = (samples.len() * 2) as u32;
    let byte_rate = sample_rate * channels as u32 * 2;
    w.write_all(b"RIFF").unwrap();
    w.write_all(&(36 + data_len).to_le_bytes()).unwrap();
    w.write_all(b"WAVEfmt ").unwrap();
    w.write_all(&16u32.to_le_bytes()).unwrap();
    w.write_all(&1u16.to_le_bytes()).unwrap();
    w.write_all(&channels.to_le_bytes()).unwrap();
    w.write_all(&sample_rate.to_le_bytes()).unwrap();
    w.write_all(&byte_rate.to_le_bytes()).unwrap();
    w.write_all(&(channels * 2).to_le_bytes()).unwrap();
    w.write_all(&16u16.to_le_bytes()).unwrap();
    w.write_all(b"data").unwrap();
    w.write_all(&data_len.to_le_bytes()).unwrap();
    for &s in samples {
        w.write_all(&s.to_le_bytes()).unwrap();
    }
}
