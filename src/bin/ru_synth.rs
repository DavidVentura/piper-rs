use std::io::{Read, Write};
use std::path::Path;

use piper_rs::{init_espeak, Backend, PiperModel};

// Synthesize a single chunk of Russian text into a WAV.
//
// Usage:
//     ru_synth <config.onnx.json> <out.wav> [text]
//
// If `text` is omitted, reads from stdin (useful for piping from
// scripts/ru_stress.py).
//
// Text can contain U+0301 combining acute marks — the patched ru_rules in
// the vendored espeak-ng will honor them as primary-stress hints.
fn main() {
    let mut args = std::env::args().skip(1);
    let config_path = args.next().expect(
        "usage: ru_synth <config.onnx.json> <out.wav> [text]\n       echo 'text' | ru_synth <config.onnx.json> <out.wav>",
    );
    let output_path = args.next().expect(
        "usage: ru_synth <config.onnx.json> <out.wav> [text]\n       echo 'text' | ru_synth <config.onnx.json> <out.wav>",
    );

    let text = match args.next() {
        Some(t) => t,
        None => {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .expect("failed to read text from stdin");
            buf.trim().to_owned()
        }
    };

    if text.is_empty() {
        eprintln!("no text provided");
        std::process::exit(2);
    }

    let espeak_data = std::env::var("PIPER_ESPEAKNG_DATA_DIRECTORY")
        .expect("set PIPER_ESPEAKNG_DATA_DIRECTORY to a directory containing espeak-ng-data/");
    init_espeak(Path::new(&espeak_data)).expect("failed to initialize espeak-ng");

    let onnx_path = config_path.replace(".onnx.json", ".onnx");
    let mut model = PiperModel::new(
        Path::new(&onnx_path),
        Path::new(&config_path),
        &Backend::Cpu,
    )
    .expect("failed to load piper model");

    eprintln!("synthesizing: {text:?}");
    let (samples, sample_rate) = model.synthesize(&text, None).expect("synthesis failed");

    let samples_i16: Vec<i16> = samples
        .iter()
        .map(|&s| (s * i16::MAX as f32) as i16)
        .collect();
    let mut file = std::fs::File::create(&output_path).expect("failed to create output file");
    write_wav(&mut file, &samples_i16, sample_rate, 1);
    println!(
        "saved {output_path} ({} samples @ {sample_rate} Hz)",
        samples_i16.len()
    );
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
