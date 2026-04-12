/*
ORT_DYLIB_PATH=./libonnxruntime.so cargo run --example sherpa_vits_wav -- \
    /path/to/model.onnx /path/to/config.json cantonese_sample.wav
*/

use piper_rs::SherpaVitsModel;
use std::io::Write;
use std::path::Path;

//const SAMPLE_TEXT: &str = "你好，世界！今日天氣好好。";
const SAMPLE_TEXT: &str = "好,以下是 123 字。Google 翻譯";

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .expect("Please specify model.onnx path");
    let config_path = std::env::args()
        .nth(2)
        .expect("Please specify config.json path");
    let output_path = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "sherpa_vits_sample.wav".to_string());

    let mut model = SherpaVitsModel::new(
        Path::new(&model_path),
        Path::new(&config_path),
        &piper_rs::Backend::Cpu,
    )
    .unwrap();

    let phonemes = model.phonemize(SAMPLE_TEXT).unwrap();
    let token_ids = model.token_ids(SAMPLE_TEXT);
    println!("text: {}", SAMPLE_TEXT);
    println!("phonemes: {}", phonemes);
    println!("token_ids: {:?}", token_ids);

    let (samples, sample_rate) = model.synthesize(SAMPLE_TEXT, None, None).unwrap();
    let samples_i16: Vec<i16> = samples
        .iter()
        .map(|&s| (s * i16::MAX as f32) as i16)
        .collect();

    let mut file = std::fs::File::create(&output_path).unwrap();
    write_wav(&mut file, &samples_i16, sample_rate, 1);
    println!("Saved to {}", output_path);
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
