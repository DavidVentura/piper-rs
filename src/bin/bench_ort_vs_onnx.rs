use piper_rs::{init_espeak, PiperModel};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn bench_model(model_path: &Path, config_path: &Path, phonemes: &str, iters: usize) {
    let label = model_path.file_name().unwrap().to_string_lossy();
    println!("\n=== {} ===", label);

    let t0 = Instant::now();
    let mut model = PiperModel::new(model_path, config_path, &piper_rs::Backend::Cpu).unwrap();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("load             : {:7.1} ms", load_ms);

    let t0 = Instant::now();
    let (samples, sr) = model.synthesize_phonemes(phonemes, None).unwrap();
    let first_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "first inference  : {:7.1} ms  ({} samples @ {} Hz)",
        first_ms,
        samples.len(),
        sr
    );

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = model.synthesize_phonemes(phonemes, None).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "warm inference   : mean {:.1} ms  min {:.1}  max {:.1}  (n={})",
        mean, min, max, iters
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let espeak_data = args
        .next()
        .expect("Usage: bench_ort_vs_onnx <espeak-ng-data-dir> <model.onnx> <model.onnx.json> [more model+config pairs...]");
    init_espeak(Path::new(&espeak_data)).expect("espeak init failed");

    let rest: Vec<String> = args.collect();
    assert!(
        rest.len() >= 2 && rest.len() % 2 == 0,
        "expected pairs of (model.onnx, model.onnx.json) after espeak data dir"
    );

    let pairs: Vec<(PathBuf, PathBuf)> = rest
        .chunks_exact(2)
        .map(|c| (PathBuf::from(&c[0]), PathBuf::from(&c[1])))
        .collect();

    let (warm_model, warm_config) = &pairs[0];
    let warmup = PiperModel::new(warm_model, warm_config, &piper_rs::Backend::Cpu)
        .expect("warmup load failed");
    let phonemes = warmup.phonemize("hola").expect("phonemize failed");
    drop(warmup);
    println!("phonemes for \"hola\": {:?}", phonemes);

    let iters = 5;
    for (model_path, config_path) in &pairs {
        bench_model(model_path, config_path, &phonemes, iters);
    }
}
