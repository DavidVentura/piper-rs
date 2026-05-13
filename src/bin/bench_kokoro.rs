use piper_rs::{init_espeak, KokoroModel};
use std::path::Path;
use std::time::Instant;

fn rss_mb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: f64 = rest
                .trim()
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
                .unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

fn main() {
    let mut args = std::env::args().skip(1);
    let espeak_data = args
        .next()
        .expect("Usage: bench_kokoro <espeak-ng-data-dir> <kokoro.onnx> <voices.bin> [text]");
    let model_path = args.next().expect("missing kokoro.onnx");
    let voices_path = args.next().expect("missing voices.bin");
    let text = args
        .next()
        .unwrap_or_else(|| "Hello! This is Kokoro, a text to speech model.".to_string());

    let rss_start = rss_mb();
    println!("rss start        : {:7.1} MB", rss_start);

    init_espeak(Path::new(&espeak_data)).expect("espeak init failed");
    let rss_post_espeak = rss_mb();
    println!(
        "rss after espeak : {:7.1} MB  (Δ {:+.1})",
        rss_post_espeak,
        rss_post_espeak - rss_start
    );

    let t0 = Instant::now();
    let mut model = KokoroModel::new(
        Path::new(&model_path),
        Path::new(&voices_path),
        "en-us",
        &piper_rs::Backend::Cpu,
    )
    .expect("KokoroModel::new failed");
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let rss_post_load = rss_mb();
    println!(
        "load (total)     : {:7.1} ms  rss {:7.1} MB  (Δ {:+.1})",
        load_ms,
        rss_post_load,
        rss_post_load - rss_post_espeak
    );

    let voice_id = model
        .voices()
        .and_then(|v| v.get("af_heart").copied())
        .unwrap_or(0);

    let t0 = Instant::now();
    let phonemes = model.phonemize(&text).expect("phonemize failed");
    let phon_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "phonemize        : {:7.1} ms  ({} chars)",
        phon_ms,
        phonemes.chars().count()
    );

    let t0 = Instant::now();
    let (samples, sr) = model
        .synthesize_phonemes(&phonemes, Some(voice_id), None)
        .expect("synth failed");
    let first_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let audio_s = samples.len() as f64 / sr as f64;
    println!(
        "first inference  : {:7.1} ms  ({:.2}s audio @ {} Hz)  rtf={:.2}",
        first_ms,
        audio_s,
        sr,
        first_ms / 1000.0 / audio_s
    );

    let iters = 3;
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = model
            .synthesize_phonemes(&phonemes, Some(voice_id), None)
            .expect("synth failed");
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "warm inference   : mean {:.1} ms  (n={})  rtf={:.2}",
        mean,
        iters,
        mean / 1000.0 / audio_s
    );

    // Also benchmark a very short input — first-sentence latency proxy.
    let short = model.phonemize("Hello.").expect("short phonemize failed");
    let t0 = Instant::now();
    let (s, sr) = model
        .synthesize_phonemes(&short, Some(voice_id), None)
        .expect("short synth failed");
    let short_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let short_audio = s.len() as f64 / sr as f64;
    let rss_end = rss_mb();
    println!(
        "short \"Hello.\"   : {:7.1} ms  ({:.2}s audio)  rtf={:.2}  rss {:7.1} MB",
        short_ms,
        short_audio,
        short_ms / 1000.0 / short_audio,
        rss_end
    );
}
