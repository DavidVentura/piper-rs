use piper_rs::init_espeak;
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
    let espeak_data = std::env::args()
        .nth(1)
        .expect("Usage: probe_espeak_langs <espeak-ng-data-dir>");

    println!("rss before init  : {:6.1} MB", rss_mb());

    let t = Instant::now();
    init_espeak(Path::new(&espeak_data)).expect("espeak init failed");
    println!(
        "init_espeak      : {:6.1} ms  rss {:6.1} MB",
        t.elapsed().as_secs_f64() * 1000.0,
        rss_mb()
    );

    let langs = [
        "en-US", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "uk", "tr", "cs", "sv", "da",
        "fi", "ar", "ja", "cmn", "ko",
    ];

    println!("\n--- first call per language (sets voice + phonemizes 'hello') ---");
    for lang in &langs {
        let t = Instant::now();
        let _ = espeak_rs::text_to_phonemes("hello world", lang, None).expect("phonemize failed");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("  {:<8} : {:6.1} ms  rss {:6.1} MB", lang, ms, rss_mb());
    }

    println!("\n--- second call per language (cached voice/dict) ---");
    for lang in &langs {
        let t = Instant::now();
        let _ = espeak_rs::text_to_phonemes("hello world", lang, None).expect("phonemize failed");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("  {:<8} : {:6.1} ms", lang, ms);
    }

    println!("\nfinal rss        : {:6.1} MB", rss_mb());
}
