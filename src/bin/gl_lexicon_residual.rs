//! Reduce a full `word\tids` Galician lexicon to the residual: only the entries
//! the Galician g2p cannot reproduce. At runtime cotovia_vits regenerates the
//! rest via galician_g2p, so this is lossless. The residual is DERIVED from the
//! current g2p — regenerate it whenever gl_g2p changes.
//!
//! Usage: gl_lexicon_residual <full_lexicon.txt> <residual_out.txt>
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: gl_lexicon_residual <full.txt> <residual.txt>");
        std::process::exit(2);
    }
    let text = std::fs::read_to_string(&args[1]).expect("read full lexicon");
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(&args[2]).expect("create residual"));
    let (mut kept, mut total) = (0usize, 0usize);
    for line in text.lines() {
        let Some((word, ids_str)) = line.split_once('\t') else {
            continue;
        };
        total += 1;
        let gold: Vec<i64> = ids_str
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        if piper_rs::galician_g2p(&word.to_lowercase()) != gold {
            writeln!(out, "{line}").unwrap();
            kept += 1;
        }
    }
    eprintln!(
        "residual: kept {kept}/{total} entries ({:.1}% dropped)",
        100.0 * (total - kept) as f64 / total as f64
    );
}
