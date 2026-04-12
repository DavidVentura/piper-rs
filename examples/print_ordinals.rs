fn main() {
    let en = espeak_ng::EspeakNg::new("en").unwrap();
    for w in ["1st", "2nd", "3rd", "4th", "21st", "100th"] {
        println!("en  {w:6} => {:?}", en.text_to_phonemes(w));
    }
    let es = espeak_ng::EspeakNg::new("es").unwrap();
    for w in ["1º", "2º", "3º"] {
        println!("es  {w:6} => {:?}", es.text_to_phonemes(w));
    }
}
