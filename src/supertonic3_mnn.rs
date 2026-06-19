use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mnn_sys::{InferenceConfig, ModuleEngine, NamedInput, TensorData};
use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

use crate::{PiperError, PiperResult};

const AVAILABLE_LANGS: &[&str] = &[
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
    "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
];

#[derive(Debug, Clone, Deserialize)]
struct Config {
    ae: AeConfig,
    ttl: TtlConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct AeConfig {
    sample_rate: u32,
    base_chunk_size: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct TtlConfig {
    latent_dim: u32,
    chunk_compress_factor: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct VoiceStyleData {
    style_ttl: StyleComponent,
    style_dp: StyleComponent,
}

#[derive(Debug, Clone, Deserialize)]
struct StyleComponent {
    data: Vec<Vec<Vec<f32>>>,
    dims: Vec<usize>,
    #[serde(rename = "type")]
    dtype: String,
}

#[derive(Debug, Clone)]
struct Style {
    ttl: Vec<f32>,
    ttl_shape: Vec<usize>,
    dp: Vec<f32>,
    dp_shape: Vec<usize>,
}

#[derive(Debug, Clone)]
struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

pub struct Supertonic3MnnModel {
    config: Config,
    unicode_indexer: Vec<i64>,
    dp: ModuleEngine,
    text_encoder: ModuleEngine,
    vector_estimator: ModuleEngine,
    vocoder: ModuleEngine,
}

impl Supertonic3MnnModel {
    pub fn new(mnn_dir: &Path, asset_dir: &Path) -> PiperResult<Self> {
        Self::new_with_config(mnn_dir, asset_dir, InferenceConfig::new().with_threads(4))
    }

    pub fn new_with_config(
        mnn_dir: &Path,
        asset_dir: &Path,
        inference_config: InferenceConfig,
    ) -> PiperResult<Self> {
        let config: Config = read_json(&asset_dir.join("tts.json"))?;
        let unicode_indexer: Vec<i64> = read_json(&asset_dir.join("unicode_indexer.json"))?;

        let dp = load_engine(
            &mnn_dir.join("duration_predictor.mnn"),
            &["text_ids", "style_dp", "text_mask"],
            &["duration"],
            &inference_config,
        )?;
        let text_encoder = load_engine(
            &mnn_dir.join("text_encoder.mnn"),
            &["text_ids", "style_ttl", "text_mask"],
            &["text_emb"],
            &inference_config,
        )?;
        let vector_estimator = load_engine(
            &mnn_dir.join("vector_estimator.mnn"),
            &[
                "noisy_latent",
                "text_emb",
                "style_ttl",
                "latent_mask",
                "text_mask",
                "current_step",
                "total_step",
            ],
            &["denoised_latent"],
            &inference_config,
        )?;
        let vocoder = load_engine(
            &mnn_dir.join("vocoder.mnn"),
            &["latent"],
            &["wav_tts"],
            &inference_config,
        )?;

        Ok(Self {
            config,
            unicode_indexer,
            dp,
            text_encoder,
            vector_estimator,
            vocoder,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.ae.sample_rate
    }

    pub fn synthesize(
        &self,
        text: &str,
        lang: &str,
        voice_style_path: &Path,
        total_step: usize,
        speed: f32,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let style = load_voice_style(voice_style_path)?;
        let started_at = Instant::now();
        let (samples, audio_s) =
            self.synthesize_with_style(text, lang, &style, total_step, speed)?;
        let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        let rtf = if audio_s > 0.0 {
            elapsed_ms / 1000.0 / audio_s as f64
        } else {
            0.0
        };
        log::info!(
            "supertonic3_mnn.inference: took_ms={:.1} audio_s={:.2} rtf={:.3} samples={}",
            elapsed_ms,
            audio_s,
            rtf,
            samples.len()
        );
        Ok((samples, self.sample_rate()))
    }

    fn synthesize_with_style(
        &self,
        text: &str,
        lang: &str,
        style: &Style,
        total_step: usize,
        speed: f32,
    ) -> PiperResult<(Vec<f32>, f32)> {
        if style.ttl_shape.first().copied() != Some(1) || style.dp_shape.first().copied() != Some(1)
        {
            return Err(PiperError::FailedToLoadResource(
                "Supertonic3MnnModel currently supports one voice style at a time".to_string(),
            ));
        }

        let max_len = if lang == "ko" || lang == "ja" {
            120
        } else {
            300
        };
        let chunks = chunk_text(text, max_len);
        let mut out = Vec::new();
        let mut total_duration = 0.0f32;

        for (idx, chunk) in chunks.iter().enumerate() {
            let (mut samples, duration) = self.infer_one(chunk, lang, style, total_step, speed)?;
            if idx > 0 {
                let silence_len = (0.3 * self.sample_rate() as f32) as usize;
                out.extend(std::iter::repeat(0.0).take(silence_len));
                total_duration += 0.3;
            }
            out.append(&mut samples);
            total_duration += duration;
        }

        Ok((out, total_duration))
    }

    fn infer_one(
        &self,
        text: &str,
        lang: &str,
        style: &Style,
        total_step: usize,
        speed: f32,
    ) -> PiperResult<(Vec<f32>, f32)> {
        let (text_ids, text_mask, text_len) = process_text(&self.unicode_indexer, text, lang)?;
        let text_ids_shape = [1usize, text_len];
        let text_mask_shape = [1usize, 1, text_len];

        let duration_out = self.run_first(
            &self.dp,
            &[
                NamedInput {
                    name: "text_ids",
                    data: TensorData::I32(&text_ids),
                    shape: &text_ids_shape,
                },
                NamedInput {
                    name: "style_dp",
                    data: TensorData::F32(&style.dp),
                    shape: &style.dp_shape,
                },
                NamedInput {
                    name: "text_mask",
                    data: TensorData::F32(&text_mask),
                    shape: &text_mask_shape,
                },
            ],
            "duration",
        )?;
        let mut duration = duration_out.data.first().copied().ok_or_else(|| {
            PiperError::InferenceError("duration_predictor returned no duration".to_string())
        })?;
        if speed > 0.0 {
            duration /= speed;
        }
        duration = duration.max(0.05);

        let text_emb = self.run_first(
            &self.text_encoder,
            &[
                NamedInput {
                    name: "text_ids",
                    data: TensorData::I32(&text_ids),
                    shape: &text_ids_shape,
                },
                NamedInput {
                    name: "style_ttl",
                    data: TensorData::F32(&style.ttl),
                    shape: &style.ttl_shape,
                },
                NamedInput {
                    name: "text_mask",
                    data: TensorData::F32(&text_mask),
                    shape: &text_mask_shape,
                },
            ],
            "text_emb",
        )?;

        let (mut latent, latent_mask, latent_shape, latent_mask_shape) =
            self.sample_noisy_latent(duration);
        let total_step_buf = [total_step as f32];

        for step in 0..total_step {
            let current_step_buf = [step as f32];
            let denoised = self.run_first(
                &self.vector_estimator,
                &[
                    NamedInput {
                        name: "noisy_latent",
                        data: TensorData::F32(&latent),
                        shape: &latent_shape,
                    },
                    NamedInput {
                        name: "text_emb",
                        data: TensorData::F32(&text_emb.data),
                        shape: &text_emb.shape,
                    },
                    NamedInput {
                        name: "style_ttl",
                        data: TensorData::F32(&style.ttl),
                        shape: &style.ttl_shape,
                    },
                    NamedInput {
                        name: "latent_mask",
                        data: TensorData::F32(&latent_mask),
                        shape: &latent_mask_shape,
                    },
                    NamedInput {
                        name: "text_mask",
                        data: TensorData::F32(&text_mask),
                        shape: &text_mask_shape,
                    },
                    NamedInput {
                        name: "current_step",
                        data: TensorData::F32(&current_step_buf),
                        shape: &[1],
                    },
                    NamedInput {
                        name: "total_step",
                        data: TensorData::F32(&total_step_buf),
                        shape: &[1],
                    },
                ],
                "denoised_latent",
            )?;
            latent = denoised.data;
        }

        let wav = self.run_first(
            &self.vocoder,
            &[NamedInput {
                name: "latent",
                data: TensorData::F32(&latent),
                shape: &latent_shape,
            }],
            "wav_tts",
        )?;
        let trim_len = (duration * self.sample_rate() as f32) as usize;
        let mut samples = wav.data;
        samples.truncate(trim_len.min(samples.len()));
        Ok((samples, duration))
    }

    fn sample_noisy_latent(&self, duration: f32) -> (Vec<f32>, Vec<f32>, Vec<usize>, Vec<usize>) {
        let wav_len = (duration * self.sample_rate() as f32) as usize;
        let chunk_size =
            (self.config.ae.base_chunk_size * self.config.ttl.chunk_compress_factor) as usize;
        let latent_len = ((wav_len + chunk_size - 1) / chunk_size).max(1);
        let latent_dim =
            (self.config.ttl.latent_dim * self.config.ttl.chunk_compress_factor) as usize;
        let latent_shape = vec![1, latent_dim, latent_len];
        let latent_mask_shape = vec![1, 1, latent_len];
        let latent_mask = vec![1.0f32; latent_len];
        let mut rng = XorShift64::new(0x5355_5054_4f4e_4943);
        let mut latent = Vec::with_capacity(latent_dim * latent_len);
        while latent.len() < latent_dim * latent_len {
            let (a, b) = rng.normal_pair();
            latent.push(a);
            if latent.len() < latent_dim * latent_len {
                latent.push(b);
            }
        }
        (latent, latent_mask, latent_shape, latent_mask_shape)
    }

    fn run_first(
        &self,
        engine: &ModuleEngine,
        inputs: &[NamedInput<'_>],
        output_name: &str,
    ) -> PiperResult<Tensor> {
        let outputs = engine
            .run_named_dynamic(inputs, &[output_name])
            .map_err(|e| PiperError::InferenceError(format!("MNN inference failed: {}", e)))?;
        let output = outputs.into_iter().next().ok_or_else(|| {
            PiperError::InferenceError(format!("MNN returned no `{}` output", output_name))
        })?;
        Ok(Tensor {
            data: output.data,
            shape: output.shape,
        })
    }
}

fn load_engine(
    path: &Path,
    input_names: &[&str],
    output_names: &[&str],
    config: &InferenceConfig,
) -> PiperResult<ModuleEngine> {
    ModuleEngine::from_file(path, input_names, output_names, Some(config.clone())).map_err(|e| {
        PiperError::FailedToLoadResource(format!(
            "Failed to load Supertonic-3 MNN module `{}`: {}",
            path.display(),
            e
        ))
    })
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> PiperResult<T> {
    let file = File::open(path).map_err(|e| {
        PiperError::FailedToLoadResource(format!("Failed to open `{}`: {}", path.display(), e))
    })?;
    serde_json::from_reader(BufReader::new(file)).map_err(|e| {
        PiperError::FailedToLoadResource(format!("Failed to parse `{}`: {}", path.display(), e))
    })
}

fn load_voice_style(path: &Path) -> PiperResult<Style> {
    let data: VoiceStyleData = read_json(path)?;
    Ok(Style {
        ttl: flatten_component(&data.style_ttl, path, "style_ttl")?,
        ttl_shape: data.style_ttl.dims.clone(),
        dp: flatten_component(&data.style_dp, path, "style_dp")?,
        dp_shape: data.style_dp.dims.clone(),
    })
}

fn flatten_component(component: &StyleComponent, path: &Path, name: &str) -> PiperResult<Vec<f32>> {
    if component.dtype != "float32" {
        return Err(PiperError::FailedToLoadResource(format!(
            "`{}` in `{}` has unsupported dtype `{}`",
            name,
            path.display(),
            component.dtype
        )));
    }
    if component.dims.len() != 3 {
        return Err(PiperError::FailedToLoadResource(format!(
            "`{}` in `{}` must be rank 3",
            name,
            path.display()
        )));
    }
    let expected: usize = component.dims.iter().product();
    let mut flat = Vec::with_capacity(expected);
    for batch in &component.data {
        for row in batch {
            flat.extend_from_slice(row);
        }
    }
    if flat.len() != expected {
        return Err(PiperError::FailedToLoadResource(format!(
            "`{}` in `{}` has {} values, expected {} from dims {:?}",
            name,
            path.display(),
            flat.len(),
            expected,
            component.dims
        )));
    }
    Ok(flat)
}

fn process_text(
    unicode_indexer: &[i64],
    text: &str,
    lang: &str,
) -> PiperResult<(Vec<i32>, Vec<f32>, usize)> {
    let text = preprocess_text(text, lang)?;
    let mut ids = Vec::with_capacity(text.chars().count());
    for ch in text.chars() {
        let idx = ch as usize;
        let id = unicode_indexer.get(idx).copied().ok_or_else(|| {
            PiperError::PhonemizationError(format!(
                "Character U+{:04X} is outside Supertonic-3 unicode_indexer",
                idx
            ))
        })?;
        ids.push(i32::try_from(id).map_err(|_| {
            PiperError::PhonemizationError(format!(
                "Supertonic-3 token id {} for U+{:04X} does not fit int32",
                id, idx
            ))
        })?);
    }
    let len = ids.len();
    Ok((ids, vec![1.0; len], len))
}

fn preprocess_text(text: &str, lang: &str) -> PiperResult<String> {
    if !AVAILABLE_LANGS.contains(&lang) {
        return Err(PiperError::PhonemizationError(format!(
            "Invalid Supertonic-3 language `{}`",
            lang
        )));
    }

    let mut text: String = text
        .nfkd()
        .flat_map(|ch| match ch {
            '\u{2013}' | '\u{2011}' | '\u{2014}' => Some('-'),
            '_' | '[' | ']' | '|' | '/' | '#' | '\u{2192}' | '\u{2190}' => Some(' '),
            '\u{201c}' | '\u{201d}' => Some('"'),
            '\u{2018}' | '\u{2019}' | '\u{00b4}' | '`' => Some('\''),
            '\u{2665}' | '\u{2606}' | '\u{2661}' | '\u{00a9}' | '\\' => None,
            ch if is_emoji_or_symbol(ch) => None,
            ch => Some(ch),
        })
        .collect();

    text = text.replace('@', " at ");
    text = text.replace("e.g.,", "for example, ");
    text = text.replace("i.e.,", "that is, ");
    text = collapse_spaces(&text);
    text = fix_punctuation_spacing(&text);
    text = collapse_repeated_quotes(&text);
    text = collapse_spaces(&text);

    if !text.is_empty() && !ends_with_terminal_punctuation(&text) {
        text.push('.');
    }

    Ok(format!("<{}>{}", lang, text))
}

fn is_emoji_or_symbol(ch: char) -> bool {
    let code = ch as u32;
    matches!(
        code,
        0x1F600..=0x1F64F
            | 0x1F300..=0x1F5FF
            | 0x1F680..=0x1F6FF
            | 0x1F700..=0x1F77F
            | 0x1F780..=0x1F7FF
            | 0x1F800..=0x1F8FF
            | 0x1F900..=0x1F9FF
            | 0x1FA00..=0x1FA6F
            | 0x1FA70..=0x1FAFF
            | 0x2600..=0x26FF
            | 0x2700..=0x27BF
            | 0x1F1E6..=0x1F1FF
    )
}

fn collapse_spaces(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

fn fix_punctuation_spacing(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if matches!(ch, ',' | '.' | '!' | '?' | ';' | ':' | '\'') && out.ends_with(' ') {
            out.pop();
        }
        out.push(ch);
    }
    out
}

fn collapse_repeated_quotes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev = '\0';
    for ch in text.chars() {
        if (ch == '"' || ch == '\'') && ch == prev {
            continue;
        }
        out.push(ch);
        prev = ch;
    }
    out
}

fn ends_with_terminal_punctuation(text: &str) -> bool {
    text.chars().last().is_some_and(|ch| {
        matches!(
            ch,
            '.' | '!'
                | '?'
                | ';'
                | ':'
                | ','
                | '\''
                | '"'
                | ')'
                | ']'
                | '}'
                | '\u{2026}'
                | '\u{3002}'
                | '\u{300d}'
                | '\u{3011}'
                | '\u{3009}'
                | '\u{300b}'
                | '\u{203a}'
                | '\u{00bb}'
        )
    })
}

fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut chunks = Vec::new();
    for paragraph in text.split("\n\n").map(str::trim).filter(|s| !s.is_empty()) {
        if paragraph.chars().count() <= max_len {
            chunks.push(paragraph.to_string());
            continue;
        }

        let mut current = String::new();
        for sentence in split_sentences(paragraph) {
            let candidate_len = current.chars().count() + sentence.chars().count() + 1;
            if !current.is_empty() && candidate_len > max_len {
                chunks.push(current.trim().to_string());
                current.clear();
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(sentence.trim());
        }
        if !current.trim().is_empty() {
            chunks.push(current.trim().to_string());
        }
    }

    if chunks.is_empty() {
        vec![String::new()]
    } else {
        chunks
    }
}

fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    for (idx, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let next = idx + ch.len_utf8();
            if text[next..].starts_with(' ') && !is_abbreviation(&text[start..next]) {
                sentences.push(&text[start..next]);
                start = next;
            }
        }
    }
    if start < text.len() {
        sentences.push(&text[start..]);
    }
    sentences
}

fn is_abbreviation(text: &str) -> bool {
    const ABBREVIATIONS: &[&str] = &[
        "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.", "St.", "Ave.", "Rd.", "Blvd.", "Dept.",
        "Inc.", "Ltd.", "Co.", "Corp.", "etc.", "vs.", "i.e.", "e.g.", "Ph.D.",
    ];
    let trimmed = text.trim_end();
    ABBREVIATIONS.iter().any(|abbr| trimmed.ends_with(abbr))
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32_open01(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        let mantissa = ((x >> 40) as u32).max(1);
        mantissa as f32 / ((1u32 << 24) as f32)
    }

    fn normal_pair(&mut self) -> (f32, f32) {
        let u1 = self.next_f32_open01().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32_open01();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (radius * theta.cos(), radius * theta.sin())
    }
}

#[allow(dead_code)]
fn _asset_path(asset_dir: &Path, file: &str) -> PathBuf {
    asset_dir.join(file)
}
