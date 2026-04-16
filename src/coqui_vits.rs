use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;

use crate::vits_tokenize;
use crate::{build_session, Backend, PiperError, PiperResult};

const DEFAULT_NOISE_SCALE: f32 = 0.3;
const DEFAULT_NOISE_SCALE_W: f32 = 0.3;
const DEFAULT_LENGTH_SCALE: f32 = 1.0;

#[derive(Default, Deserialize)]
struct AudioConfig {
    sample_rate: u32,
}

#[derive(Default, Deserialize)]
struct ModelArgsConfig {
    inference_noise_scale: Option<f32>,
    inference_noise_scale_dp: Option<f32>,
    length_scale: Option<f32>,
}

#[derive(Default, Deserialize)]
struct CharactersConfig {
    #[serde(default)]
    blank: Option<String>,
}

#[derive(Default, Deserialize)]
struct RawConfig {
    audio: AudioConfig,
    inference_noise_scale: Option<f32>,
    inference_noise_scale_dp: Option<f32>,
    length_scale: Option<f32>,
    #[serde(default)]
    model_args: ModelArgsConfig,
    #[serde(default)]
    add_blank: bool,
    #[serde(default)]
    characters: Option<CharactersConfig>,
}

pub struct CoquiVitsModel {
    session: Session,
    sample_rate: u32,
    token_to_id: HashMap<String, i64>,
    max_token_chars: usize,
    blank_id: Option<i64>,
    language_id: i64,
    speaker_id_map: HashMap<String, i64>,
    default_speaker_id: Option<i64>,
    has_langid_input: bool,
    speaker_input_name: Option<String>,
    noise_scale: f32,
    noise_scale_w: f32,
    length_scale: f32,
}

pub struct CoquiFrontendDebug {
    pub normalized: String,
    pub token_ids: Vec<i64>,
    pub dropped: Vec<String>,
}

impl CoquiVitsModel {
    pub fn new(
        model_path: &Path,
        config_path: &Path,
        language_code: &str,
        backend: &Backend,
    ) -> PiperResult<Self> {
        let file = File::open(config_path).map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "Failed to open config `{}`: {}",
                config_path.display(),
                e
            ))
        })?;
        let raw: RawConfig = serde_json::from_reader(file).map_err(|e| {
            PiperError::FailedToLoadResource(format!(
                "Failed to parse config `{}`: {}",
                config_path.display(),
                e
            ))
        })?;
        let config_dir = config_path.parent().ok_or_else(|| {
            PiperError::FailedToLoadResource(format!(
                "Config `{}` does not have a parent directory",
                config_path.display()
            ))
        })?;
        let token_to_id = vits_tokenize::load_tokens(&config_dir.join("tokens.txt"))?;
        let max_token_chars = vits_tokenize::max_token_chars(&token_to_id);
        let language_ids = load_optional_id_map(&config_dir.join("language_ids.json"))?;
        let speaker_id_map = load_optional_id_map(&config_dir.join("speaker_ids.json"))?;

        let session = build_session(model_path, backend)?;
        let input_names = session
            .inputs()
            .iter()
            .map(|input| input.name().to_owned())
            .collect::<Vec<_>>();
        let has_langid_input = input_names.iter().any(|name| name == "langid");
        let speaker_input_name = input_names
            .iter()
            .find(|name| matches!(name.as_str(), "sid" | "speaker_id" | "speaker"))
            .cloned();

        let language_id = resolve_language_id(language_code, &language_ids)?;
        let default_speaker_id = if speaker_input_name.is_some() && !speaker_id_map.is_empty() {
            speaker_id_map.values().copied().min()
        } else {
            None
        };

        let blank_id = if raw.add_blank {
            let blank_token = raw
                .characters
                .and_then(|c| c.blank)
                .unwrap_or_else(|| "<BLNK>".to_string());
            Some(*token_to_id.get(&blank_token).ok_or_else(|| {
                PiperError::FailedToLoadResource(format!(
                    "add_blank is true but blank token `{blank_token}` not found in tokens"
                ))
            })?)
        } else {
            None
        };

        Ok(Self {
            session,
            sample_rate: raw.audio.sample_rate,
            token_to_id,
            max_token_chars,
            blank_id,
            language_id,
            speaker_id_map,
            default_speaker_id,
            has_langid_input,
            speaker_input_name,
            noise_scale: raw
                .model_args
                .inference_noise_scale
                .or(raw.inference_noise_scale)
                .unwrap_or(DEFAULT_NOISE_SCALE),
            noise_scale_w: raw
                .model_args
                .inference_noise_scale_dp
                .or(raw.inference_noise_scale_dp)
                .unwrap_or(DEFAULT_NOISE_SCALE_W),
            length_scale: raw
                .model_args
                .length_scale
                .or(raw.length_scale)
                .unwrap_or(DEFAULT_LENGTH_SCALE),
        })
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        speaker_id: Option<i64>,
        speed: Option<f32>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let normalized = self.phonemize(text)?;
        self.synthesize_phonemes(&normalized, speaker_id, speed)
    }

    pub fn synthesize_phonemes(
        &mut self,
        phonemes: &str,
        speaker_id: Option<i64>,
        speed: Option<f32>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let ids = self.text_to_ids(phonemes);
        if ids.is_empty() {
            return Ok((Vec::new(), self.sample_rate));
        }

        let samples = infer(
            &mut self.session,
            &ids,
            self.language_id,
            speaker_id.or(self.default_speaker_id),
            speed.unwrap_or(1.0),
            self.noise_scale,
            self.length_scale,
            self.noise_scale_w,
            self.has_langid_input,
            self.speaker_input_name.as_deref(),
        )?;
        Ok((samples, self.sample_rate))
    }

    pub fn phonemize(&self, text: &str) -> PiperResult<String> {
        Ok(vits_tokenize::normalize_text(
            text,
            &self.token_to_id,
            self.max_token_chars,
        ))
    }

    pub fn token_ids(&self, text: &str) -> Vec<i64> {
        self.text_to_ids(text)
    }

    pub fn debug_frontend(&self, text: &str) -> CoquiFrontendDebug {
        let normalized =
            vits_tokenize::normalize_text(text, &self.token_to_id, self.max_token_chars);
        let token_ids = self.text_to_ids(&normalized);
        let dropped =
            vits_tokenize::collect_dropped_tokens(text, &self.token_to_id, self.max_token_chars);
        CoquiFrontendDebug {
            normalized,
            token_ids,
            dropped,
        }
    }

    pub fn voices(&self) -> Option<&HashMap<String, i64>> {
        if self.default_speaker_id.is_some() {
            Some(&self.speaker_id_map)
        } else {
            None
        }
    }

    fn text_to_ids(&self, text: &str) -> Vec<i64> {
        let ids = vits_tokenize::tokenize_to_ids(text, &self.token_to_id, self.max_token_chars);
        match self.blank_id {
            Some(blank_id) => vits_tokenize::intersperse(&ids, blank_id),
            None => ids,
        }
    }
}

fn load_optional_id_map(path: &Path) -> PiperResult<HashMap<String, i64>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let file = File::open(path).map_err(|e| {
        PiperError::FailedToLoadResource(format!("Failed to open `{}`: {}", path.display(), e))
    })?;
    serde_json::from_reader::<_, HashMap<String, i64>>(file).map_err(|e| {
        PiperError::FailedToLoadResource(format!("Failed to parse `{}`: {}", path.display(), e))
    })
}

fn resolve_language_id(
    language_code: &str,
    language_ids: &HashMap<String, i64>,
) -> PiperResult<i64> {
    if language_ids.is_empty() {
        return Ok(0);
    }

    if let Some(id) = language_ids.get(language_code) {
        return Ok(*id);
    }

    let normalized = language_code
        .split(['-', '_'])
        .next()
        .unwrap_or(language_code);
    if let Some(id) = language_ids.get(normalized) {
        return Ok(*id);
    }

    Err(PiperError::FailedToLoadResource(format!(
        "No language ID found for `{language_code}`"
    )))
}

#[allow(clippy::too_many_arguments)]
fn infer(
    session: &mut Session,
    ids: &[i64],
    language_id: i64,
    speaker_id: Option<i64>,
    speed: f32,
    noise_scale: f32,
    length_scale: f32,
    noise_scale_w: f32,
    has_langid_input: bool,
    speaker_input_name: Option<&str>,
) -> PiperResult<Vec<f32>> {
    let input = Array2::<i64>::from_shape_vec((1, ids.len()), ids.to_vec()).unwrap();
    let input_lengths = Array1::<i64>::from_iter([ids.len() as i64]);
    let scales =
        Array1::<f32>::from_iter([noise_scale, length_scale / speed.max(0.1), noise_scale_w]);

    let input_t = Tensor::<i64>::from_array((
        [1, ids.len()],
        input.into_raw_vec_and_offset().0.into_boxed_slice(),
    ))
    .unwrap();
    let lengths_t = Tensor::<i64>::from_array((
        [1],
        input_lengths.into_raw_vec_and_offset().0.into_boxed_slice(),
    ))
    .unwrap();
    let scales_t =
        Tensor::<f32>::from_array(([3], scales.into_raw_vec_and_offset().0.into_boxed_slice()))
            .unwrap();

    let mut inputs = ort::inputs! {
        "input" => input_t,
        "input_lengths" => lengths_t,
        "scales" => scales_t,
    };

    if has_langid_input {
        let lang_t =
            Tensor::<i64>::from_array(([1], vec![language_id].into_boxed_slice())).unwrap();
        inputs.push(("langid".into(), lang_t.into()));
    }

    if let Some(name) = speaker_input_name {
        if let Some(speaker_id) = speaker_id {
            let sid_t =
                Tensor::<i64>::from_array(([1], vec![speaker_id].into_boxed_slice())).unwrap();
            inputs.push((name.into(), sid_t.into()));
        }
    }

    let outputs = session
        .run(inputs)
        .map_err(|e| PiperError::InferenceError(format!("Inference failed: {}", e)))?;

    let (_, audio) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| PiperError::InferenceError(format!("Failed to extract output: {}", e)))?;

    let mut samples = audio.to_vec();
    crate::normalize_audio(&mut samples);
    Ok(samples)
}
