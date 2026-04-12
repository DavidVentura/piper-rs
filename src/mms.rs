use std::collections::HashMap;
use std::path::Path;

use ndarray::Array1;
use ort::session::Session;
use ort::value::Tensor;

use crate::vits_tokenize;
use crate::{build_session, Backend, PiperError, PiperResult};

const DEFAULT_NOISE_SCALE: f32 = 0.667;
const DEFAULT_NOISE_SCALE_W: f32 = 0.8;
const DEFAULT_LENGTH_SCALE: f32 = 1.0;
const DEFAULT_SAMPLE_RATE: u32 = 16_000;

pub struct MmsModel {
    session: Session,
    sample_rate: u32,
    token_to_id: HashMap<String, i64>,
    max_token_chars: usize,
    blank_id: Option<i64>,
}

impl MmsModel {
    pub fn new(model_path: &Path, tokens_path: &Path, backend: &Backend) -> PiperResult<Self> {
        let token_to_id = vits_tokenize::load_tokens(tokens_path)?;
        let max_token_chars = vits_tokenize::max_token_chars(&token_to_id);
        let session = build_session(model_path, backend)?;
        let (sample_rate, blank_id) = {
            let metadata = session.metadata().ok();
            let sample_rate = metadata
                .as_ref()
                .and_then(|m| m.custom("sample_rate"))
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(DEFAULT_SAMPLE_RATE);
            let add_blank = metadata
                .as_ref()
                .and_then(|m| m.custom("add_blank"))
                .is_some_and(|v| v == "1" || v == "true");
            (sample_rate, if add_blank { Some(0i64) } else { None })
        };

        Ok(Self {
            session,
            sample_rate,
            token_to_id,
            max_token_chars,
            blank_id,
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
        _speaker_id: Option<i64>,
        speed: Option<f32>,
    ) -> PiperResult<(Vec<f32>, u32)> {
        let ids = self.text_to_ids(phonemes);
        if ids.is_empty() {
            return Ok((Vec::new(), self.sample_rate));
        }

        let samples = infer(
            &mut self.session,
            &ids,
            speed.unwrap_or(1.0),
            DEFAULT_NOISE_SCALE,
            DEFAULT_NOISE_SCALE_W,
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

    pub fn voices(&self) -> Option<&HashMap<String, i64>> {
        None
    }

    fn text_to_ids(&self, text: &str) -> Vec<i64> {
        let ids = vits_tokenize::tokenize_to_ids(text, &self.token_to_id, self.max_token_chars);
        match self.blank_id {
            Some(blank_id) => vits_tokenize::intersperse(&ids, blank_id),
            None => ids,
        }
    }
}

fn infer(
    session: &mut Session,
    ids: &[i64],
    speed: f32,
    noise_scale: f32,
    noise_scale_w: f32,
) -> PiperResult<Vec<f32>> {
    let input = Array1::<i64>::from_iter(ids.iter().copied());
    let input_len = Array1::<i64>::from_iter([ids.len() as i64]);
    let noise = Array1::<f32>::from_iter([noise_scale]);
    let length = Array1::<f32>::from_iter([DEFAULT_LENGTH_SCALE / speed.max(0.1)]);
    let noise_w = Array1::<f32>::from_iter([noise_scale_w]);

    let input_t = Tensor::<i64>::from_array((
        [1, ids.len()],
        input.into_raw_vec_and_offset().0.into_boxed_slice(),
    ))
    .unwrap();
    let input_len_t = Tensor::<i64>::from_array((
        [1],
        input_len.into_raw_vec_and_offset().0.into_boxed_slice(),
    ))
    .unwrap();
    let noise_t =
        Tensor::<f32>::from_array(([1], noise.into_raw_vec_and_offset().0.into_boxed_slice()))
            .unwrap();
    let length_t =
        Tensor::<f32>::from_array(([1], length.into_raw_vec_and_offset().0.into_boxed_slice()))
            .unwrap();
    let noise_w_t =
        Tensor::<f32>::from_array(([1], noise_w.into_raw_vec_and_offset().0.into_boxed_slice()))
            .unwrap();

    let outputs = session
        .run(ort::inputs![
            input_t,
            input_len_t,
            noise_t,
            length_t,
            noise_w_t
        ])
        .map_err(|e| PiperError::InferenceError(format!("Inference failed: {}", e)))?;

    let (_, audio) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| PiperError::InferenceError(format!("Failed to extract output: {}", e)))?;

    Ok(audio.to_vec())
}
