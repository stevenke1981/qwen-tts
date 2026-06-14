//! Pipeline orchestration: ties the talker transformer, MTP code predictor heads,
//! and the qwen-tts-codec DAC decoder into a single `synthesize()` entry point.
//!
//! The pipeline:
//!   1. Tokenize input text (instruct)
//!   2. Talker forward pass → hidden states
//!   3. Code predictor → audio code frames
//!   4. Codec decode frames → WAV samples

use std::path::Path;

use candle_core::Device;
use qwen_tts_runtime::SynthesisRequest;
use rand::Rng;
use rand::SeedableRng;

use crate::code_predictor::CodePredictor;
use crate::sampling::sample_token;
use crate::talker::Talker;

/// Max token IDs to generate before producing audio codes.
const MAX_TEXT_TOKENS: usize = 256;

/// Pads the input sequence to at least this length for the talker.
const MIN_SEQ_LEN: usize = 8;

/// Loaded pipeline ready for synthesis.
pub struct Pipeline {
    talker: Talker,
    code_predictor: CodePredictor,
    /// Number of audio codebook levels.
    num_codebooks: usize,
    device: Device,
}

/// Synthesis configuration derived from [`SynthesisRequest`].
struct SynthConfig {
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    max_new_tokens: usize,
    seed: Option<u64>,
}

impl Pipeline {
    /// Load the talker GGUF and codec weights.
    ///
    /// `talker_path` — path to the talker Qwen2 GGUF file.
    /// `codec_path`  — path to a GGUF file containing the MTP code-predictor heads.
    pub fn new(talker_path: &Path, codec_path: &Path) -> anyhow::Result<Self> {
        let device = Device::Cpu;

        let talker = Talker::from_gguf(talker_path, &device)?;
        let code_predictor = CodePredictor::from_gguf(codec_path, &device)?;
        let num_codebooks = code_predictor.num_codebooks();

        Ok(Self {
            talker,
            code_predictor,
            num_codebooks,
            device,
        })
    }

    /// Full TTS synthesis: text → WAV samples.
    ///
    /// For the initial implementation we tokenize the instruct text, run it
    /// through the talker, predict code frames, and decode.
    pub fn synthesize(&self, request: &SynthesisRequest) -> anyhow::Result<Vec<i16>> {
        let cfg = self.build_config(request);
        let mut rng = match cfg.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        // For now, the tokenizer is a simple placeholder that uses a small
        // fixed lookup. Real implementation will load tokenizer.json.
        let instruct = request.instruct.as_deref().unwrap_or("");
        let mut input_ids = self.simple_tokenize(instruct);

        // Pad to minimum length
        if input_ids.len() < MIN_SEQ_LEN {
            let mut padded = vec![0u32; MIN_SEQ_LEN - input_ids.len()];
            padded.extend_from_slice(&input_ids);
            input_ids = padded;
        }

        // Run talker autoregressively
        // We use a simple approach: run the talker once, get hidden, predict codes
        let seq_len = input_ids.len();
        let input_tensor = candle_core::Tensor::from_slice(
            &input_ids,
            (1, seq_len),
            &self.device,
        )?;

        let logits = self.talker.forward(&input_tensor)?;
        // logits: [1, vocab_size]

        // Sample next token
        let logits_flat: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
        let (next_token, _prob) = sample_token(
            &logits_flat,
            cfg.temperature,
            cfg.top_k,
            cfg.top_p,
            &mut rng,
        );

        // TODO: full autoregressive generation
        // For now, use the talker's final hidden state directly as the
        // audio code prediction input (simplification — real impl will
        // accumulate hidden states across generated text tokens).
        let _ = next_token;

        // Get the talker's hidden state (last layer, last position).
        // In the current `forward()` API we only get logits, not hidden states.
        // For a proper implementation we need to modify the talker to return
        // hidden states as well.
        //
        // For now, we use logits → code predictor as a placeholder.
        // The actual code predictor takes hidden states, not logits.

        // Simulate code prediction: predict NUM_CODEBOOKS code tokens
        // per audio frame. For the initial implementation we predict
        // a single frame.
        let mut audio_codes = Vec::new();
        for _ in 0..cfg.max_new_tokens {
            let code_frame: Vec<u32> = (0..self.num_codebooks)
                .map(|_| rng.gen_range(0..1024))
                .collect();
            audio_codes.push(code_frame);
        }

        // Decode code frames to audio using qwen-tts-codec
        // For now, produce silence — the codec integration will come in a
        // follow-up change once the forward pass properly returns hidden states.
        let sample_rate = 44100u32;
        let num_samples = audio_codes.len() * 1024; // rough estimate: 1024 PCM frames per code step
        let silence = vec![0i16; num_samples];

        log::info!(
            "pure-rust synth: {} code frames → {} samples @ {}Hz",
            audio_codes.len(),
            num_samples,
            sample_rate,
        );

        Ok(silence)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn build_config(&self, request: &SynthesisRequest) -> SynthConfig {
        let temperature = request.temperature.unwrap_or(1.0) as f32;
        let top_k = request.top_k.map(|v| v as usize);
        let top_p = request.top_p.map(|v| v as f32);
        let max_new_tokens = request.max_new_tokens.unwrap_or(1024) as usize;
        SynthConfig {
            temperature,
            top_k,
            top_p,
            max_new_tokens,
            seed: request.seed.map(|s| s as u64),
        }
    }

    /// Simple fallback tokenizer (ASCII-by-character).
    fn simple_tokenize(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![0];
        }
        // Use byte-level fallback: each byte becomes a token ID
        text.bytes().map(|b| b as u32 + 1).collect()
    }
}
