//! Pipeline orchestration: autoregressive talker → code predictor → codec decoder.

use std::path::Path;

use candle_core::{Device, IndexOp, Tensor};
use qwen_tts_codec::CodecDecoder;
use qwen_tts_runtime::SynthesisRequest;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::code_predictor::CodePredictor;
use crate::sampling;
use crate::talker::Talker;
use crate::tokenizer::HfTokenizer;

/// Loaded pipeline ready for synthesis.
pub struct Pipeline {
    talker: Talker,
    code_predictor: CodePredictor,
    codec_decoder: CodecDecoder,
    tokenizer: Option<HfTokenizer>,
    device: Device,
}

impl Pipeline {
    /// Load all model weights, discover tokenizer alongside talker GGUF.
    pub fn new(talker_path: &Path, codec_path: &Path) -> anyhow::Result<Self> {
        let device = Device::Cpu;

        let (talker, code_predictor) = Talker::load_with_predictor(talker_path, &device)?;
        let codec_decoder = CodecDecoder::load(codec_path)
            .map_err(|e| anyhow::anyhow!("failed to load codec decoder: {e}"))?;

        let tokenizer = discover_tokenizer(talker_path).ok();

        log::info!(
            "pipeline loaded: talker={}, code_predictor={}, tokenizer={}",
            talker_path.display(),
            codec_path.display(),
            if tokenizer.is_some() { "found" } else { "not-found" },
        );

        Ok(Self { talker, code_predictor, codec_decoder, tokenizer, device })
    }

    /// Synthesize speech: text → WAV samples via autoregressive code generation.
    pub fn synthesize(&self, request: &SynthesisRequest) -> anyhow::Result<Vec<i16>> {
        let raw_text = &request.text;
        if raw_text.trim().is_empty() {
            anyhow::bail!("text cannot be empty");
        }

        // --- 1. Tokenize ---
        let input_ids = if let Some(ref tok) = self.tokenizer {
            let mut ids = tok.encode(raw_text)?;
            let bos = tok.bos_id();
            if bos != 0 && !ids.is_empty() && ids[0] != bos {
                ids.insert(0, bos);
            }
            ids
        } else {
            byte_tokenize(raw_text)
        };
        if input_ids.is_empty() {
            anyhow::bail!("tokenization produced empty input");
        }

        // Pad to minimum sequence length
        let min_seq: usize = 8;
        let pad_len = min_seq.saturating_sub(input_ids.len());
        let mut padded = vec![0u32; pad_len];
        padded.extend_from_slice(&input_ids);
        let seq_len = padded.len();
        let input_tensor = Tensor::from_slice(&padded, (1, seq_len), &self.device)?;

        // --- 2. Initial text embeddings ---
        let mut seq = self.talker.embed_text(&input_tensor)?;
        // seq: [1, seq_len, d_model]

        // --- 3. Sampling configuration ---
        let temperature = request.temperature.unwrap_or(1.0);
        let top_k = request.top_k.map(|v| v as usize);
        let top_p = request.top_p.map(|v| v as f32);
        let seed = request.seed.map(|s| s as u64);
        let mut rng: StdRng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let num_frames = request.max_new_tokens.unwrap_or(1024) as usize;
        let do_sample = request.do_sample.unwrap_or(true);

        // --- 4. Autoregressive frame generation ---
        let _num_acoustic = self.code_predictor.num_acoustic();
        let codec_codebooks: usize = 16;
        let mut all_codes: Vec<i32> = Vec::with_capacity(num_frames * codec_codebooks);

        for frame_idx in 0..num_frames {
            let current_len = seq.dims()[1];

            // 4a. Talker forward on current sequence
            let hidden = self.talker.forward_embeddings(&seq)?;
            // hidden: [1, current_len, d_model]
            let last_hidden = hidden.i((0, current_len - 1, ..))?.unsqueeze(0)?;
            // last_hidden: [1, 1, d_model]

            // 4b. Predict codebook 0 via codec_head
            let cb0_logits = self.talker.predict_codebook0(&last_hidden)?;
            // cb0_logits: [1, 1, vocab_cb0]
            let cb0_token = if do_sample {
                sample_logits_tensor(&cb0_logits, temperature, top_k, top_p, &mut rng)?
            } else {
                sampling::sample_argmax(&cb0_logits)?
            };
            all_codes.push(cb0_token as i32);

            // 4c. Embed codebook 0 token and append to sequence
            let cb0_emb = self.talker.embed_codebook0(cb0_token)?;
            // cb0_emb: [1, 1, d_model]
            seq = Tensor::cat(&[&seq, &cb0_emb], 1)?;

            // 4d. Predict acoustic codebooks 1..N via code predictor
            let frame = self
                .code_predictor
                .predict_one_frame_sampled(&last_hidden, temperature, top_k, top_p, &mut rng)?;
            for &token in &frame {
                all_codes.push(token as i32);
            }

            // Pad to full 16 codebooks if necessary
            while all_codes.len() < (frame_idx + 1) * codec_codebooks {
                all_codes.push(0);
            }

            // Optional: early stopping on EOS (codebook 0 = 0 is padding/silence)
            // For now we just generate all requested frames.
            _ = frame_idx;
        }

        // Trim to exact multiples
        let total_frames = all_codes.len() / codec_codebooks;
        all_codes.truncate(total_frames * codec_codebooks);

        // --- 5. Decode audio via codec ---
        let audio_f32 = self.codec_decoder.decode(&all_codes, total_frames);

        // --- 6. Convert f32 → i16 PCM ---
        let audio_i16: Vec<i16> = audio_f32
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();

        log::info!(
            "pure-rust synth: {} text tokens → {} code frames ({} code tokens) → {} audio samples ({}s)",
            seq_len,
            total_frames,
            all_codes.len(),
            audio_i16.len(),
            audio_i16.len() / 24000,
        );

        Ok(audio_i16)
    }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Sample from a `[batch, 1, vocab]` or `[batch, vocab]` logits tensor.
fn sample_logits_tensor(
    logits: &Tensor,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rng: &mut impl rand::Rng,
) -> anyhow::Result<u32> {
    // Flatten to 1D regardless of input rank
    let logits_1d: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    let (token, _prob) = sampling::sample_token(&logits_1d, temperature, top_k, top_p, rng);
    Ok(token)
}

/// Look for `tokenizer.json` in the same directory as the talker GGUF.
fn discover_tokenizer(talker_path: &Path) -> anyhow::Result<HfTokenizer> {
    let parent = talker_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("talker path has no parent"))?;

    for name in &["tokenizer.json", "tokenizer.json"] {
        let candidate = parent.join(name);
        if candidate.exists() {
            return HfTokenizer::from_file(&candidate);
        }
    }

    anyhow::bail!(
        "no tokenizer.json found next to {}",
        talker_path.display()
    );
}

/// Simple byte-level tokenizer (fallback when no BPE tokenizer available).
fn byte_tokenize(text: &str) -> Vec<u32> {
    text.bytes().map(|b| b as u32 + 1).collect()
}
