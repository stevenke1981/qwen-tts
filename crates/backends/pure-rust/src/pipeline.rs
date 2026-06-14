//! Pipeline orchestration: talker → code predictor → codec decoder.

use std::path::Path;

use candle_core::Device;
use candle_core::IndexOp;
use qwen_tts_codec::CodecDecoder;
use qwen_tts_runtime::SynthesisRequest;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::code_predictor::CodePredictor;
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
    ///
    /// `talker_path` — GGUF with both talker + code predictor weights.
    /// `codec_path`  — GGUF with DAC audio decoder weights.
    pub fn new(talker_path: &Path, codec_path: &Path) -> anyhow::Result<Self> {
        let device = Device::Cpu;

        // Talker + code predictor share the same GGUF
        let (talker, code_predictor) = Talker::load_with_predictor(talker_path, &device)?;

        // Codec decoder from separate GGUF
        let codec_decoder = CodecDecoder::load(codec_path)
            .map_err(|e| anyhow::anyhow!("failed to load codec decoder: {e}"))?;

        // Discover tokenizer.json alongside talker GGUF
        let tokenizer = discover_tokenizer(talker_path).ok();

        log::info!(
            "pipeline loaded: talker={}, code_predictor={}, tokenizer={}",
            talker_path.display(),
            codec_path.display(),
            if tokenizer.is_some() { "found" } else { "not-found" },
        );

        Ok(Self {
            talker,
            code_predictor,
            codec_decoder,
            tokenizer,
            device,
        })
    }

    /// Synthesize speech: text → WAV samples.
    pub fn synthesize(&self, request: &SynthesisRequest) -> anyhow::Result<Vec<i16>> {
        // 1. Encode text with BPE tokenizer (fallback: byte-level)
        let raw_text = &request.text;
        if raw_text.trim().is_empty() {
            anyhow::bail!("text cannot be empty");
        }

        // Build input IDs with BOS
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

        // Pad to minimum sequence length for the talker
        let min_seq: usize = 8;
        let pad_len = min_seq.saturating_sub(input_ids.len());
        let mut padded = vec![0u32; pad_len];
        padded.extend_from_slice(&input_ids);

        let seq_len = padded.len();
        let input_tensor =
            candle_core::Tensor::from_slice(&padded, (1, seq_len), &self.device)?;

        // 2. Talker forward: get hidden state at the last position
        let (_, hidden) = self.talker.forward_hidden(&input_tensor)?;
        let hidden = hidden.i((0, seq_len - 1, ..))?;
        let hidden = hidden.unsqueeze(0)?;

        // 3. Sampling configuration
        let temperature = request.temperature.unwrap_or(1.0);
        let top_k = request.top_k.map(|v| v as usize);
        let top_p = request.top_p.map(|v| v as f32);
        let seed = request.seed.map(|s| s as u64);
        let mut rng: StdRng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let num_frames = request.max_new_tokens.unwrap_or(1024) as usize;

        // 4. Code predictor: predict frames of acoustic codes with sampling
        let mut all_codes: Vec<i32> = Vec::new();
        let _num_acoustic = self.code_predictor.num_acoustic();

        for _frame_idx in 0..num_frames {
            // Predict one frame using the talker hidden state.
            // We pass the same hidden state for each frame (simplified).
            let frame = self
                .code_predictor
                .predict_one_frame_sampled(&hidden, temperature, top_k, top_p, &mut rng)?;

            // Codebook 0 (not predicted by code predictor) — use 0 placeholder
            all_codes.push(0);

            // Acoustic codebooks 1..N
            for &token in &frame {
                all_codes.push(token as i32);
            }
        }

        // Pad to match codec expectation: [num_frames, 16]
        let codec_codebooks: usize = 16;
        while all_codes.len() < num_frames * codec_codebooks {
            all_codes.push(0);
        }

        // 5. Decode audio via codec
        let audio_f32 = self.codec_decoder.decode(&all_codes, num_frames);

        // 6. Convert f32 [-1,1] → i16 PCM
        let audio_i16: Vec<i16> = audio_f32
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();

        log::info!(
            "pure-rust synth: {} text tokens → {} code frames → {} audio samples ({}s)",
            seq_len,
            num_frames,
            audio_i16.len(),
            audio_i16.len() / 24000,
        );

        Ok(audio_i16)
    }
}

// -----------------------------------------------------------------------
// Tokenizer discovery
// -----------------------------------------------------------------------

/// Look for `tokenizer.json` in the same directory as the talker GGUF.
fn discover_tokenizer(talker_path: &Path) -> anyhow::Result<HfTokenizer> {
    let parent = talker_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("talker path has no parent"))?;

    // Try common names
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
