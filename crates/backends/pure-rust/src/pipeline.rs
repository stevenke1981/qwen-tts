//! Pipeline orchestration: talker → code predictor → codec decoder.

use std::path::Path;

use candle_core::Device;
use candle_core::IndexOp;
use qwen_tts_codec::CodecDecoder;
use qwen_tts_runtime::SynthesisRequest;

use crate::code_predictor::CodePredictor;
use crate::talker::Talker;

/// Loaded pipeline ready for synthesis.
pub struct Pipeline {
    talker: Talker,
    code_predictor: CodePredictor,
    codec_decoder: CodecDecoder,
    device: Device,
}

#[allow(dead_code)]
struct SynthConfig {
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    max_new_tokens: usize,
    seed: Option<u64>,
}

impl Pipeline {
    /// Load all model weights.
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

        Ok(Self {
            talker,
            code_predictor,
            codec_decoder,
            device,
        })
    }

    /// Synthesize speech: text → WAV samples.
    pub fn synthesize(&self, request: &SynthesisRequest) -> anyhow::Result<Vec<i16>> {
        let cfg = self.build_config(request);

        // 1. Tokenize instruct text (simple fallback — real tokenizer later)
        let instruct = request.instruct.as_deref().unwrap_or("");
        let input_ids = self.simple_tokenize(instruct);
        if input_ids.is_empty() {
            anyhow::bail!("empty instruct text");
        }

        // Pad to minimum sequence length for the talker
        let min_seq: usize = 8;
        let pad_len = min_seq.saturating_sub(input_ids.len());
        let mut padded = vec![0u32; pad_len];
        padded.extend_from_slice(&input_ids);

        let seq_len = padded.len();
        let input_tensor = candle_core::Tensor::from_slice(&padded, (1, seq_len), &self.device)?;

        // 2. Talker forward: get hidden state at the last position
        let (_, hidden) = self.talker.forward_hidden(&input_tensor)?;
        // hidden: [batch=1, seq_len, d_model]
        // Take the last position
        let hidden = hidden.i((0, seq_len - 1, ..))?;
        // hidden: [d_model]
        let hidden = hidden.unsqueeze(0)?;
        // hidden: [1, d_model]

        // 3. Code predictor: predict frames of acoustic codes
        let num_frames = cfg.max_new_tokens;
        let mut all_codes: Vec<i32> = Vec::new();
        let _num_acoustic = self.code_predictor.num_acoustic();

        for _frame_idx in 0..num_frames {
            let frames = self.code_predictor.predict_one_frame(&hidden)?;
            // frames[0] has num_acoustic code tokens
            if let Some(frame) = frames.first() {
                // Codebook 0 (predicted by talker) — for now use 0
                all_codes.push(0);
                // Codebooks 1..N
                for &token in frame {
                    all_codes.push(token as i32);
                }
            }
        }

        // Pad/truncate to match codec expectation: [num_frames, 16]
        let codec_codebooks: usize = 16;
        while all_codes.len() < num_frames * codec_codebooks {
            all_codes.push(0);
        }

        // 4. Decode audio via codec
        let audio_f32 = self.codec_decoder.decode(&all_codes, num_frames);

        // 5. Convert f32 [-1,1] → i16 PCM
        let audio_i16: Vec<i16> = audio_f32
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();

        log::info!(
            "pure-rust synth: {} code frames → {} audio samples",
            num_frames,
            audio_i16.len(),
        );

        Ok(audio_i16)
    }

    fn build_config(&self, request: &SynthesisRequest) -> SynthConfig {
        SynthConfig {
            temperature: request.temperature.unwrap_or(1.0),
            top_k: request.top_k.map(|v| v as usize),
            top_p: request.top_p.map(|v| v as f32),
            max_new_tokens: request.max_new_tokens.unwrap_or(1024) as usize,
            seed: request.seed.map(|s| s as u64),
        }
    }

    /// Simple fallback tokenizer (byte-level).
    fn simple_tokenize(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }
        text.bytes().map(|b| b as u32 + 1).collect()
    }
}
