//! Audio code token predictor (MTP heads) for Qwen3-TTS.
//!
//! The code predictor takes the final hidden state from the talker transformer
//! and produces logits over audio code tokens for each codebook level.
//!
//! Architecture: per-level linear head (with optional layer-norm). Each head
//! predicts one code token per time-step.

use std::path::Path;

use candle_core::{Device, IndexOp, Module, Tensor, D};
use candle_nn::RmsNorm;

use crate::config::ModelConfig;

/// Type alias: a list of code token IDs (one per codebook level).
pub type CodeFrame = Vec<u32>;

/// Number of audio codebook levels in Qwen3-TTS (DAC 44kHz).
pub const NUM_CODEBOOKS: usize = 4;

/// Code predictor: linear heads per codebook level, loaded from a GGUF file.
pub struct CodePredictor {
    config: ModelConfig,
    heads: Vec<LinearHead>,
    device: Device,
}

struct LinearHead {
    norm: Option<RmsNorm>,
    weight: Tensor,
    bias: Option<Tensor>,
}

impl CodePredictor {
    /// Load code predictor from a dedicated GGUF file.
    ///
    /// The GGUF is expected to contain tensors named:
    /// - `code_predictor.{lvl}.weight`
    /// - `code_predictor.{lvl}.bias` (optional)
    /// - `code_predictor.{lvl}.norm.weight` (optional)
    pub fn from_gguf(path: &Path, device: &Device) -> anyhow::Result<Self> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("failed to open codec GGUF {path:?}: {e}"))?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("bad GGUF header: {e}"))?;

        let cfg = ModelConfig::from_gguf(&content.metadata);

        let mut load = |name: &str| -> anyhow::Result<Tensor> {
            let qt = content
                .tensor(&mut file, name, device)
                .map_err(|e| anyhow::anyhow!("missing tensor {name}: {e}"))?;
            qt.dequantize(device)
                .map_err(|e| anyhow::anyhow!("dequantize {name}: {e}"))
        };

        let mut heads = Vec::with_capacity(NUM_CODEBOOKS);
        for lvl in 0..NUM_CODEBOOKS {
            let weight = load(&format!("code_predictor.{lvl}.weight"))?;

            let bias = if content.tensor_infos.contains_key(&format!("code_predictor.{lvl}.bias")) {
                Some(load(&format!("code_predictor.{lvl}.bias"))?)
            } else {
                None
            };

            let norm = if content.tensor_infos.contains_key(&format!("code_predictor.{lvl}.norm.weight")) {
                let w = load(&format!("code_predictor.{lvl}.norm.weight"))?;
                Some(RmsNorm::new(w, cfg.norm_eps))
            } else {
                None
            };

            heads.push(LinearHead { norm, weight, bias });
        }

        Ok(Self {
            config: cfg,
            heads,
            device: device.clone(),
        })
    }

    /// Predict code tokens from talker hidden states.
    ///
    /// `hidden`: talker output hidden state `[batch, d_model]`.
    ///
    /// Returns a `Vec<CodeFrame>`: for each item in the batch, a frame of
    /// `NUM_CODEBOOKS` code token IDs.
    pub fn predict_codes(&self, hidden: &Tensor) -> anyhow::Result<Vec<CodeFrame>> {
        let batch = hidden.dims()[0];
        let mut out_frames = vec![CodeFrame::with_capacity(NUM_CODEBOOKS); batch];

        for (lvl, head) in self.heads.iter().enumerate() {
            let mut x = hidden.clone();

            // Optional pre-norm
            if let Some(ref norm) = head.norm {
                x = norm.forward(&x)?;
            }

            // Linear projection
            let mut logits = head.weight.matmul(&x.t()?)?.t()?;
            if let Some(ref bias) = head.bias {
                logits = (logits + bias)?;
            }

            // Argmax per item
            for b in 0..batch {
                let row = logits.i(b)?;
                let idx = row.argmax(D::Minus1)?;
                let token: u32 = idx.flatten_all()?.to_vec0()?;
                out_frames[b].push(token);
            }
        }

        Ok(out_frames)
    }

    /// Return the number of codebook levels.
    #[must_use]
    pub fn num_codebooks(&self) -> usize {
        self.heads.len()
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Basic test with fake loaded weights (empty predictor = 4 heads at size 0).
    #[test]
    fn test_code_predictor_structure() {
        let dev = Device::Cpu;
        // We can't test from_gguf without a real file, but verify NUM_CODEBOOKS
        assert_eq!(NUM_CODEBOOKS, 4);
        let _ = dev;
    }

    #[test]
    fn test_code_frame_type() {
        let mut frame = CodeFrame::with_capacity(NUM_CODEBOOKS);
        frame.push(0);
        frame.push(1);
        frame.push(2);
        frame.push(3);
        assert_eq!(frame.len(), NUM_CODEBOOKS);
    }
}
