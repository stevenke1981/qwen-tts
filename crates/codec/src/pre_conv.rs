//! Pre-convolution layer: causal Conv1d (k=3, 512 → 1024).
//!
//! Maps the quantizer output [512, T] to transformer input [1024, T].
//! Weight is F16 (stored as Q8_0 in the GGUF, loaded as F32 at load time).

use crate::conv::conv1d_causal;
use crate::gguf::GgufFile;

/// Pre-convolution: causal Conv1d(k=3, stride=1, dilation=1, 512→1024).
pub struct PreConv {
    /// Weight tensor [1024, 512, 3] (C_out, C_in, k) in row-major.
    pub weight: Vec<f32>,
    /// Bias tensor [1024].
    pub bias: Vec<f32>,
}

impl PreConv {
    /// Load weights from the GGUF file.
    ///
    /// Tensor names:
    /// - `tok_dec.pre_conv.weight` — shape [3, 512, 1024] F16
    /// - `tok_dec.pre_conv.bias`   — shape [1024] F32
    pub fn from_gguf(gguf: &mut GgufFile) -> Result<Self, String> {
        let weight = gguf.read_tensor_f32("tok_dec.pre_conv.weight")?;
        let bias = gguf.read_tensor_f32("tok_dec.pre_conv.bias")?;

        // Verify shapes
        // GGUF shape: ne=[3, 512, 1024] = (k, C_in, C_out)
        // Our layout: [C_out=1024, C_in=512, k=3]
        assert_eq!(weight.len(), 3 * 512 * 1024, "pre_conv.weight size mismatch");
        assert_eq!(bias.len(), 1024, "pre_conv.bias size mismatch");

        Ok(Self { weight, bias })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `input` — [512, T] C-first
    /// * `t` — number of frames
    ///
    /// # Returns
    /// `[1024, T]` C-first
    pub fn forward(&self, input: &[f32], t: usize) -> Vec<f32> {
        conv1d_causal(input, 512, t, &self.weight, 1024, 3, Some(&self.bias), 1, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use std::path::PathBuf;

    fn codec_gguf_path() -> PathBuf {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .expect("crates/codec should be two levels below workspace");
        workspace.join("models").join("qwen-tokenizer-12hz-Q8_0.gguf")
    }

    #[test]
    fn pre_conv_load_and_forward() {
        let path = codec_gguf_path();
        assert!(path.exists(), "GGUF not found");
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let pre_conv = PreConv::from_gguf(&mut gguf).expect("load pre_conv");

        assert_eq!(pre_conv.weight.len(), 3 * 512 * 1024);
        assert_eq!(pre_conv.bias.len(), 1024);

        // Small test: 8 frames of dummy input [512, 8]
        let t = 8;
        let input = vec![0.1f32; 512 * t];
        let output = pre_conv.forward(&input, t);

        assert_eq!(output.len(), 1024 * t, "output size should be [1024, {t}]");

        // Check output is finite and non-zero
        assert!(output.iter().all(|v| v.is_finite()), "non-finite output");
        let has_signal = output.iter().any(|&v| v.abs() > 1e-4);
        assert!(has_signal, "output appears all zero");

        let mean = output.iter().sum::<f32>() / output.len() as f32;
        println!("pre_conv forward [512, {t}] → [1024, {t}]: mean={mean:.6}");
    }

    #[test]
    fn pre_conv_identity_bias() {
        // Short circuit: verify forward with constant input produces
        // correct output shape for various frame counts
        let path = codec_gguf_path();
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let pre_conv = PreConv::from_gguf(&mut gguf).expect("load pre_conv");

        for t in [1, 2, 5] {
            let input = vec![0.5f32; 512 * t];
            let output = pre_conv.forward(&input, t);
            assert_eq!(output.len(), 1024 * t, "t={t} failed shape check");
        }
    }
}
