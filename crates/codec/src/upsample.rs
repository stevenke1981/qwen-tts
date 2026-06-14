//! Upsample stage: 2× ConvNeXt-based upsampler for the Qwen3-TTS 12Hz codec decoder.
//!
//! Architecture (C-first throughout):
//! ```text
//! input [1024, T]
//!   → Block 0: CausalTransConv1d(k=2, s=2) → ConvNeXtBlock
//!   → Block 1: CausalTransConv1d(k=2, s=2) → ConvNeXtBlock
//!   → output [1024, T*4]
//! ```
//!
//! Each ConvNeXtBlock:
//! ```text
//!   depthwise causal Conv1d(k=7)
//!   → LayerNorm (affine, eps=1e-6)
//!   → pwconv1 Linear(1024 → 4096)
//!   → GELU (tanh approximation)
//!   → pwconv2 Linear(4096 → 1024)
//!   → LayerScale(gamma) per-channel multiply
//!   → + skip connection
//! ```

use crate::gguf::GgufFile;
use crate::tconv::conv_transpose1d_causal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NUM_BLOCKS: usize = 2;
const CHANNELS: usize = 1024;       // latent_dim
const DWDEPTH_KERNEL: usize = 7;   // depthwise conv kernel
const UPSAMPLE_RATIO: usize = 2;   // per block
const FFN_MULTIPLIER: usize = 4;   // 1024 → 4096

// ---------------------------------------------------------------------------
// Helper: GELU activation (tanh approximation)
// ---------------------------------------------------------------------------

/// GELU activation using the tanh approximation:
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
#[inline]
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56; // sqrt(2.0 / pi)
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

/// Apply GELU to a tensor `[C, T]` in-place.
fn gelu_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = gelu_tanh(*v);
    }
}

// ---------------------------------------------------------------------------
// Helper: depthwise causal Conv1d
// ---------------------------------------------------------------------------

/// Depthwise causal Conv1d: each channel has its own kernel.
///
/// # Arguments
/// - `input`: `[C, T]` C-first
/// - `c`: number of channels
/// - `t`: time steps
/// - `weight`: `[C, k]` in row-major
/// - `k`: kernel size
/// - `bias`: optional `[C]`
/// - `dilation`: dilation factor
///
/// # Returns
/// `[C, T]` C-first (same shape)
fn depthwise_conv1d_causal(
    input: &[f32],
    c: usize,
    t: usize,
    weight: &[f32],
    k: usize,
    bias: Option<&[f32]>,
    dilation: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; c * t];
    for ch in 0..c {
        let w_base = ch * k;
        let in_base = ch * t;
        let out_base = ch * t;
        for ot in 0..t {
            let it = ot; // stride = 1
            let mut sum = 0.0f32;
            for ki in 0..k {
                let ii = it as isize - ((k - 1 - ki) * dilation) as isize;
                if ii < 0 {
                    continue;
                }
                let ii = ii as usize;
                sum += weight[w_base + ki] * input[in_base + ii];
            }
            if let Some(b) = bias {
                sum += b[ch];
            }
            output[out_base + ot] = sum;
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Helper: LayerNorm (affine) over channel dimension
// ---------------------------------------------------------------------------

/// Apply LayerNorm over the channel dimension of a `[C, T]` tensor.
/// Normalizes each time step independently over all C channels.
fn layer_norm_cfirst(data: &mut [f32], c: usize, t: usize, weight: &[f32], bias: &[f32], eps: f32) {
    for ti in 0..t {
        // Compute mean and variance over channels at this time step
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        for ch in 0..c {
            let val = data[ch * t + ti];
            sum += val;
            sum_sq += val * val;
        }
        let mean = sum / c as f32;
        let var = sum_sq / c as f32 - mean * mean;
        let inv_std = 1.0 / (var + eps).sqrt();

        for ch in 0..c {
            let idx = ch * t + ti;
            data[idx] = (data[idx] - mean) * inv_std * weight[ch] + bias[ch];
        }
    }
}

// ---------------------------------------------------------------------------
// ConvNeXtBlock
// ---------------------------------------------------------------------------

/// One ConvNeXt block: dwconv → LayerNorm → pwconv1 → GELU → pwconv2 → gamma + skip.
pub(crate) struct ConvNeXtBlock {
    dwconv_weight: Vec<f32>,   // [C=1024, k=7]
    dwconv_bias: Vec<f32>,     // [C=1024]
    norm_weight: Vec<f32>,     // [C=1024]
    norm_bias: Vec<f32>,       // [C=1024]
    pwconv1_weight: Vec<f32>,  // [4*C=4096, C=1024]
    pwconv1_bias: Vec<f32>,    // [4*C=4096]
    pwconv2_weight: Vec<f32>,  // [C=1024, 4*C=4096]
    pwconv2_bias: Vec<f32>,    // [C=1024]
    gamma: Vec<f32>,           // [C=1024] per-channel LayerScale
    c: usize,                  // channels
}

impl ConvNeXtBlock {
    fn load_from_gguf(gguf: &mut GgufFile, block_idx: usize) -> Result<Self, String> {
        let prefix = format!("tok_dec.upsample.{block_idx}");
        let c = CHANNELS;
        let ffn = c * FFN_MULTIPLIER;

        // Depthwise conv: GGUF shape [k=7, C=1024, 1] = (k, C, 1)
        // Data layout: for grp 0..1, for ch 0..C, for ki 0..k
        // But since grp=1 (1 group): index = ch * k + ki → [C, k]
        let dwconv_weight = gguf.read_tensor_f32(&format!("{prefix}.dwconv.weight"))?;
        let dwconv_bias = gguf.read_tensor_f32(&format!("{prefix}.dwconv.bias"))?;
        assert_eq!(dwconv_bias.len(), c, "{prefix}.dwconv.bias size");
        // Weight expected: ne=[k=7, c=1024, 1] → total 7*1024 elements
        assert_eq!(dwconv_weight.len(), DWDEPTH_KERNEL * c, "{prefix}.dwconv.weight size");

        // LayerNorm
        let norm_weight = gguf.read_tensor_f32(&format!("{prefix}.norm.weight"))?;
        let norm_bias = gguf.read_tensor_f32(&format!("{prefix}.norm.bias"))?;
        assert_eq!(norm_weight.len(), c, "{prefix}.norm.weight size");
        assert_eq!(norm_bias.len(), c, "{prefix}.norm.bias size");

        // pwconv1: Linear(1024 → 4096). GGUF shape: [1024, 4096]
        // Our layout: weight [4096, 1024], bias [4096]
        let pwconv1_weight = gguf.read_tensor_f32(&format!("{prefix}.pwconv1.weight"))?;
        let pwconv1_bias = gguf.read_tensor_f32(&format!("{prefix}.pwconv1.bias"))?;
        assert_eq!(pwconv1_weight.len(), ffn * c, "{prefix}.pwconv1.weight size");
        assert_eq!(pwconv1_bias.len(), ffn, "{prefix}.pwconv1.bias size");

        // pwconv2: Linear(4096 → 1024). GGUF shape: [4096, 1024]
        // Our layout: weight [1024, 4096], bias [1024]
        let pwconv2_weight = gguf.read_tensor_f32(&format!("{prefix}.pwconv2.weight"))?;
        let pwconv2_bias = gguf.read_tensor_f32(&format!("{prefix}.pwconv2.bias"))?;
        assert_eq!(pwconv2_weight.len(), c * ffn, "{prefix}.pwconv2.weight size");
        assert_eq!(pwconv2_bias.len(), c, "{prefix}.pwconv2.bias size");

        // Gamma (LayerScale)
        let gamma = gguf.read_tensor_f32(&format!("{prefix}.gamma"))?;
        assert_eq!(gamma.len(), c, "{prefix}.gamma size");

        Ok(Self {
            dwconv_weight,
            dwconv_bias,
            norm_weight,
            norm_bias,
            pwconv1_weight,
            pwconv1_bias,
            pwconv2_weight,
            pwconv2_bias,
            gamma,
            c,
        })
    }

    /// Forward pass.
    ///
    /// Input:  `[C, T]` C-first
    /// Output: `[C, T]` C-first (same shape)
    fn forward(&self, input: &[f32], t: usize) -> Vec<f32> {
        let c = self.c;

        // 1. Depthwise causal Conv1d(k=7)
        let mut x = depthwise_conv1d_causal(
            input, c, t,
            &self.dwconv_weight, DWDEPTH_KERNEL,
            Some(&self.dwconv_bias),
            1,
        );

        // 2. LayerNorm over channel dim
        layer_norm_cfirst(&mut x, c, t, &self.norm_weight, &self.norm_bias, 1e-6);

        // 3. pwconv1: Linear(C → 4C) via matmul
        // weight [4C, C] · x [C, T] → [4C, T]
        let ffn = c * FFN_MULTIPLIER;
        let mut x_ffn = vec![0.0f32; ffn * t];
        let w1 = &self.pwconv1_weight;
        let b1 = &self.pwconv1_bias;
        for oc in 0..ffn {
            let out_base = oc * t;
            let w_base = oc * c;
            for ic in 0..c {
                let w_val = w1[w_base + ic];
                let in_base = ic * t;
                for ti in 0..t {
                    x_ffn[out_base + ti] += w_val * x[in_base + ti];
                }
            }
            // Add bias
            let bval = b1[oc];
            for ti in 0..t {
                x_ffn[out_base + ti] += bval;
            }
        }

        // 4. GELU
        gelu_inplace(&mut x_ffn);

        // 5. pwconv2: Linear(4C → C)
        // weight [C, 4C] · x_ffn [4C, T] → [C, T]
        x = vec![0.0f32; c * t];
        let w2 = &self.pwconv2_weight;
        let b2 = &self.pwconv2_bias;
        for oc in 0..c {
            let out_base = oc * t;
            let w_base = oc * ffn;
            for ic in 0..ffn {
                let w_val = w2[w_base + ic];
                let in_base = ic * t;
                for ti in 0..t {
                    x[out_base + ti] += w_val * x_ffn[in_base + ti];
                }
            }
            let bval = b2[oc];
            for ti in 0..t {
                x[out_base + ti] += bval;
            }
        }

        // 6. LayerScale: per-channel multiply
        for ch in 0..c {
            let g = self.gamma[ch];
            let base = ch * t;
            for ti in 0..t {
                x[base + ti] *= g;
            }
        }

        // 7. Skip connection
        for i in 0..c * t {
            x[i] += input[i];
        }

        x
    }
}

// ---------------------------------------------------------------------------
// UpsampleBlock
// ---------------------------------------------------------------------------

/// One upsample block: CausalTransConv1d(k=2, s=2) → ConvNeXtBlock.
pub(crate) struct UpsampleBlock {
    /// Transposed conv weight. GGUF shape: [K=2, OC=1024, IC=1024]
    /// Our layout: [IC=1024, OC=1024, K=2]
    conv_t_weight: Vec<f32>,
    /// Transposed conv bias [OC=1024]
    conv_t_bias: Vec<f32>,
    /// ConvNeXt block
    convnext: ConvNeXtBlock,
}

impl UpsampleBlock {
    fn load_from_gguf(gguf: &mut GgufFile, block_idx: usize) -> Result<Self, String> {
        let prefix = format!("tok_dec.upsample.{block_idx}");
        // Transposed conv: GGUF shape [K=2, OC=1024, IC=1024]
        let conv_t_weight = gguf.read_tensor_f32(&format!("{prefix}.conv.weight"))?;
        let conv_t_bias = gguf.read_tensor_f32(&format!("{prefix}.conv.bias"))?;
        assert_eq!(conv_t_bias.len(), CHANNELS, "{prefix}.conv.bias size");

        let convnext = ConvNeXtBlock::load_from_gguf(gguf, block_idx)?;

        Ok(Self {
            conv_t_weight,
            conv_t_bias,
            convnext,
        })
    }

    /// Forward pass.
    ///
    /// Input:  `[C, T]` C-first
    /// Output: `[C, T*2]` C-first
    fn forward(&self, input: &[f32], t: usize) -> Vec<f32> {
        // 1. CausalTransConv1d(k=2, s=2): [C, T] → [C, T*2]
        let x = conv_transpose1d_causal(
            input, CHANNELS, t,
            &self.conv_t_weight, CHANNELS, 2, UPSAMPLE_RATIO,
            Some(&self.conv_t_bias),
        );
        let t2 = t * UPSAMPLE_RATIO;

        // 2. ConvNeXtBlock: [C, T*2] → [C, T*2]
        self.convnext.forward(&x, t2)
    }
}

// ---------------------------------------------------------------------------
// Upsampler
// ---------------------------------------------------------------------------

/// 2-block upsampler: input [1024, T] → output [1024, T*4].
pub struct Upsampler {
    pub(crate) blocks: Vec<UpsampleBlock>,
    pub channels: usize,
}

impl Upsampler {
    /// Load all upsample weights from the GGUF file.
    pub fn from_gguf(gguf: &mut GgufFile) -> Result<Self, String> {
        let mut blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            blocks.push(UpsampleBlock::load_from_gguf(gguf, i)?);
        }
        Ok(Self {
            blocks,
            channels: CHANNELS,
        })
    }

    /// Full upsample forward pass.
    ///
    /// Input:  `[C, T]` C-first (C=1024)
    /// Output: `[C, T * 4]` C-first
    pub fn forward(&self, input: &[f32], t: usize) -> Vec<f32> {
        let mut x = input.to_vec();
        let mut t_cur = t;
        for block in &self.blocks {
            x = block.forward(&x, t_cur);
            t_cur *= UPSAMPLE_RATIO;
        }
        x
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use std::path::PathBuf;

    fn codec_gguf_path() -> PathBuf {
        let md = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let ws = md.parent().and_then(|p| p.parent()).unwrap();
        ws.join("models").join("qwen-tokenizer-12hz-Q8_0.gguf")
    }

    #[test]
    fn upsample_load_all_weights() {
        let path = codec_gguf_path();
        assert!(path.exists(), "GGUF not found");
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let up = Upsampler::from_gguf(&mut gguf).expect("load upsampler");

        assert_eq!(up.blocks.len(), NUM_BLOCKS);
        assert_eq!(up.channels, CHANNELS);

        // Verify each block's weight sizes
        for (i, block) in up.blocks.iter().enumerate() {
            // Transposed conv weight: [IC=1024, OC=1024, K=2]
            assert_eq!(block.conv_t_weight.len(), 1024 * 1024 * 2, "block {i} conv_t_weight");
            assert_eq!(block.conv_t_bias.len(), 1024, "block {i} conv_t_bias");
            // ConvNeXt
            assert_eq!(block.convnext.dwconv_weight.len(), DWDEPTH_KERNEL * CHANNELS);
            assert_eq!(block.convnext.dwconv_bias.len(), CHANNELS);
            assert_eq!(block.convnext.norm_weight.len(), CHANNELS);
            assert_eq!(block.convnext.norm_bias.len(), CHANNELS);
            assert_eq!(block.convnext.pwconv1_weight.len(), 4096 * CHANNELS);
            assert_eq!(block.convnext.pwconv1_bias.len(), 4096);
            assert_eq!(block.convnext.pwconv2_weight.len(), CHANNELS * 4096);
            assert_eq!(block.convnext.pwconv2_bias.len(), CHANNELS);
            assert_eq!(block.convnext.gamma.len(), CHANNELS);
        }

        println!("Upsampler loaded: {} blocks, {} channels", NUM_BLOCKS, CHANNELS);
    }

    #[test]
    fn upsample_single_block_shape() {
        let path = codec_gguf_path();
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let up = Upsampler::from_gguf(&mut gguf).expect("load upsampler");

        // Test block 0 forward: [1024, 4] → [1024, 8]
        let block0 = &up.blocks[0];
        let t_in = 4;
        let input = vec![0.01f32; CHANNELS * t_in];
        let out = block0.forward(&input, t_in);
        let expected_t = t_in * UPSAMPLE_RATIO;
        assert_eq!(out.len(), CHANNELS * expected_t, "block0 output shape");
        assert!(out.iter().all(|v| v.is_finite()), "block0 non-finite");
        println!("Block 0 forward [{}, {t_in}] → [{}, {expected_t}]: ok", CHANNELS, CHANNELS);
    }

    #[test]
    fn upsample_full_forward_shape() {
        let path = codec_gguf_path();
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let up = Upsampler::from_gguf(&mut gguf).expect("load upsampler");

        for t in [1, 2, 4] {
            let input = vec![0.005f32; CHANNELS * t];
            let output = up.forward(&input, t);
            let expected_t = t * 4;
            assert_eq!(
                output.len(),
                CHANNELS * expected_t,
                "t={t} output should be [{}, {expected_t}]",
                CHANNELS
            );
            assert!(output.iter().all(|v| v.is_finite()), "t={t} non-finite");
            let has_signal = output.iter().any(|&v| v.abs() > 1e-6);
            assert!(has_signal, "t={t} output appears all zero");
            println!("Upsample t={t}: [{}, {t}] → [{}, {expected_t}] mean={:.6}",
                CHANNELS, CHANNELS,
                output.iter().sum::<f32>() / output.len() as f32
            );
        }
    }

    #[test]
    fn gelu_tanh_basics() {
        assert!((gelu_tanh(0.0) - 0.0).abs() < 1e-6, "gelu(0) = 0");
        assert!((gelu_tanh(1.0) - 0.841_192).abs() < 1e-4, "gelu(1) ≈ 0.841");
        assert!((gelu_tanh(-1.0) - (-0.158_808)).abs() < 1e-4, "gelu(-1) ≈ -0.159");
    }

    #[test]
    fn depthwise_conv_simple() {
        let c = 2;
        let t = 4;
        // Input: ch0=[1, 2, 3, 4], ch1=[5, 6, 7, 8]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // Weight: ch0=[0.5, 1.0], ch1=[0.1, 0.2] (k=2)
        let weight = vec![0.5, 1.0, 0.1, 0.2];
        let bias = vec![0.0, 0.0];

        let out = depthwise_conv1d_causal(&input, c, t, &weight, 2, Some(&bias), 1);
        assert_eq!(out.len(), c * t);

        // Ch0: k=2 causal: out[t] = 0.5*in[t-1] + 1.0*in[t]
        // t=0: 0.5*0 + 1.0*1 = 1.0
        // t=1: 0.5*1 + 1.0*2 = 2.5
        // t=2: 0.5*2 + 1.0*3 = 4.0
        // t=3: 0.5*3 + 1.0*4 = 5.5
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.5).abs() < 1e-6);
        assert!((out[2] - 4.0).abs() < 1e-6);
        assert!((out[3] - 5.5).abs() < 1e-6);

        // Ch1: out[t] = 0.1*in[t-1] + 0.2*in[t]
        // t=0: 0.1*0 + 0.2*5 = 1.0
        // t=1: 0.1*5 + 0.2*6 = 1.7
        // t=2: 0.1*6 + 0.2*7 = 2.0
        // t=3: 0.1*7 + 0.2*8 = 2.3
        assert!((out[4] - 1.0).abs() < 1e-6);
        assert!((out[5] - 1.7).abs() < 1e-6);
        assert!((out[6] - 2.0).abs() < 1e-6);
        assert!((out[7] - 2.3).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_basic() {
        let c = 4;
        let t = 3;
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        // data shape is [4, 3]:
        // ch0: t0=1, t1=5, t2=9
        // ch1: t0=2, t1=6, t2=10
        // ch2: t0=3, t1=7, t2=11
        // ch3: t0=4, t1=8, t2=12

        let weight = vec![1.0; c];
        let bias = vec![0.0; c];

        layer_norm_cfirst(&mut data, c, t, &weight, &bias, 1e-6);

        // After LN: at t=0, values [1,2,3,4] with mean=2.5
        // var = (1.5^2 + 0.5^2 + 0.5^2 + 1.5^2)/4 = (2.25+0.25+0.25+2.25)/4 = 1.25
        // std = sqrt(1.25) ≈ 1.118
        // normalized: [-1.5, -0.5, 0.5, 1.5] / 1.118 = [-1.342, -0.447, 0.447, 1.342]
        let t0_expected = [-1.3416407, -0.4472136, 0.4472136, 1.3416407];
        for ch in 0..c {
            assert!((data[ch * t + 0] - t0_expected[ch]).abs() < 1e-5,
                "ch{ch} t0: got {}, expected {}", data[ch * t + 0], t0_expected[ch]);
        }
    }
}
