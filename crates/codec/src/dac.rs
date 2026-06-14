//! DAC decoder: 4-block acoustic decoder for the qwen3-tts 12Hz tokenizer.
//!
//! Architecture (C-first throughout):
//! ```text
//! input [1024, T]
//!   → conv_pre (k=7, 1024→1536)
//!   → 4 blocks (SnakeBeta → TransConv(k=2*stride, stride) → 3×ResUnit(dil=1,3,9))
//!   → snake_post
//!   → conv_post (k=7, 96→1)
//!   → output [1, T * 1920]
//! ```
//!
//! Block dimensions:
//! | Block | In_C | Out_C | Stride | ConvT_K |
//! |-------|------|-------|--------|---------|
//! | dec.1 | 1536 | 768   | 8      | 16      |
//! | dec.2 | 768  | 384   | 5      | 10      |
//! | dec.3 | 384  | 192   | 4      | 8       |
//! | dec.4 | 192  | 96    | 3      | 6       |

use crate::conv::conv1d_causal;
use crate::gguf::GgufFile;
use crate::resunit::ResUnit;
use crate::snake::snake_beta_inplace;
use crate::tconv::conv_transpose1d_causal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NUM_BLOCKS: usize = 4;
const RES_UNITS_PER_BLOCK: usize = 3;

/// Strides for the 4 DAC blocks.
const BLOCK_STRIDES: [usize; NUM_BLOCKS] = [8, 5, 4, 3];

/// Channel dimensions: [C_in for block 0, C_in for block 1, ... , C_out for last block].
const BLOCK_CHANNELS: [usize; NUM_BLOCKS + 1] = [1536, 768, 384, 192, 96];

/// ResUnit dilations (same for all blocks).
const RES_DILATIONS: [usize; RES_UNITS_PER_BLOCK] = [1, 3, 9];

// ---------------------------------------------------------------------------
// DacBlock
// ---------------------------------------------------------------------------

/// One DAC block: SnakeBeta → CausalTransConv → 3× ResUnit.
pub struct DacBlock {
    pub snake_alpha: Vec<f32>,
    pub snake_beta: Vec<f32>,
    pub conv_t_weight: Vec<f32>,   // Transposed conv weight
    pub conv_t_bias: Vec<f32>,
    pub res_units: Vec<ResUnit>,
    pub in_ch: usize,
    pub out_ch: usize,
    pub stride: usize,
}

impl DacBlock {
    fn load_from_gguf(gguf: &mut GgufFile, block_idx: usize) -> Result<Self, String> {
        let py_idx = block_idx + 1; // Python ModuleList: dec.0=conv_pre, dec.{1..4}=blocks
        let prefix = format!("tok_dec.dec.{py_idx}");

        let stride = BLOCK_STRIDES[block_idx];
        let in_ch = BLOCK_CHANNELS[block_idx];
        let out_ch = BLOCK_CHANNELS[block_idx + 1];

        // SnakeBeta
        let snake_alpha = gguf.read_tensor_f32(&format!("{prefix}.snake.alpha"))?;
        let snake_beta = gguf.read_tensor_f32(&format!("{prefix}.snake.beta"))?;
        assert_eq!(snake_alpha.len(), in_ch);
        assert_eq!(snake_beta.len(), in_ch);

        // Transposed conv
        let conv_t_weight = gguf.read_tensor_f32(&format!("{prefix}.conv_t.weight"))?;
        let conv_t_bias = gguf.read_tensor_f32(&format!("{prefix}.conv_t.bias"))?;
        assert_eq!(conv_t_bias.len(), out_ch);

        // ResUnits
        let mut res_units = Vec::with_capacity(RES_UNITS_PER_BLOCK);
        for r in 0..RES_UNITS_PER_BLOCK {
            let rp = format!("{prefix}.res.{r}");
            let dilation = RES_DILATIONS[r];

            let act1_alpha = gguf.read_tensor_f32(&format!("{rp}.act1.alpha"))?;
            let act1_beta = gguf.read_tensor_f32(&format!("{rp}.act1.beta"))?;
            let conv1_weight = gguf.read_tensor_f32(&format!("{rp}.conv1.weight"))?;
            let conv1_bias = gguf.read_tensor_f32(&format!("{rp}.conv1.bias"))?;
            let act2_alpha = gguf.read_tensor_f32(&format!("{rp}.act2.alpha"))?;
            let act2_beta = gguf.read_tensor_f32(&format!("{rp}.act2.beta"))?;
            let conv2_weight = gguf.read_tensor_f32(&format!("{rp}.conv2.weight"))?;
            let conv2_bias = gguf.read_tensor_f32(&format!("{rp}.conv2.bias"))?;

            // Verify shapes
            assert_eq!(act1_alpha.len(), out_ch, "{rp}.act1.alpha size");
            assert_eq!(conv1_bias.len(), out_ch, "{rp}.conv1.bias size");
            assert_eq!(act2_alpha.len(), out_ch, "{rp}.act2.alpha size");
            assert_eq!(conv2_bias.len(), out_ch, "{rp}.conv2.bias size");

            res_units.push(ResUnit {
                conv1_weight,
                conv1_bias,
                conv2_weight,
                conv2_bias,
                snake_alpha: act1_alpha,
                snake_beta: act1_beta,
                snake2_alpha: act2_alpha,
                snake2_beta: act2_beta,
                dilation,
            });
        }

        Ok(Self {
            snake_alpha,
            snake_beta,
            conv_t_weight,
            conv_t_bias,
            res_units,
            in_ch,
            out_ch,
            stride,
        })
    }

    /// Forward pass.
    ///
    /// Input:  `[in_ch, T]` C-first
    /// Output: `[out_ch, T * stride]` C-first
    fn forward(&self, input: &[f32], t: usize) -> Vec<f32> {
        // 1. SnakeBeta
        let mut x = input.to_vec();
        snake_beta_inplace(&mut x, self.in_ch, t, &self.snake_alpha, &self.snake_beta);

        // 2. Causal Transposed Conv1d
        let kernel = 2 * self.stride;
        x = conv_transpose1d_causal(
            &x, self.in_ch, t,
            &self.conv_t_weight, self.out_ch, kernel, self.stride,
            Some(&self.conv_t_bias),
        );
        let t_after_conv = t * self.stride;

        // 3. 3× ResUnit
        for ru in &self.res_units {
            x = ru.forward(&x, self.out_ch, t_after_conv);
        }

        x
    }
}

// ---------------------------------------------------------------------------
// DacDecoder
// ---------------------------------------------------------------------------

/// Full DAC decoder: conv_pre → 4 blocks → snake_post → conv_post.
pub struct DacDecoder {
    pub conv_pre_weight: Vec<f32>,
    pub conv_pre_bias: Vec<f32>,
    pub blocks: Vec<DacBlock>,
    pub post_snake_alpha: Vec<f32>,
    pub post_snake_beta: Vec<f32>,
    pub conv_post_weight: Vec<f32>,
    pub conv_post_bias: Vec<f32>,
}

impl DacDecoder {
    /// Load all DAC weights from the GGUF file.
    pub fn from_gguf(gguf: &mut GgufFile) -> Result<Self, String> {
        // conv_pre (dec.0): Conv1d(k=7, 1024→1536)
        let conv_pre_weight = gguf.read_tensor_f32("tok_dec.dec.0.conv.weight")?;
        let conv_pre_bias = gguf.read_tensor_f32("tok_dec.dec.0.conv.bias")?;
        assert_eq!(conv_pre_bias.len(), 1536);

        // 4 blocks
        let mut blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            blocks.push(DacBlock::load_from_gguf(gguf, i)?);
        }

        // snake_post (dec.5)
        let post_snake_alpha = gguf.read_tensor_f32("tok_dec.dec.5.snake.alpha")?;
        let post_snake_beta = gguf.read_tensor_f32("tok_dec.dec.5.snake.beta")?;
        assert_eq!(post_snake_alpha.len(), 96);

        // conv_post (dec.6): Conv1d(k=7, 96→1)
        let conv_post_weight = gguf.read_tensor_f32("tok_dec.dec.6.conv.weight")?;
        let conv_post_bias = gguf.read_tensor_f32("tok_dec.dec.6.conv.bias")?;
        assert_eq!(conv_post_bias.len(), 1);

        Ok(Self {
            conv_pre_weight,
            conv_pre_bias,
            blocks,
            post_snake_alpha,
            post_snake_beta,
            conv_post_weight,
            conv_post_bias,
        })
    }

    /// Full DAC forward pass.
    ///
    /// Input:  `[1024, T]` C-first
    /// Output: `[1, T * 1920]` C-first (raw audio samples @ 24 kHz)
    ///
    /// The caller should clamp the output to [-1, 1].
    pub fn forward(&self, input: &[f32], t: usize) -> Vec<f32> {
        // 1. conv_pre: [1024, T] → [1536, T]
        let mut x = conv1d_causal(
            input, 1024, t,
            &self.conv_pre_weight, 1536, 7,
            Some(&self.conv_pre_bias),
            1, 1,
        );
        let mut t_cur = t;

        // 2. 4 DAC blocks
        for block in &self.blocks {
            x = block.forward(&x, t_cur);
            t_cur *= block.stride;
        }

        // 3. snake_post [96, T*1920]
        snake_beta_inplace(&mut x, 96, t_cur, &self.post_snake_alpha, &self.post_snake_beta);

        // 4. conv_post: [96, T*1920] → [1, T*1920]
        // Note: conv_post weight is stored as [7, 96] 2D in GGUF (OC=1 is implicit).
        // We pass c_out=1 to handle this correctly.
        let output = conv1d_causal(
            &x, 96, t_cur,
            &self.conv_post_weight, 1, 7,
            Some(&self.conv_post_bias),
            1, 1,
        );

        output
    }

    /// Total upsample factor (480x for T → T*1920).
    pub const fn total_factor() -> usize {
        8 * 5 * 4 * 3 // = 480
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
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
    fn dac_load_all_weights() {
        let path = codec_gguf_path();
        assert!(path.exists(), "GGUF not found");
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let dac = DacDecoder::from_gguf(&mut gguf).expect("load DAC decoder");

        // Verify counts
        assert_eq!(dac.blocks.len(), NUM_BLOCKS);

        // Check each block
        let expected_chs: [(usize, usize, usize); NUM_BLOCKS] = [
            (1536, 768, 8),
            (768, 384, 5),
            (384, 192, 4),
            (192, 96, 3),
        ];
        for (i, (in_ch, out_ch, stride)) in expected_chs.iter().enumerate() {
            let b = &dac.blocks[i];
            assert_eq!(b.in_ch, *in_ch, "block {i} in_ch");
            assert_eq!(b.out_ch, *out_ch, "block {i} out_ch");
            assert_eq!(b.stride, *stride, "block {i} stride");
            assert_eq!(b.res_units.len(), RES_UNITS_PER_BLOCK, "block {i} res units");
            assert_eq!(b.snake_alpha.len(), *in_ch, "block {i} snake alpha");

            // Verify resunit channel count
            for (r, ru) in b.res_units.iter().enumerate() {
                assert_eq!(ru.snake_alpha.len(), *out_ch, "block {i} res {r} alpha");
                assert_eq!(ru.conv1_bias.len(), *out_ch, "block {i} res {r} conv1_bias");
            }
        }

        // Post layers
        assert_eq!(dac.post_snake_alpha.len(), 96);
        assert_eq!(dac.conv_post_bias.len(), 1);

        println!("DAC decoder loaded successfully: {} blocks, conv_pre+post", NUM_BLOCKS);
    }

    #[test]
    fn dac_block_output_shape() {
        let path = codec_gguf_path();
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let dac = DacDecoder::from_gguf(&mut gguf).expect("load DAC decoder");

        // Test block 0 individually: [1536, 4] → [768, 32]
        let block0 = &dac.blocks[0];
        assert_eq!(block0.stride, 8);
        let t_in = 4;
        let input = vec![0.01f32; block0.in_ch * t_in];
        let out = block0.forward(&input, t_in);
        let expected_t = t_in * block0.stride;
        assert_eq!(
            out.len(),
            block0.out_ch * expected_t,
            "block0 output shape"
        );
        assert!(out.iter().all(|v| v.is_finite()), "block0 non-finite");

        println!(
            "Block 0 forward [1536, {t_in}] → [768, {expected_t}]: ok"
        );
    }

    #[test]
    fn dac_decoder_shapes() {
        let path = codec_gguf_path();
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let dac = DacDecoder::from_gguf(&mut gguf).expect("load DAC decoder");

        // Test forward with small input
        // Input: [1024, 2] → should produce [1, 2*8*5*4*3] = [1, 960]
        let t = 2;
        let input = vec![0.01f32; 1024 * t];
        let output = dac.forward(&input, t);

        let expected_samples = t * 8 * 5 * 4 * 3; // = 960
        assert_eq!(
            output.len(),
            1 * expected_samples,
            "DAC decoder output length should be [1, {expected_samples}]"
        );
        assert!(output.iter().all(|v| v.is_finite()), "non-finite output");

        // Check not all zeros
        let has_signal = output.iter().any(|&v| v.abs() > 1e-6);
        assert!(
            has_signal,
            "DAC output appears all zero — signal expected"
        );

        let mean = output.iter().sum::<f32>() / output.len() as f32;
        println!(
            "DAC forward [1024, {t}] → [1, {expected_samples}]: mean={mean:.6}, range=[{:.4}, {:.4}]",
            output.iter().cloned().fold(f32::NAN, f32::min),
            output.iter().cloned().fold(f32::NAN, f32::max),
        );
    }

    #[test]
    fn dac_decoder_multi_frame() {
        let path = codec_gguf_path();
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let dac = DacDecoder::from_gguf(&mut gguf).expect("load DAC decoder");

        for t in [1, 3, 5] {
            let input = vec![0.005f32; 1024 * t];
            let output = dac.forward(&input, t);
            let expected = t * 8 * 5 * 4 * 3;
            assert_eq!(output.len(), expected, "t={t} output length should be {expected}");
            assert!(output.iter().all(|v| v.is_finite()), "t={t} non-finite");
            println!("DAC t={t}: output length {} (expected {expected})", output.len());
        }
    }
}
