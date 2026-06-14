//! Full codec decoder pipeline: codes → quantizer → pre_conv → transformer → upsample → DAC → audio.
//!
//! Layout flow (C-first throughout):
//! ```text
//! codes [T, 16] i32
//!   → quantizer.decode → [512, T]
//!   → pre_conv.forward → [1024, T]
//!   → transformer.forward → [1024, T]
//!   → upsample.forward → [1024, T*4]
//!   → dac.forward → [1, T*1920]
//!   → clamp(-1, 1)
//!   → audio [T*1920] f32  mono 24 kHz
//! ```

use crate::dac::DacDecoder;
use crate::gguf::GgufFile;
use crate::pre_conv::PreConv;
use crate::pre_transformer::PreTransformer;
use crate::quantizer::QuantizerDecoder;
use crate::upsample::Upsampler;
use std::path::Path;

/// Total hop length: 1920 samples per frame.
pub const HOP_LENGTH: usize = 1920;

/// Number of codebooks per frame.
pub const NUM_CODEBOOKS: usize = 16;

/// Sample rate of the output audio.
pub const SAMPLE_RATE: u32 = 24000;

/// Complete codec decoder, combining all modules.
pub struct CodecDecoder {
    pub quantizer: QuantizerDecoder,
    pub pre_conv: PreConv,
    pub transformer: PreTransformer,
    pub upsample: Upsampler,
    pub dac: DacDecoder,
}

impl CodecDecoder {
    /// Load all weights from a codec GGUF file.
    ///
    /// # Errors
    ///
    /// Returns an error if the GGUF cannot be opened or any module fails to load.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mut gguf = GgufFile::open(path)?;

        // Load modules in pipeline order
        let quantizer = QuantizerDecoder::from_gguf(&mut gguf)?;
        let pre_conv = PreConv::from_gguf(&mut gguf)?;
        let transformer = PreTransformer::from_gguf(&mut gguf)?;
        let upsample = Upsampler::from_gguf(&mut gguf)?;
        let dac = DacDecoder::from_gguf(&mut gguf)?;

        Ok(Self {
            quantizer,
            pre_conv,
            transformer,
            upsample,
            dac,
        })
    }

    /// Full decode: codes → audio samples.
    ///
    /// # Arguments
    ///
    /// * `codes` — flat i32 buffer, `[T, 16]` row-major (T frames × 16 codebooks)
    /// * `num_frames` — number of frames T
    ///
    /// # Returns
    ///
    /// Audio samples as `[T * 1920]` f32, clamped to [-1, 1].
    pub fn decode(&self, codes: &[i32], num_frames: usize) -> Vec<f32> {
        assert_eq!(codes.len(), num_frames * NUM_CODEBOOKS, "codes buffer size mismatch");

        // 1. Quantizer decode: [T, 16] → [512, T]
        let x = self.quantizer.decode(codes, num_frames);

        // 2. pre_conv: [512, T] → [1024, T]
        let x = self.pre_conv.forward(&x, num_frames);

        // 3. transformer: [1024, T] → [1024, T]
        let x = self.transformer.forward(&x, num_frames);

        // 4. upsample: [1024, T] → [1024, T*4]
        let x = self.upsample.forward(&x, num_frames);

        // 5. DAC decoder: [1024, T*4] → [1, T*1920]
        let t_upsampled = num_frames * 4;
        let x = self.dac.forward(&x, t_upsampled);

        // 6. Clamp to [-1, 1] and convert to 1D audio
        let total_samples = num_frames * HOP_LENGTH;
        let mut audio = Vec::with_capacity(total_samples);
        for i in 0..total_samples {
            let sample = x[i].clamp(-1.0, 1.0);
            audio.push(sample);
        }

        audio
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
        let md = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let ws = md.parent().and_then(|p| p.parent()).unwrap();
        ws.join("models").join("qwen-tokenizer-12hz-Q8_0.gguf")
    }

    #[test]
    fn codec_decoder_load_all_modules() {
        let path = codec_gguf_path();
        assert!(path.exists(), "GGUF not found");
        let decoder = CodecDecoder::load(&path).expect("load CodecDecoder");

        // Verify all modules are loaded
        assert!(decoder.quantizer.codebooks.len() == 16);
        assert_eq!(decoder.pre_conv.weight.len(), 3 * 512 * 1024);
        assert_eq!(decoder.transformer.layers.len(), 8);
        assert_eq!(decoder.upsample.blocks.len(), 2);
        assert_eq!(decoder.dac.blocks.len(), 4);

        println!("CodecDecoder loaded: all 5 modules ready");
    }

    #[test]
    fn codec_decode_synthetic_codes() {
        let path = codec_gguf_path();
        let decoder = CodecDecoder::load(&path).expect("load CodecDecoder");

        // Synthetic codes: 2 frames
        let num_frames = 2;
        let codes: Vec<i32> = (0..num_frames * NUM_CODEBOOKS)
            .map(|i| ((i * 7 + 13) % 2048) as i32)
            .collect();

        let audio = decoder.decode(&codes, num_frames);
        let expected_samples = num_frames * HOP_LENGTH;

        assert_eq!(audio.len(), expected_samples, "audio length should be {expected_samples}");

        // All finite and in range [-1, 1]
        assert!(audio.iter().all(|&s| s.is_finite()), "non-finite sample");
        assert!(audio.iter().all(|&s| s >= -1.0 && s <= 1.0), "sample out of range");

        // Non-zero audio (should have signal)
        let has_signal = audio.iter().any(|&s| s.abs() > 1e-4);
        assert!(has_signal, "audio is all zero — decoder may not be working");

        let mean = audio.iter().sum::<f32>() / audio.len() as f32;
        println!("Decoded {num_frames} frames → {} samples @ {} Hz: mean={mean:.6}",
            audio.len(), SAMPLE_RATE);
    }

    #[test]
    fn codec_decode_multiple_frames() {
        let path = codec_gguf_path();
        let decoder = CodecDecoder::load(&path).expect("load CodecDecoder");

        for num_frames in [1, 3, 5] {
            let codes: Vec<i32> = (0..num_frames * NUM_CODEBOOKS)
                .map(|i| ((i * 31 + 7) % 2048) as i32)
                .collect();

            let audio = decoder.decode(&codes, num_frames);
            let expected = num_frames * HOP_LENGTH;
            assert_eq!(audio.len(), expected, "t={num_frames} audio length");
            assert!(audio.iter().all(|&s| s.is_finite()), "t={num_frames} non-finite");
            assert!(audio.iter().all(|&s| s >= -1.0 && s <= 1.0), "t={num_frames} out of range");

            // Check that different codes produce different audio
            if num_frames > 1 {
                let first_half: f32 = audio[..expected / 2].iter().map(|&x| x.abs()).sum();
                let second_half: f32 = audio[expected / 2..].iter().map(|&x| x.abs()).sum();
                // They should not be identical (different frames produce different signal)
                let ratio = (first_half - second_half).abs() / (first_half + second_half + 1e-8);
                println!("t={num_frames}: first_half={first_half:.4}, second_half={second_half:.4}, ratio={ratio:.4}");
            }

            println!("CodecDecoder t={num_frames}: audio length {expected}, range=[{:.4}, {:.4}]",
                audio.iter().cloned().fold(f32::NAN, f32::min),
                audio.iter().cloned().fold(f32::NAN, f32::max),
            );
        }
    }

    #[test]
    fn codec_decode_deterministic() {
        let path = codec_gguf_path();
        let decoder = CodecDecoder::load(&path).expect("load CodecDecoder");

        let codes: Vec<i32> = (0..3 * NUM_CODEBOOKS)
            .map(|i| ((i * 131) % 2048) as i32)
            .collect();

        // Decode twice
        let audio1 = decoder.decode(&codes, 3);
        let audio2 = decoder.decode(&codes, 3);

        assert_eq!(audio1.len(), audio2.len());
        for (i, (a, b)) in audio1.iter().zip(audio2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "sample {i} differs: {a} vs {b}");
        }

        println!("Deterministic check: {} samples identical", audio1.len());
    }
}
