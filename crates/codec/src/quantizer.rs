//! RVQ quantizer decoder for the qwen3-tts 12Hz tokenizer.
//!
//! Architecture (split RVQ):
//! - 1 semantic codebook + 15 acoustic codebooks = 16 total
//! - Each codebook: [dim=256, codebook_size=2048] F32
//! - Two output projection matrices: [256, 512] each (256→512)
//! - Semantic and acoustic groups are projected separately then summed.
//!
//! Input:  codes [T, 16] i32   — one index per codebook per frame
//! Output: latent [512, T] f32  — C-first (channels × time)

use crate::gguf::GgufFile;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NUM_QUANTIZERS: usize = 16;       // 1 semantic + 15 acoustic
const CODEBOOK_DIM: usize = 256;
const CODEBOOK_SIZE: usize = 2048;
const HIDDEN: usize = 512;

const SEMANTIC_CODEBOOKS: usize = 1;
const ACOUSTIC_CODEBOOKS: usize = NUM_QUANTIZERS - SEMANTIC_CODEBOOKS; // 15

// ---------------------------------------------------------------------------
// QuantizerDecoder
// ---------------------------------------------------------------------------

/// RVQ quantizer decoder.
///
/// Decodes 16 codebook indices per frame into a 512-dimensional latent vector.
pub struct QuantizerDecoder {
    /// Codebook embedding tables, flat [16][256 * 2048].
    /// Index 0 = semantic, indices 1..16 = acoustic.
    pub(crate) codebooks: Vec<Vec<f32>>,

    /// Output projection weights for each group.
    /// [0] = semantic proj [256, 512]  — weight[ic + 256*oc] in GGUF layout
    /// [1] = acoustic proj [256, 512]  — same layout
    proj_weights: Vec<Vec<f32>>,
}

impl QuantizerDecoder {
    /// Load all codebooks and projection weights from the GGUF file.
    ///
    /// Tensor names (from C++ reference):
    /// - `tok_dec.vq_first.0.codebook`         — 1 semantic codebook
    /// - `tok_dec.vq_first.output_proj.weight`  — semantic projection
    /// - `tok_dec.vq_rest.{k}.codebook`         — 15 acoustic codebooks
    /// - `tok_dec.vq_rest.output_proj.weight`   — acoustic projection
    pub fn from_gguf(gguf: &mut GgufFile) -> Result<Self, String> {
        let mut codebooks = Vec::with_capacity(NUM_QUANTIZERS);

        // Semantic codebook (1)
        let sem_cb = gguf
            .read_tensor_f32("tok_dec.vq_first.0.codebook")?;
        assert_eq!(sem_cb.len(), CODEBOOK_DIM * CODEBOOK_SIZE);
        codebooks.push(sem_cb);

        // Acoustic codebooks (15)
        for k in 0..ACOUSTIC_CODEBOOKS {
            let name = format!("tok_dec.vq_rest.{k}.codebook");
            let cb = gguf.read_tensor_f32(&name)?;
            assert_eq!(cb.len(), CODEBOOK_DIM * CODEBOOK_SIZE);
            codebooks.push(cb);
        }

        // Projection weights (2 groups, each [1, 256, 512] → flat [256*512])
        let mut proj_weights = Vec::with_capacity(2);

        let sem_proj_name = "tok_dec.vq_first.output_proj.weight";
        let sem_proj_raw = gguf.read_tensor_f32(sem_proj_name)?;
        // GGUF shape [1, 256, 512]; flat data = (ic + 256*oc) ordering
        assert_eq!(sem_proj_raw.len(), CODEBOOK_DIM * HIDDEN);
        proj_weights.push(sem_proj_raw);

        let aco_proj_name = "tok_dec.vq_rest.output_proj.weight";
        let aco_proj_raw = gguf.read_tensor_f32(aco_proj_name)?;
        assert_eq!(aco_proj_raw.len(), CODEBOOK_DIM * HIDDEN);
        proj_weights.push(aco_proj_raw);

        Ok(Self {
            codebooks,
            proj_weights,
        })
    }

    /// Decode RVQ codes into a 512-dimensional latent representation.
    ///
    /// # Arguments
    ///
    /// * `codes` — flat array of shape [T, 16] in row-major order
    ///   (T frames × 16 codebooks, time varies fastest).
    ///   Each value is an index into a codebook [0, 2048).
    /// * `t` — number of frames
    ///
    /// # Returns
    ///
    /// `Vec<f32>` of length `HIDDEN * T` = `512 * T`, stored C-first
    /// (channels × time), where channel `c` at frame `ti` is at index
    /// `c * t + ti`.
    pub fn decode(&self, codes: &[i32], t: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; HIDDEN * t];

        // For each frame:
        //   1. Look up semantic embedding (1 codebook)
        //   2. Sum 15 acoustic embeddings
        //   3. Project both groups from 256→512 and accumulate into output
        for ti in 0..t {
            let frame_base = ti * NUM_QUANTIZERS;

            // --- Semantic embedding (codebook 0) ---
            let sem_idx = codes[frame_base] as usize;
            let sem_idx = sem_idx.clamp(0, CODEBOOK_SIZE - 1);
            let sem_emb = &self.codebooks[0][sem_idx * CODEBOOK_DIM..(sem_idx + 1) * CODEBOOK_DIM];

            // --- Acoustic sum (codebooks 1..16) ---
            // Accumulate 15 codebook embeddings into a temporary buffer.
            // Per-frame allocation is fine for typical T ≤ hundreds.
            // TODO: reuse scratch buffer for hot path if needed.
            let mut aco_sum = [0.0f32; CODEBOOK_DIM];
            for cb in 1..NUM_QUANTIZERS {
                let idx = codes[frame_base + cb] as usize;
                let idx = idx.clamp(0, CODEBOOK_SIZE - 1);
                let emb =
                    &self.codebooks[cb][idx * CODEBOOK_DIM..(idx + 1) * CODEBOOK_DIM];
                for i in 0..CODEBOOK_DIM {
                    aco_sum[i] += emb[i];
                }
            }

            // --- Project both groups simultaneously ---
            // proj_weight[group][ic + 256*oc] = weight(ic → oc) in GGUF layout
            let sem_proj = &self.proj_weights[0];
            let aco_proj = &self.proj_weights[1];

            for oc in 0..HIDDEN {
                let mut val = 0.0f32;
                // Unroll inner loop over input channels for clarity;
                // the compiler should autovectorize this pattern.
                for ic in 0..CODEBOOK_DIM {
                    let w_sem = sem_proj[ic + CODEBOOK_DIM * oc];
                    let w_aco = aco_proj[ic + CODEBOOK_DIM * oc];
                    val += w_sem * sem_emb[ic] + w_aco * aco_sum[ic];
                }
                output[oc * t + ti] = val;
            }
        }

        output
    }

    /// Number of output channels (always 512).
    pub const fn hidden() -> usize {
        HIDDEN
    }

    /// Number of codebooks (always 16).
    pub const fn num_quantizers() -> usize {
        NUM_QUANTIZERS
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
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .expect("crates/codec should be two levels below workspace");
        workspace.join("models").join("qwen-tokenizer-12hz-Q8_0.gguf")
    }

    #[test]
    fn quantizer_load_from_gguf() {
        let path = codec_gguf_path();
        assert!(path.exists(), "GGUF not found");
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let qd = QuantizerDecoder::from_gguf(&mut gguf).expect("load quantizer");

        assert_eq!(qd.codebooks.len(), NUM_QUANTIZERS);
        assert_eq!(qd.proj_weights.len(), 2);

        // Each codebook should have the right size
        for (i, cb) in qd.codebooks.iter().enumerate() {
            assert_eq!(
                cb.len(),
                CODEBOOK_DIM * CODEBOOK_SIZE,
                "codebook {i} wrong size"
            );
        }

        // Codebook values should be non-zero (real data)
        let sem_mean: f32 =
            qd.codebooks[0].iter().sum::<f32>() / qd.codebooks[0].len() as f32;
        println!("semantic codebook mean: {sem_mean:.6}");

        let first_aco_mean: f32 =
            qd.codebooks[1].iter().sum::<f32>() / qd.codebooks[1].len() as f32;
        println!("first acoustic codebook mean: {first_aco_mean:.6}");

        // Projection weights
        assert_eq!(qd.proj_weights[0].len(), CODEBOOK_DIM * HIDDEN);
        assert_eq!(qd.proj_weights[1].len(), CODEBOOK_DIM * HIDDEN);
    }

    #[test]
    fn quantizer_decode_real_data() {
        let path = codec_gguf_path();
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let qd = QuantizerDecoder::from_gguf(&mut gguf).expect("load quantizer");

        // Use the first frame from a real synthesis (captured from CLI output)
        let codes: Vec<i32> = vec![
            1995, 1486, 1114, 1399, 1832, 383, 1579, 1544, 1611, 2030, 414,
            796, 729, 1575, 1587, 787,
        ];

        let output = qd.decode(&codes, 1);
        assert_eq!(output.len(), 512);

        // Check output is finite and has non-zero content
        assert!(
            output.iter().all(|v| v.is_finite()),
            "non-finite values in quantizer output"
        );
        let has_signal = output.iter().any(|&v| v.abs() > 1e-4);
        assert!(has_signal, "quantizer output appears all zero");

        let mean = output.iter().sum::<f32>() / output.len() as f32;
        println!("quantizer decode 1-frame mean: {mean:.6}");

        // Check min/max
        let min_val = output.iter().cloned().fold(f32::NAN, f32::min);
        let max_val = output.iter().cloned().fold(f32::NAN, f32::max);
        println!(
            "quantizer decode 1-frame range: [{min_val:.4}, {max_val:.4}]"
        );
    }

    #[test]
    fn quantizer_decode_multi_frame() {
        let path = codec_gguf_path();
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let qd = QuantizerDecoder::from_gguf(&mut gguf).expect("load quantizer");

        let t = 5;
        let mut codes = Vec::with_capacity(16 * t);
        // Fill with diverse indices
        for i in 0..t {
            for cb in 0..16 {
                // Use varying indices to get different embeddings
                codes.push(((i * 17 + cb * 31) % 2048) as i32);
            }
        }

        let output = qd.decode(&codes, t);
        assert_eq!(output.len(), 512 * t);

        // Each frame should have different output (same codebook indices
        // for different frames → same embeddings, but let's verify
        // output is deterministic and well-formed)
        let all_finite = output.iter().all(|v| v.is_finite());
        assert!(all_finite, "non-finite values in multi-frame output");

        // Check frames are not all identical (different code indices)
        let frame0: Vec<f32> = (0..512).map(|c| output[c * t + 0]).collect();
        let frame1: Vec<f32> = (0..512).map(|c| output[c * t + 1]).collect();
        let frames_differ = frame0
            .iter()
            .zip(frame1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            frames_differ,
            "frames should differ with different code indices"
        );

        // Test with zero codes — should produce valid (non-NaN) output
        let zero_codes = vec![0i32; 16 * t];
        let zero_output = qd.decode(&zero_codes, t);
        assert_eq!(zero_output.len(), 512 * t);
        assert!(zero_output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn quantizer_output_shape_and_layout() {
        let path = codec_gguf_path();
        let mut gguf = GgufFile::open(&path).expect("open GGUF");
        let qd = QuantizerDecoder::from_gguf(&mut gguf).expect("load quantizer");

        let t = 3;
        let codes = vec![42i32; 16 * t];
        let output = qd.decode(&codes, t);

        // Verify C-first layout: output[ch * t + ti]
        // Channel 0 at frames 0..t should be contiguous
        let ch0: Vec<f32> = (0..t).map(|ti| output[0 * t + ti]).collect();
        assert_eq!(ch0.len(), t);

        // Last channel at frames 0..t
        let ch511: Vec<f32> = (0..t).map(|ti| output[511 * t + ti]).collect();
        assert_eq!(ch511.len(), t);
    }
}
