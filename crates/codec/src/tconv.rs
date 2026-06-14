//! Causal transposed 1D convolution (for upsampling in DAC decoder).
//!
//! Maps `[C_in, T]` → `[C_out, T × stride]` by scattering each input element
//! across stride output positions. Causal by construction: output[t] depends
//! only on input[0..=⌊t/stride⌋].
//!
//! # Weight layout
//! `[C_in, C_out, k]` in row-major (note: input channel outer vs regular conv).

// ---------------------------------------------------------------------------
// Transposed Conv1d
// ---------------------------------------------------------------------------

/// Causal transposed 1D convolution.
///
/// # Arguments
/// - `input`:  `[C_in, T]` in row-major
/// - `c_in`:   number of input channels
/// - `t_in`:   input time steps
/// - `weight`: `[C_in, C_out, k]` in row-major
/// - `c_out`:  number of output channels
/// - `k`:      kernel size
/// - `stride`: upsampling factor
/// - `bias`:   optional `[C_out]`
///
/// # Returns
/// `[C_out, T_out]` where `T_out = T_in * stride`.
pub fn conv_transpose1d_causal(
    input: &[f32],
    c_in: usize,
    t_in: usize,
    weight: &[f32],
    c_out: usize,
    k: usize,
    stride: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let t_out = t_in * stride;
    let mut output = vec![0.0f32; c_out * t_out];

    // For each input position, scatter-add to output
    for it in 0..t_in {
        for ic in 0..c_in {
            let w_base = ic * c_out * k;
            let in_val = input[ic * t_in + it];

            for ki in 0..k {
                let ot = it * stride + ki;
                if ot >= t_out {
                    break;
                }

                for oc in 0..c_out {
                    let w_idx = w_base + oc * k + ki;
                    output[oc * t_out + ot] += in_val * weight[w_idx];
                }
            }
        }
    }

    // Add bias
    if let Some(b) = bias {
        for oc in 0..c_out {
            let out_off = oc * t_out;
            let bval = b[oc];
            for ot in 0..t_out {
                output[out_off + ot] += bval;
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Single channel, simple upsampling.
    #[test]
    fn transpose_single_channel() {
        // input: [1, 2] = [1.0, 2.0]
        let input = vec![1.0, 2.0];
        // weight: [1, 1, 3] = [0.5, 1.0, 0.5]
        let weight = vec![0.5, 1.0, 0.5];
        // stride=2, k=3

        // input[0]=1.0 contributes to output positions 0,1,2:
        //   ot=0: 1.0*0.5 = 0.5
        //   ot=1: 1.0*1.0 = 1.0
        //   ot=2: 1.0*0.5 = 0.5
        // input[1]=2.0 contributes to output positions 2,3,4:
        //   ot=2: 2.0*0.5 = 1.0
        //   ot=3: 2.0*1.0 = 2.0
        //   ot=4: 2.0*0.5 = 1.0
        // Total: [0.5, 1.0, 1.5, 2.0, 1.0]
        let result = conv_transpose1d_causal(&input, 1, 2, &weight, 1, 3, 2, None);
        assert_eq!(result.len(), 4); // T_out = 2*2 = 4

        let expected = [0.5, 1.0, 1.5, 2.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "output[{i}]: got {got}, expected {exp}");
        }
    }

    /// With bias.
    #[test]
    fn transpose_with_bias() {
        let input = vec![1.0, 2.0];
        let weight = vec![0.5, 0.5]; // k=2
        let bias = vec![10.0];

        // input[0]=1 contributes to ot=0,1: 0.5, 0.5
        // input[1]=2 contributes to ot=2,3: 1.0, 1.0
        // bias=10 to all 4 output positions
        let result = conv_transpose1d_causal(&input, 1, 2, &weight, 1, 2, 2, Some(&bias));
        assert_eq!(result.len(), 4);
        assert!((result[0] - 10.5).abs() < 1e-6);
        assert!((result[1] - 10.5).abs() < 1e-6);
        assert!((result[2] - 11.0).abs() < 1e-6);
        assert!((result[3] - 11.0).abs() < 1e-6);
    }

    /// Multiple output channels.
    #[test]
    fn transpose_multi_output() {
        // input: [1, 2]
        let input = vec![1.0, 2.0];
        // weight: [1, 2, 2] = oc0: [0.5, 0.5], oc1: [1.0, 1.0]
        let weight = vec![0.5, 0.5, 1.0, 1.0];
        let bias = vec![0.0, 1.0];
        // stride=2, k=2

        // oc0: input[0]=1 → ot[0,1]: 0.5,0.5
        //      input[1]=2 → ot[2,3]: 1.0,1.0
        //      total: [0.5, 0.5, 1.0, 1.0]
        // oc1: input[0]=1 → ot[0,1]: 1.0,1.0
        //      input[1]=2 → ot[2,3]: 2.0,2.0
        //      bias=1: [2.0, 2.0, 3.0, 3.0]
        let result = conv_transpose1d_causal(&input, 1, 2, &weight, 2, 2, 2, Some(&bias));
        assert_eq!(result.len(), 8); // 2 * 4

        let oc0_expected = [0.5, 0.5, 1.0, 1.0];
        for (i, &exp) in oc0_expected.iter().enumerate() {
            assert!((result[i] - exp).abs() < 1e-6, "oc0[{i}]: got {}, expected {}", result[i], exp);
        }

        let oc1_expected = [2.0, 2.0, 3.0, 3.0];
        for (i, &exp) in oc1_expected.iter().enumerate() {
            assert!((result[4 + i] - exp).abs() < 1e-6, "oc1[{i}]: got {}, expected {}", result[4 + i], exp);
        }
    }

    /// Multiple input channels.
    #[test]
    fn transpose_multi_input() {
        // input: [2, 2] = ch0: [1, 2], ch1: [3, 4]
        let input = vec![1.0, 2.0, 3.0, 4.0];
        // weight: [2, 1, 2] = ic0: [0.5, 0.5], ic1: [1.0, 1.0]
        let weight = vec![0.5, 0.5, 1.0, 1.0];
        // stride=2, k=2

        // ch0 contributes: [1*0.5, 1*0.5, 2*0.5, 2*0.5] = [0.5, 0.5, 1.0, 1.0]
        // ch1 contributes: [3*1.0, 3*1.0, 4*1.0, 4*1.0] = [3.0, 3.0, 4.0, 4.0]
        // total: [3.5, 3.5, 5.0, 5.0]
        let result = conv_transpose1d_causal(&input, 2, 2, &weight, 1, 2, 2, None);
        assert_eq!(result.len(), 4);
        let expected = [3.5, 3.5, 5.0, 5.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "output[{i}]: got {got}, expected {exp}");
        }
    }

    /// Edge case: k=1 (no scattering, just scale + bias).
    #[test]
    fn transpose_kernel_1() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![2.0]; // k=1
        // stride=2, k=1: ot = it*2 + 0 = [0, 2, 4]
        // output[0] = 1*2 = 2
        // output[2] = 2*2 = 4
        // output[4] = 3*2 = 6
        // others are 0
        let result = conv_transpose1d_causal(&input, 1, 3, &weight, 1, 1, 2, None);
        assert_eq!(result.len(), 6);
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - 4.0).abs() < 1e-6);
        assert!((result[3] - 0.0).abs() < 1e-6);
        assert!((result[4] - 6.0).abs() < 1e-6);
        assert!((result[5] - 0.0).abs() < 1e-6);
    }

    /// Integration: load real transposed conv weight from GGUF.
    #[test]
    fn transpose_from_real_gguf_weight() {
        let path = {
            let md = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let ws = md.parent().and_then(|p| p.parent()).unwrap();
            ws.join("models").join("qwen-tokenizer-12hz-Q8_0.gguf")
        };

        let mut gguf = crate::gguf::GgufFile::open(&path).expect("open GGUF");

        // dec.1.conv_t.weight: [16, 768, 1536] → k=16, C_out=768, C_in=1536
        let info = gguf.tensor("tok_dec.dec.1.conv_t.weight").expect("conv_t.weight");
        assert_eq!(info.ggml_type, 1, "F16 expected");
        let k = info.shape[0] as usize;   // 16
        let c_out = info.shape[1] as usize; // 768
        let c_in = info.shape[2] as usize;  // 1536

        let weight = gguf.read_tensor_f32("tok_dec.dec.1.conv_t.weight")
            .expect("read weight");

        // input: [1536, 2]
        let t = 2;
        let input: Vec<f32> = (0..c_in * t).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

        let output = conv_transpose1d_causal(&input, c_in, t, &weight, c_out, k, 8, None);
        let t_out = t * 8;
        assert_eq!(output.len(), c_out * t_out);

        assert!(output.iter().all(|v| v.is_finite()), "all finite");
        let has_nz = output.iter().any(|v| v.abs() > 0.001);
        assert!(has_nz, "non-zero output expected");

        let mean = output.iter().sum::<f32>() / output.len() as f32;
        println!("transpose(dec.1.conv_t) [{c_in}→{c_out}×{k}] stride=8: mean={mean:.6}");
    }
}
