//! Causal 1D convolution forward kernel.
//!
//! Supports arbitrary kernel size, dilation, stride, multiple input/output channels,
//! and optional bias. Designed for the qwen3-tts codec decoder's Conv1d and ResUnit blocks.
//!
//! # Causal constraint
//! Left-pads by `(k-1) * dilation` zeros so output[t] depends only on input[0..=t].
//! This is important for streaming / real-time inference where future input is unknown.
//!
//! # Tensor layout
//! - Input:  `[C_in, T]` in row-major (C_in × T)
//! - Weight: `[C_out, C_in, k]` in row-major
//! - Bias:   optional `[C_out]`

// ---------------------------------------------------------------------------
// Causal Conv1d
// ---------------------------------------------------------------------------

/// Causal 1D convolution.
///
/// # Arguments
/// - `input`:  `[C_in, T]` in row-major
/// - `c_in`:   number of input channels
/// - `t_in`:   input time steps
/// - `weight`: `[C_out, C_in, k]` in row-major
/// - `c_out`:  number of output channels
/// - `k`:      kernel size
/// - `bias`:   optional `[C_out]`
/// - `dilation`: dilation factor (default 1)
/// - `stride`:   stride (default 1)
///
/// # Returns
/// `[C_out, T_out]` where `T_out = T_in / stride`.
pub fn conv1d_causal(
    input: &[f32],
    c_in: usize,
    t_in: usize,
    weight: &[f32],
    c_out: usize,
    k: usize,
    bias: Option<&[f32]>,
    dilation: usize,
    stride: usize,
) -> Vec<f32> {
    let t_out = t_in / stride;
    let mut output = vec![0.0f32; c_out * t_out];

    // For each output channel
    for oc in 0..c_out {
        let w_base = oc * c_in * k;
        let out_base = oc * t_out;

        for ot in 0..t_out {
            let it = ot * stride; // center position in input

            let mut sum = 0.0f32;

            // Sum over input channels
            for ic in 0..c_in {
                let w_ch = w_base + ic * k;

                // Sum over kernel positions (causal: look left from center)
                for ki in 0..k {
                    // input index = it - (k-1 - ki) * dilation
                    let ii = it as isize - ((k - 1 - ki) * dilation) as isize;
                    if ii < 0 {
                        continue; // left-padding zeros
                    }
                    let ii = ii as usize;

                    let w_idx = w_ch + ki;
                    let in_idx = ic * t_in + ii;
                    sum += weight[w_idx] * input[in_idx];
                }
            }

            if let Some(b) = bias {
                sum += b[oc];
            }

            output[out_base + ot] = sum;
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

    /// Simple 1×1×3 causal conv with known expected output.
    #[test]
    fn causal_conv1d_simple() {
        // input: [1, 4] = [1.0, 2.0, 3.0, 4.0]
        let input = vec![1.0, 2.0, 3.0, 4.0];
        // weight: [1, 1, 3] = [0.5, 1.0, 0.5]
        let weight = vec![0.5, 1.0, 0.5];
        // bias: [1] = [0.0]
        let bias = vec![0.0];

        // Expected (k=3, dilation=1, stride=1):
        // output[0] = 0.5*pad + 1.0*pad + 0.5*1.0 = 0.5
        // output[1] = 0.5*pad + 1.0*1.0 + 0.5*2.0 = 2.0
        // output[2] = 0.5*1.0 + 1.0*2.0 + 0.5*3.0 = 4.0
        // output[3] = 0.5*2.0 + 1.0*3.0 + 0.5*4.0 = 6.0
        let result = conv1d_causal(&input, 1, 4, &weight, 1, 3, Some(&bias), 1, 1);
        assert_eq!(result.len(), 4);

        let expected = [0.5, 2.0, 4.0, 6.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "output[{i}]: got {got}, expected {exp}");
        }
    }

    /// Causal conv with bias offset.
    #[test]
    fn causal_conv1d_with_bias() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 1.0]; // k=2
        let bias = vec![10.0];

        // Expected (k=2, dilation=1, stride=1):
        // output[0] = 1.0*pad + 1.0*1.0 + 10 = 11.0
        // output[1] = 1.0*1.0 + 1.0*2.0 + 10 = 13.0
        // output[2] = 1.0*2.0 + 1.0*3.0 + 10 = 15.0
        let result = conv1d_causal(&input, 1, 3, &weight, 1, 2, Some(&bias), 1, 1);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 11.0).abs() < 1e-6);
        assert!((result[1] - 13.0).abs() < 1e-6);
        assert!((result[2] - 15.0).abs() < 1e-6);
    }

    /// No bias.
    #[test]
    fn causal_conv1d_no_bias() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![0.5, 0.5];
        let result = conv1d_causal(&input, 1, 3, &weight, 1, 2, None, 1, 1);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 1.5).abs() < 1e-6);
        assert!((result[2] - 2.5).abs() < 1e-6);
    }

    /// Causal conv with dilation=3 (like ResUnit residual blocks).
    #[test]
    fn causal_conv1d_dilation() {
        // input: [1, 10], k=3, dilation=3
        let input: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        // weight: uniform [0.1, 0.2, 0.3]
        let weight = vec![0.1, 0.2, 0.3];

        // Expected (k=3, dilation=3):
        // output[t] = 0.1*in[t-6] + 0.2*in[t-3] + 0.3*in[t]
        // where negative indices are padded with zero.
        //
        // t=0: 0.1*0 + 0.2*0 + 0.3*1 = 0.3
        // t=1: 0.1*0 + 0.2*0 + 0.3*2 = 0.6
        // t=2: 0.1*0 + 0.2*0 + 0.3*3 = 0.9
        // t=3: 0.1*0 + 0.2*1 + 0.3*4 = 1.4
        // t=4: 0.1*0 + 0.2*2 + 0.3*5 = 1.9
        // t=5: 0.1*0 + 0.2*3 + 0.3*6 = 2.4
        // t=6: 0.1*1 + 0.2*4 + 0.3*7 = 3.0
        // t=7: 0.1*2 + 0.2*5 + 0.3*8 = 3.6
        // t=8: 0.1*3 + 0.2*6 + 0.3*9 = 4.2
        // t=9: 0.1*4 + 0.2*7 + 0.3*10 = 4.8

        let result = conv1d_causal(&input, 1, 10, &weight, 1, 3, None, 3, 1);
        assert_eq!(result.len(), 10);

        let expected = [0.3, 0.6, 0.9, 1.4, 1.9, 2.4, 3.0, 3.6, 4.2, 4.8];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "output[{i}]: got {got}, expected {exp}");
        }
    }

    /// Multiple input channels, single output channel.
    #[test]
    fn causal_conv1d_multi_input_channel() {
        // input: [2, 3]
        // ch0: [1, 2, 3], ch1: [4, 5, 6]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // weight: [1, 2, 2] = ch0: [0.5, 1.0], ch1: [0.1, 0.2]
        let weight = vec![0.5, 1.0, 0.1, 0.2];

        // Expected (k=2, dilation=1):
        // t=0: 0.5*0 + 1.0*1 + 0.1*0 + 0.2*4 = 1.8
        // t=1: 0.5*1 + 1.0*2 + 0.1*4 + 0.2*5 = 3.9
        // t=2: 0.5*2 + 1.0*3 + 0.1*5 + 0.2*6 = 5.7
        let result = conv1d_causal(&input, 2, 3, &weight, 1, 2, None, 1, 1);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.8).abs() < 1e-6, "result[0] = {}", result[0]);
        assert!((result[1] - 3.9).abs() < 1e-6, "result[1] = {}", result[1]);
        assert!((result[2] - 5.7).abs() < 1e-6, "result[2] = {}", result[2]);
    }

    /// Multiple output channels.
    #[test]
    fn causal_conv1d_multi_output_channel() {
        // input: [1, 4]
        let input = vec![1.0, 2.0, 3.0, 4.0];
        // weight: [2, 1, 3] = oc0: [1, 1, 1], oc1: [0.5, 0.5, 0.5]
        let weight = vec![1.0, 1.0, 1.0, 0.5, 0.5, 0.5];
        let bias = vec![0.0, 1.0];

        // Causal k=3, dilation=1:
        // weight[0]→input[t-2], weight[1]→input[t-1], weight[2]→input[t]
        //
        // oc0 (weight=[1,1,1], bias=0):
        //   t=0: 1*0 + 1*0 + 1*1 = 1
        //   t=1: 1*0 + 1*1 + 1*2 = 3
        //   t=2: 1*1 + 1*2 + 1*3 = 6
        //   t=3: 1*2 + 1*3 + 1*4 = 9
        //
        // oc1 (weight=[0.5,0.5,0.5], bias=1):
        //   t=0: 0.5*0 + 0.5*0 + 0.5*1 + 1 = 1.5
        //   t=1: 0.5*0 + 0.5*1 + 0.5*2 + 1 = 2.5
        //   t=2: 0.5*1 + 0.5*2 + 0.5*3 + 1 = 4.0
        //   t=3: 0.5*2 + 0.5*3 + 0.5*4 + 1 = 5.5
        let result = conv1d_causal(&input, 1, 4, &weight, 2, 3, Some(&bias), 1, 1);
        assert_eq!(result.len(), 8); // 2 * 4

        // oc0 at indices [0..4)
        let oc0_expected = [1.0, 3.0, 6.0, 9.0];
        for (i, &exp) in oc0_expected.iter().enumerate() {
            assert!((result[i] - exp).abs() < 1e-6, "oc0[{i}]: got {}, expected {}", result[i], exp);
        }

        // oc1 at indices [4..8)
        let oc1_expected = [1.5, 2.5, 4.0, 5.5];
        for (i, &exp) in oc1_expected.iter().enumerate() {
            assert!((result[4 + i] - exp).abs() < 1e-6, "oc1[{i}]: got {}, expected {}", result[4 + i], exp);
        }
    }

    /// Stride > 1.
    #[test]
    fn causal_conv1d_stride() {
        // input: [1, 6]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // weight: [1, 1, 3] = [0.5, 1.0, 0.5]
        let weight = vec![0.5, 1.0, 0.5];

        // stride=2, k=3, dilation=1:
        // t_out = 3
        // ot=0 → it=0:
        //   ki=0: ii=0-2=-2<0 skip
        //   ki=1: ii=0-1=-1<0 skip
        //   ki=2: ii=0 → input[0]=1, weight[2]=0.5 → 0.5
        //   sum=0.5
        // ot=1 → it=2:
        //   ki=0: ii=2-2=0 → input[0]=1, weight[0]=0.5 → 0.5
        //   ki=1: ii=2-1=1 → input[1]=2, weight[1]=1.0 → 2.0
        //   ki=2: ii=2-0=2 → input[2]=3, weight[2]=0.5 → 1.5
        //   sum=4.0
        // ot=2 → it=4:
        //   ki=0: ii=4-2=2 → input[2]=3, weight[0]=0.5 → 1.5
        //   ki=1: ii=4-1=3 → input[3]=4, weight[1]=1.0 → 4.0
        //   ki=2: ii=4-0=4 → input[4]=5, weight[2]=0.5 → 2.5
        //   sum=8.0
        let result = conv1d_causal(&input, 1, 6, &weight, 1, 3, None, 1, 2);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);
        assert!((result[2] - 8.0).abs() < 1e-6);
    }

    /// Edge case: kernel size 1 (just multiply by weight + bias).
    #[test]
    fn causal_conv1d_kernel_1() {
        let input = vec![2.0, 4.0, 6.0];
        let weight = vec![0.5]; // k=1
        let bias = vec![1.0];

        let result = conv1d_causal(&input, 1, 3, &weight, 1, 1, Some(&bias), 1, 1);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-6); // 2*0.5 + 1 = 2.0
        assert!((result[1] - 3.0).abs() < 1e-6); // 4*0.5 + 1 = 3.0
        assert!((result[2] - 4.0).abs() < 1e-6); // 6*0.5 + 1 = 4.0
    }

    /// Integration test: load a real F16 conv weight from GGUF and run it.
    /// Verifies no panics and output is finite + non-zero.
    #[test]
    fn conv1d_from_real_gguf_weight() {
        let path = {
            let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let workspace = manifest_dir.parent().and_then(|p| p.parent()).unwrap();
            workspace.join("models").join("qwen-tokenizer-12hz-Q8_0.gguf")
        };

        let mut gguf = crate::gguf::GgufFile::open(&path)
            .expect("should open codec GGUF");

        // Use pre_conv.weight: shape [3, 512, 1024] → k=3, C_in=512, C_out=1024
        let info = gguf.tensor("tok_dec.pre_conv.weight").expect("pre_conv.weight");
        assert_eq!(info.ggml_type, 1, "expected F16 type");
        let k = info.shape[0] as usize;   // 3
        let c_in = info.shape[1] as usize; // 512
        let c_out = info.shape[2] as usize; // 1024

        let weight = gguf.read_tensor_f32("tok_dec.pre_conv.weight")
            .expect("should read weight");

        // Create a random-ish input: [512, 8]
        let t = 8;
        let input: Vec<f32> = (0..c_in * t).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        let output = conv1d_causal(&input, c_in, t, &weight, c_out, k, None, 1, 1);
        assert_eq!(output.len(), c_out * t);

        // All finite
        assert!(output.iter().all(|v| v.is_finite()), "all values should be finite");

        // Non-zero output
        let has_nonzero = output.iter().any(|v| v.abs() > 0.001);
        assert!(has_nonzero, "output should have non-zero values");

        let mean = output.iter().sum::<f32>() / output.len() as f32;
        println!("conv1d_causal(pre_conv.weight) [{c_out}×{c_in}×{k}] input [{c_in}×{t}]: mean={mean:.6}");
    }
}
