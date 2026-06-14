//! ResUnit: residual block with dual SnakeBeta activations.
//!
//! Architecture:
//! ```text
//! input → SnakeBeta → Conv1d(k=7, dilation=d) → SnakeBeta → Conv1d(k=1) → + skip → output
//! ```
//!
//! All convolutions are causal with stride=1.

use crate::conv::conv1d_causal;
use crate::snake::snake_beta_inplace;

/// Residual unit used inside each DAC decoder block.
///
/// Each block has 3 ResUnits with dilations [1, 3, 9].
pub struct ResUnit {
    /// Conv1(k=7) weight: [C, C, 7] in row-major (GGUF: [7, C, C])
    pub conv1_weight: Vec<f32>,
    /// Conv1 bias: [C]
    pub conv1_bias: Vec<f32>,
    /// Conv2(k=1) weight: [C, C, 1]
    pub conv2_weight: Vec<f32>,
    /// Conv2 bias: [C]
    pub conv2_bias: Vec<f32>,
    /// SnakeBeta alpha for first activation: [C]
    pub snake_alpha: Vec<f32>,
    /// SnakeBeta beta for first activation: [C]
    pub snake_beta: Vec<f32>,
    /// SnakeBeta alpha for second activation: [C]
    pub snake2_alpha: Vec<f32>,
    /// SnakeBeta beta for second activation: [C]
    pub snake2_beta: Vec<f32>,
    /// Dilation for conv1 (1, 3, or 9)
    pub dilation: usize,
}

impl ResUnit {
    /// Forward pass.
    ///
    /// # Arguments
    /// * `input` — [C, T] C-first
    /// * `c` — number of channels
    /// * `t` — number of time steps
    ///
    /// # Returns
    /// `[C, T]` C-first (same shape as input, stride=1 throughout)
    pub fn forward(&self, input: &[f32], c: usize, t: usize) -> Vec<f32> {
        // 1. SnakeBeta → Conv1d(k=7, dilation=d)
        let mut x = input.to_vec();
        snake_beta_inplace(&mut x, c, t, &self.snake_alpha, &self.snake_beta);

        x = conv1d_causal(
            &x, c, t, &self.conv1_weight, c, 7, Some(&self.conv1_bias), self.dilation, 1,
        );

        // 2. SnakeBeta → Conv1d(k=1)
        snake_beta_inplace(&mut x, c, t, &self.snake2_alpha, &self.snake2_beta);

        x = conv1d_causal(
            &x, c, t, &self.conv2_weight, c, 1, Some(&self.conv2_bias), 1, 1,
        );

        // 3. Skip connection
        for i in 0..c * t {
            x[i] += input[i];
        }

        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resunit_output_same_shape() {
        // Minimal test: ResUnit with identity-like weights
        let c = 4;
        let t = 8;

        // Create a ResUnit with simple weights
        // conv1: k=7, C=C=4 (all zeros except center tap = 1.0 for causal)
        let mut conv1_w = vec![0.0f32; c * c * 7];
        for oc in 0..c {
            for ic in 0..c {
                if oc == ic {
                    // Center (last position for causal): ki=6
                    conv1_w[oc * c * 7 + ic * 7 + 6] = 1.0;
                }
            }
        }
        let conv1_b = vec![0.0f32; c];

        // conv2: k=1, identity
        let mut conv2_w = vec![0.0f32; c * c * 1];
        for oc in 0..c {
            conv2_w[oc * c * 1 + oc * 1 + 0] = 1.0;
        }
        let conv2_b = vec![0.0f32; c];

        // Snake params: alpha=beta=0 → x + sin²(0)/0 → this would be NaN
        // Use alpha=-10 (exp(-10) ≈ 0) → x + sin²(0)/1 = x
        let snake_alpha = vec![-10.0f32; c];
        let snake_beta = vec![1.0f32; c];
        let snake2_alpha = vec![-10.0f32; c];
        let snake2_beta = vec![1.0f32; c];

        let ru = ResUnit {
            conv1_weight: conv1_w,
            conv1_bias: conv1_b,
            conv2_weight: conv2_w,
            conv2_bias: conv2_b,
            snake_alpha,
            snake_beta,
            snake2_alpha,
            snake2_beta,
            dilation: 1,
        };

        let input = vec![0.5f32; c * t];
        let output = ru.forward(&input, c, t);

        assert_eq!(output.len(), c * t, "output shape should match input");
        assert!(output.iter().all(|v| v.is_finite()), "non-finite output");

        // With identity conv and near-zero snake, output ≈ input + input = 2*input
        // (because of skip connection: conv2(snake(conv1(snake(x)))) ≈ x, then + skip = 2x)
        let mean_ratio = output.iter().zip(input.iter()).map(|(o, i)| o / i).sum::<f32>() / (c * t) as f32;
        println!("ResUnit identity test: output/input ratio ≈ {mean_ratio:.4}");
        assert!(
            (mean_ratio - 2.0).abs() < 0.1,
            "expected ratio ~2.0 with identity convs, got {mean_ratio:.4}"
        );
    }
}
