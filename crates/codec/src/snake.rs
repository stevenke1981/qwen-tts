//! SnakeBeta activation: y = x + sin²(x · exp(α)) / (exp(β) + eps)
//!
//! The C++ reference pre-computes a = exp(alpha) and inv_b = 1 / (exp(beta) + 1e-9),
//! then evaluates: y = x + sin²(a * x) * inv_b.
//!
//! We compute `a = exp(alpha)` and `inv_b = 1 / (exp(beta) + 1e-9)` per element
//! at each forward pass since it's cheap and matches the C++ semantics exactly.
//!
//! IMPORTANT: The stored `beta` parameter is NOT the denominator directly — it is
//! first exponentiated (like alpha) and then inverted with an epsilon guard.

/// SnakeBeta pointwise: y = x + sin²(x · e^α) / (e^β + eps)
#[inline]
pub fn snake_beta(x: f32, alpha: f32, beta: f32) -> f32 {
    let a = alpha.exp();
    let inv_b = 1.0 / (beta.exp() + 1e-9);
    let sin_val = (x * a).sin();
    x + sin_val * sin_val * inv_b
}

/// Apply SnakeBeta to every element of a tensor `[C, T]` in-place.
///
/// Each channel `c` uses `alpha[c]` and `beta[c]`.
pub fn snake_beta_inplace(
    tensor: &mut [f32],
    c: usize,
    t: usize,
    alpha: &[f32],
    beta: &[f32],
) {
    assert_eq!(alpha.len(), c);
    assert_eq!(beta.len(), c);
    for ch in 0..c {
        let a = alpha[ch];
        let b = beta[ch];
        for ti in 0..t {
            let idx = ch * t + ti;
            tensor[idx] = snake_beta(tensor[idx], a, b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snake_beta_identity_for_zero() {
        // snake_beta(0, α, β) = 0 + sin²(0) / β = 0
        for alpha in [-2.0, 0.0, 2.0] {
            for beta in [0.5, 1.0, 2.0] {
                let result = snake_beta(0.0, alpha, beta);
                assert!(
                    (result - 0.0).abs() < 1e-6,
                    "snake_beta(0, {alpha}, {beta}) = {result}, expected 0"
                );
            }
        }
    }

    #[test]
    fn snake_beta_small_input() {
        // snake_beta(0.5, 0.0, 1.0)
        //   = 0.5 + sin²(0.5 * e^0) * inv_b
        //   where a = exp(0.0) = 1.0, inv_b = 1/(exp(1.0) + 1e-9) ≈ 1/2.71828
        // sin(0.5) ≈ 0.4794, sin² ≈ 0.2298
        // result ≈ 0.5 + 0.2298 / 2.71828 ≈ 0.5 + 0.08454 ≈ 0.58454
        let result = snake_beta(0.5, 0.0, 1.0);
        let a = (0.0f32).exp();
        let inv_b = 1.0 / ((1.0f32).exp() + 1e-9);
        let expected = 0.5 + (0.5f32 * a).sin().powi(2) * inv_b;
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn snake_beta_negative_input() {
        let result = snake_beta(-1.0, 0.0, 2.0);
        // -1.0 + sin²(-1.0) * inv_b where inv_b = 1/(exp(2.0) + 1e-9)
        let a = (0.0f32).exp();
        let inv_b = 1.0 / ((2.0f32).exp() + 1e-9);
        let expected = -1.0 + (-1.0f32 * a).sin().powi(2) * inv_b;
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn snake_beta_inplace_modifies() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        let alpha = vec![0.0f32, 1.0];
        let beta = vec![1.0f32, 1.0];
        // data shape: [C=2, T=2]
        snake_beta_inplace(&mut data, 2, 2, &alpha, &beta);

        // Channel 0: snake_beta(1.0, 0.0, 1.0)
        //   a=exp(0)=1, inv_b=1/(exp(1)+1e-9) ≈ 1/2.71828
        let inv_b0 = 1.0 / ((1.0f32).exp() + 1e-9);
        let expected_0 = 1.0 + (1.0f32).sin().powi(2) * inv_b0;
        assert!((data[0] - expected_0).abs() < 1e-5, "ch0 t0");
        // Channel 1: snake_beta(4.0, 1.0, 1.0)
        //   a=exp(1)=E, inv_b=1/(exp(1)+1e-9)
        let a1 = std::f32::consts::E;
        let inv_b1 = 1.0 / ((1.0f32).exp() + 1e-9);
        let expected_1 = 4.0 + (4.0 * a1).sin().powi(2) * inv_b1;
        assert!((data[3] - expected_1).abs() < 1e-5, "ch1 t1");
    }
}
