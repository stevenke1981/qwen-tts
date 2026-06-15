//! Logit post-processing and token sampling (top-k, top-p, temperature).
//!
//! All operations work on flat `Vec<f32>` slices for simplicity and CPU
//! performance — candle tensor ops for top-k/top-p are not available in the
//! version used here, and manual iteration over the vocabulary (≤152k tokens)
//! is fast enough for CPU inference.

use std::collections::HashSet;

use rand::Rng;

use candle_core::Tensor;

/// Apply temperature scaling to a logits slice (in-place).
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if (temperature - 1.0).abs() <= f32::EPSILON {
        return;
    }
    for v in logits.iter_mut() {
        *v /= temperature;
    }
}

/// Apply top-k filtering (in-place): set all logits to -inf except the top-k.
pub fn apply_top_k(logits: &mut [f32], k: usize) {
    if k >= logits.len() {
        return;
    }
    // Partial sort: find the k-th largest value
    let mut sorted = logits.to_vec();
    sorted.select_nth_unstable_by(logits.len() - k, |a, b| a.partial_cmp(b).unwrap());
    let threshold = sorted[logits.len() - k];
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Apply top-p (nucleus) filtering (in-place): keep the smallest set of tokens
/// whose cumulative probability exceeds `p`.
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 - f32::EPSILON {
        return;
    }

    // Softmax to probabilities
    let probs = softmax(logits);

    // Sort by descending probability
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    // Cumulative sum
    let mut cum = 0.0f32;
    let mut cutoff = logits.len();
    for (rank, &idx) in indices.iter().enumerate() {
        cum += probs[idx];
        if cum > p {
            cutoff = rank;
            break;
        }
    }

    // Zero out all logits not in the nucleus
    let mut keep = vec![false; logits.len()];
    for &idx in indices.iter().take(cutoff + 1) {
        keep[idx] = true;
    }
    for (i, v) in logits.iter_mut().enumerate() {
        if !keep[i] {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Compute softmax in-place (modifies logits to probabilities).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for v in &mut exps {
            *v /= sum;
        }
    }
    exps
}

/// Sample a single token from logits using argmax (greedy).
///
/// `logits`: `[batch, 1, vocab_size]` — returns `u32` token ID for batch 0.
pub fn sample_argmax(logits: &Tensor) -> anyhow::Result<u32> {
    use candle_core::IndexOp;
    let logits_1d = logits.i((0, 0, ..))?;
    let token = logits_1d.argmax(0)?;
    let val: u32 = token.to_vec0()?;
    Ok(val)
}

/// Mask logits in `[lo, hi)` to -inf, except `keep` is left untouched.
/// Mirrors C++ `apply_suppress` in sampling.h.
pub fn apply_suppress(logits: &mut [f32], lo: usize, hi: usize, keep: u32) {
    let lo = lo.min(logits.len());
    let hi = hi.min(logits.len());
    for i in lo..hi {
        if i as u32 != keep {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

/// HF-style repetition penalty over **unique** tokens in history.
///
/// For each unique token `t` in `history`:
///   if logits[t] >= 0 → logits[t] /= penalty
///   if logits[t] <  0 → logits[t] *= penalty
///
/// Mirrors C++ `apply_repetition_penalty` in sampling.h (uses per-call
/// bool-flag vector to deduplicate).
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() <= f32::EPSILON || history.is_empty() {
        return;
    }
    let n_vocab = logits.len();
    let mut seen = HashSet::new();
    for &tok in history {
        let t = tok as usize;
        if t >= n_vocab {
            continue;
        }
        if !seen.insert(t) {
            continue;
        }
        let s = logits[t];
        logits[t] = if s < 0.0 { s * penalty } else { s / penalty };
    }
}

/// Sample a single token ID from a flat logits array.
///
/// Pipeline: repetition_penalty(history) → temperature → top_k → top_p →
/// softmax → multinomial.
///
/// When `temperature <= 0.0`, returns argmax (greedy) ignoring all other
/// sampling params.
///
/// `history` — optional c0 token history for repetition penalty (None = skip).
///
/// Returns `(token_id, probability_of_selected_token)`.
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rng: &mut impl Rng,
    history: Option<&[u32]>,
    repetition_penalty: f32,
) -> (u32, f32) {
    // Greedy: argmax, skip all sampling chain
    if temperature <= 0.0 {
        let idx = argmax_idx(logits);
        return (idx, 1.0);
    }

    let mut logits = logits.to_vec();

    apply_repetition_penalty(&mut logits, history.unwrap_or(&[]), repetition_penalty);
    apply_temperature(&mut logits, temperature);
    if let Some(k) = top_k {
        apply_top_k(&mut logits, k);
    }
    if let Some(p) = top_p {
        apply_top_p(&mut logits, p);
    }

    let probs = softmax(&logits);

    // Weighted sampling
    let total: f32 = probs.iter().sum();
    if total <= 0.0 {
        // Fallback: argmax
        let (idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));
        return (idx as u32, 1.0);
    }

    let r = rng.gen::<f32>() * total;
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if cum >= r {
            return (i as u32, p);
        }
    }

    // Fallback (shouldn't reach here)
    ((probs.len() - 1) as u32, probs[probs.len() - 1])
}

/// Argmax: return index of the highest logit.
pub fn argmax_idx(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_logits() -> Vec<f32> {
        vec![2.0, 1.0, 0.5, 0.1, 0.0, -0.5, -1.0, -2.0]
    }

    #[test]
    fn test_temperature_1_no_op() {
        let mut logits = test_logits();
        let original = logits.clone();
        apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_temperature_scales() {
        let mut logits = vec![10.0, 0.0];
        apply_temperature(&mut logits, 0.5);
        assert!((logits[0] - 20.0).abs() < 1e-5);
        assert_eq!(logits[1], 0.0);
    }

    #[test]
    fn test_top_k_keeps_exactly_k() {
        let mut logits = test_logits();
        apply_top_k(&mut logits, 2);
        let finite: Vec<f32> = logits.iter().copied().filter(|v| v.is_finite()).collect();
        assert_eq!(finite.len(), 2);
    }

    #[test]
    fn test_top_k_1_is_argmax() {
        let mut logits = test_logits();
        apply_top_k(&mut logits, 1);
        let idx = argmax_idx(&logits);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_top_p_keeps_nonzero_prob() {
        let mut logits = vec![100.0, 1.0, 0.1, 0.01];
        apply_top_p(&mut logits, 0.9);
        // Token 0 should dominate
        assert!(logits[0].is_finite());
    }

    #[test]
    fn test_sample_deterministic_with_seed() {
        let logits = test_logits();
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);
        let (t1, _) = sample_token(&logits, 0.8, Some(5), None, &mut rng1, None, 1.0);
        let (t2, _) = sample_token(&logits, 0.8, Some(5), None, &mut rng2, None, 1.0);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_argmax_idx() {
        assert_eq!(argmax_idx(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax_idx(&[-10.0, -5.0, -1.0]), 2);
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let probs = softmax(&[1.0, 2.0, 3.0, 4.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
