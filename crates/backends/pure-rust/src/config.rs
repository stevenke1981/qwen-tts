//! Model hyperparameters parsed from talker GGUF metadata keys.
//!
//! Expected GGUF metadata keys follow both `llama-*` and `qwen3-tts-*`
//! conventions. We try the architecture-specific prefix first, then fall
//! back to `llama.*` for compatibility with llama.cpp-based tools.

use std::collections::HashMap;
use candle_core::quantized::gguf_file;

/// Architecture hyperparameters read from the talker GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f64,
    pub norm_eps: f64,
}

impl ModelConfig {
    /// Parse a [`ModelConfig`] from the metadata map of a loaded GGUF file.
    ///
    /// Every field has a hard-coded default that is overridden when the
    /// corresponding metadata key exists in the GGUF file.
    ///
    /// We try architecture-specific keys first (`qwen3-tts.talker.*` for
    /// talker, `qwen3-tts.code_pred.*` for code predictor), then generic
    /// architecture keys (`qwen3-tts.*`), then standard keys (`llama.*`).
    /// This ensures qwen3-tts GGUFs work correctly alongside standard
    /// llama.cpp models.
    pub fn from_gguf(metadata: &HashMap<String, gguf_file::Value>) -> Self {
        Self {
            d_model: get_u32_any(metadata, &[
                "qwen3-tts.talker.embedding_length",
                "qwen3-tts.embedding_length",
                "llama.embedding_length",
            ])
            .unwrap_or(2048) as usize,
            n_layers: get_u32_any(metadata, &[
                "qwen3-tts.talker.block_count",
                "qwen3-tts.block_count",
                "llama.block_count",
            ])
            .unwrap_or(24) as usize,
            n_heads: get_u32_any(metadata, &[
                "qwen3-tts.talker.attention.head_count",
                "qwen3-tts.attention.head_count",
                "llama.attention.head_count",
            ])
            .unwrap_or(16) as usize,
            n_kv_heads: get_u32_any(metadata, &[
                "qwen3-tts.talker.attention.head_count_kv",
                "qwen3-tts.attention.head_count_kv",
                "llama.attention.head_count_kv",
            ])
            .unwrap_or(16) as usize,
            vocab_size: get_u32_any(metadata, &[
                "qwen3-tts.talker.text_vocab_size",
                "qwen3-tts.talker.vocab_size",
                "qwen3-tts.vocab_size",
                "llama.vocab_size",
            ])
            .unwrap_or(152064) as usize,
            max_seq_len: get_u32_any(metadata, &[
                "qwen3-tts.talker.context_length",
                "qwen3-tts.context_length",
                "llama.context_length",
            ])
            .unwrap_or(8192) as usize,
            rope_theta: get_f64_any(metadata, &[
                "qwen3-tts.talker.rope.freq_base",
                "qwen3-tts.code_pred.rope.freq_base",
                "qwen3-tts.rope.freq_base",
                "llama.rope.freq_base",
            ])
            .unwrap_or(10_000_000.0),
            norm_eps: get_f64_any(metadata, &[
                "qwen3-tts.talker.attention.layer_norm_rms_epsilon",
                "qwen3-tts.attention.layer_norm_rms_epsilon",
                "llama.attention.layer_norm_rms_epsilon",
            ])
            .unwrap_or(1e-6),
        }
    }

    /// Head dimension = d_model / n_heads
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

/// Try each key in order, return the first match.
fn get_u32_any(map: &HashMap<String, gguf_file::Value>, keys: &[&str]) -> Option<u32> {
    for key in keys {
        if let Some(v) = map.get(*key) {
            if let Ok(n) = v.to_u32() {
                return Some(n);
            }
        }
    }
    None
}

/// Try each key in order, return the first match (accepts f32 or f64).
fn get_f64_any(map: &HashMap<String, gguf_file::Value>, keys: &[&str]) -> Option<f64> {
    for key in keys {
        if let Some(v) = map.get(*key) {
            if let Ok(n) = v.to_f32() {
                return Some(f64::from(n));
            }
            if let Ok(n) = v.to_f64() {
                return Some(n);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file::Value;

    #[test]
    fn defaults_are_reasonable() {
        let config = ModelConfig {
            d_model: 2048,
            n_layers: 24,
            n_heads: 16,
            n_kv_heads: 16,
            vocab_size: 152064,
            max_seq_len: 8192,
            rope_theta: 1_000_000.0,
            norm_eps: 1e-6,
        };
        assert_eq!(config.head_dim(), 128);
        assert!(config.rope_theta > 0.0);
        assert!(config.norm_eps > 0.0);
    }

    #[test]
    fn head_dim_round_trip() {
        let config = ModelConfig {
            d_model: 4096,
            n_heads: 32,
            ..ModelConfig {
                d_model: 2048,
                n_layers: 24,
                n_heads: 16,
                n_kv_heads: 16,
                vocab_size: 152064,
                max_seq_len: 8192,
                rope_theta: 1_000_000.0,
                norm_eps: 1e-6,
            }
        };
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn get_u32_from_empty_map() {
        let map = HashMap::new();
        assert_eq!(get_u32_any(&map, &["nonexistent"]), None);
    }

    #[test]
    fn get_u32_falls_through_to_second_key() {
        let mut map = HashMap::new();
        map.insert("b".to_string(), Value::U32(42));
        assert_eq!(get_u32_any(&map, &["a", "b"]), Some(42));
        assert_eq!(get_u32_any(&map, &["a", "c"]), None);
    }

    #[test]
    fn get_f64_from_empty_map() {
        let map = HashMap::new();
        assert_eq!(get_f64_any(&map, &["nonexistent"]), None);
    }

    #[test]
    fn get_f64_falls_through() {
        let mut map = HashMap::new();
        map.insert("b".to_string(), Value::F32(3.14));
        assert!((get_f64_any(&map, &["a", "b"]).unwrap() - 3.14).abs() < 1e-6);
    }
}
