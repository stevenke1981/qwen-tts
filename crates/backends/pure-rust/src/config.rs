//! Model hyperparameters parsed from talker GGUF metadata keys.
//!
//! Expected GGUF metadata keys follow the `llama-*` convention used by
//! llama.cpp / candle for Qwen2 architectures.

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
    pub fn from_gguf(metadata: &HashMap<String, gguf_file::Value>) -> Self {
        Self {
            d_model: get_u32(metadata, "llama.embedding_length").unwrap_or(2048) as usize,
            n_layers: get_u32(metadata, "llama.block_count").unwrap_or(24) as usize,
            n_heads: get_u32(metadata, "llama.attention.head_count").unwrap_or(16) as usize,
            n_kv_heads: get_u32(metadata, "llama.attention.head_count_kv").unwrap_or(16) as usize,
            vocab_size: get_u32(metadata, "llama.vocab_size").unwrap_or(152064) as usize,
            max_seq_len: get_u32(metadata, "llama.context_length").unwrap_or(8192) as usize,
            rope_theta: get_f64(metadata, "llama.rope.freq_base").unwrap_or(1_000_000.0),
            norm_eps: get_f64(metadata, "llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
        }
    }

    /// Head dimension = d_model / n_heads
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

fn get_u32(map: &HashMap<String, gguf_file::Value>, key: &str) -> Option<u32> {
    map.get(key)?.to_u32().ok()
}

fn get_f64(map: &HashMap<String, gguf_file::Value>, key: &str) -> Option<f64> {
    // GGUF stores floats as f32 usually, so we try f32 first then f64
    if let Ok(v) = map.get(key)?.to_f32() {
        Some(f64::from(v))
    } else if let Ok(v) = map.get(key)?.to_f64() {
        Some(v)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(get_u32(&map, "nonexistent"), None);
    }

    #[test]
    fn get_f64_from_empty_map() {
        let map = HashMap::new();
        assert_eq!(get_f64(&map, "nonexistent"), None);
    }
}
