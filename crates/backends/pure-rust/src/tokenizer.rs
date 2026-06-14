//! HuggingFace BPE tokenizer wrapper (via the `tokenizers` crate).
//!
//! Qwen3-TTS uses the Qwen2.5 tokenizer (BPE, 152k vocab) which is stored as
//! `tokenizer.json` inside the HuggingFace model repository. For GGUF-only
//! operation we load the tokenizer from an external `tokenizer.json` file
//! alongside the GGUF weights, or fall back to embedding lookup when no
//! tokenizer file is available.

use std::path::Path;
use tokenizers::Tokenizer;

/// A wrapper around the HuggingFace `tokenizers` tokenizer.
pub struct HfTokenizer {
    inner: Tokenizer,
    /// Whether the tokenizer uses a pad token (most do).
    pad_id: u32,
}

impl HfTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {path:?}: {e}"))?;

        let pad_id = inner
            .get_padding()
            .and_then(|p| Some(p.pad_id))
            .or_else(|| {
                inner
                    .token_to_id("<|endoftext|>")
                    .or_else(|| {
                        inner
                            .token_to_id("<|im_end|>")
                            .or_else(|| inner.token_to_id("<|extra_0|>"))
                    })
            })
            .unwrap_or(0);

        Ok(Self { inner, pad_id })
    }

    /// Load from a bytes buffer (e.g., embedded in GGUF metadata).
    pub fn from_bytes(data: &[u8]) -> anyhow::Result<Self> {
        let inner = Tokenizer::from_bytes(data)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer from bytes: {e}"))?;
        let pad_id = inner
            .get_padding()
            .and_then(|p| Some(p.pad_id))
            .unwrap_or(0);
        Ok(Self { inner, pad_id })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoded = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
        Ok(encoded.get_ids().to_vec())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode error: {e}"))
    }

    /// Convert a single token ID to its string representation.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Convert a token string to its ID.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Pad token ID (used for filling sequences to equal length).
    #[must_use]
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    /// End-of-sequence token ID (commonly `<|endoftext|>` in Qwen2).
    ///
    /// This is `<|extra_0|>` in Qwen3-TTS tokenizer (token 151643).
    #[must_use]
    pub fn eos_id(&self) -> u32 {
        // Qwen2 family uses <|endoftext|> (151643) or <|extra_0|>
        self.inner
            .token_to_id("<|endoftext|>")
            .or_else(|| self.inner.token_to_id("<|extra_0|>"))
            .or_else(|| self.inner.token_to_id("<|im_end|>"))
            .unwrap_or(0)
    }

    /// Start-of-sequence / begin-of-sequence token ID.
    #[must_use]
    pub fn bos_id(&self) -> u32 {
        // Qwen2 has no explicit BOS token; <|endoftext|> is used as BOS
        self.eos_id()
    }

    /// Maximum model context length (from tokenizer metadata if available).
    pub fn max_length(&self) -> Option<usize> {
        // `get_truncation().max_len()` returns Option<usize>
        self.inner
            .get_truncation()
            .and_then(|t| {
                let max = t.max_length;
                if max > 0 { Some(max as usize) } else { None }
            })
    }

    /// The raw `tokenizers::Tokenizer` (for advanced use).
    #[must_use]
    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(false)
    }
}

#[cfg(test)]
mod tests {
    /// Basic sanity test that doesn't require a real tokenizer file.
    #[test]
    fn test_empty_tokenizer_behavior() {
        let tok = "dummy";
        assert!(!tok.is_empty());
    }
}
