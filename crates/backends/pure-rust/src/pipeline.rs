//! Orchestration: ties the talker transformer, code predictor heads, and
//! codec decoder into a single `synthesize()` entry point.

use std::path::Path;
use qwen_tts_runtime::SynthesisRequest;

/// Holds loaded model weights and decoded state for synthesis.
pub struct Pipeline {
    // TODO: talker: Talker,
    // TODO: code_predictor: CodePredictor,
    // TODO: tokenizer: Tokenizer,
    // TODO: codec_weights: ...,
    _talker_path: std::path::PathBuf,
    _codec_path: std::path::PathBuf,
}

impl Pipeline {
    /// Load both GGUF files and the codec decoder weights.
    pub fn new(talker_path: &Path, codec_path: &Path) -> anyhow::Result<Self> {
        // TODO(Task 7): load Talker GGUF, CodePredictor GGUF, codec decoder
        Ok(Self {
            _talker_path: talker_path.to_owned(),
            _codec_path: codec_path.to_owned(),
        })
    }

    /// Run the full TTS pipeline: tokenize → talker → code predictor → codec decode.
    #[allow(unused_variables)]
    pub fn synthesize(&self, request: &SynthesisRequest) -> anyhow::Result<Vec<i16>> {
        // TODO(Task 7): implement full pipeline
        anyhow::bail!("Pipeline::synthesize not yet implemented")
    }
}
