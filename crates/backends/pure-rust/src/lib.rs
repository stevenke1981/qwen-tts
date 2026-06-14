//! Pure Rust TTS backend using candle for the talker transformer and
//! qwen-tts-codec for the DAC decoder. No C++ FFI required.

pub mod config;
pub mod code_predictor;
pub mod pipeline;
pub mod sampling;
pub mod talker;
pub mod tokenizer;

use std::path::PathBuf;
use qwen_tts_runtime::{BackendError, BackendResult, DeviceKind, RuntimeBackend, SynthesisRequest, SynthesisResponse};

use pipeline::Pipeline;

/// The public backend type — implements [`RuntimeBackend`] using pure Rust.
pub struct PureRustBackend {
    talker_path: PathBuf,
    codec_path: PathBuf,
    pipeline: Option<Pipeline>,
}

impl PureRustBackend {
    #[must_use]
    pub fn new(talker_path: PathBuf, codec_path: PathBuf) -> Self {
        Self {
            talker_path,
            codec_path,
            pipeline: None,
        }
    }

    fn init_pipeline(&mut self) -> BackendResult<()> {
        if self.pipeline.is_none() {
            let p = Pipeline::new(&self.talker_path, &self.codec_path)
                .map_err(|e| BackendError::Unavailable(format!("pipeline init failed: {e}")))?;
            self.pipeline = Some(p);
        }
        Ok(())
    }
}

impl RuntimeBackend for PureRustBackend {
    fn name(&self) -> &'static str {
        "pure-rust"
    }

    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn is_available(&self) -> bool {
        self.talker_path.exists() && self.codec_path.exists()
    }

    fn synthesize(&self, request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        // We use &mut self via interior: Pipeline behind OnceLock would need
        // &self -> &mut self coercion, so we require &mut self for lazy init.
        // For now, return a clear "not implemented" error that guides users.
        let _ = (request, &self.talker_path, &self.codec_path);
        Err(BackendError::Unavailable(
            "pure-rust backend: pipeline not yet implemented (Task 7)".into()
        ))
    }
}
