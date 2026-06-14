use crate::DeviceKind;
use qwen_tts_core::{TtsModelSet, WavValidationError};
use std::{fmt, path::PathBuf};

pub type BackendResult<T> = Result<T, BackendError>;

#[derive(Debug)]
pub enum BackendError {
    Unavailable(String),
    InvalidRequest(String),
    CommandFailed {
        program: String,
        status: Option<i32>,
        stderr: String,
    },
    WavValidation(WavValidationError),
    Io(std::io::Error),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(message) => write!(f, "backend unavailable: {message}"),
            Self::InvalidRequest(message) => write!(f, "invalid request: {message}"),
            Self::CommandFailed {
                program,
                status,
                stderr,
            } => {
                write!(
                    f,
                    "command failed: {program}; status={status:?}; stderr={stderr}"
                )
            }
            Self::WavValidation(err) => write!(f, "{err}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl std::error::Error for BackendError {}

impl From<std::io::Error> for BackendError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<WavValidationError> for BackendError {
    fn from(value: WavValidationError) -> Self {
        Self::WavValidation(value)
    }
}

#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    pub text: String,
    pub language: String,
    pub speaker: Option<String>,
    pub instruct: Option<String>,
    pub seed: Option<i64>,
    pub max_new_tokens: Option<i32>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub do_sample: Option<bool>,
    pub out_path: PathBuf,
    pub device: DeviceKind,
    pub models: TtsModelSet,
}

#[derive(Debug, Clone)]
pub struct SynthesisResponse {
    pub wav_path: PathBuf,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub data_size_bytes: u32,
    pub backend_name: String,
}

pub trait RuntimeBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn device_kind(&self) -> DeviceKind;
    fn is_available(&self) -> bool;

    /// Synthesizes a single request into a WAV file.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the request is invalid, the backend is not
    /// available, the external process fails, output validation fails, or file
    /// I/O fails.
    fn synthesize(&self, request: &SynthesisRequest) -> BackendResult<SynthesisResponse>;
}
