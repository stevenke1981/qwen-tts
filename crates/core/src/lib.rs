//! Core model types, graph description, common tensor/audio structs.
//! This crate intentionally has no GPU dependency.

pub mod audio;
pub mod gguf;
pub mod graph;
pub mod model;
pub mod ops;
pub mod wav;

pub use audio::{AudioBuffer, AudioSpec};
pub use gguf::{GgufProbe, GgufProbeError};
pub use graph::{GraphNode, NodeKind, TtsGraph};
pub use model::{CodecModel, TalkerModel, TtsModelSet};
pub use wav::{
    read_wav_metadata, validate_wav_file, validate_wav_metadata, WavMetadata, WavValidationError,
    WavValidationResult,
};
