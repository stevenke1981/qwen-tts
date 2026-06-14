//! Rust FFI bindings for the qwentts.cpp public C ABI (qwen.h).
//!
//! Links against the pre-built `qwen-core.lib` static library which contains
//! the full TTS pipeline: talker LM forward, code predictor MTP head, codec
//! decoder, BPE tokenizer, prompt builder, and sampling.
//!
//! Every type mirrors the C struct layout exactly with `#[repr(C)]`. Functions
//! are declared as `extern "C"` and called through the FFI boundary. The
//! pre-built library lives at `vendor/qwentts.cpp/build/Release/qwen-core.lib`.
//!
//! # Safety
//!
//! All functions are `unsafe` because they call into C++ code across the FFI
//! boundary. The safe wrappers in [`super::context`] manage pointer lifetimes
//! and error handling.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_void};

// ---------------------------------------------------------------------------
// Constants — matches qwen.h
// ---------------------------------------------------------------------------

pub const QT_ABI_VERSION: i32 = 2;

pub const QT_STATUS_OK: i32 = 0;
pub const QT_STATUS_INVALID_PARAMS: i32 = -1;
pub const QT_STATUS_MODE_INVALID: i32 = -2;
pub const QT_STATUS_GENERATE_FAILED: i32 = -3;
pub const QT_STATUS_OOM: i32 = -4;
pub const QT_STATUS_CANCELLED: i32 = -5;

// ---------------------------------------------------------------------------
// Opaque handle — definition lives in qwen.cpp
// ---------------------------------------------------------------------------

/// Opaque context handle. Only used as `*mut qt_context` through FFI.
pub enum qt_context {}

// ---------------------------------------------------------------------------
// qt_audio — output audio buffer (POD struct)
// ---------------------------------------------------------------------------

/// Output audio buffer. `samples` is `malloc`-allocated, owned by the struct,
/// released by `qt_audio_free`. Mono f32 PCM at 24 kHz.
#[repr(C)]
pub struct qt_audio {
    pub samples: *mut f32,
    pub n_samples: i32,
    pub sample_rate: i32,
    pub channels: i32,
}

// ---------------------------------------------------------------------------
// qt_init_params — initialisation parameters
// ---------------------------------------------------------------------------

/// Initialisation parameters. Both GGUF paths are required.
#[repr(C)]
pub struct qt_init_params {
    pub abi_version: i32,
    pub talker_path: *const c_char,
    pub codec_path: *const c_char,
    pub use_fa: bool,
    pub clamp_fp16: bool,
}

// ---------------------------------------------------------------------------
// Callback type aliases — function pointers for cancellation / streaming
// ---------------------------------------------------------------------------

/// Cooperative cancellation callback. Return `true` to abort synthesis.
pub type qt_cancel_cb =
    Option<unsafe extern "C" fn(user_data: *mut c_void) -> bool>;

/// Streaming audio output callback. Return `false` to abort.
pub type qt_audio_chunk_cb =
    Option<unsafe extern "C" fn(samples: *const f32, n_samples: i32, user_data: *mut c_void) -> bool>;

/// Logging callback.
pub type qt_log_cb =
    Option<unsafe extern "C" fn(level: i32, msg: *const c_char, user_data: *mut c_void)>;

// ---------------------------------------------------------------------------
// qt_tts_params — synthesis parameters (see qwen.h for field docs)
// ---------------------------------------------------------------------------

/// Synthesis parameters. Abi-versioned struct.
#[repr(C)]
pub struct qt_tts_params {
    pub abi_version: i32,

    // Required: input text and language
    pub text: *const c_char,
    pub lang: *const c_char,

    // Optional: style instruction and speaker name
    pub instruct: *const c_char,
    pub speaker: *const c_char,

    // Optional voice reference (base mode)
    pub ref_audio_24k: *const f32,
    pub ref_n_samples: i32,
    pub ref_text: *const c_char,

    // Sampling
    pub seed: i64,
    pub max_new_tokens: i32,
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub subtalker_do_sample: bool,
    pub subtalker_temperature: f32,
    pub subtalker_top_k: i32,
    pub subtalker_top_p: f32,

    // Debug
    pub dump_dir: *const c_char,

    // Cancellation
    pub cancel: qt_cancel_cb,
    pub cancel_user_data: *mut c_void,

    // Streaming
    pub on_chunk: qt_audio_chunk_cb,
    pub on_chunk_user_data: *mut c_void,

    // Codec decode framing
    pub codec_chunk_sec: f32,
    pub codec_left_context_sec: f32,

    // ABI v2: pre-encoded latent voice reference
    pub ref_spk_emb: *const f32,
    pub ref_spk_dim: i32,
    pub ref_codes: *const i32,
    pub ref_T: i32,
}

// ---------------------------------------------------------------------------
// FFI function declarations — extern "C" symbols from qwen-core.lib
// ---------------------------------------------------------------------------

extern "C" {
    /// Return version string ("<git-hash> (<date>)").
    pub fn qt_version() -> *const c_char;

    /// Return last error message for the calling thread.
    pub fn qt_last_error() -> *const c_char;

    /// Release a `qt_audio` buffer. Safe on zero-initialised struct.
    pub fn qt_audio_free(a: *mut qt_audio);

    /// Fill `p` with default init params.
    pub fn qt_init_default_params(p: *mut qt_init_params);

    /// Initialise the TTS pipeline. Returns NULL on failure.
    pub fn qt_init(params: *const qt_init_params) -> *mut qt_context;

    /// Release the context and all owned resources. Safe on NULL.
    pub fn qt_free(ctx: *mut qt_context);

    /// Fill `p` with default synthesis params.
    pub fn qt_tts_default_params(p: *mut qt_tts_params);

    /// Run full TTS synthesis. Returns negative status on failure.
    pub fn qt_synthesize(
        ctx: *mut qt_context,
        params: *const qt_tts_params,
        out: *mut qt_audio,
    ) -> i32;

    /// Number of RVQ codebooks in the loaded codec.
    pub fn qt_num_codebooks(ctx: *const qt_context) -> i32;

    /// Convert duration in seconds to frame count at 12.5 Hz.
    pub fn qt_duration_sec_to_tokens(ctx: *const qt_context, duration_sec: f32) -> i32;

    /// Number of named speakers in the loaded model.
    pub fn qt_n_speakers(ctx: *const qt_context) -> i32;

    /// Name of speaker `i` (NULL if out of range).
    pub fn qt_speaker_name(ctx: *const qt_context, i: i32) -> *const c_char;
}
