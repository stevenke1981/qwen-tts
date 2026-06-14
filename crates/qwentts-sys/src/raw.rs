//! Raw FFI function declarations from `qwen.h`.
//!
//! All functions are `unsafe` and map 1:1 to the C ABI exported by
//! `qwentts.cpp/src/qwen.cpp`.  The shared library (`qwen.dll` /
//! `libqwen.so` / `libqwen.dylib`) is linked by `build.rs`.
//!
//! See `vendor/qwentts.cpp/src/qwen.h` (v2 ABI) for the authoritative
//! declaration.

// Allow C-style naming — this is an FFI binding that must match qwen.h
// exactly.  Rust convention warns about every non-CamelCase name.
#![allow(non_camel_case_types, non_snake_case)]

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use std::ffi::{c_char, c_float, c_int, c_void};

// ---------------------------------------------------------------------------
// ABI version constant
// ---------------------------------------------------------------------------

/// Struct ABI version.  Every public POD struct carries this at offset 0.
/// The library rejects structs whose `abi_version` exceeds the build-time
/// `QT_ABI_VERSION`.
pub const QT_ABI_VERSION: i32 = 2;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Status code returned by every fallible entry.  `QT_STATUS_OK` is always
/// zero so `if rc != 0` reads as failure.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum qt_status {
    QT_STATUS_OK = 0,
    QT_STATUS_INVALID_PARAMS = -1,
    QT_STATUS_MODE_INVALID = -2,
    QT_STATUS_GENERATE_FAILED = -3,
    QT_STATUS_OOM = -4,
    QT_STATUS_CANCELLED = -5,
}

/// Log severity for the global log callback.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum qt_log_level {
    QT_LOG_DEBUG = 0,
    QT_LOG_INFO = 1,
    QT_LOG_WARN = 2,
    QT_LOG_ERROR = 3,
}

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaq ue context handle.  Defined in C++ as `struct qt_context`.
/// Never inspect the internals; always pass through the FFI functions.
pub enum qt_context {}

// ---------------------------------------------------------------------------
// POD structs
// ---------------------------------------------------------------------------

/// Output audio buffer.
///
/// The `samples` pointer is malloc-allocated by `qt_synthesize`, owned by
/// the struct, and must be released with `qt_audio_free`.  Do **not** free
/// samples directly.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct qt_audio {
    pub samples: *mut c_float,   // mono PCM, malloc allocated
    pub n_samples: c_int,        // length in samples
    pub sample_rate: c_int,      // 24000 (codec rate)
    pub channels: c_int,         // 1 (mono)
}

/// Initialisation parameters.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct qt_init_params {
    pub abi_version: c_int,
    pub talker_path: *const c_char,
    pub codec_path: *const c_char,
    pub use_fa: bool,
    pub clamp_fp16: bool,
}

// ---------------------------------------------------------------------------
// Callback type aliases
// ---------------------------------------------------------------------------

/// Cooperative cancellation callback.  Return `true` to request abort.
pub type qt_cancel_cb =
    Option<unsafe extern "C" fn(user_data: *mut c_void) -> bool>;

/// Streaming output callback.  Return `false` to abort (equivalent to
/// `qt_cancel_cb` returning `true`).
pub type qt_audio_chunk_cb = Option<
    unsafe extern "C" fn(
        samples: *const c_float,
        n_samples: c_int,
        user_data: *mut c_void,
    ) -> bool,
>;

/// Logging callback.  `msg` is a NUL-terminated UTF-8 string with no
/// trailing newline.
pub type qt_log_cb = Option<
    unsafe extern "C" fn(
        level: qt_log_level,
        msg: *const c_char,
        user_data: *mut c_void,
    ),
>;

/// Synthesis parameters.
///
/// Every string field is NUL-terminated UTF-8; `NULL` maps to empty/default.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct qt_tts_params {
    pub abi_version: c_int,

    // Input text and language hint
    pub text: *const c_char,
    pub lang: *const c_char,
    pub instruct: *const c_char,
    pub speaker: *const c_char,

    // Voice reference (base mode voice cloning)
    pub ref_audio_24k: *const c_float,
    pub ref_n_samples: c_int,
    pub ref_text: *const c_char,

    // Sampling configuration
    pub seed: i64,
    pub max_new_tokens: c_int,
    pub do_sample: bool,
    pub temperature: c_float,
    pub top_k: c_int,
    pub top_p: c_float,
    pub repetition_penalty: c_float,
    pub subtalker_do_sample: bool,
    pub subtalker_temperature: c_float,
    pub subtalker_top_k: c_int,
    pub subtalker_top_p: c_float,

    // Debug dump directory
    pub dump_dir: *const c_char,

    // Cooperative cancellation
    pub cancel: qt_cancel_cb,
    pub cancel_user_data: *mut c_void,

    // Streaming output
    pub on_chunk: qt_audio_chunk_cb,
    pub on_chunk_user_data: *mut c_void,

    // Codec decode framing
    pub codec_chunk_sec: c_float,
    pub codec_left_context_sec: c_float,

    // ABI v2: pre-encoded voice reference
    pub ref_spk_emb: *const c_float,
    pub ref_spk_dim: c_int,
    pub ref_codes: *const i32, // int32_t row-major [num_codebooks, ref_T]
    pub ref_T: c_int,
}

// ---------------------------------------------------------------------------
// FFI function declarations
// ---------------------------------------------------------------------------

extern "C" {
    #[link_name = "qt_version"]
    pub fn qt_version() -> *const c_char;

    #[link_name = "qt_last_error"]
    pub fn qt_last_error() -> *const c_char;

    #[link_name = "qt_audio_free"]
    pub fn qt_audio_free(a: *mut qt_audio);

    #[link_name = "qt_init_default_params"]
    pub fn qt_init_default_params(p: *mut qt_init_params);

    #[link_name = "qt_init"]
    pub fn qt_init(params: *const qt_init_params) -> *mut qt_context;

    #[link_name = "qt_free"]
    pub fn qt_free(q: *mut qt_context);

    #[link_name = "qt_log_set"]
    pub fn qt_log_set(cb: qt_log_cb, user_data: *mut c_void);

    #[link_name = "qt_tts_default_params"]
    pub fn qt_tts_default_params(p: *mut qt_tts_params);

    #[link_name = "qt_num_codebooks"]
    pub fn qt_num_codebooks(q: *const qt_context) -> c_int;

    #[link_name = "qt_synthesize"]
    pub fn qt_synthesize(
        q: *mut qt_context,
        params: *const qt_tts_params,
        out: *mut qt_audio,
    ) -> qt_status;

    #[link_name = "qt_duration_sec_to_tokens"]
    pub fn qt_duration_sec_to_tokens(
        q: *const qt_context,
        duration_sec: c_float,
    ) -> c_int;

    #[link_name = "qt_n_speakers"]
    pub fn qt_n_speakers(q: *const qt_context) -> c_int;

    #[link_name = "qt_speaker_name"]
    pub fn qt_speaker_name(
        q: *const qt_context,
        i: c_int,
    ) -> *const c_char;
}
