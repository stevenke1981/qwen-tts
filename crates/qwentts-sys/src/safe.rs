//! Safe Rust wrapper around the raw FFI functions.
//!
//! [`QwenTts`] manages initialisation, synthesis, and resource cleanup
//! without exposing `unsafe` to the caller.
//!
//! # Example (non-streaming)
//!
//! ```rust,no_run
//! use qwen_tts_sys::safe::QwenTts;
//!
//! let mut init = QwenTts::init_params();
//! init.talker_path = std::ffi::CString::new("model.gguf").unwrap().into_raw();
//! init.codec_path = std::ffi::CString::new("codec.gguf").unwrap().into_raw();
//!
//! let tts = QwenTts::new(&init).expect("qt_init failed");
//!
//! let mut params = QwenTts::tts_params();
//! params.text = std::ffi::CString::new("Hello world").unwrap().into_raw();
//!
//! let audio = tts.synthesize(&params).expect("synthesis failed");
//! println!("Got {} audio samples", audio.len());
//! ```

use std::ffi::CStr;
use std::ptr::NonNull;

use crate::raw::{
    self, qt_audio, qt_init_params, qt_status, qt_tts_params, qt_context,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error wrapping the failed status code and the last error message.
#[derive(Debug, Clone)]
pub struct Error {
    pub status: qt_status,
    pub message: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "qwen error ({:?}): {}", self.status, self.message)
    }
}

impl std::error::Error for Error {}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// Safe handle
// ---------------------------------------------------------------------------

/// Safe handle to an initialised qwentts.cpp context.
///
/// Created with [`QwenTts::new`], released automatically on `Drop`.
pub struct QwenTts {
    inner: NonNull<qt_context>,
}

// `qt_context` is opaque and managed by the C side; the handle is safe
// to send and share as long as the inner pointer is stable (it is — the
// C side never moves it behind our back).
unsafe impl Send for QwenTts {}
unsafe impl Sync for QwenTts {}

impl QwenTts {
    /// Return the default initialisation parameters.
    ///
    /// The caller **must** fill `talker_path` and `codec_path` before
    /// passing to [`QwenTts::new`].
    pub fn init_params() -> qt_init_params {
        let mut p = std::mem::MaybeUninit::<qt_init_params>::zeroed();
        unsafe {
            raw::qt_init_default_params(p.as_mut_ptr());
            p.assume_init()
        }
    }

    /// Return the default synthesis parameters.
    ///
    /// The caller **must** fill `text` (and optionally `lang`, `instruct`,
    /// `speaker`, voice reference fields) before passing to
    /// [`QwenTts::synthesize`].
    pub fn tts_params() -> qt_tts_params {
        let mut p = std::mem::MaybeUninit::<qt_tts_params>::zeroed();
        unsafe {
            raw::qt_tts_default_params(p.as_mut_ptr());
            p.assume_init()
        }
    }

    /// Initialise a new qwentts.cpp context.
    ///
    /// # Safety
    ///
    /// `params.talker_path` and `params.codec_path` must point to valid,
    /// NUL-terminated C strings.  The caller is responsible for keeping
    /// those pointers alive for the duration of the call — the library
    /// copies the paths internally during `qt_init`.
    pub fn new(params: &qt_init_params) -> Result<Self> {
        let ptr = unsafe { raw::qt_init(params as *const qt_init_params) };
        match NonNull::new(ptr) {
            Some(inner) => Ok(Self { inner }),
            None => {
                let msg = last_error();
                Err(Error {
                    status: qt_status::QT_STATUS_INVALID_PARAMS,
                    message: msg,
                })
            }
        }
    }

    /// Run TTS synthesis in buffered mode.
    ///
    /// Returns the mono float PCM audio samples at 24 kHz.
    ///
    /// # Safety
    ///
    /// String pointer fields in `params` (`text`, `lang`, etc.) must
    /// point to valid NUL-terminated C strings that stay alive for the
    /// duration of the call.
    ///
    /// When `params.on_chunk` is non-`None`, the streaming path is used
    /// and the returned vector is empty on success (audio is delivered
    /// through the callback instead).  For simple buffered usage, leave
    /// `params.on_chunk` as `None`.
    pub unsafe fn synthesize(
        &self,
        params: &qt_tts_params,
    ) -> Result<Vec<f32>> {
        let mut out = std::mem::MaybeUninit::<qt_audio>::zeroed();
        let status =
            raw::qt_synthesize(self.inner.as_ptr(), params, out.as_mut_ptr());

        if status != qt_status::QT_STATUS_OK {
            let msg = last_error();
            return Err(Error {
                status,
                message: msg,
            });
        }

        let mut audio = out.assume_init();
        let samples = if audio.n_samples > 0 && !audio.samples.is_null() {
            let slice =
                std::slice::from_raw_parts(audio.samples, audio.n_samples as usize);
            let v = slice.to_vec();
            v
        } else {
            Vec::new()
        };

        // Free the C-allocated buffer (safe because we have copied the
        // data into `v`).  `qt_audio` is a POD struct so Rust's Drop
        // does nothing; we still need to tell the C side to free the
        // malloc'd samples pointer.
        unsafe { raw::qt_audio_free(&mut audio) };

        Ok(samples)
    }

    // ── Info queries ────────────────────────────────────────────────────

    /// Number of RVQ codebooks of the loaded codec.
    pub fn num_codebooks(&self) -> i32 {
        unsafe { raw::qt_num_codebooks(self.inner.as_ptr()) }
    }

    /// Number of named speakers in the loaded model.
    pub fn n_speakers(&self) -> i32 {
        unsafe { raw::qt_n_speakers(self.inner.as_ptr()) }
    }

    /// Name of speaker `i` (valid for `0 .. n_speakers()`).
    pub fn speaker_name(&self, i: i32) -> Option<&str> {
        let ptr = unsafe { raw::qt_speaker_name(self.inner.as_ptr(), i) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { c_str_to_str(ptr) })
        }
    }

    /// Convert a duration in seconds to a codec frame count.
    pub fn duration_sec_to_tokens(&self, secs: f32) -> i32 {
        unsafe { raw::qt_duration_sec_to_tokens(self.inner.as_ptr(), secs) }
    }
}

impl Drop for QwenTts {
    fn drop(&mut self) {
        unsafe { raw::qt_free(self.inner.as_ptr()) };
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read the thread-local last error message.
fn last_error() -> String {
    unsafe {
        let ptr = raw::qt_last_error();
        if ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// Safety: `ptr` must be a valid NUL-terminated UTF-8 C string.
unsafe fn c_str_to_str<'a>(ptr: *const std::ffi::c_char) -> &'a str {
    CStr::from_ptr(ptr).to_str().unwrap_or("(invalid utf-8)")
}
