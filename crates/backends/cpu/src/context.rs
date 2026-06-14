//! Safe Rust wrapper around the qwentts.cpp `qt_context` lifecycle.
//!
//! `NativeContext` owns a raw `*mut qt_context` pointer and ensures
//! `qt_free` is called on drop. Construction loads both GGUF files and
//! initialises the GGML backend pair. `synthesize` runs the full TTS
//! pipeline and returns mono f32 PCM samples at 24 kHz.

use crate::ffi::{
    qt_audio, qt_audio_free, qt_context, qt_free, qt_init, qt_init_default_params, qt_init_params,
    qt_last_error, qt_synthesize, qt_tts_params, QT_ABI_VERSION, QT_STATUS_OK,
};
use std::ffi::CString;
use std::path::Path;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Opaque error wrapping a diagnostic from `qt_last_error`.
#[derive(Debug)]
pub struct NativeError(pub String);

impl std::fmt::Display for NativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "qwentts native error: {}", self.0)
    }
}

impl std::error::Error for NativeError {}

// ---------------------------------------------------------------------------
// NativeContext
// ---------------------------------------------------------------------------

/// Owns a `qt_context` handle with RAII semantics.
///
/// The handle is lazily initialised by `new` and released in `Drop`.
/// `synthesize` is single-threaded for one context (the internal pipeline
/// is not re-entrant), so callers should serialise access.
pub struct NativeContext {
    inner: *mut qt_context,
}

// The C++ backend is safe to call from different threads as long as calls
// are serialised per context. The Mutex in CpuBackend provides this.
unsafe impl Send for NativeContext {}
unsafe impl Sync for NativeContext {}

impl NativeContext {
    /// Create a new context, loading talker and codec GGUF files.
    ///
    /// # Errors
    ///
    /// Returns `NativeError` if either GGUF path is invalid, the architecture
    /// doesn't match `qwen3-tts`, or the backend initialisation fails.
    pub fn new(talker_path: &Path, codec_path: &Path) -> Result<Self, NativeError> {
        let talker_cstr = Self::path_to_cstr(talker_path, "talker")?;
        let codec_cstr = Self::path_to_cstr(codec_path, "codec")?;

        let mut init_params = qt_init_params {
            abi_version: QT_ABI_VERSION,
            talker_path: talker_cstr.as_ptr(),
            codec_path: codec_cstr.as_ptr(),
            use_fa: false, // CPU only; flash attention is GPU-only
            clamp_fp16: false,
        };

        unsafe {
            qt_init_default_params(&mut init_params);
            // Override fields we set manually
            init_params.talker_path = talker_cstr.as_ptr();
            init_params.codec_path = codec_cstr.as_ptr();
            init_params.use_fa = false;

            let ctx = qt_init(&init_params);
            if ctx.is_null() {
                return Err(NativeError(Self::last_error()));
            }
            Ok(Self { inner: ctx })
        }
    }

    /// Run full TTS synthesis. Returns mono f32 PCM samples at 24 kHz.
    ///
    /// # Errors
    ///
    /// Returns `NativeError` on invalid params, mode mismatch, OOM,
    /// cancellation, or internal pipeline failure.
    pub fn synthesize(
        &self,
        text: &str,
        language: &str,
        speaker: Option<&str>,
    ) -> Result<Vec<f32>, NativeError> {
        let text_cstr = CString::new(text)
            .map_err(|e| NativeError(format!("null byte in text: {e}")))?;
        let lang_cstr = CString::new(language)
            .map_err(|e| NativeError(format!("null byte in language: {e}")))?;
        let speaker_cstr = speaker
            .map(|s| {
                CString::new(s)
                    .map_err(|e| NativeError(format!("null byte in speaker: {e}")))
            })
            .transpose()?;

        let params = qt_tts_params {
            abi_version: QT_ABI_VERSION,
            text: text_cstr.as_ptr(),
            lang: lang_cstr.as_ptr(),
            instruct: std::ptr::null(),
            speaker: speaker_cstr
                .as_ref()
                .map_or(std::ptr::null(), |c| c.as_ptr()),
            ref_audio_24k: std::ptr::null(),
            ref_n_samples: 0,
            ref_text: std::ptr::null(),
            seed: -1,
            max_new_tokens: 2048,
            do_sample: true,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            subtalker_do_sample: true,
            subtalker_temperature: 0.9,
            subtalker_top_k: 50,
            subtalker_top_p: 1.0,
            dump_dir: std::ptr::null(),
            cancel: None,
            cancel_user_data: std::ptr::null_mut(),
            on_chunk: None,
            on_chunk_user_data: std::ptr::null_mut(),
            codec_chunk_sec: 24.0,
            codec_left_context_sec: 2.0,
            ref_spk_emb: std::ptr::null(),
            ref_spk_dim: 0,
            ref_codes: std::ptr::null(),
            ref_T: 0,
        };

        let mut audio = qt_audio {
            samples: std::ptr::null_mut(),
            n_samples: 0,
            sample_rate: 0,
            channels: 0,
        };

        unsafe {
            let status = qt_synthesize(self.inner, &params, &mut audio);

            if status != QT_STATUS_OK {
                let err = Self::last_error();
                // Free any partial output
                if !audio.samples.is_null() {
                    qt_audio_free(&mut audio);
                }
                return Err(NativeError(err));
            }

            if audio.samples.is_null() || audio.n_samples <= 0 {
                return Err(NativeError("synthesis returned empty audio".into()));
            }

            // Copy f32 samples from C heap to Rust Vec
            #[allow(clippy::cast_sign_loss)]
            let n = audio.n_samples as usize;
            let mut samples = Vec::with_capacity(n);
            std::ptr::copy_nonoverlapping(audio.samples, samples.as_mut_ptr(), n);
            samples.set_len(n);

            // Free the C allocation
            qt_audio_free(&mut audio);

            Ok(samples)
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Convert a `Path` to a `CString` for FFI.
    fn path_to_cstr(path: &Path, label: &str) -> Result<CString, NativeError> {
        let s = path
            .to_str()
            .ok_or_else(|| NativeError(format!("{label} path contains invalid UTF-8")))?;
        CString::new(s)
            .map_err(|e| NativeError(format!("null byte in {label} path: {e}")))
    }

    /// Retrieve the last error message from the C library.
    fn last_error() -> String {
        unsafe {
            let ptr = qt_last_error();
            if ptr.is_null() {
                return "unknown error".into();
            }
            std::ffi::CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    }
}

impl Drop for NativeContext {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                qt_free(self.inner);
            }
            self.inner = std::ptr::null_mut();
        }
    }
}
