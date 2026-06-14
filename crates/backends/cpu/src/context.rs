//! Safe Rust wrapper around the qwentts.cpp `qt_context` lifecycle.
//!
//! `NativeContext` owns a raw `*mut qt_context` pointer and ensures
//! `qt_free` is called on drop. Construction loads both GGUF files and
//! initialises the GGML backend pair. `synthesize` runs the full TTS
//! pipeline and returns mono f32 PCM samples at 24 kHz.

use crate::ffi::{
    qt_audio, qt_audio_free, qt_context, qt_free, qt_init, qt_init_default_params, qt_last_error,
    qt_synthesize, qt_tts_default_params, qt_tts_params, qt_version, QT_ABI_VERSION, QT_STATUS_OK,
};
use qwen_tts_runtime::SynthesisRequest;
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

        unsafe {
            // Log library version
            let ver = qt_version();
            if !ver.is_null() {
                let ver_str = std::ffi::CStr::from_ptr(ver).to_string_lossy();
                eprintln!("[qwen-tts-cpu] qt_version: {ver_str}");
            }

            eprintln!(
                "[qwen-tts-cpu] loading talker: {}",
                talker_path.display()
            );
            eprintln!(
                "[qwen-tts-cpu] loading codec:  {}",
                codec_path.display()
            );

            // Use default params then override paths and CPU-specific flags
            let mut init_params = std::mem::zeroed();
            qt_init_default_params(&mut init_params);
            init_params.talker_path = talker_cstr.as_ptr();
            init_params.codec_path = codec_cstr.as_ptr();
            init_params.use_fa = false; // CPU always uses F32 chain
            init_params.clamp_fp16 = false;

            let ctx = qt_init(&init_params);
            if ctx.is_null() {
                let err = Self::last_error();
                eprintln!("[qwen-tts-cpu] qt_init FAILED: {err}");
                return Err(NativeError(err));
            }

            eprintln!("[qwen-tts-cpu] context initialised successfully");
            Ok(Self { inner: ctx })
        }
    }

    /// Run full TTS synthesis. Returns mono f32 PCM samples at 24 kHz.
    ///
    /// # Errors
    ///
    /// Returns `NativeError` on invalid params, mode mismatch, OOM,
    /// cancellation, or internal pipeline failure.
    pub fn synthesize(&self, req: &SynthesisRequest) -> Result<Vec<f32>, NativeError> {
        // ── Prepare C strings ────────────────────────────────────────────
        let text_cstr = CString::new(req.text.as_str())
            .map_err(|e| NativeError(format!("null byte in text: {e}")))?;
        let lang_cstr = CString::new(req.language.as_str())
            .map_err(|e| NativeError(format!("null byte in language: {e}")))?;
        let speaker_cstr = match &req.speaker {
            Some(s) => Some(
                CString::new(s.as_str())
                    .map_err(|e| NativeError(format!("null byte in speaker: {e}")))?,
            ),
            None => None,
        };
        let instruct_cstr = match &req.instruct {
            Some(s) => Some(
                CString::new(s.as_str())
                    .map_err(|e| NativeError(format!("null byte in instruct: {e}")))?,
            ),
            None => None,
        };
        let ref_audio_f32: Option<Vec<f32>> = match &req.ref_audio_path {
            Some(path) => {
                eprintln!("[qwen-tts-cpu] loading ref audio: {}", path.display());
                Some(read_wav_f32_mono(path)?)
            }
            None => None,
        };
        let ref_text_cstr = match &req.ref_text {
            Some(s) => Some(
                CString::new(s.as_str())
                    .map_err(|e| NativeError(format!("null byte in ref_text: {e}")))?,
            ),
            None => None,
        };

        eprintln!(
            "[qwen-tts-cpu] synthesize: lang={} speaker={} instruct={} text_len={} ref={}",
            req.language,
            req.speaker.as_deref().unwrap_or("(none)"),
            req.instruct.as_deref().unwrap_or("(none)"),
            req.text.len(),
            req.ref_audio_path
                .as_ref()
                .map_or("none", |p| p.to_str().unwrap_or("?")),
        );

        // ── Build params ─────────────────────────────────────────────────
        let mut params: qt_tts_params;
        unsafe {
            params = std::mem::zeroed();
            qt_tts_default_params(&mut params);
        }
        params.abi_version = QT_ABI_VERSION;
        params.text = text_cstr.as_ptr();
        params.lang = lang_cstr.as_ptr();
        params.speaker = speaker_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        params.instruct = instruct_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());

        // Sampling params (None = keep C defaults)
        if let Some(seed) = req.seed {
            params.seed = seed;
        }
        if let Some(tokens) = req.max_new_tokens {
            params.max_new_tokens = tokens;
        }
        if let Some(temp) = req.temperature {
            params.temperature = temp;
        }
        if let Some(k) = req.top_k {
            params.top_k = k;
        }
        if let Some(p) = req.top_p {
            params.top_p = p;
        }
        if let Some(rp) = req.repetition_penalty {
            params.repetition_penalty = rp;
        }
        if let Some(sample) = req.do_sample {
            params.do_sample = sample;
        }

        // Voice reference (voice cloning via ref_audio + ref_text)
        params.ref_audio_24k = ref_audio_f32.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        params.ref_n_samples = ref_audio_f32.as_ref().map_or(0, |v| v.len() as i32);
        params.ref_text = ref_text_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());

        // ── Run synthesis ────────────────────────────────────────────────
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
                eprintln!("[qwen-tts-cpu] synthesize FAILED (status={status}): {err}");
                if !audio.samples.is_null() {
                    qt_audio_free(&mut audio);
                }
                return Err(NativeError(err));
            }

            if audio.samples.is_null() || audio.n_samples <= 0 {
                eprintln!(
                    "[qwen-tts-cpu] synthesize returned empty audio (samples={:?}, n_samples={})",
                    audio.samples, audio.n_samples
                );
                return Err(NativeError("synthesis returned empty audio".into()));
            }

            #[allow(clippy::cast_sign_loss)]
            let n = audio.n_samples as usize;
            eprintln!(
                "[qwen-tts-cpu] output: n_samples={n} sample_rate={} channels={}",
                audio.sample_rate, audio.channels
            );

            let mut samples = Vec::with_capacity(n);
            std::ptr::copy_nonoverlapping(audio.samples, samples.as_mut_ptr(), n);
            samples.set_len(n);

            let (min, max) = samples
                .iter()
                .fold((f32::MAX, f32::MIN), |(mn, mx), &s| (mn.min(s), mx.max(s)));
            let mean = samples.iter().sum::<f32>() / n.max(1) as f32;
            let first_10: Vec<f32> = samples.iter().take(10).copied().collect();
            eprintln!(
                "[qwen-tts-cpu] sample stats: min={min:.4} max={max:.4} mean={mean:.4} first_10={first_10:?}"
            );

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

// ---------------------------------------------------------------------------
// read_wav_f32_mono — hand-parse 16-bit PCM WAV → Vec<f32> (no dep)
// ---------------------------------------------------------------------------

/// Parse a WAV file containing 16-bit signed PCM mono audio into f32
/// samples normalized to [-1.0, 1.0].
///
/// # Errors
///
/// Returns `NativeError` if the file cannot be read, is not a valid WAV,
/// or has an unsupported format (not 16-bit PCM mono).
pub fn read_wav_f32_mono(path: &Path) -> Result<Vec<f32>, NativeError> {
    let raw = std::fs::read(path)
        .map_err(|e| NativeError(format!("failed to read WAV `{}`: {e}", path.display())))?;

    if raw.len() < 44 {
        return Err(NativeError(format!(
            "WAV `{}` too short ({} bytes, need ≥ 44)",
            path.display(),
            raw.len()
        )));
    }
    if &raw[..4] != b"RIFF" || &raw[8..12] != b"WAVE" {
        return Err(NativeError(format!(
            "`{}` is not a WAV file (RIFF/WAVE header missing)",
            path.display()
        )));
    }

    let num_channels = u16::from_le_bytes([raw[22], raw[23]]);
    let sample_rate = u32::from_le_bytes([raw[24], raw[25], raw[26], raw[27]]);
    let bits_per_sample = u16::from_le_bytes([raw[34], raw[35]]);

    if num_channels != 1 {
        return Err(NativeError(format!(
            "WAV `{}` has {num_channels} channels, expected 1 (mono)",
            path.display()
        )));
    }
    if bits_per_sample != 16 {
        return Err(NativeError(format!(
            "WAV `{}` has {bits_per_sample} bits/sample, expected 16",
            path.display()
        )));
    }

    eprintln!(
        "[qwen-tts-cpu] read_wav_f32_mono: {} channels={num_channels} rate={sample_rate} bits={bits_per_sample}",
        path.display()
    );

    // Locate "data" chunk beyond the 44-byte fixed header
    let data_start = raw[44..]
        .windows(8)
        .position(|win| &win[..4] == b"data")
        .map(|pos| pos + 44);

    let data_offset = match data_start {
        Some(offset) => {
            let chunk_size =
                u32::from_le_bytes([raw[offset + 4], raw[offset + 5], raw[offset + 6], raw[offset + 7]]);
            offset + 8 // skip past "data" + size
                + if chunk_size as usize <= raw.len().saturating_sub(offset + 8) {
                    0
                } else {
                    0 // trust the RIFF size over chunk size
                }
        }
        None => 44, // assume data starts at offset 44 (common)
    };

    // Use remaining bytes from data_offset to end of file
    let pcm_data = &raw[data_offset..];
    let n_samp = pcm_data.len() / 2;

    let inv_32768 = 1.0 / 32768.0;
    let mut samples = Vec::with_capacity(n_samp);
    for chunk in pcm_data.chunks_exact(2) {
        let i16_val = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(f32::from(i16_val) * inv_32768);
    }

    eprintln!(
        "[qwen-tts-cpu] read_wav_f32_mono: decoded {} samples @ {} Hz",
        samples.len(),
        sample_rate
    );
    Ok(samples)
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
