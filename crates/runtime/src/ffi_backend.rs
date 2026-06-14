//! In-process FFI backend using the `qwen-tts-sys` crate.
//!
//! [`FfiBackend`] replaces `ExternalQwenTtsBackend` by calling the shared
//! library directly instead of spawning a subprocess.  This avoids
//! process overhead, pipes, and filesystem I/O for the audio stream.

use crate::{
    BackendError, BackendResult, DeviceKind, RuntimeBackend, SynthesisRequest, SynthesisResponse,
};
use qwen_tts_core::{validate_wav_file, AudioSpec};
use qwen_tts_sys::safe::QwenTts;
use std::{ffi::CString, fs, path::PathBuf};
use tracing::{info, instrument};

/// In-process FFI backend.
///
/// Holds the model file paths and device kind.  A new `qt_context` is
/// created for every `synthesize` call so the C library's internal state
/// stays fresh.
#[derive(Debug, Clone)]
pub struct FfiBackend {
    /// Path to the talker GGUF model.
    pub talker_path: PathBuf,
    /// Path to the codec GGUF model.
    pub codec_path: PathBuf,
    /// Device hint (CPU / CUDA / …).
    pub device: DeviceKind,
}

impl FfiBackend {
    #[must_use]
    pub fn new(
        talker_path: impl Into<PathBuf>,
        codec_path: impl Into<PathBuf>,
        device: DeviceKind,
    ) -> Self {
        Self {
            talker_path: talker_path.into(),
            codec_path: codec_path.into(),
            device,
        }
    }
}

impl RuntimeBackend for FfiBackend {
    fn name(&self) -> &'static str {
        "qwentts.cpp-ffi"
    }

    fn device_kind(&self) -> DeviceKind {
        self.device
    }

    fn is_available(&self) -> bool {
        self.talker_path.exists() && self.codec_path.exists()
    }

    #[instrument(skip(self, request), fields(backend = self.name(), text_len = request.text.chars().count()))]
    fn synthesize(&self, request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        // ── Validate ────────────────────────────────────────────────────
        if request.text.trim().is_empty() {
            return Err(BackendError::InvalidRequest("text cannot be empty".into()));
        }
        if !self.is_available() {
            return Err(BackendError::Unavailable(format!(
                "model files not found: talker={}, codec={}",
                self.talker_path.display(),
                self.codec_path.display()
            )));
        }

        // ── Prepare C strings ───────────────────────────────────────────
        let talker_cstr =
            CString::new(self.talker_path.as_os_str().as_encoded_bytes())
                .map_err(|_| BackendError::InvalidRequest("talker path contains NUL".into()))?;
        let codec_cstr =
            CString::new(self.codec_path.as_os_str().as_encoded_bytes())
                .map_err(|_| BackendError::InvalidRequest("codec path contains NUL".into()))?;
        let text_cstr =
            CString::new(request.text.as_str())
                .map_err(|_| BackendError::InvalidRequest("text contains NUL".into()))?;
        let lang_cstr =
            CString::new(request.language.as_str())
                .map_err(|_| BackendError::InvalidRequest("language contains NUL".into()))?;
        let speaker_cstr = match &request.speaker {
            Some(s) => Some(
                CString::new(s.as_str())
                    .map_err(|_| BackendError::InvalidRequest("speaker contains NUL".into()))?,
            ),
            None => None,
        };

        // ── Initialise context ──────────────────────────────────────────
        let mut init = QwenTts::init_params();
        init.talker_path = talker_cstr.as_ptr();
        init.codec_path = codec_cstr.as_ptr();

        let tts = QwenTts::new(&init).map_err(|e| {
            BackendError::Unavailable(format!(
                "qt_init failed (talker={}, codec={}): {e}",
                self.talker_path.display(),
                self.codec_path.display()
            ))
        })?;

        // ── Run synthesis (buffered) ────────────────────────────────────
        let mut params = QwenTts::tts_params();
        params.text = text_cstr.as_ptr();
        params.lang = lang_cstr.as_ptr();
        params.speaker = speaker_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());

        let samples = unsafe {
            // Safety: the C string pointers above stay valid for the
            // duration of the call.
            tts.synthesize(&params)
        }
        .map_err(|e| {
            BackendError::CommandFailed {
                program: "qt_synthesize (FFI)".into(),
                status: Some(e.status as i32),
                stderr: e.message,
            }
        })?;

        if samples.is_empty() {
            return Err(BackendError::InvalidRequest(
                "synthesis returned empty audio".into(),
            ));
        }

        // ── Write WAV file ─────────────────────────────────────────────
        if let Some(parent) = request
            .out_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }

        write_wav_f32(&request.out_path, 24000, &samples)?;

        // ── Validate & respond ──────────────────────────────────────────
        let metadata = validate_wav_file(&request.out_path, AudioSpec::default())?;

        info!(
            backend = self.name(),
            output = %request.out_path.display(),
            sample_rate_hz = 24000_u32,
            channels = 1_u16,
            "FFI synthesis finished"
        );

        Ok(SynthesisResponse {
            wav_path: request.out_path.clone(),
            sample_rate_hz: metadata.sample_rate_hz,
            channels: metadata.channels,
            bits_per_sample: metadata.bits_per_sample,
            data_size_bytes: metadata.data_size_bytes,
            backend_name: self.name().to_owned(),
        })
    }
}

// ---------------------------------------------------------------------------
// WAV writer (mono f32 → 16-bit PCM)
// ---------------------------------------------------------------------------

/// Write a mono f32 PCM buffer as a 16-bit WAV file.
fn write_wav_f32(path: &PathBuf, sample_rate: u32, samples: &[f32]) -> BackendResult<()> {
    use std::io::Write;

    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample / 8);
    let block_align = num_channels * (bits_per_sample / 8);
    let data_size = u32::try_from(samples.len())
        .map_err(|_| BackendError::InvalidRequest("audio too long for WAV".into()))?
        .wrapping_mul(u32::from(block_align));

    let file = fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);

    // RIFF header
    writer.write_all(b"RIFF")?;
    writer.write_all(&(36 + data_size).to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt chunk
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;  // chunk size
    writer.write_all(&1u16.to_le_bytes())?;   // PCM
    writer.write_all(&num_channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bits_per_sample.to_le_bytes())?;

    // data chunk
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;

    // Convert f32 [-1, 1] → i16
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample_i16 = (clamped * i16::MAX as f32) as i16;
        writer.write_all(&sample_i16.to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}
