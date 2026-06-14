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
use std::{ffi::CString, fs, path::{Path, PathBuf}};
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
    /// Enable Flash Attention (GPU backends only).
    pub use_flash_attn: bool,
    /// Clamp fp16 values to avoid NaN artifacts.
    pub clamp_fp16: bool,
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
            use_flash_attn: false,
            clamp_fp16: false,
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
        let instruct_cstr = match &request.instruct {
            Some(s) => Some(
                CString::new(s.as_str())
                    .map_err(|_| BackendError::InvalidRequest("instruct contains NUL".into()))?,
            ),
            None => None,
        };
        let ref_audio_samples: Option<Vec<f32>> = match &request.ref_audio_path {
            Some(path) => Some(read_wav_f32_mono(path)?),
            None => None,
        };
        let ref_text_cstr = match &request.ref_text {
            Some(s) => Some(
                CString::new(s.as_str())
                    .map_err(|_| BackendError::InvalidRequest("ref_text contains NUL".into()))?,
            ),
            None => None,
        };

        // ── Initialise context ──────────────────────────────────────────
        let mut init = QwenTts::init_params();
        init.talker_path = talker_cstr.as_ptr();
        init.codec_path = codec_cstr.as_ptr();
        init.use_fa = self.use_flash_attn;
        init.clamp_fp16 = self.clamp_fp16;

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
        params.instruct = instruct_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());

        // Sampling params (None = keep C defaults)
        if let Some(seed) = request.seed {
            params.seed = seed;
        }
        if let Some(tokens) = request.max_new_tokens {
            params.max_new_tokens = tokens;
        }
        if let Some(temp) = request.temperature {
            params.temperature = temp;
        }
        if let Some(k) = request.top_k {
            params.top_k = k;
        }
        if let Some(p) = request.top_p {
            params.top_p = p;
        }
        if let Some(rp) = request.repetition_penalty {
            params.repetition_penalty = rp;
        }
        if let Some(sample) = request.do_sample {
            params.do_sample = sample;
        }

        // Voice reference (voice cloning)
        params.ref_audio_24k = ref_audio_samples.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        params.ref_n_samples = ref_audio_samples.as_ref().map_or(0, |v| v.len() as i32);
        params.ref_text = ref_text_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());

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

/// Read a 16-bit PCM mono WAV file into f32 samples normalized to [-1, 1].
///
/// Only supports the subset needed for voice reference: 16-bit PCM, 1 channel.
fn read_wav_f32_mono(path: &Path) -> BackendResult<Vec<f32>> {
    let data = fs::read(path).map_err(|e| {
        BackendError::InvalidRequest(format!("cannot read ref audio WAV: {e}"))
    })?;

    if data.len() < 44 {
        return Err(BackendError::InvalidRequest(
            "ref audio WAV too small".into(),
        ));
    }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(BackendError::InvalidRequest(
            "ref audio: not a WAV file".into(),
        ));
    }

    // Parse chunks
    let mut offset = 12;
    let mut audio_format = None;
    let mut num_channels = None;
    let mut bits_per_sample = None;
    let mut data_offset = None;
    let mut data_size = None;

    loop {
        if offset + 8 > data.len() {
            return Err(BackendError::InvalidRequest(
                "ref audio: truncated WAV header".into(),
            ));
        }
        let chunk_id = &data[offset..offset + 4];
        let chunk_size =
            u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        match chunk_id {
            b"fmt " => {
                if offset + 16 > data.len() {
                    return Err(BackendError::InvalidRequest(
                        "ref audio: truncated fmt chunk".into(),
                    ));
                }
                audio_format =
                    Some(u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()));
                num_channels =
                    Some(u16::from_le_bytes(data[offset + 2..offset + 4].try_into().unwrap()));
                bits_per_sample =
                    Some(u16::from_le_bytes(data[offset + 14..offset + 16].try_into().unwrap()));
            }
            b"data" => {
                data_offset = Some(offset);
                data_size = Some(chunk_size);
            }
            _ => {}
        }
        offset += chunk_size;
        if chunk_id == b"data" {
            break;
        }
        if offset >= data.len() {
            return Err(BackendError::InvalidRequest(
                "ref audio: data chunk not found".into(),
            ));
        }
    }

    let af = audio_format
        .ok_or_else(|| BackendError::InvalidRequest("ref audio: missing fmt chunk".into()))?;
    let ch = num_channels
        .ok_or_else(|| BackendError::InvalidRequest("ref audio: missing channels".into()))?;
    let bps = bits_per_sample
        .ok_or_else(|| BackendError::InvalidRequest("ref audio: missing bits_per_sample".into()))?;
    let d_off = data_offset
        .ok_or_else(|| BackendError::InvalidRequest("ref audio: missing data chunk".into()))?;
    let d_sz = data_size.unwrap_or(0);

    if af != 1 {
        return Err(BackendError::InvalidRequest(format!(
            "ref audio: only PCM supported, got format {af}"
        )));
    }
    if ch != 1 {
        return Err(BackendError::InvalidRequest(format!(
            "ref audio: only mono supported, got {ch} channels"
        )));
    }
    if bps != 16 {
        return Err(BackendError::InvalidRequest(format!(
            "ref audio: only 16-bit PCM supported, got {bps} bits"
        )));
    }

    let sample_bytes = (bps / 8) as usize;
    let total_samples = d_sz / sample_bytes;
    let mut samples = Vec::with_capacity(total_samples);

    for i in 0..total_samples {
        let byte_start = d_off + i * sample_bytes;
        if byte_start + sample_bytes > data.len() {
            break;
        }
        let raw = i16::from_le_bytes(data[byte_start..byte_start + 2].try_into().unwrap());
        samples.push(f32::from(raw) / i16::MAX as f32);
    }

    if samples.is_empty() {
        return Err(BackendError::InvalidRequest(
            "ref audio: no samples found".into(),
        ));
    }

    info!("loaded ref audio: {} samples", samples.len());
    Ok(samples)
}
