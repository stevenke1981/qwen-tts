//! Pure Rust TTS backend using candle for the talker transformer and
//! qwen-tts-codec for the DAC decoder. No C++ FFI required.

pub mod config;
pub mod code_predictor;
pub mod custom_ops;
pub mod pipeline;
pub mod qgemv;
pub mod sampling;
pub mod talker;
pub mod timing;
pub mod tokenizer;

use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Mutex;

use candle_core::Device;
use qwen_tts_runtime::{BackendError, BackendResult, DeviceKind, RuntimeBackend, SynthesisRequest, SynthesisResponse};

use pipeline::Pipeline;

/// The public backend type — implements [`RuntimeBackend`] using pure Rust.
pub struct PureRustBackend {
    talker_path: PathBuf,
    codec_path: PathBuf,
    device: DeviceKind,
    pipeline: Mutex<Option<Pipeline>>,
}

impl PureRustBackend {
    #[must_use]
    pub fn new(talker_path: PathBuf, codec_path: PathBuf, device: DeviceKind) -> Self {
        Self {
            talker_path,
            codec_path,
            device,
            pipeline: Mutex::new(None),
        }
    }

    fn ensure_pipeline(&self) -> BackendResult<()> {
        let mut guard = self
            .pipeline
            .lock()
            .map_err(|e| BackendError::Unavailable(format!("mutex poisoned: {e}")))?;

        if guard.is_some() {
            return Ok(());
        }

        let candle_device = device_to_candle(self.device)?;
        let p = Pipeline::new(&self.talker_path, &self.codec_path, &candle_device)
            .map_err(|e| BackendError::Unavailable(format!("pipeline init failed: {e}")))?;
        *guard = Some(p);
        Ok(())
    }
}

/// Convert a [`DeviceKind`] to a candle [`Device`].
///
/// Supports `Cpu`, `Cuda`, and `Auto` (Auto tries CUDA first, falls back to CPU).
fn device_to_candle(kind: DeviceKind) -> BackendResult<Device> {
    match kind {
        DeviceKind::Cpu => Ok(Device::Cpu),
        DeviceKind::Cuda => Device::new_cuda(0)
            .map_err(|e| BackendError::Unavailable(format!("CUDA device 0 not available: {e}"))),
        DeviceKind::Auto => {
            // Try CUDA first, silently fall back to CPU
            Ok(Device::new_cuda(0).unwrap_or(Device::Cpu))
        }
        other => Err(BackendError::Unavailable(format!(
            "device {other} not supported by pure-rust backend (use cpu, cuda, or auto)"
        ))),
    }
}

impl RuntimeBackend for PureRustBackend {
    fn name(&self) -> &'static str {
        "pure-rust"
    }

    fn device_kind(&self) -> DeviceKind {
        self.device
    }

    fn is_available(&self) -> bool {
        self.talker_path.exists() && self.codec_path.exists()
    }

    fn synthesize(&self, request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        if request.text.trim().is_empty() {
            return Err(BackendError::InvalidRequest("text cannot be empty".into()));
        }

        self.ensure_pipeline()?;

        let mut guard = self
            .pipeline
            .lock()
            .map_err(|e| BackendError::Unavailable(format!("mutex poisoned: {e}")))?;

        let pipeline = guard.as_mut().ok_or_else(|| {
            BackendError::Unavailable("pipeline not initialised".into())
        })?;

        let audio_i16 = pipeline
            .synthesize_simple(request)
            .map_err(|e| BackendError::InvalidRequest(format!("synthesis failed: {e}")))?;

        drop(guard);

        // Ensure output directory exists
        if let Some(parent) = request.out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }

        // Write WAV
        write_pcm16_wav(&request.out_path, &audio_i16)?;

        let data_size = (audio_i16.len() * 2) as u32;

        Ok(SynthesisResponse {
            wav_path: request.out_path.clone(),
            sample_rate_hz: 24000,
            channels: 1,
            bits_per_sample: 16,
            data_size_bytes: data_size,
            backend_name: self.name().to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// WAV helper
// ---------------------------------------------------------------------------

fn write_pcm16_wav(path: &PathBuf, samples: &[i16]) -> BackendResult<()> {
    use std::io::Write;

    let data_size_bytes = u32::try_from(samples.len() * 2)
        .map_err(|_| BackendError::InvalidRequest("generated WAV is too large".into()))?;
    let byte_rate = 24000u32 * 1u32 * 16u32 / 8;
    let block_align = 1u16 * 16u16 / 8;
    let riff_size = 36 + data_size_bytes;

    let mut writer = BufWriter::new(std::fs::File::create(path)?);
    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;   // mono
    writer.write_all(&24000u32.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_size_bytes.to_le_bytes())?;
    for sample in samples {
        writer.write_all(&sample.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}
