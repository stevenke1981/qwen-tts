//! Native CPU backend — Milestone 2: FFI inference via qwen-core.
//!
//! Links against the pre-built `qwen-core.lib` (static) which contains the
//! complete qwentts.cpp TTS pipeline (talker LM, code predictor, codec decoder,
//! BPE tokenizer, prompt builder, sampling). ggml is loaded as a shared library
//! (ggml.dll, ggml-base.dll, ggml-cpu.dll).
//!
//! The `CpuBackend` lazily initialises a `NativeContext` on the first
//! `synthesize` call and reuses it for subsequent calls. Output is written
//! as a 24 kHz mono 16-bit PCM WAV file.

use qwen_tts_core::{validate_wav_file, AudioSpec};
use qwen_tts_runtime::{
    BackendError, BackendResult, DeviceKind, RuntimeBackend, SynthesisRequest, SynthesisResponse,
};
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    sync::Mutex,
};

mod context;
mod ffi;

use context::NativeContext;

const SAMPLE_RATE_HZ: u32 = 24_000;
const CHANNELS: u16 = 1;
const BITS_PER_SAMPLE: u16 = 16;

/// Native CPU backend that uses the qwentts.cpp pipeline via FFI.
pub struct CpuBackend {
    /// Lazily initialised native inference context. Serialised by a Mutex
    /// because the C++ pipeline is not re-entrant per context.
    ctx: Mutex<Option<NativeContext>>,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ctx: Mutex::new(None),
        }
    }

    /// Ensure the native context is initialised with the given model paths.
    fn ensure_context(&self, request: &SynthesisRequest) -> BackendResult<()> {
        let mut guard = self
            .ctx
            .lock()
            .map_err(|e| BackendError::InvalidRequest(format!("mutex poisoned: {e}")))?;

        if guard.is_some() {
            return Ok(());
        }

        let ctx = NativeContext::new(&request.models.talker.path, &request.models.codec.path)
            .map_err(|e| {
                BackendError::InvalidRequest(format!(
                    "failed to initialise native TTS engine: {e}"
                ))
            })?;

        *guard = Some(ctx);
        Ok(())
    }
}

impl RuntimeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "native-cpu-ffi"
    }

    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn is_available(&self) -> bool {
        true
    }

    fn synthesize(&self, request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        if request.text.trim().is_empty() {
            return Err(BackendError::InvalidRequest("text cannot be empty".into()));
        }

        // Ensure output directory exists
        if let Some(parent) = request
            .out_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }

        // Initialise the native context on the first call
        self.ensure_context(request)?;

        // Acquire the lock and run inference
        let guard = self
            .ctx
            .lock()
            .map_err(|e| BackendError::InvalidRequest(format!("mutex poisoned: {e}")))?;

        let ctx = guard.as_ref().ok_or_else(|| {
            BackendError::InvalidRequest("native context not initialised".into())
        })?;

        let f32_samples = ctx
            .synthesize(
                &request.text,
                &request.language,
                request.speaker.as_deref(),
            )
            .map_err(|e| BackendError::InvalidRequest(format!("synthesis failed: {e}")))?;

        drop(guard); // release the lock before file I/O

        // Convert f32 PCM to i16 and write WAV
        let i16_samples = f32_samples_to_i16(&f32_samples);
        write_pcm16_wav(&request.out_path, &i16_samples)?;

        let metadata = validate_wav_file(&request.out_path, AudioSpec::default())?;

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
// PCM conversion and WAV writing
// ---------------------------------------------------------------------------

/// Convert `[-1.0, 1.0]` f32 samples to clamped 16-bit PCM.
#[allow(clippy::cast_possible_truncation)]
fn f32_samples_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| {
            let clamped = s.clamp(-1.0, 1.0);
            (clamped * f32::from(i16::MAX)).round() as i16
        })
        .collect()
}

/// Write 16-bit mono PCM samples as a WAV file.
fn write_pcm16_wav(path: &Path, samples: &[i16]) -> BackendResult<()> {
    let data_size_bytes = u32::try_from(samples.len() * 2)
        .map_err(|_| BackendError::InvalidRequest("generated WAV is too large".into()))?;
    let byte_rate = SAMPLE_RATE_HZ * u32::from(CHANNELS) * u32::from(BITS_PER_SAMPLE) / 8;
    let block_align = CHANNELS * BITS_PER_SAMPLE / 8;
    let riff_size = 36 + data_size_bytes;

    let mut writer = BufWriter::new(fs::File::create(path)?);
    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16_u32.to_le_bytes())?;
    writer.write_all(&1_u16.to_le_bytes())?;
    writer.write_all(&CHANNELS.to_le_bytes())?;
    writer.write_all(&SAMPLE_RATE_HZ.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&BITS_PER_SAMPLE.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_size_bytes.to_le_bytes())?;
    for sample in samples {
        writer.write_all(&sample.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_metadata_is_correct() {
        let backend = CpuBackend::new();
        assert!(backend.is_available());
        assert_eq!(backend.device_kind(), DeviceKind::Cpu);
        assert_eq!(backend.name(), "native-cpu-ffi");
    }

    #[test]
    fn rejects_empty_text() {
        let backend = CpuBackend::new();
        let request = SynthesisRequest {
            text: "".to_owned(),
            language: "english".to_owned(),
            speaker: None,
            out_path: Path::new("test.wav").to_path_buf(),
            device: DeviceKind::Cpu,
            models: qwen_tts_core::TtsModelSet::new("dummy.gguf", "dummy.gguf"),
        };
        let result = backend.synthesize(&request);
        assert!(result.is_err());
        match result {
            Err(BackendError::InvalidRequest(msg)) => {
                assert!(
                    msg.contains("empty"),
                    "expected empty-text error, got: {msg}"
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    /// Verify f32→i16 conversion clamps and scales correctly.
    #[test]
    fn f32_to_i16_clamps_properly() {
        let input = vec![-1.0, 0.0, 0.5, 1.0, 1.5, -2.0];
        let output = f32_samples_to_i16(&input);
        assert_eq!(output.len(), 6);
        // Scaling by i16::MAX (32767) yields slightly asymmetric range [-32767, 32767]
        assert_eq!(output[0], -32767); // -1.0 → -32767
        assert_eq!(output[1], 0); // 0.0 → 0
        assert_eq!(output[3], i16::MAX); // 1.0 → 32767
        assert_eq!(output[4], i16::MAX); // 1.5 clamped to 32767
        assert_eq!(output[5], -32767); // -2.0 clamped to -32767
    }

    /// Verify WAV header structure is correct for a trivial sample.
    #[test]
    fn wav_header_is_valid() {
        let dir = std::env::temp_dir().join("qwen-tts-cpu-test-wav");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test.wav");

        let samples: Vec<i16> = vec![0, 100, -100, 32767, -32768];
        write_pcm16_wav(&path, &samples).unwrap();

        let metadata = fs::metadata(&path).unwrap();
        // RIFF(12) + fmt(24) + data(8) + samples*2
        let expected_len = 44 + samples.len() * 2;
        assert_eq!(metadata.len() as usize, expected_len);

        let _ = fs::remove_dir_all(&dir);
    }
}
