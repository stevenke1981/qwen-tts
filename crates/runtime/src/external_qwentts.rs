use crate::{
    BackendError, BackendResult, DeviceKind, RuntimeBackend, SynthesisRequest, SynthesisResponse,
};
use qwen_tts_core::{validate_wav_file, AudioSpec};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};
use tracing::{debug, error, info};

#[derive(Debug, Clone)]
pub struct ExternalQwenTtsBackend {
    pub qwen_tts_bin: PathBuf,
    pub device: DeviceKind,
}

impl ExternalQwenTtsBackend {
    #[must_use]
    pub fn new(qwen_tts_bin: impl Into<PathBuf>, device: DeviceKind) -> Self {
        Self {
            qwen_tts_bin: qwen_tts_bin.into(),
            device,
        }
    }
}

impl RuntimeBackend for ExternalQwenTtsBackend {
    fn name(&self) -> &'static str {
        "qwentts.cpp-cli"
    }

    fn device_kind(&self) -> DeviceKind {
        self.device
    }

    fn is_available(&self) -> bool {
        self.qwen_tts_bin.exists()
    }

    fn synthesize(&self, request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        if request.text.trim().is_empty() {
            return Err(BackendError::InvalidRequest("text cannot be empty".into()));
        }
        if !self.is_available() {
            return Err(BackendError::Unavailable(format!(
                "qwen-tts executable not found at {}",
                self.qwen_tts_bin.display()
            )));
        }
        if let Some(parent) = request
            .out_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }

        let mut command = Command::new(&self.qwen_tts_bin);
        command
            .arg("--model")
            .arg(&request.models.talker.path)
            .arg("--codec")
            .arg(&request.models.codec.path)
            .arg("--lang")
            .arg(&request.language)
            .arg("-o")
            .arg(&request.out_path);

        if let Some(speaker) = &request.speaker {
            command.arg("--speaker").arg(speaker);
        }

        debug!(
            backend = self.name(),
            program = %self.qwen_tts_bin.display(),
            talker_model = %request.models.talker.path.display(),
            codec_model = %request.models.codec.path.display(),
            output = %request.out_path.display(),
            device = %request.device,
            text_len = request.text.chars().count(),
            "starting qwentts process"
        );

        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        {
            let stdin = child.stdin.as_mut().ok_or_else(|| {
                BackendError::InvalidRequest("failed to open qwentts stdin".into())
            })?;
            stdin.write_all(request.text.as_bytes())?;
        }

        let output = child.wait_with_output()?;
        if !output.status.success() {
            error!(
                backend = self.name(),
                program = %self.qwen_tts_bin.display(),
                status = ?output.status.code(),
                stderr = %String::from_utf8_lossy(&output.stderr),
                "qwentts process failed"
            );
            return Err(BackendError::CommandFailed {
                program: self.qwen_tts_bin.display().to_string(),
                status: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }

        info!(
            backend = self.name(),
            output = %request.out_path.display(),
            sample_rate_hz = 24_000_u32,
            channels = 1_u16,
            "qwentts synthesis finished"
        );

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
