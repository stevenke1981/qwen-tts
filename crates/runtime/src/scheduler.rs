use crate::{BackendError, BackendResult, RuntimeBackend, SynthesisRequest, SynthesisResponse};

pub struct Scheduler {
    backends: Vec<Box<dyn RuntimeBackend>>,
}

#[derive(Debug)]
pub struct BatchSynthesisItem {
    pub index: usize,
    pub result: BackendResult<SynthesisResponse>,
}

#[derive(Debug, Default)]
pub struct BatchSynthesisResponse {
    pub items: Vec<BatchSynthesisItem>,
}

impl BatchSynthesisResponse {
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.items.iter().filter(|item| item.result.is_ok()).count()
    }

    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.items.len() - self.success_count()
    }

    #[must_use]
    pub fn is_success(&self) -> bool {
        self.failure_count() == 0
    }
}

impl Scheduler {
    #[must_use]
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    pub fn register<B>(&mut self, backend: B)
    where
        B: RuntimeBackend + 'static,
    {
        self.backends.push(Box::new(backend));
    }

    /// Synthesizes a single request using the best matching registered backend.
    ///
    /// # Errors
    ///
    /// Returns an error when no backend is available or the selected backend
    /// fails to synthesize the request.
    pub fn synthesize(&self, request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        let selected = self
            .backends
            .iter()
            .find(|backend| {
                backend.device_kind() == request.device || request.device.to_string() == "auto"
            })
            .or_else(|| self.backends.iter().find(|backend| backend.is_available()));

        match selected {
            Some(backend) => backend.synthesize(request),
            None => Err(BackendError::Unavailable(
                "no registered backend is available".into(),
            )),
        }
    }

    #[must_use]
    pub fn synthesize_batch(&self, requests: &[SynthesisRequest]) -> BatchSynthesisResponse {
        let items = requests
            .iter()
            .enumerate()
            .map(|(index, request)| BatchSynthesisItem {
                index,
                result: self.synthesize(request),
            })
            .collect();

        BatchSynthesisResponse { items }
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Scheduler;
    use crate::{
        BackendError, BackendResult, DeviceKind, RuntimeBackend, SynthesisRequest,
        SynthesisResponse,
    };
    use qwen_tts_core::TtsModelSet;
    use std::path::PathBuf;

    #[test]
    fn batch_synthesis_aggregates_successes_and_failures() {
        let mut scheduler = Scheduler::new();
        scheduler.register(FakeBackend);
        let requests = vec![request("one"), request("fail"), request("two")];

        let response = scheduler.synthesize_batch(&requests);

        assert_eq!(response.items.len(), 3);
        assert_eq!(response.success_count(), 2);
        assert_eq!(response.failure_count(), 1);
        assert!(!response.is_success());
        assert_eq!(response.items[0].index, 0);
        assert!(response.items[0].result.is_ok());
        assert_eq!(response.items[1].index, 1);
        assert!(matches!(
            response.items[1].result,
            Err(BackendError::InvalidRequest(_))
        ));
        assert_eq!(response.items[2].index, 2);
        assert!(response.items[2].result.is_ok());
    }

    struct FakeBackend;

    impl RuntimeBackend for FakeBackend {
        fn name(&self) -> &'static str {
            "fake"
        }

        fn device_kind(&self) -> DeviceKind {
            DeviceKind::Cpu
        }

        fn is_available(&self) -> bool {
            true
        }

        fn synthesize(&self, request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
            if request.text == "fail" {
                return Err(BackendError::InvalidRequest("forced failure".into()));
            }

            Ok(SynthesisResponse {
                wav_path: request.out_path.clone(),
                sample_rate_hz: 24_000,
                channels: 1,
                bits_per_sample: 16,
                data_size_bytes: 4,
                backend_name: self.name().to_owned(),
            })
        }
    }

    fn request(text: &str) -> SynthesisRequest {
        SynthesisRequest {
            text: text.to_owned(),
            language: "Chinese".to_owned(),
            speaker: None,
            instruct: None,
            ref_audio_path: None,
            ref_text: None,
            seed: None,
            max_new_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            do_sample: None,
            out_path: PathBuf::from(format!("{text}.wav")),
            device: DeviceKind::Cpu,
            models: TtsModelSet::new("talker.gguf", "codec.gguf"),
        }
    }
}
