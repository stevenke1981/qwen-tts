use qwen_tts_runtime::{
    BackendError, BackendResult, DeviceKind, RuntimeBackend, SynthesisRequest, SynthesisResponse,
};

#[derive(Debug, Default, Clone)]
pub struct RocmBackend;

impl RuntimeBackend for RocmBackend {
    fn name(&self) -> &'static str {
        "native-rocm-placeholder"
    }
    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Rocm
    }
    fn is_available(&self) -> bool {
        false
    }
    fn synthesize(&self, _request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        Err(BackendError::Unavailable(
            "native ROCm/HIP backend is not implemented yet".into(),
        ))
    }
}
