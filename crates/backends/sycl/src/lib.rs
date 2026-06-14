use qwen_tts_runtime::{
    BackendError, BackendResult, DeviceKind, RuntimeBackend, SynthesisRequest, SynthesisResponse,
};

#[derive(Debug, Default, Clone)]
pub struct SyclBackend;

impl RuntimeBackend for SyclBackend {
    fn name(&self) -> &'static str {
        "native-sycl-placeholder"
    }
    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Sycl
    }
    fn is_available(&self) -> bool {
        false
    }
    fn synthesize(&self, _request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        Err(BackendError::Unavailable(
            "native SYCL/oneAPI backend is not implemented yet".into(),
        ))
    }
}
