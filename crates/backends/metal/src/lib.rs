use qwen_tts_runtime::{
    BackendError, BackendResult, DeviceKind, RuntimeBackend, SynthesisRequest, SynthesisResponse,
};

#[derive(Debug, Default, Clone)]
pub struct MetalBackend;

impl RuntimeBackend for MetalBackend {
    fn name(&self) -> &'static str {
        "native-metal-placeholder"
    }
    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Metal
    }
    fn is_available(&self) -> bool {
        false
    }
    fn synthesize(&self, _request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        Err(BackendError::Unavailable(
            "native Metal backend is not implemented yet; qwentts.cpp Metal path is recommended first".into(),
        ))
    }
}
