use qwen_tts_runtime::{
    BackendError, BackendResult, DeviceKind, RuntimeBackend, SynthesisRequest, SynthesisResponse,
};

#[derive(Debug, Default, Clone)]
pub struct CpuBackend;

impl RuntimeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "native-cpu-placeholder"
    }
    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }
    fn is_available(&self) -> bool {
        false
    }
    fn synthesize(&self, _request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        Err(BackendError::Unavailable(
            "native CPU graph is not implemented yet; use qwentts.cpp CLI backend for MVP".into(),
        ))
    }
}
