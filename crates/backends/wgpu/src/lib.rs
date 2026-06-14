use qwen_tts_runtime::{
    BackendError, BackendResult, DeviceKind, RuntimeBackend, SynthesisRequest, SynthesisResponse,
};

#[derive(Debug, Default, Clone)]
pub struct WgpuBackend;

impl RuntimeBackend for WgpuBackend {
    fn name(&self) -> &'static str {
        "native-wgpu-placeholder"
    }
    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Wgpu
    }
    fn is_available(&self) -> bool {
        false
    }
    fn synthesize(&self, _request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        Err(BackendError::Unavailable(
            "native WGPU compute backend is not implemented yet".into(),
        ))
    }
}
