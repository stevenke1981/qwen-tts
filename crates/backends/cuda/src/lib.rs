use qwen_tts_runtime::{
    BackendError, BackendResult, DeviceKind, RuntimeBackend, SynthesisRequest, SynthesisResponse,
};

#[derive(Debug, Default, Clone)]
pub struct CudaBackend;

impl RuntimeBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "native-cuda-placeholder"
    }
    fn device_kind(&self) -> DeviceKind {
        DeviceKind::Cuda
    }
    fn is_available(&self) -> bool {
        false
    }
    fn synthesize(&self, _request: &SynthesisRequest) -> BackendResult<SynthesisResponse> {
        Err(BackendError::Unavailable(
            "native CUDA kernels are not implemented yet; plan to integrate cudarc/cust or qwentts.cpp FFI".into(),
        ))
    }
}
