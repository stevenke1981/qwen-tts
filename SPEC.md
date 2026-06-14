# SPEC.md — Qwen TTS

## Goal

Build a Rust local text-to-speech app using Qwen3-TTS GGUF models.

The architecture must support:

- GGUF model inspection.
- Default GGUF model download into `./models`.
- Text-to-WAV generation through qwentts.cpp.
- A clean Rust runtime abstraction.
- A native egui GUI for setup and synthesis.
- Future native CPU/CUDA/ROCm/Metal/WGPU/SYCL backend implementations.
- Future streaming playback.

## Non-goals for MVP

- Full native Qwen3-TTS graph execution in pure Rust.
- Training or fine-tuning.
- Real-time voice cloning UI.

## Crate responsibilities

### `crates/core`

No GPU dependencies. Owns:

- Model path structs.
- GGUF header probe.
- TTS graph description.
- Audio buffer/spec types.
- Common op enums.

### `crates/runtime`

Owns:

- `RuntimeBackend` trait.
- `DeviceKind` enum.
- `Scheduler`.
- `SynthesisRequest` / `SynthesisResponse`.
- `ExternalQwenTtsBackend`, which calls the qwentts.cpp CLI.
- Default GGUF model catalog/status/download helpers.

### `crates/app`

Native egui desktop app:

- Model folder status.
- Default GGUF download button.
- Text, language, speaker, device, runtime binary, and output path controls.
- Background download and synthesis workers.

### `crates/backends/cpu`

Future native CPU backend. Suggested implementation path:

- `rayon` for parallel CPU scheduling.
- `ndarray` or custom packed tensor views.
- GGML-compatible quantized matmul kernels.

### `crates/backends/cuda`

Future native CUDA backend. Suggested implementation path:

- `cudarc` or `cust` for CUDA driver/runtime integration.
- Custom kernels for quantized matmul and codec decode hot paths.
- Optional FFI bridge to qwentts.cpp first.

### `crates/backends/rocm`

Future native AMD GPU backend.

- HIP bindings are likely required.
- Keep this isolated because ROCm install/toolchain requirements differ from CUDA.

### `crates/backends/metal`

Future Apple Silicon backend.

- `metal` crate or MLX-style integration.
- Prioritize M-series devices because local TTS usage is common on Mac laptops.

### `crates/backends/wgpu`

Future cross-platform GPU fallback.

- Useful for Vulkan/Metal/DX12 portability.
- Start with compute experiments, not full graph execution.

### `crates/backends/sycl`

Future Intel / oneAPI / AdaptiveCpp backend.

- Lower priority unless Intel GPU support becomes a hard requirement.

### `crates/cli`

Binary entrypoint:

- `inspect`
- `graph`
- `models status`
- `models download`
- `setup-script`
- `synth`

## Data flow

```text
text
  ↓
CLI parses request
  ↓
runtime scheduler selects backend
  ↓
ExternalQwenTtsBackend invokes qwen-tts
  ↓
talker GGUF → acoustic codes
  ↓
codec GGUF → 24 kHz mono WAV
  ↓
output.wav
```

## Error handling

- Missing executable returns `BackendError::Unavailable`.
- Empty text returns `BackendError::InvalidRequest`.
- Failed qwen-tts process returns `BackendError::CommandFailed` with stderr.
- GGUF header errors return `GgufProbeError`.

## Future FFI design

Add a new crate:

```text
crates/qwentts-sys/
crates/qwentts-safe/
```

Suggested split:

- `qwentts-sys`: bindgen-generated unsafe C ABI.
- `qwentts-safe`: safe Rust wrapper with ownership, context lifetime, and error mapping.

Then `runtime` can swap `ExternalQwenTtsBackend` for `QwenTtsFfiBackend` without changing CLI commands.
