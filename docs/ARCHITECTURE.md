# Architecture

## Recommended split

Use the workspace split proposed by Steven, with one adjustment: keep qwentts.cpp integration in the runtime layer first, then add native backends later.

```text
CLI / GUI
  ↓
runtime::Scheduler
  ↓
RuntimeBackend trait
  ↓
ExternalQwenTtsBackend or native backend
  ↓
Qwen3-TTS GGUF talker + codec
  ↓
24 kHz mono WAV
```

## Why not implement all GPU backends immediately?

Each GPU stack has different build-time and runtime requirements. If all backends are hard dependencies, a Windows CPU user could fail to build because ROCm or SYCL is unavailable. Isolating each backend crate keeps the workspace maintainable.

## Backend priority

1. qwentts.cpp CLI adapter — fastest MVP.
2. qwentts.cpp FFI — stable app integration.
3. Native CPU — correctness reference.
4. CUDA — NVIDIA acceleration.
5. Metal — Apple Silicon.
6. WGPU — cross-platform GPU fallback.
7. ROCm — AMD GPU.
8. SYCL — Intel/oneAPI.
