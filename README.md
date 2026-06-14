# Qwen TTS

Rust workspace for building a local speech generation app with Qwen3-TTS GGUF
models.

Four backends are available:

| Backend | Mode | Default | Description |
|---------|------|---------|-------------|
| `ffi` | In-process FFI | ✅ (since v0.1) | Direct calls into qwentts.cpp shared library |
| `qwentts` | Subprocess | | External `qwen-tts` CLI executable |
| `native-cpu` | In-process FFI | | Wraps qwentts.cpp via `qwentts-sys` with WAV write in Rust |
| `pure-rust` | Pure Rust (candle) | 🚧 WIP | Zero C++ dependency — candle 0.10.x, GGUF dequant→F32 |

## Layout

```text
qwen_tts/
├── Cargo.toml
├── Cargo.lock
├── crates/
│   ├── core/          — Model paths, GGUF probe, audio spec types
│   ├── runtime/       — Backend trait, scheduler, config, model download
│   ├── backends/
│   │   ├── cpu/       — Native CPU backend (FFI to qwentts.cpp)
│   │   ├── pure-rust/ — Pure Rust backend (candle 0.10.x, no C++) 🚧
│   │   ├── cuda/      — (skeleton)
│   │   ├── rocm/      — (skeleton)
│   │   ├── metal/     — (skeleton)
│   │   ├── wgpu/      — (skeleton)
│   │   └── sycl/      — (skeleton)
│   ├── qwentts-sys/   — Unsafe raw FFI + safe Rust wrapper
│   ├── cli/           — CLI binary
│   └── app/           — egui desktop GUI
├── vendor/
│   └── qwentts.cpp/   — Upstream C++ TTS library
├── examples/
├── scripts/
├── docs/
└── models/
```

## Model files

The app can download the default qwentts.cpp GGUF files from
`Serveurperso/Qwen3-TTS-GGUF` into `./models`:

```bash
cargo run -p qwen-tts-cli -- models download
```

The CLI prints GGUF download progress while files are being fetched.

Dry-run and status checks:

```bash
cargo run -p qwen-tts-cli -- models download --dry-run
cargo run -p qwen-tts-cli -- models status
```

The default files are:

```text
qwen-talker-1.7b-base-Q8_0.gguf
qwen-tokenizer-12hz-Q8_0.gguf
```

The talker model converts text into acoustic codes; the codec/tokenizer model decodes those codes into 24 kHz mono WAV.
When `synth` uses the default model paths and either file is missing, it downloads the default GGUF files before synthesis.

## Build Rust workspace

```bash
cargo build --workspace
```

## Windows release binary

The checked-in Windows build artifacts are available at:

```text
dist/qwen-tts.exe
dist/qwen-tts-gui.exe
```

Verify it with:

```powershell
Get-FileHash dist/qwen-tts.exe -Algorithm SHA256
Get-FileHash dist/qwen-tts-gui.exe -Algorithm SHA256
Get-Content dist/SHA256SUMS.txt
```

## Run the egui desktop app

The GUI supports all three backends. With the `ffi` feature (default), it
uses the in-process FFI backend for synthesis:

```bash
# Default (FFI in-process backend)
cargo run -p qwen-tts-app --features ffi

# Fallback: subprocess backend (requires qwen-tts executable)
cargo run -p qwen-tts-app
```

The GUI uses the project-level `models/` folder by default. When it opens and
the default GGUF files are missing, it asks whether to download them into that
folder and shows download progress in the status bar.

Synthesis controls:
- **Text**, **language**, **speaker**, **instruct** (voice style guide)
- **Reference audio** for voice cloning (WAV path + reference text)
- **Backend selection** (FFI / Native CPU / qwentts.cpp subprocess)
- **Device** selection (Auto / CPU / CUDA / ROCm / Metal / WGPU / SYCL)
- **Advanced params** — seed, temperature, top-k, top-p, repetition penalty,
  max tokens, do_sample, flash attention, clamp fp16

Playback:
- **Auto-play** after synthesis completes
- Manual **Play / Pause / Stop** controls
- **Progress bar** with elapsed / total time display
- Powered by `rodio` (cross-platform, non-blocking)

## Build qwentts.cpp runtime

```bash
cargo run -p qwen-tts-cli -- backend status
cargo run -p qwen-tts-cli -- backend setup
```

The GUI also shows backend status and can run the same setup flow from the
`建置 backend` button.

Script generator for manual qwentts.cpp builds:

```bash
cargo run -p qwen-tts-cli -- setup-script --target cpu > setup.sh
bash setup.sh
```

## Inspect GGUF headers

```bash
cargo run -p qwen-tts-cli -- inspect \
  --talker models/qwen-talker-1.7b-base-Q8_0.gguf \
  --codec models/qwen-tokenizer-12hz-Q8_0.gguf
```

## Generate speech

Default (FFI in-process backend):

```bash
cargo run -p qwen-tts-cli -- synth \
  --text "你好，這是 Qwen TTS 語音合成測試。" \
  --lang Chinese
```

Use the legacy subprocess backend:

```bash
cargo run -p qwen-tts-cli -- synth \
  --text "你好" \
  --backend qwentts
```

All available flags:

| Flag | Description |
|------|-------------|
| `--text` | Input text to synthesise |
| `--lang` | Language (Chinese, English, Japanese, etc.) |
| `--speaker` | Speaker ID or name |
| `--instruct` | Voice style / emotion guide |
| `--ref-audio` | Reference WAV path (voice cloning) |
| `--ref-text` | Transcription of the reference audio |
| `--seed` | Random seed for reproducibility |
| `--temperature` | Sampling temperature (0.0–2.0) |
| `--top-k` | Top-K sampling |
| `--top-p` | Top-P (nucleus) sampling |
| `--repetition-penalty` | Repetition penalty (≥ 1.0) |
| `--max-tokens` | Maximum output tokens |
| `--no-sample` | Disable random sampling (greedy decode) |
| `--backend` | `ffi` (default), `qwentts`, `native-cpu`, or `pure-rust` (🚧 WIP) |
| `--device` | `auto`, `cpu`, `cuda`, `rocm`, `metal`, `wgpu`, `sycl` |
| `--out` | Output WAV path (default: `output/voice-<timestamp>.wav`) |

When `--out` is omitted, the WAV is written to `output/voice-<timestamp>.wav`.

If your `qwen-tts` binary is elsewhere (only needed for `--backend qwentts`):

```bash
QWEN_TTS_BIN=/path/to/qwen-tts cargo run -p qwen-tts-cli -- synth --backend qwentts --text "測試"
```

## Roadmap

### ✅ Completed
- [x] CLI: GGUF inspect, model download, TOML config, batch synth
- [x] FFI: In-process qwentts.cpp backend (`--backend ffi`), voice cloning
- [x] GUI: egui desktop app with model mgmt, synthesis form, audio playback
- [x] CLI/GUI integration test suite (4 e2e, 4 unit)
- [x] Codec decoder + DAC in pure Rust (54 tests)
- [x] Pure-Rust talker transformer: Qwen2 (RoPE, GQA, SwiGLU, RMSNorm)
- [x] Pure-Rust code predictor (MTP heads, 15 acoustic codebooks)
- [x] BPE tokenizer via `tokenizers` 0.21
- [x] Top-k / top-p / temperature sampling (pure Rust)
- [x] Pipeline wiring: tokenize → talker → code predictor → codec decoder → WAV
- [x] Tensor naming + metadata correction for real qwen3-tts GGUF
- [x] GGUF probe utilities for tensor/metadata discovery

### 🚧 In Progress — Pure Rust Backend
```
Status: 83% (10/12 tasks)

Core modules      ■■■■■■■■■■  all implemented
Unit tests        ■■■■■■■■■■  19/19 pass
E2E structure     ■■■■■■■■■■  3/3 pass
Heavy e2e (8GB+)  □□□□□□□□□□  #[ignore] — needs real GGUF load
Codebook 0 pred   □□□□□□□□□□  placeholder (talker.codec_head.weight)
MRoPE support     □□□□□□□□□□  talker uses standard RoPE, model needs MRoPE
```

### 📋 Next Steps
- [ ] Run heavy E2E tests (`test_talker_loads`, `test_code_predictor_loads`, `test_pipeline_full_synthesize`) — requires ~8-10 GB RAM
- [ ] Replace codebook-0 placeholder with actual `talker.codec_head.weight` prediction
- [ ] Implement MRoPE (Multi-Resolution RoPE) for talker — see `qwen3-tts.talker.rope.mrope_section`
- [ ] Implement autoregressive frame generation (recurrent hidden state)
- [ ] Cross-validate: pure-rust output == FFI backend output (same seed, bit-exact)
- [ ] Native CUDA / Metal / WGPU / ROCm / SYCL backends
