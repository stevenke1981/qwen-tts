# Qwen TTS

Rust workspace for building a local speech generation app with Qwen3-TTS GGUF.

The MVP uses Rust as the app/runtime layer and delegates actual Qwen3-TTS inference to `qwentts.cpp`'s `qwen-tts` executable. The project is intentionally split into crates so native CPU/CUDA/Metal/WGPU/ROCm/SYCL backends can be added later without rewriting the CLI or scheduler.

## Layout

```text
qwen_tts/
├── Cargo.toml
├── Cargo.lock
├── crates/
│   ├── core/
│   ├── runtime/
│   ├── backends/
│   │   ├── cpu/
│   │   ├── cuda/
│   │   ├── rocm/
│   │   ├── metal/
│   │   ├── wgpu/
│   │   └── sycl/
│   └── cli/
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

```bash
cargo run -p qwen-tts-app
```

The GUI can check/download the default GGUF files, edit synthesis settings, and run text-to-WAV generation through the external qwentts.cpp runtime.

## Build qwentts.cpp runtime

```bash
cargo run -p qwen-tts-cli -- setup-script --target cpu > setup.sh
bash setup.sh
```

CUDA example:

```bash
cargo run -p qwen-tts-cli -- setup-script --target cuda > setup.sh
bash setup.sh
```

## Inspect GGUF headers

```bash
cargo run -p qwen-tts-cli -- inspect \
  --talker models/qwen-talker-1.7b-base-Q8_0.gguf \
  --codec models/qwen-tokenizer-12hz-Q8_0.gguf
```

## Generate speech

```bash
cargo run -p qwen-tts-cli -- synth \
  --text "你好，這是 Rust 本機語音生成測試。" \
  --lang Chinese \
  --device auto \
  --out output.wav
```

If your `qwen-tts` binary is elsewhere:

```bash
QWEN_TTS_BIN=/path/to/qwen-tts cargo run -p qwen-tts-cli -- synth --text "測試" --out output.wav
```

## Roadmap

1. Keep CLI + qwentts.cpp path as the MVP.
2. Replace CLI process execution with qwentts.cpp C ABI through Rust FFI.
3. Add native backend implementations crate by crate.
4. Add GUI crate later, for example Tauri / egui / Slint.
