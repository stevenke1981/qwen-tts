# Qwen TTS Windows Release

This folder contains the compiled Windows CLI binary:

- `qwen-tts.exe`
- `SHA256SUMS.txt`

The binary is the Rust app/runtime layer. Speech synthesis still requires the
external `qwen-tts` runtime from qwentts.cpp and the Qwen3-TTS GGUF model files
described in the repository README.

Quick check:

```powershell
.\qwen-tts.exe --help
```
