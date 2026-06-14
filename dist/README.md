# Qwen TTS Windows Release

This folder contains the compiled Windows CLI binary:

- `qwen-tts.exe`
- `qwen-tts-gui.exe`
- `SHA256SUMS.txt`

The binary is the Rust app/runtime layer. Speech synthesis still requires the
external `qwen-tts` runtime from qwentts.cpp and the Qwen3-TTS GGUF model files
described in the repository README. The CLI and GUI can download the default
GGUF files into `./models`.

Quick check:

```powershell
.\qwen-tts.exe --help
.\qwen-tts-gui.exe
```
