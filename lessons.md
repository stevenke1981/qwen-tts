---
## Lesson #1 - 2026-06-14
**Trigger:** Generated OpenCode cache appeared as an untracked `${PROJECT_ROOT}/` folder before release staging.
**Rule:** Before `git add -A`, run `git status --short --ignored` and add accidental local cache folders to `.gitignore` instead of staging them.
**Source:** complete compiled version handoff

---
## Lesson #2 - 2026-06-14
**Trigger:** Release copy failed because `dist/qwen-tts-gui.exe` was still running and locked on Windows.
**Rule:** Before copying release GUI binaries into `dist`, check for running `qwen-tts-gui` processes and stop the old `dist` executable if it locks the target file.
**Source:** qwentts backend implementation
