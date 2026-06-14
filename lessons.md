---
## Lesson #1 - 2026-06-14
**Trigger:** Generated OpenCode cache appeared as an untracked `${PROJECT_ROOT}/` folder before release staging.
**Rule:** Before `git add -A`, run `git status --short --ignored` and add accidental local cache folders to `.gitignore` instead of staging them.
**Source:** complete compiled version handoff
