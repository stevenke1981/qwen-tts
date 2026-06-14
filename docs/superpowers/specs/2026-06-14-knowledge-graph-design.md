# Knowledge Graph Design — Qwen TTS

**Date:** 2026-06-14
**Complexity:** L3 (multi-layer, multi-deliverable)
**Status:** Design approved

## Goal

Build a three-layer knowledge graph for the Qwen TTS workspace, covering codebase relationships, architectural decisions, and domain knowledge — both queryable (via codebase-memory-mcp) and visual (via Mermaid diagrams and documentation).

---

## Layer 1 — Codebase Knowledge Graph (codebase-memory-mcp)

### Tool
`codebase-memory-mcp` — OpenCode's built-in codebase indexing.

### Scope
All 9 workspace crates:

| Crate | Path | Role |
|-------|------|------|
| `qwen-tts-core` | `crates/core/` | Model types, GGUF probe, graph description, audio types |
| `qwen-tts-runtime` | `crates/runtime/` | RuntimeBackend trait, scheduler, config, logging |
| `qwen-tts-cli` | `crates/cli/` | Binary entrypoint: inspect, graph, setup-script, synth |
| `qwen-tts-backend-cpu` | `crates/backends/cpu/` | Future native CPU backend |
| `qwen-tts-backend-cuda` | `crates/backends/cuda/` | Future native CUDA backend |
| `qwen-tts-backend-rocm` | `crates/backends/rocm/` | Future native ROCm backend |
| `qwen-tts-backend-metal` | `crates/backends/metal/` | Future native Metal backend |
| `qwen-tts-backend-wgpu` | `crates/backends/wgpu/` | Future cross-platform GPU backend |
| `qwen-tts-backend-sycl` | `crates/backends/sycl/` | Future Intel/oneAPI backend |

### Key Symbols to Index

**Core crate:**
- `GgufProbe` — GGUF header inspector
- `GgufProbeError` — probe error enum
- `TtsGraph` — TTS pipeline description
- `GraphNode`, `NodeKind` — graph node types
- `AudioBuffer`, `AudioSpec` — audio data types
- `TalkerModel`, `CodecModel`, `TtsModelSet` — model path types
- `WavMetadata`, `read_wav_metadata`, `validate_wav_file` — WAV utilities

**Runtime crate:**
- `RuntimeBackend` trait — backend abstraction
- `SynthesisRequest`, `SynthesisResponse` — request/response types
- `Scheduler` — backend selection and dispatch
- `ExternalQwenTtsBackend` — CLI process adapter
- `DeviceKind` — device enum (Auto, Cpu, Cuda, Rocm, Metal, Wgpu, Sycl)
- `RuntimeConfig` — TOML config
- `BatchSynthesisItem`, `BatchSynthesisResponse` — batch API

**CLI crate:**
- CLI command structs: `InspectArgs`, `SynthArgs`, `SetupScriptArgs`
- Entry functions: `inspect()`, `synth()`, `graph()`, `setup_script()`
- Helper: `path_from_arg_env_or_default()`

### Relationships to Capture
- Workspace member → Cargo.toml dependencies
- CLI → runtime (Scheduler, ExternalQwenTtsBackend)
- Runtime → core (model types, graph types, audio types)
- RuntimeBackend trait → ExternalQwenTtsBackend (impl)
- SynthesisRequest → TtsModelSet (composition)
- Scheduler → RuntimeBackend (dispatch)
- GgufProbe → GgufProbeError (error mapping)
- TtsGraph → GraphNode (composition)

### Deliverable
- Indexed graph DB at `.codebase-memory/graph.db.zst`
- Architecture overview via `cbm_get_architecture`
- Call paths via `cbm_trace_path`
- ADRs stored via `cbm_manage_adr`

---

## Layer 2 — Architecture Decision Records (ADRs) + Mermaid Visualization

### ADR List

| ADR # | Title | Decision |
|-------|-------|----------|
| ADR-001 | Workspace Split | 9 crates with core/runtime/backends/cli; resolver 2 |
| ADR-002 | Backend Isolation | Each backend is an independent crate with optional compilation |
| ADR-003 | External Process Adapter | MVP uses ExternalQwenTtsBackend to shell out to qwentts.cpp CLI |
| ADR-004 | Data Flow Architecture | text → CLI → Scheduler → RuntimeBackend → WAV |
| ADR-005 | TTS Graph as Description | TtsGraph is a pipeline descriptor, not an executor |
| ADR-006 | FFI Migration Path | Future crates/qwentts-sys + crates/qwentts-safe to replace process execution |

### Mermaid Diagrams

1. **Crate dependency graph** — workspace member relationships
2. **Data flow diagram** — text input through WAV output
3. **Backend priority tree** — implementation order with decision rationale
4. **Module composition** — internal module structure per crate
5. **RuntimeBackend trait hierarchy** — trait definition, impls, and planned impls

### Deliverable
- ADRs stored in codebase-memory-mcp via `cbm_manage_adr`
- Mermaid diagrams embedded in `docs/knowledge-graph/`
- Architecture overview document

---

## Layer 3 — Domain Knowledge Graph

### Domain Topics

| Topic | Coverage | Code Mapping |
|-------|----------|-------------|
| GGUF Format | File structure, metadata KV, tensor layout, quantization types | `core/src/gguf.rs` — `GgufProbe` |
| Qwen3-TTS Architecture | Talker model (text→acoustic codes), Codec model (codes→WAV), 24kHz mono | `core/src/graph.rs` — `TtsGraph` |
| qwentts.cpp | Repository structure, build system (CMake), C ABI, CLI interface | `runtime/src/external_qwentts.rs` |
| WAV Format | RIFF header, sample format, PCM data, metadata chunks | `core/src/wav.rs` |
| GPU Backend Strategy | Comparative analysis of CUDA/ROCm/Metal/WGPU/SYCL | `crates/backends/*/` |

### Cross-Domain Relationships
```
GGUF File → GgufProbe → metadata KV / tensor info
Qwen3-TTS Talker → acoustic codes → Qwen3-TTS Codec → WAV audio
text → TtsGraph (pipeline description) → RuntimeBackend → qwentts.cpp
SynthesisRequest → TtsModelSet (talker + codec) → ExternalQwenTtsBackend
```

### Deliverable
- Domain knowledge document at `docs/knowledge-graph/domain-knowledge.md`
- Cross-reference map connecting domain concepts to code symbols
- External reference links (qwentts.cpp repo, GGUF spec, Qwen3-TTS papers)

---

## Directory Structure

```
docs/
├── superpowers/
│   └── specs/
│       └── 2026-06-14-knowledge-graph-design.md  (this file)
└── knowledge-graph/
    ├── architecture-overview.md        (Layer 2 + Mermaid diagrams)
    ├── domain-knowledge.md             (Layer 3)
    └── diagrams/
        ├── crate-dependencies.mmd
        ├── data-flow.mmd
        ├── backend-priority.mmd
        └── module-composition.mmd
```

## Implementation Order

1. Index repository via `cbm_index_repository`
2. Create ADRs via `cbm_manage_adr` (6 ADRs)
3. Write domain knowledge document
4. Write architecture overview with embedded Mermaid diagrams
5. Store snippet references for key code paths
6. Final verification — query the graph, check diagram rendering

## Definition of Done

- [ ] `cbm_get_architecture` returns populated architecture for project
- [ ] 6 ADRs stored and retrievable
- [ ] Mermaid diagrams render correctly in docs
- [ ] Domain knowledge document covers all 5 topics
- [ ] Cross-reference map links domain concepts → code symbols
- [ ] All files committed with conventional commit messages
