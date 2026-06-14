//! Runtime trait, scheduler, and external qwentts.cpp adapter.

pub mod backend;
pub mod backend_setup;
pub mod config;
pub mod device;
pub mod external_qwentts;
pub mod logging;
pub mod models;
pub mod scheduler;

pub use backend::{
    BackendError, BackendResult, RuntimeBackend, SynthesisRequest, SynthesisResponse,
};
pub use backend_setup::{
    backend_executable_candidates, backend_status, default_backend_executable,
    default_backend_source_dir, find_qwentts_executable, setup_qwentts_backend, BackendSetupError,
    BackendSetupResult, BackendStatus, DEFAULT_QWENTTS_REPO,
};
pub use config::{ConfigError, ConfigResult, RuntimeConfig};
pub use device::DeviceKind;
pub use external_qwentts::ExternalQwenTtsBackend;
pub use logging::{init_logging, init_logging_with, LoggingError, LoggingOptions, LoggingResult};
pub use models::{
    default_model_set, default_model_status, ensure_default_models,
    ensure_default_models_with_progress, resolve_url_for_file, DefaultModelFile,
    DefaultModelStatus, ModelDownloadError, ModelDownloadProgress, ModelDownloadResult,
    ModelFileStatus, DEFAULT_CODEC_FILE, DEFAULT_MODELS_DIR, DEFAULT_MODEL_FILES,
    DEFAULT_MODEL_REPO, DEFAULT_TALKER_FILE,
};
pub use scheduler::{BatchSynthesisItem, BatchSynthesisResponse, Scheduler};
