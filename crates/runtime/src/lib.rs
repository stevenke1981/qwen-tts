//! Runtime trait, scheduler, and external qwentts.cpp adapter.

pub mod backend;
pub mod config;
pub mod device;
pub mod external_qwentts;
pub mod logging;
pub mod scheduler;

pub use backend::{
    BackendError, BackendResult, RuntimeBackend, SynthesisRequest, SynthesisResponse,
};
pub use config::{ConfigError, ConfigResult, RuntimeConfig};
pub use device::DeviceKind;
pub use external_qwentts::ExternalQwenTtsBackend;
pub use logging::{init_logging, init_logging_with, LoggingError, LoggingOptions, LoggingResult};
pub use scheduler::{BatchSynthesisItem, BatchSynthesisResponse, Scheduler};
