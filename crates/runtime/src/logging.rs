use std::fmt;
use tracing_subscriber::{fmt as tracing_fmt, EnvFilter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoggingOptions {
    pub env_filter: String,
    pub with_ansi: bool,
    pub with_thread_ids: bool,
    pub with_targets: bool,
}

impl LoggingOptions {
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            env_filter: std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_owned()),
            ..Self::default()
        }
    }
}

impl Default for LoggingOptions {
    fn default() -> Self {
        Self {
            env_filter: "info".to_owned(),
            with_ansi: true,
            with_thread_ids: false,
            with_targets: true,
        }
    }
}

#[derive(Debug)]
pub enum LoggingError {
    Filter(tracing_subscriber::filter::ParseError),
    Init(String),
}

impl fmt::Display for LoggingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Filter(err) => write!(f, "invalid log filter: {err}"),
            Self::Init(err) => write!(f, "failed to initialize logging: {err}"),
        }
    }
}

impl std::error::Error for LoggingError {}

impl From<tracing_subscriber::filter::ParseError> for LoggingError {
    fn from(value: tracing_subscriber::filter::ParseError) -> Self {
        Self::Filter(value)
    }
}

pub type LoggingResult<T> = Result<T, LoggingError>;

/// Initializes structured logging from `RUST_LOG` or the default `info` filter.
///
/// # Errors
///
/// Returns an error when the log filter is invalid or a global subscriber has
/// already been installed.
pub fn init_logging() -> LoggingResult<()> {
    init_logging_with(LoggingOptions::from_env())
}

/// Initializes structured logging with explicit options.
///
/// # Errors
///
/// Returns an error when the log filter is invalid or a global subscriber has
/// already been installed.
pub fn init_logging_with(options: LoggingOptions) -> LoggingResult<()> {
    let env_filter = EnvFilter::try_new(options.env_filter)?;

    tracing_fmt()
        .with_env_filter(env_filter)
        .with_ansi(options.with_ansi)
        .with_thread_ids(options.with_thread_ids)
        .with_target(options.with_targets)
        .try_init()
        .map_err(|err| LoggingError::Init(err.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_filter_is_info() {
        assert_eq!(LoggingOptions::default().env_filter, "info");
    }

    #[test]
    fn rejects_invalid_filter() {
        let options = LoggingOptions {
            env_filter: "qwen_tts_runtime=notalevel".to_owned(),
            ..LoggingOptions::default()
        };

        assert!(matches!(
            init_logging_with(options),
            Err(LoggingError::Filter(_))
        ));
    }
}
