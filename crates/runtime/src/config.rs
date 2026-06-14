use crate::{DeviceKind, ExternalQwenTtsBackend, SynthesisRequest, DEFAULT_OUTPUT_DIR};
use qwen_tts_core::TtsModelSet;
use serde::Deserialize;
use std::{
    fmt, fs,
    path::{Path, PathBuf},
    str::FromStr,
};

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Parse(toml::de::Error),
    Invalid(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "failed to read config: {err}"),
            Self::Parse(err) => write!(f, "failed to parse config TOML: {err}"),
            Self::Invalid(message) => write!(f, "invalid config: {message}"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<std::io::Error> for ConfigError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<toml::de::Error> for ConfigError {
    fn from(value: toml::de::Error) -> Self {
        Self::Parse(value)
    }
}

pub type ConfigResult<T> = Result<T, ConfigError>;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct RuntimeConfig {
    #[serde(alias = "qwentts_bin", alias = "qwentts_executable")]
    pub qwen_tts_bin: PathBuf,
    pub talker_model: PathBuf,
    pub codec_model: PathBuf,
    pub default_lang: String,
    #[serde(deserialize_with = "deserialize_device_kind")]
    pub default_device: DeviceKind,
    pub default_threads: Option<usize>,
    pub output_dir: PathBuf,
}

impl RuntimeConfig {
    /// Loads runtime configuration from a TOML file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, the TOML cannot be
    /// parsed, or the parsed configuration is invalid.
    pub fn load(path: impl AsRef<Path>) -> ConfigResult<Self> {
        Self::from_toml_str(&fs::read_to_string(path)?)
    }

    /// Parses runtime configuration from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns an error when the TOML cannot be parsed or the parsed
    /// configuration is invalid.
    pub fn from_toml_str(value: &str) -> ConfigResult<Self> {
        let config = toml::from_str::<Self>(value)?;
        config.validate()?;
        Ok(config)
    }

    /// Validates required config fields and numeric limits.
    ///
    /// # Errors
    ///
    /// Returns an error when required paths or language values are empty, or
    /// when `default_threads` is set to zero.
    pub fn validate(&self) -> ConfigResult<()> {
        if self.qwen_tts_bin.as_os_str().is_empty() {
            return Err(ConfigError::Invalid("qwen_tts_bin cannot be empty".into()));
        }
        if self.talker_model.as_os_str().is_empty() {
            return Err(ConfigError::Invalid("talker_model cannot be empty".into()));
        }
        if self.codec_model.as_os_str().is_empty() {
            return Err(ConfigError::Invalid("codec_model cannot be empty".into()));
        }
        if self.default_lang.trim().is_empty() {
            return Err(ConfigError::Invalid("default_lang cannot be empty".into()));
        }
        if self.default_threads == Some(0) {
            return Err(ConfigError::Invalid(
                "default_threads must be greater than zero".into(),
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn model_set(&self) -> TtsModelSet {
        TtsModelSet::new(self.talker_model.clone(), self.codec_model.clone())
    }

    #[must_use]
    pub fn external_backend(&self) -> ExternalQwenTtsBackend {
        ExternalQwenTtsBackend::new(self.qwen_tts_bin.clone(), self.default_device)
    }

    #[must_use]
    pub fn synthesis_request(
        &self,
        text: impl Into<String>,
        out_path: impl Into<PathBuf>,
    ) -> SynthesisRequest {
        SynthesisRequest {
            text: text.into(),
            language: self.default_lang.clone(),
            speaker: None,
            instruct: None,
            out_path: out_path.into(),
            device: self.default_device,
            models: self.model_set(),
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            qwen_tts_bin: default_qwentts_bin(),
            talker_model: PathBuf::from("./models/qwen-talker-1.7b-base-Q8_0.gguf"),
            codec_model: PathBuf::from("./models/qwen-tokenizer-12hz-Q8_0.gguf"),
            default_lang: "Chinese".to_owned(),
            default_device: DeviceKind::Auto,
            default_threads: None,
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
        }
    }
}

fn default_qwentts_bin() -> PathBuf {
    if cfg!(windows) {
        PathBuf::from("./vendor/qwentts.cpp/build/Release/qwen-tts.exe")
    } else {
        PathBuf::from("./vendor/qwentts.cpp/build/qwen-tts")
    }
}

fn deserialize_device_kind<'de, D>(deserializer: D) -> Result<DeviceKind, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    DeviceKind::from_str(&value).map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_example_config() {
        let config =
            RuntimeConfig::from_toml_str(include_str!("../../../qwen-tts.toml.example")).unwrap();

        assert_eq!(config.default_lang, "Chinese");
        assert_eq!(config.default_device, DeviceKind::Auto);
        assert_eq!(config.default_threads, Some(4));
        assert_eq!(config.output_dir, PathBuf::from("output"));
        assert_eq!(
            config.model_set().talker.path,
            PathBuf::from("./models/qwen-talker-1.7b-base-Q8_0.gguf")
        );
    }

    #[test]
    fn parses_inline_config_aliases() {
        let config = RuntimeConfig::from_toml_str(
            r#"
qwentts_executable = "bin/qwen-tts"
talker_model = "models/talker.gguf"
codec_model = "models/codec.gguf"
default_lang = "English"
default_device = "cuda"
default_threads = 8
output_dir = "generated"
"#,
        )
        .unwrap();

        assert_eq!(config.qwen_tts_bin, PathBuf::from("bin/qwen-tts"));
        assert_eq!(config.default_device, DeviceKind::Cuda);
        assert_eq!(config.default_threads, Some(8));
        assert_eq!(config.external_backend().device, DeviceKind::Cuda);
    }

    #[test]
    fn rejects_zero_threads() {
        let err = RuntimeConfig::from_toml_str("default_threads = 0").unwrap_err();

        assert!(matches!(err, ConfigError::Invalid(_)));
    }
}
