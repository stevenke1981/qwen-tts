use qwen_tts_core::TtsModelSet;
use std::{
    fmt, fs,
    io::{self, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

pub const DEFAULT_MODEL_REPO: &str = "Serveurperso/Qwen3-TTS-GGUF";
pub const DEFAULT_MODELS_DIR: &str = "./models";
pub const DEFAULT_TALKER_FILE: &str = "qwen-talker-1.7b-base-Q8_0.gguf";
pub const DEFAULT_CODEC_FILE: &str = "qwen-tokenizer-12hz-Q8_0.gguf";

const HF_RESOLVE_BASE: &str = "https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main";

#[derive(Debug)]
pub enum ModelDownloadError {
    Io(io::Error),
    Http(String),
}

impl fmt::Display for ModelDownloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "model download I/O error: {err}"),
            Self::Http(message) => write!(f, "model download HTTP error: {message}"),
        }
    }
}

impl std::error::Error for ModelDownloadError {}

impl From<io::Error> for ModelDownloadError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub type ModelDownloadResult<T> = Result<T, ModelDownloadError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DefaultModelFile {
    pub role: &'static str,
    pub file_name: &'static str,
    pub url: &'static str,
}

impl DefaultModelFile {
    #[must_use]
    pub fn path_in(self, models_dir: impl AsRef<Path>) -> PathBuf {
        models_dir.as_ref().join(self.file_name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDownloadProgress {
    pub role: &'static str,
    pub file_name: &'static str,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub finished: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFileStatus {
    pub file: DefaultModelFile,
    pub path: PathBuf,
    pub exists: bool,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultModelStatus {
    pub models_dir: PathBuf,
    pub files: Vec<ModelFileStatus>,
}

impl DefaultModelStatus {
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.files.iter().all(|file| file.exists)
    }

    #[must_use]
    pub fn missing_files(&self) -> Vec<&ModelFileStatus> {
        self.files.iter().filter(|file| !file.exists).collect()
    }
}

pub const DEFAULT_MODEL_FILES: [DefaultModelFile; 2] = [
    DefaultModelFile {
        role: "talker",
        file_name: DEFAULT_TALKER_FILE,
        url: concat!(
            "https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main/",
            "qwen-talker-1.7b-base-Q8_0.gguf"
        ),
    },
    DefaultModelFile {
        role: "codec",
        file_name: DEFAULT_CODEC_FILE,
        url: concat!(
            "https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main/",
            "qwen-tokenizer-12hz-Q8_0.gguf"
        ),
    },
];

#[must_use]
pub fn default_model_set(models_dir: impl AsRef<Path>) -> TtsModelSet {
    TtsModelSet::new(
        models_dir.as_ref().join(DEFAULT_TALKER_FILE),
        models_dir.as_ref().join(DEFAULT_CODEC_FILE),
    )
}

#[must_use]
pub fn default_model_status(models_dir: impl AsRef<Path>) -> DefaultModelStatus {
    let models_dir = models_dir.as_ref().to_path_buf();
    let files = DEFAULT_MODEL_FILES
        .iter()
        .copied()
        .map(|file| {
            let path = file.path_in(&models_dir);
            let metadata = fs::metadata(&path).ok();
            ModelFileStatus {
                file,
                path,
                exists: metadata.as_ref().is_some_and(fs::Metadata::is_file),
                size_bytes: metadata.map(|value| value.len()),
            }
        })
        .collect();

    DefaultModelStatus { models_dir, files }
}

/// Ensures the default GGUF files exist in the given models directory.
///
/// # Errors
///
/// Returns an error when the directory cannot be created, a model cannot be
/// downloaded, or the completed download cannot be moved into place.
pub fn ensure_default_models(
    models_dir: impl AsRef<Path>,
) -> ModelDownloadResult<DefaultModelStatus> {
    ensure_default_models_with_progress(models_dir, |_| {})
}

/// Ensures the default GGUF files exist and reports byte-level download progress.
///
/// # Errors
///
/// Returns an error when the directory cannot be created, a model cannot be
/// downloaded, or the completed download cannot be moved into place.
pub fn ensure_default_models_with_progress(
    models_dir: impl AsRef<Path>,
    mut on_progress: impl FnMut(ModelDownloadProgress),
) -> ModelDownloadResult<DefaultModelStatus> {
    let models_dir = models_dir.as_ref();
    fs::create_dir_all(models_dir)?;

    for file in DEFAULT_MODEL_FILES {
        let destination = file.path_in(models_dir);
        if destination.is_file() {
            continue;
        }
        download_model_file_with_progress(file, &destination, &mut on_progress)?;
    }

    Ok(default_model_status(models_dir))
}

/// Downloads a single model file to `destination` through a temporary `.part` file.
///
/// # Errors
///
/// Returns an error when the HTTP request fails, the server returns a non-2xx
/// status, or local file I/O fails.
pub fn download_model_file(url: &str, destination: &Path) -> ModelDownloadResult<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }

    let temporary = destination.with_extension(format!(
        "{}.part",
        destination
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("download")
    ));

    let mut response = ureq::get(url)
        .call()
        .map_err(|err| ModelDownloadError::Http(err.to_string()))?;
    if !response.status().is_success() {
        return Err(ModelDownloadError::Http(format!(
            "GET {url} returned {}",
            response.status()
        )));
    }

    let mut reader = response.body_mut().as_reader();
    let mut writer = BufWriter::new(fs::File::create(&temporary)?);
    io::copy(&mut reader, &mut writer)?;
    writer.flush()?;
    drop(writer);

    fs::rename(&temporary, destination)?;
    Ok(())
}

fn download_model_file_with_progress(
    file: DefaultModelFile,
    destination: &Path,
    mut on_progress: impl FnMut(ModelDownloadProgress),
) -> ModelDownloadResult<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }

    let temporary = destination.with_extension(format!(
        "{}.part",
        destination
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("download")
    ));

    let mut response = ureq::get(file.url)
        .call()
        .map_err(|err| ModelDownloadError::Http(err.to_string()))?;
    if !response.status().is_success() {
        return Err(ModelDownloadError::Http(format!(
            "GET {} returned {}",
            file.url,
            response.status()
        )));
    }
    let total_bytes = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());

    let mut reader = response.body_mut().as_reader();
    let mut writer = BufWriter::new(fs::File::create(&temporary)?);
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut downloaded_bytes = 0_u64;
    on_progress(ModelDownloadProgress {
        role: file.role,
        file_name: file.file_name,
        downloaded_bytes,
        total_bytes,
        finished: false,
    });

    loop {
        let read_bytes = reader.read(&mut buffer)?;
        if read_bytes == 0 {
            break;
        }
        writer.write_all(&buffer[..read_bytes])?;
        downloaded_bytes += u64::try_from(read_bytes).unwrap_or(0);
        on_progress(ModelDownloadProgress {
            role: file.role,
            file_name: file.file_name,
            downloaded_bytes,
            total_bytes,
            finished: false,
        });
    }
    writer.flush()?;
    drop(writer);

    fs::rename(&temporary, destination)?;
    on_progress(ModelDownloadProgress {
        role: file.role,
        file_name: file.file_name,
        downloaded_bytes,
        total_bytes,
        finished: true,
    });
    Ok(())
}

#[must_use]
pub fn resolve_url_for_file(file_name: &str) -> String {
    format!("{HF_RESOLVE_BASE}/{file_name}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_paths_point_into_models_dir() {
        let model_set = default_model_set("models");

        assert_eq!(
            model_set.talker.path,
            PathBuf::from("models").join(DEFAULT_TALKER_FILE)
        );
        assert_eq!(
            model_set.codec.path,
            PathBuf::from("models").join(DEFAULT_CODEC_FILE)
        );
    }

    #[test]
    fn default_status_reports_missing_files_without_download() {
        let unique_dir =
            std::env::temp_dir().join(format!("qwen-tts-missing-models-{}", std::process::id()));
        let _ = fs::remove_dir_all(&unique_dir);

        let status = default_model_status(&unique_dir);

        assert!(!status.is_complete());
        assert_eq!(status.missing_files().len(), DEFAULT_MODEL_FILES.len());
    }

    #[test]
    fn default_urls_match_expected_hugging_face_files() {
        for file in DEFAULT_MODEL_FILES {
            assert_eq!(file.url, resolve_url_for_file(file.file_name));
        }
    }
}
