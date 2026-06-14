use std::{
    env, fmt, fs, io,
    path::{Path, PathBuf},
    process::Command,
};

pub const DEFAULT_QWENTTS_REPO: &str = "https://github.com/ServeurpersoCom/qwentts.cpp.git";

#[derive(Debug)]
pub enum BackendSetupError {
    Io(io::Error),
    CommandFailed {
        program: String,
        status: Option<i32>,
        stderr: String,
    },
    MissingExecutable(PathBuf),
}

impl fmt::Display for BackendSetupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "backend setup I/O error: {err}"),
            Self::CommandFailed {
                program,
                status,
                stderr,
            } => write!(
                f,
                "backend setup command failed: {program}; status={status:?}; stderr={stderr}"
            ),
            Self::MissingExecutable(path) => {
                write!(
                    f,
                    "qwentts backend executable was not built at {}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for BackendSetupError {}

impl From<io::Error> for BackendSetupError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub type BackendSetupResult<T> = Result<T, BackendSetupError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendStatus {
    pub source_dir: PathBuf,
    pub expected_executable: PathBuf,
    pub resolved_executable: Option<PathBuf>,
}

impl BackendStatus {
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.resolved_executable.is_some()
    }
}

#[must_use]
pub fn backend_status(project_root: impl AsRef<Path>, configured: Option<&Path>) -> BackendStatus {
    let project_root = project_root.as_ref();
    BackendStatus {
        source_dir: default_backend_source_dir(project_root),
        expected_executable: default_backend_executable(project_root),
        resolved_executable: find_qwentts_executable(project_root, configured),
    }
}

#[must_use]
pub fn default_backend_source_dir(project_root: impl AsRef<Path>) -> PathBuf {
    project_root.as_ref().join("vendor").join("qwentts.cpp")
}

#[must_use]
pub fn default_backend_executable(project_root: impl AsRef<Path>) -> PathBuf {
    let build_dir = default_backend_source_dir(project_root).join("build");
    if cfg!(windows) {
        build_dir.join("Release").join("qwen-tts.exe")
    } else {
        build_dir.join("qwen-tts")
    }
}

#[must_use]
pub fn find_qwentts_executable(
    project_root: impl AsRef<Path>,
    configured: Option<&Path>,
) -> Option<PathBuf> {
    if let Some(path) = configured.filter(|path| path.is_file()) {
        return Some(path.to_path_buf());
    }
    if let Some(path) = env::var_os("QWEN_TTS_BIN")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
    {
        return Some(path);
    }

    backend_executable_candidates(project_root)
        .into_iter()
        .find(|path| path.is_file())
}

#[must_use]
pub fn backend_executable_candidates(project_root: impl AsRef<Path>) -> Vec<PathBuf> {
    let project_root = project_root.as_ref();
    let source_dir = default_backend_source_dir(project_root);
    let mut candidates = Vec::new();

    if cfg!(windows) {
        candidates.extend([
            source_dir
                .join("build")
                .join("Release")
                .join("qwen-tts.exe"),
            source_dir.join("build").join("qwen-tts.exe"),
            source_dir
                .join("build")
                .join("bin")
                .join("Release")
                .join("qwen-tts.exe"),
            source_dir.join("build").join("bin").join("qwen-tts.exe"),
        ]);
    } else {
        candidates.extend([
            source_dir.join("build").join("qwen-tts"),
            source_dir.join("build").join("bin").join("qwen-tts"),
        ]);
    }

    if cfg!(windows) {
        candidates.extend(sibling_backend_candidates(project_root));
    }

    candidates
}

#[cfg(windows)]
fn sibling_backend_candidates(project_root: &Path) -> Vec<PathBuf> {
    let Some(parent) = project_root.parent() else {
        return Vec::new();
    };
    ["qwen3_tts_rust_app"]
        .into_iter()
        .map(|name| {
            parent
                .join(name)
                .join("qwentts.cpp")
                .join("build")
                .join("Release")
                .join("qwen-tts.exe")
        })
        .collect()
}

#[cfg(not(windows))]
fn sibling_backend_candidates(_project_root: &Path) -> Vec<PathBuf> {
    Vec::new()
}

/// Clones or updates qwentts.cpp and builds the CPU backend.
///
/// # Errors
///
/// Returns an error when git/cmake fail, filesystem operations fail, or the
/// expected backend executable is not produced.
pub fn setup_qwentts_backend(project_root: impl AsRef<Path>) -> BackendSetupResult<BackendStatus> {
    let project_root = project_root.as_ref();
    let source_dir = default_backend_source_dir(project_root);
    fs::create_dir_all(source_dir.parent().unwrap_or(project_root))?;

    if source_dir.join(".git").is_dir() {
        run_command("git", &["pull", "--ff-only"], Some(&source_dir))?;
        run_command(
            "git",
            &["submodule", "update", "--init", "--recursive"],
            Some(&source_dir),
        )?;
    } else {
        let vendor_dir = source_dir.parent().unwrap_or(project_root);
        run_command(
            "git",
            &[
                "clone",
                "--recurse-submodules",
                DEFAULT_QWENTTS_REPO,
                "qwentts.cpp",
            ],
            Some(vendor_dir),
        )?;
    }

    let build_dir = source_dir.join("build");
    fs::create_dir_all(&build_dir)?;
    let source_arg = source_dir.display().to_string();
    let build_arg = build_dir.display().to_string();
    if run_command(
        "cmake",
        &["-S", &source_arg, "-B", &build_arg, "-DGGML_BLAS=ON"],
        Some(project_root),
    )
    .is_err()
    {
        run_command(
            "cmake",
            &["-S", &source_arg, "-B", &build_arg, "-DGGML_BLAS=OFF"],
            Some(project_root),
        )?;
    }
    run_command(
        "cmake",
        &[
            "--build", &build_arg, "--config", "Release", "--target", "qwen-tts",
        ],
        Some(project_root),
    )?;

    let status = backend_status(project_root, None);
    if status.resolved_executable.is_none() {
        return Err(BackendSetupError::MissingExecutable(
            status.expected_executable.clone(),
        ));
    }
    Ok(status)
}

fn run_command(program: &str, args: &[&str], cwd: Option<&Path>) -> BackendSetupResult<()> {
    let mut command = Command::new(program);
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let output = command.output()?;
    if output.status.success() {
        return Ok(());
    }

    Err(BackendSetupError::CommandFailed {
        program: format!("{} {}", program, args.join(" ")),
        status: output.status.code(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_backend_path_points_under_vendor() {
        let root = PathBuf::from("repo");
        assert!(default_backend_executable(&root).starts_with(root.join("vendor")));
    }

    #[test]
    fn configured_existing_executable_wins() {
        let temp_dir = env::temp_dir().join(format!("qwen-backend-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        let exe = temp_dir.join(if cfg!(windows) {
            "qwen-tts.exe"
        } else {
            "qwen-tts"
        });
        fs::write(&exe, b"stub").unwrap();

        assert_eq!(find_qwentts_executable("repo", Some(&exe)), Some(exe));

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
