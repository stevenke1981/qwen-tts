use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct TalkerModel {
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct CodecModel {
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct TtsModelSet {
    pub talker: TalkerModel,
    pub codec: CodecModel,
}

impl TtsModelSet {
    #[must_use]
    pub fn new(talker: impl Into<PathBuf>, codec: impl Into<PathBuf>) -> Self {
        Self {
            talker: TalkerModel {
                path: talker.into(),
            },
            codec: CodecModel { path: codec.into() },
        }
    }
}
