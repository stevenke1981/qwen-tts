use std::{
    fmt,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufProbe {
    pub magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

#[derive(Debug)]
pub enum GgufProbeError {
    Io(std::io::Error),
    InvalidMagic([u8; 4]),
}

impl fmt::Display for GgufProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::InvalidMagic(magic) => write!(f, "invalid GGUF magic: {magic:?}"),
        }
    }
}

impl std::error::Error for GgufProbeError {}

impl From<std::io::Error> for GgufProbeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl GgufProbe {
    /// Minimal GGUF header probe. For full metadata parsing, swap this for
    /// llama-gguf / gguf-rs later.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or its magic bytes are not
    /// the expected `GGUF` marker.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufProbeError> {
        let mut file = File::open(path)?;
        let mut magic = [0_u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(GgufProbeError::InvalidMagic(magic));
        }

        let version = read_u32_le(&mut file)?;
        let tensor_count = read_u64_le(&mut file)?;
        let metadata_kv_count = read_u64_le(&mut file)?;
        file.seek(SeekFrom::Start(0))?;

        Ok(Self {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }
}

fn read_u32_le(reader: &mut impl Read) -> Result<u32, std::io::Error> {
    let mut buf = [0_u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(reader: &mut impl Read) -> Result<u64, std::io::Error> {
    let mut buf = [0_u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}
