use std::{
    collections::BTreeMap,
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
    InvalidUtf8(std::string::FromUtf8Error),
    LengthOverflow(u64),
    UnsupportedValueType(u32),
}

impl fmt::Display for GgufProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::InvalidMagic(magic) => write!(f, "invalid GGUF magic: {magic:?}"),
            Self::InvalidUtf8(err) => write!(f, "invalid GGUF UTF-8 string: {err}"),
            Self::LengthOverflow(len) => write!(f, "GGUF metadata length is too large: {len}"),
            Self::UnsupportedValueType(value_type) => {
                write!(f, "unsupported GGUF metadata value type: {value_type}")
            }
        }
    }
}

impl std::error::Error for GgufProbeError {}

impl From<std::io::Error> for GgufProbeError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<std::string::FromUtf8Error> for GgufProbeError {
    fn from(value: std::string::FromUtf8Error) -> Self {
        Self::InvalidUtf8(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValue {
    U32(u32),
    F32(f32),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufMetadata {
    values: BTreeMap<String, GgufMetadataValue>,
}

impl GgufMetadata {
    /// Reads only selected metadata keys from a GGUF file.
    ///
    /// Large string arrays, such as tokenizer vocabularies, are skipped without
    /// allocation unless their key is explicitly requested.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, the magic bytes are
    /// invalid, UTF-8 metadata is malformed, or a selected value has an
    /// unsupported type.
    pub fn read_selected(
        path: impl AsRef<Path>,
        selected_keys: &[&str],
    ) -> Result<Self, GgufProbeError> {
        let mut file = File::open(path)?;
        read_header(&mut file)?;
        let _version = read_u32_le(&mut file)?;
        let _tensor_count = read_u64_le(&mut file)?;
        let metadata_kv_count = read_u64_le(&mut file)?;
        let mut values = BTreeMap::new();

        for _ in 0..metadata_kv_count {
            let key = read_string(&mut file)?;
            let value_type = read_u32_le(&mut file)?;
            if selected_keys.iter().any(|selected| *selected == key) {
                let value = read_metadata_value(&mut file, value_type)?;
                values.insert(key, value);
            } else {
                skip_metadata_value(&mut file, value_type)?;
            }
        }

        Ok(Self { values })
    }

    #[must_use]
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.values.get(key) {
            Some(GgufMetadataValue::String(value)) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.values.get(key) {
            Some(GgufMetadataValue::U32(value)) => Some(*value),
            _ => None,
        }
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
        let magic = read_header(&mut file)?;
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

fn read_header(reader: &mut impl Read) -> Result<[u8; 4], GgufProbeError> {
    let mut magic = [0_u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(GgufProbeError::InvalidMagic(magic));
    }
    Ok(magic)
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

fn read_string(reader: &mut impl Read) -> Result<String, GgufProbeError> {
    let len = read_u64_le(reader)?;
    let len = usize::try_from(len).map_err(|_| GgufProbeError::LengthOverflow(len))?;
    let mut bytes = vec![0_u8; len];
    reader.read_exact(&mut bytes)?;
    Ok(String::from_utf8(bytes)?)
}

fn read_metadata_value(
    reader: &mut impl Read,
    value_type: u32,
) -> Result<GgufMetadataValue, GgufProbeError> {
    match value_type {
        4 => Ok(GgufMetadataValue::U32(read_u32_le(reader)?)),
        6 => {
            let mut buf = [0_u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::F32(f32::from_le_bytes(buf)))
        }
        7 => {
            let mut buf = [0_u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(GgufMetadataValue::Bool(buf[0] != 0))
        }
        8 => Ok(GgufMetadataValue::String(read_string(reader)?)),
        other => Err(GgufProbeError::UnsupportedValueType(other)),
    }
}

fn skip_metadata_value(
    reader: &mut (impl Read + Seek),
    value_type: u32,
) -> Result<(), GgufProbeError> {
    match value_type {
        0 | 1 | 7 => skip_bytes(reader, 1)?,
        2 | 3 => skip_bytes(reader, 2)?,
        4..=6 => skip_bytes(reader, 4)?,
        8 => {
            let len = read_u64_le(reader)?;
            skip_bytes(reader, len)?;
        }
        9 => {
            let element_type = read_u32_le(reader)?;
            let len = read_u64_le(reader)?;
            for _ in 0..len {
                skip_metadata_value(reader, element_type)?;
            }
        }
        10..=12 => skip_bytes(reader, 8)?,
        other => return Err(GgufProbeError::UnsupportedValueType(other)),
    }
    Ok(())
}

fn skip_bytes(reader: &mut impl Seek, len: u64) -> Result<(), std::io::Error> {
    let len = i64::try_from(len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "GGUF metadata skip length is too large",
        )
    })?;
    reader.seek(SeekFrom::Current(len))?;
    Ok(())
}
