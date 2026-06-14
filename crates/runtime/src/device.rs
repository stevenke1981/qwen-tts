use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Auto,
    Cpu,
    Cuda,
    Rocm,
    Metal,
    Wgpu,
    Sycl,
}

impl FromStr for DeviceKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "rocm" | "hip" => Ok(Self::Rocm),
            "metal" => Ok(Self::Metal),
            "wgpu" | "vulkan" => Ok(Self::Wgpu),
            "sycl" | "oneapi" => Ok(Self::Sycl),
            other => Err(format!("unknown device kind: {other}")),
        }
    }
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Metal => "metal",
            Self::Wgpu => "wgpu",
            Self::Sycl => "sycl",
        };
        write!(f, "{value}")
    }
}
