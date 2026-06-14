/// Placeholder for backend-independent common ops.
/// Keep this crate CPU/GPU neutral; backend-specific kernels belong under `crates/backends/*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I32,
    Q4KM,
    Q8_0,
}
