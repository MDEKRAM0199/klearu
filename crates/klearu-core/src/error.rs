use thiserror::Error;

#[derive(Error, Debug)]
pub enum KlearuError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("index out of bounds: {index} >= {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    #[error("empty input: {0}")]
    EmptyInput(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("network error: {0}")]
    Network(String),

    #[error("training error: {0}")]
    Training(String),

    #[error("serialization error: {0}")]
    Serialization(String),
}

pub type Result<T> = std::result::Result<T, KlearuError>;
