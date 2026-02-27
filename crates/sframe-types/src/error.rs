use thiserror::Error;

#[derive(Error, Debug)]
pub enum SFrameError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Type error: {0}")]
    Type(String),

    #[error("Format error: {0}")]
    Format(String),
}

pub type Result<T> = std::result::Result<T, SFrameError>;
