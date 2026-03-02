pub use klearu_core as core;

#[cfg(feature = "simd")]
pub use klearu_accel as accel;

#[cfg(feature = "bf16")]
pub use klearu_accel as accel_bf16;

#[cfg(feature = "mongoose")]
pub use klearu_mongoose as mongoose;

#[cfg(feature = "bolt")]
pub use klearu_bolt as bolt;

#[cfg(feature = "deja-vu")]
pub use klearu_dejavu as dejavu;

#[cfg(feature = "llm")]
pub use klearu_llm as llm;

/// Prelude for convenient imports.
pub mod prelude {
    pub use klearu_core::config::*;
    pub use klearu_core::data::{Dataset, Example};
    pub use klearu_core::error::{KlearuError, Result};
    pub use klearu_core::network::Network;
    pub use klearu_core::tensor::{AlignedVec, SparseVector};
}
