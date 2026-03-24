//! Tensor attributes and related functionality.

// SliceParamAttr and SliceParamsAttr live in the memref dialect since they are
// shared between memref and tensor slice ops.
pub use crate::memref::attributes::{SliceParamAttr, SliceParamsAttr};
