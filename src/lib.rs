//! Raw FFI bindings to MNN (Mobile Neural Network) by Alibaba.
//!
//! This crate provides low-level unsafe bindings to the MNN library.
//! For a safe Rust API, use the `mnn` crate instead.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bindings_exist() {
        // Just ensure the types are available
        let _: MNNC_ForwardType = MNNC_ForwardType_MNNC_FORWARD_CPU;
        let _: MNNC_DataType = MNNC_DataType_MNNC_DTYPE_FLOAT;
        let _: MNNC_ErrorCode = MNNC_ErrorCode_MNNC_OK;
    }
}
