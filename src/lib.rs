
// lib.rs
mod types;
mod strict_array;
mod strict_number;
mod strict_string;
mod strict_object;
mod strict_bigint;
mod strict_function;
mod loops;

mod strict_async;
mod threads;
mod utils;

pub use types::HeapType;
pub use types::Schema;
// Add GPU types exports
pub use types::{JsGPUType, GPUMemoryManager, GPUBufferInfo, GPUBufferUsage, create_gpu_type, get_available_gpu_types};
// Add SIMD types exports
pub use types::{JsSIMDType, create_simd_type, get_available_simd_types, get_simd_type_for_use_case};

pub use strict_array::StrictArray;
pub use strict_number::StrictNumber;
pub use strict_string::{StrictString, StringEncoding};
pub use strict_object::StrictObject; 
pub use utils::StrictBoolean;
pub use strict_bigint::StrictBigInt;
pub use strict_function::StrictFunction;
pub use strict_async::{
    StrictAsync, StrictPromise, StrictTimeout, strict_fetch
};

#[wasm_bindgen]
pub fn init_thread_manager(config: JsValue) -> Result<threads::ThreadManager, JsValue> {
    threads::ThreadManager::new(config)
}

pub use loops::{StrictForLoop, StrictWhileLoop};

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn get_memory() -> JsValue {
    wasm_bindgen::memory()
}

// SIMD helper functions
#[wasm_bindgen]
pub fn create_simd_f32x4() -> JsSIMDType {
    JsSIMDType::new("f32x4").unwrap()
}

#[wasm_bindgen]
pub fn create_simd_i32x4() -> JsSIMDType {
    JsSIMDType::new("i32x4").unwrap()
}

#[wasm_bindgen]
pub fn create_simd_u8x16() -> JsSIMDType {
    JsSIMDType::new("u8x16").unwrap()
}

#[cfg(test)]
mod tests;

