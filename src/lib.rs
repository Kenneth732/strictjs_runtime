
// lib.rs
mod types;
mod strict_array;
mod strict_number;
mod strict_string;
mod strict_object;
mod strict_bigint;
mod strict_function;
// mod reactive_system;
mod loops;
mod strict_async;
mod threads;
mod utils;

pub use types::HeapType;
pub use types::Schema;
// Add GPU types exports
pub use types::{JsGPUType, GPUMemoryManager, GPUBufferInfo, GPUBufferUsage, create_gpu_type, get_available_gpu_types};

pub use strict_array::StrictArray;
pub use strict_number::StrictNumber;
pub use strict_string::{StrictString, StringEncoding};
pub use strict_object::StrictObject; 
pub use utils::StrictBoolean;
pub use strict_bigint::StrictBigInt;
pub use strict_function::StrictFunction;
// Change this line to import from the module directly
pub use strict_async::{
    StrictAsync, StrictPromise, StrictTimeout, strict_fetch
};

#[wasm_bindgen]
pub fn init_thread_manager(config: JsValue) -> Result<threads::ThreadManager, JsValue> {
    threads::ThreadManager::new(config)
}

// pub use reactive_system::{ReactiveCell, ReactiveSystem, Computed};
pub use loops::{StrictForLoop, StrictWhileLoop};

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn get_memory() -> JsValue {
    wasm_bindgen::memory()
}

// You can also add some GPU-specific helper functions if needed
#[wasm_bindgen]
pub fn init_gpu_memory_manager() -> GPUMemoryManager {
    GPUMemoryManager::new()
}

#[wasm_bindgen]
pub fn create_gpu_tensor_f32() -> JsGPUType {
    JsGPUType::new("tensor_f32").unwrap()
}

#[wasm_bindgen]
pub fn create_gpu_matrix_f32() -> JsGPUType {
    JsGPUType::new("matrix_f32").unwrap()
}

#[wasm_bindgen]
pub fn create_gpu_vector_f32() -> JsGPUType {
    JsGPUType::new("vector_f32").unwrap()
}

#[cfg(test)]
mod tests;
