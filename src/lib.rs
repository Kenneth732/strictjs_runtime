
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



#[cfg(test)]
mod tests;



