use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum OptimizationMode {
    Sequential,
    Batched,
    GPU,
    SIMD,
    Auto,
}