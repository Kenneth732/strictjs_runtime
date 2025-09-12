
// src/utils/boolean.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct StrictBoolean {
    value: bool,
}

#[wasm_bindgen]
impl StrictBoolean {
    #[wasm_bindgen(constructor)]
    pub fn new(value: bool) -> StrictBoolean {
        StrictBoolean { value }
    }

    #[wasm_bindgen(js_name = get)]
    pub fn get(&self) -> bool {
        self.value
    }

    #[wasm_bindgen(js_name = set)]
    pub fn set(&mut self, value: bool) {
        self.value = value;
    }

    #[wasm_bindgen(js_name = toggle)]
    pub fn toggle(&mut self) {
        self.value = !self.value;
    }

    #[wasm_bindgen(js_name = and)]
    pub fn and(&self, other: bool) -> bool {
        self.value && other
    }

    #[wasm_bindgen(js_name = or)]
    pub fn or(&self, other: bool) -> bool {
        self.value || other
    }

    #[wasm_bindgen(js_name = not)]
    pub fn not(&self) -> bool {
        !self.value
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        self.value.to_string()
    }
}


