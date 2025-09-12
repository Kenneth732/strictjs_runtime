
// src/strict_number/implementation.rs
use wasm_bindgen::prelude::*;
use crate::types::{HeapType, Num};

#[wasm_bindgen]
pub struct StrictNumber {
    value: Num,
    heap: HeapType,
}

#[wasm_bindgen]
impl StrictNumber {
    #[wasm_bindgen(constructor)]
    pub fn new(val: f64, heap: HeapType) -> StrictNumber {
        StrictNumber {
            value: Num::from_f64(&heap, val),
            heap,
        }
    }

    pub fn heap(&self) -> HeapType {
        self.heap
    }

    pub fn get(&self) -> f64 {
        self.value.to_f64()
    }

    pub fn set(&mut self, val: f64) {
        self.value = Num::from_f64(&self.heap, val);
    }

    pub fn add(&mut self, delta: f64) {
        self.value.add_assign(&self.heap, delta);
    }

    pub fn sub(&mut self, delta: f64) {
        self.value.sub_assign(&self.heap, delta);
    }

    #[wasm_bindgen(js_name = mul)]
    pub fn mul(&mut self, factor: f64) {
        let cur = self.value.to_f64();
        self.value = Num::from_f64(&self.heap, cur * factor);
    }
    
    #[wasm_bindgen(js_name = div)]
    pub fn div(&mut self, divisor: f64) {
        let cur = self.value.to_f64(); // Define cur here
        
        if divisor == 0.0 {
            // Handle division by zero (clamp to max/min)
            let clamped = match self.heap {
                HeapType::U8 | HeapType::U16 | HeapType::U32 => f64::MAX,
                _ => if cur < 0.0 { f64::MIN } else { f64::MAX },
            };
            self.value = Num::from_f64(&self.heap, clamped);
        } else {
            self.value = Num::from_f64(&self.heap, cur / divisor);
        }
    }
    
    // JavaScript's valueOf() equivalent
    #[wasm_bindgen(js_name = valueOf)]
    pub fn value_of(&self) -> f64 {
        self.value.to_f64()
    }
}

