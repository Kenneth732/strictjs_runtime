
// src/loops/for_loop.rs
use crate::types::HeapType;
use crate::utils::clamp_f64;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct StrictForLoop {
    current: f64,
    end: f64,
    step: f64,
    heap_type: HeapType,
    max_iterations: usize,
    iteration_count: usize,
}

#[wasm_bindgen]
impl StrictForLoop {
    #[wasm_bindgen(constructor)]
    pub fn new(
        start: f64,
        end: f64,
        step: f64,
        heap_type: HeapType,
        max_iterations: usize,
    ) -> Self {
        // Clamp initial values based on heap type
        let clamped_start = Self::clamp_value(start, heap_type);
        let clamped_end = Self::clamp_value(end, heap_type);
        let clamped_step = Self::clamp_value(step, heap_type);

        StrictForLoop {
            current: clamped_start,
            end: clamped_end,
            step: clamped_step,
            heap_type,
            max_iterations,
            iteration_count: 0,
        }
    }

    #[wasm_bindgen(js_name = hasNext)]
    pub fn has_next(&self) -> bool {
        // Check bounds and iteration limit
        ((self.step > 0.0 && self.current <= self.end)
            || (self.step < 0.0 && self.current >= self.end))
            && self.iteration_count < self.max_iterations
    }

    #[wasm_bindgen(js_name = next)]
    pub fn next(&mut self) -> Result<f64, JsValue> {
        if self.iteration_count >= self.max_iterations {
            return Err(JsValue::from_str("Maximum iterations exceeded"));
        }

        if !self.has_next() {
            return Err(JsValue::from_str("Loop completed"));
        }

        let current_value = self.current;
        self.current = Self::clamp_value(self.current + self.step, self.heap_type);
        self.iteration_count += 1;

        Ok(current_value)
    }

    #[wasm_bindgen(js_name = getIterationCount)]
    pub fn get_iteration_count(&self) -> usize {
        self.iteration_count
    }

    #[wasm_bindgen(js_name = reset)]
    pub fn reset(&mut self) {
        let start_value = self.current - (self.step * self.iteration_count as f64);
        self.current = Self::clamp_value(start_value, self.heap_type);
        self.iteration_count = 0;
    }

    #[wasm_bindgen(js_name = getCurrent)]
    pub fn get_current(&self) -> f64 {
        self.current
    }

    #[wasm_bindgen(js_name = getProgress)]
    pub fn get_progress(&self) -> f64 {
        if self.max_iterations == 0 {
            return 0.0;
        }
        (self.iteration_count as f64 / self.max_iterations as f64) * 100.0
    }

    #[wasm_bindgen(js_name = getStep)]
    pub fn get_step(&self) -> f64 {
        self.step
    }

    #[wasm_bindgen(js_name = getEnd)]
    pub fn get_end(&self) -> f64 {
        self.end
    }

    #[wasm_bindgen(js_name = getHeapType)]
    pub fn get_heap_type(&self) -> HeapType {
        self.heap_type
    }

    fn clamp_value(value: f64, heap_type: HeapType) -> f64 {
        match heap_type {
            HeapType::Number => value,
            HeapType::U8 => clamp_f64(value, 0.0, u8::MAX as f64),
            HeapType::I8 => clamp_f64(value, i8::MIN as f64, i8::MAX as f64),
            HeapType::U16 => clamp_f64(value, 0.0, u16::MAX as f64),
            HeapType::I16 => clamp_f64(value, i16::MIN as f64, i16::MAX as f64),
            HeapType::U32 => clamp_f64(value, 0.0, u32::MAX as f64),
            HeapType::I32 => clamp_f64(value, i32::MIN as f64, i32::MAX as f64),
            HeapType::U64 => clamp_f64(value, 0.0, u64::MAX as f64),
            HeapType::I64 => clamp_f64(value, i64::MIN as f64, i64::MAX as f64),
            HeapType::F32 => value as f32 as f64, // Convert to f32 and back to f64
            HeapType::F64 => value,
            HeapType::Bool => {
                if value != 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            HeapType::Str => value,
            HeapType::Str16 => value,
            HeapType::Any => value,
            HeapType::Struct => value,
            HeapType::Array => value,
            HeapType::Map => value,
            HeapType::Date => value,
            HeapType::Buffer => value,
            HeapType::Symbol => value,
            HeapType::Null => value,
            HeapType::Undefined => value,
        }
    }
}


