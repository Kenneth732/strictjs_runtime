
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct StrictWhileLoop {
    max_iterations: usize,
    iteration_count: usize,
    condition: js_sys::Function,
}

#[wasm_bindgen]
impl StrictWhileLoop {
    #[wasm_bindgen(constructor)]
    pub fn new(condition: js_sys::Function, max_iterations: usize) -> Self {
        StrictWhileLoop {
            max_iterations,
            iteration_count: 0,
            condition,
        }
    }

    #[wasm_bindgen(js_name = shouldContinue)]
    pub fn should_continue(&mut self) -> Result<bool, JsValue> {
        if self.iteration_count >= self.max_iterations {
            return Ok(false);
        }

        let condition_result = self.condition.call0(&JsValue::NULL)?;
        condition_result.as_bool().ok_or_else(|| JsValue::from_str("Condition must return boolean"))
    }

    #[wasm_bindgen(js_name = increment)]
    pub fn increment(&mut self) -> Result<usize, JsValue> {
        if self.iteration_count >= self.max_iterations {
            return Err(JsValue::from_str("Maximum iterations exceeded"));
        }

        self.iteration_count += 1;
        Ok(self.iteration_count)
    }

    #[wasm_bindgen(js_name = getIterationCount)]
    pub fn get_iteration_count(&self) -> usize {
        self.iteration_count
    }

    #[wasm_bindgen(js_name = reset)]
    pub fn reset(&mut self) {
        self.iteration_count = 0;
    }

    #[wasm_bindgen(js_name = run)]
    pub fn run(&mut self, callback: js_sys::Function) -> Result<usize, JsValue> {
        while self.should_continue()? {
            let iteration = self.increment()?;
            callback.call1(&JsValue::NULL, &JsValue::from(iteration))?;
        }
        Ok(self.iteration_count)
    }
}

