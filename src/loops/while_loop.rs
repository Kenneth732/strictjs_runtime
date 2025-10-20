// src/loops/while_loop.rs
use wasm_bindgen::prelude::*;
use js_sys::{Function, Array, Object, Reflect};
use crate::types::OptimizationMode;


// #[wasm_bindgen]
// #[derive(Clone, Copy, Debug)]
// pub enum OptimizationMode {
//     Sequential,
//     Batched,
//     GPU,
//     SIMD,
//     Auto,
// }

#[wasm_bindgen]
pub struct StrictWhileLoop {
    max_iterations: usize,
    iteration_count: usize,
    condition: Function,
    optimization_mode: OptimizationMode,
    batch_size: usize,
    progress_callback: Option<Function>,
    condition_cache: Option<bool>,
    last_condition_check: usize,
}

#[wasm_bindgen]
impl StrictWhileLoop {
    #[wasm_bindgen(constructor)]
    pub fn new(condition: Function, max_iterations: usize) -> Self {
        StrictWhileLoop {
            max_iterations,
            iteration_count: 0,
            condition,
            optimization_mode: OptimizationMode::Sequential,
            batch_size: 1,
            progress_callback: None,
            condition_cache: None,
            last_condition_check: 0,
        }
    }

    #[wasm_bindgen(js_name = newWithOptimization)]
    pub fn new_with_optimization(
        condition: Function,
        max_iterations: usize,
        optimization_mode: OptimizationMode,
        batch_size: usize,
    ) -> Self {
        StrictWhileLoop {
            max_iterations,
            iteration_count: 0,
            condition,
            optimization_mode,
            batch_size,
            progress_callback: None,
            condition_cache: None,
            last_condition_check: 0,
        }
    }

    #[wasm_bindgen(js_name = shouldContinue)]
    pub fn should_continue(&mut self) -> Result<bool, JsValue> {
        if self.iteration_count >= self.max_iterations {
            return Ok(false);
        }

        if let Some(cached) = self.condition_cache {
            if self.iteration_count == self.last_condition_check {
                return Ok(cached);
            }
        }

        let condition_result = self.condition.call0(&JsValue::NULL)?;
        let should_continue = condition_result.as_bool()
            .ok_or_else(|| JsValue::from_str("Condition must return boolean"))?;

        self.condition_cache = Some(should_continue);
        self.last_condition_check = self.iteration_count;

        Ok(should_continue)
    }

    #[wasm_bindgen(js_name = shouldContinueBatch)]
    pub fn should_continue_batch(&mut self, batch_size: usize) -> Result<bool, JsValue> {
        if self.iteration_count >= self.max_iterations {
            return Ok(false);
        }

        let remaining_iterations = self.max_iterations - self.iteration_count;
        if remaining_iterations >= batch_size {
            let condition_result = self.condition.call0(&JsValue::NULL)?;
            condition_result.as_bool()
                .ok_or_else(|| JsValue::from_str("Condition must return boolean"))
        } else {
            self.should_continue()
        }
    }

    #[wasm_bindgen(js_name = increment)]
    pub fn increment(&mut self) -> Result<usize, JsValue> {
        if self.iteration_count >= self.max_iterations {
            return Err(JsValue::from_str("Maximum iterations exceeded"));
        }

        self.iteration_count += 1;
        self.call_progress_callback()?;
        Ok(self.iteration_count)
    }

    #[wasm_bindgen(js_name = incrementBatch)]
    pub fn increment_batch(&mut self, batch_size: usize) -> Result<usize, JsValue> {
        if self.iteration_count >= self.max_iterations {
            return Err(JsValue::from_str("Maximum iterations exceeded"));
        }

        let actual_batch_size = std::cmp::min(batch_size, self.max_iterations - self.iteration_count);
        self.iteration_count += actual_batch_size;
        self.call_progress_callback()?;
        Ok(self.iteration_count)
    }

    #[wasm_bindgen(js_name = run)]
    pub fn run(&mut self, callback: Function) -> Result<usize, JsValue> {
        while self.should_continue()? {
            let iteration = self.increment()?;
            callback.call1(&JsValue::NULL, &JsValue::from(iteration))?;
        }
        Ok(self.iteration_count)
    }

    #[wasm_bindgen(js_name = runBatch)]
    pub fn run_batch(&mut self, callback: Function) -> Result<usize, JsValue> {
        while self.should_continue_batch(self.batch_size)? {
            let batch_start = self.iteration_count;
            let iteration = self.increment_batch(self.batch_size)?;
            
            let batch_info = Object::new();
            Reflect::set(&batch_info, &"start".into(), &(batch_start as f64).into())?;
            Reflect::set(&batch_info, &"end".into(), &(iteration as f64).into())?;
            Reflect::set(&batch_info, &"size".into(), &((iteration - batch_start) as f64).into())?;
            
            callback.call1(&JsValue::NULL, &batch_info.into())?;
        }
        Ok(self.iteration_count)
    }

    #[wasm_bindgen(js_name = runUntilConvergence)]
    pub fn run_until_convergence(
        &mut self, 
        callback: Function,
        convergence_threshold: f64,
        max_iterations_without_improvement: usize,
    ) -> Result<JsValue, JsValue> {
        let mut best_value = f64::MAX;
        let mut iterations_without_improvement = 0;
        let history = Array::new();

        while self.should_continue()? && iterations_without_improvement < max_iterations_without_improvement {
            let iteration = self.increment()?;
            
            let result = callback.call1(&JsValue::NULL, &JsValue::from(iteration))?;
            let current_value = result.as_f64()
                .ok_or_else(|| JsValue::from_str("Callback must return a numeric value"))?;

            history.push(&JsValue::from_f64(current_value));

            if current_value < best_value - convergence_threshold {
                best_value = current_value;
                iterations_without_improvement = 0;
            } else {
                iterations_without_improvement += 1;
            }
        }

        let convergence_info = Object::new();
        Reflect::set(&convergence_info, &"iterations".into(), &(self.iteration_count as f64).into())?;
        Reflect::set(&convergence_info, &"bestValue".into(), &best_value.into())?;
        Reflect::set(&convergence_info, &"converged".into(), &(iterations_without_improvement < max_iterations_without_improvement).into())?;
        Reflect::set(&convergence_info, &"history".into(), &history.into())?;

        Ok(convergence_info.into())
    }

    #[wasm_bindgen(js_name = setProgressCallback)]
    pub fn set_progress_callback(&mut self, callback: Function) {
        self.progress_callback = Some(callback);
    }

    #[wasm_bindgen(js_name = getIterationCount)]
    pub fn get_iteration_count(&self) -> usize {
        self.iteration_count
    }

    #[wasm_bindgen(js_name = reset)]
    pub fn reset(&mut self) {
        self.iteration_count = 0;
        self.condition_cache = None;
        self.last_condition_check = 0;
    }

    #[wasm_bindgen(js_name = getProgress)]
    pub fn get_progress(&self) -> f64 {
        if self.max_iterations == 0 {
            return 0.0;
        }
        (self.iteration_count as f64 / self.max_iterations as f64) * 100.0
    }

    #[wasm_bindgen(js_name = getOptimizationMode)]
    pub fn get_optimization_mode(&self) -> OptimizationMode {
        self.optimization_mode
    }

    #[wasm_bindgen(js_name = getBatchSize)]
    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    #[wasm_bindgen(js_name = getMaxIterations)]
    pub fn get_max_iterations(&self) -> usize {
        self.max_iterations
    }

    fn call_progress_callback(&self) -> Result<(), JsValue> {
        if let Some(callback) = &self.progress_callback {
            let progress = self.get_progress();
            
            let progress_info = Object::new();
            Reflect::set(&progress_info, &"progress".into(), &progress.into())?;
            Reflect::set(&progress_info, &"iteration".into(), &(self.iteration_count as f64).into())?;
            Reflect::set(&progress_info, &"total".into(), &(self.max_iterations as f64).into())?;
            
            callback.call1(&JsValue::NULL, &progress_info.into())?;
        }
        Ok(())
    }
}