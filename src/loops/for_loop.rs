

// src/loops/for_loop.rs
use crate::types::HeapType;
use crate::types::gpu_types::JsGPUType;
use crate::types::simd_types::JsSIMDType;
use crate::utils::clamp_f64;
use wasm_bindgen::prelude::*;
use js_sys::{Function, Array, Object, Reflect};
use crate::types::OptimizationMode; //Add this

#[wasm_bindgen]
#[derive(Clone)]
pub struct StrictForLoop {
    // Numeric loop fields
    current: f64,
    end: f64,
    step: f64,
    
    // Array loop fields
    array_data: Option<Array>,
    current_index: usize,
    
    // Common fields
    heap_type: HeapType,
    max_iterations: usize,
    iteration_count: usize,
    gpu_type: Option<JsGPUType>,
    simd_type: Option<JsSIMDType>,
    batch_size: usize,
    optimization_mode: OptimizationMode,
    progress_callback: Option<Function>,
    is_array_iteration: bool,
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
        let clamped_start = Self::clamp_value(start, heap_type);
        let clamped_end = Self::clamp_value(end, heap_type);
        let clamped_step = Self::clamp_value(step, heap_type);

        let optimization_mode = Self::detect_optimization_mode(heap_type);
        let (gpu_type, simd_type, batch_size) = Self::get_optimization_settings(heap_type, optimization_mode);

        StrictForLoop {
            current: clamped_start,
            end: clamped_end,
            step: clamped_step,
            array_data: None,
            current_index: 0,
            heap_type,
            max_iterations,
            iteration_count: 0,
            gpu_type,
            simd_type,
            batch_size,
            optimization_mode,
            progress_callback: None,
            is_array_iteration: false,
        }
    }

    #[wasm_bindgen(js_name = newWithOptimization)]
    pub fn new_with_optimization(
        start: f64,
        end: f64,
        step: f64,
        heap_type: HeapType,
        max_iterations: usize,
        optimization_mode: OptimizationMode,
    ) -> Self {
        let clamped_start = Self::clamp_value(start, heap_type);
        let clamped_end = Self::clamp_value(end, heap_type);
        let clamped_step = Self::clamp_value(step, heap_type);

        let (gpu_type, simd_type, batch_size) = Self::get_optimization_settings(heap_type, optimization_mode);

        StrictForLoop {
            current: clamped_start,
            end: clamped_end,
            step: clamped_step,
            array_data: None,
            current_index: 0,
            heap_type,
            max_iterations,
            iteration_count: 0,
            gpu_type,
            simd_type,
            batch_size,
            optimization_mode,
            progress_callback: None,
            is_array_iteration: false,
        }
    }

    #[wasm_bindgen(js_name = newForArray)]
    pub fn new_for_array(
        data: Array,
        heap_type: HeapType,
        max_iterations: usize,
    ) -> Self {
        let data_length = data.length() as usize;
        let optimization_mode = Self::detect_optimization_mode(heap_type);
        let (gpu_type, simd_type, batch_size) = Self::get_optimization_settings(heap_type, optimization_mode);

        StrictForLoop {
            current: 0.0,
            end: 0.0,
            step: 0.0,
            array_data: Some(data),
            current_index: 0,
            heap_type,
            max_iterations: max_iterations.min(data_length),
            iteration_count: 0,
            gpu_type,
            simd_type,
            batch_size,
            optimization_mode,
            progress_callback: None,
            is_array_iteration: true,
        }
    }

    #[wasm_bindgen(js_name = newForArrayWithOptimization)]
    pub fn new_for_array_with_optimization(
        data: Array,
        heap_type: HeapType,
        max_iterations: usize,
        optimization_mode: OptimizationMode,
    ) -> Self {
        let data_length = data.length() as usize;
        let (gpu_type, simd_type, batch_size) = Self::get_optimization_settings(heap_type, optimization_mode);

        StrictForLoop {
            current: 0.0,
            end: 0.0,
            step: 0.0,
            array_data: Some(data),
            current_index: 0,
            heap_type,
            max_iterations: max_iterations.min(data_length),
            iteration_count: 0,
            gpu_type,
            simd_type,
            batch_size,
            optimization_mode,
            progress_callback: None,
            is_array_iteration: true,
        }
    }

    #[wasm_bindgen(js_name = hasNext)]
    pub fn has_next(&self) -> bool {
        if self.is_array_iteration {
            if let Some(data) = &self.array_data {
                self.current_index < data.length() as usize
                    && self.iteration_count < self.max_iterations
            } else {
                false
            }
        } else {
            ((self.step > 0.0 && self.current <= self.end)
                || (self.step < 0.0 && self.current >= self.end))
                && self.iteration_count < self.max_iterations
        }
    }

    #[wasm_bindgen(js_name = next)]
    pub fn next(&mut self) -> Result<JsValue, JsValue> {
        if self.iteration_count >= self.max_iterations {
            return Err(JsValue::from_str("Maximum iterations exceeded"));
        }

        if !self.has_next() {
            return Err(JsValue::from_str("Loop completed"));
        }

        let result = if self.is_array_iteration {
            if let Some(data) = &self.array_data {
                let value = data.get(self.current_index as u32);
                self.current_index += 1;
                value
            } else {
                return Err(JsValue::from_str("Array data not available"));
            }
        } else {
            let current_value = self.current;
            self.current = Self::clamp_value(self.current + self.step, self.heap_type);
            JsValue::from_f64(current_value)
        };

        self.iteration_count += 1;
        self.call_progress_callback()?;
        Ok(result)
    }

    #[wasm_bindgen(js_name = nextValue)]
    pub fn next_value(&mut self) -> Result<f64, JsValue> {
        let value = self.next()?;
        value.as_f64()
            .ok_or_else(|| JsValue::from_str("Cannot convert value to number"))
    }

    #[wasm_bindgen(js_name = nextBatch)]
    pub fn next_batch(&mut self) -> Result<Array, JsValue> {
        if self.iteration_count >= self.max_iterations {
            return Err(JsValue::from_str("Maximum iterations exceeded"));
        }

        let batch = Array::new();
        let mut items_in_batch = 0;

        while items_in_batch < self.batch_size && self.has_next() {
            let value = self.next()?;
            batch.push(&value);
            items_in_batch += 1;
        }

        Ok(batch)
    }

    #[wasm_bindgen(js_name = nextNumericBatch)]
    pub fn next_numeric_batch(&mut self) -> Result<Array, JsValue> {
        if self.iteration_count >= self.max_iterations {
            return Err(JsValue::from_str("Maximum iterations exceeded"));
        }

        let batch = Array::new();
        let mut items_in_batch = 0;

        while items_in_batch < self.batch_size && self.has_next() {
            let value = self.next_value()?;
            batch.push(&JsValue::from_f64(value));
            items_in_batch += 1;
        }

        Ok(batch)
    }

    #[wasm_bindgen(js_name = nextGPUBatch)]
    pub fn next_gpu_batch(&mut self) -> Result<JsValue, JsValue> {
        if !self.is_array_iteration {
            let gpu_type = self.gpu_type.clone();
            if let Some(gpu_type) = gpu_type {
                let batch = self.next_numeric_batch()?;
                let buffer_info = Self::create_gpu_buffer(&batch, &gpu_type)?;
                Ok(buffer_info.into())
            } else {
                Err(JsValue::from_str("GPU optimization not available for this loop type"))
            }
        } else {
            Err(JsValue::from_str("GPU batches only available for numeric loops"))
        }
    }

    #[wasm_bindgen(js_name = nextSIMDBatch)]
    pub fn next_simd_batch(&mut self) -> Result<Array, JsValue> {
        if !self.is_array_iteration {
            let simd_type = self.simd_type.clone();
            if let Some(simd_type) = simd_type {
                let batch = self.next_numeric_batch()?;
                let optimized_batch = Self::apply_simd_optimization(&batch, &simd_type)?;
                Ok(optimized_batch)
            } else {
                Err(JsValue::from_str("SIMD optimization not available for this loop type"))
            }
        } else {
            Err(JsValue::from_str("SIMD batches only available for numeric loops"))
        }
    }

    #[wasm_bindgen(js_name = forEach)]
    pub fn for_each(&mut self, callback: Function) -> Result<usize, JsValue> {
        while self.has_next() {
            let value = self.next()?;
            callback.call1(&JsValue::NULL, &value)?;
        }
        Ok(self.iteration_count)
    }

    #[wasm_bindgen(js_name = forEachValue)]
    pub fn for_each_value(&mut self, callback: Function) -> Result<usize, JsValue> {
        while self.has_next() {
            let value = self.next_value()?;
            callback.call1(&JsValue::NULL, &JsValue::from_f64(value))?;
        }
        Ok(self.iteration_count)
    }

    #[wasm_bindgen(js_name = forEachBatch)]
    pub fn for_each_batch(&mut self, callback: Function) -> Result<usize, JsValue> {
        while self.has_next() {
            let batch = self.next_batch()?;
            callback.call1(&JsValue::NULL, &batch)?;
        }
        Ok(self.iteration_count)
    }

    #[wasm_bindgen(js_name = forEachNumericBatch)]
    pub fn for_each_numeric_batch(&mut self, callback: Function) -> Result<usize, JsValue> {
        while self.has_next() {
            let batch = self.next_numeric_batch()?;
            callback.call1(&JsValue::NULL, &batch)?;
        }
        Ok(self.iteration_count)
    }

    #[wasm_bindgen(js_name = forEachTensorBatch)]
    pub fn for_each_tensor_batch(&mut self, callback: Function) -> Result<usize, JsValue> {
        if !self.is_array_iteration {
            while self.has_next() {
                let batch = self.next_numeric_batch()?;
                callback.call1(&JsValue::NULL, &batch)?;
            }
            Ok(self.iteration_count)
        } else {
            Err(JsValue::from_str("Tensor batches only available for numeric loops"))
        }
    }

    #[wasm_bindgen(js_name = forEachMatrixRow)]
    pub fn for_each_matrix_row(&mut self, callback: Function, row_size: usize) -> Result<usize, JsValue> {
        if !self.is_array_iteration {
            let mut row = Array::new();
            let mut rows_processed = 0;
            
            while self.has_next() {
                let value = self.next_value()?;
                row.push(&JsValue::from_f64(value));
                
                if row.length() as usize == row_size {
                    callback.call1(&JsValue::NULL, &row)?;
                    row = Array::new();
                    rows_processed += 1;
                }
            }
            
            if row.length() > 0 {
                callback.call1(&JsValue::NULL, &row)?;
                rows_processed += 1;
            }
            
            Ok(rows_processed)
        } else {
            Err(JsValue::from_str("Matrix rows only available for numeric loops"))
        }
    }

    #[wasm_bindgen(js_name = forEachObject)]
    pub fn for_each_object(&mut self, callback: Function) -> Result<usize, JsValue> {
        if self.is_array_iteration {
            while self.has_next() {
                let value = self.next()?;
                if value.is_object() {
                    callback.call1(&JsValue::NULL, &value)?;
                }
            }
            Ok(self.iteration_count)
        } else {
            Err(JsValue::from_str("Object iteration only available for array loops"))
        }
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
        if self.is_array_iteration {
            self.current_index = 0;
        } else {
            let start_value = self.current - (self.step * self.iteration_count as f64);
            self.current = Self::clamp_value(start_value, self.heap_type);
        }
        self.iteration_count = 0;
    }

    #[wasm_bindgen(js_name = getCurrent)]
    pub fn get_current(&self) -> JsValue {
        if self.is_array_iteration {
            if let Some(data) = &self.array_data {
                if self.current_index < data.length() as usize {
                    data.get(self.current_index as u32)
                } else {
                    JsValue::NULL
                }
            } else {
                JsValue::NULL
            }
        } else {
            JsValue::from_f64(self.current)
        }
    }

    #[wasm_bindgen(js_name = getCurrentValue)]
    pub fn get_current_value(&self) -> Result<f64, JsValue> {
        if self.is_array_iteration {
            if let Some(data) = &self.array_data {
                if self.current_index < data.length() as usize {
                    let value = data.get(self.current_index as u32);
                    value.as_f64()
                        .ok_or_else(|| JsValue::from_str("Current value is not a number"))
                } else {
                    Err(JsValue::from_str("No current value available"))
                }
            } else {
                Err(JsValue::from_str("Array data not available"))
            }
        } else {
            Ok(self.current)
        }
    }

    #[wasm_bindgen(js_name = getProgress)]
    pub fn get_progress(&self) -> f64 {
        if self.max_iterations == 0 {
            return 0.0;
        }
        (self.iteration_count as f64 / self.max_iterations as f64) * 100.0
    }

    #[wasm_bindgen(js_name = getStep)]
    pub fn get_step(&self) -> Result<f64, JsValue> {
        if self.is_array_iteration {
            Err(JsValue::from_str("Step not available for array iteration"))
        } else {
            Ok(self.step)
        }
    }

    #[wasm_bindgen(js_name = getEnd)]
    pub fn get_end(&self) -> Result<f64, JsValue> {
        if self.is_array_iteration {
            if let Some(data) = &self.array_data {
                Ok(data.length() as f64)
            } else {
                Err(JsValue::from_str("Array data not available"))
            }
        } else {
            Ok(self.end)
        }
    }

    #[wasm_bindgen(js_name = getHeapType)]
    pub fn get_heap_type(&self) -> HeapType {
        self.heap_type
    }

    #[wasm_bindgen(js_name = getOptimizationMode)]
    pub fn get_optimization_mode(&self) -> OptimizationMode {
        self.optimization_mode
    }

    #[wasm_bindgen(js_name = getBatchSize)]
    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    #[wasm_bindgen(js_name = getGPUType)]
    pub fn get_gpu_type(&self) -> Option<JsGPUType> {
        self.gpu_type.clone()
    }

    #[wasm_bindgen(js_name = getSIMDType)]
    pub fn get_simd_type(&self) -> Option<JsSIMDType> {
        self.simd_type.clone()
    }

    #[wasm_bindgen(js_name = isArrayIteration)]
    pub fn is_array_iteration(&self) -> bool {
        self.is_array_iteration
    }

    #[wasm_bindgen(js_name = getArrayLength)]
    pub fn get_array_length(&self) -> Result<usize, JsValue> {
        if self.is_array_iteration {
            if let Some(data) = &self.array_data {
                Ok(data.length() as usize)
            } else {
                Err(JsValue::from_str("Array data not available"))
            }
        } else {
            Err(JsValue::from_str("Not an array iteration"))
        }
    }

    // Internal helper methods
    fn detect_optimization_mode(heap_type: HeapType) -> OptimizationMode {
        match heap_type {
            HeapType::TensorF32 | HeapType::TensorF64 | HeapType::MatrixF32 | HeapType::MatrixF64 => {
                OptimizationMode::GPU
            }
            HeapType::VectorF32 | HeapType::VectorF64 | HeapType::VectorI32 => {
                OptimizationMode::SIMD
            }
            HeapType::TensorU8 | HeapType::TensorI8 | HeapType::Quantized8 => {
                OptimizationMode::SIMD
            }
            HeapType::TensorU16 | HeapType::TensorI16 | HeapType::Quantized16 => {
                OptimizationMode::SIMD
            }
            _ => OptimizationMode::Sequential,
        }
    }

    fn get_optimization_settings(heap_type: HeapType, mode: OptimizationMode) -> (Option<JsGPUType>, Option<JsSIMDType>, usize) {
        match (heap_type, mode) {
            (HeapType::TensorF32, OptimizationMode::GPU) => (
                Some(JsGPUType::new("tensor_f32").unwrap()),
                Some(JsSIMDType::new("f32x8").unwrap()),
                1024,
            ),
            (HeapType::MatrixF32, OptimizationMode::GPU) => (
                Some(JsGPUType::new("matrix_f32").unwrap()),
                Some(JsSIMDType::new("f32x8").unwrap()),
                512,
            ),
            (HeapType::VectorF32, OptimizationMode::SIMD) => (
                None,
                Some(JsSIMDType::new("f32x4").unwrap()),
                64,
            ),
            (HeapType::TensorI8, OptimizationMode::SIMD) => (
                None,
                Some(JsSIMDType::new("i8x16").unwrap()),
                128,
            ),
            (HeapType::TensorU8, OptimizationMode::SIMD) => (
                None,
                Some(JsSIMDType::new("u8x16").unwrap()),
                128,
            ),
            _ => (None, None, 1),
        }
    }

    fn create_gpu_buffer(batch: &Array, gpu_type: &JsGPUType) -> Result<JsValue, JsValue> {
        let buffer_size = batch.length() as usize * gpu_type.element_size();
        
        let buffer_info = Object::new();
        Reflect::set(&buffer_info, &"type".into(), &gpu_type.to_string().into())?;
        Reflect::set(&buffer_info, &"size".into(), &(buffer_size as f64).into())?;
        Reflect::set(&buffer_info, &"batchSize".into(), &(batch.length() as f64).into())?;
        Reflect::set(&buffer_info, &"elementSize".into(), &(gpu_type.element_size() as f64).into())?;
        
        Ok(buffer_info.into())
    }

    fn apply_simd_optimization(batch: &Array, simd_type: &JsSIMDType) -> Result<Array, JsValue> {
        let optimized = Array::new();
        
        for i in 0..batch.length() {
            let value = batch.get(i);
            if let Some(num) = value.as_f64() {
                let processed = match simd_type.to_string().as_str() {
                    "f32x4" | "f32x8" => num as f32 as f64,
                    "i8x16" | "u8x16" => (num as i8) as f64,
                    "i16x8" | "u16x8" => (num as i16) as f64,
                    _ => num,
                };
                optimized.push(&JsValue::from_f64(processed));
            }
        }
        
        Ok(optimized)
    }

    fn call_progress_callback(&self) -> Result<(), JsValue> {
        if let Some(callback) = &self.progress_callback {
            let progress = self.get_progress();
            let current = self.get_current();
            let iteration = self.get_iteration_count();
            
            let progress_info = Object::new();
            Reflect::set(&progress_info, &"progress".into(), &progress.into())?;
            Reflect::set(&progress_info, &"current".into(), &current.into())?;
            Reflect::set(&progress_info, &"iteration".into(), &(iteration as f64).into())?;
            Reflect::set(&progress_info, &"total".into(), &(self.max_iterations as f64).into())?;
            Reflect::set(&progress_info, &"heapType".into(), &self.heap_type.to_string().into())?;
            Reflect::set(&progress_info, &"isArrayIteration".into(), &self.is_array_iteration.into())?;
            
            callback.call1(&JsValue::NULL, &progress_info.into())?;
        }
        Ok(())
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
            HeapType::F32 => value as f32 as f64,
            HeapType::F64 => value,
            HeapType::Bool => {
                if value != 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            HeapType::TensorF32 | HeapType::MatrixF32 | HeapType::VectorF32 => value as f32 as f64,
            HeapType::TensorF64 | HeapType::MatrixF64 | HeapType::VectorF64 => value,
            HeapType::TensorI32 | HeapType::VectorI32 => clamp_f64(value, i32::MIN as f64, i32::MAX as f64),
            HeapType::TensorU8 | HeapType::Quantized8 => clamp_f64(value, 0.0, 255.0),
            HeapType::TensorI8 => clamp_f64(value, -128.0, 127.0),
            HeapType::TensorU16 | HeapType::Quantized16 => clamp_f64(value, 0.0, 65535.0),
            HeapType::TensorI16 => clamp_f64(value, -32768.0, 32767.0),
            _ => value,
        }
    }
}


