

// src/strict_array/implementation.rs
use crate::types::{HeapType, Num};
use js_sys::{Array, Function, Uint8Array, Object};
use wasm_bindgen::prelude::*;
use std::convert::TryInto;

#[wasm_bindgen]
#[derive(Clone)]
pub struct StrictArray {
    heap: HeapType,
    buffer: Vec<u8>,
    len: usize,
    element_size: usize,
}

#[wasm_bindgen]
impl StrictArray {
    #[wasm_bindgen(constructor)]
    pub fn new(heap: HeapType, len: usize) -> StrictArray {
        let element_size = heap.element_size();
        StrictArray {
            heap,
            buffer: vec![0; len * element_size],
            len,
            element_size,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[wasm_bindgen(getter)]
    pub fn heap(&self) -> HeapType {
        self.heap
    }

    #[wasm_bindgen(getter)]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    #[wasm_bindgen(getter)]
    pub fn byte_len(&self) -> usize {
        self.buffer.len()
    }

    #[wasm_bindgen(getter)]
    pub fn element_size(&self) -> usize {
        self.element_size
    }

    #[wasm_bindgen(js_name = isTensorType)]
    pub fn is_tensor_type(&self) -> bool {
        self.heap.is_tensor_type()
    }

    #[wasm_bindgen(js_name = isMatrixType)]
    pub fn is_matrix_type(&self) -> bool {
        self.heap.is_matrix_type()
    }

    #[wasm_bindgen(js_name = isVectorType)]
    pub fn is_vector_type(&self) -> bool {
        self.heap.is_vector_type()
    }

    #[wasm_bindgen(js_name = isQuantizedType)]
    pub fn is_quantized_type(&self) -> bool {
        self.heap.is_quantized_type()
    }

    #[wasm_bindgen(js_name = isSparseType)]
    pub fn is_sparse_type(&self) -> bool {
        self.heap.is_sparse_type()
    }

    #[wasm_bindgen(js_name = getRecommendedBackend)]
    pub fn get_recommended_backend(&self) -> String {
        self.heap.recommended_backend().to_string()
    }

    #[wasm_bindgen(js_name = getOptimalLayout)]
    pub fn get_optimal_layout(&self) -> String {
        self.heap.optimal_layout().to_string()
    }

    #[wasm_bindgen(js_name = estimateMemoryFootprint)]
    pub fn estimate_memory_footprint(&self) -> usize {
        self.heap.memory_footprint(self.len)
    }

    #[wasm_bindgen(js_name = getMLOperations)]
    pub fn get_ml_operations(&self) -> Array {
        self.heap.get_ml_operations()
            .into_iter()
            .map(JsValue::from)
            .collect()
    }

    #[wasm_bindgen(js_name = canPerformOperation)]
    pub fn can_perform_operation(&self, operation: &str) -> bool {
        self.heap.supports_operation(operation)
    }

    fn bounds_check(&self, index: usize) -> Result<(), JsValue> {
        if index >= self.len {
            return Err(JsValue::from_str(&format!(
                "Index {} out of bounds (length: {})", 
                index, self.len
            )));
        }
        Ok(())
    }

    fn get_bytes(&self, index: usize) -> Result<&[u8], JsValue> {
        self.bounds_check(index)?;

        let start = index * self.element_size;
        let end = start + self.element_size;

        if end > self.buffer.len() {
            return Err(JsValue::from_str("Buffer corruption detected"));
        }

        Ok(&self.buffer[start..end])
    }

    fn get_bytes_mut(&mut self, index: usize) -> Result<&mut [u8], JsValue> {
        self.bounds_check(index)?;

        let start = index * self.element_size;
        let end = start + self.element_size;

        if end > self.buffer.len() {
            return Err(JsValue::from_str("Buffer corruption detected"));
        }

        Ok(&mut self.buffer[start..end])
    }

    #[wasm_bindgen(js_name = getValue)]
    pub fn get_value(&self, index: usize) -> Result<JsValue, JsValue> {
        let bytes = self.get_bytes(index)?;

        match self.heap {
            HeapType::Number => {
                if bytes.len() == 8 {
                    let array: [u8; 8] = bytes.try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for f64"))?;
                    Ok(JsValue::from_f64(f64::from_le_bytes(array)))
                } else {
                    Err(JsValue::from_str("Invalid element size for Number type"))
                }
            }
            HeapType::U8 => {
                if bytes.len() >= 1 {
                    Ok(JsValue::from_f64(bytes[0] as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for U8 type"))
                }
            }
            HeapType::I8 => {
                if bytes.len() >= 1 {
                    Ok(JsValue::from_f64(bytes[0] as i8 as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for I8 type"))
                }
            }
            HeapType::U16 => {
                if bytes.len() >= 2 {
                    let array: [u8; 2] = bytes[0..2].try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for u16"))?;
                    Ok(JsValue::from_f64(u16::from_le_bytes(array) as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for U16 type"))
                }
            }
            HeapType::I16 => {
                if bytes.len() >= 2 {
                    let array: [u8; 2] = bytes[0..2].try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for i16"))?;
                    Ok(JsValue::from_f64(i16::from_le_bytes(array) as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for I16 type"))
                }
            }
            HeapType::U32 => {
                if bytes.len() >= 4 {
                    let array: [u8; 4] = bytes[0..4].try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for u32"))?;
                    Ok(JsValue::from_f64(u32::from_le_bytes(array) as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for U32 type"))
                }
            }
            HeapType::I32 => {
                if bytes.len() >= 4 {
                    let array: [u8; 4] = bytes[0..4].try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for i32"))?;
                    Ok(JsValue::from_f64(i32::from_le_bytes(array) as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for I32 type"))
                }
            }
            HeapType::Bool => {
                if bytes.len() >= 1 {
                    Ok(JsValue::from_bool(bytes[0] != 0))
                } else {
                    Err(JsValue::from_str("Invalid element size for Bool type"))
                }
            }
            HeapType::U64 => {
                if bytes.len() >= 8 {
                    let array: [u8; 8] = bytes[0..8].try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for u64"))?;
                    Ok(JsValue::from_f64(u64::from_le_bytes(array) as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for U64 type"))
                }
            }
            HeapType::I64 => {
                if bytes.len() >= 8 {
                    let array: [u8; 8] = bytes[0..8].try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for i64"))?;
                    Ok(JsValue::from_f64(i64::from_le_bytes(array) as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for I64 type"))
                }
            }
            HeapType::F32 => {
                if bytes.len() >= 4 {
                    let array: [u8; 4] = bytes[0..4].try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for f32"))?;
                    Ok(JsValue::from_f64(f32::from_le_bytes(array) as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for F32 type"))
                }
            }
            HeapType::F64 => {
                if bytes.len() >= 8 {
                    let array: [u8; 8] = bytes[0..8].try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for f64"))?;
                    Ok(JsValue::from_f64(f64::from_le_bytes(array)))
                } else {
                    Err(JsValue::from_str("Invalid element size for F64 type"))
                }
            }
            HeapType::Str | HeapType::Str16 => {
                let array = Uint8Array::new_with_length(bytes.len() as u32);
                array.copy_from(bytes);
                Ok(array.into())
            }
            
            // === AI/ML TYPES ===
            HeapType::TensorF32 | HeapType::MatrixF32 | HeapType::VectorF32 | 
            HeapType::WeightF32 | HeapType::BiasF32 | HeapType::GradientF32 | 
            HeapType::Activation | HeapType::Embedding | HeapType::Attention => {
                if bytes.len() >= 4 {
                    let array: [u8; 4] = bytes[0..4].try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for f32"))?;
                    Ok(JsValue::from_f64(f32::from_le_bytes(array) as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for F32 tensor type"))
                }
            }
            
            HeapType::TensorF64 | HeapType::MatrixF64 | HeapType::VectorF64 |
            HeapType::MatrixC64 => {
                if bytes.len() >= 8 {
                    let array: [u8; 8] = bytes[0..8].try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for f64"))?;
                    Ok(JsValue::from_f64(f64::from_le_bytes(array)))
                } else {
                    Err(JsValue::from_str("Invalid element size for F64 tensor type"))
                }
            }
            
            HeapType::TensorI32 | HeapType::VectorI32 | HeapType::MatrixC32 => {
                if bytes.len() >= 4 {
                    let array: [u8; 4] = bytes[0..4].try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for i32"))?;
                    Ok(JsValue::from_f64(i32::from_le_bytes(array) as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for I32 tensor type"))
                }
            }
            
            HeapType::TensorU8 | HeapType::Quantized8 | HeapType::TensorI8 => {
                if bytes.len() >= 1 {
                    Ok(JsValue::from_f64(bytes[0] as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for 8-bit tensor type"))
                }
            }
            
            HeapType::TensorU16 | HeapType::Quantized16 | HeapType::TensorI16 => {
                if bytes.len() >= 2 {
                    let array: [u8; 2] = bytes[0..2].try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for 16-bit tensor"))?;
                    Ok(JsValue::from_f64(u16::from_le_bytes(array) as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for 16-bit tensor type"))
                }
            }
            
            HeapType::SparseMatrix | HeapType::GPUTensor | HeapType::SIMDVector => {
                // For complex types, return the raw bytes
                let array = Uint8Array::new_with_length(bytes.len() as u32);
                array.copy_from(bytes);
                Ok(array.into())
            }
            
            _ => {
                let array = Uint8Array::new_with_length(bytes.len() as u32);
                array.copy_from(bytes);
                Ok(array.into())
            }
        }
    }

    #[wasm_bindgen(js_name = setValue)]
    pub fn set_value(&mut self, index: usize, value: JsValue) -> Result<(), JsValue> {
        let heap_type = self.heap; 
        let bytes_mut = self.get_bytes_mut(index)?;

        match heap_type {
            HeapType::Str | HeapType::Str16 => {
                if let Ok(uint8_array) = value.dyn_into::<Uint8Array>() {
                    let copy_len = bytes_mut.len().min(uint8_array.length() as usize);
                    uint8_array.copy_to(&mut bytes_mut[0..copy_len]);
                    Ok(())
                } else {
                    Err(JsValue::from_str("String types require Uint8Array values"))
                }
            }
            _ => {
                // For numeric types, convert to f64 first
                let num_value = value.as_f64()
                    .ok_or_else(|| JsValue::from_str("Numeric types require number values"))?;
                
                let num = Num::from_f64(&heap_type, num_value);
                Self::write_num_to_bytes_static(heap_type, num, bytes_mut)
            }
        }
    }

    fn write_num_to_bytes_static(heap_type: HeapType, num: Num, bytes_mut: &mut [u8]) -> Result<(), JsValue> {
        match num {
            Num::Number(v) => {
                if bytes_mut.len() >= 8 {
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..8].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for Number type"))
                }
            }
            Num::U8(v) => {
                if bytes_mut.len() >= 1 {
                    bytes_mut[0] = v;
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for U8 type"))
                }
            }
            Num::I8(v) => {
                if bytes_mut.len() >= 1 {
                    bytes_mut[0] = v as u8;
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for I8 type"))
                }
            }
            Num::U16(v) => {
                if bytes_mut.len() >= 2 {
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..2].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for U16 type"))
                }
            }
            Num::I16(v) => {
                if bytes_mut.len() >= 2 {
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..2].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for I16 type"))
                }
            }
            Num::U32(v) => {
                if bytes_mut.len() >= 4 {
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..4].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for U32 type"))
                }
            }
            Num::I32(v) => {
                if bytes_mut.len() >= 4 {
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..4].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for I32 type"))
                }
            }
            Num::Bool(v) => {
                if bytes_mut.len() >= 1 {
                    bytes_mut[0] = v as u8;
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for Bool type"))
                }
            }
            Num::U64(v) => {
                if bytes_mut.len() >= 8 {
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..8].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for U64 type"))
                }
            }
            Num::I64(v) => {
                if bytes_mut.len() >= 8 {
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..8].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for I64 type"))
                }
            }
            Num::F32(v) => {
                if bytes_mut.len() >= 4 {
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..4].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for F32 type"))
                }
            }
            Num::F64(v) => {
                if bytes_mut.len() >= 8 {
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..8].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for F64 type"))
                }
            }
            
            // === AI/ML TYPES ===
            Num::TensorF32(_) | Num::MatrixF32(_) | Num::VectorF32(_) | 
            Num::WeightF32(_) | Num::BiasF32(_) | Num::GradientF32(_) | 
            Num::Activation(_) | Num::Embedding(_) | Num::Attention(_) => {
                if bytes_mut.len() >= 4 {
                    let v = num.to_f64() as f32;
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..4].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for F32 tensor type"))
                }
            }
            
            Num::TensorF64(_) | Num::MatrixF64(_) | Num::VectorF64(_) |
            Num::MatrixC64(_) => {
                if bytes_mut.len() >= 8 {
                    let v = num.to_f64();
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..8].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for F64 tensor type"))
                }
            }
            
            Num::TensorI32(_) | Num::VectorI32(_) | Num::MatrixC32(_) => {
                if bytes_mut.len() >= 4 {
                    let v = num.to_f64() as i32;
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..4].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for I32 tensor type"))
                }
            }
            
            Num::TensorU8(_) | Num::Quantized8(_) => {
                if bytes_mut.len() >= 1 {
                    let v = num.to_f64() as u8;
                    bytes_mut[0] = v;
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for U8 tensor type"))
                }
            }
            
            Num::TensorI8(_) => {
                if bytes_mut.len() >= 1 {
                    let v = num.to_f64() as i8;
                    bytes_mut[0] = v as u8;
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for I8 tensor type"))
                }
            }
            
            Num::TensorU16(_) | Num::Quantized16(_) => {
                if bytes_mut.len() >= 2 {
                    let v = num.to_f64() as u16;
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..2].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for U16 tensor type"))
                }
            }
            
            Num::TensorI16(_) => {
                if bytes_mut.len() >= 2 {
                    let v = num.to_f64() as i16;
                    let bytes = v.to_le_bytes();
                    bytes_mut[0..2].copy_from_slice(&bytes);
                    Ok(())
                } else {
                    Err(JsValue::from_str("Insufficient space for I16 tensor type"))
                }
            }
            
            Num::SparseMatrix(_) | Num::GPUTensor(_) | Num::SIMDVector(_) => {
                // For complex types, write default value
                let default_value = heap_type.get_default_value().to_le_bytes();
                let copy_len = bytes_mut.len().min(default_value.len());
                bytes_mut[0..copy_len].copy_from_slice(&default_value[0..copy_len]);
                Ok(())
            }
            
            _ => {
                let default_value = heap_type.get_default_value().to_le_bytes();
                let copy_len = bytes_mut.len().min(default_value.len());
                bytes_mut[0..copy_len].copy_from_slice(&default_value[0..copy_len]);
                Ok(())
            }
        }
    }

    #[wasm_bindgen(js_name = copyToArrayBuffer)]
    pub fn copy_to_array_buffer(&self) -> Result<js_sys::ArrayBuffer, JsValue> {
        let array_buffer = js_sys::ArrayBuffer::new(self.buffer.len() as u32);
        let uint8_array = Uint8Array::new(&array_buffer);
        uint8_array.copy_from(&self.buffer);
        Ok(array_buffer)
    }

    #[wasm_bindgen(js_name = copyFromArrayBuffer)]
    pub fn copy_from_array_buffer(&mut self, array_buffer: &js_sys::ArrayBuffer) -> Result<(), JsValue> {
        let uint8_array = Uint8Array::new(array_buffer);
        if uint8_array.length() as usize != self.buffer.len() {
            return Err(JsValue::from_str("ArrayBuffer size mismatch"));
        }
        uint8_array.copy_to(&mut self.buffer);
        Ok(())
    }

    #[wasm_bindgen(js_name = toUint8Array)]
    pub fn to_uint8_array(&self) -> Uint8Array {
        let array = Uint8Array::new_with_length(self.buffer.len() as u32);
        array.copy_from(&self.buffer);
        array
    }

    #[wasm_bindgen(js_name = fromUint8Array)]
    pub fn from_uint8_array(heap: HeapType, array: &Uint8Array) -> Result<StrictArray, JsValue> {
        let element_size = heap.element_size();
        let len = array.length() as usize / element_size;
        
        if array.length() as usize % element_size != 0 {
            return Err(JsValue::from_str("Uint8Array length must be multiple of element size"));
        }

        let mut strict_array = StrictArray::new(heap, len);
        array.copy_to(&mut strict_array.buffer);
        
        Ok(strict_array)
    }

    #[wasm_bindgen(js_name = map)]
    pub fn map(&self, js_function: JsValue) -> Result<StrictArray, JsValue> {
        let function: Function = js_function
            .dyn_into()
            .map_err(|_| JsValue::from_str("Argument must be a function"))?;

        let mut new_array = StrictArray::new(self.heap, self.len);
        let args = Array::new_with_length(3);

        for i in 0..self.len {
            let current_value = self.get_value(i)?;

            args.set(0, current_value.clone());
            args.set(1, JsValue::from_f64(i as f64));
            args.set(2, self.heap.to_js_value());

            let mapped_value = function.apply(&JsValue::NULL, &args)?;
            new_array.set_value(i, mapped_value)?;
        }

        Ok(new_array)
    }

    #[wasm_bindgen(js_name = forEach)]
    pub fn for_each(&self, js_function: JsValue) -> Result<(), JsValue> {
        let function: Function = js_function
            .dyn_into()
            .map_err(|_| JsValue::from_str("Argument must be a function"))?;

        let args = Array::new_with_length(3);

        for i in 0..self.len {
            let current_value = self.get_value(i)?;

            args.set(0, current_value);
            args.set(1, JsValue::from_f64(i as f64));
            args.set(2, self.heap.to_js_value());

            function.apply(&JsValue::NULL, &args)?;
        }

        Ok(())
    }

    #[wasm_bindgen(js_name = reduce)]
    pub fn reduce(&self, js_function: JsValue, initial_value: JsValue) -> Result<JsValue, JsValue> {
        let function: Function = js_function
            .dyn_into()
            .map_err(|_| JsValue::from_str("Argument must be a function"))?;

        let mut accumulator = initial_value;
        let args = Array::new_with_length(4);

        for i in 0..self.len {
            let current_value = self.get_value(i)?;

            args.set(0, accumulator.clone());
            args.set(1, current_value);
            args.set(2, JsValue::from_f64(i as f64));
            args.set(3, self.heap.to_js_value());

            accumulator = function.apply(&JsValue::NULL, &args)?;
        }

        Ok(accumulator)
    }

    #[wasm_bindgen(js_name = setRange)]
    pub fn set_range(&mut self, start: usize, values: Array) -> Result<(), JsValue> {
        if start + values.length() as usize > self.len {
            return Err(JsValue::from_str("Range out of bounds"));
        }

        for i in 0..values.length() {
            let value = values.get(i);
            self.set_value(start + i as usize, value)?;
        }

        Ok(())
    }

    #[wasm_bindgen(js_name = getRange)]
    pub fn get_range(&self, start: usize, count: usize) -> Result<Array, JsValue> {
        if start + count > self.len {
            return Err(JsValue::from_str("Range out of bounds"));
        }

        let result = Array::new();
        for i in 0..count {
            result.push(&self.get_value(start + i)?);
        }

        Ok(result)
    }

    #[wasm_bindgen(js_name = fill)]
    pub fn fill(&mut self, value: JsValue) -> Result<(), JsValue> {
        for i in 0..self.len {
            self.set_value(i, value.clone())?;
        }
        Ok(())
    }

    #[wasm_bindgen(js_name = resize)]
    pub fn resize(&mut self, new_len: usize) -> Result<(), JsValue> {
        if new_len == self.len {
            return Ok(());
        }

        let new_capacity = new_len * self.element_size;
        let mut new_buffer = vec![0; new_capacity];
        
        // Copy existing data safely
        let copy_len = self.buffer.len().min(new_capacity);
        new_buffer[0..copy_len].copy_from_slice(&self.buffer[0..copy_len]);
        
        self.buffer = new_buffer;
        self.len = new_len;
        
        Ok(())
    }

    #[wasm_bindgen(js_name = clear)]
    pub fn clear(&mut self) -> Result<(), JsValue> {
        for byte in self.buffer.iter_mut() {
            *byte = 0;
        }
        Ok(())
    }

    #[wasm_bindgen(js_name = clone)]
    pub fn clone_array(&self) -> StrictArray {
        self.clone()
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        format!("StrictArray[heap: {:?}, len: {}, element_size: {}]", 
               self.heap, self.len, self.element_size)
    }

    // === AI/ML SPECIFIC METHODS ===

    #[wasm_bindgen(js_name = toFloat32Array)]
    pub fn to_float32_array(&self) -> Result<js_sys::Float32Array, JsValue> {
        if !self.heap.is_numeric() {
            return Err(JsValue::from_str("Array type is not numeric"));
        }

        if self.element_size != 4 {
            return Err(JsValue::from_str("Element size must be 4 for Float32Array"));
        }

        let float32_array = js_sys::Float32Array::new_with_length(self.len as u32);
        
        for i in 0..self.len {
            let value = self.get_value(i)?;
            if let Some(f64_value) = value.as_f64() {
                float32_array.set_index(i as u32, f64_value as f32);
            } else {
                return Err(JsValue::from_str("Non-numeric value found in array"));
            }
        }
        
        Ok(float32_array)
    }

    #[wasm_bindgen(js_name = toFloat64Array)]
    pub fn to_float64_array(&self) -> Result<js_sys::Float64Array, JsValue> {
        if !self.heap.is_numeric() {
            return Err(JsValue::from_str("Array type is not numeric"));
        }

        if self.element_size != 8 {
            return Err(JsValue::from_str("Element size must be 8 for Float64Array"));
        }

        let float64_array = js_sys::Float64Array::new_with_length(self.len as u32);
        
        for i in 0..self.len {
            let value = self.get_value(i)?;
            if let Some(f64_value) = value.as_f64() {
                float64_array.set_index(i as u32, f64_value);
            } else {
                return Err(JsValue::from_str("Non-numeric value found in array"));
            }
        }
        
        Ok(float64_array)
    }

    #[wasm_bindgen(js_name = fromFloat32Array)]
    pub fn from_float32_array(heap: HeapType, array: &js_sys::Float32Array) -> Result<StrictArray, JsValue> {
        if !heap.is_numeric() {
            return Err(JsValue::from_str("Heap type must be numeric"));
        }

        let len = array.length() as usize;
        let mut strict_array = StrictArray::new(heap, len);
        
        for i in 0..len {
            let value = array.get_index(i as u32);
            strict_array.set_value(i, JsValue::from_f64(value as f64))?;
        }
        
        Ok(strict_array)
    }

    #[wasm_bindgen(js_name = fromFloat64Array)]
    pub fn from_float64_array(heap: HeapType, array: &js_sys::Float64Array) -> Result<StrictArray, JsValue> {
        if !heap.is_numeric() {
            return Err(JsValue::from_str("Heap type must be numeric"));
        }

        let len = array.length() as usize;
        let mut strict_array = StrictArray::new(heap, len);
        
        for i in 0..len {
            let value = array.get_index(i as u32);
            strict_array.set_value(i, JsValue::from_f64(value))?;
        }
        
        Ok(strict_array)
    }

    #[wasm_bindgen(js_name = sum)]
    pub fn sum(&self) -> Result<f64, JsValue> {
        if !self.heap.is_numeric() {
            return Err(JsValue::from_str("Array type is not numeric"));
        }

        let mut total = 0.0;
        for i in 0..self.len {
            let value = self.get_value(i)?;
            if let Some(f64_value) = value.as_f64() {
                total += f64_value;
            } else {
                return Err(JsValue::from_str("Non-numeric value found in array"));
            }
        }
        
        Ok(total)
    }

    #[wasm_bindgen(js_name = average)]
    pub fn average(&self) -> Result<f64, JsValue> {
        if self.len == 0 {
            return Ok(0.0);
        }
        Ok(self.sum()? / self.len as f64)
    }

    #[wasm_bindgen(js_name = min)]
    pub fn min(&self) -> Result<f64, JsValue> {
        if !self.heap.is_numeric() {
            return Err(JsValue::from_str("Array type is not numeric"));
        }

        if self.len == 0 {
            return Ok(0.0);
        }

        let mut min_val = f64::MAX;
        for i in 0..self.len {
            let value = self.get_value(i)?;
            if let Some(f64_value) = value.as_f64() {
                if f64_value < min_val {
                    min_val = f64_value;
                }
            } else {
                return Err(JsValue::from_str("Non-numeric value found in array"));
            }
        }
        
        Ok(min_val)
    }

    #[wasm_bindgen(js_name = max)]
    pub fn max(&self) -> Result<f64, JsValue> {
        if !self.heap.is_numeric() {
            return Err(JsValue::from_str("Array type is not numeric"));
        }

        if self.len == 0 {
            return Ok(0.0);
        }

        let mut max_val = f64::MIN;
        for i in 0..self.len {
            let value = self.get_value(i)?;
            if let Some(f64_value) = value.as_f64() {
                if f64_value > max_val {
                    max_val = f64_value;
                }
            } else {
                return Err(JsValue::from_str("Non-numeric value found in array"));
            }
        }
        
        Ok(max_val)
    }

    #[wasm_bindgen(js_name = normalize)]
    pub fn normalize(&mut self) -> Result<(), JsValue> {
        if !self.heap.is_numeric() {
            return Err(JsValue::from_str("Array type is not numeric"));
        }

        let min_val = self.min()?;
        let max_val = self.max()?;
        let range = max_val - min_val;

        if range == 0.0 {
            // All values are the same, set to 0
            for i in 0..self.len {
                self.set_value(i, JsValue::from_f64(0.0))?;
            }
            return Ok(());
        }

        for i in 0..self.len {
            let value = self.get_value(i)?;
            if let Some(f64_value) = value.as_f64() {
                let normalized = (f64_value - min_val) / range;
                self.set_value(i, JsValue::from_f64(normalized))?;
            } else {
                return Err(JsValue::from_str("Non-numeric value found in array"));
            }
        }
        
        Ok(())
    }

    // === ADVANCED AI/ML OPERATIONS ===

    #[wasm_bindgen(js_name = dotProduct)]
    pub fn dot_product(&self, other: &StrictArray) -> Result<f64, JsValue> {
        if !self.heap.is_numeric() || !other.heap.is_numeric() {
            return Err(JsValue::from_str("Both arrays must be numeric for dot product"));
        }

        if self.len != other.len {
            return Err(JsValue::from_str("Arrays must have same length for dot product"));
        }

        let mut result = 0.0;
        for i in 0..self.len {
            let val1 = self.get_value(i)?;
            let val2 = other.get_value(i)?;
            
            if let (Some(f1), Some(f2)) = (val1.as_f64(), val2.as_f64()) {
                result += f1 * f2;
            } else {
                return Err(JsValue::from_str("Non-numeric values found in arrays"));
            }
        }
        
        Ok(result)
    }

    #[wasm_bindgen(js_name = convolution)]
    pub fn convolution(&self, kernel: &StrictArray) -> Result<StrictArray, JsValue> {
        if !self.heap.is_numeric() || !kernel.heap.is_numeric() {
            return Err(JsValue::from_str("Both arrays must be numeric for convolution"));
        }

        let kernel_len = kernel.len();
        let output_len = self.len.saturating_sub(kernel_len - 1);
        
        if output_len == 0 {
            return Err(JsValue::from_str("Kernel larger than input array"));
        }

        let mut result = StrictArray::new(self.heap, output_len);
        
        for i in 0..output_len {
            let mut sum = 0.0;
            for j in 0..kernel_len {
                let input_val = self.get_value(i + j)?;
                let kernel_val = kernel.get_value(j)?;
                
                if let (Some(f1), Some(f2)) = (input_val.as_f64(), kernel_val.as_f64()) {
                    sum += f1 * f2;
                } else {
                    return Err(JsValue::from_str("Non-numeric values found in arrays"));
                }
            }
            result.set_value(i, JsValue::from_f64(sum))?;
        }
        
        Ok(result)
    }

    #[wasm_bindgen(js_name = activation)]
    pub fn activation(&mut self, activation_type: &str) -> Result<(), JsValue> {
        if !self.heap.is_numeric() {
            return Err(JsValue::from_str("Array must be numeric for activation functions"));
        }

        for i in 0..self.len {
            let value = self.get_value(i)?;
            if let Some(f64_value) = value.as_f64() {
                let activated = match activation_type.to_lowercase().as_str() {
                    "relu" => f64_value.max(0.0),
                    "sigmoid" => 1.0 / (1.0 + (-f64_value).exp()),
                    "tanh" => f64_value.tanh(),
                    "leaky_relu" => if f64_value > 0.0 { f64_value } else { 0.01 * f64_value },
                    "softplus" => (1.0 + f64_value.exp()).ln(),
                    _ => return Err(JsValue::from_str(&format!("Unknown activation function: {}", activation_type))),
                };
                self.set_value(i, JsValue::from_f64(activated))?;
            } else {
                return Err(JsValue::from_str("Non-numeric value found in array"));
            }
        }
        
        Ok(())
    }

    #[wasm_bindgen(js_name = batchNormalization)]
    pub fn batch_normalization(&mut self, epsilon: f64) -> Result<(), JsValue> {
        if !self.heap.is_numeric() {
            return Err(JsValue::from_str("Array must be numeric for batch normalization"));
        }

        let mean = self.average()?;
        let variance = self.variance()?;
        let std_dev = (variance + epsilon).sqrt();

        for i in 0..self.len {
            let value = self.get_value(i)?;
            if let Some(f64_value) = value.as_f64() {
                let normalized = (f64_value - mean) / std_dev;
                self.set_value(i, JsValue::from_f64(normalized))?;
            } else {
                return Err(JsValue::from_str("Non-numeric value found in array"));
            }
        }
        
        Ok(())
    }

    #[wasm_bindgen(js_name = variance)]
    pub fn variance(&self) -> Result<f64, JsValue> {
        if !self.heap.is_numeric() {
            return Err(JsValue::from_str("Array type is not numeric"));
        }

        if self.len <= 1 {
            return Ok(0.0);
        }

        let mean = self.average()?;
        let mut sum_sq_diff = 0.0;

        for i in 0..self.len {
            let value = self.get_value(i)?;
            if let Some(f64_value) = value.as_f64() {
                let diff = f64_value - mean;
                sum_sq_diff += diff * diff;
            } else {
                return Err(JsValue::from_str("Non-numeric value found in array"));
            }
        }
        
        Ok(sum_sq_diff / (self.len - 1) as f64)
    }

    #[wasm_bindgen(js_name = standardDeviation)]
    pub fn standard_deviation(&self) -> Result<f64, JsValue> {
        Ok(self.variance()?.sqrt())
    }

    // === TENSOR/MATRIX OPERATIONS ===

    #[wasm_bindgen(js_name = reshape)]
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<StrictArray, JsValue> {
        let total_elements: usize = new_shape.iter().product();
        
        if total_elements != self.len {
            return Err(JsValue::from_str(&format!(
                "New shape must have same number of elements: {} != {}", 
                total_elements, self.len
            )));
        }

        // For now, return a clone since we're just changing metadata
        let reshaped = self.clone();
        
        Ok(reshaped)
    }

    #[wasm_bindgen(js_name = transpose)]
    pub fn transpose(&self, rows: usize, cols: usize) -> Result<StrictArray, JsValue> {
        if rows * cols != self.len {
            return Err(JsValue::from_str("Rows * cols must equal array length"));
        }

        let mut transposed = StrictArray::new(self.heap, self.len);
        
        for i in 0..rows {
            for j in 0..cols {
                let original_index = i * cols + j;
                let transposed_index = j * rows + i;
                
                let value = self.get_value(original_index)?;
                transposed.set_value(transposed_index, value)?;
            }
        }
        
        Ok(transposed)
    }

    #[wasm_bindgen(js_name = matrixMultiply)]
    pub fn matrix_multiply(&self, other: &StrictArray, a_rows: usize, a_cols: usize, b_cols: usize) -> Result<StrictArray, JsValue> {
        if !self.heap.is_numeric() || !other.heap.is_numeric() {
            return Err(JsValue::from_str("Both arrays must be numeric for matrix multiplication"));
        }

        if self.len != a_rows * a_cols {
            return Err(JsValue::from_str("First array size doesn't match specified dimensions"));
        }

        if other.len != a_cols * b_cols {
            return Err(JsValue::from_str("Second array size doesn't match specified dimensions"));
        }

        let result_len = a_rows * b_cols;
        let mut result = StrictArray::new(self.heap, result_len);

        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = 0.0;
                for k in 0..a_cols {
                    let a_index = i * a_cols + k;
                    let b_index = k * b_cols + j;
                    
                    let a_val = self.get_value(a_index)?;
                    let b_val = other.get_value(b_index)?;
                    
                    if let (Some(f1), Some(f2)) = (a_val.as_f64(), b_val.as_f64()) {
                        sum += f1 * f2;
                    } else {
                        return Err(JsValue::from_str("Non-numeric values found in arrays"));
                    }
                }
                let result_index = i * b_cols + j;
                result.set_value(result_index, JsValue::from_f64(sum))?;
            }
        }
        
        Ok(result)
    }

    // === QUANTIZATION OPERATIONS ===

    #[wasm_bindgen(js_name = quantize)]
    pub fn quantize(&self, bits: u8) -> Result<StrictArray, JsValue> {
        if !self.heap.is_numeric() {
            return Err(JsValue::from_str("Array must be numeric for quantization"));
        }

        if bits != 8 && bits != 16 {
            return Err(JsValue::from_str("Only 8-bit and 16-bit quantization supported"));
        }

        let min_val = self.min()?;
        let max_val = self.max()?;
        let range = max_val - min_val;

        let target_heap = if bits == 8 {
            if min_val >= 0.0 {
                HeapType::Quantized8
            } else {
                HeapType::TensorI8
            }
        } else {
            if min_val >= 0.0 {
                HeapType::Quantized16
            } else {
                HeapType::TensorI16
            }
        };

        let mut quantized = StrictArray::new(target_heap, self.len);
        let max_quant_value = if bits == 8 { u8::MAX as f64 } else { u16::MAX as f64 };

        for i in 0..self.len {
            let value = self.get_value(i)?;
            if let Some(f64_value) = value.as_f64() {
                let normalized = (f64_value - min_val) / range;
                let quant_value = (normalized * max_quant_value).round() as f64;
                quantized.set_value(i, JsValue::from_f64(quant_value))?;
            } else {
                return Err(JsValue::from_str("Non-numeric value found in array"));
            }
        }
        
        Ok(quantized)
    }

    #[wasm_bindgen(js_name = dequantize)]
    pub fn dequantize(&self, original_min: f64, original_max: f64) -> Result<StrictArray, JsValue> {
        if !self.heap.is_quantized_type() && !self.is_tensor_type() {
            return Err(JsValue::from_str("Array must be quantized or tensor type for dequantization"));
        }

        let range = original_max - original_min;
        let max_quant_value = match self.heap {
            HeapType::Quantized8 | HeapType::TensorU8 | HeapType::TensorI8 => u8::MAX as f64,
            HeapType::Quantized16 | HeapType::TensorU16 | HeapType::TensorI16 => u16::MAX as f64,
            _ => return Err(JsValue::from_str("Unsupported type for dequantization")),
        };

        let mut dequantized = StrictArray::new(HeapType::F32, self.len);

        for i in 0..self.len {
            let value = self.get_value(i)?;
            if let Some(f64_value) = value.as_f64() {
                let normalized = f64_value / max_quant_value;
                let dequant_value = normalized * range + original_min;
                dequantized.set_value(i, JsValue::from_f64(dequant_value))?;
            } else {
                return Err(JsValue::from_str("Non-numeric value found in array"));
            }
        }
        
        Ok(dequantized)
    }

    // === PERFORMANCE OPTIMIZATIONS ===

    #[wasm_bindgen(js_name = setValuesBatch)]
    pub fn set_values_batch(&mut self, values: Array) -> Result<(), JsValue> {
        if values.length() as usize != self.len {
            return Err(JsValue::from_str("Batch size must match array length"));
        }

        for i in 0..values.length() {
            self.set_value(i as usize, values.get(i))?;
        }
        Ok(())
    }

    #[wasm_bindgen(js_name = getValuesBatch)]
    pub fn get_values_batch(&self, indices: Vec<usize>) -> Result<Array, JsValue> {
        let result = Array::new();
        for &index in &indices {
            result.push(&self.get_value(index)?);
        }
        Ok(result)
    }

    #[wasm_bindgen(js_name = applyStrided)]
    pub fn apply_strided(&mut self, start: usize, stride: usize, js_function: JsValue) -> Result<(), JsValue> {
        let function: Function = js_function
            .dyn_into()
            .map_err(|_| JsValue::from_str("Argument must be a function"))?;

        let args = Array::new_with_length(3);

        for i in (start..self.len).step_by(stride) {
            let current_value = self.get_value(i)?;

            args.set(0, current_value.clone());
            args.set(1, JsValue::from_f64(i as f64));
            args.set(2, self.heap.to_js_value());

            let new_value = function.apply(&JsValue::NULL, &args)?;
            self.set_value(i, new_value)?;
        }

        Ok(())
    }

    // === MEMORY AND PERFORMANCE METHODS ===

    #[wasm_bindgen(js_name = getMemoryInfo)]
    pub fn get_memory_info(&self) -> JsValue {
        let obj = js_sys::Object::new();
        
        let _ = js_sys::Reflect::set(&obj, &"totalBytes".into(), &(self.buffer.len() as f64).into());
        let _ = js_sys::Reflect::set(&obj, &"elementCount".into(), &(self.len as f64).into());
        let _ = js_sys::Reflect::set(&obj, &"elementSize".into(), &(self.element_size as f64).into());
        let _ = js_sys::Reflect::set(&obj, &"estimatedFootprint".into(), &(self.estimate_memory_footprint() as f64).into());
        let _ = js_sys::Reflect::set(&obj, &"recommendedBackend".into(), &self.get_recommended_backend().into());
        let _ = js_sys::Reflect::set(&obj, &"optimalLayout".into(), &self.get_optimal_layout().into());
        
        obj.into()
    }

    #[wasm_bindgen(js_name = getCapabilities)]
    pub fn get_capabilities(&self) -> JsValue {
        self.heap.get_capabilities_js()
    }

    #[wasm_bindgen(js_name = isCompatibleWith)]
    pub fn is_compatible_with(&self, other: &StrictArray) -> bool {
        self.heap.is_binary_compatible_with(&other.heap)
    }

    #[wasm_bindgen(js_name = getBinaryResultType)]
    pub fn get_binary_result_type(&self, other: &StrictArray) -> Option<HeapType> {
        self.heap.get_binary_result_type(&other.heap)
    }

    // === SERIALIZATION ===

    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        let obj = js_sys::Object::new();
        
        let _ = js_sys::Reflect::set(&obj, &"type".into(), &self.heap.to_js_value());
        let _ = js_sys::Reflect::set(&obj, &"length".into(), &(self.len as f64).into());
        let _ = js_sys::Reflect::set(&obj, &"elementSize".into(), &(self.element_size as f64).into());
        
        // Convert data to base64 for compact serialization
        let data_array = self.to_uint8_array();
        let data_string = js_sys::JSON::stringify(&data_array.into())?;
        let _ = js_sys::Reflect::set(&obj, &"data".into(), &data_string);
        
        Ok(obj.into())
    }

// In src/strict_array/implementation.rs, fix the JSON parsing line:
#[wasm_bindgen(js_name = fromJSON)]
pub fn from_json(json: JsValue) -> Result<StrictArray, JsValue> {
    let obj: Object = json.dyn_into()?;
    
    let heap_type_val = js_sys::Reflect::get(&obj, &"type".into())?;
    let length_val = js_sys::Reflect::get(&obj, &"length".into())?;
    let data_val = js_sys::Reflect::get(&obj, &"data".into())?;
    
    let heap_type = HeapType::from_js_value(heap_type_val)?;
    let _length = length_val.as_f64().ok_or("Invalid length")? as usize;
    
    let data_string = data_val.as_string().ok_or("Invalid data string")?;
    let data_array: Uint8Array = js_sys::JSON::parse(&data_string)?.dyn_into()?;
    
    StrictArray::from_uint8_array(heap_type, &data_array)
}
}

// Internal helper methods
impl StrictArray {
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buffer
    }

    pub fn validate(&self) -> Result<(), JsValue> {
        if self.buffer.len() != self.len * self.element_size {
            return Err(JsValue::from_str("Array buffer size mismatch"));
        }
        Ok(())
    }

    pub fn to_vec_f64(&self) -> Result<Vec<f64>, JsValue> {
        if !self.heap.is_numeric() {
            return Err(JsValue::from_str("Array type is not numeric"));
        }

        let mut result = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let value = self.get_value(i)?;
            if let Some(f64_value) = value.as_f64() {
                result.push(f64_value);
            } else {
                return Err(JsValue::from_str("Non-numeric value found in array"));
            }
        }
        Ok(result)
    }

    pub fn get_element_bytes(&self, index: usize) -> Result<&[u8], JsValue> {
        self.get_bytes(index)
    }

    pub fn get_element_bytes_mut(&mut self, index: usize) -> Result<&mut [u8], JsValue> {
        self.get_bytes_mut(index)
    }
}

#[wasm_bindgen]
pub struct ArrayIterator {
    array: StrictArray,
    index: usize,
}

#[wasm_bindgen]
impl ArrayIterator {
    #[wasm_bindgen(js_name = next)]
    pub fn next(&mut self) -> Result<JsValue, JsValue> {
        if self.index < self.array.len() {
            let value = self.array.get_value(self.index)?;
            self.index += 1;
            Ok(value)
        } else {
            Ok(JsValue::NULL)
        }
    }

    #[wasm_bindgen(js_name = hasNext)]
    pub fn has_next(&self) -> bool {
        self.index < self.array.len()
    }

    #[wasm_bindgen(js_name = reset)]
    pub fn reset(&mut self) {
        self.index = 0;
    }
}

#[wasm_bindgen(js_name = getIterator)]
pub fn get_iterator(array: &StrictArray) -> ArrayIterator {
    ArrayIterator {
        array: array.clone(),
        index: 0,
    }
}

// Factory functions for common AI/ML array types
#[wasm_bindgen(js_name = createTensor)]
pub fn create_tensor(heap: HeapType, shape: Vec<usize>) -> Result<StrictArray, JsValue> {
    let total_elements: usize = shape.iter().product();
    Ok(StrictArray::new(heap, total_elements))
}

#[wasm_bindgen(js_name = createVector)]
pub fn create_vector(heap: HeapType, length: usize) -> Result<StrictArray, JsValue> {
    Ok(StrictArray::new(heap, length))
}

#[wasm_bindgen(js_name = createMatrix)]
pub fn create_matrix(heap: HeapType, rows: usize, cols: usize) -> Result<StrictArray, JsValue> {
    Ok(StrictArray::new(heap, rows * cols))
}

#[wasm_bindgen(js_name = createZeros)]
pub fn create_zeros(heap: HeapType, length: usize) -> Result<StrictArray, JsValue> {
    let mut array = StrictArray::new(heap, length);
    array.clear()?;
    Ok(array)
}

#[wasm_bindgen(js_name = createOnes)]
pub fn create_ones(heap: HeapType, length: usize) -> Result<StrictArray, JsValue> {
    let mut array = StrictArray::new(heap, length);
    array.fill(JsValue::from_f64(1.0))?;
    Ok(array)
}

#[wasm_bindgen(js_name = createRange)]
pub fn create_range(heap: HeapType, start: f64, end: f64, step: f64) -> Result<StrictArray, JsValue> {
    let length = ((end - start) / step).ceil() as usize;
    let mut array = StrictArray::new(heap, length);
    
    for i in 0..length {
        let value = start + (i as f64) * step;
        array.set_value(i, JsValue::from_f64(value))?;
    }
    
    Ok(array)
}


