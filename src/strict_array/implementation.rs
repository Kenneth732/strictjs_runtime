
// src/strict_array/implementation.rs
use crate::types::{HeapType, Num};
use js_sys::{Array, Function, Uint8Array};
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
                    // Note: u64 may lose precision when converted to f64
                    Ok(JsValue::from_f64(u64::from_le_bytes(array) as f64))
                } else {
                    Err(JsValue::from_str("Invalid element size for U64 type"))
                }
            }
            HeapType::I64 => {
                if bytes.len() >= 8 {
                    let array: [u8; 8] = bytes[0..8].try_into()
                        .map_err(|_| JsValue::from_str("Invalid byte length for i64"))?;
                    // Note: i64 may lose precision when converted to f64
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
                // For string types, return the bytes as a Uint8Array
                let array = Uint8Array::new_with_length(bytes.len() as u32);
                array.copy_from(bytes);
                Ok(array.into())
            }
            _ => {
                // For unsupported complex types, return the raw bytes
                let array = Uint8Array::new_with_length(bytes.len() as u32);
                array.copy_from(bytes);
                Ok(array.into())
            }
        }
    }

    #[wasm_bindgen(js_name = setValue)]
    pub fn set_value(&mut self, index: usize, value: JsValue) -> Result<(), JsValue> {
        let heap_type = self.heap; // Copy heap type before mutable borrow
        let bytes_mut = self.get_bytes_mut(index)?;

        match heap_type {
            HeapType::Str | HeapType::Str16 => {
                // For string types, expect Uint8Array
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
            _ => {
                // For unsupported types, set to default value
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
}

// Additional safety and utility methods
impl StrictArray {
    /// Creates a slice view of the underlying buffer for safe access
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }

    /// Creates a mutable slice view of the underlying buffer for safe access
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buffer
    }

    /// Validates that the array is properly sized for its type
    pub fn validate(&self) -> Result<(), JsValue> {
        if self.buffer.len() != self.len * self.element_size {
            return Err(JsValue::from_str("Array buffer size mismatch"));
        }
        Ok(())
    }

    /// Safe conversion to Vec<f64> for numeric types
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

    /// Zero-copy access to raw bytes for a specific element
    pub fn get_element_bytes(&self, index: usize) -> Result<&[u8], JsValue> {
        self.get_bytes(index)
    }

    /// Zero-copy mutable access to raw bytes for a specific element
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
}

#[wasm_bindgen(js_name = getIterator)]
pub fn get_iterator(array: &StrictArray) -> ArrayIterator {
    ArrayIterator {
        array: array.clone(),
        index: 0,
    }
}



