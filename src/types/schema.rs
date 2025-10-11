
// src/types/schema.rs
use wasm_bindgen::prelude::*;
use js_sys::{Object, Array};
use std::collections::HashMap;
// Remove unused import: use super::HeapType;

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Schema {
    #[wasm_bindgen(skip)]
    pub fields: HashMap<String, FieldType>,
    #[wasm_bindgen(skip)]
    pub nested_schemas: HashMap<String, Schema>,
    #[wasm_bindgen(skip)]
    pub metadata: HashMap<String, String>,
}

#[derive(Clone, Debug)]
pub enum FieldType {
    Simple(String),
    Array(Box<FieldType>),
    Nested,
    Tensor(usize), // Tensor with dimensions
    Matrix(usize, usize), // Matrix with rows, cols
    Vector(usize), // Vector with length
    SparseMatrix,
    Quantized(String), // Quantized type with precision
    GPU(String), // GPU-specific type
    SIMD(String), // SIMD-specific type
}

#[wasm_bindgen]
impl Schema {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Schema {
        Schema {
            fields: HashMap::new(),
            nested_schemas: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    #[wasm_bindgen(js_name = addField)]
    pub fn add_field(&mut self, field: &str, type_str: &str) {
        self.fields.insert(field.to_string(), FieldType::Simple(type_str.to_string()));
        self.nested_schemas.remove(field);
    }

    #[wasm_bindgen(js_name = addArrayField)]
    pub fn add_array_field(&mut self, field: &str, element_type: &str) {
        self.fields.insert(field.to_string(), FieldType::Array(Box::new(FieldType::Simple(element_type.to_string()))));
        self.nested_schemas.remove(field);
    }

    #[wasm_bindgen(js_name = addNestedArrayField)]
    pub fn add_nested_array_field(&mut self, field: &str, schema: Schema) {
        self.fields.insert(field.to_string(), FieldType::Array(Box::new(FieldType::Nested)));
        self.nested_schemas.insert(field.to_string(), schema);
    }

    #[wasm_bindgen(js_name = addNestedField)]
    pub fn add_nested_field(&mut self, field: &str, schema: Schema) {
        self.fields.insert(field.to_string(), FieldType::Nested);
        self.nested_schemas.insert(field.to_string(), schema);
    }

    // === AI/ML ENHANCEMENTS ===

    #[wasm_bindgen(js_name = addTensorField)]
    pub fn add_tensor_field(&mut self, field: &str, dimensions: usize) {
        self.fields.insert(field.to_string(), FieldType::Tensor(dimensions));
    }

    #[wasm_bindgen(js_name = addMatrixField)]
    pub fn add_matrix_field(&mut self, field: &str, rows: usize, cols: usize) {
        self.fields.insert(field.to_string(), FieldType::Matrix(rows, cols));
    }

    #[wasm_bindgen(js_name = addVectorField)]
    pub fn add_vector_field(&mut self, field: &str, length: usize) {
        self.fields.insert(field.to_string(), FieldType::Vector(length));
    }

    #[wasm_bindgen(js_name = addSparseMatrixField)]
    pub fn add_sparse_matrix_field(&mut self, field: &str) {
        self.fields.insert(field.to_string(), FieldType::SparseMatrix);
    }

    #[wasm_bindgen(js_name = addQuantizedField)]
    pub fn add_quantized_field(&mut self, field: &str, precision: &str) {
        self.fields.insert(field.to_string(), FieldType::Quantized(precision.to_string()));
    }

    #[wasm_bindgen(js_name = addGPUField)]
    pub fn add_gpu_field(&mut self, field: &str, gpu_type: &str) {
        self.fields.insert(field.to_string(), FieldType::GPU(gpu_type.to_string()));
    }

    #[wasm_bindgen(js_name = addSIMDField)]
    pub fn add_simd_field(&mut self, field: &str, simd_type: &str) {
        self.fields.insert(field.to_string(), FieldType::SIMD(simd_type.to_string()));
    }

    #[wasm_bindgen(js_name = addMetadata)]
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    #[wasm_bindgen(js_name = getMetadata)]
    pub fn get_metadata(&self, key: &str) -> Option<String> {
        self.metadata.get(key).cloned()
    }

    #[wasm_bindgen(js_name = getFieldType)]
    pub fn get_field_type(&self, field: &str) -> Option<String> {
        self.fields.get(field).and_then(|ft| match ft {
            FieldType::Simple(t) => Some(t.clone()),
            FieldType::Array(inner) => match **inner {
                FieldType::Simple(ref t) => Some(format!("Array<{}>", t)),
                FieldType::Nested => Some("Array<Nested>".to_string()),
                FieldType::Tensor(dims) => Some(format!("Array<Tensor[{}]>", dims)),
                FieldType::Matrix(rows, cols) => Some(format!("Array<Matrix[{}x{}]>", rows, cols)),
                FieldType::Vector(len) => Some(format!("Array<Vector[{}]>", len)),
                FieldType::SparseMatrix => Some("Array<SparseMatrix>".to_string()),
                FieldType::Quantized(ref prec) => Some(format!("Array<Quantized[{}]>", prec)),
                FieldType::GPU(ref t) => Some(format!("Array<GPU[{}]>", t)),
                FieldType::SIMD(ref t) => Some(format!("Array<SIMD[{}]>", t)),
                FieldType::Array(_) => Some("Array<Array>".to_string()), // Handle nested arrays
            },
            FieldType::Nested => Some("Nested".to_string()),
            FieldType::Tensor(dims) => Some(format!("Tensor[{}]", dims)),
            FieldType::Matrix(rows, cols) => Some(format!("Matrix[{}x{}]", rows, cols)),
            FieldType::Vector(len) => Some(format!("Vector[{}]", len)),
            FieldType::SparseMatrix => Some("SparseMatrix".to_string()),
            FieldType::Quantized(prec) => Some(format!("Quantized[{}]", prec)),
            FieldType::GPU(t) => Some(format!("GPU[{}]", t)),
            FieldType::SIMD(t) => Some(format!("SIMD[{}]", t)),
        })
    }

    #[wasm_bindgen(js_name = getFieldTypeInfo)]
    pub fn get_field_type_info(&self, field: &str) -> JsValue {
        let obj = Object::new();
        
        if let Some(field_type) = self.fields.get(field) {
            match field_type {
                FieldType::Simple(t) => {
                    js_sys::Reflect::set(&obj, &"type".into(), &"simple".into()).unwrap();
                    js_sys::Reflect::set(&obj, &"value".into(), &t.into()).unwrap();
                }
                FieldType::Array(inner) => {
                    js_sys::Reflect::set(&obj, &"type".into(), &"array".into()).unwrap();
                    if let Some(element_info) = self.get_field_type_info_internal(inner) {
                        js_sys::Reflect::set(&obj, &"elementType".into(), &element_info).unwrap();
                    }
                }
                FieldType::Nested => {
                    js_sys::Reflect::set(&obj, &"type".into(), &"nested".into()).unwrap();
                }
                FieldType::Tensor(dims) => {
                    js_sys::Reflect::set(&obj, &"type".into(), &"tensor".into()).unwrap();
                    js_sys::Reflect::set(&obj, &"dimensions".into(), &(*dims as f64).into()).unwrap();
                }
                FieldType::Matrix(rows, cols) => {
                    js_sys::Reflect::set(&obj, &"type".into(), &"matrix".into()).unwrap();
                    js_sys::Reflect::set(&obj, &"rows".into(), &(*rows as f64).into()).unwrap();
                    js_sys::Reflect::set(&obj, &"columns".into(), &(*cols as f64).into()).unwrap();
                }
                FieldType::Vector(len) => {
                    js_sys::Reflect::set(&obj, &"type".into(), &"vector".into()).unwrap();
                    js_sys::Reflect::set(&obj, &"length".into(), &(*len as f64).into()).unwrap();
                }
                FieldType::SparseMatrix => {
                    js_sys::Reflect::set(&obj, &"type".into(), &"sparse_matrix".into()).unwrap();
                }
                FieldType::Quantized(prec) => {
                    js_sys::Reflect::set(&obj, &"type".into(), &"quantized".into()).unwrap();
                    js_sys::Reflect::set(&obj, &"precision".into(), &prec.into()).unwrap();
                }
                FieldType::GPU(t) => {
                    js_sys::Reflect::set(&obj, &"type".into(), &"gpu".into()).unwrap();
                    js_sys::Reflect::set(&obj, &"gpuType".into(), &t.into()).unwrap();
                }
                FieldType::SIMD(t) => {
                    js_sys::Reflect::set(&obj, &"type".into(), &"simd".into()).unwrap();
                    js_sys::Reflect::set(&obj, &"simdType".into(), &t.into()).unwrap();
                }
            }
        }
        
        obj.into()
    }

    fn get_field_type_info_internal(&self, field_type: &FieldType) -> Option<JsValue> {
        let obj = Object::new();
        
        match field_type {
            FieldType::Simple(t) => {
                js_sys::Reflect::set(&obj, &"type".into(), &"simple".into()).unwrap();
                js_sys::Reflect::set(&obj, &"value".into(), &t.into()).unwrap();
            }
            FieldType::Tensor(dims) => {
                js_sys::Reflect::set(&obj, &"type".into(), &"tensor".into()).unwrap();
                js_sys::Reflect::set(&obj, &"dimensions".into(), &(*dims as f64).into()).unwrap();
            }
            FieldType::Matrix(rows, cols) => {
                js_sys::Reflect::set(&obj, &"type".into(), &"matrix".into()).unwrap();
                js_sys::Reflect::set(&obj, &"rows".into(), &(*rows as f64).into()).unwrap();
                js_sys::Reflect::set(&obj, &"columns".into(), &(*cols as f64).into()).unwrap();
            }
            FieldType::Vector(len) => {
                js_sys::Reflect::set(&obj, &"type".into(), &"vector".into()).unwrap();
                js_sys::Reflect::set(&obj, &"length".into(), &(*len as f64).into()).unwrap();
            }
            FieldType::SparseMatrix => {
                js_sys::Reflect::set(&obj, &"type".into(), &"sparse_matrix".into()).unwrap();
            }
            FieldType::Quantized(prec) => {
                js_sys::Reflect::set(&obj, &"type".into(), &"quantized".into()).unwrap();
                js_sys::Reflect::set(&obj, &"precision".into(), &prec.into()).unwrap();
            }
            FieldType::GPU(t) => {
                js_sys::Reflect::set(&obj, &"type".into(), &"gpu".into()).unwrap();
                js_sys::Reflect::set(&obj, &"gpuType".into(), &t.into()).unwrap();
            }
            FieldType::SIMD(t) => {
                js_sys::Reflect::set(&obj, &"type".into(), &"simd".into()).unwrap();
                js_sys::Reflect::set(&obj, &"simdType".into(), &t.into()).unwrap();
            }
            FieldType::Array(inner) => {
                js_sys::Reflect::set(&obj, &"type".into(), &"array".into()).unwrap();
                if let Some(element_info) = self.get_field_type_info_internal(inner) {
                    js_sys::Reflect::set(&obj, &"elementType".into(), &element_info).unwrap();
                }
            }
            FieldType::Nested => {
                js_sys::Reflect::set(&obj, &"type".into(), &"nested".into()).unwrap();
            }
        }
        
        Some(obj.into())
    }

    #[wasm_bindgen(js_name = getNestedSchema)]
    pub fn get_nested_schema(&self, field: &str) -> Option<Schema> {
        self.nested_schemas.get(field).cloned()
    }

    #[wasm_bindgen(js_name = hasField)]
    pub fn has_field(&self, field: &str) -> bool {
        self.fields.contains_key(field) || self.nested_schemas.contains_key(field)
    }

    #[wasm_bindgen(js_name = isNestedField)]
    pub fn is_nested_field(&self, field: &str) -> bool {
        matches!(self.fields.get(field), Some(FieldType::Nested)) || 
        matches!(self.fields.get(field), Some(FieldType::Array(inner)) if matches!(**inner, FieldType::Nested))
    }

    #[wasm_bindgen(js_name = isArrayField)]
    pub fn is_array_field(&self, field: &str) -> bool {
        matches!(self.fields.get(field), Some(FieldType::Array(_)))
    }

    #[wasm_bindgen(js_name = isTensorField)]
    pub fn is_tensor_field(&self, field: &str) -> bool {
        matches!(self.fields.get(field), Some(FieldType::Tensor(_)))
    }

    #[wasm_bindgen(js_name = isMatrixField)]
    pub fn is_matrix_field(&self, field: &str) -> bool {
        matches!(self.fields.get(field), Some(FieldType::Matrix(_, _)))
    }

    #[wasm_bindgen(js_name = isVectorField)]
    pub fn is_vector_field(&self, field: &str) -> bool {
        matches!(self.fields.get(field), Some(FieldType::Vector(_)))
    }

    #[wasm_bindgen(js_name = isGPUField)]
    pub fn is_gpu_field(&self, field: &str) -> bool {
        matches!(self.fields.get(field), Some(FieldType::GPU(_)))
    }

    #[wasm_bindgen(js_name = isSIMDField)]
    pub fn is_simd_field(&self, field: &str) -> bool {
        matches!(self.fields.get(field), Some(FieldType::SIMD(_)))
    }

    #[wasm_bindgen(js_name = fieldNames)]
    pub fn field_names(&self) -> Vec<JsValue> {
        let mut names: Vec<JsValue> = self.fields.keys()
            .map(|k| JsValue::from_str(k))
            .collect();
        
        let nested_names: Vec<JsValue> = self.nested_schemas.keys()
            .map(|k| JsValue::from_str(k))
            .collect();
        
        names.extend(nested_names);
        names
    }

    #[wasm_bindgen(js_name = fieldCount)]
    pub fn field_count(&self) -> usize {
        self.fields.len() + self.nested_schemas.len()
    }

    #[wasm_bindgen(js_name = toJS)]
    pub fn to_js(&self) -> JsValue {
        let obj = Object::new();
        
        // Add metadata if present
        if !self.metadata.is_empty() {
            let metadata_obj = Object::new();
            for (key, value) in &self.metadata {
                js_sys::Reflect::set(&metadata_obj, &JsValue::from_str(key), &JsValue::from_str(value)).unwrap();
            }
            js_sys::Reflect::set(&obj, &"__metadata__".into(), &metadata_obj.into()).unwrap();
        }
        
        for (key, field_type) in &self.fields {
            let type_str = match field_type {
                FieldType::Simple(t) => t.clone(),
                FieldType::Array(inner) => {
                    let inner_str = match **inner {
                        FieldType::Simple(ref t) => t.clone(),
                        FieldType::Nested => "Nested".to_string(),
                        FieldType::Tensor(dims) => format!("Tensor[{}]", dims),
                        FieldType::Matrix(rows, cols) => format!("Matrix[{}x{}]", rows, cols),
                        FieldType::Vector(len) => format!("Vector[{}]", len),
                        FieldType::SparseMatrix => "SparseMatrix".to_string(),
                        FieldType::Quantized(ref prec) => format!("Quantized[{}]", prec),
                        FieldType::GPU(ref t) => format!("GPU[{}]", t),
                        FieldType::SIMD(ref t) => format!("SIMD[{}]", t),
                        FieldType::Array(_) => "Array".to_string(), // Handle nested arrays
                    };
                    format!("Array<{}>", inner_str)
                }
                FieldType::Nested => "Nested".to_string(),
                FieldType::Tensor(dims) => format!("Tensor[{}]", dims),
                FieldType::Matrix(rows, cols) => format!("Matrix[{}x{}]", rows, cols),
                FieldType::Vector(len) => format!("Vector[{}]", len),
                FieldType::SparseMatrix => "SparseMatrix".to_string(),
                FieldType::Quantized(prec) => format!("Quantized[{}]", prec),
                FieldType::GPU(t) => format!("GPU[{}]", t),
                FieldType::SIMD(t) => format!("SIMD[{}]", t),
            };
            js_sys::Reflect::set(&obj, &JsValue::from_str(key), &JsValue::from_str(&type_str)).unwrap();
        }
        
        for (key, schema) in &self.nested_schemas {
            js_sys::Reflect::set(&obj, &JsValue::from_str(key), &schema.to_js()).unwrap();
        }
        
        obj.into()
    }

    #[wasm_bindgen(js_name = fromJSObject)]
    pub fn from_js_object(js_obj: &Object) -> Result<Schema, JsValue> {
        let mut schema = Schema::new();
        let keys = Object::keys(js_obj);
        
        // Check for metadata
        if let Ok(metadata_value) = js_sys::Reflect::get(js_obj, &"__metadata__".into()) {
            if metadata_value.is_object() {
                let metadata_obj = metadata_value.dyn_into::<Object>()?;
                let metadata_keys = Object::keys(&metadata_obj);
                for i in 0..metadata_keys.length() {
                    let key = metadata_keys.get(i);
                    let key_str = key.as_string().ok_or_else(|| JsValue::from_str("Invalid metadata key"))?;
                    let value = js_sys::Reflect::get(&metadata_obj, &key)?;
                    if value.is_string() {
                        schema.add_metadata(&key_str, &value.as_string().unwrap());
                    }
                }
            }
        }
        
        for i in 0..keys.length() {
            let key = keys.get(i);
            let key_str = key.as_string().ok_or_else(|| JsValue::from_str("Invalid key"))?;
            
            // Skip metadata key
            if key_str == "__metadata__" {
                continue;
            }
            
            let value = js_sys::Reflect::get(js_obj, &key)?;
            
            if value.is_string() {
                let type_str = value.as_string().unwrap();
                if type_str.starts_with("Array<") && type_str.ends_with(">") {
                    let element_type = type_str.trim_start_matches("Array<").trim_end_matches(">");
                    
                    // Check for AI/ML types in arrays
                    if element_type.starts_with("Tensor[") {
                        let dims_str = element_type.trim_start_matches("Tensor[").trim_end_matches("]");
                        let dims = dims_str.parse::<usize>().unwrap_or(0);
                        schema.fields.insert(key_str, FieldType::Array(Box::new(FieldType::Tensor(dims))));
                    } else if element_type.starts_with("Matrix[") {
                        let dims_str = element_type.trim_start_matches("Matrix[").trim_end_matches("]");
                        let parts: Vec<&str> = dims_str.split('x').collect();
                        if parts.len() == 2 {
                            let rows = parts[0].parse::<usize>().unwrap_or(0);
                            let cols = parts[1].parse::<usize>().unwrap_or(0);
                            schema.fields.insert(key_str, FieldType::Array(Box::new(FieldType::Matrix(rows, cols))));
                        } else {
                            schema.add_array_field(&key_str, element_type);
                        }
                    } else if element_type.starts_with("Vector[") {
                        let len_str = element_type.trim_start_matches("Vector[").trim_end_matches("]");
                        let len = len_str.parse::<usize>().unwrap_or(0);
                        schema.fields.insert(key_str, FieldType::Array(Box::new(FieldType::Vector(len))));
                    } else if element_type.starts_with("Quantized[") {
                        let prec = element_type.trim_start_matches("Quantized[").trim_end_matches("]");
                        schema.fields.insert(key_str, FieldType::Array(Box::new(FieldType::Quantized(prec.to_string()))));
                    } else if element_type.starts_with("GPU[") {
                        let gpu_type = element_type.trim_start_matches("GPU[").trim_end_matches("]");
                        schema.fields.insert(key_str, FieldType::Array(Box::new(FieldType::GPU(gpu_type.to_string()))));
                    } else if element_type.starts_with("SIMD[") {
                        let simd_type = element_type.trim_start_matches("SIMD[").trim_end_matches("]");
                        schema.fields.insert(key_str, FieldType::Array(Box::new(FieldType::SIMD(simd_type.to_string()))));
                    } else {
                        schema.add_array_field(&key_str, element_type);
                    }
                } else if type_str.starts_with("Tensor[") {
                    let dims_str = type_str.trim_start_matches("Tensor[").trim_end_matches("]");
                    let dims = dims_str.parse::<usize>().unwrap_or(0);
                    schema.add_tensor_field(&key_str, dims);
                } else if type_str.starts_with("Matrix[") {
                    let dims_str = type_str.trim_start_matches("Matrix[").trim_end_matches("]");
                    let parts: Vec<&str> = dims_str.split('x').collect();
                    if parts.len() == 2 {
                        let rows = parts[0].parse::<usize>().unwrap_or(0);
                        let cols = parts[1].parse::<usize>().unwrap_or(0);
                        schema.add_matrix_field(&key_str, rows, cols);
                    } else {
                        schema.add_field(&key_str, &type_str);
                    }
                } else if type_str.starts_with("Vector[") {
                    let len_str = type_str.trim_start_matches("Vector[").trim_end_matches("]");
                    let len = len_str.parse::<usize>().unwrap_or(0);
                    schema.add_vector_field(&key_str, len);
                } else if type_str.starts_with("Quantized[") {
                    let prec = type_str.trim_start_matches("Quantized[").trim_end_matches("]");
                    schema.add_quantized_field(&key_str, prec);
                } else if type_str.starts_with("GPU[") {
                    let gpu_type = type_str.trim_start_matches("GPU[").trim_end_matches("]");
                    schema.add_gpu_field(&key_str, gpu_type);
                } else if type_str.starts_with("SIMD[") {
                    let simd_type = type_str.trim_start_matches("SIMD[").trim_end_matches("]");
                    schema.add_simd_field(&key_str, simd_type);
                } else {
                    schema.add_field(&key_str, &type_str);
                }
            } else if value.is_object() {
                let nested_obj = value.dyn_into::<Object>()?;
                let nested_schema = Schema::from_js_object(&nested_obj)?;
                schema.add_nested_field(&key_str, nested_schema);
            } else {
                return Err(JsValue::from_str("Field value must be string (type) or object (nested schema)"));
            }
        }
        
        Ok(schema)
    }

    #[wasm_bindgen(js_name = estimateMemoryUsage)]
    pub fn estimate_memory_usage(&self, sample_size: usize) -> f64 {
        let mut total_bytes = 0.0;
        
        for field_type in self.fields.values() {
            total_bytes += Self::estimate_field_memory(field_type, sample_size);
        }
        
        for nested_schema in self.nested_schemas.values() {
            total_bytes += nested_schema.estimate_memory_usage(sample_size);
        }
        
        total_bytes
    }

    fn estimate_field_memory(field_type: &FieldType, sample_size: usize) -> f64 {
        match field_type {
            FieldType::Simple(t) => {
                // Estimate based on type
                match t.as_str() {
                    "f32" | "i32" | "u32" => 4.0 * sample_size as f64,
                    "f64" | "i64" | "u64" => 8.0 * sample_size as f64,
                    "u8" | "i8" | "bool" => 1.0 * sample_size as f64,
                    "u16" | "i16" => 2.0 * sample_size as f64,
                    _ => 8.0 * sample_size as f64, // Default for strings/objects
                }
            }
            FieldType::Array(inner) => {
                // Array overhead + elements
                16.0 + Self::estimate_field_memory(inner, sample_size)
            }
            FieldType::Tensor(dims) => {
                // Tensor with dimensions
                (4.0 * (*dims as f64)).powi(*dims as i32) * sample_size as f64
            }
            FieldType::Matrix(rows, cols) => {
                // Matrix size
                (4.0 * (*rows as f64) * (*cols as f64)) * sample_size as f64
            }
            FieldType::Vector(len) => {
                // Vector size
                (4.0 * (*len as f64)) * sample_size as f64
            }
            FieldType::SparseMatrix => {
                // Sparse matrix (assume 10% density)
                (4.0 * sample_size as f64) * 0.1
            }
            FieldType::Quantized(_) => {
                // Quantized data (1-2 bytes per element)
                1.5 * sample_size as f64
            }
            FieldType::GPU(_) => {
                // GPU data has additional overhead
                8.0 * sample_size as f64 + 1024.0 // Base overhead for GPU buffers
            }
            FieldType::SIMD(_) => {
                // SIMD aligned data
                16.0 * sample_size as f64
            }
            FieldType::Nested => {
                // Nested objects have significant overhead
                64.0 * sample_size as f64
            }
        }
    }

    #[wasm_bindgen(js_name = getOptimizationHints)]
    pub fn get_optimization_hints(&self) -> JsValue {
        let hints = Array::new();
        
        for (field_name, field_type) in &self.fields {
            match field_type {
                FieldType::Tensor(_) | FieldType::Matrix(_, _) | FieldType::Vector(_) => {
                    hints.push(&JsValue::from_str(&format!("Field '{}' would benefit from GPU acceleration", field_name)));
                }
                FieldType::SparseMatrix => {
                    hints.push(&JsValue::from_str(&format!("Field '{}' should use sparse storage format", field_name)));
                }
                FieldType::Quantized(_) => {
                    hints.push(&JsValue::from_str(&format!("Field '{}' is memory-efficient (quantized)", field_name)));
                }
                FieldType::GPU(_) => {
                    hints.push(&JsValue::from_str(&format!("Field '{}' is optimized for GPU", field_name)));
                }
                FieldType::SIMD(_) => {
                    hints.push(&JsValue::from_str(&format!("Field '{}' is optimized for SIMD", field_name)));
                }
                FieldType::Array(inner) if matches!(**inner, FieldType::Tensor(_) | FieldType::Matrix(_, _) | FieldType::Vector(_)) => {
                    hints.push(&JsValue::from_str(&format!("Array field '{}' contains tensor data - consider batching", field_name)));
                }
                _ => {}
            }
        }
        
        hints.into()
    }
}

impl FieldType {
    pub fn is_array(&self) -> bool {
        matches!(self, FieldType::Array(_))
    }
    
    pub fn get_element_type(&self) -> Option<&FieldType> {
        match self {
            FieldType::Array(inner) => Some(inner),
            _ => None,
        }
    }

    pub fn is_ai_ml_type(&self) -> bool {
        matches!(
            self,
            FieldType::Tensor(_) |
            FieldType::Matrix(_, _) |
            FieldType::Vector(_) |
            FieldType::SparseMatrix |
            FieldType::Quantized(_) |
            FieldType::GPU(_) |
            FieldType::SIMD(_)
        )
    }
}

