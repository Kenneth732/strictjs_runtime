
//src/strict_object/implementation.rs
use wasm_bindgen::prelude::*;
use js_sys::{Object, Reflect, Array};
use crate::types::schema::{Schema, FieldType};
use std::collections::HashMap;

#[wasm_bindgen]
pub struct StrictObject {
    schema: Schema,
    data: Object,
    field_cache: HashMap<String, FieldInfo>,
    depth: u32,
}

#[derive(Clone)]
enum FieldInfo {
    Simple(String),
    Nested(Schema),
    Array(Box<FieldInfo>),
    Tensor,           // Simplified - dimensions handled by schema
    Matrix,           // Simplified - dimensions handled by schema  
    Vector,           // Simplified - length handled by schema
    SparseMatrix,
    Quantized,        // Simplified - precision handled by schema
    GPU,              // Simplified - type handled by schema
    SIMD,             // Simplified - type handled by schema
}

#[wasm_bindgen]
impl StrictObject {
    #[wasm_bindgen(constructor)]
    pub fn new(schema: JsValue) -> Result<StrictObject, JsValue> {
        let parsed_schema = if let Ok(js_obj) = schema.dyn_into::<Object>() {
            Schema::from_js_object(&js_obj)?
        } else {
            return Err(JsValue::from_str("Schema must be a JavaScript object"));
        };
        
        let mut obj = StrictObject {
            schema: parsed_schema.clone(),
            data: Object::new(),
            field_cache: HashMap::with_capacity(parsed_schema.field_count()),
            depth: 0,
        };
        
        obj.initialize_fields()?;
        Ok(obj)
    }

    fn initialize_fields(&mut self) -> Result<(), JsValue> {
        for field in self.schema.field_names() {
            let field_str = field.as_string().unwrap();
            
            if let Some(field_type) = self.schema.fields.get(&field_str) {
                match field_type {
                    FieldType::Simple(type_str) => {
                        self.field_cache.insert(field_str.clone(), FieldInfo::Simple(type_str.clone()));
                        self.set_default_value(&field_str, type_str)?;
                    }
                    FieldType::Array(element_type) => {
                        let element_info = match **element_type {
                            FieldType::Simple(ref t) => FieldInfo::Simple(t.clone()),
                            FieldType::Nested => {
                                if let Some(nested_schema) = self.schema.get_nested_schema(&field_str) {
                                    FieldInfo::Nested(nested_schema)
                                } else {
                                    return Err(JsValue::from_str(&format!("Missing schema for nested array field: {}", field_str)));
                                }
                            }
                            FieldType::Tensor(_) => FieldInfo::Tensor,
                            FieldType::Matrix(_, _) => FieldInfo::Matrix,
                            FieldType::Vector(_) => FieldInfo::Vector,
                            FieldType::SparseMatrix => FieldInfo::SparseMatrix,
                            FieldType::Quantized(_) => FieldInfo::Quantized,
                            FieldType::GPU(_) => FieldInfo::GPU,
                            FieldType::SIMD(_) => FieldInfo::SIMD,
                            FieldType::Array(_) => return Err(JsValue::from_str("Nested arrays not supported")),
                        };
                        self.field_cache.insert(field_str.clone(), FieldInfo::Array(Box::new(element_info)));
                        Reflect::set(&self.data, &field, &Array::new().into())?;
                    }
                    FieldType::Nested => {
                        if let Some(nested_schema) = self.schema.get_nested_schema(&field_str) {
                            self.field_cache.insert(field_str.clone(), FieldInfo::Nested(nested_schema.clone()));
                            let nested_obj = self.create_nested_object(&nested_schema)?;
                            Reflect::set(&self.data, &field, &nested_obj.to_js())?;
                        }
                    }
                    FieldType::Tensor(_) => {
                        self.field_cache.insert(field_str.clone(), FieldInfo::Tensor);
                        Reflect::set(&self.data, &field, &Array::new().into())?;
                    }
                    FieldType::Matrix(_, _) => {
                        self.field_cache.insert(field_str.clone(), FieldInfo::Matrix);
                        Reflect::set(&self.data, &field, &Array::new().into())?;
                    }
                    FieldType::Vector(_) => {
                        self.field_cache.insert(field_str.clone(), FieldInfo::Vector);
                        Reflect::set(&self.data, &field, &Array::new().into())?;
                    }
                    FieldType::SparseMatrix => {
                        self.field_cache.insert(field_str.clone(), FieldInfo::SparseMatrix);
                        Reflect::set(&self.data, &field, &Object::new().into())?;
                    }
                    FieldType::Quantized(_) => {
                        self.field_cache.insert(field_str.clone(), FieldInfo::Quantized);
                        Reflect::set(&self.data, &field, &Array::new().into())?;
                    }
                    FieldType::GPU(_) => {
                        self.field_cache.insert(field_str.clone(), FieldInfo::GPU);
                        Reflect::set(&self.data, &field, &Array::new().into())?;
                    }
                    FieldType::SIMD(_) => {
                        self.field_cache.insert(field_str.clone(), FieldInfo::SIMD);
                        Reflect::set(&self.data, &field, &Array::new().into())?;
                    }
                }
            }
        }
        Ok(())
    }

    fn create_nested_object(&self, schema: &Schema) -> Result<StrictObject, JsValue> {
        if self.depth > 50 {
            return Err(JsValue::from_str("Maximum nesting depth exceeded"));
        }
        
        let mut nested_obj = StrictObject {
            schema: schema.clone(),
            data: Object::new(),
            field_cache: HashMap::with_capacity(schema.field_count()),
            depth: self.depth + 1,
        };
        
        nested_obj.initialize_fields()?;
        Ok(nested_obj)
    }

    fn set_default_value(&self, field: &str, type_str: &str) -> Result<(), JsValue> {
        let default_value = match type_str {
            "string" => JsValue::from_str(""),
            "bool" | "boolean" => JsValue::from_bool(false),
            "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "f32" | "f64" => JsValue::from_f64(0.0),
            _ => JsValue::NULL,
        };
        
        Reflect::set(&self.data, &JsValue::from_str(field), &default_value)
            .map(|_| ())
            .map_err(|_| JsValue::from_str("Failed to set default value"))
    }

    #[wasm_bindgen(js_name = setField)]
    pub fn set_field(&mut self, field: &str, value: JsValue) -> Result<(), JsValue> {
        let field_info = self.field_cache.get(field).cloned();
        
        if let Some(field_info) = field_info {
            match field_info {
                FieldInfo::Nested(nested_schema) => {
                    self.set_nested_field(field, &nested_schema, value)
                }
                FieldInfo::Array(element_info) => {
                    self.set_array_field(field, &element_info, value)
                }
                FieldInfo::Simple(type_str) => {
                    self.set_simple_field(field, &type_str, value)
                }
                FieldInfo::Tensor => {
                    self.set_tensor_field(field, value)
                }
                FieldInfo::Matrix => {
                    self.set_matrix_field(field, value)
                }
                FieldInfo::Vector => {
                    self.set_vector_field(field, value)
                }
                FieldInfo::SparseMatrix => {
                    self.set_sparse_matrix_field(field, value)
                }
                FieldInfo::Quantized => {
                    self.set_quantized_field(field, value)
                }
                FieldInfo::GPU => {
                    self.set_gpu_field(field, value)
                }
                FieldInfo::SIMD => {
                    self.set_simd_field(field, value)
                }
            }
        } else {
            Err(JsValue::from_str(&format!("Field '{}' not in schema", field)))
        }
    }

    // AI/ML field setters
    fn set_tensor_field(&self, field: &str, value: JsValue) -> Result<(), JsValue> {
        if value.is_object() || value.is_array() {
            Reflect::set(&self.data, &JsValue::from_str(field), &value)
                .map(|_| ())
                .map_err(|_| JsValue::from_str("Failed to set tensor field"))
        } else {
            Err(JsValue::from_str("Tensor fields require object or array values"))
        }
    }

    fn set_matrix_field(&self, field: &str, value: JsValue) -> Result<(), JsValue> {
        if value.is_object() || value.is_array() {
            Reflect::set(&self.data, &JsValue::from_str(field), &value)
                .map(|_| ())
                .map_err(|_| JsValue::from_str("Failed to set matrix field"))
        } else {
            Err(JsValue::from_str("Matrix fields require object or array values"))
        }
    }

    fn set_vector_field(&self, field: &str, value: JsValue) -> Result<(), JsValue> {
        if value.is_array() {
            Reflect::set(&self.data, &JsValue::from_str(field), &value)
                .map(|_| ())
                .map_err(|_| JsValue::from_str("Failed to set vector field"))
        } else {
            Err(JsValue::from_str("Vector fields require array values"))
        }
    }

    fn set_sparse_matrix_field(&self, field: &str, value: JsValue) -> Result<(), JsValue> {
        if value.is_object() {
            Reflect::set(&self.data, &JsValue::from_str(field), &value)
                .map(|_| ())
                .map_err(|_| JsValue::from_str("Failed to set sparse matrix field"))
        } else {
            Err(JsValue::from_str("Sparse matrix fields require object values"))
        }
    }

    fn set_quantized_field(&self, field: &str, value: JsValue) -> Result<(), JsValue> {
        if value.is_array() {
            Reflect::set(&self.data, &JsValue::from_str(field), &value)
                .map(|_| ())
                .map_err(|_| JsValue::from_str("Failed to set quantized field"))
        } else {
            Err(JsValue::from_str("Quantized fields require array values"))
        }
    }

    fn set_gpu_field(&self, field: &str, value: JsValue) -> Result<(), JsValue> {
        if value.is_object() || value.is_array() {
            Reflect::set(&self.data, &JsValue::from_str(field), &value)
                .map(|_| ())
                .map_err(|_| JsValue::from_str("Failed to set GPU field"))
        } else {
            Err(JsValue::from_str("GPU fields require object or array values"))
        }
    }

    fn set_simd_field(&self, field: &str, value: JsValue) -> Result<(), JsValue> {
        if value.is_array() {
            Reflect::set(&self.data, &JsValue::from_str(field), &value)
                .map(|_| ())
                .map_err(|_| JsValue::from_str("Failed to set SIMD field"))
        } else {
            Err(JsValue::from_str("SIMD fields require array values"))
        }
    }

    fn set_nested_field(&mut self, field: &str, schema: &Schema, value: JsValue) -> Result<(), JsValue> {
        if value.is_object() {
            let nested_obj_value = value.dyn_into::<Object>()?;
            let nested_obj = StrictObject::new_with_data(schema.to_js(), nested_obj_value.into())?;
            Reflect::set(&self.data, &JsValue::from_str(field), &nested_obj.to_js())
                .map(|_| ())
                .map_err(|_| JsValue::from_str("Failed to set nested field"))
        } else {
            Err(JsValue::from_str("Nested fields require object values"))
        }
    }

    fn set_array_field(&mut self, field: &str, element_info: &FieldInfo, value: JsValue) -> Result<(), JsValue> {
        if let Ok(js_array) = value.dyn_into::<Array>() {
            let validated_array = Array::new();
            
            for i in 0..js_array.length() {
                let element_value = js_array.get(i);
                let validated_element = self.validate_array_element(element_info, element_value)?;
                validated_array.push(&validated_element);
            }
            
            Reflect::set(&self.data, &JsValue::from_str(field), &validated_array)
                .map(|_| ())
                .map_err(|_| JsValue::from_str("Failed to set array field"))
        } else {
            Err(JsValue::from_str("Array fields require array values"))
        }
    }

    fn validate_array_element(&self, element_info: &FieldInfo, value: JsValue) -> Result<JsValue, JsValue> {
        match element_info {
            FieldInfo::Simple(type_str) => {
                self.validate_simple_value(type_str, value)
            }
            FieldInfo::Nested(schema) => {
                if value.is_object() {
                    let obj_value = value.dyn_into::<Object>()?;
                    let nested_obj = StrictObject::new_with_data(schema.to_js(), obj_value.into())?;
                    Ok(nested_obj.to_js())
                } else {
                    Err(JsValue::from_str("Nested array elements require object values"))
                }
            }
            FieldInfo::Tensor => {
                if value.is_object() || value.is_array() {
                    Ok(value)
                } else {
                    Err(JsValue::from_str("Tensor array elements require object or array values"))
                }
            }
            FieldInfo::Matrix => {
                if value.is_object() || value.is_array() {
                    Ok(value)
                } else {
                    Err(JsValue::from_str("Matrix array elements require object or array values"))
                }
            }
            FieldInfo::Vector => {
                if value.is_array() {
                    Ok(value)
                } else {
                    Err(JsValue::from_str("Vector array elements require array values"))
                }
            }
            FieldInfo::SparseMatrix => {
                if value.is_object() {
                    Ok(value)
                } else {
                    Err(JsValue::from_str("Sparse matrix array elements require object values"))
                }
            }
            FieldInfo::Quantized => {
                if value.is_array() {
                    Ok(value)
                } else {
                    Err(JsValue::from_str("Quantized array elements require array values"))
                }
            }
            FieldInfo::GPU => {
                if value.is_object() || value.is_array() {
                    Ok(value)
                } else {
                    Err(JsValue::from_str("GPU array elements require object or array values"))
                }
            }
            FieldInfo::SIMD => {
                if value.is_array() {
                    Ok(value)
                } else {
                    Err(JsValue::from_str("SIMD array elements require array values"))
                }
            }
            FieldInfo::Array(_) => {
                Err(JsValue::from_str("Nested arrays not supported"))
            }
        }
    }

    fn validate_simple_value(&self, type_str: &str, value: JsValue) -> Result<JsValue, JsValue> {
        match type_str {
            "string" => {
                if value.is_string() {
                    Ok(value)
                } else {
                    Ok(JsValue::from_str(&value.as_string().unwrap_or_default()))
                }
            }
            "bool" | "boolean" => {
                let bool_value = value.as_f64().map(|n| n != 0.0)
                    .or_else(|| value.as_bool())
                    .unwrap_or(false);
                Ok(JsValue::from_bool(bool_value))
            }
            _ => {
                let num_value = value.as_f64().unwrap_or(0.0);
                let clamped_value = self.clamp_value(type_str, num_value);
                Ok(JsValue::from_f64(clamped_value))
            }
        }
    }

    fn set_simple_field(&self, field: &str, type_str: &str, value: JsValue) -> Result<(), JsValue> {
        let validated_value = self.validate_simple_value(type_str, value)?;
        Reflect::set(&self.data, &JsValue::from_str(field), &validated_value)
            .map(|_| ())
            .map_err(|_| JsValue::from_str("Failed to set simple field"))
    }

    #[wasm_bindgen(js_name = newWithData)]
    pub fn new_with_data(schema: JsValue, initial_data: JsValue) -> Result<StrictObject, JsValue> {
        let mut obj = StrictObject::new(schema)?;
        
        if let Ok(data_obj) = initial_data.dyn_into::<Object>() {
            let keys = Object::keys(&data_obj);
            
            for i in 0..keys.length() {
                let key = keys.get(i);
                let key_str = key.as_string().ok_or_else(|| JsValue::from_str("Invalid key"))?;
                
                let value = Reflect::get(&data_obj, &key)?;
                obj.set_field(&key_str, value)?;
            }
        }
        
        Ok(obj)
    }

    #[wasm_bindgen(js_name = getField)]
    pub fn get_field(&self, field: &str) -> Result<JsValue, JsValue> {
        Reflect::get(&self.data, &JsValue::from_str(field))
            .map_err(|_| JsValue::from_str(&format!("Field '{}' not found", field)))
    }

    #[wasm_bindgen(js_name = getArrayField)]
    pub fn get_array_field(&self, field: &str) -> Result<Array, JsValue> {
        let value = self.get_field(field)?;
        value.dyn_into::<Array>()
            .map_err(|_| JsValue::from_str(&format!("Field '{}' is not an array", field)))
    }

    #[wasm_bindgen(js_name = getArrayElement)]
    pub fn get_array_element(&self, field: &str, index: usize) -> Result<JsValue, JsValue> {
        let array = self.get_array_field(field)?;
        if index < array.length() as usize {
            Ok(array.get(index as u32))
        } else {
            Err(JsValue::from_str("Array index out of bounds"))
        }
    }

    #[wasm_bindgen(js_name = pushToArray)]
    pub fn push_to_array(&mut self, field: &str, value: JsValue) -> Result<(), JsValue> {
        if let Some(field_info) = self.field_cache.get(field) {
            match field_info {
                FieldInfo::Array(element_info) => {
                    let array = self.get_array_field(field)?;
                    let validated_value = self.validate_array_element(element_info, value)?;
                    array.push(&validated_value);
                    Ok(())
                }
                _ => Err(JsValue::from_str(&format!("Field '{}' is not an array", field))),
            }
        } else {
            Err(JsValue::from_str(&format!("Field '{}' not found", field)))
        }
    }

    #[wasm_bindgen(js_name = getFieldAsString)]
    pub fn get_field_as_string(&self, field: &str) -> Result<String, JsValue> {
        let value = self.get_field(field)?;
        value.as_string()
            .ok_or_else(|| JsValue::from_str(&format!("Field '{}' is not a string", field)))
    }

    #[wasm_bindgen(js_name = getFieldAsNumber)]
    pub fn get_field_as_number(&self, field: &str) -> Result<f64, JsValue> {
        let value = self.get_field(field)?;
        value.as_f64()
            .ok_or_else(|| JsValue::from_str(&format!("Field '{}' is not a number", field)))
    }

    #[wasm_bindgen(js_name = getFieldAsBoolean)]
    pub fn get_field_as_boolean(&self, field: &str) -> Result<bool, JsValue> {
        let value = self.get_field(field)?;
        if let Some(num) = value.as_f64() {
            Ok(num != 0.0)
        } else if let Some(b) = value.as_bool() {
            Ok(b)
        } else {
            Ok(false)
        }
    }

    #[wasm_bindgen(js_name = getNestedObject)]
    pub fn get_nested_object(&self, field: &str) -> Result<StrictObject, JsValue> {
        if self.schema.is_nested_field(field) {
            let value = self.get_field(field)?;
            if let Ok(obj_value) = value.dyn_into::<Object>() {
                if let Some(nested_schema) = self.schema.get_nested_schema(field) {
                    return StrictObject::new_with_data(nested_schema.to_js(), obj_value.into());
                }
            }
            Err(JsValue::from_str(&format!("Field '{}' is not a valid nested object", field)))
        } else {
            Err(JsValue::from_str(&format!("Field '{}' is not a nested object", field)))
        }
    }

    fn clamp_value(&self, type_str: &str, value: f64) -> f64 {
        match type_str {
            "u8" => value.clamp(0.0, u8::MAX as f64),
            "i8" => value.clamp(i8::MIN as f64, i8::MAX as f64),
            "u16" => value.clamp(0.0, u16::MAX as f64),
            "i16" => value.clamp(i16::MIN as f64, i16::MAX as f64),
            "u32" => value.clamp(0.0, u32::MAX as f64),
            "i32" => value.clamp(i32::MIN as f64, i32::MAX as f64),
            "bool" | "boolean" => {
                if value != 0.0 { 1.0 } else { 0.0 }
            }
            _ => value,
        }
    }

    #[wasm_bindgen(js_name = getSchema)]
    pub fn get_schema(&self) -> JsValue {
        self.schema.to_js()
    }

    #[wasm_bindgen(js_name = toJS)]
    pub fn to_js(&self) -> JsValue {
        self.data.clone().into()
    }

    #[wasm_bindgen(js_name = fieldNames)]
    pub fn field_names(&self) -> Vec<JsValue> {
        self.schema.field_names()
    }

    #[wasm_bindgen(js_name = isNestedField)]
    pub fn is_nested_field(&self, field: &str) -> bool {
        self.schema.is_nested_field(field)
    }

    #[wasm_bindgen(js_name = isArrayField)]
    pub fn is_array_field(&self, field: &str) -> bool {
        self.schema.is_array_field(field)
    }

    #[wasm_bindgen(js_name = arrayLength)]
    pub fn array_length(&self, field: &str) -> Result<usize, JsValue> {
        let array = self.get_array_field(field)?;
        Ok(array.length() as usize)
    }

    // AI/ML specific field type checks
    #[wasm_bindgen(js_name = isTensorField)]
    pub fn is_tensor_field(&self, field: &str) -> bool {
        self.schema.is_tensor_field(field)
    }

    #[wasm_bindgen(js_name = isMatrixField)]
    pub fn is_matrix_field(&self, field: &str) -> bool {
        self.schema.is_matrix_field(field)
    }

    #[wasm_bindgen(js_name = isVectorField)]
    pub fn is_vector_field(&self, field: &str) -> bool {
        self.schema.is_vector_field(field)
    }

    #[wasm_bindgen(js_name = isGPUField)]
    pub fn is_gpu_field(&self, field: &str) -> bool {
        self.schema.is_gpu_field(field)
    }

    #[wasm_bindgen(js_name = isSIMDField)]
    pub fn is_simd_field(&self, field: &str) -> bool {
        self.schema.is_simd_field(field)
    }

    // AI/ML specific getters with validation
    #[wasm_bindgen(js_name = getTensorField)]
    pub fn get_tensor_field(&self, field: &str) -> Result<JsValue, JsValue> {
        if self.is_tensor_field(field) {
            self.get_field(field)
        } else {
            Err(JsValue::from_str(&format!("Field '{}' is not a tensor", field)))
        }
    }

    #[wasm_bindgen(js_name = getMatrixField)]
    pub fn get_matrix_field(&self, field: &str) -> Result<JsValue, JsValue> {
        if self.is_matrix_field(field) {
            self.get_field(field)
        } else {
            Err(JsValue::from_str(&format!("Field '{}' is not a matrix", field)))
        }
    }

    #[wasm_bindgen(js_name = getVectorField)]
    pub fn get_vector_field(&self, field: &str) -> Result<JsValue, JsValue> {
        if self.is_vector_field(field) {
            self.get_field(field)
        } else {
            Err(JsValue::from_str(&format!("Field '{}' is not a vector", field)))
        }
    }

    #[wasm_bindgen(js_name = getGPUField)]
    pub fn get_gpu_field(&self, field: &str) -> Result<JsValue, JsValue> {
        if self.is_gpu_field(field) {
            self.get_field(field)
        } else {
            Err(JsValue::from_str(&format!("Field '{}' is not a GPU field", field)))
        }
    }

    #[wasm_bindgen(js_name = getSIMDField)]
    pub fn get_simd_field(&self, field: &str) -> Result<JsValue, JsValue> {
        if self.is_simd_field(field) {
            self.get_field(field)
        } else {
            Err(JsValue::from_str(&format!("Field '{}' is not a SIMD field", field)))
        }
    }
}

