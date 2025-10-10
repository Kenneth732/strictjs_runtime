// src/strict_function/implementation.rs
use wasm_bindgen::prelude::*;
use js_sys::{Function, Array, Object, Reflect};
use crate::types::heap_type::{HeapType, TypeCapabilities, JsHeapType, JsTypeCapabilities};
use std::collections::HashMap;
use std::sync::Mutex;

#[wasm_bindgen]
pub struct StrictFunction {
    js_function: Function,
    arg_types: Vec<HeapType>,
    return_type: HeapType,
    arg_type_strings: Vec<&'static str>,
    return_type_string: &'static str,
    // Cache for processed argument patterns to improve performance
    #[wasm_bindgen(skip)]
    arg_pattern_cache: Mutex<HashMap<String, Array>>,
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct StrictFunctionResult {
    value: JsValue,
    result_type: String,
    capabilities: JsTypeCapabilities,
}

#[wasm_bindgen]
impl StrictFunctionResult {
    #[wasm_bindgen(getter)]
    pub fn value(&self) -> JsValue {
        self.value.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn result_type(&self) -> String {
        self.result_type.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn capabilities(&self) -> JsTypeCapabilities {
        self.capabilities
    }
    
    #[wasm_bindgen(js_name = toNumber)]
    pub fn to_number(&self) -> Option<f64> {
        self.value.as_f64()
    }
    
    #[wasm_bindgen(js_name = toBoolean)]
    pub fn to_boolean(&self) -> Option<bool> {
        self.value.as_bool()
    }
    
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> Option<String> {
        self.value.as_string()
    }
    
    #[wasm_bindgen(js_name = toObject)]
    pub fn to_object(&self) -> Option<Object> {
        self.value.clone().dyn_into::<Object>().ok()
    }
    
    #[wasm_bindgen(js_name = supportsOperation)]
    pub fn supports_operation(&self, operation: &str) -> bool {
        let heap_type = HeapType::from_string(&self.result_type).unwrap_or(HeapType::Any);
        heap_type.supports_operation(operation)
    }
    
    #[wasm_bindgen(js_name = getRecommendedOperations)]
    pub fn get_recommended_operations(&self) -> Array {
        let heap_type = HeapType::from_string(&self.result_type).unwrap_or(HeapType::Any);
        heap_type.recommended_operations()
            .into_iter()
            .map(JsValue::from)
            .collect()
    }
}

// Helper trait for internal conversions
trait CapabilitiesExt {
    fn to_js(&self) -> JsTypeCapabilities;
}

impl CapabilitiesExt for TypeCapabilities {
    fn to_js(&self) -> JsTypeCapabilities {
        JsTypeCapabilities::new(self.bits())
    }
}

trait HeapTypeExt {
    fn to_js(&self) -> JsHeapType;
}

impl HeapTypeExt for HeapType {
    fn to_js(&self) -> JsHeapType {
        JsHeapType::new(*self)
    }
}

#[wasm_bindgen]
impl StrictFunction {
    #[wasm_bindgen(constructor)]
    pub fn new(
        js_function: Function,
        arg_types: JsValue,
        return_type: HeapType
    ) -> Result<StrictFunction, JsValue> {
        let arg_types_array: Array = arg_types.dyn_into()?;
        let arg_count = arg_types_array.length() as usize;
        
        let mut arg_types_vec = Vec::with_capacity(arg_count);
        let mut arg_type_strings = Vec::with_capacity(arg_count);
        
        for i in 0..arg_types_array.length() {
            let heap_type_val = arg_types_array.get(i);
            let heap_type = HeapType::from_js_value(heap_type_val)?;
            arg_types_vec.push(heap_type);
            arg_type_strings.push(heap_type_to_str(heap_type));
        }
        
        let return_type_string = heap_type_to_str(return_type);
        
        Ok(StrictFunction {
            js_function,
            arg_types: arg_types_vec,
            return_type,
            arg_type_strings,
            return_type_string,
            arg_pattern_cache: Mutex::new(HashMap::new()),
        })
    }
    
    // NEW: Create function with capability-based validation
    #[wasm_bindgen(js_name = createWithCapabilities)]
    pub fn create_with_capabilities(
        js_function: Function,
        arg_types: JsValue,
        return_type: HeapType,
        required_capabilities: JsTypeCapabilities
    ) -> Result<StrictFunction, JsValue> {
        let strict_func = Self::new(js_function, arg_types, return_type)?;
        
        // Validate that the function signature supports required capabilities
        let func_capabilities = strict_func.get_capabilities();
        if !func_capabilities.supports_all(&required_capabilities) {
            return Err(JsValue::from_str(&format!(
                "Function does not support required capabilities. Missing: {:?}",
                required_capabilities.without(&func_capabilities).to_names()
            )));
        }
        
        Ok(strict_func)
    }
    
    #[wasm_bindgen(js_name = callComplex)]
    pub fn call_complex(&self, args: JsValue, context: JsValue) -> Result<StrictFunctionResult, JsValue> {
        let args_array: Array = args.dyn_into()?;
        let processed_args = self.process_arguments(&args_array)?;
        
        let context = if context.is_null() || context.is_undefined() {
            JsValue::NULL
        } else {
            context
        };
        
        let result = self.js_function.apply(&context, &processed_args)
            .map_err(|e| self.enhance_error(e, "Function call failed"))?;
        
        self.validate_and_wrap_result(result)
    }
    
    // Enhanced argument processing with capabilities
    fn process_arguments(&self, args_array: &Array) -> Result<Array, JsValue> {
        let arg_count = args_array.length() as usize;
        
        // Check cache first
        if let Some(cached) = self.get_cached_arguments(args_array) {
            return Ok(cached);
        }
        
        let processed_args = Array::new_with_length(arg_count as u32);
        
        for i in 0..arg_count {
            let arg_val = args_array.get(i as u32);
            let processed_val = if (i as usize) < self.arg_types.len() {
                self.process_argument_enhanced(i, arg_val)?
            } else {
                // For extra arguments, pass through but validate type if possible
                self.validate_any_argument(i, arg_val)?
            };
            processed_args.set(i as u32, processed_val);
        }
        
        // Cache the processed arguments
        self.cache_arguments(args_array, &processed_args);
        
        Ok(processed_args)
    }
    
    fn process_argument_enhanced(&self, index: usize, arg_val: JsValue) -> Result<JsValue, JsValue> {
        let arg_type = self.arg_types[index];
        let capabilities = arg_type.capabilities();
        
        // Use capabilities for smarter type conversion
        if capabilities.contains(TypeCapabilities::NUMERIC_OPS) {
            self.convert_to_numeric(arg_val, arg_type, index)
        } else if capabilities.contains(TypeCapabilities::STRING_OPS) {
            self.convert_to_string(arg_val, index)
        } else if capabilities.contains(TypeCapabilities::ITERABLE) {
            self.convert_to_iterable(arg_val, arg_type, index)
        } else {
            // Strict type checking for non-convertible types
            self.validate_strict_type(arg_val, arg_type, index)
        }
    }
    
    fn convert_to_numeric(&self, arg_val: JsValue, target_type: HeapType, index: usize) -> Result<JsValue, JsValue> {
        if let Some(num) = arg_val.as_f64() {
            Ok(JsValue::from_f64(self.clamp_value(target_type, num)))
        } else if let Some(s) = arg_val.as_string() {
            s.parse::<f64>()
                .map(|n| JsValue::from_f64(self.clamp_value(target_type, n)))
                .map_err(|_| self.enhance_error_with_type_info(
                    JsValue::from_str("Cannot convert string to number"),
                    index,
                    "numeric"
                ))
        } else if let Some(b) = arg_val.as_bool() {
            Ok(JsValue::from_f64(if b { 1.0 } else { 0.0 }))
        } else if arg_val.is_null() || arg_val.is_undefined() {
            Ok(JsValue::from_f64(0.0))
        } else {
            Err(self.enhance_error_with_type_info(
                JsValue::from_str("Value cannot be converted to numeric type"),
                index,
                "numeric"
            ))
        }
    }
    
    fn convert_to_string(&self, arg_val: JsValue, index: usize) -> Result<JsValue, JsValue> {
        if let Some(s) = arg_val.as_string() {
            Ok(JsValue::from_str(&s))
        } else if let Some(num) = arg_val.as_f64() {
            Ok(JsValue::from_str(&num.to_string()))
        } else if let Some(b) = arg_val.as_bool() {
            Ok(JsValue::from_str(if b { "true" } else { "false" }))
        } else if arg_val.is_null() {
            Ok(JsValue::from_str("null"))
        } else if arg_val.is_undefined() {
            Ok(JsValue::from_str("undefined"))
        } else if arg_val.is_object() {
            Ok(JsValue::from_str("[object]"))
        } else {
            Err(self.enhance_error_with_type_info(
                JsValue::from_str("Value cannot be converted to string"),
                index,
                "string"
            ))
        }
    }
    
    fn convert_to_iterable(&self, arg_val: JsValue, target_type: HeapType, index: usize) -> Result<JsValue, JsValue> {
        if arg_val.is_object() {
            Ok(arg_val)
        } else if target_type == HeapType::Array {
            if let Some(s) = arg_val.as_string() {
                // Convert string to array of characters
                let array = Array::new();
                for ch in s.chars() {
                    array.push(&JsValue::from_str(&ch.to_string()));
                }
                Ok(array.into())
            } else {
                Err(self.enhance_error_with_type_info(
                    JsValue::from_str("Value cannot be converted to iterable type"),
                    index,
                    "iterable"
                ))
            }
        } else {
            Err(self.enhance_error_with_type_info(
                JsValue::from_str("Value cannot be converted to iterable type"),
                index,
                "iterable"
            ))
        }
    }
    
    fn validate_strict_type(&self, arg_val: JsValue, target_type: HeapType, index: usize) -> Result<JsValue, JsValue> {
        match target_type {
            HeapType::Struct | HeapType::Array | HeapType::Map | HeapType::Date | HeapType::Buffer => {
                if arg_val.is_object() {
                    Ok(arg_val)
                } else {
                    Err(self.enhance_error_with_type_info(
                        JsValue::from_str("Must be an object"),
                        index,
                        "object"
                    ))
                }
            },
            HeapType::Null => {
                if arg_val.is_null() {
                    Ok(arg_val)
                } else {
                    Err(self.enhance_error_with_type_info(
                        JsValue::from_str("Must be null"),
                        index,
                        "null"
                    ))
                }
            },
            HeapType::Undefined => {
                if arg_val.is_undefined() {
                    Ok(arg_val)
                } else {
                    Err(self.enhance_error_with_type_info(
                        JsValue::from_str("Must be undefined"),
                        index,
                        "undefined"
                    ))
                }
            },
            HeapType::Symbol => {
                if arg_val.is_symbol() {
                    Ok(arg_val)
                } else {
                    Err(self.enhance_error_with_type_info(
                        JsValue::from_str("Must be a symbol"),
                        index,
                        "symbol"
                    ))
                }
            },
            HeapType::Bool => {
                if let Some(b) = arg_val.as_bool() {
                    Ok(JsValue::from_bool(b))
                } else {
                    Err(self.enhance_error_with_type_info(
                        JsValue::from_str("Must be a boolean"),
                        index,
                        "boolean"
                    ))
                }
            },
            _ => Ok(arg_val), // For Any and other types, pass through
        }
    }
    
    fn validate_any_argument(&self, index: usize, arg_val: JsValue) -> Result<JsValue, JsValue> {
        // For extra arguments beyond defined types, do basic validation
        if arg_val.is_symbol() && !self.supports_operation("symbol") {
            return Err(JsValue::from_str(&format!(
                "Extra argument {}: symbol not supported by function capabilities", index
            )));
        }
        Ok(arg_val)
    }
    
    fn enhance_error(&self, error: JsValue, message: &str) -> JsValue {
        let error_string = error.as_string().unwrap_or_default();
        JsValue::from_str(&format!("{}: {}", message, error_string))
    }
    
    fn enhance_error_with_type_info(&self, error: JsValue, index: usize, _operation: &str) -> JsValue {
        let arg_type = &self.arg_types[index];
        let capabilities = arg_type.capabilities();
        let supported_ops = capabilities.to_names().join(", ");
        
        let base_error = error.as_string().unwrap_or_default();
        JsValue::from_str(&format!(
            "Argument {} ({}): {} - Supported operations: [{}]",
            index, arg_type.to_string(), base_error, supported_ops
        ))
    }
    
    // Cache management methods
    fn get_cached_arguments(&self, args_array: &Array) -> Option<Array> {
        let cache_key = self.generate_cache_key(args_array).ok()?;
        let cache = self.arg_pattern_cache.lock().ok()?;
        cache.get(&cache_key).cloned()
    }
    
    fn cache_arguments(&self, args_array: &Array, processed_args: &Array) {
        if let Ok(cache_key) = self.generate_cache_key(args_array) {
            if let Ok(mut cache) = self.arg_pattern_cache.lock() {
                if cache.len() < 100 { // Limit cache size
                    cache.insert(cache_key, processed_args.clone());
                }
            }
        }
    }
    
    fn generate_cache_key(&self, args_array: &Array) -> Result<String, JsValue> {
        let mut key_parts = Vec::new();
        
        for i in 0..args_array.length() {
            let arg_val = args_array.get(i);
            if let Some(s) = arg_val.as_string() {
                key_parts.push(s);
            } else if let Some(n) = arg_val.as_f64() {
                key_parts.push(n.to_string());
            } else if let Some(b) = arg_val.as_bool() {
                key_parts.push(b.to_string());
            } else {
                key_parts.push(format!("{:?}", arg_val));
            }
        }
        
        Ok(key_parts.join("|"))
    }
    
    fn validate_and_wrap_result(&self, result: JsValue) -> Result<StrictFunctionResult, JsValue> {
        let result_type = self.return_type;
        let capabilities = result_type.capabilities();
        
        let (value, result_type_str) = match result_type {
            HeapType::Number => {
                if result.as_f64().is_some() {
                    (result, "number")
                } else {
                    return Err(JsValue::from_str("Function must return a number"));
                }
            },
            HeapType::Any => {
                self.wrap_any_result(result)
            },
            HeapType::U8 | HeapType::I8 | HeapType::U16 | HeapType::I16 
            | HeapType::U32 | HeapType::I32 | HeapType::U64 | HeapType::I64 => {
                let num = result.as_f64()
                    .or_else(|| result.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                    .ok_or(JsValue::from_str("Function must return a number or boolean"))?;
                let clamped = self.clamp_value(result_type, num);
                (JsValue::from_f64(clamped), "number")
            },
            HeapType::F32 | HeapType::F64 => {
                if let Some(num) = result.as_f64() {
                    (JsValue::from_f64(num), "number")
                } else {
                    return Err(JsValue::from_str("Function must return a number"));
                }
            },
            HeapType::Bool => {
                if let Some(b) = result.as_bool() {
                    (JsValue::from_bool(b), "boolean")
                } else if let Some(num) = result.as_f64() {
                    let bool_val = num != 0.0;
                    (JsValue::from_bool(bool_val), "boolean")
                } else {
                    return Err(JsValue::from_str("Function must return a boolean or number"));
                }
            },
            HeapType::Str | HeapType::Str16 => {
                if let Some(s) = result.as_string() {
                    (JsValue::from_str(&s), "string")
                } else {
                    return Err(JsValue::from_str("Function must return a string"));
                }
            },
            HeapType::Struct | HeapType::Array | HeapType::Map | HeapType::Date | HeapType::Buffer => {
                if result.is_object() {
                    (result, "object")
                } else {
                    return Err(JsValue::from_str("Function must return an object"));
                }
            },
            HeapType::Null => {
                if result.is_null() {
                    (JsValue::NULL, "null")
                } else {
                    return Err(JsValue::from_str("Function must return null"));
                }
            },
            HeapType::Undefined => {
                if result.is_undefined() {
                    (JsValue::UNDEFINED, "undefined")
                } else {
                    return Err(JsValue::from_str("Function must return undefined"));
                }
            },
            HeapType::Symbol => {
                if result.is_symbol() {
                    (result, "symbol")
                } else {
                    return Err(JsValue::from_str("Function must return a symbol"));
                }
            },
        };
        
        Ok(StrictFunctionResult {
            value,
            result_type: result_type_str.to_string(),
            capabilities: capabilities.to_js(),
        })
    }
    
    fn wrap_any_result(&self, result: JsValue) -> (JsValue, &'static str) {
        if let Some(num) = result.as_f64() {
            (JsValue::from_f64(num), "number")
        } else if let Some(b) = result.as_bool() {
            (JsValue::from_bool(b), "boolean")
        } else if let Some(s) = result.as_string() {
            (JsValue::from_str(&s), "string")
        } else if result.is_null() {
            (JsValue::NULL, "null")
        } else if result.is_undefined() {
            (JsValue::UNDEFINED, "undefined")
        } else if result.is_symbol() {
            (result, "symbol")
        } else if result.is_object() {
            (result, "object")
        } else {
            (result, "unknown")
        }
    }

    #[wasm_bindgen(js_name = validateArguments)]
    pub fn validate_arguments(&self, args: JsValue) -> Result<(), JsValue> {
        let args_array: Array = args.dyn_into()?;
        
        if args_array.length() < (self.arg_types.len() as u32) {
            return Err(JsValue::from_str(&format!(
                "Expected at least {} arguments, got {}",
                self.arg_types.len(),
                args_array.length()
            )));
        }
        
        for i in 0..self.arg_types.len() {
            let arg_val = args_array.get(i as u32);
            self.validate_argument(i, arg_val)?;
        }
        
        Ok(())
    }
    
    fn validate_argument(&self, index: usize, arg_val: JsValue) -> Result<(), JsValue> {
        let arg_type = self.arg_types[index];
        let capabilities = arg_type.capabilities();
        
        // Use capabilities for validation
        if capabilities.contains(TypeCapabilities::NUMERIC_OPS) {
            self.validate_numeric_argument(arg_val, index)
        } else if capabilities.contains(TypeCapabilities::STRING_OPS) {
            self.validate_string_argument(arg_val, index)
        } else {
            self.validate_strict_argument(arg_val, arg_type, index)
        }
    }
    
    fn validate_numeric_argument(&self, arg_val: JsValue, index: usize) -> Result<(), JsValue> {
        if !arg_val.as_f64().is_some() && !arg_val.is_string() && !arg_val.as_bool().is_some() {
            return Err(JsValue::from_str(&format!(
                "Argument {} must be a number, string, or boolean", index
            )));
        }
        
        if arg_val.is_string() {
            let str_val = arg_val.as_string().unwrap();
            if str_val.parse::<f64>().is_err() {
                return Err(JsValue::from_str(&format!(
                    "Argument {}: string cannot be converted to number", index
                )));
            }
        }
        Ok(())
    }
    
    fn validate_string_argument(&self, _arg_val: JsValue, _index: usize) -> Result<(), JsValue> {
        // Strings accept any type (they'll be converted)
        Ok(())
    }
    
    fn validate_strict_argument(&self, arg_val: JsValue, arg_type: HeapType, index: usize) -> Result<(), JsValue> {
        match arg_type {
            HeapType::Struct | HeapType::Array | HeapType::Map | HeapType::Date | HeapType::Buffer => {
                if !arg_val.is_object() {
                    return Err(JsValue::from_str(&format!(
                        "Argument {} must be an object", index
                    )));
                }
            },
            HeapType::Null => {
                if !arg_val.is_null() {
                    return Err(JsValue::from_str(&format!(
                        "Argument {} must be null", index
                    )));
                }
            },
            HeapType::Undefined => {
                if !arg_val.is_undefined() {
                    return Err(JsValue::from_str(&format!(
                        "Argument {} must be undefined", index
                    )));
                }
            },
            HeapType::Symbol => {
                if !arg_val.is_symbol() {
                    return Err(JsValue::from_str(&format!(
                        "Argument {} must be a symbol", index
                    )));
                }
            },
            HeapType::Bool => {
                if arg_val.as_bool().is_none() {
                    return Err(JsValue::from_str(&format!(
                        "Argument {} must be a boolean", index
                    )));
                }
            },
            _ => {} // Any and other types are more flexible
        }
        Ok(())
    }
    
    #[wasm_bindgen(js_name = call)]
    pub fn call(&self, args: JsValue) -> Result<f64, JsValue> {
        let args_array: Array = args.dyn_into()?;
        
        if args_array.length() as usize != self.arg_types.len() {
            return Err(JsValue::from_str(&format!(
                "Expected {} arguments, got {}",
                self.arg_types.len(),
                args_array.length()
            )));
        }
        
        let processed_args = self.process_arguments(&args_array)?;
        
        let result = self.js_function.apply(&JsValue::NULL, &processed_args)
            .map_err(|e| JsValue::from_str(&format!("Function call failed: {:?}", e)))?;
        
        self.process_result(result)
    }

    fn process_result(&self, result: JsValue) -> Result<f64, JsValue> {
        match self.return_type {
            HeapType::Number => {
                result.as_f64()
                    .ok_or(JsValue::from_str("Function must return a number"))
            },
            HeapType::Any => {
                result.as_f64()
                    .or_else(|| result.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                    .or_else(|| result.as_string().map(|s| s.len() as f64))
                    .or(Some(0.0))
                    .ok_or(JsValue::from_str("Invalid return value"))
            },
            HeapType::U8 | HeapType::I8 | HeapType::U16 | HeapType::I16 
            | HeapType::U32 | HeapType::I32 | HeapType::Bool => {
                let num = result.as_f64()
                    .or_else(|| result.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                    .ok_or(JsValue::from_str("Function must return a number or boolean"))?;
                
                Ok(self.clamp_value(self.return_type, num))
            },
            HeapType::U64 | HeapType::I64 => {
                result.as_f64()
                    .ok_or(JsValue::from_str("Function must return a number"))
            },
            HeapType::F32 | HeapType::F64 => {
                result.as_f64()
                    .ok_or(JsValue::from_str("Function must return a number"))
            },
            HeapType::Str | HeapType::Str16 => {
                Ok(result.as_string().map(|s| s.len() as f64).unwrap_or(0.0))
            },
            HeapType::Struct | HeapType::Array | HeapType::Map | HeapType::Date | HeapType::Buffer => {
                Ok(if result.is_object() { 1.0 } else { 0.0 })
            },
            HeapType::Null => {
                Ok(if result.is_null() { 1.0 } else { 0.0 })
            },
            HeapType::Undefined => {
                Ok(if result.is_undefined() { 1.0 } else { 0.0 })
            },
            HeapType::Symbol => {
                Ok(if result.is_symbol() { 1.0 } else { 0.0 })
            },
        }
    }

    fn clamp_value(&self, heap_type: HeapType, value: f64) -> f64 {
        match heap_type {
            HeapType::Number => value,
            HeapType::U8 => value.clamp(0.0, u8::MAX as f64),
            HeapType::I8 => value.clamp(i8::MIN as f64, i8::MAX as f64),
            HeapType::U16 => value.clamp(0.0, u16::MAX as f64),
            HeapType::I16 => value.clamp(i16::MIN as f64, i16::MAX as f64),
            HeapType::U32 => value.clamp(0.0, u32::MAX as f64),
            HeapType::I32 => value.clamp(i32::MIN as f64, i32::MAX as f64),
            HeapType::U64 => value.clamp(0.0, u64::MAX as f64),
            HeapType::I64 => value.clamp(i64::MIN as f64, i64::MAX as f64),
            HeapType::F32 | HeapType::F64 => value,
            HeapType::Bool => {
                if value != 0.0 { 1.0 } else { 0.0 }
            },
            _ => value,
        }
    }

    // Enhanced capability-based methods
    #[wasm_bindgen(js_name = supportsOperation)]
    pub fn supports_operation(&self, operation: &str) -> bool {
        // Check if all argument types support this operation
        let args_support = self.arg_types.iter().all(|arg_type| 
            arg_type.supports_operation(operation)
        );
        
        // Check if return type supports this operation
        let return_supports = self.return_type.supports_operation(operation);
        
        args_support && return_supports
    }

    #[wasm_bindgen(js_name = getCapabilities)]
    pub fn get_capabilities(&self) -> JsTypeCapabilities {
        if self.arg_types.is_empty() {
            return self.return_type.capabilities().to_js();
        }
        
        let mut caps = self.return_type.capabilities();
        
        // Function capabilities are intersection of all argument capabilities
        for arg_type in &self.arg_types {
            caps = caps.intersection(arg_type.capabilities());
        }
        
        // Add callable capability since this is a function
        caps = caps.union(TypeCapabilities::CALLABLE);
        
        caps.to_js()
    }

    #[wasm_bindgen(js_name = getCompatibleOperations)]
    pub fn get_compatible_operations(&self) -> Array {
        if self.arg_types.is_empty() {
            return self.return_type.recommended_operations()
                .into_iter()
                .map(JsValue::from)
                .collect();
        }
        
        let mut common_ops = self.arg_types[0].capabilities();
        
        for arg_type in &self.arg_types[1..] {
            common_ops = common_ops.intersection(arg_type.capabilities());
        }
        
        // Also intersect with return type capabilities
        common_ops = common_ops.intersection(self.return_type.capabilities());
        
        common_ops.to_names()
            .into_iter()
            .map(JsValue::from)
            .collect()
    }

    // Smart result type inference
    #[wasm_bindgen(js_name = inferResultType)]
    pub fn infer_result_type(&self, args: JsValue) -> Result<JsHeapType, JsValue> {
        let args_array: Array = args.dyn_into()?;
        
        if args_array.length() == 0 || !self.return_type.is_numeric() {
            return Ok(self.return_type.to_js());
        }
        
        // For numeric operations, infer the result type based on input types
        let mut result_type = self.return_type;
        
        for i in 0..args_array.length().min(self.arg_types.len() as u32) {
            let arg_val = args_array.get(i);
            if let Some(arg_type) = self.infer_value_type(arg_val) {
                if let Some(promoted) = result_type.get_binary_result_type(&arg_type) {
                    result_type = promoted;
                }
            }
        }
        
        Ok(result_type.to_js())
    }
    
    fn infer_value_type(&self, value: JsValue) -> Option<HeapType> {
        if value.is_string() {
            Some(HeapType::Str)
        } else if value.as_f64().is_some() {
            Some(HeapType::Number)
        } else if value.as_bool().is_some() {
            Some(HeapType::Bool)
        } else if value.is_object() {
            Some(HeapType::Any)
        } else if value.is_null() {
            Some(HeapType::Null)
        } else if value.is_undefined() {
            Some(HeapType::Undefined)
        } else if value.is_symbol() {
            Some(HeapType::Symbol)
        } else {
            None
        }
    }

    // Batch argument processing
    #[wasm_bindgen(js_name = processArgumentsBatch)]
    pub fn process_arguments_batch(&self, args_batch: JsValue) -> Result<Array, JsValue> {
        let batches: Array = args_batch.dyn_into()?;
        let results = Array::new_with_length(batches.length());
        
        for i in 0..batches.length() {
            let args = batches.get(i);
            let processed = self.process_arguments(&args.dyn_into()?)?;
            results.set(i, processed.into());
        }
        
        Ok(results)
    }

    // Get type compatibility report
    #[wasm_bindgen(js_name = getTypeCompatibility)]
    pub fn get_type_compatibility(&self, args: JsValue) -> Result<JsValue, JsValue> {
        let args_array: Array = args.dyn_into()?;
        let report = Object::new();
        
        for i in 0..args_array.length().min(self.arg_types.len() as u32) {
            let arg_val = args_array.get(i);
            let arg_type = self.arg_types[i as usize];
            let compatible = self.validate_argument(i as usize, arg_val).is_ok();
            
            let info = Object::new();
            Reflect::set(&info, &"expected".into(), &arg_type.to_js_value())?;
            Reflect::set(&info, &"compatible".into(), &compatible.into())?;
            Reflect::set(&info, &"capabilities".into(), &arg_type.get_capabilities_js())?;
            
            if compatible {
                let recommended = arg_type.recommended_operations();
                let recommended_array = Array::new_with_length(recommended.len() as u32);
                for (j, op) in recommended.iter().enumerate() {
                    recommended_array.set(j as u32, JsValue::from_str(op));
                }
                Reflect::set(&info, &"recommendedOperations".into(), &recommended_array.into())?;
            }
            
            Reflect::set(&report, &i.into(), &info.into())?;
        }
        
        Ok(report.into())
    }

    // Chain functions with type compatibility checking
    #[wasm_bindgen(js_name = chain)]
    pub fn chain(&self, next_function: &StrictFunction) -> Result<StrictFunction, JsValue> {
        if next_function.arg_types.is_empty() {
            return Err(JsValue::from_str("Next function must accept at least one argument"));
        }
        
        if self.return_type != next_function.arg_types[0] {
            return Err(JsValue::from_str(&format!(
                "Return type of first function ({}) must match argument type of second function ({})",
                self.return_type.to_string(),
                next_function.arg_types[0].to_string()
            )));
        }
        
        // Create combined argument types (our args + their remaining args)
        let mut combined_arg_types = self.arg_types.clone();
        combined_arg_types.extend_from_slice(&next_function.arg_types[1..]);
        
        let combined_arg_array = Array::new_with_length(combined_arg_types.len() as u32);
        for (i, arg_type) in combined_arg_types.iter().enumerate() {
            combined_arg_array.set(i as u32, arg_type.to_js_value());
        }
        
        // Create chained function
        let chained_js_func = Function::new_no_args("
            return function(...args) {
                const firstResult = this.firstFunc.call(args);
                const remainingArgs = [firstResult].concat(args.slice(this.argCount));
                return this.secondFunc.call(remainingArgs);
            }
        ").call0(&JsValue::undefined())?.dyn_into::<Function>()?;
        
        // Set properties on the chained function
        Reflect::set(&chained_js_func, &"firstFunc".into(), &self.js_function)?;
        Reflect::set(&chained_js_func, &"secondFunc".into(), &next_function.js_function)?;
        Reflect::set(&chained_js_func, &"argCount".into(), &(self.arg_types.len() as u32).into())?;
        
        StrictFunction::new(chained_js_func, combined_arg_array.into(), next_function.return_type)
    }

    // Existing getter methods
    #[wasm_bindgen(js_name = getArgTypes)]
    pub fn get_arg_types(&self) -> JsValue {
        let array = Array::new_with_length(self.arg_type_strings.len() as u32);
        for (i, &type_str) in self.arg_type_strings.iter().enumerate() {
            array.set(i as u32, JsValue::from_str(type_str));
        }
        array.into()
    }

    #[wasm_bindgen(js_name = getReturnType)]
    pub fn get_return_type(&self) -> JsValue {
        JsValue::from_str(self.return_type_string)
    }
    
    #[wasm_bindgen(js_name = getFunction)]
    pub fn get_function(&self) -> Function {
        self.js_function.clone()
    }
    
    #[wasm_bindgen(js_name = getArgCount)]
    pub fn get_arg_count(&self) -> usize {
        self.arg_types.len()
    }
    
    // Performance and utility methods
    #[wasm_bindgen(js_name = clearCache)]
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.arg_pattern_cache.lock() {
            cache.clear();
        }
    }
    
    #[wasm_bindgen(js_name = getCacheSize)]
    pub fn get_cache_size(&self) -> usize {
        self.arg_pattern_cache.lock()
            .map(|cache| cache.len())
            .unwrap_or(0)
    }
    
    // Create optimized version for specific argument patterns
    #[wasm_bindgen(js_name = optimizeForArgs)]
    pub fn optimize_for_args(&self, common_args: JsValue) -> Result<StrictFunction, JsValue> {
        let args_array: Array = common_args.dyn_into()?;
        let _ = self.process_arguments(&args_array)?; // Warm up cache
        
        // Return a clone with pre-warmed cache
        Ok(StrictFunction {
            js_function: self.js_function.clone(),
            arg_types: self.arg_types.clone(),
            return_type: self.return_type,
            arg_type_strings: self.arg_type_strings.clone(),
            return_type_string: self.return_type_string,
            arg_pattern_cache: Mutex::new(HashMap::new()), // Fresh cache
        })
    }
}

fn heap_type_to_str(heap_type: HeapType) -> &'static str {
    match heap_type {
        HeapType::Number => "number",
        HeapType::U8 => "u8",
        HeapType::I8 => "i8",
        HeapType::U16 => "u16",
        HeapType::I16 => "i16",
        HeapType::U32 => "u32",
        HeapType::I32 => "i32",
        HeapType::U64 => "u64",
        HeapType::I64 => "i64",
        HeapType::F32 => "f32",
        HeapType::F64 => "f64",
        HeapType::Bool => "bool",
        HeapType::Str => "string",
        HeapType::Str16 => "string16",
        HeapType::Any => "any",
        HeapType::Struct => "struct",
        HeapType::Array => "array",
        HeapType::Map => "map",
        HeapType::Date => "date",
        HeapType::Buffer => "buffer",
        HeapType::Null => "null",
        HeapType::Undefined => "undefined",
        HeapType::Symbol => "symbol",
    }
}