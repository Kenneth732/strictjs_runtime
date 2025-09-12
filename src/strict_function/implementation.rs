
// src/strict_function/implementation.rs
use wasm_bindgen::prelude::*;
use js_sys::{Function, Array, Object, Uint8Array, ArrayBuffer};
use crate::types::HeapType;

#[wasm_bindgen]
pub struct StrictFunction {
    js_function: Function,
    arg_types: Vec<HeapType>,
    return_type: HeapType,
    arg_type_strings: Vec<&'static str>,
    return_type_string: &'static str,
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct StrictFunctionResult {
    value: JsValue,
    result_type: String,
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
        })
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
    
    fn process_arguments(&self, args_array: &Array) -> Result<Array, JsValue> {
        let arg_count = args_array.length() as usize;
        let processed_args = Array::new_with_length(arg_count as u32);
        
        for i in 0..arg_count {
            let arg_val = args_array.get(i as u32);
            let processed_val = if (i as usize) < self.arg_types.len() {
                self.process_argument(i, arg_val)?
            } else {
                arg_val
            };
            processed_args.set(i as u32, processed_val);
        }
        
        Ok(processed_args)
    }
    
    fn process_argument(&self, index: usize, arg_val: JsValue) -> Result<JsValue, JsValue> {
        match self.arg_types[index] {
            HeapType::Number => {
                if arg_val.as_f64().is_some() {
                    Ok(arg_val)
                } else if let Some(s) = arg_val.as_string() {
                    s.parse::<f64>()
                        .map(JsValue::from_f64)
                        .map_err(|_| JsValue::from_str(&format!(
                            "Argument {}: could not convert string to number", index
                        )))
                } else {
                    Err(JsValue::from_str(&format!(
                        "Argument {} must be a number or numeric string", index
                    )))
                }
            },
            HeapType::Any => Ok(arg_val),
            HeapType::U8 | HeapType::I8 | HeapType::U16 | HeapType::I16 
            | HeapType::U32 | HeapType::I32 | HeapType::U64 | HeapType::I64 => {
                if let Some(num) = arg_val.as_f64() {
                    Ok(JsValue::from_f64(self.clamp_value(self.arg_types[index], num)))
                } else if let Some(s) = arg_val.as_string() {
                    s.parse::<f64>()
                        .map(|n| JsValue::from_f64(self.clamp_value(self.arg_types[index], n)))
                        .map_err(|_| JsValue::from_str(&format!(
                            "Argument {}: could not convert string to number", index
                        )))
                } else {
                    Err(JsValue::from_str(&format!(
                        "Argument {} must be a number or string", index
                    )))
                }
            },
            HeapType::F32 | HeapType::F64 => {
                if let Some(num) = arg_val.as_f64() {
                    Ok(JsValue::from_f64(num))
                } else if let Some(s) = arg_val.as_string() {
                    s.parse::<f64>()
                        .map(JsValue::from_f64)
                        .map_err(|_| JsValue::from_str(&format!(
                            "Argument {}: could not convert string to floating point number", index
                        )))
                } else {
                    Err(JsValue::from_str(&format!(
                        "Argument {} must be a number or numeric string", index
                    )))
                }
            },
            HeapType::Bool => {
                if let Some(b) = arg_val.as_bool() {
                    Ok(JsValue::from_bool(b))
                } else if let Some(num) = arg_val.as_f64() {
                    Ok(JsValue::from_bool(num != 0.0))
                } else if let Some(s) = arg_val.as_string() {
                    let lower = s.to_lowercase();
                    Ok(JsValue::from_bool(lower == "true" || lower == "1"))
                } else {
                    Err(JsValue::from_str(&format!(
                        "Argument {} must be a boolean, number, or string", index
                    )))
                }
            },
            HeapType::Str | HeapType::Str16 => {
                if let Some(s) = arg_val.as_string() {
                    Ok(JsValue::from_str(&s))
                } else if arg_val.is_null() || arg_val.is_undefined() {
                    Ok(JsValue::from_str(""))
                } else {
                    Ok(arg_val)
                }
            },
            HeapType::Struct | HeapType::Array | HeapType::Map | HeapType::Date | HeapType::Buffer => {
                if arg_val.is_object() {
                    Ok(arg_val)
                } else {
                    Err(JsValue::from_str(&format!(
                        "Argument {} must be an object", index
                    )))
                }
            },
            HeapType::Null => {
                if arg_val.is_null() {
                    Ok(arg_val)
                } else {
                    Err(JsValue::from_str(&format!(
                        "Argument {} must be null", index
                    )))
                }
            },
            HeapType::Undefined => {
                if arg_val.is_undefined() {
                    Ok(arg_val)
                } else {
                    Err(JsValue::from_str(&format!(
                        "Argument {} must be undefined", index
                    )))
                }
            },
            HeapType::Symbol => {
                if arg_val.is_symbol() {
                    Ok(arg_val)
                } else {
                    Err(JsValue::from_str(&format!(
                        "Argument {} must be a symbol", index
                    )))
                }
            },
        }
    }
    
    fn enhance_error(&self, error: JsValue, message: &str) -> JsValue {
        let error_string = error.as_string().unwrap_or_default();
        JsValue::from_str(&format!("{}: {}", message, error_string))
    }
    
    fn validate_and_wrap_result(&self, result: JsValue) -> Result<StrictFunctionResult, JsValue> {
        match self.return_type {
            HeapType::Number => {
                if result.as_f64().is_some() {
                    Ok(StrictFunctionResult {
                        value: result,
                        result_type: "number".to_string(),
                    })
                } else {
                    Err(JsValue::from_str("Function must return a number"))
                }
            },
            HeapType::Any => {
                if let Some(num) = result.as_f64() {
                    Ok(StrictFunctionResult {
                        value: JsValue::from_f64(num),
                        result_type: "number".to_string(),
                    })
                } else if let Some(b) = result.as_bool() {
                    Ok(StrictFunctionResult {
                        value: JsValue::from_bool(b),
                        result_type: "boolean".to_string(),
                    })
                } else if let Some(s) = result.as_string() {
                    Ok(StrictFunctionResult {
                        value: JsValue::from_str(&s),
                        result_type: "string".to_string(),
                    })
                } else if result.is_null() {
                    Ok(StrictFunctionResult {
                        value: JsValue::NULL,
                        result_type: "null".to_string(),
                    })
                } else if result.is_undefined() {
                    Ok(StrictFunctionResult {
                        value: JsValue::UNDEFINED,
                        result_type: "undefined".to_string(),
                    })
                } else if result.is_symbol() {
                    Ok(StrictFunctionResult {
                        value: result,
                        result_type: "symbol".to_string(),
                    })
                } else if result.is_object() {
                    Ok(StrictFunctionResult {
                        value: result,
                        result_type: "object".to_string(),
                    })
                } else {
                    Ok(StrictFunctionResult {
                        value: result,
                        result_type: "unknown".to_string(),
                    })
                }
            },
            HeapType::U8 | HeapType::I8 | HeapType::U16 | HeapType::I16 
            | HeapType::U32 | HeapType::I32 | HeapType::U64 | HeapType::I64 => {
                let num = result.as_f64()
                    .or_else(|| result.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                    .ok_or(JsValue::from_str("Function must return a number or boolean"))?;
                let clamped = self.clamp_value(self.return_type, num);
                Ok(StrictFunctionResult {
                    value: JsValue::from_f64(clamped),
                    result_type: "number".to_string(),
                })
            },
            HeapType::F32 | HeapType::F64 => {
                if let Some(num) = result.as_f64() {
                    Ok(StrictFunctionResult {
                        value: JsValue::from_f64(num),
                        result_type: "number".to_string(),
                    })
                } else {
                    Err(JsValue::from_str("Function must return a number"))
                }
            },
            HeapType::Bool => {
                if let Some(b) = result.as_bool() {
                    Ok(StrictFunctionResult {
                        value: JsValue::from_bool(b),
                        result_type: "boolean".to_string(),
                    })
                } else if let Some(num) = result.as_f64() {
                    let bool_val = num != 0.0;
                    Ok(StrictFunctionResult {
                        value: JsValue::from_bool(bool_val),
                        result_type: "boolean".to_string(),
                    })
                } else {
                    Err(JsValue::from_str("Function must return a boolean or number"))
                }
            },
            HeapType::Str | HeapType::Str16 => {
                if let Some(s) = result.as_string() {
                    Ok(StrictFunctionResult {
                        value: JsValue::from_str(&s),
                        result_type: "string".to_string(),
                    })
                } else {
                    Err(JsValue::from_str("Function must return a string"))
                }
            },
            HeapType::Struct | HeapType::Array | HeapType::Map | HeapType::Date | HeapType::Buffer => {
                if result.is_object() {
                    Ok(StrictFunctionResult {
                        value: result,
                        result_type: "object".to_string(),
                    })
                } else {
                    Err(JsValue::from_str("Function must return an object"))
                }
            },
            HeapType::Null => {
                if result.is_null() {
                    Ok(StrictFunctionResult {
                        value: JsValue::NULL,
                        result_type: "null".to_string(),
                    })
                } else {
                    Err(JsValue::from_str("Function must return null"))
                }
            },
            HeapType::Undefined => {
                if result.is_undefined() {
                    Ok(StrictFunctionResult {
                        value: JsValue::UNDEFINED,
                        result_type: "undefined".to_string(),
                    })
                } else {
                    Err(JsValue::from_str("Function must return undefined"))
                }
            },
            HeapType::Symbol => {
                if result.is_symbol() {
                    Ok(StrictFunctionResult {
                        value: result,
                        result_type: "symbol".to_string(),
                    })
                } else {
                    Err(JsValue::from_str("Function must return a symbol"))
                }
            },
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
        match self.arg_types[index] {
            HeapType::Number => {
                if !arg_val.as_f64().is_some() && !arg_val.is_string() {
                    return Err(JsValue::from_str(&format!(
                        "Argument {} must be a number or string", index
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
            },
            HeapType::Any => Ok(()),
            HeapType::U8 | HeapType::I8 | HeapType::U16 | HeapType::I16 
            | HeapType::U32 | HeapType::I32 | HeapType::U64 | HeapType::I64 => {
                if arg_val.as_f64().is_none() && !arg_val.is_string() {
                    return Err(JsValue::from_str(&format!(
                        "Argument {} must be a number or string", index
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
            },
            HeapType::F32 | HeapType::F64 => {
                if !arg_val.as_f64().is_some() && !arg_val.is_string() {
                    return Err(JsValue::from_str(&format!(
                        "Argument {} must be a number or string", index
                    )));
                }
                
                if arg_val.is_string() {
                    let str_val = arg_val.as_string().unwrap();
                    if str_val.parse::<f64>().is_err() {
                        return Err(JsValue::from_str(&format!(
                            "Argument {}: string cannot be converted to floating point number", index
                        )));
                    }
                }
                Ok(())
            },
            HeapType::Bool => {
                if arg_val.as_bool().is_none() && arg_val.as_f64().is_none() && !arg_val.is_string() {
                    return Err(JsValue::from_str(&format!(
                        "Argument {} must be a boolean, number, or string", index
                    )));
                }
                
                if arg_val.is_string() {
                    let str_val = arg_val.as_string().unwrap().to_lowercase();
                    if str_val != "true" && str_val != "false" && str_val != "1" && str_val != "0" {
                        return Err(JsValue::from_str(&format!(
                            "Argument {}: string must be 'true', 'false', '1', or '0'", index
                        )));
                    }
                }
                Ok(())
            },
            HeapType::Str | HeapType::Str16 => Ok(()),
            HeapType::Struct | HeapType::Array | HeapType::Map | HeapType::Date | HeapType::Buffer => {
                if !arg_val.is_object() {
                    return Err(JsValue::from_str(&format!(
                        "Argument {} must be an object", index
                    )));
                }
                Ok(())
            },
            HeapType::Null => {
                if !arg_val.is_null() {
                    return Err(JsValue::from_str(&format!(
                        "Argument {} must be null", index
                    )));
                }
                Ok(())
            },
            HeapType::Undefined => {
                if !arg_val.is_undefined() {
                    return Err(JsValue::from_str(&format!(
                        "Argument {} must be undefined", index
                    )));
                }
                Ok(())
            },
            HeapType::Symbol => {
                if !arg_val.is_symbol() {
                    return Err(JsValue::from_str(&format!(
                        "Argument {} must be a symbol", index
                    )));
                }
                Ok(())
            },
        }
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
        
        let processed_args = Array::new_with_length(self.arg_types.len() as u32);
        
        for i in 0..self.arg_types.len() {
            let arg_val = args_array.get(i as u32);
            let processed_val = self.process_argument(i, arg_val)?;
            processed_args.set(i as u32, processed_val);
        }
        
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
            HeapType::Str | HeapType::Str16 => value,
            HeapType::Any => value,
            HeapType::Struct | HeapType::Array | HeapType::Map | HeapType::Date | HeapType::Buffer => value,
            HeapType::Null | HeapType::Undefined | HeapType::Symbol => value,
        }
    }

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


