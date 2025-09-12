

// // // src/types/heap_type.rs
use js_sys::{Array, ArrayBuffer, Object, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HeapType {
   
    Number, 
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    Bool,
    Str,  
    Str16, 

    Any,    
    Struct, 
    Array,
    Map,    
    Date,   
    Buffer, 

    Null,
    Undefined,
    Symbol,
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct JsHeapType(HeapType);

#[wasm_bindgen]
impl JsHeapType {
    #[wasm_bindgen(getter)]
    pub fn element_size(&self) -> usize {
        self.0.element_size()
    }

    #[wasm_bindgen(getter, js_name = isPrimitive)]
    pub fn is_primitive(&self) -> bool {
        self.0.is_primitive()
    }

    #[wasm_bindgen(getter, js_name = isComplex)]
    pub fn is_complex(&self) -> bool {
        self.0.is_complex()
    }

    #[wasm_bindgen(getter, js_name = isNumeric)]
    pub fn is_numeric(&self) -> bool {
        self.0.is_numeric()
    }

    #[wasm_bindgen(js_name = fromString)]
    pub fn from_string(type_str: &str) -> Result<JsHeapType, JsValue> {
        HeapType::from_string(type_str).map(JsHeapType)
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }

    #[wasm_bindgen(js_name = fromJSValue)]
    pub fn from_js_value(js_value: JsValue) -> Result<JsHeapType, JsValue> {
        HeapType::from_js_value(js_value).map(JsHeapType)
    }

    #[wasm_bindgen(js_name = getDefaultValue)]
    pub fn get_default_value(&self) -> f64 {
        self.0.get_default_value()
    }

    #[wasm_bindgen(js_name = canStoreInArray)]
    pub fn can_store_in_array(&self) -> bool {
        self.0.can_store_in_array()
    }

    #[wasm_bindgen(js_name = alignment)]
    pub fn alignment(&self) -> usize {
        self.0.alignment()
    }

    #[wasm_bindgen(js_name = toJSValue)]
    pub fn to_js_value(&self) -> JsValue {
        self.0.to_js_value()
    }
}

impl HeapType {
    pub fn element_size(&self) -> usize {
        match self {
          
            HeapType::U8 | HeapType::I8 | HeapType::Bool => 1,
            HeapType::U16 | HeapType::I16 | HeapType::F32 => 2,
            HeapType::U32 | HeapType::I32 => 4,
            HeapType::U64 | HeapType::I64 | HeapType::F64 | HeapType::Number => 8,

            HeapType::Str
            | HeapType::Str16
            | HeapType::Any
            | HeapType::Struct
            | HeapType::Array
            | HeapType::Map
            | HeapType::Date
            | HeapType::Buffer => 8, 

            // Special types
            HeapType::Null | HeapType::Undefined | HeapType::Symbol => 0,
        }
    }

    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            HeapType::U8
                | HeapType::I8
                | HeapType::U16
                | HeapType::I16
                | HeapType::U32
                | HeapType::I32
                | HeapType::U64
                | HeapType::I64
                | HeapType::F32
                | HeapType::F64
                | HeapType::Number
                | HeapType::Bool
        )
    }

    pub fn is_complex(&self) -> bool {
        matches!(
            self,
            HeapType::Str
                | HeapType::Str16
                | HeapType::Struct
                | HeapType::Array
                | HeapType::Map
                | HeapType::Date
                | HeapType::Buffer
        )
    }

    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            HeapType::U8
                | HeapType::I8
                | HeapType::U16
                | HeapType::I16
                | HeapType::U32
                | HeapType::I32
                | HeapType::U64
                | HeapType::I64
                | HeapType::F32
                | HeapType::F64
                | HeapType::Number
        )
    }

    pub fn from_string(type_str: &str) -> Result<HeapType, JsValue> {
        match type_str.to_lowercase().as_str() {
            "number" => Ok(HeapType::Number),
            "u8" => Ok(HeapType::U8),
            "i8" => Ok(HeapType::I8),
            "u16" => Ok(HeapType::U16),
            "i16" => Ok(HeapType::I16),
            "u32" => Ok(HeapType::U32),
            "i32" => Ok(HeapType::I32),
            "u64" => Ok(HeapType::U64),
            "i64" => Ok(HeapType::I64),
            "f32" => Ok(HeapType::F32),
            "f64" => Ok(HeapType::F64),
            "bool" => Ok(HeapType::Bool),
            "string" => Ok(HeapType::Str),
            "string16" => Ok(HeapType::Str16),
            "any" => Ok(HeapType::Any),
            "struct" => Ok(HeapType::Struct),
            "array" => Ok(HeapType::Array),
            "map" => Ok(HeapType::Map),
            "date" => Ok(HeapType::Date),
            "buffer" => Ok(HeapType::Buffer),
            "null" => Ok(HeapType::Null),
            "undefined" => Ok(HeapType::Undefined),
            "symbol" => Ok(HeapType::Symbol),
            _ => Err(JsValue::from_str(&format!("Unknown type: {}", type_str))),
        }
    }

    pub fn to_string(&self) -> String {
        match self {
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
        .to_string()
    }

    pub fn to_js_value(&self) -> JsValue {
        match self {
            HeapType::Number => JsValue::from_str("number"),
            HeapType::U8 => JsValue::from_str("u8"),
            HeapType::I8 => JsValue::from_str("i8"),
            HeapType::U16 => JsValue::from_str("u16"),
            HeapType::I16 => JsValue::from_str("i16"),
            HeapType::U32 => JsValue::from_str("u32"),
            HeapType::I32 => JsValue::from_str("i32"),
            HeapType::U64 => JsValue::from_str("u64"),
            HeapType::I64 => JsValue::from_str("i64"),
            HeapType::F32 => JsValue::from_str("f32"),
            HeapType::F64 => JsValue::from_str("f64"),
            HeapType::Bool => JsValue::from_str("bool"),
            HeapType::Str => JsValue::from_str("string"),
            HeapType::Str16 => JsValue::from_str("string16"),
            HeapType::Any => JsValue::from_str("any"),
            HeapType::Struct => JsValue::from_str("struct"),
            HeapType::Array => JsValue::from_str("array"),
            HeapType::Map => JsValue::from_str("map"),
            HeapType::Date => JsValue::from_str("date"),
            HeapType::Buffer => JsValue::from_str("buffer"),
            HeapType::Null => JsValue::from_str("null"),
            HeapType::Undefined => JsValue::from_str("undefined"),
            HeapType::Symbol => JsValue::from_str("symbol"),
        }
    }

    pub fn from_js_value(js_value: JsValue) -> Result<HeapType, JsValue> {
        if js_value.is_null() {
            return Ok(HeapType::Null);
        }
        if js_value.is_undefined() {
            return Ok(HeapType::Undefined);
        }
        if let Some(_) = js_value.as_string() {
            return Ok(HeapType::Str);
        }
        if let Some(_) = js_value.as_f64() {
            return Ok(HeapType::Number);
        }
        if let Some(_) = js_value.as_bool() {
            return Ok(HeapType::Bool);
        }
        if js_value.is_object() {
            let obj = js_value.dyn_into::<Object>()?;
            if Array::is_array(&obj) {
                return Ok(HeapType::Array);
            }
           
            if obj.has_own_property(&JsValue::from_str("getTime")) {
                return Ok(HeapType::Date);
            }
            if obj.is_instance_of::<Uint8Array>() || obj.is_instance_of::<ArrayBuffer>() {
                return Ok(HeapType::Buffer);
            }
            return Ok(HeapType::Struct);
        }
        if js_value.is_symbol() {
            return Ok(HeapType::Symbol);
        }

        Err(JsValue::from_str("Unsupported JavaScript value type"))
    }

    pub fn get_default_value(&self) -> f64 {
        match self {
            HeapType::U8 | HeapType::U16 | HeapType::U32 | HeapType::U64 => 0.0,
            HeapType::I8 | HeapType::I16 | HeapType::I32 | HeapType::I64 => 0.0,
            HeapType::F32 | HeapType::F64 | HeapType::Number => 0.0,
            HeapType::Bool => 0.0,
            HeapType::Str
            | HeapType::Str16
            | HeapType::Any
            | HeapType::Struct
            | HeapType::Array
            | HeapType::Map
            | HeapType::Date
            | HeapType::Buffer => 0.0, 
            HeapType::Null | HeapType::Undefined | HeapType::Symbol => 0.0,
        }
    }

    pub fn can_store_in_array(&self) -> bool {
        !matches!(
            self,
            HeapType::Null | HeapType::Undefined | HeapType::Symbol
        )
    }

    pub fn alignment(&self) -> usize {
        match self {
            HeapType::U8 | HeapType::I8 | HeapType::Bool => 1,
            HeapType::U16 | HeapType::I16 | HeapType::F32 => 2,
            HeapType::U32 | HeapType::I32 => 4,
            HeapType::U64
            | HeapType::I64
            | HeapType::F64
            | HeapType::Number
            | HeapType::Str
            | HeapType::Str16
            | HeapType::Any
            | HeapType::Struct
            | HeapType::Array
            | HeapType::Map
            | HeapType::Date
            | HeapType::Buffer => 8,
            HeapType::Null | HeapType::Undefined | HeapType::Symbol => 1,
        }
    }

    pub fn from_type_str(type_str: &str) -> Option<HeapType> {
        HeapType::from_string(type_str).ok()
    }

    pub fn is_string_type(&self) -> bool {
        matches!(self, HeapType::Str | HeapType::Str16)
    }

    pub fn is_container_type(&self) -> bool {
        matches!(self, HeapType::Array | HeapType::Struct | HeapType::Map)
    }

    pub fn requires_managed_memory(&self) -> bool {
        self.is_complex() || matches!(self, HeapType::Any)
    }
}






