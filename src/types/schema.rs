
// // src/types/schema.rs
use wasm_bindgen::prelude::*;
use js_sys::Object;
use std::collections::HashMap;

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Schema {
    #[wasm_bindgen(skip)]
    pub fields: HashMap<String, FieldType>,
    #[wasm_bindgen(skip)]
    pub nested_schemas: HashMap<String, Schema>,
}

#[derive(Clone, Debug)]
pub enum FieldType {
    Simple(String),
    Array(Box<FieldType>),
    Nested,
}

#[wasm_bindgen]
impl Schema {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Schema {
        Schema {
            fields: HashMap::new(),
            nested_schemas: HashMap::new(),
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

    #[wasm_bindgen(js_name = getFieldType)]
    pub fn get_field_type(&self, field: &str) -> Option<String> {
        self.fields.get(field).and_then(|ft| match ft {
            FieldType::Simple(t) => Some(t.clone()),
            FieldType::Array(inner) => match **inner {
                FieldType::Simple(ref t) => Some(format!("Array<{}>", t)),
                FieldType::Nested => Some("Array<Nested>".to_string()),
                _ => None,
            },
            FieldType::Nested => Some("Nested".to_string()),
        })
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
        
        for (key, field_type) in &self.fields {
            let type_str = match field_type {
                FieldType::Simple(t) => t.clone(),
                FieldType::Array(inner) => {
                    let inner_str = match **inner {
                        FieldType::Simple(ref t) => t.clone(),
                        FieldType::Nested => "Nested".to_string(),
                        _ => "Unknown".to_string(),
                    };
                    format!("Array<{}>", inner_str)
                }
                FieldType::Nested => "Nested".to_string(),
            };
            js_sys::Reflect::set(&obj, &JsValue::from_str(key), &JsValue::from_str(&type_str)).unwrap();
        }
        
        for (key, schema) in &self.nested_schemas {
            js_sys::Reflect::set(&obj, &JsValue::from_str(key), &schema.to_js()).unwrap();
        }
        
        obj.into()
    }

    pub fn from_js_object(js_obj: &Object) -> Result<Schema, JsValue> {
        let mut schema = Schema::new();
        let keys = Object::keys(js_obj);
        
        for i in 0..keys.length() {
            let key = keys.get(i);
            let key_str = key.as_string().ok_or_else(|| JsValue::from_str("Invalid key"))?;
            
            let value = js_sys::Reflect::get(js_obj, &key)?;
            
            if value.is_string() {
                let type_str = value.as_string().unwrap();
                if type_str.starts_with("Array<") && type_str.ends_with(">") {
                    let element_type = type_str.trim_start_matches("Array<").trim_end_matches(">");
                    schema.add_array_field(&key_str, element_type);
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
}



