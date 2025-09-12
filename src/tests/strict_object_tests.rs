use wasm_bindgen_test::*;
use super::super::{StrictObject, HeapType};
use wasm_bindgen::JsValue;
use js_sys::Object;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_object_creation_with_schema() {
    let schema = Object::new();
    js_sys::Reflect::set(&schema, &"age".into(), &"u8".into()).unwrap();
    js_sys::Reflect::set(&schema, &"score".into(), &"i16".into()).unwrap();
    
    let obj = StrictObject::new(schema.into()).unwrap();
    assert_eq!(obj.get_field("age").unwrap(), 0.0);
    assert_eq!(obj.get_field("score").unwrap(), 0.0);
}

#[wasm_bindgen_test]
fn test_object_clamping() {
    let schema = Object::new();
    js_sys::Reflect::set(&schema, &"age".into(), &"u8".into()).unwrap();
    
    let mut obj = StrictObject::new(schema.into()).unwrap();
    obj.set_field("age", 300.0).unwrap(); // Should clamp to 255
    assert_eq!(obj.get_field("age").unwrap(), 255.0);
    
    obj.set_field("age", -10.0).unwrap(); // Should clamp to 0
    assert_eq!(obj.get_field("age").unwrap(), 0.0);
}

#[wasm_bindgen_test]
fn test_object_error_handling() {
    let schema = Object::new();
    js_sys::Reflect::set(&schema, &"age".into(), &"u8".into()).unwrap();
    
    let mut obj = StrictObject::new(schema.into()).unwrap();
    
    // Setting unknown field should error
    assert!(obj.set_field("unknown", 100.0).is_err());
    
    // Getting unknown field should error
    assert!(obj.get_field("unknown").is_err());
}

#[wasm_bindgen_test]
fn test_get_schema() {
    let schema = Object::new();
    js_sys::Reflect::set(&schema, &"age".into(), &"u8".into()).unwrap();
    js_sys::Reflect::set(&schema, &"score".into(), &"i16".into()).unwrap();
    
    let obj = StrictObject::new(schema.into()).unwrap();
    let returned_schema = obj.get_schema();
    
    // Verify the returned schema contains the expected fields
    assert!(js_sys::Reflect::has(&returned_schema, &"age".into()).unwrap());
    assert!(js_sys::Reflect::has(&returned_schema, &"score".into()).unwrap());
}