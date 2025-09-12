use wasm_bindgen_test::*;
use super::super::{StrictArray, HeapType};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_array_creation() {
    let arr = StrictArray::new(HeapType::U8, 10);
    assert_eq!(arr.len(), 10);
    assert_eq!(arr.heap(), HeapType::U8);
    assert_eq!(arr.byte_len(), 10); // 10 elements * 1 byte each
}

#[wasm_bindgen_test]
fn test_array_get_set() {
    let mut arr = StrictArray::new(HeapType::U16, 3);
    
    arr.set(0, 1000.0).unwrap();
    arr.set(1, 2000.0).unwrap();
    arr.set(2, 3000.0).unwrap();
    
    assert_eq!(arr.get(0).unwrap(), 1000.0);
    assert_eq!(arr.get(1).unwrap(), 2000.0);
    assert_eq!(arr.get(2).unwrap(), 3000.0);
}

#[wasm_bindgen_test]
fn test_array_bounds_checking() {
    let mut arr = StrictArray::new(HeapType::U8, 2);
    
    // These should work
    assert!(arr.set(0, 100.0).is_ok());
    assert!(arr.set(1, 200.0).is_ok());
    
    // These should fail
    assert!(arr.set(2, 300.0).is_err()); // Out of bounds
    assert!(arr.get(2).is_err()); // Out of bounds
}

#[wasm_bindgen_test]
fn test_array_clamping() {
    let mut arr = StrictArray::new(HeapType::U8, 2);
    
    arr.set(0, 300.0).unwrap(); // Should clamp to 255
    arr.set(1, -50.0).unwrap(); // Should clamp to 0
    
    assert_eq!(arr.get(0).unwrap(), 255.0);
    assert_eq!(arr.get(1).unwrap(), 0.0);
}