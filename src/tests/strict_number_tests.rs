use wasm_bindgen_test::*;
use super::super::{StrictNumber, HeapType};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_u8_clamping() {
    let num = StrictNumber::new(300.0, HeapType::U8);
    assert_eq!(num.get(), 255.0);
    
    let num = StrictNumber::new(-10.0, HeapType::U8);
    assert_eq!(num.get(), 0.0);
}

#[wasm_bindgen_test]
fn test_i16_clamping() {
    let num = StrictNumber::new(-40000.0, HeapType::I16);
    assert_eq!(num.get(), -32768.0); // i16::MIN
    
    let num = StrictNumber::new(40000.0, HeapType::I16);
    assert_eq!(num.get(), 32767.0); // i16::MAX
}

#[wasm_bindgen_test]
fn test_arithmetic_operations() {
    let mut num = StrictNumber::new(100.0, HeapType::U8);
    num.add(50.0);
    assert_eq!(num.get(), 150.0);
    
    num.add(200.0); // Should clamp to 255
    assert_eq!(num.get(), 255.0);
    
    num.sub(300.0); // Should clamp to 0
    assert_eq!(num.get(), 0.0);
}

#[wasm_bindgen_test]
fn test_heap_type_getter() {
    let num = StrictNumber::new(100.0, HeapType::U32);
    assert_eq!(num.heap(), HeapType::U32);
}