use wasm_bindgen_test::*;
use super::super::StrictString;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_string_truncation() {
    let s = StrictString::new("Hello World".to_string(), 5);
    assert_eq!(s.get(), "Hello");
    assert_eq!(s.len_chars(), 5);
}

#[wasm_bindgen_test]
fn test_unicode_strings() {
    let s = StrictString::new("Hello 😊 World".to_string(), 7);
    assert_eq!(s.get(), "Hello 😊"); // Should preserve emoji
    assert_eq!(s.len_chars(), 7); // H,e,l,l,o, ,😊
}

#[wasm_bindgen_test]
fn test_string_operations() {
    let mut s = StrictString::new("Hello".to_string(), 10);
    
    s.push(" World");
    assert_eq!(s.get(), "Hello World");
    assert_eq!(s.len_chars(), 11);
    
    // Test truncation on push
    s.push("!!! This should be truncated");
    assert_eq!(s.get(), "Hello Worl"); // Truncated to 10 chars
    assert_eq!(s.len_chars(), 10);
}

#[wasm_bindgen_test]
fn test_string_set() {
    let mut s = StrictString::new("Initial".to_string(), 5);
    s.set("This is too long".to_string());
    assert_eq!(s.get(), "This ");
    assert_eq!(s.len_chars(), 5);
}