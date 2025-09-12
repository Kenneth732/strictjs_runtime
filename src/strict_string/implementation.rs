
// src/strict_string/implementation.rs
use wasm_bindgen::prelude::*;
use js_sys::{Array, Function};

#[wasm_bindgen]
pub struct StrictString {
    value: String,
    max_chars: usize,
    encoding: StringEncoding,
}

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StringEncoding {
    Utf8,
    Utf16,
    Ascii,
}

fn truncate_to_chars(s: &str, max_chars: usize, encoding: StringEncoding) -> String {
    if max_chars == 0 {
        return String::new();
    }
    
    match encoding {
        StringEncoding::Ascii => {
            // For ASCII, we can use bytes directly
            s.chars()
                .take(max_chars)
                .filter(|c| c.is_ascii())
                .collect()
        }
        StringEncoding::Utf8 | StringEncoding::Utf16 => {
            // For Unicode, count grapheme clusters for proper character counting
            let mut result = String::new();
            let mut count = 0;
            
            for c in s.chars() {
                if count >= max_chars {
                    break;
                }
                result.push(c);
                count += 1;
            }
            result
        }
    }
}

fn validate_string(s: &str, encoding: StringEncoding) -> Result<(), String> {
    match encoding {
        StringEncoding::Ascii => {
            if s.chars().all(|c| c.is_ascii()) {
                Ok(())
            } else {
                Err("String contains non-ASCII characters".to_string())
            }
        }
        StringEncoding::Utf8 => {
            // UTF-8 is always valid in Rust strings
            Ok(())
        }
        StringEncoding::Utf16 => {
            // Rust strings are UTF-8, but we can handle UTF-16 content
            Ok(())
        }
    }
}

#[wasm_bindgen]
impl StrictString {
    #[wasm_bindgen(constructor)]
    pub fn new(val: String, max_chars: usize) -> StrictString {
        Self::new_with_encoding(val, max_chars, StringEncoding::Utf8)
    }

    #[wasm_bindgen(js_name = newWithEncoding)]
    pub fn new_with_encoding(val: String, max_chars: usize, encoding: StringEncoding) -> StrictString {
        let truncated = truncate_to_chars(&val, max_chars, encoding);
        StrictString {
            value: truncated,
            max_chars,
            encoding,
        }
    }

    pub fn get(&self) -> String {
        self.value.clone()
    }

    #[wasm_bindgen(js_name = getBytes)]
    pub fn get_bytes(&self) -> Vec<u8> {
        self.value.as_bytes().to_vec()
    }

    #[wasm_bindgen(js_name = getBytes16)]
    pub fn get_bytes_16(&self) -> Vec<u16> {
        self.value.encode_utf16().collect()
    }

    pub fn set(&mut self, val: String) -> Result<(), JsValue> {
        if let Err(e) = validate_string(&val, self.encoding) {
            return Err(JsValue::from_str(&e));
        }
        self.value = truncate_to_chars(&val, self.max_chars, self.encoding);
        Ok(())
    }

    pub fn push(&mut self, extra: &str) -> Result<(), JsValue> {
        if let Err(e) = validate_string(extra, self.encoding) {
            return Err(JsValue::from_str(&e));
        }
        
        let mut s = String::new();
        s.push_str(&self.value);
        s.push_str(extra);
        self.value = truncate_to_chars(&s, self.max_chars, self.encoding);
        Ok(())
    }

    #[wasm_bindgen(js_name = pushChar)]
    pub fn push_char(&mut self, c: char) -> Result<(), JsValue> {
        if self.encoding == StringEncoding::Ascii && !c.is_ascii() {
            return Err(JsValue::from_str("Non-ASCII character not allowed in ASCII encoding"));
        }
        
        if self.len_chars() >= self.max_chars {
            return Err(JsValue::from_str("String length exceeded"));
        }
        
        self.value.push(c);
        Ok(())
    }

    #[wasm_bindgen(js_name = popChar)]
    pub fn pop_char(&mut self) -> Option<char> {
        self.value.pop()
    }

    pub fn len_chars(&self) -> usize {
        self.value.chars().count()
    }

    #[wasm_bindgen(js_name = lenBytes)]
    pub fn len_bytes(&self) -> usize {
        self.value.len()
    }

    #[wasm_bindgen(js_name = lenBytes16)]
    pub fn len_bytes_16(&self) -> usize {
        self.value.encode_utf16().count() * 2
    }

    pub fn max_chars(&self) -> usize {
        self.max_chars
    }

    #[wasm_bindgen(js_name = getEncoding)]
    pub fn get_encoding(&self) -> StringEncoding {
        self.encoding
    }

    #[wasm_bindgen(js_name = setEncoding)]
    pub fn set_encoding(&mut self, encoding: StringEncoding) -> Result<(), JsValue> {
        if let Err(e) = validate_string(&self.value, encoding) {
            return Err(JsValue::from_str(&e));
        }
        self.encoding = encoding;
        Ok(())
    }

    #[wasm_bindgen(js_name = substring)]
    pub fn substring(&self, start: usize, end: usize) -> Result<StrictString, JsValue> {
        if start > end || end > self.len_chars() {
            return Err(JsValue::from_str("Invalid substring range"));
        }
        
        let substring: String = self.value.chars()
            .skip(start)
            .take(end - start)
            .collect();
            
        Ok(StrictString {
            value: substring,
            max_chars: self.max_chars,
            encoding: self.encoding,
        })
    }

    #[wasm_bindgen(js_name = charAt)]
    pub fn char_at(&self, index: usize) -> Option<String> {
        self.value.chars()
            .nth(index)
            .map(|c| c.to_string())
    }

    #[wasm_bindgen(js_name = indexOf)]
    pub fn index_of(&self, search: &str) -> isize {
        self.value.find(search)
            .map(|pos| pos as isize)
            .unwrap_or(-1)
    }

    #[wasm_bindgen(js_name = lastIndexOf)]
    pub fn last_index_of(&self, search: &str) -> isize {
        self.value.rfind(search)
            .map(|pos| pos as isize)
            .unwrap_or(-1)
    }

    #[wasm_bindgen(js_name = startsWith)]
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.value.starts_with(prefix)
    }

    #[wasm_bindgen(js_name = endsWith)]
    pub fn ends_with(&self, suffix: &str) -> bool {
        self.value.ends_with(suffix)
    }

    #[wasm_bindgen(js_name = contains)]
    pub fn contains(&self, search: &str) -> bool {
        self.value.contains(search)
    }

    #[wasm_bindgen(js_name = replace)]
    pub fn replace(&mut self, old: &str, new: &str) -> Result<(), JsValue> {
        if let Err(e) = validate_string(new, self.encoding) {
            return Err(JsValue::from_str(&e));
        }
        
        let replaced = self.value.replace(old, new);
        self.value = truncate_to_chars(&replaced, self.max_chars, self.encoding);
        Ok(())
    }

    #[wasm_bindgen(js_name = toLowerCase)]
    pub fn to_lower_case(&mut self) {
        self.value = self.value.to_lowercase();
    }

    #[wasm_bindgen(js_name = toUpperCase)]
    pub fn to_upper_case(&mut self) {
        self.value = self.value.to_uppercase();
    }

    #[wasm_bindgen(js_name = trim)]
    pub fn trim(&mut self) {
        self.value = self.value.trim().to_string();
    }

    #[wasm_bindgen(js_name = trimStart)]
    pub fn trim_start(&mut self) {
        self.value = self.value.trim_start().to_string();
    }

    #[wasm_bindgen(js_name = trimEnd)]
    pub fn trim_end(&mut self) {
        self.value = self.value.trim_end().to_string();
    }

    #[wasm_bindgen(js_name = padStart)]
    pub fn pad_start(&mut self, target_length: usize, pad_char: &str) -> Result<(), JsValue> {
        if pad_char.chars().count() != 1 {
            return Err(JsValue::from_str("Pad character must be a single character"));
        }
        
        if let Err(e) = validate_string(pad_char, self.encoding) {
            return Err(JsValue::from_str(&e));
        }
        
        if self.len_chars() >= target_length {
            return Ok(());
        }
        
        let pad_count = target_length - self.len_chars();
        let padding: String = std::iter::repeat(pad_char)
            .take(pad_count)
            .collect();
            
        self.value = format!("{}{}", padding, self.value);
        self.value = truncate_to_chars(&self.value, self.max_chars, self.encoding);
        Ok(())
    }

    #[wasm_bindgen(js_name = padEnd)]
    pub fn pad_end(&mut self, target_length: usize, pad_char: &str) -> Result<(), JsValue> {
        if pad_char.chars().count() != 1 {
            return Err(JsValue::from_str("Pad character must be a single character"));
        }
        
        if let Err(e) = validate_string(pad_char, self.encoding) {
            return Err(JsValue::from_str(&e));
        }
        
        if self.len_chars() >= target_length {
            return Ok(());
        }
        
        let pad_count = target_length - self.len_chars();
        let padding: String = std::iter::repeat(pad_char)
            .take(pad_count)
            .collect();
            
        self.value = format!("{}{}", self.value, padding);
        self.value = truncate_to_chars(&self.value, self.max_chars, self.encoding);
        Ok(())
    }

    #[wasm_bindgen(js_name = split)]
    pub fn split(&self, delimiter: &str) -> Array {
        let parts: Array = self.value.split(delimiter)
            .map(|part| JsValue::from_str(part))
            .collect();
        parts
    }

    #[wasm_bindgen(js_name = map)]
    pub fn map(&self, js_function: JsValue) -> Result<StrictString, JsValue> {
        let function = js_function.dyn_into::<Function>()
            .map_err(|_| JsValue::from_str("Argument must be a function"))?;

        let mut result = String::new();
        
        for (i, c) in self.value.chars().enumerate() {
            let args = Array::new();
            args.push(&JsValue::from_str(&c.to_string()));
            args.push(&JsValue::from_f64(i as f64));
            
            let mapped_char = function.apply(&JsValue::NULL, &args)?
                .as_string()
                .ok_or_else(|| JsValue::from_str("Map function must return a string"))?;
                
            result.push_str(&mapped_char);
        }
        
        Ok(StrictString {
            value: truncate_to_chars(&result, self.max_chars, self.encoding),
            max_chars: self.max_chars,
            encoding: self.encoding,
        })
    }

    #[wasm_bindgen(js_name = filter)]
    pub fn filter(&self, js_function: JsValue) -> Result<StrictString, JsValue> {
        let function = js_function.dyn_into::<Function>()
            .map_err(|_| JsValue::from_str("Argument must be a function"))?;

        let mut result = String::new();
        
        for (i, c) in self.value.chars().enumerate() {
            let args = Array::new();
            args.push(&JsValue::from_str(&c.to_string()));
            args.push(&JsValue::from_f64(i as f64));
            
            let should_keep = function.apply(&JsValue::NULL, &args)?
                .as_bool()
                .unwrap_or(false);
                
            if should_keep {
                result.push(c);
            }
        }
        
        Ok(StrictString {
            value: truncate_to_chars(&result, self.max_chars, self.encoding),
            max_chars: self.max_chars,
            encoding: self.encoding,
        })
    }
}

