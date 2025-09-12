
// src/strict_bigint/implementation.rs
use wasm_bindgen::prelude::*;
use js_sys::BigInt;
use crate::types::HeapType;

#[wasm_bindgen]
pub struct StrictBigInt {
    value: i64,
    heap: HeapType,
}

#[wasm_bindgen]
impl StrictBigInt {
    #[wasm_bindgen(constructor)]
    pub fn new(value: JsValue, heap: HeapType) -> Result<StrictBigInt, JsValue> {
        let num_value = if value.is_string() {
            let str_val = value.as_string().unwrap();
            str_val.parse::<i64>().map_err(|_| JsValue::from_str("Invalid number string"))?
        } else if let Some(num) = value.as_f64() {
            num as i64
        } else {
            return Err(JsValue::from_str("Value must be number or string"));
        };

        Ok(StrictBigInt {
            value: Self::clamp_value(heap, num_value),
            heap,
        })
    }

    #[wasm_bindgen(js_name = get)]
    pub fn get(&self) -> JsValue {
        BigInt::from(self.value).into()
    }

    #[wasm_bindgen(js_name = set)]
    pub fn set(&mut self, value: JsValue) -> Result<(), JsValue> {
        let num_value = if value.is_string() {
            let str_val = value.as_string().unwrap();
            str_val.parse::<i64>().map_err(|_| JsValue::from_str("Invalid number string"))?
        } else if let Some(num) = value.as_f64() {
            num as i64
        } else {
            return Err(JsValue::from_str("Value must be number or string"));
        };

        self.value = Self::clamp_value(self.heap, num_value);
        Ok(())
    }

    #[wasm_bindgen(js_name = add)]
    pub fn add(&mut self, delta: JsValue) -> Result<(), JsValue> {
        let delta_val = if delta.is_string() {
            delta.as_string().unwrap().parse::<i64>().map_err(|_| JsValue::from_str("Invalid delta"))?
        } else if let Some(num) = delta.as_f64() {
            num as i64
        } else {
            return Err(JsValue::from_str("Delta must be number or string"));
        };

        self.value = Self::clamp_value(self.heap, self.value.saturating_add(delta_val));
        Ok(())
    }

    fn clamp_value(heap: HeapType, value: i64) -> i64 {
        match heap {
            HeapType::I64 => value,
            HeapType::U64 => value.clamp(0, i64::MAX),
            _ => value, // Shouldn't happen for BigInt
        }
    }
}

