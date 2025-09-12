
// // src/reactive_system/implementation.rs
use crate::types::HeapType;
use crate::StrictFunction;
use js_sys::Array;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct ReactiveSystem {
    cells: HashMap<String, ReactiveCell>,
    computations: HashMap<String, Computed>,
    dependencies: HashMap<String, Vec<String>>,
}

#[wasm_bindgen]
pub struct ReactiveCell {
    value: f64,
    heap_type: HeapType,
    #[allow(dead_code)]
    subscribers: Vec<String>,
}

#[wasm_bindgen]
pub struct Computed {
    computation: StrictFunction,
    dependencies: Vec<String>,
    last_value: f64,
}

#[wasm_bindgen]
impl ReactiveSystem {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ReactiveSystem {
        ReactiveSystem {
            cells: HashMap::new(),
            computations: HashMap::new(),
            dependencies: HashMap::new(),
        }
    }

    #[wasm_bindgen(js_name = defineCell)]
    pub fn define_cell(
        &mut self,
        name: &str,
        initial_value: f64,
        heap_type: HeapType,
    ) -> Result<(), JsValue> {
        let clamped_value = self.clamp_value(heap_type, initial_value);

        self.cells.insert(
            name.to_string(),
            ReactiveCell {
                value: clamped_value,
                heap_type,
                subscribers: Vec::new(),
            },
        );

        Ok(())
    }

    #[wasm_bindgen(js_name = setCell)]
    pub fn set_cell(&mut self, name: &str, new_value: f64) -> Result<(), JsValue> {
        let heap_type = {
            let cell = self
                .cells
                .get(name)
                .ok_or_else(|| JsValue::from_str(&format!("Cell '{}' not found", name)))?;
            cell.heap_type
        };

        let clamped_value = self.clamp_value(heap_type, new_value);

        let cell = self
            .cells
            .get_mut(name)
            .ok_or_else(|| JsValue::from_str(&format!("Cell '{}' not found", name)))?;
        cell.value = clamped_value;

        self.notify_subscribers(name)?;

        Ok(())
    }

    #[wasm_bindgen(js_name = getCell)]
    pub fn get_cell(&self, name: &str) -> Result<f64, JsValue> {
        self.cells
            .get(name)
            .map(|cell| cell.value)
            .ok_or_else(|| JsValue::from_str(&format!("Cell '{}' not found", name)))
    }

    #[wasm_bindgen(js_name = defineComputed)]
    pub fn define_computed(
        &mut self,
        name: &str,
        computation: StrictFunction,
        dependencies: JsValue,
    ) -> Result<(), JsValue> {
        let deps_array: Array = dependencies.dyn_into()?;
        let mut deps_vec = Vec::new();

        for i in 0..deps_array.length() {
            let dep_name = deps_array
                .get(i)
                .as_string()
                .ok_or_else(|| JsValue::from_str("Dependency names must be strings"))?;
            deps_vec.push(dep_name);
        }

        let args = self.get_dependency_values(&deps_vec)?;
        let initial_value = computation.call(args)?;

        for dep in &deps_vec {
            self.add_subscriber(dep, name.to_string());
        }

        self.computations.insert(
            name.to_string(),
            Computed {
                computation,
                dependencies: deps_vec,
                last_value: initial_value,
            },
        );

        Ok(())
    }

    #[wasm_bindgen(js_name = getComputed)]
    pub fn get_computed(&self, name: &str) -> Result<f64, JsValue> {
        self.computations
            .get(name)
            .map(|comp| comp.last_value)
            .ok_or_else(|| JsValue::from_str(&format!("Computed '{}' not found", name)))
    }

    fn notify_subscribers(&mut self, cell_name: &str) -> Result<(), JsValue> {
        let subscribers: Vec<String> = self
            .dependencies
            .get(cell_name)
            .map(|subs| subs.clone())
            .unwrap_or_default();

        for subscriber in subscribers {
            let dependencies: Vec<String> = self
                .computations
                .get(&subscriber)
                .map(|comp| comp.dependencies.clone())
                .ok_or_else(|| {
                    JsValue::from_str(&format!("Computed '{}' not found", subscriber))
                })?;

            let args = self.get_dependency_values(&dependencies)?;

            if let Some(computed) = self.computations.get_mut(&subscriber) {
                computed.last_value = computed.computation.call(args)?;
                self.notify_subscribers(&subscriber)?;
            }
        }
        Ok(())
    }

    fn get_dependency_values(&self, dependencies: &[String]) -> Result<JsValue, JsValue> {
        let args = Array::new();
        for dep in dependencies {
            if let Some(cell) = self.cells.get(dep) {
                args.push(&JsValue::from_f64(cell.value));
            } else if let Some(computed) = self.computations.get(dep) {
                args.push(&JsValue::from_f64(computed.last_value));
            } else {
                return Err(JsValue::from_str(&format!(
                    "Dependency '{}' not found",
                    dep
                )));
            }
        }
        Ok(args.into())
    }

    fn add_subscriber(&mut self, dependency: &str, subscriber: String) {
        self.dependencies
            .entry(dependency.to_string())
            .or_insert_with(Vec::new)
            .push(subscriber);
    }

    fn clamp_value(&self, heap_type: HeapType, value: f64) -> f64 {
        match heap_type {
            HeapType::U8 => value.clamp(0.0, u8::MAX as f64),
            HeapType::I8 => value.clamp(i8::MIN as f64, i8::MAX as f64),
            HeapType::U16 => value.clamp(0.0, u16::MAX as f64),
            HeapType::I16 => value.clamp(i16::MIN as f64, i16::MAX as f64),
            HeapType::U32 => value.clamp(0.0, u32::MAX as f64),
            HeapType::I32 => value.clamp(i32::MIN as f64, i32::MAX as f64),
            HeapType::U64 => value.clamp(0.0, u64::MAX as f64),
            HeapType::I64 => value.clamp(i64::MIN as f64, i64::MAX as f64),
            HeapType::Bool => {
                if value != 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            _ => value, // No clamping for all other types
        }
    }

    #[wasm_bindgen(js_name = getSystemState)]
    pub fn get_system_state(&self) -> JsValue {
        let state = js_sys::Object::new();

        let cells_obj = js_sys::Object::new();
        for (name, cell) in &self.cells {
            js_sys::Reflect::set(
                &cells_obj,
                &JsValue::from_str(name),
                &JsValue::from_f64(cell.value),
            )
            .unwrap();
        }
        js_sys::Reflect::set(&state, &JsValue::from_str("cells"), &cells_obj).unwrap();

        let computed_obj = js_sys::Object::new();
        for (name, computed) in &self.computations {
            js_sys::Reflect::set(
                &computed_obj,
                &JsValue::from_str(name),
                &JsValue::from_f64(computed.last_value),
            )
            .unwrap();
        }
        js_sys::Reflect::set(&state, &JsValue::from_str("computed"), &computed_obj).unwrap();

        state.into()
    }
}


