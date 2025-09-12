

// src/threads/task.rs
use wasm_bindgen::prelude::*;
use js_sys::{Function, Promise};
use crate::threads::types::{ThreadPriority, ThreadState};
use crate::types::HeapType;

#[wasm_bindgen]
#[derive(Clone)]
pub struct ThreadTask {
    id: String,
    function: Function,
    args: Vec<JsValue>,
    #[allow(dead_code)] 
    result_type: HeapType,
    state: ThreadState,
    priority: ThreadPriority,
}


#[wasm_bindgen]
impl ThreadTask {
    #[wasm_bindgen(constructor)]
    pub fn new(
        function: Function,
        args: JsValue,
        result_type: HeapType,
        priority: ThreadPriority
    ) -> Result<ThreadTask, JsValue> {
        let args_array: js_sys::Array = args.dyn_into()?;
        let mut args_vec = Vec::with_capacity(args_array.length() as usize);
        
        for i in 0..args_array.length() {
            args_vec.push(args_array.get(i));
        }
        
        Ok(ThreadTask {
            id: uuid::Uuid::new_v4().to_string(),
            function,
            args: args_vec,
            result_type,
            state: ThreadState::Idle,
            priority,
        })
    }
    
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn state(&self) -> ThreadState {
        self.state
    }
    
    #[wasm_bindgen(getter)]
    pub fn priority(&self) -> ThreadPriority {
        self.priority
    }
    
    pub async fn execute(&mut self) -> Result<JsValue, JsValue> {
        self.state = ThreadState::Running;
        
        let result = self.function.apply(&JsValue::NULL, &js_sys::Array::from_iter(self.args.iter()))
            .map_err(|e| {
                self.state = ThreadState::Error;
                e
            })?;
            
        self.state = ThreadState::Completed;
        Ok(result)
    }
    
    pub fn to_promise(&self) -> Promise {
        let function = self.function.clone();
        let args = js_sys::Array::from_iter(self.args.iter().cloned());
        
        Promise::new(&mut |resolve, reject| {
            let result = function.apply(&JsValue::NULL, &args);
            match result {
                Ok(value) => {
                    let _ = resolve.call1(&JsValue::NULL, &value);
                },
                Err(error) => {
                    let _ = reject.call1(&JsValue::NULL, &error);
                },
            }
        })
    }
}

