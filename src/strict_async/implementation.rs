
// // src/strict_async/implementation.rs
use wasm_bindgen::prelude::*;
use js_sys::{Promise, Function, Array};
use crate::types::HeapType;
use wasm_bindgen_futures::JsFuture;
use std::collections::VecDeque;
use std::cell::RefCell;
use std::rc::Rc;

#[wasm_bindgen]
pub struct StrictAsync {
    task_queue: VecDeque<AsyncTask>,
    max_concurrent: usize,
    running_tasks: usize,
    // Track closures for proper cleanup
    active_closures: Rc<RefCell<Vec<JsValue>>>, // Store JS references to closures
}

#[derive(Clone)]
struct AsyncTask {
    promise: Promise,
    callback: Option<Function>,
    error_handler: Option<Function>,
    heap_type: HeapType,
}

#[wasm_bindgen]
impl StrictAsync {
    #[wasm_bindgen(constructor)]
    pub fn new(max_concurrent: usize) -> StrictAsync {
        StrictAsync {
            task_queue: VecDeque::new(),
            max_concurrent,
            running_tasks: 0,
            active_closures: Rc::new(RefCell::new(Vec::new())),
        }
    }

    #[wasm_bindgen(js_name = addTask)]
    pub fn add_task(
        &mut self,
        promise: Promise,
        callback: Option<Function>,
        error_handler: Option<Function>,
        return_type: HeapType,
    ) {
        self.task_queue.push_back(AsyncTask {
            promise,
            callback,
            error_handler,
            heap_type: return_type,
        });
    }

    #[wasm_bindgen(js_name = runTasks)]
    pub async fn run_tasks(&mut self) -> Result<JsValue, JsValue> {
        let results = Array::new();
        
        while let Some(task) = self.task_queue.pop_front() {
            if self.running_tasks >= self.max_concurrent {
                self.task_queue.push_front(task);
                break;
            }
            
            self.running_tasks += 1;
            let result = self.execute_task(task).await;
            self.running_tasks -= 1;
            
            results.push(&result?);
        }
        
        // Clean up any remaining closures
        self.cleanup_closures();
        
        Ok(results.into())
    }

    async fn execute_task(&self, task: AsyncTask) -> Result<JsValue, JsValue> {
        let result = JsFuture::from(task.promise).await;
        
        match result {
            Ok(value) => {
                if let Some(callback) = task.callback {
                    let processed_value = self.process_result(value, task.heap_type)?;
                    let args = Array::new();
                    args.push(&processed_value);
                    callback.apply(&JsValue::NULL, &args)?;
                    Ok(processed_value)
                } else {
                    self.process_result(value, task.heap_type)
                }
            }
            Err(error) => {
                if let Some(error_handler) = task.error_handler {
                    let args = Array::new();
                    args.push(&error);
                    error_handler.apply(&JsValue::NULL, &args)?;
                    Ok(JsValue::NULL)
                } else {
                    Err(error)
                }
            }
        }
    }

    fn process_result(&self, value: JsValue, heap_type: HeapType) -> Result<JsValue, JsValue> {
        match heap_type {
            HeapType::U8 | HeapType::I8 | HeapType::U16 | HeapType::I16 
            | HeapType::U32 | HeapType::I32 | HeapType::Bool => {
                if let Some(num) = value.as_f64() {
                    let clamped = self.clamp_value(heap_type, num);
                    Ok(JsValue::from_f64(clamped))
                } else if value.is_truthy() {
                    match heap_type {
                        HeapType::Bool => Ok(JsValue::from_bool(true)),
                        _ => Ok(JsValue::from_f64(1.0)),
                    }
                } else {
                    match heap_type {
                        HeapType::Bool => Ok(JsValue::from_bool(false)),
                        _ => Ok(JsValue::from_f64(0.0)),
                    }
                }
            }
            HeapType::Str => {
                if let Some(str_val) = value.as_string() {
                    Ok(JsValue::from_str(&str_val))
                } else {
                    Ok(JsValue::from_str(""))
                }
            }
            _ => Ok(value),
        }
    }

    fn clamp_value(&self, heap_type: HeapType, value: f64) -> f64 {
        match heap_type {
            HeapType::U8 => value.clamp(0.0, u8::MAX as f64),
            HeapType::I8 => value.clamp(i8::MIN as f64, i8::MAX as f64),
            HeapType::U16 => value.clamp(0.0, u16::MAX as f64),
            HeapType::I16 => value.clamp(i16::MIN as f64, i16::MAX as f64),
            HeapType::U32 => value.clamp(0.0, u32::MAX as f64),
            HeapType::I32 => value.clamp(i32::MIN as f64, i32::MAX as f64),
            HeapType::Bool => if value != 0.0 { 1.0 } else { 0.0 },
            _ => value,
        }
    }

    #[wasm_bindgen(js_name = getQueueSize)]
    pub fn get_queue_size(&self) -> usize {
        self.task_queue.len()
    }

    #[wasm_bindgen(js_name = getRunningTasks)]
    pub fn get_running_tasks(&self) -> usize {
        self.running_tasks
    }

    #[wasm_bindgen(js_name = clearQueue)]
    pub fn clear_queue(&mut self) {
        self.task_queue.clear();
    }

    #[wasm_bindgen(js_name = cleanup)]
    pub fn cleanup(&self) {
        self.cleanup_closures();
    }

    fn cleanup_closures(&self) {
        self.active_closures.borrow_mut().clear();
    }
}

impl Drop for StrictAsync {
    fn drop(&mut self) {
        self.cleanup_closures();
    }
}

#[wasm_bindgen]
pub struct StrictPromise {
    promise: Promise,
    return_type: HeapType,
    // Track closures for cleanup
    closures: Rc<RefCell<Vec<JsValue>>>, // Store JS references
}

#[wasm_bindgen]
impl StrictPromise {
    #[wasm_bindgen(constructor)]
    pub fn new(executor: Function, return_type: HeapType) -> Result<StrictPromise, JsValue> {
        let closures = Rc::new(RefCell::new(Vec::new()));
        
        let promise = Promise::new(&mut |resolve, reject| {
            let args = Array::new();
            args.push(&resolve);
            args.push(&reject);
            
            if let Err(e) = executor.apply(&JsValue::NULL, &args) {
                let _ = reject.call1(&JsValue::NULL, &e);
            }
        });
        
        Ok(StrictPromise {
            promise,
            return_type,
            closures,
        })
    }

    #[wasm_bindgen(js_name = awaitValue)]
    pub async fn await_value(&self) -> Result<JsValue, JsValue> {
        let result = JsFuture::from(self.promise.clone()).await?;
        Ok(self.clamp_result(result))
    }

    fn clamp_result(&self, value: JsValue) -> JsValue {
        match self.return_type {
            HeapType::U8 | HeapType::I8 | HeapType::U16 | HeapType::I16 
            | HeapType::U32 | HeapType::I32 | HeapType::Bool => {
                if let Some(num) = value.as_f64() {
                    let clamped = match self.return_type {
                        HeapType::U8 => num.clamp(0.0, u8::MAX as f64),
                        HeapType::I8 => num.clamp(i8::MIN as f64, i8::MAX as f64),
                        HeapType::U16 => num.clamp(0.0, u16::MAX as f64),
                        HeapType::I16 => num.clamp(i16::MIN as f64, i16::MAX as f64),
                        HeapType::U32 => num.clamp(0.0, u32::MAX as f64),
                        HeapType::I32 => num.clamp(i32::MIN as f64, i32::MAX as f64),
                        HeapType::Bool => if num != 0.0 { 1.0 } else { 0.0 },
                        _ => num,
                    };
                    JsValue::from_f64(clamped)
                } else if value.is_truthy() {
                    match self.return_type {
                        HeapType::Bool => JsValue::from_bool(true),
                        _ => JsValue::from_f64(1.0),
                    }
                } else {
                    match self.return_type {
                        HeapType::Bool => JsValue::from_bool(false),
                        _ => JsValue::from_f64(0.0),
                    }
                }
            }
            _ => value,
        }
    }

    #[wasm_bindgen]
    pub fn then(&self, on_fulfilled: Function) -> Result<StrictPromise, JsValue> {
        let return_type = self.return_type;
        let closures = Rc::new(RefCell::new(Vec::new()));
        
        let closure = Closure::wrap(Box::new(move |value: JsValue| {
            let args = Array::new();
            args.push(&value);
            let _ = on_fulfilled.apply(&JsValue::NULL, &args);
        }) as Box<dyn FnMut(JsValue)>);
        
        // Store the JS value of the closure to keep it alive
        closures.borrow_mut().push(closure.as_ref().into());
        
        let new_promise = self.promise.then(&closure);
        
        // Forget the closure to prevent it from being dropped
        closure.forget();
        
        Ok(StrictPromise {
            promise: new_promise,
            return_type,
            closures,
        })
    }

    #[wasm_bindgen]
    pub fn catch(&self, on_rejected: Function) -> Result<StrictPromise, JsValue> {
        let return_type = self.return_type;
        let closures = Rc::new(RefCell::new(Vec::new()));
        
        let closure = Closure::wrap(Box::new(move |error: JsValue| {
            let args = Array::new();
            args.push(&error);
            let _ = on_rejected.apply(&JsValue::NULL, &args);
        }) as Box<dyn FnMut(JsValue)>);
        
        // Store the JS value of the closure to keep it alive
        closures.borrow_mut().push(closure.as_ref().into());
        
        let new_promise = self.promise.catch(&closure);
        
        // Forget the closure to prevent it from being dropped
        closure.forget();
        
        Ok(StrictPromise {
            promise: new_promise,
            return_type,
            closures,
        })
    }

    #[wasm_bindgen(js_name = cleanup)]
    pub fn cleanup(&self) {
        self.closures.borrow_mut().clear();
    }
}

impl Drop for StrictPromise {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[wasm_bindgen]
pub struct StrictTimeout {
    duration: f64,
    callback: Function,
    return_type: HeapType,
    timeout_id: Option<i32>,
    closure: Option<JsValue>, // Store JS reference to closure
}

#[wasm_bindgen]
impl StrictTimeout {
    #[wasm_bindgen(constructor)]
    pub fn new(duration: f64, callback: Function, return_type: HeapType) -> StrictTimeout {
        StrictTimeout {
            duration,
            callback,
            return_type,
            timeout_id: None,
            closure: None,
        }
    }

    pub async fn start(&mut self) -> Result<JsValue, JsValue> {
        // Clean up any existing timeout
        self.cancel();
        
        let promise = Promise::new(&mut |resolve, reject| {
            let callback_clone = self.callback.clone();
            let resolve_clone = resolve.clone();
            let reject_clone = reject.clone();
            
            let closure = Closure::wrap(Box::new(move || {
                match callback_clone.call0(&JsValue::NULL) {
                    Ok(value) => {
                        let _ = resolve_clone.call1(&JsValue::NULL, &value);
                    }
                    Err(e) => {
                        let _ = reject_clone.call1(&JsValue::NULL, &e);
                    }
                }
            }) as Box<dyn FnMut()>);
            
            let timeout_id = web_sys::window()
                .and_then(|w| {
                    w.set_timeout_with_callback_and_timeout_and_arguments(
                        closure.as_ref().unchecked_ref(),
                        self.duration as i32,
                        &Array::new(),
                    ).ok()
                });
            
            self.timeout_id = timeout_id;
            self.closure = Some(closure.as_ref().into());
            
            // Forget the closure to prevent it from being dropped
            closure.forget();
        });
        
        let result = JsFuture::from(promise).await?;
        Ok(self.clamp_result(result))
    }

    #[wasm_bindgen(js_name = cancel)]
    pub fn cancel(&mut self) {
        if let Some(timeout_id) = self.timeout_id.take() {
            if let Some(window) = web_sys::window() {
                let _ = window.clear_timeout_with_handle(timeout_id);
            }
        }
        self.closure = None;
    }

    fn clamp_result(&self, value: JsValue) -> JsValue {
        match self.return_type {
            HeapType::U8 | HeapType::I8 | HeapType::U16 | HeapType::I16 
            | HeapType::U32 | HeapType::I32 | HeapType::Bool => {
                if let Some(num) = value.as_f64() {
                    let clamped = match self.return_type {
                        HeapType::U8 => num.clamp(0.0, u8::MAX as f64),
                        HeapType::I8 => num.clamp(i8::MIN as f64, i8::MAX as f64),
                        HeapType::U16 => num.clamp(0.0, u16::MAX as f64),
                        HeapType::I16 => num.clamp(i16::MIN as f64, i16::MAX as f64),
                        HeapType::U32 => num.clamp(0.0, u32::MAX as f64),
                        HeapType::I32 => num.clamp(i32::MIN as f64, i32::MAX as f64),
                        HeapType::Bool => if num != 0.0 { 1.0 } else { 0.0 },
                        _ => num,
                    };
                    JsValue::from_f64(clamped)
                } else if value.is_truthy() {
                    match self.return_type {
                        HeapType::Bool => JsValue::from_bool(true),
                        _ => JsValue::from_f64(1.0),
                    }
                } else {
                    match self.return_type {
                        HeapType::Bool => JsValue::from_bool(false),
                        _ => JsValue::from_f64(0.0),
                    }
                }
            }
            _ => value,
        }
    }
}

impl Drop for StrictTimeout {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[wasm_bindgen]
pub async fn strict_fetch(url: &str, return_type: HeapType) -> Result<JsValue, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window object"))?;
    let response_promise = window.fetch_with_str(url);
    let response = JsFuture::from(response_promise).await?;
    let response: web_sys::Response = response.dyn_into()?;
    
    if response.ok() {
        let text_promise = response.text()?;
        let text = JsFuture::from(text_promise).await?;
        
        match return_type {
            HeapType::Str => Ok(text),
            _ => {
                if let Some(num_str) = text.as_string() {
                    if let Ok(num) = num_str.parse::<f64>() {
                        let clamped = match return_type {
                            HeapType::U8 => num.clamp(0.0, u8::MAX as f64),
                            HeapType::I8 => num.clamp(i8::MIN as f64, i8::MAX as f64),
                            HeapType::U16 => num.clamp(0.0, u16::MAX as f64),
                            HeapType::I16 => num.clamp(i16::MIN as f64, i16::MAX as f64),
                            HeapType::U32 => num.clamp(0.0, u32::MAX as f64),
                            HeapType::I32 => num.clamp(i32::MIN as f64, i32::MAX as f64),
                            HeapType::Bool => if num != 0.0 { 1.0 } else { 0.0 },
                            _ => num,
                        };
                        Ok(JsValue::from_f64(clamped))
                    } else {
                        Ok(JsValue::from_f64(0.0))
                    }
                } else {
                    Ok(JsValue::from_f64(0.0))
                }
            }
        }
    } else {
        Err(JsValue::from_str("HTTP error"))
    }
}




