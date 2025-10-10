

// src/strict_async/implementation.rs
use wasm_bindgen::prelude::*;
use js_sys::{Promise, Function, Array};
use crate::types::HeapType;
use wasm_bindgen_futures::JsFuture;
use std::collections::{HashSet, BinaryHeap};
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use futures::future::join_all;
use web_sys;

#[wasm_bindgen]
pub struct StrictAsync {
    task_queue: BinaryHeap<AsyncTask>, // Changed from VecDeque to BinaryHeap for priority
    max_concurrent: usize,
    running_tasks: AtomicUsize,
    active_closures: RefCell<HashSet<JsValue>>,
    last_error: RefCell<Option<String>>,
    next_task_id: AtomicUsize,
}

struct AsyncTask {
    promise: Promise,
    callback: Option<Function>,
    error_handler: Option<Function>,
    heap_type: HeapType,
    task_id: usize,
    priority: TaskPriority,
}

// Implement manual comparison for AsyncTask (since Promise doesn't implement PartialEq)
impl PartialEq for AsyncTask {
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
    }
}

impl Eq for AsyncTask {}

impl Ord for AsyncTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.priority.cmp(&self.priority) // Reverse for max-heap behavior
            .then_with(|| self.task_id.cmp(&other.task_id))
    }
}

impl PartialOrd for AsyncTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for TaskPriority {
    fn default() -> Self {
        TaskPriority::Normal
    }
}

#[wasm_bindgen]
impl StrictAsync {
    #[wasm_bindgen(constructor)]
    pub fn new(max_concurrent: usize) -> StrictAsync {
        StrictAsync {
            task_queue: BinaryHeap::new(),
            max_concurrent,
            running_tasks: AtomicUsize::new(0),
            active_closures: RefCell::new(HashSet::new()),
            last_error: RefCell::new(None),
            next_task_id: AtomicUsize::new(1),
        }
    }

    #[wasm_bindgen(js_name = addTask)]
    pub fn add_task(
        &mut self,
        promise: Promise,
        callback: Option<Function>,
        error_handler: Option<Function>,
        return_type: HeapType,
    ) -> usize {
        self.add_task_with_priority(promise, callback, error_handler, return_type, TaskPriority::Normal)
    }

    #[wasm_bindgen(js_name = addTaskWithPriority)]
    pub fn add_task_with_priority(
        &mut self,
        promise: Promise,
        callback: Option<Function>,
        error_handler: Option<Function>,
        return_type: HeapType,
        priority: TaskPriority,
    ) -> usize {
        let task_id = self.next_task_id.fetch_add(1, Ordering::SeqCst);
        let task = AsyncTask {
            promise,
            callback,
            error_handler,
            heap_type: return_type,
            task_id,
            priority,
        };
        
        self.task_queue.push(task);
        task_id
    }

    #[wasm_bindgen(js_name = runTasks)]
    pub async fn run_tasks(&mut self) -> Result<JsValue, JsValue> {
        let results = Array::new();
        let mut tasks_to_run = Vec::new();
        
        // Collect tasks to run concurrently based on priority
        while let Some(task) = self.task_queue.pop() {
            if self.running_tasks.load(Ordering::SeqCst) + tasks_to_run.len() >= self.max_concurrent {
                // Put it back if we can't run it now
                self.task_queue.push(task);
                break;
            }
            tasks_to_run.push(task);
        }
        
        // Execute tasks concurrently
        let task_futures: Vec<_> = tasks_to_run.into_iter().map(|task| {
            self.execute_task_concurrent(task)
        }).collect();
        
        let task_results = join_all(task_futures).await;
        
        for result in task_results {
            match result {
                Ok(value) => {
                    let _ = results.push(&value);
                }
                Err(e) => {
                    *self.last_error.borrow_mut() = Some(format!("Task error: {:?}", e));
                    let _ = results.push(&JsValue::NULL);
                }
            }
        }
        
        self.cleanup_closures();
        Ok(results.into())
    }

    async fn execute_task_concurrent(&self, task: AsyncTask) -> Result<JsValue, JsValue> {
        self.running_tasks.fetch_add(1, Ordering::SeqCst);
        let result = self.execute_task(task).await;
        self.running_tasks.fetch_sub(1, Ordering::SeqCst);
        result
    }

    async fn execute_task(&self, task: AsyncTask) -> Result<JsValue, JsValue> {
        let result = JsFuture::from(task.promise).await;
        
        match result {
            Ok(value) => {
                let processed_value = self.process_result(value, task.heap_type)?;
                if let Some(callback) = task.callback {
                    let args = Array::new();
                    args.push(&processed_value);
                    callback.apply(&JsValue::NULL, &args)?;
                }
                Ok(processed_value)
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
            | HeapType::U32 | HeapType::I32 | HeapType::U64 | HeapType::I64
            | HeapType::F32 | HeapType::F64 | HeapType::Number | HeapType::Bool => {
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
            HeapType::Str | HeapType::Str16 => {
                if let Some(str_val) = value.as_string() {
                    Ok(JsValue::from_str(&str_val))
                } else {
                    Ok(JsValue::from_str(""))
                }
            }
            HeapType::Array => {
                if value.is_array() {
                    Ok(value)
                } else {
                    Ok(Array::new().into())
                }
            }
            HeapType::Buffer => {
                if value.is_instance_of::<js_sys::Uint8Array>() 
                    || value.is_instance_of::<js_sys::ArrayBuffer>() {
                    Ok(value)
                } else {
                    Ok(js_sys::Uint8Array::new_with_length(0).into())
                }
            }
            HeapType::Date => {
                if value.is_instance_of::<js_sys::Date>() {
                    Ok(value)
                } else {
                    Ok(js_sys::Date::new_0().into())
                }
            }
            HeapType::Null => Ok(JsValue::NULL),
            HeapType::Undefined => Ok(JsValue::UNDEFINED),
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
            HeapType::U64 => value.clamp(0.0, u64::MAX as f64),
            HeapType::I64 => value.clamp(i64::MIN as f64, i64::MAX as f64),
            HeapType::F32 => value as f32 as f64,
            HeapType::F64 => value,
            HeapType::Number => value,
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
        self.running_tasks.load(Ordering::SeqCst)
    }

    #[wasm_bindgen(js_name = clearQueue)]
    pub fn clear_queue(&mut self) {
        self.task_queue.clear();
    }

    #[wasm_bindgen(js_name = cancelTask)]
    pub fn cancel_task(&mut self, task_id: usize) -> bool {
        // Convert BinaryHeap to Vec to search for the task
        let mut tasks: Vec<AsyncTask> = self.task_queue.drain().collect();
        let mut found = false;
        
        // Remove the task with matching ID
        tasks.retain(|task| {
            if task.task_id == task_id {
                found = true;
                false
            } else {
                true
            }
        });
        
        // Rebuild the BinaryHeap
        for task in tasks {
            self.task_queue.push(task);
        }
        
        found
    }

    #[wasm_bindgen(js_name = setMaxConcurrent)]
    pub fn set_max_concurrent(&mut self, max: usize) {
        self.max_concurrent = max;
    }

    #[wasm_bindgen(js_name = getLastError)]
    pub fn get_last_error(&self) -> Option<String> {
        self.last_error.borrow().clone()
    }

    #[wasm_bindgen(js_name = cleanup)]
    pub fn cleanup(&self) {
        self.cleanup_closures();
        *self.last_error.borrow_mut() = None;
    }

    #[wasm_bindgen(js_name = getNextTaskId)]
    pub fn get_next_task_id(&self) -> usize {
        self.next_task_id.load(Ordering::SeqCst)
    }

    fn cleanup_closures(&self) {
        let mut closures = self.active_closures.borrow_mut();
        closures.clear();
    }
}

impl Drop for StrictAsync {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[wasm_bindgen]
pub struct StrictPromise {
    promise: Promise,
    return_type: HeapType,
    closures: RefCell<Vec<JsValue>>,
}

#[wasm_bindgen]
impl StrictPromise {
    #[wasm_bindgen(constructor)]
    pub fn new(executor: Function, return_type: HeapType) -> Result<StrictPromise, JsValue> {
        let closures = RefCell::new(Vec::new());
        
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
        Ok(self.process_result(result))
    }

    fn process_result(&self, value: JsValue) -> JsValue {
        match self.return_type {
            HeapType::U8 | HeapType::I8 | HeapType::U16 | HeapType::I16 
            | HeapType::U32 | HeapType::I32 | HeapType::U64 | HeapType::I64
            | HeapType::F32 | HeapType::F64 | HeapType::Number | HeapType::Bool => {
                if let Some(num) = value.as_f64() {
                    let clamped = self.clamp_value(num);
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

    fn clamp_value(&self, value: f64) -> f64 {
        match self.return_type {
            HeapType::U8 => value.clamp(0.0, u8::MAX as f64),
            HeapType::I8 => value.clamp(i8::MIN as f64, i8::MAX as f64),
            HeapType::U16 => value.clamp(0.0, u16::MAX as f64),
            HeapType::I16 => value.clamp(i16::MIN as f64, i16::MAX as f64),
            HeapType::U32 => value.clamp(0.0, u32::MAX as f64),
            HeapType::I32 => value.clamp(i32::MIN as f64, i32::MAX as f64),
            HeapType::U64 => value.clamp(0.0, u64::MAX as f64),
            HeapType::I64 => value.clamp(i64::MIN as f64, i64::MAX as f64),
            HeapType::F32 => value as f32 as f64,
            HeapType::F64 => value,
            HeapType::Number => value,
            HeapType::Bool => if value != 0.0 { 1.0 } else { 0.0 },
            _ => value,
        }
    }

    #[wasm_bindgen]
    pub fn then(&self, on_fulfilled: Function) -> Result<StrictPromise, JsValue> {
        use wasm_bindgen::closure::Closure;
        
        let return_type = self.return_type;
        
        let closure = Closure::once(move |value: JsValue| {
            let args = Array::new();
            args.push(&value);
            let _ = on_fulfilled.apply(&JsValue::NULL, &args);
        });
        
        let new_promise = self.promise.then(&closure);
        
        // Store the closure for cleanup
        self.closures.borrow_mut().push(closure.into_js_value());
        
        Ok(StrictPromise {
            promise: new_promise,
            return_type,
            closures: self.closures.clone(),
        })
    }

    #[wasm_bindgen]
    pub fn catch(&self, on_rejected: Function) -> Result<StrictPromise, JsValue> {
        use wasm_bindgen::closure::Closure;
        
        let return_type = self.return_type;
        
        let closure = Closure::once(move |error: JsValue| {
            let args = Array::new();
            args.push(&error);
            let _ = on_rejected.apply(&JsValue::NULL, &args);
        });
        
        let new_promise = self.promise.catch(&closure);
        
        // Store the closure for cleanup
        self.closures.borrow_mut().push(closure.into_js_value());
        
        Ok(StrictPromise {
            promise: new_promise,
            return_type,
            closures: self.closures.clone(),
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
    closure: Option<JsValue>,
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
        self.cancel();
        
        let promise = Promise::new(&mut |resolve, _reject| {
            let callback_clone = self.callback.clone();
            let resolve_clone = resolve.clone();
            
            let closure = Closure::once(move || {
                match callback_clone.call0(&JsValue::NULL) {
                    Ok(value) => {
                        let _ = resolve_clone.call1(&JsValue::NULL, &value);
                    }
                    Err(_e) => {
                        // Still resolve but with error indicator
                        let _ = resolve_clone.call1(&JsValue::NULL, &JsValue::NULL);
                    }
                }
            });
            
            self.timeout_id = web_sys::window()
                .and_then(|w| {
                    w.set_timeout_with_callback_and_timeout_and_arguments(
                        closure.as_ref().unchecked_ref(),
                        self.duration as i32,
                        &Array::new(),
                    ).ok()
                });
            
            self.closure = Some(closure.into_js_value());
        });
        
        let result = JsFuture::from(promise).await?;
        Ok(self.process_result(result))
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

    fn process_result(&self, value: JsValue) -> JsValue {
        match self.return_type {
            HeapType::U8 | HeapType::I8 | HeapType::U16 | HeapType::I16 
            | HeapType::U32 | HeapType::I32 | HeapType::U64 | HeapType::I64
            | HeapType::F32 | HeapType::F64 | HeapType::Number | HeapType::Bool => {
                if let Some(num) = value.as_f64() {
                    let clamped = self.clamp_value(num);
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

    fn clamp_value(&self, value: f64) -> f64 {
        match self.return_type {
            HeapType::U8 => value.clamp(0.0, u8::MAX as f64),
            HeapType::I8 => value.clamp(i8::MIN as f64, i8::MAX as f64),
            HeapType::U16 => value.clamp(0.0, u16::MAX as f64),
            HeapType::I16 => value.clamp(i16::MIN as f64, i16::MAX as f64),
            HeapType::U32 => value.clamp(0.0, u32::MAX as f64),
            HeapType::I32 => value.clamp(i32::MIN as f64, i32::MAX as f64),
            HeapType::U64 => value.clamp(0.0, u64::MAX as f64),
            HeapType::I64 => value.clamp(i64::MIN as f64, i64::MAX as f64),
            HeapType::F32 => value as f32 as f64,
            HeapType::F64 => value,
            HeapType::Number => value,
            HeapType::Bool => if value != 0.0 { 1.0 } else { 0.0 },
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
        match return_type {
            HeapType::Str | HeapType::Str16 => {
                let text_promise = response.text()?;
                let text = JsFuture::from(text_promise).await?;
                Ok(text)
            }
            HeapType::Array => {
                let json_promise = response.json()?;
                let json = JsFuture::from(json_promise).await?;
                Ok(json)
            }
            HeapType::Buffer => {
                let array_buffer_promise = response.array_buffer()?;
                let array_buffer = JsFuture::from(array_buffer_promise).await?;
                Ok(array_buffer)
            }
            _ => {
                let text_promise = response.text()?;
                let text = JsFuture::from(text_promise).await?;
                
                if let Some(num_str) = text.as_string() {
                    if let Ok(num) = num_str.parse::<f64>() {
                        let clamped = match return_type {
                            HeapType::U8 => num.clamp(0.0, u8::MAX as f64),
                            HeapType::I8 => num.clamp(i8::MIN as f64, i8::MAX as f64),
                            HeapType::U16 => num.clamp(0.0, u16::MAX as f64),
                            HeapType::I16 => num.clamp(i16::MIN as f64, i16::MAX as f64),
                            HeapType::U32 => num.clamp(0.0, u32::MAX as f64),
                            HeapType::I32 => num.clamp(i32::MIN as f64, i32::MAX as f64),
                            HeapType::U64 => num.clamp(0.0, u64::MAX as f64),
                            HeapType::I64 => num.clamp(i64::MIN as f64, i64::MAX as f64),
                            HeapType::F32 => num as f32 as f64,
                            HeapType::F64 => num,
                            HeapType::Number => num,
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

