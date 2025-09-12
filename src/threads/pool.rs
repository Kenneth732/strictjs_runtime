
// src/threads/pool.rs
use wasm_bindgen::prelude::*;
use js_sys::Function;
use std::collections::{VecDeque, HashMap};
use crate::threads::task::ThreadTask;
use crate::threads::types::ThreadPriority; // Removed ThreadState

#[wasm_bindgen]
#[derive(Clone)]
pub struct ThreadPool {
    max_threads: usize,
    active_tasks: HashMap<String, ThreadTask>,
    pending_tasks: VecDeque<ThreadTask>,
    completed_tasks: HashMap<String, ThreadTask>,
}

#[wasm_bindgen]
impl ThreadPool {
    #[wasm_bindgen(constructor)]
    pub fn new(max_threads: usize) -> ThreadPool {
        ThreadPool {
            max_threads,
            active_tasks: HashMap::new(),
            pending_tasks: VecDeque::new(),
            completed_tasks: HashMap::new(),
        }
    }
    
    #[wasm_bindgen(js_name = submitTask)]
    pub fn submit_task(&mut self, task: ThreadTask) -> String {
        let task_id = task.id().clone();
        self.pending_tasks.push_back(task);
        task_id
    }
    
    #[wasm_bindgen(js_name = submitFunction)]
    pub fn submit_function(
        &mut self,
        function: Function,
        args: JsValue,
        result_type: JsValue,
        priority: ThreadPriority
    ) -> Result<String, JsValue> {
        let heap_type = crate::types::HeapType::from_js_value(result_type)?;
        let task = ThreadTask::new(function, args, heap_type, priority)?;
        let task_id = task.id().clone();
        self.pending_tasks.push_back(task);
        Ok(task_id)
    }
    
#[wasm_bindgen(js_name = executeNext)]
pub async fn execute_next(&mut self) -> Result<JsValue, JsValue> {
    if self.active_tasks.len() >= self.max_threads {
        return Err(JsValue::from_str("Thread pool is at capacity"));
    }
    
    if let Some(mut task) = self.pending_tasks.pop_front() {
        let task_id = task.id().clone();
        
        // Execute the task first, then store it
        let result = task.execute().await;
        
        // Store the task in completed tasks
        self.completed_tasks.insert(task_id, task);
        
        result
    } else {
        Err(JsValue::from_str("No pending tasks"))
    }
}
    
#[wasm_bindgen(js_name = getTaskStatus)]
pub fn get_task_status(&self, task_id: &str) -> JsValue {
    let state = self.active_tasks.get(task_id)
        .map(|t| t.state())
        .or_else(|| self.pending_tasks.iter().find(|t| t.id() == task_id).map(|t| t.state()))
        .or_else(|| self.completed_tasks.get(task_id).map(|t| t.state()));
    
    // Convert to JsValue instead of Option
    match state {
        Some(s) => JsValue::from_str(&format!("{:?}", s)),
        None => JsValue::NULL,
    }
}
    
    #[wasm_bindgen(js_name = cancelTask)]
    pub fn cancel_task(&mut self, task_id: &str) -> bool {
        if self.active_tasks.contains_key(task_id) {
            // Can't cancel active tasks in this simple implementation
            false
        } else {
            self.pending_tasks.retain(|t| t.id() != task_id);
            true
        }
    }
    
#[wasm_bindgen(js_name = getCompletedResult)]
pub fn get_completed_result(&self, task_id: &str) -> JsValue {
    // Return JsValue::NULL for now - in real implementation, return actual result
    if self.completed_tasks.contains_key(task_id) {
        JsValue::NULL
    } else {
        JsValue::UNDEFINED
    }
}
    
    #[wasm_bindgen(getter)]
    pub fn active_count(&self) -> usize {
        self.active_tasks.len()
    }
    
    #[wasm_bindgen(getter)]
    pub fn pending_count(&self) -> usize {
        self.pending_tasks.len()
    }
    
    #[wasm_bindgen(getter)]
    pub fn completed_count(&self) -> usize {
        self.completed_tasks.len()
    }
}

