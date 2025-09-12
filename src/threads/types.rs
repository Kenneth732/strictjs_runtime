

// src/threads/types.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum ThreadState {
    Idle,
    Running,
    Paused,
    Completed,
    Error,
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct ThreadConfig {
    pub priority: ThreadPriority,
    pub stack_size: usize,
    pub timeout_ms: u32,
    pub max_retries: u32,
}

impl Default for ThreadConfig {
    fn default() -> Self {
        ThreadConfig {
            priority: ThreadPriority::Normal,
            stack_size: 1024 * 1024, // 1MB
            timeout_ms: 5000,
            max_retries: 3,
        }
    }
}

