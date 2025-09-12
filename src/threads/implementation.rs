
// src/threads/implementation.rs
use wasm_bindgen::prelude::*;
use js_sys::{Promise, Function};
use std::collections::HashMap;
use crate::threads::pool::ThreadPool;
use crate::threads::types::{ThreadPriority, ThreadConfig}; // Added ThreadConfig back

#[wasm_bindgen]
pub struct ThreadManager {
    main_pool: ThreadPool,
    worker_pools: HashMap<String, ThreadPool>,
    #[allow(dead_code)]
    config: ThreadConfig,
}

#[wasm_bindgen]
impl ThreadManager {
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<ThreadManager, JsValue> {
        let _config_obj: js_sys::Object = config.dyn_into()?;
        let thread_config = ThreadConfig::default();
        
        Ok(ThreadManager {
            main_pool: ThreadPool::new(4),
            worker_pools: HashMap::new(),
            config: thread_config,
        })
    }
    
    #[wasm_bindgen(js_name = createPool)]
    pub fn create_pool(&mut self, name: &str, max_threads: usize) -> bool {
        if !self.worker_pools.contains_key(name) {
            self.worker_pools.insert(name.to_string(), ThreadPool::new(max_threads));
            true
        } else {
            false
        }
    }
    
    #[wasm_bindgen(js_name = getPool)]
    pub fn get_pool(&self, name: &str) -> Option<ThreadPool> {
        self.worker_pools.get(name).cloned()
    }
    
    #[wasm_bindgen(js_name = submitToPool)]
    pub fn submit_to_pool(
        &mut self,
        pool_name: &str,
        function: Function,
        args: JsValue,
        result_type: JsValue,
        priority: ThreadPriority
    ) -> Result<String, JsValue> {
        if let Some(pool) = self.worker_pools.get_mut(pool_name) {
            pool.submit_function(function, args, result_type, priority)
        } else {
            self.main_pool.submit_function(function, args, result_type, priority)
        }
    }
    
    #[wasm_bindgen(js_name = executeBatch)]
    pub async fn execute_batch(&mut self, pool_name: Option<String>, count: usize) -> Result<JsValue, JsValue> {
        let pool = if let Some(name) = pool_name {
            self.worker_pools.get_mut(&name).ok_or_else(|| JsValue::from_str("Pool not found"))?
        } else {
            &mut self.main_pool
        };
        
        let results = js_sys::Array::new(); // Removed mut
        
        for _ in 0..count {
            match pool.execute_next().await {
                Ok(result) => {
                    let _ = results.push(&result);
                },
                Err(_) => break,
            }
        }
        
        Ok(results.into())
    }

    #[wasm_bindgen(js_name = parallelMap)]
    pub async fn parallel_map(
        &self,
        array: JsValue,
        mapper: Function,
        result_type: JsValue,
        _pool_name: Option<String> // Prefix with underscore
    ) -> Result<JsValue, JsValue> {
        let array: js_sys::Array = array.dyn_into()?;
        let _heap_type = crate::types::HeapType::from_js_value(result_type)?;
        
        let mut promises = Vec::with_capacity(array.length() as usize);
        
        for i in 0..array.length() {
            let item = array.get(i);
            let args = js_sys::Array::new();
            args.push(&item);
            args.push(&JsValue::from_f64(i as f64));
            
            let promise = Promise::new(&mut |resolve, reject| {
                let result = mapper.apply(&JsValue::NULL, &args);
                match result {
                    Ok(value) => {
                        let _ = resolve.call1(&JsValue::NULL, &value);
                    },
                    Err(error) => {
                        let _ = reject.call1(&JsValue::NULL, &error);
                    },
                }
            });
            
            promises.push(promise);
        }
        
        let results = Promise::all(&js_sys::Array::from_iter(promises.iter()));
        Ok(results.into())
    }
}

