
// src/types/gpu_types.rs
use wasm_bindgen::prelude::*;
use super::HeapType;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GPUType {
    TensorF32,    
    TensorI32,    
    TensorU32,    
    MatrixF32,
    MatrixF64,    
    VectorF32,    
    ImageRGBA, 
    ComputeBuffer,
}

#[derive(Clone, Copy, Debug)]
pub enum MemoryLayout {
    Linear,
    Strided,
    ColumnMajor,
    RowMajor,
    Packed,
    Channels,
}

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct JsGPUType {
    inner: GPUType,
}

#[wasm_bindgen]
impl JsGPUType {
    #[wasm_bindgen(constructor)]
    pub fn new(type_str: &str) -> Result<JsGPUType, JsValue> {
        let gpu_type = match type_str.to_lowercase().as_str() {
            "tensor_f32" => GPUType::TensorF32,
            "tensor_i32" => GPUType::TensorI32,
            "tensor_u32" => GPUType::TensorU32,
            "matrix_f32" => GPUType::MatrixF32,
            "matrix_f64" => GPUType::MatrixF64,
            "vector_f32" => GPUType::VectorF32,
            "image_rgba" => GPUType::ImageRGBA,
            "compute_buffer" => GPUType::ComputeBuffer,
            _ => return Err(JsValue::from_str(&format!("Unknown GPU type: {}", type_str))),
        };
        Ok(JsGPUType { inner: gpu_type })
    }

    #[wasm_bindgen(js_name = isTensor)]
    pub fn is_tensor(&self) -> bool {
        matches!(self.inner, GPUType::TensorF32 | GPUType::TensorI32 | GPUType::TensorU32)
    }

    #[wasm_bindgen(js_name = isMatrix)]
    pub fn is_matrix(&self) -> bool {
        matches!(self.inner, GPUType::MatrixF32 | GPUType::MatrixF64)
    }

    #[wasm_bindgen(js_name = getInfo)]
    pub fn get_info(&self) -> JsValue {
        let info = self.inner.get_info();
        let obj = js_sys::Object::new();
        
        js_sys::Reflect::set(&obj, &"elementSize".into(), &info.element_size.into()).unwrap();
        js_sys::Reflect::set(&obj, &"alignment".into(), &info.alignment.into()).unwrap();
        js_sys::Reflect::set(&obj, &"preferredBackend".into(), &info.preferred_backend.into()).unwrap();
        js_sys::Reflect::set(&obj, &"memoryLayout".into(), &info.memory_layout.to_string().into()).unwrap();
        
        obj.into()
    }

    #[wasm_bindgen(js_name = getBackends)]
    pub fn get_backends(&self) -> js_sys::Array {
        self.inner.get_backends()
            .into_iter()
            .map(JsValue::from)
            .collect()
    }

    #[wasm_bindgen(js_name = elementSize)]
    pub fn element_size(&self) -> usize {
        self.inner.element_size()
    }

    #[wasm_bindgen(js_name = alignment)]
    pub fn alignment(&self) -> usize {
        self.inner.alignment()
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        self.inner.to_string()
    }

    #[wasm_bindgen(js_name = supportedOperations)]
    pub fn supported_operations(&self) -> js_sys::Array {
        self.inner.supported_operations()
            .into_iter()
            .map(JsValue::from)
            .collect()
    }

    #[wasm_bindgen(js_name = isCompatibleWith)]
    pub fn is_compatible_with(&self, other: &JsGPUType) -> bool {
        self.inner.is_compatible_with(&other.inner)
    }

    #[wasm_bindgen(js_name = getHeapType)]
    pub fn get_heap_type(&self) -> HeapType {
        self.inner.get_heap_type()
    }

    #[wasm_bindgen(js_name = computeCapabilities)]
    pub fn compute_capabilities(&self) -> js_sys::Array {
        self.inner.compute_capabilities()
            .into_iter()
            .map(JsValue::from)
            .collect()
    }

    #[wasm_bindgen(js_name = optimalWorkgroupSize)]
    pub fn optimal_workgroup_size(&self) -> js_sys::Array {
        let (x, y, z) = self.inner.optimal_workgroup_size();
        let arr = js_sys::Array::new();
        arr.push(&x.into());
        arr.push(&y.into());
        arr.push(&z.into());
        arr
    }
}

impl GPUType {
    pub fn get_info(&self) -> GPUTypeInfo {
        match self {
            GPUType::TensorF32 => GPUTypeInfo {
                element_size: 4,
                alignment: 16,
                preferred_backend: "webgpu".to_string(),
                memory_layout: MemoryLayout::Strided,
            },
            GPUType::TensorI32 => GPUTypeInfo {
                element_size: 4,
                alignment: 16,
                preferred_backend: "webgpu".to_string(),
                memory_layout: MemoryLayout::Strided,
            },
            GPUType::TensorU32 => GPUTypeInfo {
                element_size: 4,
                alignment: 16,
                preferred_backend: "webgpu".to_string(),
                memory_layout: MemoryLayout::Strided,
            },
            GPUType::MatrixF32 => GPUTypeInfo {
                element_size: 4,
                alignment: 16,
                preferred_backend: "webgpu".to_string(),
                memory_layout: MemoryLayout::ColumnMajor,
            },
            GPUType::MatrixF64 => GPUTypeInfo {
                element_size: 8,
                alignment: 16,
                preferred_backend: "webgpu".to_string(),
                memory_layout: MemoryLayout::ColumnMajor,
            },
            GPUType::VectorF32 => GPUTypeInfo {
                element_size: 4,
                alignment: 16,
                preferred_backend: "webgpu".to_string(),
                memory_layout: MemoryLayout::Packed,
            },
            GPUType::ImageRGBA => GPUTypeInfo {
                element_size: 4,
                alignment: 4,
                preferred_backend: "webgpu".to_string(),
                memory_layout: MemoryLayout::Channels,
            },
            GPUType::ComputeBuffer => GPUTypeInfo {
                element_size: 1,
                alignment: 16,
                preferred_backend: "webgpu".to_string(),
                memory_layout: MemoryLayout::Linear,
            },
        }
    }

    pub fn get_backends(&self) -> Vec<String> {
        match self {
            GPUType::TensorF32 | GPUType::TensorI32 | GPUType::TensorU32 => {
                vec!["webgpu".to_string(), "webgl".to_string(), "wasm".to_string()]
            }
            GPUType::MatrixF32 | GPUType::MatrixF64 => {
                vec!["webgpu".to_string(), "wasm".to_string()]
            }
            GPUType::VectorF32 => {
                vec!["webgpu".to_string(), "webgl".to_string(), "wasm".to_string()]
            }
            GPUType::ImageRGBA => {
                vec!["webgpu".to_string(), "webgl".to_string()]
            }
            GPUType::ComputeBuffer => {
                vec!["webgpu".to_string(), "wasm".to_string()]
            }
        }
    }

    pub fn element_size(&self) -> usize {
        self.get_info().element_size
    }

    pub fn alignment(&self) -> usize {
        self.get_info().alignment
    }

    pub fn to_string(&self) -> String {
        match self {
            GPUType::TensorF32 => "tensor_f32",
            GPUType::TensorI32 => "tensor_i32",
            GPUType::TensorU32 => "tensor_u32",
            GPUType::MatrixF32 => "matrix_f32",
            GPUType::MatrixF64 => "matrix_f64",
            GPUType::VectorF32 => "vector_f32",
            GPUType::ImageRGBA => "image_rgba",
            GPUType::ComputeBuffer => "compute_buffer",
        }.to_string()
    }

    pub fn supported_operations(&self) -> Vec<String> {
        match self {
            GPUType::TensorF32 | GPUType::TensorI32 | GPUType::TensorU32 => {
                vec![
                    "matmul".to_string(),
                    "convolution".to_string(),
                    "pooling".to_string(),
                    "activation".to_string(),
                    "normalization".to_string(),
                ]
            }
            GPUType::MatrixF32 | GPUType::MatrixF64 => {
                vec![
                    "multiply".to_string(),
                    "transpose".to_string(),
                    "inverse".to_string(),
                    "determinant".to_string(),
                    "decomposition".to_string(),
                ]
            }
            GPUType::VectorF32 => {
                vec![
                    "dot".to_string(),
                    "cross".to_string(),
                    "normalize".to_string(),
                    "length".to_string(),
                    "distance".to_string(),
                ]
            }
            GPUType::ImageRGBA => {
                vec![
                    "filter".to_string(),
                    "transform".to_string(),
                    "blend".to_string(),
                    "sample".to_string(),
                    "encode".to_string(),
                ]
            }
            GPUType::ComputeBuffer => {
                vec![
                    "read".to_string(),
                    "write".to_string(),
                    "copy".to_string(),
                    "map".to_string(),
                    "unmap".to_string(),
                ]
            }
        }
    }

    pub fn is_compatible_with(&self, other: &GPUType) -> bool {
        match (self, other) {
            (GPUType::TensorF32, GPUType::TensorF32) => true,
            (GPUType::TensorI32, GPUType::TensorI32) => true,
            (GPUType::TensorU32, GPUType::TensorU32) => true,
            (GPUType::MatrixF32, GPUType::MatrixF32) => true,
            (GPUType::MatrixF64, GPUType::MatrixF64) => true,
            (GPUType::VectorF32, GPUType::VectorF32) => true,
            (GPUType::ImageRGBA, GPUType::ImageRGBA) => true,
            (GPUType::ComputeBuffer, GPUType::ComputeBuffer) => true,
            _ => false,
        }
    }

    pub fn get_heap_type(&self) -> HeapType {
        match self {
            GPUType::TensorF32 | GPUType::MatrixF32 | GPUType::VectorF32 => HeapType::F32,
            GPUType::MatrixF64 => HeapType::F64,
            GPUType::TensorI32 => HeapType::I32,
            GPUType::TensorU32 => HeapType::U32,
            GPUType::ImageRGBA | GPUType::ComputeBuffer => HeapType::Buffer,
        }
    }

    pub fn compute_capabilities(&self) -> Vec<&'static str> {
        match self {
            GPUType::TensorF32 => vec!["neural_networks", "matrix_ops", "convolution"],
            GPUType::TensorI32 => vec!["integer_math", "indexing", "quantization"],
            GPUType::TensorU32 => vec!["unsigned_math", "addressing", "bit_ops"],
            GPUType::MatrixF32 => vec!["linear_algebra", "decomposition", "transforms"],
            GPUType::MatrixF64 => vec!["precision_math", "scientific", "financial"],
            GPUType::VectorF32 => vec!["geometry", "physics", "graphics"],
            GPUType::ImageRGBA => vec!["image_processing", "computer_vision", "graphics"],
            GPUType::ComputeBuffer => vec!["general_compute", "data_processing", "streaming"],
        }
    }

    pub fn optimal_workgroup_size(&self) -> (u32, u32, u32) {
        match self {
            GPUType::TensorF32 | GPUType::TensorI32 | GPUType::TensorU32 => (16, 16, 1),
            GPUType::MatrixF32 | GPUType::MatrixF64 => (8, 8, 1),
            GPUType::VectorF32 => (64, 1, 1),
            GPUType::ImageRGBA => (16, 16, 1),
            GPUType::ComputeBuffer => (64, 1, 1),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GPUTypeInfo {
    pub element_size: usize,
    pub alignment: usize,
    pub preferred_backend: String,
    pub memory_layout: MemoryLayout,
}

impl MemoryLayout {
    pub fn to_string(&self) -> String {
        match self {
            MemoryLayout::Linear => "linear",
            MemoryLayout::Strided => "strided",
            MemoryLayout::ColumnMajor => "column_major",
            MemoryLayout::RowMajor => "row_major",
            MemoryLayout::Packed => "packed",
            MemoryLayout::Channels => "channels",
        }.to_string()
    }
}

impl From<GPUType> for HeapType {
    fn from(gpu_type: GPUType) -> Self {
        gpu_type.get_heap_type()
    }
}

#[wasm_bindgen]
pub struct GPUMemoryManager {
    buffers: std::collections::HashMap<usize, GPUBufferInfo>,
    next_id: usize,
}

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct GPUBufferInfo {
    buffer_id: usize,
    buffer_size: usize,
    buffer_usage: GPUBufferUsage,
}

#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct GPUBufferUsage(u32);

impl GPUBufferUsage {
    pub const COPY_SRC: Self = Self(1 << 0);
    pub const COPY_DST: Self = Self(1 << 1);
    pub const STORAGE: Self = Self(1 << 2);
    pub const UNIFORM: Self = Self(1 << 3);
    pub const VERTEX: Self = Self(1 << 4);
    pub const INDEX: Self = Self(1 << 5);
}

#[wasm_bindgen]
impl GPUMemoryManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            buffers: std::collections::HashMap::new(),
            next_id: 1,
        }
    }

    #[wasm_bindgen(js_name = createBuffer)]
    pub fn create_buffer(&mut self, _gpu_type: &JsGPUType, size: usize, usage: GPUBufferUsage) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let buffer_info = GPUBufferInfo {
            buffer_id: id,
            buffer_size: size,
            buffer_usage: usage,
        };

        self.buffers.insert(id, buffer_info);
        id
    }

    #[wasm_bindgen(js_name = getBufferInfo)]
    pub fn get_buffer_info(&self, id: usize) -> Option<GPUBufferInfo> {
        self.buffers.get(&id).cloned()
    }

    #[wasm_bindgen(js_name = destroyBuffer)]
    pub fn destroy_buffer(&mut self, id: usize) -> bool {
        self.buffers.remove(&id).is_some()
    }

    #[wasm_bindgen(js_name = getTotalMemory)]
    pub fn get_total_memory(&self) -> usize {
        self.buffers.values().map(|info| info.buffer_size).sum()
    }

    #[wasm_bindgen(js_name = getBufferCount)]
    pub fn get_buffer_count(&self) -> usize {
        self.buffers.len()
    }
}

#[wasm_bindgen]
impl GPUBufferInfo {
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> usize {
        self.buffer_id
    }

    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.buffer_size
    }

    #[wasm_bindgen(getter)]
    pub fn usage(&self) -> GPUBufferUsage {
        self.buffer_usage
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        format!(
            "GPUBufferInfo(id: {}, size: {}, usage: {})",
            self.buffer_id,
            self.buffer_size,
            self.buffer_usage.bits()
        )
    }
}

#[wasm_bindgen]
impl GPUBufferUsage {
    #[wasm_bindgen(constructor)]
    pub fn new(bits: u32) -> Self {
        Self(bits)
    }

    #[wasm_bindgen(js_name = bits)]
    pub fn bits(&self) -> u32 {
        self.0
    }

    #[wasm_bindgen(js_name = contains)]
    pub fn contains(&self, other: &GPUBufferUsage) -> bool {
        (self.0 & other.0) != 0
    }

    #[wasm_bindgen(js_name = with)]
    pub fn with(&self, other: &GPUBufferUsage) -> GPUBufferUsage {
        GPUBufferUsage(self.0 | other.0)
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        let mut flags = Vec::new();
        if self.contains(&GPUBufferUsage::COPY_SRC) {
            flags.push("COPY_SRC");
        }
        if self.contains(&GPUBufferUsage::COPY_DST) {
            flags.push("COPY_DST");
        }
        if self.contains(&GPUBufferUsage::STORAGE) {
            flags.push("STORAGE");
        }
        if self.contains(&GPUBufferUsage::UNIFORM) {
            flags.push("UNIFORM");
        }
        if self.contains(&GPUBufferUsage::VERTEX) {
            flags.push("VERTEX");
        }
        if self.contains(&GPUBufferUsage::INDEX) {
            flags.push("INDEX");
        }
        format!("GPUBufferUsage({})", flags.join(" | "))
    }
}

// Helper function to create GPU types from JS
#[wasm_bindgen(js_name = createGPUType)]
pub fn create_gpu_type(type_str: &str) -> Result<JsGPUType, JsValue> {
    JsGPUType::new(type_str)
}

// Helper function to get all available GPU types
#[wasm_bindgen(js_name = getAvailableGPUTypes)]
pub fn get_available_gpu_types() -> js_sys::Array {
    let types = vec![
        "tensor_f32", "tensor_i32", "tensor_u32",
        "matrix_f32", "matrix_f64", "vector_f32",
        "image_rgba", "compute_buffer"
    ];
    
    types.into_iter()
        .map(JsValue::from)
        .collect()
}

