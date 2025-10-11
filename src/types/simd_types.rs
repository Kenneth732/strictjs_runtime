

// src/types/simd_types.rs
use wasm_bindgen::prelude::*;
use super::HeapType;

// SIMD types for high-performance computations
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SIMDType {
    F32x4,  // 4x f32 for 3D math
    F32x8,  // 8x f32 for ML matrices
    F64x2,  // 2x f64 for precision math
    F64x4,  // 4x f64
    I32x4,  // 4x i32 for indices
    U32x4,  // 4x u32 for graphics
    Boolx4, // Mask operations
    I16x8,  // 8x i16 for audio processing
    U16x8,  // 8x u16 for image processing
    I8x16,  // 16x i8 for ML quantization
    U8x16,  // 16x u8 for pixel data
}

// SIMD operations that can be performed
#[derive(Clone, Copy, Debug, PartialEq, Eq)] // FIXED: Added PartialEq, Eq
pub enum SIMDOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Min,
    Max,
    Sqrt,
    Abs,
    Neg,
    And,
    Or,
    Xor,
    Not,
    ShiftLeft,
    ShiftRight,
    CompareEqual,
    CompareGreater,
    CompareLess,
    DotProduct,
    CrossProduct,
    Normalize,
    Shuffle,
    Blend,
}


#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct JsSIMDType {
    inner: SIMDType,
}

#[wasm_bindgen]
impl JsSIMDType {
    #[wasm_bindgen(constructor)]
    pub fn new(type_str: &str) -> Result<JsSIMDType, JsValue> {
        let simd_type = match type_str.to_lowercase().as_str() {
            "f32x4" => SIMDType::F32x4,
            "f32x8" => SIMDType::F32x8,
            "f64x2" => SIMDType::F64x2,
            "f64x4" => SIMDType::F64x4,
            "i32x4" => SIMDType::I32x4,
            "u32x4" => SIMDType::U32x4,
            "boolx4" => SIMDType::Boolx4,
            "i16x8" => SIMDType::I16x8,
            "u16x8" => SIMDType::U16x8,
            "i8x16" => SIMDType::I8x16,
            "u8x16" => SIMDType::U8x16,
            _ => return Err(JsValue::from_str(&format!("Unknown SIMD type: {}", type_str))),
        };
        Ok(JsSIMDType { inner: simd_type })
    }

    #[wasm_bindgen(js_name = elementCount)]
    pub fn element_count(&self) -> usize {
        self.inner.element_count()
    }

    #[wasm_bindgen(js_name = alignment)]
    pub fn alignment(&self) -> usize {
        self.inner.alignment()
    }

    #[wasm_bindgen(js_name = totalSize)]
    pub fn total_size(&self) -> usize {
        self.inner.total_size()
    }

    #[wasm_bindgen(js_name = isFloatingPoint)]
    pub fn is_floating_point(&self) -> bool {
        matches!(self.inner, SIMDType::F32x4 | SIMDType::F32x8 | SIMDType::F64x2 | SIMDType::F64x4)
    }

    #[wasm_bindgen(js_name = isInteger)]
    pub fn is_integer(&self) -> bool {
        matches!(self.inner, SIMDType::I32x4 | SIMDType::U32x4 | SIMDType::I16x8 | SIMDType::U16x8 | SIMDType::I8x16 | SIMDType::U8x16)
    }

    #[wasm_bindgen(js_name = isBoolean)]
    pub fn is_boolean(&self) -> bool {
        matches!(self.inner, SIMDType::Boolx4)
    }

    #[wasm_bindgen(js_name = supportedOperations)]
    pub fn supported_operations(&self) -> js_sys::Array {
        self.inner.supported_operations()
            .into_iter()
            .map(|op| JsValue::from(op.to_string()))
            .collect()
    }

    #[wasm_bindgen(js_name = elementType)]
    pub fn element_type(&self) -> HeapType {
        self.inner.element_type()
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        self.inner.to_string()
    }

    #[wasm_bindgen(js_name = getInfo)]
    pub fn get_info(&self) -> JsValue {
        let info = self.inner.get_info();
        let obj = js_sys::Object::new();
        
        js_sys::Reflect::set(&obj, &"elementCount".into(), &info.element_count.into()).unwrap();
        js_sys::Reflect::set(&obj, &"elementSize".into(), &info.element_size.into()).unwrap();
        js_sys::Reflect::set(&obj, &"totalSize".into(), &info.total_size.into()).unwrap();
        js_sys::Reflect::set(&obj, &"alignment".into(), &info.alignment.into()).unwrap();
        js_sys::Reflect::set(&obj, &"isFloatingPoint".into(), &info.is_floating_point.into()).unwrap();
        js_sys::Reflect::set(&obj, &"isInteger".into(), &info.is_integer.into()).unwrap();
        js_sys::Reflect::set(&obj, &"isBoolean".into(), &info.is_boolean.into()).unwrap();
        js_sys::Reflect::set(&obj, &"laneCount".into(), &info.lane_count.into()).unwrap();
        
        obj.into()
    }

    #[wasm_bindgen(js_name = canPerformOperation)]
    pub fn can_perform_operation(&self, operation: &str) -> bool {
        self.inner.can_perform_operation(operation)
    }

    #[wasm_bindgen(js_name = getOptimalLaneCount)]
    pub fn get_optimal_lane_count(&self) -> usize {
        self.inner.optimal_lane_count()
    }

    #[wasm_bindgen(js_name = isCompatibleWith)]
    pub fn is_compatible_with(&self, other: &JsSIMDType) -> bool {
        self.inner.is_compatible_with(&other.inner)
    }

    #[wasm_bindgen(js_name = getBinaryResultType)]
    pub fn get_binary_result_type(&self, other: &JsSIMDType) -> Option<JsSIMDType> {
        self.inner.get_binary_result_type(&other.inner).map(JsSIMDType::from)
    }
}

// Internal implementation for SIMDType
impl SIMDType {
    pub fn element_count(&self) -> usize {
        match self {
            SIMDType::F32x4 => 4,
            SIMDType::F32x8 => 8,
            SIMDType::F64x2 => 2,
            SIMDType::F64x4 => 4,
            SIMDType::I32x4 => 4,
            SIMDType::U32x4 => 4,
            SIMDType::Boolx4 => 4,
            SIMDType::I16x8 => 8,
            SIMDType::U16x8 => 8,
            SIMDType::I8x16 => 16,
            SIMDType::U8x16 => 16,
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            SIMDType::F32x4 | SIMDType::F32x8 => 4,
            SIMDType::F64x2 | SIMDType::F64x4 => 8,
            SIMDType::I32x4 | SIMDType::U32x4 => 4,
            SIMDType::Boolx4 => 1,
            SIMDType::I16x8 | SIMDType::U16x8 => 2,
            SIMDType::I8x16 | SIMDType::U8x16 => 1,
        }
    }

    pub fn total_size(&self) -> usize {
        self.element_count() * self.element_size()
    }

    pub fn alignment(&self) -> usize {
        match self {
            SIMDType::F32x4 | SIMDType::I32x4 | SIMDType::U32x4 | SIMDType::Boolx4 => 16,
            SIMDType::F32x8 | SIMDType::F64x4 => 32,
            SIMDType::F64x2 => 16,
            SIMDType::I16x8 | SIMDType::U16x8 => 16,
            SIMDType::I8x16 | SIMDType::U8x16 => 16,
        }
    }

    pub fn supported_operations(&self) -> Vec<SIMDOperation> {
        match self {
            SIMDType::F32x4 | SIMDType::F32x8 => {
                vec![
                    SIMDOperation::Add,
                    SIMDOperation::Subtract,
                    SIMDOperation::Multiply,
                    SIMDOperation::Divide,
                    SIMDOperation::Min,
                    SIMDOperation::Max,
                    SIMDOperation::Sqrt,
                    SIMDOperation::Abs,
                    SIMDOperation::Neg,
                    SIMDOperation::CompareEqual,
                    SIMDOperation::CompareGreater,
                    SIMDOperation::CompareLess,
                    SIMDOperation::DotProduct,
                    SIMDOperation::CrossProduct,
                    SIMDOperation::Normalize,
                    SIMDOperation::Shuffle,
                    SIMDOperation::Blend,
                ]
            }
            SIMDType::F64x2 | SIMDType::F64x4 => {
                vec![
                    SIMDOperation::Add,
                    SIMDOperation::Subtract,
                    SIMDOperation::Multiply,
                    SIMDOperation::Divide,
                    SIMDOperation::Min,
                    SIMDOperation::Max,
                    SIMDOperation::Sqrt,
                    SIMDOperation::Abs,
                    SIMDOperation::Neg,
                    SIMDOperation::CompareEqual,
                    SIMDOperation::CompareGreater,
                    SIMDOperation::CompareLess,
                    SIMDOperation::Shuffle,
                    SIMDOperation::Blend,
                ]
            }
            SIMDType::I32x4 | SIMDType::U32x4 => {
                vec![
                    SIMDOperation::Add,
                    SIMDOperation::Subtract,
                    SIMDOperation::Multiply,
                    SIMDOperation::And,
                    SIMDOperation::Or,
                    SIMDOperation::Xor,
                    SIMDOperation::Not,
                    SIMDOperation::ShiftLeft,
                    SIMDOperation::ShiftRight,
                    SIMDOperation::CompareEqual,
                    SIMDOperation::CompareGreater,
                    SIMDOperation::CompareLess,
                    SIMDOperation::Shuffle,
                    SIMDOperation::Blend,
                ]
            }
            SIMDType::Boolx4 => {
                vec![
                    SIMDOperation::And,
                    SIMDOperation::Or,
                    SIMDOperation::Xor,
                    SIMDOperation::Not,
                    SIMDOperation::Blend,
                ]
            }
            SIMDType::I16x8 | SIMDType::U16x8 => {
                vec![
                    SIMDOperation::Add,
                    SIMDOperation::Subtract,
                    SIMDOperation::Multiply,
                    SIMDOperation::And,
                    SIMDOperation::Or,
                    SIMDOperation::Xor,
                    SIMDOperation::ShiftLeft,
                    SIMDOperation::ShiftRight,
                    SIMDOperation::CompareEqual,
                    SIMDOperation::CompareGreater,
                    SIMDOperation::CompareLess,
                    SIMDOperation::Shuffle,
                    SIMDOperation::Blend,
                ]
            }
            SIMDType::I8x16 | SIMDType::U8x16 => {
                vec![
                    SIMDOperation::Add,
                    SIMDOperation::Subtract,
                    SIMDOperation::And,
                    SIMDOperation::Or,
                    SIMDOperation::Xor,
                    SIMDOperation::CompareEqual,
                    SIMDOperation::CompareGreater,
                    SIMDOperation::CompareLess,
                    SIMDOperation::Shuffle,
                    SIMDOperation::Blend,
                ]
            }
        }
    }

    pub fn element_type(&self) -> HeapType {
        match self {
            SIMDType::F32x4 | SIMDType::F32x8 => HeapType::F32,
            SIMDType::F64x2 | SIMDType::F64x4 => HeapType::F64,
            SIMDType::I32x4 => HeapType::I32,
            SIMDType::U32x4 => HeapType::U32,
            SIMDType::Boolx4 => HeapType::Bool,
            SIMDType::I16x8 => HeapType::I16,
            SIMDType::U16x8 => HeapType::U16,
            SIMDType::I8x16 => HeapType::I8,
            SIMDType::U8x16 => HeapType::U8,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            SIMDType::F32x4 => "f32x4",
            SIMDType::F32x8 => "f32x8",
            SIMDType::F64x2 => "f64x2",
            SIMDType::F64x4 => "f64x4",
            SIMDType::I32x4 => "i32x4",
            SIMDType::U32x4 => "u32x4",
            SIMDType::Boolx4 => "boolx4",
            SIMDType::I16x8 => "i16x8",
            SIMDType::U16x8 => "u16x8",
            SIMDType::I8x16 => "i8x16",
            SIMDType::U8x16 => "u8x16",
        }.to_string()
    }

    pub fn get_info(&self) -> SIMDTypeInfo {
        SIMDTypeInfo {
            element_count: self.element_count(),
            element_size: self.element_size(),
            total_size: self.total_size(),
            alignment: self.alignment(),
            is_floating_point: self.is_floating_point(),
            is_integer: self.is_integer(),
            is_boolean: self.is_boolean(),
            lane_count: self.optimal_lane_count(),
        }
    }

    pub fn can_perform_operation(&self, operation: &str) -> bool {
        let op = match operation.to_lowercase().as_str() {
            "add" => Some(SIMDOperation::Add),
            "subtract" => Some(SIMDOperation::Subtract),
            "multiply" => Some(SIMDOperation::Multiply),
            "divide" => Some(SIMDOperation::Divide),
            "min" => Some(SIMDOperation::Min),
            "max" => Some(SIMDOperation::Max),
            "sqrt" => Some(SIMDOperation::Sqrt),
            "abs" => Some(SIMDOperation::Abs),
            "neg" => Some(SIMDOperation::Neg),
            "and" => Some(SIMDOperation::And),
            "or" => Some(SIMDOperation::Or),
            "xor" => Some(SIMDOperation::Xor),
            "not" => Some(SIMDOperation::Not),
            "shiftleft" => Some(SIMDOperation::ShiftLeft),
            "shiftright" => Some(SIMDOperation::ShiftRight),
            "compareequal" => Some(SIMDOperation::CompareEqual),
            "comparegreater" => Some(SIMDOperation::CompareGreater),
            "compareless" => Some(SIMDOperation::CompareLess),
            "dotproduct" => Some(SIMDOperation::DotProduct),
            "crossproduct" => Some(SIMDOperation::CrossProduct),
            "normalize" => Some(SIMDOperation::Normalize),
            "shuffle" => Some(SIMDOperation::Shuffle),
            "blend" => Some(SIMDOperation::Blend),
            _ => None,
        };

        if let Some(op) = op {
            self.supported_operations().contains(&op)
        } else {
            false
        }
    }

    pub fn optimal_lane_count(&self) -> usize {
        match self {
            SIMDType::F32x4 | SIMDType::I32x4 | SIMDType::U32x4 | SIMDType::Boolx4 => 4,
            SIMDType::F32x8 => 8,
            SIMDType::F64x2 => 2,
            SIMDType::F64x4 => 4,
            SIMDType::I16x8 | SIMDType::U16x8 => 8,
            SIMDType::I8x16 | SIMDType::U8x16 => 16,
        }
    }

    pub fn is_floating_point(&self) -> bool {
        matches!(self, SIMDType::F32x4 | SIMDType::F32x8 | SIMDType::F64x2 | SIMDType::F64x4)
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, SIMDType::I32x4 | SIMDType::U32x4 | SIMDType::I16x8 | SIMDType::U16x8 | SIMDType::I8x16 | SIMDType::U8x16)
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self, SIMDType::Boolx4)
    }

    pub fn is_compatible_with(&self, other: &SIMDType) -> bool {
        self.element_type() == other.element_type() && self.element_count() == other.element_count()
    }

    pub fn get_binary_result_type(&self, other: &SIMDType) -> Option<SIMDType> {
        if !self.is_compatible_with(other) {
            return None;
        }

        // For binary operations, result type is the same as input types
        Some(*self)
    }
}

// Internal struct for SIMD type info
#[derive(Clone, Debug)]
pub struct SIMDTypeInfo {
    pub element_count: usize,
    pub element_size: usize,
    pub total_size: usize,
    pub alignment: usize,
    pub is_floating_point: bool,
    pub is_integer: bool,
    pub is_boolean: bool,
    pub lane_count: usize,
}

impl SIMDOperation {
    pub fn to_string(&self) -> String {
        match self {
            SIMDOperation::Add => "add",
            SIMDOperation::Subtract => "subtract",
            SIMDOperation::Multiply => "multiply",
            SIMDOperation::Divide => "divide",
            SIMDOperation::Min => "min",
            SIMDOperation::Max => "max",
            SIMDOperation::Sqrt => "sqrt",
            SIMDOperation::Abs => "abs",
            SIMDOperation::Neg => "neg",
            SIMDOperation::And => "and",
            SIMDOperation::Or => "or",
            SIMDOperation::Xor => "xor",
            SIMDOperation::Not => "not",
            SIMDOperation::ShiftLeft => "shift_left",
            SIMDOperation::ShiftRight => "shift_right",
            SIMDOperation::CompareEqual => "compare_equal",
            SIMDOperation::CompareGreater => "compare_greater",
            SIMDOperation::CompareLess => "compare_less",
            SIMDOperation::DotProduct => "dot_product",
            SIMDOperation::CrossProduct => "cross_product",
            SIMDOperation::Normalize => "normalize",
            SIMDOperation::Shuffle => "shuffle",
            SIMDOperation::Blend => "blend",
        }.to_string()
    }
}

// Conversion from SIMDType to JsSIMDType
impl From<SIMDType> for JsSIMDType {
    fn from(simd_type: SIMDType) -> Self {
        JsSIMDType { inner: simd_type }
    }
}

// SIMD performance optimizations
impl SIMDType {
    pub fn recommended_use_cases(&self) -> Vec<&'static str> {
        match self {
            SIMDType::F32x4 => vec!["3d_math", "vector_operations", "graphics", "physics"],
            SIMDType::F32x8 => vec!["matrix_operations", "neural_networks", "scientific_computing"],
            SIMDType::F64x2 => vec!["precision_math", "financial_calculations"],
            SIMDType::F64x4 => vec!["scientific_simulations", "high_precision_math"],
            SIMDType::I32x4 => vec!["indices", "coordinates", "integer_math"],
            SIMDType::U32x4 => vec!["graphics", "bit_operations", "hashing"],
            SIMDType::Boolx4 => vec!["masking", "conditional_operations", "branching"],
            SIMDType::I16x8 => vec!["audio_processing", "signal_processing"],
            SIMDType::U16x8 => vec!["image_processing", "color_operations"],
            SIMDType::I8x16 => vec!["ml_quantization", "neural_networks"],
            SIMDType::U8x16 => vec!["pixel_operations", "compression"],
        }
    }

    pub fn performance_multiplier(&self) -> f64 {
        match self {
            SIMDType::F32x4 => 4.0,
            SIMDType::F32x8 => 8.0,
            SIMDType::F64x2 => 2.0,
            SIMDType::F64x4 => 4.0,
            SIMDType::I32x4 => 4.0,
            SIMDType::U32x4 => 4.0,
            SIMDType::Boolx4 => 4.0,
            SIMDType::I16x8 => 8.0,
            SIMDType::U16x8 => 8.0,
            SIMDType::I8x16 => 16.0,
            SIMDType::U8x16 => 16.0,
        }
    }
}

// Helper functions for JS interaction
#[wasm_bindgen(js_name = createSIMDType)]
pub fn create_simd_type(type_str: &str) -> Result<JsSIMDType, JsValue> {
    JsSIMDType::new(type_str)
}

#[wasm_bindgen(js_name = getAvailableSIMDTypes)]
pub fn get_available_simd_types() -> js_sys::Array {
    let types = vec![
        "f32x4", "f32x8", "f64x2", "f64x4",
        "i32x4", "u32x4", "boolx4",
        "i16x8", "u16x8", "i8x16", "u8x16"
    ];
    
    types.into_iter()
        .map(JsValue::from)
        .collect()
}

#[wasm_bindgen(js_name = getSIMDTypeForUseCase)]
pub fn get_simd_type_for_use_case(use_case: &str) -> Option<JsSIMDType> {
    let use_case_lower = use_case.to_lowercase();
    
    // Find the best SIMD type for the given use case
    let all_types = vec![
        SIMDType::F32x4, SIMDType::F32x8, SIMDType::F64x2, SIMDType::F64x4,
        SIMDType::I32x4, SIMDType::U32x4, SIMDType::Boolx4,
        SIMDType::I16x8, SIMDType::U16x8, SIMDType::I8x16, SIMDType::U8x16,
    ];
    
    for simd_type in all_types {
        let use_cases = simd_type.recommended_use_cases();
        if use_cases.iter().any(|uc| uc.to_lowercase() == use_case_lower) {
            return Some(JsSIMDType::from(simd_type));
        }
    }
    
    None
}

