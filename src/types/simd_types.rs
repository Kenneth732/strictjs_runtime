use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SIMDType {
    F32x4,  // 4x f32 for 3D math
    F32x8,  // 8x f32 for ML matrices
    F64x2,  // 2x f64 for precision math
    F64x4,  // 4x f64
    I32x4,  // 4x i32 for indices
    U32x4,  // 4x u32 for graphics
    Boolx4, // Mask operations
}

// Remove #[wasm_bindgen] from the impl block and use static methods instead
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
        }
    }

    pub fn alignment(&self) -> usize {
        match self {
            SIMDType::F32x4 | SIMDType::I32x4 | SIMDType::U32x4 | SIMDType::Boolx4 => 16,
            SIMDType::F32x8 => 32,
            SIMDType::F64x2 => 16,
            SIMDType::F64x4 => 32,
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            SIMDType::F32x4 | SIMDType::F32x8 => 4,
            SIMDType::F64x2 | SIMDType::F64x4 => 8,
            SIMDType::I32x4 | SIMDType::U32x4 => 4,
            SIMDType::Boolx4 => 1,
        }
    }

    pub fn total_size(&self) -> usize {
        self.element_count() * self.element_size()
    }

    pub fn supported_operations(&self) -> Vec<&'static str> {
        match self {
            SIMDType::F32x4 | SIMDType::F32x8 => vec![
                "add", "sub", "mul", "div", "sqrt", "min", "max",
                "dot_product", "cross_product", "normalize", "fma"
            ],
            SIMDType::F64x2 | SIMDType::F64x4 => vec![
                "add", "sub", "mul", "div", "sqrt", "min", "max", "fma"
            ],
            SIMDType::I32x4 | SIMDType::U32x4 => vec![
                "add", "sub", "mul", "and", "or", "xor", "shift_left", "shift_right"
            ],
            SIMDType::Boolx4 => vec!["and", "or", "xor", "not", "select", "any", "all"],
        }
    }

    pub fn is_floating_point(&self) -> bool {
        matches!(self, SIMDType::F32x4 | SIMDType::F32x8 | SIMDType::F64x2 | SIMDType::F64x4)
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, SIMDType::I32x4 | SIMDType::U32x4)
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self, SIMDType::Boolx4)
    }

    pub fn from_string(type_str: &str) -> Result<SIMDType, String> {
        match type_str.to_lowercase().as_str() {
            "f32x4" => Ok(SIMDType::F32x4),
            "f32x8" => Ok(SIMDType::F32x8),
            "f64x2" => Ok(SIMDType::F64x2),
            "f64x4" => Ok(SIMDType::F64x4),
            "i32x4" => Ok(SIMDType::I32x4),
            "u32x4" => Ok(SIMDType::U32x4),
            "boolx4" => Ok(SIMDType::Boolx4),
            _ => Err(format!("Unknown SIMD type: {}", type_str)),
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
        }.to_string()
    }

    pub fn get_optimal_operation(&self, operation: &str) -> String {
        match (self, operation) {
            (SIMDType::F32x4, "dot_product") => "simd_f32x4_dot".to_string(),
            (SIMDType::F32x4, "cross_product") => "simd_f32x4_cross".to_string(),
            (SIMDType::F32x4, "normalize") => "simd_f32x4_normalize".to_string(),
            (SIMDType::F32x8, "matmul") => "simd_f32x8_matmul".to_string(),
            _ => format!("simd_{}_{}", self.to_string(), operation),
        }
    }

    pub fn is_supported(&self) -> bool {
        // Check if the current platform supports this SIMD type
        // This would need actual feature detection
        true // Placeholder
    }

    pub fn lane_count(&self) -> usize {
        self.element_count()
    }

    pub fn is_wide(&self) -> bool {
        matches!(self, SIMDType::F32x8 | SIMDType::F64x4)
    }

    pub fn recommended_use_case(&self) -> &'static str {
        match self {
            SIMDType::F32x4 => "3D graphics, vector math, physics",
            SIMDType::F32x8 => "ML matrices, large data processing",
            SIMDType::F64x2 => "Precision math, scientific computing",
            SIMDType::F64x4 => "High-precision ML, financial calculations",
            SIMDType::I32x4 => "Integer math, indices, game logic",
            SIMDType::U32x4 => "Graphics, bit operations, hashing",
            SIMDType::Boolx4 => "Masking, conditional operations",
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct JsSIMDType(pub SIMDType);

#[wasm_bindgen]
impl JsSIMDType {
    #[wasm_bindgen(constructor)]
    pub fn new(simd_type: SIMDType) -> Self {
        JsSIMDType(simd_type)
    }

    #[wasm_bindgen(getter)]
    pub fn value(&self) -> SIMDType {
        self.0
    }

    #[wasm_bindgen(js_name = elementCount)]
    pub fn element_count(&self) -> usize {
        self.0.element_count()
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }

    #[wasm_bindgen(js_name = alignment)]
    pub fn alignment(&self) -> usize {
        self.0.alignment()
    }

    #[wasm_bindgen(js_name = elementSize)]
    pub fn element_size(&self) -> usize {
        self.0.element_size()
    }

    #[wasm_bindgen(js_name = totalSize)]
    pub fn total_size(&self) -> usize {
        self.0.total_size()
    }

    #[wasm_bindgen(js_name = supportedOperations)]
    pub fn supported_operations(&self) -> js_sys::Array {
        self.0.supported_operations()
            .into_iter()
            .map(JsValue::from)
            .collect()
    }

    #[wasm_bindgen(js_name = isFloatingPoint)]
    pub fn is_floating_point(&self) -> bool {
        self.0.is_floating_point()
    }

    #[wasm_bindgen(js_name = isInteger)]
    pub fn is_integer(&self) -> bool {
        self.0.is_integer()
    }

    #[wasm_bindgen(js_name = isBoolean)]
    pub fn is_boolean(&self) -> bool {
        self.0.is_boolean()
    }

    #[wasm_bindgen(js_name = isSupported)]
    pub fn is_supported(&self) -> bool {
        self.0.is_supported()
    }

    #[wasm_bindgen(js_name = getOptimalOperation)]
    pub fn get_optimal_operation(&self, operation: &str) -> String {
        self.0.get_optimal_operation(operation)
    }

    #[wasm_bindgen(js_name = recommendedUseCase)]
    pub fn recommended_use_case(&self) -> String {
        self.0.recommended_use_case().to_string()
    }
}