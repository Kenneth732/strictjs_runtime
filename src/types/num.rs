
// // src/types/num.rs
use super::HeapType;
use crate::utils::clamp_f64;

#[derive(Clone, Copy, Debug)]
pub enum Num {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Number(f64),
    
    Bool(bool),
    
    Str(u32),        
    Str16(u32),      
    Any(u32),        
    Struct(u32),     
    Array(u32),     
    Map(u32),     
    Date(u32),     
    Buffer(u32),     
    
    Null,
    Undefined,
    Symbol(u32),
    
    // === AI/ML TYPES ===
    TensorF32(u32),
    TensorF64(u32),
    TensorI32(u32),
    TensorU8(u32),
    TensorI8(u32),
    TensorI16(u32),
    TensorU16(u32),
    MatrixF32(u32),
    MatrixF64(u32),
    MatrixC32(u32),
    MatrixC64(u32),
    VectorF32(u32),
    VectorF64(u32),
    VectorI32(u32),
    SparseMatrix(u32),
    Quantized8(u32),
    Quantized16(u32),
    Embedding(u32),
    Attention(u32),
    WeightF32(u32),
    BiasF32(u32),
    GradientF32(u32),
    Activation(u32),
    GPUTensor(u32),
    SIMDVector(u32),
}

impl Num {
    pub fn from_f64(heap: &HeapType, v: f64) -> Self {
        match heap {
            HeapType::Number => Num::Number(v),
            HeapType::U8 => Num::U8(clamp_f64(v, 0.0, u8::MAX as f64) as u8),
            HeapType::I8 => Num::I8(clamp_f64(v, i8::MIN as f64, i8::MAX as f64) as i8),
            HeapType::U16 => Num::U16(clamp_f64(v, 0.0, u16::MAX as f64) as u16),
            HeapType::I16 => Num::I16(clamp_f64(v, i16::MIN as f64, i16::MAX as f64) as i16),
            HeapType::U32 => Num::U32(clamp_f64(v, 0.0, u32::MAX as f64) as u32),
            HeapType::I32 => Num::I32(clamp_f64(v, i32::MIN as f64, i32::MAX as f64) as i32),
            HeapType::U64 => Num::U64(clamp_f64(v, 0.0, u64::MAX as f64) as u64),
            HeapType::I64 => Num::I64(clamp_f64(v, i64::MIN as f64, i64::MAX as f64) as i64),
            HeapType::F32 => Num::F32(v as f32),
            HeapType::F64 => Num::F64(v),
            HeapType::Bool => Num::Bool(v != 0.0),
            
            HeapType::Str => Num::Str(v as u32),
            HeapType::Str16 => Num::Str16(v as u32),
            HeapType::Any => Num::Any(v as u32),
            HeapType::Struct => Num::Struct(v as u32),
            HeapType::Array => Num::Array(v as u32),
            HeapType::Map => Num::Map(v as u32),
            HeapType::Date => Num::Date(v as u32),
            HeapType::Buffer => Num::Buffer(v as u32),
            
            HeapType::Null => Num::Null,
            HeapType::Undefined => Num::Undefined,
            HeapType::Symbol => Num::Symbol(v as u32),
            
            // === AI/ML TYPES ===
            HeapType::TensorF32 => Num::TensorF32(v as u32),
            HeapType::TensorF64 => Num::TensorF64(v as u32),
            HeapType::TensorI32 => Num::TensorI32(v as u32),
            HeapType::TensorU8 => Num::TensorU8(v as u32),
            HeapType::TensorI8 => Num::TensorI8(v as u32),
            HeapType::TensorI16 => Num::TensorI16(v as u32),
            HeapType::TensorU16 => Num::TensorU16(v as u32),
            HeapType::MatrixF32 => Num::MatrixF32(v as u32),
            HeapType::MatrixF64 => Num::MatrixF64(v as u32),
            HeapType::MatrixC32 => Num::MatrixC32(v as u32),
            HeapType::MatrixC64 => Num::MatrixC64(v as u32),
            HeapType::VectorF32 => Num::VectorF32(v as u32),
            HeapType::VectorF64 => Num::VectorF64(v as u32),
            HeapType::VectorI32 => Num::VectorI32(v as u32),
            HeapType::SparseMatrix => Num::SparseMatrix(v as u32),
            HeapType::Quantized8 => Num::Quantized8(v as u32),
            HeapType::Quantized16 => Num::Quantized16(v as u32),
            HeapType::Embedding => Num::Embedding(v as u32),
            HeapType::Attention => Num::Attention(v as u32),
            HeapType::WeightF32 => Num::WeightF32(v as u32),
            HeapType::BiasF32 => Num::BiasF32(v as u32),
            HeapType::GradientF32 => Num::GradientF32(v as u32),
            HeapType::Activation => Num::Activation(v as u32),
            HeapType::GPUTensor => Num::GPUTensor(v as u32),
            HeapType::SIMDVector => Num::SIMDVector(v as u32),
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            // Primitive numeric types
            Num::U8(v) => *v as f64,
            Num::I8(v) => *v as f64,
            Num::U16(v) => *v as f64,
            Num::I16(v) => *v as f64,
            Num::U32(v) => *v as f64,
            Num::I32(v) => *v as f64,
            Num::U64(v) => *v as f64,
            Num::I64(v) => *v as f64,
            Num::F32(v) => *v as f64,
            Num::F64(v) => *v,
            Num::Number(v) => *v,
            
            Num::Bool(v) => if *v { 1.0 } else { 0.0 },
            
            Num::Str(idx) => *idx as f64,
            Num::Str16(idx) => *idx as f64,
            Num::Any(idx) => *idx as f64,
            Num::Struct(idx) => *idx as f64,
            Num::Array(idx) => *idx as f64,
            Num::Map(idx) => *idx as f64,
            Num::Date(idx) => *idx as f64,
            Num::Buffer(idx) => *idx as f64,
            
            Num::Null => 0.0,
            Num::Undefined => 0.0,
            Num::Symbol(idx) => *idx as f64,
            
            // === AI/ML TYPES ===
            Num::TensorF32(idx) => *idx as f64,
            Num::TensorF64(idx) => *idx as f64,
            Num::TensorI32(idx) => *idx as f64,
            Num::TensorU8(idx) => *idx as f64,
            Num::TensorI8(idx) => *idx as f64,
            Num::TensorI16(idx) => *idx as f64,
            Num::TensorU16(idx) => *idx as f64,
            Num::MatrixF32(idx) => *idx as f64,
            Num::MatrixF64(idx) => *idx as f64,
            Num::MatrixC32(idx) => *idx as f64,
            Num::MatrixC64(idx) => *idx as f64,
            Num::VectorF32(idx) => *idx as f64,
            Num::VectorF64(idx) => *idx as f64,
            Num::VectorI32(idx) => *idx as f64,
            Num::SparseMatrix(idx) => *idx as f64,
            Num::Quantized8(idx) => *idx as f64,
            Num::Quantized16(idx) => *idx as f64,
            Num::Embedding(idx) => *idx as f64,
            Num::Attention(idx) => *idx as f64,
            Num::WeightF32(idx) => *idx as f64,
            Num::BiasF32(idx) => *idx as f64,
            Num::GradientF32(idx) => *idx as f64,
            Num::Activation(idx) => *idx as f64,
            Num::GPUTensor(idx) => *idx as f64,
            Num::SIMDVector(idx) => *idx as f64,
        }
    }

    pub fn add_assign(&mut self, heap: &HeapType, delta: f64) {
        if heap.is_numeric() {
            let cur = self.to_f64();
            *self = Num::from_f64(heap, cur + delta);
        }
    }

    pub fn sub_assign(&mut self, heap: &HeapType, delta: f64) {
        if heap.is_numeric() {
            let cur = self.to_f64();
            *self = Num::from_f64(heap, cur - delta);
        }
    }
}






