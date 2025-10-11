
// src/types/heap_type.rs
use js_sys::{Array, ArrayBuffer, Object, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HeapType {
    Number, 
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    Bool,
    Str,  
    Str16, 
    Any,    
    Struct, 
    Array,
    Map,    
    Date,   
    Buffer, 
    Null,
    Undefined,
    Symbol,
    
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeCapabilities(u32);

impl TypeCapabilities {
    // Basic operations
    pub const NUMERIC_OPS: TypeCapabilities = TypeCapabilities(1 << 0); 
    pub const COMPARISON_OPS: TypeCapabilities = TypeCapabilities(1 << 1); 
    pub const BITWISE_OPS: TypeCapabilities = TypeCapabilities(1 << 2); 
    pub const STRING_OPS: TypeCapabilities = TypeCapabilities(1 << 3);  
    
    // Structural operations
    pub const ITERABLE: TypeCapabilities = TypeCapabilities(1 << 4);  
    pub const INDEXABLE: TypeCapabilities = TypeCapabilities(1 << 5);  
    pub const KEYABLE: TypeCapabilities = TypeCapabilities(1 << 6);  
    
    // Function-like operations
    pub const CALLABLE: TypeCapabilities = TypeCapabilities(1 << 7); 
    pub const CONSTRUCTABLE: TypeCapabilities = TypeCapabilities(1 << 8);  
    pub const AWAITABLE: TypeCapabilities = TypeCapabilities(1 << 9);  
    pub const GENERATOR: TypeCapabilities = TypeCapabilities(1 << 10); 
    
    // Data management
    pub const SERIALIZABLE: TypeCapabilities = TypeCapabilities(1 << 11);  
    pub const CLONEABLE: TypeCapabilities = TypeCapabilities(1 << 12); 
    pub const TRANSFERABLE: TypeCapabilities = TypeCapabilities(1 << 13); 
    pub const DESTRUCTURABLE: TypeCapabilities = TypeCapabilities(1 << 14); 
    
    // Advanced operations
    pub const TYPED_ARRAY: TypeCapabilities = TypeCapabilities(1 << 15); 
    pub const ARRAY_BUFFER: TypeCapabilities = TypeCapabilities(1 << 16); 
    pub const PROMISE: TypeCapabilities = TypeCapabilities(1 << 17);
    pub const PROXY: TypeCapabilities = TypeCapabilities(1 << 18); 
    
    // Memory and performance
    pub const INLINE_OPTIMIZABLE: TypeCapabilities = TypeCapabilities(1 << 19); 
    pub const POOLABLE: TypeCapabilities = TypeCapabilities(1 << 20); 
    pub const LAZY_LOADABLE: TypeCapabilities = TypeCapabilities(1 << 21); 
    pub const ALL: TypeCapabilities = TypeCapabilities(u32::MAX);
    
    pub fn all_numeric() -> TypeCapabilities {
        TypeCapabilities(Self::NUMERIC_OPS.0 | Self::COMPARISON_OPS.0 | Self::SERIALIZABLE.0 | Self::CLONEABLE.0 | Self::TRANSFERABLE.0)
    }
    
    pub fn all_string() -> TypeCapabilities {
        TypeCapabilities(Self::STRING_OPS.0 | Self::COMPARISON_OPS.0 | Self::ITERABLE.0 | Self::SERIALIZABLE.0 | Self::CLONEABLE.0 | Self::TRANSFERABLE.0)
    }
    
    pub fn all_collection() -> TypeCapabilities {
        TypeCapabilities(Self::ITERABLE.0 | Self::INDEXABLE.0 | Self::KEYABLE.0 | Self::SERIALIZABLE.0 | Self::CLONEABLE.0 | Self::DESTRUCTURABLE.0)
    }

    #[inline]
    pub fn contains(self, other: TypeCapabilities) -> bool {
        (self.0 & other.0) != 0
    }

    #[inline]
    pub fn union(self, other: TypeCapabilities) -> TypeCapabilities {
        TypeCapabilities(self.0 | other.0)
    }

    
    
    #[inline]
    pub fn intersection(self, other: TypeCapabilities) -> TypeCapabilities {
        TypeCapabilities(self.0 & other.0)
    }
    
    #[inline]
    pub fn without(self, other: TypeCapabilities) -> TypeCapabilities {
        TypeCapabilities(self.0 & !other.0)
    }
    
    #[inline]
    pub fn with(self, other: TypeCapabilities) -> TypeCapabilities {
        self.union(other)
    }
    
    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
    
    #[inline]
    pub fn bits(&self) -> u32 {
        self.0
    }
    
    #[inline]
    pub fn from_bits(bits: u32) -> Option<TypeCapabilities> {
        Some(TypeCapabilities(bits))
    }
    
    #[inline]
    pub fn supports_all(self, other: TypeCapabilities) -> bool {
        (self.0 & other.0) == other.0
    }
    
    #[inline]
    pub fn supports_any(self, other: TypeCapabilities) -> bool {
        (self.0 & other.0) != 0
    }
    
    // Get the difference between two capability sets
    #[inline]
    pub fn difference(self, other: TypeCapabilities) -> TypeCapabilities {
        TypeCapabilities(self.0 & !other.0)
    }
    
    #[inline]
    pub fn equals(self, other: TypeCapabilities) -> bool {
        self.0 == other.0
    }
    
    #[inline]
    pub fn count(self) -> u32 {
        self.0.count_ones()
    }
    
    #[inline]
    pub fn is_subset_of(self, other: TypeCapabilities) -> bool {
        (self.0 & other.0) == self.0
    }
    
    #[inline]
    pub fn is_superset_of(self, other: TypeCapabilities) -> bool {
        (self.0 & other.0) == other.0
    }
    
    pub fn to_flags(self) -> Vec<TypeCapabilities> {
        let mut flags = Vec::new();
        let all_flags = [
            Self::NUMERIC_OPS,
            Self::COMPARISON_OPS,
            Self::BITWISE_OPS,
            Self::STRING_OPS,
            Self::ITERABLE,
            Self::INDEXABLE,
            Self::KEYABLE,
            Self::CALLABLE,
            Self::CONSTRUCTABLE,
            Self::AWAITABLE,
            Self::GENERATOR,
            Self::SERIALIZABLE,
            Self::CLONEABLE,
            Self::TRANSFERABLE,
            Self::DESTRUCTURABLE,
            Self::TYPED_ARRAY,
            Self::ARRAY_BUFFER,
            Self::PROMISE,
            Self::PROXY,
            Self::INLINE_OPTIMIZABLE,
            Self::POOLABLE,
            Self::LAZY_LOADABLE,
        ];
        
        for flag in all_flags.iter() {
            if self.contains(*flag) {
                flags.push(*flag);
            }
        }
        
        flags
    }
    
    pub fn to_names(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        
        if self.contains(Self::NUMERIC_OPS) { names.push("numeric"); }
        if self.contains(Self::COMPARISON_OPS) { names.push("comparison"); }
        if self.contains(Self::BITWISE_OPS) { names.push("bitwise"); }
        if self.contains(Self::STRING_OPS) { names.push("string"); }
        if self.contains(Self::ITERABLE) { names.push("iterable"); }
        if self.contains(Self::INDEXABLE) { names.push("indexable"); }
        if self.contains(Self::KEYABLE) { names.push("keyable"); }
        if self.contains(Self::CALLABLE) { names.push("callable"); }
        if self.contains(Self::CONSTRUCTABLE) { names.push("constructable"); }
        if self.contains(Self::AWAITABLE) { names.push("awaitable"); }
        if self.contains(Self::GENERATOR) { names.push("generator"); }
        if self.contains(Self::SERIALIZABLE) { names.push("serializable"); }
        if self.contains(Self::CLONEABLE) { names.push("cloneable"); }
        if self.contains(Self::TRANSFERABLE) { names.push("transferable"); }
        if self.contains(Self::DESTRUCTURABLE) { names.push("destructurable"); }
        if self.contains(Self::TYPED_ARRAY) { names.push("typed_array"); }
        if self.contains(Self::ARRAY_BUFFER) { names.push("array_buffer"); }
        if self.contains(Self::PROMISE) { names.push("promise"); }
        if self.contains(Self::PROXY) { names.push("proxy"); }
        if self.contains(Self::INLINE_OPTIMIZABLE) { names.push("inline_optimizable"); }
        if self.contains(Self::POOLABLE) { names.push("poolable"); }
        if self.contains(Self::LAZY_LOADABLE) { names.push("lazy_loadable"); }
        
        names
    }
}


#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct JsTypeCapabilities(TypeCapabilities);


#[wasm_bindgen]
impl JsTypeCapabilities {
    #[wasm_bindgen(constructor)]
    pub fn new(bits: u32) -> Self {
        JsTypeCapabilities(TypeCapabilities::from_bits(bits).unwrap_or(TypeCapabilities(0)))
    }

    #[wasm_bindgen(js_name = getBits)]
    pub fn get_bits(&self) -> u32 {
        self.0.bits()
    }

    #[wasm_bindgen(js_name = contains)]
    pub fn contains(&self, other: &JsTypeCapabilities) -> bool {
        self.0.contains(other.0)
    }
    
    #[wasm_bindgen(js_name = union)]
    pub fn union(&self, other: &JsTypeCapabilities) -> JsTypeCapabilities {
        JsTypeCapabilities(self.0.union(other.0))
    }
    
    #[wasm_bindgen(js_name = intersection)]
    pub fn intersection(&self, other: &JsTypeCapabilities) -> JsTypeCapabilities {
        JsTypeCapabilities(self.0.intersection(other.0))
    }
    
    #[wasm_bindgen(js_name = without)]
    pub fn without(&self, other: &JsTypeCapabilities) -> JsTypeCapabilities {
        JsTypeCapabilities(self.0.without(other.0))
    }
    
    #[wasm_bindgen(js_name = isSubsetOf)]
    pub fn is_subset_of(&self, other: &JsTypeCapabilities) -> bool {
        self.0.is_subset_of(other.0)
    }
    
    #[wasm_bindgen(js_name = isSupersetOf)]
    pub fn is_superset_of(&self, other: &JsTypeCapabilities) -> bool {
        self.0.is_superset_of(other.0)
    }
    
    #[wasm_bindgen(js_name = equals)]
    pub fn equals(&self, other: &JsTypeCapabilities) -> bool {
        self.0.equals(other.0)
    }
    
    #[wasm_bindgen(js_name = count)]
    pub fn count(&self) -> u32 {
        self.0.count()
    }
    
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    
    #[wasm_bindgen(js_name = bits)]
    pub fn bits(&self) -> u32 {
        self.0.bits()
    }
    
    #[wasm_bindgen(js_name = toNames)]
    pub fn to_names(&self) -> Array {
        self.0.to_names()
            .into_iter()
            .map(JsValue::from)
            .collect()
    }
    
    #[wasm_bindgen(js_name = supportsAll)]
    pub fn supports_all(&self, other: &JsTypeCapabilities) -> bool {
        self.0.supports_all(other.0)
    }
    
    #[wasm_bindgen(js_name = supportsAny)]
    pub fn supports_any(&self, other: &JsTypeCapabilities) -> bool {
        self.0.supports_any(other.0)
    }
}


#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct JsHeapType(pub HeapType); 

#[wasm_bindgen]
impl JsHeapType {
    #[wasm_bindgen(constructor)]
    pub fn new(heap_type: HeapType) -> Self {
        JsHeapType(heap_type)
    }
    
    #[wasm_bindgen(getter)]
    pub fn element_size(&self) -> usize {
        self.0.element_size()
    }

    #[wasm_bindgen(js_name = getHeapType)]
    pub fn get_heap_type(&self) -> HeapType {
        self.0
    }

    #[wasm_bindgen(getter, js_name = isPrimitive)]
    pub fn is_primitive(&self) -> bool {
        self.0.is_primitive()
    }

    #[wasm_bindgen(getter, js_name = isComplex)]
    pub fn is_complex(&self) -> bool {
        self.0.is_complex()
    }

    #[wasm_bindgen(getter, js_name = isNumeric)]
    pub fn is_numeric(&self) -> bool {
        self.0.is_numeric()
    }

    #[wasm_bindgen(js_name = fromString)]
    pub fn from_string(type_str: &str) -> Result<JsHeapType, JsValue> {
        HeapType::from_string(type_str).map(JsHeapType)
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }

    #[wasm_bindgen(js_name = fromJSValue)]
    pub fn from_js_value(js_value: JsValue) -> Result<JsHeapType, JsValue> {
        HeapType::from_js_value(js_value).map(JsHeapType)
    }

    #[wasm_bindgen(js_name = getDefaultValue)]
    pub fn get_default_value(&self) -> f64 {
        self.0.get_default_value()
    }

    #[wasm_bindgen(js_name = canStoreInArray)]
    pub fn can_store_in_array(&self) -> bool {
        self.0.can_store_in_array()
    }

    #[wasm_bindgen(js_name = alignment)]
    pub fn alignment(&self) -> usize {
        self.0.alignment()
    }

    #[wasm_bindgen(js_name = toJSValue)]
    pub fn to_js_value(&self) -> JsValue {
        self.0.to_js_value()
    }

    #[wasm_bindgen(js_name = supportsOperation)]
    pub fn supports_operation(&self, operation: &str) -> bool {
        self.0.supports_operation(operation)
    }

    #[wasm_bindgen(js_name = getCapabilities)]
    pub fn get_capabilities(&self) -> JsValue {
        self.0.get_capabilities_js()
    }
    
    #[wasm_bindgen(js_name = getCapabilitiesObject)]
    pub fn get_capabilities_object(&self) -> JsTypeCapabilities {
        JsTypeCapabilities(self.0.capabilities())
    }

    #[wasm_bindgen(js_name = supportsAllCapabilities)]
    pub fn supports_all_capabilities(&self, capabilities: &JsTypeCapabilities) -> bool {
        self.0.capabilities().supports_all(capabilities.0)
    }

    #[wasm_bindgen(js_name = supportsAnyCapability)]
    pub fn supports_any_capability(&self, capabilities: &JsTypeCapabilities) -> bool {
        self.0.capabilities().supports_any(capabilities.0)
    }

    #[wasm_bindgen(js_name = getCompatibleOperationsWith)]
    pub fn get_compatible_operations_with(&self, other: &JsHeapType) -> Array {
        self.0.get_compatible_operations_with(&other.0)
            .into_iter()
            .map(JsValue::from)
            .collect()
    }

    #[wasm_bindgen(js_name = recommendedOperations)]
    pub fn recommended_operations(&self) -> Array {
        self.0.recommended_operations()
            .into_iter()
            .map(JsValue::from)
            .collect()
    }

    #[wasm_bindgen(js_name = isBinaryCompatibleWith)]
    pub fn is_binary_compatible_with(&self, other: &JsHeapType) -> bool {
        self.0.is_binary_compatible_with(&other.0)
    }

    #[wasm_bindgen(js_name = getBinaryResultType)]
    pub fn get_binary_result_type(&self, other: &JsHeapType) -> Option<JsHeapType> {
        self.0.get_binary_result_type(&other.0).map(JsHeapType)
    }

    #[wasm_bindgen(js_name = isStringType)]
    pub fn is_string_type(&self) -> bool {
        self.0.is_string_type()
    }

    #[wasm_bindgen(js_name = isContainerType)]
    pub fn is_container_type(&self) -> bool {
        self.0.is_container_type()
    }

    // Check if this type requires managed memory
    #[wasm_bindgen(js_name = requiresManagedMemory)]
    pub fn requires_managed_memory(&self) -> bool {
        self.0.requires_managed_memory()
    }

    // Convert from type string (alternative to fromString with different name)
    #[wasm_bindgen(js_name = fromTypeStr)]
    pub fn from_type_str(type_str: &str) -> Option<JsHeapType> {
        HeapType::from_type_str(type_str).map(JsHeapType)
    }

    // Get the underlying HeapType enum value as string for debugging
    #[wasm_bindgen(js_name = typeName)]
    pub fn type_name(&self) -> String {
        format!("{:?}", self.0)
    }
    
    // NEW: Get capability score (number of capabilities this type has)
    #[wasm_bindgen(js_name = capabilityScore)]
    pub fn capability_score(&self) -> u32 {
        self.0.capabilities().count()
    }
    
    // NEW: Check if this type is more capable than another type
    #[wasm_bindgen(js_name = isMoreCapableThan)]
    pub fn is_more_capable_than(&self, other: &JsHeapType) -> bool {
        self.0.capabilities().count() > other.0.capabilities().count()
    }
}

impl HeapType {
    // Enhanced capabilities system with more granular control
    pub fn capabilities(&self) -> TypeCapabilities {
        match self {
            // Numeric types support numeric and comparison operations
            HeapType::Number | HeapType::U8 | HeapType::I8 | HeapType::U16 | 
            HeapType::I16 | HeapType::U32 | HeapType::I32 | HeapType::U64 | 
            HeapType::I64 | HeapType::F32 | HeapType::F64 => {
                TypeCapabilities::all_numeric()
                    .union(TypeCapabilities::INLINE_OPTIMIZABLE)
                    .union(TypeCapabilities::POOLABLE)
            }
            
            // Bool supports comparison and bitwise operations
            HeapType::Bool => {
                TypeCapabilities::COMPARISON_OPS
                    .union(TypeCapabilities::BITWISE_OPS)
                    .union(TypeCapabilities::SERIALIZABLE)
                    .union(TypeCapabilities::CLONEABLE)
                    .union(TypeCapabilities::TRANSFERABLE)
                    .union(TypeCapabilities::INLINE_OPTIMIZABLE)
            }
            
            // String types support string operations, comparison, and iteration
            HeapType::Str | HeapType::Str16 => {
                TypeCapabilities::all_string()
                    .union(TypeCapabilities::LAZY_LOADABLE)
            }
            
            // Array supports iteration, indexing, and serialization
            HeapType::Array => {
                TypeCapabilities::all_collection()
                    .union(TypeCapabilities::DESTRUCTURABLE)
                    .union(TypeCapabilities::POOLABLE)
            }
            
            // Map supports iteration and indexing
            HeapType::Map => {
                TypeCapabilities::all_collection()
                    .union(TypeCapabilities::DESTRUCTURABLE)
            }
            
            // Struct supports indexing and serialization
            HeapType::Struct => {
                TypeCapabilities::INDEXABLE
                    .union(TypeCapabilities::KEYABLE)
                    .union(TypeCapabilities::SERIALIZABLE)
                    .union(TypeCapabilities::CLONEABLE)
                    .union(TypeCapabilities::DESTRUCTURABLE)
            }
            
            // Date supports comparison and serialization
            HeapType::Date => {
                TypeCapabilities::COMPARISON_OPS
                    .union(TypeCapabilities::SERIALIZABLE)
                    .union(TypeCapabilities::CLONEABLE)
            }
            
            // Buffer supports transfer and some operations
            HeapType::Buffer => {
                TypeCapabilities::ITERABLE
                    .union(TypeCapabilities::INDEXABLE)
                    .union(TypeCapabilities::CLONEABLE)
                    .union(TypeCapabilities::TRANSFERABLE)
                    .union(TypeCapabilities::TYPED_ARRAY)
                    .union(TypeCapabilities::ARRAY_BUFFER)
                    .union(TypeCapabilities::POOLABLE)
            }
            
            // Any type is very flexible but has limitations
            HeapType::Any => {
                TypeCapabilities::SERIALIZABLE
                    .union(TypeCapabilities::CLONEABLE)
                    .union(TypeCapabilities::DESTRUCTURABLE)
            }
            
            // Special types have very limited capabilities
            HeapType::Null | HeapType::Undefined => {
                TypeCapabilities::SERIALIZABLE
            }
            
            HeapType::Symbol => {
                TypeCapabilities::CLONEABLE
            }
        }
    }

    pub fn supports_operation(&self, operation: &str) -> bool {
        let caps = self.capabilities();
        
        match operation {
            // Numeric operations
            "add" | "subtract" | "multiply" | "divide" | "modulo" | "power" => 
                caps.contains(TypeCapabilities::NUMERIC_OPS),
            
            // Comparison operations  
            "compare" | "equal" | "notEqual" | "lessThan" | "greaterThan" | 
            "lessThanOrEqual" | "greaterThanOrEqual" => 
                caps.contains(TypeCapabilities::COMPARISON_OPS),
            
            // Bitwise operations
            "bitwiseAnd" | "bitwiseOr" | "bitwiseXor" | "bitwiseNot" | 
            "leftShift" | "rightShift" | "unsignedRightShift" => 
                caps.contains(TypeCapabilities::BITWISE_OPS),
            
            // String operations
            "concat" | "slice" | "substring" | "toUpperCase" | "toLowerCase" |
            "trim" | "trimStart" | "trimEnd" | "padStart" | "padEnd" |
            "startsWith" | "endsWith" | "includes" | "indexOf" | "lastIndexOf" => 
                caps.contains(TypeCapabilities::STRING_OPS),
            
            // Iteration operations
            "iterate" | "forEach" | "map" | "filter" | "reduce" | "find" | 
            "some" | "every" | "entries" | "keys" | "values" => 
                caps.contains(TypeCapabilities::ITERABLE),
            
            // Indexing operations
            "index" | "get" | "set" | "delete" | "has" => 
                caps.contains(TypeCapabilities::INDEXABLE),
            
            // Function operations
            "call" | "apply" | "bind" => 
                caps.contains(TypeCapabilities::CALLABLE),
            
            // Construction operations
            "construct" | "new" => 
                caps.contains(TypeCapabilities::CONSTRUCTABLE),
            
            // Async operations
            "await" | "then" | "catch" | "finally" => 
                caps.contains(TypeCapabilities::AWAITABLE),
            
            // Generator operations
            "yield" | "next" | "return" | "throw" => 
                caps.contains(TypeCapabilities::GENERATOR),
            
            // Serialization operations
            "stringify" | "parse" | "toJSON" => 
                caps.contains(TypeCapabilities::SERIALIZABLE),
            
            // Cloning operations
            "clone" | "copy" | "deepClone" => 
                caps.contains(TypeCapabilities::CLONEABLE),
            
            // Transfer operations
            "transfer" | "postMessage" | "structuredClone" => 
                caps.contains(TypeCapabilities::TRANSFERABLE),
            
            // Destructuring operations
            "destructure" | "spread" => 
                caps.contains(TypeCapabilities::DESTRUCTURABLE),
            
            // Buffer operations
            "sliceBuffer" | "subarray" | "setValues" => 
                caps.contains(TypeCapabilities::TYPED_ARRAY),
            
            _ => false,
        }
    }

    pub fn get_compatible_operations_with(&self, other: &HeapType) -> Vec<&'static str> {
        let self_caps = self.capabilities();
        let other_caps = other.capabilities();
        let common_caps = self_caps.intersection(other_caps);
        
        let mut operations = Vec::new();
        
        if common_caps.contains(TypeCapabilities::NUMERIC_OPS) {
            operations.extend(&["add", "subtract", "multiply", "divide"]);
        }
        if common_caps.contains(TypeCapabilities::COMPARISON_OPS) {
            operations.extend(&["equal", "notEqual", "lessThan", "greaterThan"]);
        }
        if common_caps.contains(TypeCapabilities::BITWISE_OPS) {
            operations.extend(&["bitwiseAnd", "bitwiseOr", "bitwiseXor"]);
        }
        if common_caps.contains(TypeCapabilities::STRING_OPS) {
            operations.extend(&["concat"]);
        }
        if common_caps.contains(TypeCapabilities::ITERABLE) {
            operations.extend(&["forEach", "map", "filter"]);
        }
        
        operations
    }

    pub fn get_capabilities_js(&self) -> JsValue {
        let caps = self.capabilities();
        let obj = Object::new();
        
        let set_bool = |key: &str, value: bool| {
            js_sys::Reflect::set(&obj, &JsValue::from_str(key), &JsValue::from_bool(value)).unwrap();
        };
        
        set_bool("numeric", caps.contains(TypeCapabilities::NUMERIC_OPS));
        set_bool("comparison", caps.contains(TypeCapabilities::COMPARISON_OPS));
        set_bool("bitwise", caps.contains(TypeCapabilities::BITWISE_OPS));
        set_bool("string", caps.contains(TypeCapabilities::STRING_OPS));
        set_bool("iterable", caps.contains(TypeCapabilities::ITERABLE));
        set_bool("indexable", caps.contains(TypeCapabilities::INDEXABLE));
        set_bool("keyable", caps.contains(TypeCapabilities::KEYABLE));
        set_bool("callable", caps.contains(TypeCapabilities::CALLABLE));
        set_bool("constructable", caps.contains(TypeCapabilities::CONSTRUCTABLE));
        set_bool("awaitable", caps.contains(TypeCapabilities::AWAITABLE));
        set_bool("generator", caps.contains(TypeCapabilities::GENERATOR));
        set_bool("serializable", caps.contains(TypeCapabilities::SERIALIZABLE));
        set_bool("cloneable", caps.contains(TypeCapabilities::CLONEABLE));
        set_bool("transferable", caps.contains(TypeCapabilities::TRANSFERABLE));
        set_bool("destructurable", caps.contains(TypeCapabilities::DESTRUCTURABLE));
        set_bool("typedArray", caps.contains(TypeCapabilities::TYPED_ARRAY));
        set_bool("arrayBuffer", caps.contains(TypeCapabilities::ARRAY_BUFFER));
        set_bool("promise", caps.contains(TypeCapabilities::PROMISE));
        set_bool("proxy", caps.contains(TypeCapabilities::PROXY));
        set_bool("inlineOptimizable", caps.contains(TypeCapabilities::INLINE_OPTIMIZABLE));
        set_bool("poolable", caps.contains(TypeCapabilities::POOLABLE));
        set_bool("lazyLoadable", caps.contains(TypeCapabilities::LAZY_LOADABLE));
        
        obj.into()
    }

    pub fn recommended_operations(&self) -> Vec<&'static str> {
        let mut ops = Vec::new();
        let caps = self.capabilities();
        
        if caps.contains(TypeCapabilities::NUMERIC_OPS) {
            ops.extend(&["add", "subtract", "multiply", "divide"]);
        }
        if caps.contains(TypeCapabilities::COMPARISON_OPS) {
            ops.extend(&["equal", "notEqual", "lessThan", "greaterThan"]);
        }
        if caps.contains(TypeCapabilities::BITWISE_OPS) {
            ops.extend(&["bitwiseAnd", "bitwiseOr", "bitwiseXor"]);
        }
        if caps.contains(TypeCapabilities::STRING_OPS) {
            ops.extend(&["concat", "slice", "substring"]);
        }
        if caps.contains(TypeCapabilities::ITERABLE) {
            ops.extend(&["forEach", "map", "filter", "reduce"]);
        }
        if caps.contains(TypeCapabilities::INDEXABLE) {
            ops.extend(&["get", "set", "has"]);
        }
        
        ops
    }

    pub fn is_binary_compatible_with(&self, other: &HeapType) -> bool {
        // Both must support numeric operations
        self.supports_operation("add") && other.supports_operation("add")
    }

    pub fn get_binary_result_type(&self, other: &HeapType) -> Option<HeapType> {
        if !self.is_binary_compatible_with(other) {
            return None;
        }
        
        match (self, other) {
            // Numeric type promotions
            (HeapType::F64, _) | (_, HeapType::F64) => Some(HeapType::F64),
            (HeapType::F32, _) | (_, HeapType::F32) => Some(HeapType::F32),
            (HeapType::U64, _) | (_, HeapType::U64) => Some(HeapType::U64),
            (HeapType::I64, _) | (_, HeapType::I64) => Some(HeapType::I64),
            (HeapType::U32, _) | (_, HeapType::U32) => Some(HeapType::U32),
            (HeapType::I32, _) | (_, HeapType::I32) => Some(HeapType::I32),
            (HeapType::U16, _) | (_, HeapType::U16) => Some(HeapType::U16),
            (HeapType::I16, _) | (_, HeapType::I16) => Some(HeapType::I16),
            (HeapType::U8, _) | (_, HeapType::U8) => Some(HeapType::U8),
            (HeapType::I8, _) | (_, HeapType::I8) => Some(HeapType::I8),
            (HeapType::Number, _) | (_, HeapType::Number) => Some(HeapType::Number),
            _ => None,
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            HeapType::U8 | HeapType::I8 | HeapType::Bool => 1,
            HeapType::U16 | HeapType::I16 | HeapType::F32 => 2,
            HeapType::U32 | HeapType::I32 => 4,
            HeapType::U64 | HeapType::I64 | HeapType::F64 | HeapType::Number => 8,
            HeapType::Str
            | HeapType::Str16
            | HeapType::Any
            | HeapType::Struct
            | HeapType::Array
            | HeapType::Map
            | HeapType::Date
            | HeapType::Buffer => 8,
            HeapType::Null | HeapType::Undefined | HeapType::Symbol => 0,
        }
    }

    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            HeapType::U8
                | HeapType::I8
                | HeapType::U16
                | HeapType::I16
                | HeapType::U32
                | HeapType::I32
                | HeapType::U64
                | HeapType::I64
                | HeapType::F32
                | HeapType::F64
                | HeapType::Number
                | HeapType::Bool
        )
    }

    pub fn is_complex(&self) -> bool {
        matches!(
            self,
            HeapType::Str
                | HeapType::Str16
                | HeapType::Struct
                | HeapType::Array
                | HeapType::Map
                | HeapType::Date
                | HeapType::Buffer
        )
    }

    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            HeapType::U8
                | HeapType::I8
                | HeapType::U16
                | HeapType::I16
                | HeapType::U32
                | HeapType::I32
                | HeapType::U64
                | HeapType::I64
                | HeapType::F32
                | HeapType::F64
                | HeapType::Number
        )
    }

    pub fn from_string(type_str: &str) -> Result<HeapType, JsValue> {
        match type_str.to_lowercase().as_str() {
            "number" => Ok(HeapType::Number),
            "u8" => Ok(HeapType::U8),
            "i8" => Ok(HeapType::I8),
            "u16" => Ok(HeapType::U16),
            "i16" => Ok(HeapType::I16),
            "u32" => Ok(HeapType::U32),
            "i32" => Ok(HeapType::I32),
            "u64" => Ok(HeapType::U64),
            "i64" => Ok(HeapType::I64),
            "f32" => Ok(HeapType::F32),
            "f64" => Ok(HeapType::F64),
            "bool" => Ok(HeapType::Bool),
            "string" => Ok(HeapType::Str),
            "string16" => Ok(HeapType::Str16),
            "any" => Ok(HeapType::Any),
            "struct" => Ok(HeapType::Struct),
            "array" => Ok(HeapType::Array),
            "map" => Ok(HeapType::Map),
            "date" => Ok(HeapType::Date),
            "buffer" => Ok(HeapType::Buffer),
            "null" => Ok(HeapType::Null),
            "undefined" => Ok(HeapType::Undefined),
            "symbol" => Ok(HeapType::Symbol),
            _ => Err(JsValue::from_str(&format!("Unknown type: {}", type_str))),
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            HeapType::Number => "number",
            HeapType::U8 => "u8",
            HeapType::I8 => "i8",
            HeapType::U16 => "u16",
            HeapType::I16 => "i16",
            HeapType::U32 => "u32",
            HeapType::I32 => "i32",
            HeapType::U64 => "u64",
            HeapType::I64 => "i64",
            HeapType::F32 => "f32",
            HeapType::F64 => "f64",
            HeapType::Bool => "bool",
            HeapType::Str => "string",
            HeapType::Str16 => "string16",
            HeapType::Any => "any",
            HeapType::Struct => "struct",
            HeapType::Array => "array",
            HeapType::Map => "map",
            HeapType::Date => "date",
            HeapType::Buffer => "buffer",
            HeapType::Null => "null",
            HeapType::Undefined => "undefined",
            HeapType::Symbol => "symbol",
        }
        .to_string()
    }

    pub fn to_js_value(&self) -> JsValue {
        match self {
            HeapType::Number => JsValue::from_str("number"),
            HeapType::U8 => JsValue::from_str("u8"),
            HeapType::I8 => JsValue::from_str("i8"),
            HeapType::U16 => JsValue::from_str("u16"),
            HeapType::I16 => JsValue::from_str("i16"),
            HeapType::U32 => JsValue::from_str("u32"),
            HeapType::I32 => JsValue::from_str("i32"),
            HeapType::U64 => JsValue::from_str("u64"),
            HeapType::I64 => JsValue::from_str("i64"),
            HeapType::F32 => JsValue::from_str("f32"),
            HeapType::F64 => JsValue::from_str("f64"),
            HeapType::Bool => JsValue::from_str("bool"),
            HeapType::Str => JsValue::from_str("string"),
            HeapType::Str16 => JsValue::from_str("string16"),
            HeapType::Any => JsValue::from_str("any"),
            HeapType::Struct => JsValue::from_str("struct"),
            HeapType::Array => JsValue::from_str("array"),
            HeapType::Map => JsValue::from_str("map"),
            HeapType::Date => JsValue::from_str("date"),
            HeapType::Buffer => JsValue::from_str("buffer"),
            HeapType::Null => JsValue::from_str("null"),
            HeapType::Undefined => JsValue::from_str("undefined"),
            HeapType::Symbol => JsValue::from_str("symbol"),
        }
    }

    pub fn from_js_value(js_value: JsValue) -> Result<HeapType, JsValue> {
        if js_value.is_null() {
            return Ok(HeapType::Null);
        }
        if js_value.is_undefined() {
            return Ok(HeapType::Undefined);
        }
        if let Some(_) = js_value.as_string() {
            return Ok(HeapType::Str);
        }
        if let Some(_) = js_value.as_f64() {
            return Ok(HeapType::Number);
        }
        if let Some(_) = js_value.as_bool() {
            return Ok(HeapType::Bool);
        }
        if js_value.is_object() {
            let obj = js_value.dyn_into::<Object>()?;
            if Array::is_array(&obj) {
                return Ok(HeapType::Array);
            }
           
            if obj.has_own_property(&JsValue::from_str("getTime")) {
                return Ok(HeapType::Date);
            }
            if obj.is_instance_of::<Uint8Array>() || obj.is_instance_of::<ArrayBuffer>() {
                return Ok(HeapType::Buffer);
            }
            return Ok(HeapType::Struct);
        }
        if js_value.is_symbol() {
            return Ok(HeapType::Symbol);
        }

        Err(JsValue::from_str("Unsupported JavaScript value type"))
    }

    pub fn get_default_value(&self) -> f64 {
        match self {
            HeapType::U8 | HeapType::U16 | HeapType::U32 | HeapType::U64 => 0.0,
            HeapType::I8 | HeapType::I16 | HeapType::I32 | HeapType::I64 => 0.0,
            HeapType::F32 | HeapType::F64 | HeapType::Number => 0.0,
            HeapType::Bool => 0.0,
            HeapType::Str
            | HeapType::Str16
            | HeapType::Any
            | HeapType::Struct
            | HeapType::Array
            | HeapType::Map
            | HeapType::Date
            | HeapType::Buffer => 0.0, 
            HeapType::Null | HeapType::Undefined | HeapType::Symbol => 0.0,
        }
    }

    pub fn can_store_in_array(&self) -> bool {
        !matches!(
            self,
            HeapType::Null | HeapType::Undefined | HeapType::Symbol
        )
    }

    pub fn alignment(&self) -> usize {
        match self {
            HeapType::U8 | HeapType::I8 | HeapType::Bool => 1,
            HeapType::U16 | HeapType::I16 | HeapType::F32 => 2,
            HeapType::U32 | HeapType::I32 => 4,
            HeapType::U64
            | HeapType::I64
            | HeapType::F64
            | HeapType::Number
            | HeapType::Str
            | HeapType::Str16
            | HeapType::Any
            | HeapType::Struct
            | HeapType::Array
            | HeapType::Map
            | HeapType::Date
            | HeapType::Buffer => 8,
            HeapType::Null | HeapType::Undefined | HeapType::Symbol => 1,
        }
    }

    pub fn from_type_str(type_str: &str) -> Option<HeapType> {
        HeapType::from_string(type_str).ok()
    }

    pub fn is_string_type(&self) -> bool {
        matches!(self, HeapType::Str | HeapType::Str16)
    }

    pub fn is_container_type(&self) -> bool {
        matches!(self, HeapType::Array | HeapType::Struct | HeapType::Map)
    }

    pub fn requires_managed_memory(&self) -> bool {
        self.is_complex() || matches!(self, HeapType::Any)
    }
}
