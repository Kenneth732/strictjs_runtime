# StrictJS Runtime

> **A Type-Safe JavaScript Runtime Built with Rust and WebAssembly**  
> Bringing Rust's memory safety guarantees to JavaScript execution environments.

StrictJS is an experimental JavaScript runtime engineered in Rust and compiled to WebAssembly. It combines **Rust's memory safety** with **JavaScript's dynamic flexibility**, enabling secure and predictable execution across browser and server environments.

## 🚀 Key Features

- **🔒 Type-Safe Numeric Operations** – Automatic clamping and overflow protection with configurable bounds
- **📏 Memory-Bounded Strings** – Configurable character limits for predictable memory usage
- **🧮 Bound-Checked Arrays** – Built-in bounds checking to eliminate out-of-range access errors
- **🏗️ Schema-Enforced Objects** – Runtime type and shape enforcement for object structures
- **⚡ WebAssembly Performance** – Rust-compiled core delivering near-native execution speed
- **🧵 Concurrent Execution** – Thread pool implementation with task scheduling capabilities
- **⚛️ Reactive Programming Model** – Fine-grained reactivity system for state management
- **🧩 Modular Architecture** – Each JavaScript primitive implemented as self-contained Rust module

---

## 📦 Installation

### Browser (CDN)

```html
<script type="module">
  import {
    StrictNumber,
    StrictString,
    StrictArray,
    StrictObject,
    HeapType
  } from 'https://cdn.jsdelivr.net/npm/strictjs-runtime@latest/pkg/strictjs_runtime.js';

  // Example usage
  const safeNumber = new StrictNumber(42, { min: 0, max: 100 });
  console.log(safeNumber.value); // 42
</script>
```

### Node.js / Bundlers

```bash
npm install strictjs-runtime
```

```javascript
import { StrictNumber, StrictString } from 'strictjs-runtime';

const text = new StrictString('Hello, StrictJS!', { maxLength: 256 });
console.log(text.value);
```

---

## 🏗️ Architecture Overview

```
src/
├─ core/
│  ├─ strict_number/     # Type-safe numeric operations
│  ├─ strict_string/     # Bounded string implementation
│  ├─ strict_array/      # Bound-checked array operations
│  ├─ strict_object/     # Schema-based object validation
│  ├─ strict_function/   # Function objects & closure handling
│  ├─ strict_bigint/     # BigInt support with safety checks
│  └─ strict_async/      # Async primitives & event loop
├─ runtime/
│  ├─ loops/             # Control flow execution (for/while)
│  ├─ reactive_system/   # Reactive state management
│  ├─ threads/           # Thread pool & task scheduling
│  └─ types/             # Type definitions & heap management
├─ utils/                # Shared utilities & helpers
├─ tests/                # Comprehensive test suites
├─ error.rs              # Unified error handling
└─ lib.rs               # Primary runtime entry point
```

---

## 🚀 Getting Started

### Prerequisites

- Rust 1.70+ (for development)
- Node.js 16+ (for JavaScript integration)
- wasm-pack (for WebAssembly compilation)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Kenneth732/strictjs_runtime.git
cd strictjs_runtime

# Build the project
cargo build --release

# Run tests
cargo test --all-features

# Build WebAssembly package
wasm-pack build --target web
```

### Example Usage

```javascript
// Creating type-safe primitives
import { StrictNumber, StrictArray, StrictObject } from 'strictjs-runtime';

// Number with bounds
const age = new StrictNumber(25, { min: 0, max: 120 });

// Array with length validation
const ids = new StrictArray([1, 2, 3], { maxLength: 10 });

// Object with schema validation
const userSchema = {
  name: { type: 'string', required: true },
  age: { type: 'number', min: 0 }
};
const user = new StrictObject({ name: 'Alice', age: 30 }, userSchema);
```

---

## 🧪 Testing

The project includes comprehensive test coverage:

```bash
# Run all tests
cargo test

# Run specific test suites
cargo test --test strict_number_tests
cargo test --test strict_array_tests

# Run with verbose output
cargo test -- --nocapture
```

---

## 📊 Performance Benchmarks

Early performance metrics show promising results:

- **Number Operations**: 2-3x faster than vanilla JavaScript with bounds checking
- **String Handling**: Consistent performance with memory safety guarantees
- **Array Access**: Minimal overhead for bounds checking (∼15% slower than native)

See the [benchmarks directory](./benchmarks/) for detailed performance analysis.

---

## 🛣️ Development Roadmap

### Phase 1: Core Primitives (Current)
- [x] Type-safe numbers with clamping
- [x] Bounded string implementation
- [x] Array bounds checking
- [x] Basic object schema validation

### Phase 2: Runtime Features (Q2 2024)
- [ ] Complete ES2023 language support
- [ ] Async/await implementation
- [ ] Garbage collection prototype
- [ ] Basic standard library

### Phase 3: Production Readiness (H2 2024)
- [ ] CLI toolchain
- [ ] Package manager integration
- [ ] Production benchmarking
- [ ] Security audit

---

## 🤝 Contributing

We welcome contributions from the community. Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- Follow Rustfmt formatting guidelines
- Include comprehensive tests for new features
- Document public APIs thoroughly
- Update relevant documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

StrictJS builds upon the work of several outstanding projects:

- **V8 & SpiderMonkey** – For inspiration in JavaScript runtime design
- **Deno & Bun** – Modern runtime architecture patterns
- **Rust WebAssembly Community** – Excellent tooling and resources
- **TC39** – JavaScript language specification guidance

---

## 📞 Support

- **Documentation**: [GitHub Wiki](../../wiki)
- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Email**: [project-kennethmburu@email.com]

---

## 🔗 Related Projects

- [strictjs-compiler](https://github.com/org/strictjs-compiler) – Type-safe JavaScript compiler
- [strictjs-tools](https://github.com/org/strictjs-tools) – Development tooling ecosystem

---
