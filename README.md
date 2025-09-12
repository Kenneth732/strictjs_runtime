# StrictJS Runtime

> **Type-safe JavaScript runtime with WebAssembly**  
> Bringing Rust’s safety guarantees to JavaScript.

StrictJS is an experimental JavaScript runtime written in Rust and compiled to WebAssembly.  
It combines **Rust’s memory safety** with **JavaScript’s flexibility**, enabling secure and predictable execution both in the browser and on the server.

---

## ✨ Features

- 🔒 **Type-safe Numbers** – automatic clamping and overflow protection.
- 📏 **Bounded Strings** – configurable character limits for predictable memory usage.
- 🧮 **Safe Arrays** – built-in bounds checking to eliminate out-of-range errors.
- 🏗️ **Schema-based Objects** – enforce object shapes and types at runtime.
- ⚡ **WebAssembly Performance** – compiled Rust core delivers near-native speed.
- 🧵 **Thread Pool & Reactive System** – modern concurrency and fine-grained reactivity.
- 🧰 **Modular Design** – each JS primitive is implemented as a self-contained Rust module.

---

## 📦 Installation

### CDN (Browser)
Load the WebAssembly runtime directly from a CDN:

```html
<script type="module">
  import {
    StrictNumber,
    HeapType,
    StrictString,
    StrictArray,
    StrictObject
  } from 'https://cdn.jsdelivr.net/npm/strictjs-runtime@latest/pkg/strictjs_runtime.js';

  const num = new StrictNumber(42);
  console.log(num.value); // 42
</script>
````

### npm (Node / Bundlers)

```bash
npm install strictjs-runtime@latest
```

Then:

```js
import { StrictNumber, StrictString } from 'strictjs-runtime';

const str = new StrictString('Hello, StrictJS!');
console.log(str.length);
```

---

## 🗂️ Project Layout

```
src/
├─ loops/            # for/while loop execution
├─ reactive_system/  # reactive state engine
├─ strict_array/     # Safe Array implementation
├─ strict_async/     # Event loop & async primitives
├─ strict_bigint/    # BigInt support
├─ strict_function/  # Function objects & closures
├─ strict_number/    # Type-safe Numbers
├─ strict_object/    # Schema-based Objects
├─ strict_string/    # Bounded Strings
├─ threads/          # Thread pool, tasks, and scheduling
├─ types/            # Heap types & schema definitions
├─ utils/            # Shared utilities
├─ tests/            # Feature-specific test suites
├─ error.rs          # Centralized error handling
└─ lib.rs            # Runtime entry point
```

---

## 🚀 Getting Started (Rust Build)

For those building from source or contributing to the Rust core:

```bash
git clone https://github.com/Kenneth732/strictjs_runtime.git
cd strictjs_runtime
cargo build --release
```

You can then run the CLI (work in progress):

```bash
cargo run -- run examples/hello.js
```

---

## 🛣️ Roadmap

* [ ] Complete ES202X core features (Numbers, Strings, Arrays, Objects, Functions)
* [ ] Full async event loop & micro-task queue
* [ ] Garbage collector (mark-and-sweep prototype)
* [ ] CLI & REPL
* [ ] Package manager integration
* [ ] Comprehensive documentation and examples

Progress is tracked in [Issues](../../issues) and the [Projects](../../projects) board.

---

## 🤝 Contributing

Contributions of all kinds are welcome:

1. Fork the repo and create a feature branch.
2. Run the test suite with `cargo test`.
3. Submit a pull request with a clear description.

Please review the upcoming [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards.

---

## 📄 License

[MIT](LICENSE)

---

### Acknowledgements

Inspired by the work of V8, SpiderMonkey, Deno, Bun, and the wider Rust & WebAssembly communities.

```

