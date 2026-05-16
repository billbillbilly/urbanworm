# Installation

## Step 1 — Core package

```sh
pip install urban-worm
```

## Step 2 — Choose an inference backend

### Unsloth — recommended (GPU required)

GPU-specific PyTorch must be installed **before** the `unsloth` extra, otherwise pip falls back to a slow CPU-only build.

=== "CUDA"

    ```sh
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install "urban-worm[unsloth]"
    ```

=== "macOS (Apple Silicon)"

    ```sh
    pip install torch          # MPS is enabled by default on macOS
    pip install "urban-worm[unsloth]"
    ```

!!! tip "Supported checkpoints"
    `unsloth/Qwen3-VL-3B-Instruct`, `unsloth/Qwen3-VL-8B-Instruct`,
    `unsloth/gemma-3-4b-it`, `unsloth/Qwen2-VL-2B-Instruct`,
    `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`.
    Any vision model that `unsloth.FastVisionModel` can load should work.

---

### Ollama — lightweight (no GPU required)

Install the [Ollama application](https://ollama.com/) first:

=== "Linux"

    ```sh
    curl -fsSL https://ollama.com/install.sh | sh
    pip install "urban-worm[ollama]"
    ```

=== "macOS"

    ```sh
    brew install ollama
    pip install "urban-worm[ollama]"
    ```

=== "Windows"

    Download the installer from [ollama.com](https://ollama.com/), then:

    ```sh
    pip install "urban-worm[ollama]"
    ```

---

### llama.cpp — CLI-based

The `llama-mtmd-cli` binary must be installed separately:

```sh
# macOS / Linux
brew install llama.cpp

# Windows
winget install llama.cpp
```

Then install the Python binding:

=== "CPU"

    ```sh
    pip install "urban-worm[llamacpp]"
    ```

=== "CUDA"

    ```sh
    CMAKE_ARGS="-DGGML_CUDA=on" pip install "urban-worm[llamacpp]"
    ```

=== "Metal (macOS)"

    ```sh
    CMAKE_ARGS="-DGGML_METAL=on" pip install "urban-worm[llamacpp]"
    ```

---

### Cloud APIs (Claude / GPT-4o / Gemini)

```sh
pip install "urban-worm[api]"
```

---

## Optional extras

| Extra | What it adds |
|---|---|
| `audio` | `pydub` — needed for audio slicing (`get_sound_from_location`) |
| `all` | All inference backends + API providers (no audio) |
| `all,audio` | Everything |
| `dev` | Pytest, ruff, build tools |

```sh
pip install "urban-worm[all]"
pip install "urban-worm[all,audio]"
```

!!! warning "GPU torch + `[all]`"
    Pre-install the CUDA torch wheel **before** running `pip install "urban-worm[all]"`.
    See the Unsloth tab above for the one-liner.

---

## Dev install from source

```sh
pip install -e git+https://github.com/billbillbilly/urbanworm.git#egg=urban-worm
pip install "urban-worm[dev]"
```
