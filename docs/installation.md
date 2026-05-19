# Installation

## Step 1 — Core package

```sh
pip install urban-worm
```

## Step 2 — Choose an inference backend

### Unsloth — recommended (GPU required)

GPU-specific PyTorch must be installed **before** the `unsloth` extra, otherwise pip falls back to a slow CPU-only build.

=== "CUDA"

    First, find your CUDA version:

    ```sh
    nvidia-smi   # look for "CUDA Version: X.Y" in the top-right corner
    ```

    Map the reported version to the matching PyTorch wheel tag:

    | CUDA version | Wheel tag |
    |---|---|
    | 11.8 | `cu118` |
    | 12.1 | `cu121` |
    | 12.4 | `cu124` |
    | 12.6 | `cu126` |
    | 12.8 | `cu128` |

    !!! warning "Pin the torch version that unsloth requires"
        Unsloth requires a **specific** torch version and will not work correctly with an
        arbitrary latest release.  Before installing, check the
        [Unsloth installation guide](https://docs.unsloth.ai/get-started/installing-+-updating-unsloth)
        to find the torch version it currently requires (e.g. `2.6.0`), then pin that version
        explicitly in the first command.

    Substitute your CUDA tag and the torch version unsloth requires
    (example below uses `cu126` and `torch==2.6.0`):

    ```sh
    pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu126
    pip install "urban-worm[unsloth]" --extra-index-url https://download.pytorch.org/whl/cu126
    ```

    Not sure which CUDA tag to use? The [PyTorch install selector](https://pytorch.org/get-started/locally/) generates the exact command for your system.

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
    Pre-install the CUDA torch wheel **and** pass `--extra-index-url` with your CUDA tag
    when running `pip install "urban-worm[all]"`, otherwise pip will replace the CUDA build with a
    CPU-only torch pulled from PyPI.  Replace `cu126` with the tag that matches your driver
    (run `nvidia-smi` to check; see the CUDA version table in the Unsloth section above), and
    pin the torch version that unsloth currently requires — check the
    [Unsloth installation guide](https://docs.unsloth.ai/get-started/installing-+-updating-unsloth)
    before running these commands.

    ```sh
    pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu126
    pip install "urban-worm[all]" --extra-index-url https://download.pytorch.org/whl/cu126
    ```

---

## Dev install from source

```sh
pip install -e git+https://github.com/billbillbilly/urbanworm.git#egg=urban-worm
pip install "urban-worm[dev]"
```
