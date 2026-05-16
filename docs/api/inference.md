# Inference Backends

All backends share the same `one_inference` / `batch_inference` interface defined in
[`Inference`][urbanworm.inference.Inference.Inference].

---

## Unsloth (recommended)

GPU-accelerated local VLM inference via Unsloth's `FastVisionModel`.
Automatically spreads the model across all visible GPUs and retries
OOM-failed chunks item-by-item.

**Install:** `pip install "urban-worm[unsloth]"` (pre-install CUDA torch first)

::: urbanworm.inference.unsloth.InferenceUnsloth

---

## Ollama

Inference via a locally running [Ollama](https://ollama.com) server.
No GPU required; any GGUF-backed vision model works.

**Install:** `pip install "urban-worm[ollama]"` + the Ollama app

::: urbanworm.inference.llama.InferenceOllama

---

## llama.cpp

Inference via the `llama-mtmd-cli` binary. Supports audio input; highly
configurable sampling parameters.

**Install:** `pip install "urban-worm[llamacpp]"` + `brew install llama.cpp`

::: urbanworm.inference.llama.InferenceLlamacpp

---

## Cloud API

Inference via hosted providers (Anthropic, OpenAI, Google).

**Install:** `pip install "urban-worm[api]"`

::: urbanworm.inference.api.InferenceAPI

---

## Output schema

::: urbanworm.inference.format.create_format

::: urbanworm.inference.format.Response
