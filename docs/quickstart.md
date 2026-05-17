# Quick Start

This page walks through the core workflow end-to-end in a few code blocks.
For a runnable version, open the [Colab demo](example_colab.ipynb).

---

## 1 — Collect street views

```python
from urbanworm import GeoTaggedData

gtd = GeoTaggedData()

# Pull building footprints from OpenStreetMap
gtd.getBuildings(bbox=(-83.208, 42.374, -83.206, 42.375), source='osm')

# Fetch the closest reoriented street views for each building
gtd.get_svi_from_locations(
    key="YOUR_MAPILLARY_KEY",
    distance=30,          # search radius in metres
    reoriented=True,      # crop panorama to face the building
    multi_num=3,          # up to 3 views per location
    checkpoint_path="run/svi.jsonl",  # resume-safe
)

# Download images to disk
gtd.download_to_dir(data='svi', to_dir='run/images')
```

!!! note "No Mapillary key?"
    You can skip collection entirely and pass your own image paths directly
    to the inference constructor via `images=[...]`.

---

## 2 — Define a schema

Urban-WORM uses a plain dict to declare the structured fields the model must return.
Standard Python type hints control what values are allowed.

```python
from typing import Literal

schema = {
    "occupancy": (Literal["occupied", "unoccupied", "uncertain"], ...),
    "visual_evidence": (str, ...),
}
```

---

## 3 — Run inference

=== "Unsloth (local GPU)"

    ```python
    from urbanworm import InferenceUnsloth

    infer = InferenceUnsloth(
        llm="unsloth/Qwen2-VL-2B-Instruct",
        load_in_4bit=True,
        geo_tagged_data=gtd,
        schema=schema,
        model_dir="/data/models",   # optional: override HF cache dir
    )

    df = infer.batch_inference(
        system="You are an urban researcher assessing housing conditions.",
        prompt="Does this house look occupied or vacant? Describe the visual evidence.",
        batch_size=4,
        checkpoint_path="run/labels.jsonl",
    )
    ```

=== "Ollama (local, no GPU)"

    ```python
    from urbanworm.inference.llama import InferenceOllama

    infer = InferenceOllama(
        llm="hf.co/ggml-org/InternVL3-8B-Instruct-GGUF:Q8_0",
        geo_tagged_data=gtd,
        schema=schema,
        model_dir="/data/models",   # optional: sets OLLAMA_MODELS
    )

    df = infer.batch_inference(
        prompt="Does this house look occupied or vacant?",
        checkpoint_path="run/labels.jsonl",
    )
    ```

=== "llama.cpp"

    ```python
    from urbanworm import InferenceLlamacpp

    infer = InferenceLlamacpp(
        llm="ggml-org/InternVL3-8B-Instruct-GGUF:Q8_0",
        geo_tagged_data=gtd,
        schema=schema,
        model_dir="/data/models",   # optional: sets HF_HUB_CACHE
    )

    df = infer.batch_inference(
        prompt="Does this house look occupied or vacant?",
        checkpoint_path="run/labels.jsonl",
    )
    ```

=== "Cloud API"

    ```python
    from urbanworm import InferenceAPI

    infer = InferenceAPI(
        llm="claude-sonnet-4-5",
        provider="anthropic",
        api_key="YOUR_API_KEY",
        geo_tagged_data=gtd,
        schema=schema,
    )

    df = infer.batch_inference(
        prompt="Does this house look occupied or vacant?",
        checkpoint_path="run/labels.jsonl",
    )
    ```

---

## 4 — Export

```python
# Produces dataset/metadata.csv + dataset/images/
csv_path = gtd.export(output_dir="dataset", data="svi", labels=df)
```

---

## Multi-GPU note

When multiple CUDA GPUs are detected, `InferenceUnsloth` automatically sets
`device_map="auto"` and splits the model across all of them. You can override
the per-GPU memory budget:

```python
infer = InferenceUnsloth(
    llm="unsloth/Qwen3-VL-8B-Instruct",
    load_in_4bit=True,
    max_memory={0: "10GiB", 1: "10GiB"},  # e.g. two 12 GB cards
    schema=schema,
)
```

---

## Custom model directory

All three local backends accept a `model_dir` parameter so you can control
where downloaded model weights are stored — useful on shared servers or when
the default home directory is on a small partition.

| Backend | Effect of `model_dir` |
|---|---|
| `InferenceUnsloth` | Sets `cache_dir` in `FastVisionModel.from_pretrained()` (HuggingFace Hub cache) |
| `InferenceOllama` | Sets the `OLLAMA_MODELS` env var before each `ollama.pull()` call |
| `InferenceLlamacpp` | Sets `HF_HUB_CACHE` in the `llama-mtmd-cli` subprocess environment (only applies when downloading via `-hf`; has no effect on local GGUF paths) |

```python
# Unsloth — store weights on a large data drive
infer = InferenceUnsloth(
    llm="unsloth/Qwen2-VL-7B-Instruct",
    model_dir="/data/models",
    schema=schema,
)

# Ollama — point the client at a non-default model store
# Note: the Ollama server itself must also be started with OLLAMA_MODELS
# pointing to the same directory for new downloads to land there.
infer = InferenceOllama(
    llm="hf.co/ggml-org/InternVL3-8B-Instruct-GGUF:Q8_0",
    model_dir="/data/models",
    schema=schema,
)

# llama.cpp — redirect HuggingFace GGUF downloads
infer = InferenceLlamacpp(
    llm="ggml-org/InternVL3-8B-Instruct-GGUF:Q8_0",
    model_dir="/data/models",
    schema=schema,
)
```
