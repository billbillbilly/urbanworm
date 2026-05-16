# Urban-WORM

**W**orkflow **O**f **R**eproducible **M**ultimodal Inference

[![PyPI](https://img.shields.io/pypi/v/urban-worm.svg)](https://pypi.python.org/pypi/urban-worm)
[![PyPI Downloads](https://static.pepy.tech/badge/urban-worm)](https://pepy.tech/project/urban-worm)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/billbillbilly/urbanworm/blob/main/docs/example_colab.ipynb)

![workflow](images/urabn_worm_diagram.png)

Urban-WORM is a high-level Python interface for building geo-referenced urban datasets with model-generated ground-truth labels. It covers the full pipeline — from collecting crowdsourced street views, photos, and sounds near building footprints, through batched VLM inference, to an organised export of labelled metadata.

---

## Features

### Data collection

- Collect geotagged **street views** (Mapillary / Google), **photos** (Flickr), and **audio** (Freesound / Radio Aporee) within the proximity of building footprints or other points of interest
- Calibrate panorama orientation to face a given location; auto-compute field-of-view from building footprints
- Filter personal photos with face detection; slice audio recordings into fixed-duration clips
- **Crash-safe checkpointing** — pass `checkpoint_path` to any collection method; already-fetched locations are skipped on resume

### Inference / ground-truth labelling

- Define a structured output schema once; all backends share the same `one_inference` / `batch_inference` interface
- **[Unsloth](api/inference.md#unsloth-recommended)** — GPU-accelerated local VLM; auto-detects multiple GPUs; OOM-safe chunk retry; 2–4× faster than Ollama
- **[Ollama](api/inference.md#ollama)** — lightweight local inference, no GPU required
- **[llama.cpp](api/inference.md#llamacpp)** — highly customisable sampling; supports audio input
- **[Cloud APIs](api/inference.md#cloud-api)** — Claude, GPT-4o, Gemini via `InferenceAPI`
- **Crash-safe checkpointing** on all `batch_inference` methods

### Export

- `GeoTaggedData.export()` — one call produces a `metadata.csv` paired with an organised `images/` or `audio/` folder

---

## Quick example

```python
from urbanworm import GeoTaggedData, InferenceUnsloth
from typing import Literal

# 1 — collect street views near building footprints
gtd = GeoTaggedData()
gtd.getBuildings(bbox=(-83.208, 42.374, -83.206, 42.375), source='osm')
gtd.get_svi_from_locations(key="YOUR_MAPILLARY_KEY", distance=30, reoriented=True)

# 2 — define output schema and run inference
schema = {
    "occupancy": (Literal["occupied", "unoccupied", "uncertain"], ...),
    "visual_evidence": (str, ...),
}
infer = InferenceUnsloth(
    llm="unsloth/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
    geo_tagged_data=gtd,
    schema=schema,
)
df = infer.batch_inference(
    prompt="Does this house look occupied or vacant?",
    batch_size=4,
    checkpoint_path="labels.jsonl",
)

# 3 — export
gtd.export(output_dir="dataset", data="svi", labels=df)
```

---

## License

MIT — see [LICENSE](https://github.com/billbillbilly/urbanworm/blob/main/LICENSE).

The development of this package is supported and inspired by the city of Detroit.
