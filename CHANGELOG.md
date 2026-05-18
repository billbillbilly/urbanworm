# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] — 2026-05-17

### Fixed

- **Multi-GPU BFloat16 dtype mismatch in vendored `urbanworm/inference/unsloth.py`**
  (`InferenceUnsloth._generate_batch`) — with `device_map='auto'` across two
  GPUs, accelerate splits ViT blocks between devices and moves tensors without
  re-casting their dtype. The image processor always emits `float32`
  `pixel_values`, which caused `RuntimeError: expected scalar type BFloat16
  but found Float` deep inside the vision encoder. Fixed by adding a new
  `_apply_dtype_hooks_once()` method that registers
  `register_forward_pre_hook` on the top-level vision encoder and every ViT
  transformer block (identified by `norm1 + attn/self_attn`) to cast
  floating-point tensors to the model's compute dtype before each forward
  pass. The hook registration is deferred until after the model is loaded
  (lazy load happens inside `batch_inference`, not `__init__`) and is
  idempotent.
- **Dtype-hook flag applied too early** — `_dtype_hooks_applied` was set to
  `True` before the `model is None` guard, so a subsequent call after the
  model became available would skip hook registration entirely. Flag is now
  set only after the model reference is confirmed live.
- **VRAM budget used total memory instead of free memory** — `max_memory`
  was computed as `total_memory * 0.90`, which could trigger OOM on a GPU
  already partially occupied by other processes. Changed to
  `mem_get_info(i)[0] * 0.90` (free VRAM at load time).
- **Checkpoint resume re-processed trailing partial batch** —
  `start_idx = (len(done_records) // bs) * bs` rounded down to the nearest
  batch boundary, causing the last partial batch of a prior run to be
  re-inferred on every restart. Changed to `start_idx = len(done_records)`,
  with `done_records` (not `done_records[:start_idx]`) written to the
  result to match.
- **Silent exceptions in `_run_chunk_with_retry` halving cascade** — when a
  batch failed and `current_bs > 1`, the error was swallowed with no log
  output, making it impossible to diagnose why inference kept slowing down.
  Added `logger.warning(...)` before the retry to surface the exception
  type, message, and new batch size.
- **Missing `_model_dtype` attribute initialisation in `__init__`** — added
  `self._model_dtype = None` alongside the other `None`-initialised
  attributes for clarity and to avoid potential `AttributeError` on early
  attribute access.

### Changed

- **`classify_outdoor_unsloth.py`** — removed the now-redundant
  `_patch_infer_for_dtype_safety` runtime monkey-patch and its
  `_apply_model_dtype_hooks` helper; the dtype fix is applied inside the
  vendored package. Also removed the unused `import functools` and reordered
  `packages` in `print_runtime_versions_and_devices()` so `unsloth` appears
  before `transformers` (matching unsloth's own import-order requirement).

### Added

- **`patch_urbanworm.py`** — surgical patch script that locates the
  installed `urbanworm` package via `importlib`, uses `ast` to find the
  exact insertion point inside `InferenceUnsloth._generate_batch`, and
  injects the `_apply_dtype_hooks_once()` call. Supports `--check`
  (idempotency test), `--revert` (restore from `.bak`), and creates a
  `.bak` backup on first run.

## [0.2.2] — 2026-05-17

### Added
- **Multi-GPU support for `InferenceUnsloth`** — when more than one CUDA GPU
  is detected, `device_map="auto"` is set automatically and each GPU's VRAM
  budget is capped at 90 % of its capacity. Override with
  `max_memory={0: "10GiB", 1: "10GiB"}`. The `device` constructor parameter
  can still force a specific device map.
- **`model_dir` parameter** on all three local inference backends. Controls
  where downloaded model weights are stored:
  - `InferenceUnsloth` — passed as `cache_dir` to
    `FastVisionModel.from_pretrained` (HuggingFace Hub cache).
  - `InferenceOllama` — sets `OLLAMA_MODELS` around `ollama.pull` (saved and
    restored so other instances are not affected).
  - `InferenceLlamacpp` — sets `HF_HUB_CACHE` in the `llama-mtmd-cli`
    subprocess environment (applies when downloading via `-hf`; no effect on
    local GGUF paths).
- **Stability utilities for large-scale `InferenceUnsloth` jobs:**
  - `configure_runtime(disable_compile=True)` — sets
    `UNSLOTH_COMPILE_DISABLE`, `UNSLOTH_DISABLE_FAST_GENERATION`, and
    `TORCH_COMPILE_DISABLE` before Unsloth/Torch are imported. Called
    automatically in `__init__` (default `disable_compile=True`). Prevents
    the `AlignDevicesHook`/Torch-Dynamo recompile crashes that surface on runs
    of ~10 k+ samples.
  - `clear_compile_cache()` — removes the Unsloth compiled-model cache from
    system temp dirs; useful when a stale cache causes recompile errors.
  - `task_chunk_size` parameter on `batch_inference` — logical job-partition
    size independent of `batch_size`; reports progress at the task-chunk level
    for long runs.
  - `failed_log_path` parameter on `batch_inference` — appends permanently
    failed sample indices and error messages to a CSV so they can be rerun
    later.
  - `_log_runtime_versions()` — logs torch, CUDA, GPU, transformers,
    accelerate, and unsloth versions at `INFO` level on model load.
  - `_classify_error()` — identifies known recoverable patterns (compile/hook
    conflict, CUDA OOM, dtype mismatch) and emits a human-readable hint.
- **Halving retry cascade in `InferenceUnsloth._run_chunk_with_retry`** —
  on failure, the batch is retried at half the original size all the way down
  to 1, then fills stubs (or re-raises if `skip_errors=False`).
- **MkDocs documentation site** — `mkdocs.yml` with Material theme
  (light-blue/green + purple/green palette), mkdocstrings, mkdocs-jupyter,
  autorefs, and git-revision-date-localized. Auto-deployed to GitHub Pages on
  push to `main` via `.github/workflows/docs.yml`. Docs added:
  `docs/index.md`, `docs/installation.md`, `docs/quickstart.md`,
  `docs/api/inference.md`, `docs/api/dataset.md`, `docs/api/sources.md`,
  `docs/changelog.md`.

### Fixed
- **BFloat16 / Float32 dtype mismatch in `InferenceUnsloth._generate_batch`**
  — the image processor always emits `pixel_values` as float32, but BF16
  models raised `expected scalar type BFloat16 but found Float`. All
  floating-point input tensors are now cast to the model's compute dtype
  after the processor call.
- **`ModuleNotFoundError: No module named 'ollama'`** when importing
  `InferenceUnsloth` in environments without Ollama — caused by an eager
  `import ollama` at the top of `llama.py` and an eager import of `llama.py`
  in `__init__.py`. All four backends (`InferenceOllama`, `InferenceLlamacpp`,
  `InferenceUnsloth`, `InferenceAPI`) are now lazy via `__getattr__` in
  `urbanworm/__init__.py`. `llama.py` exposes a `_lazy_ollama()` helper that
  raises a descriptive `ImportError` only when Ollama is actually used.
- **`pydub` `SyntaxWarning` spam on Python 3.12** — pydub's own source
  contains invalid escape sequences. The module-level
  `from pydub import AudioSegment` import is replaced by a
  `_load_audio_segment()` helper that suppresses the warning with
  `warnings.filterwarnings`. All three call sites (`probe_audio_duration`,
  `clip`, `sound_url_to_temp`) updated.
- **`skip_errors=False` ignored in `InferenceUnsloth` retry ladder** — when
  the final single-item retry was exhausted, stub responses were always filled
  regardless of `skip_errors`. The flag is now checked and the exception is
  re-raised when `skip_errors=False`.
- **`clear_compile_cache` silent cwd deletion** — when `TEMP` or `TMP` env
  vars are unset, `Path("") / "unsloth_compiled_cache"` resolved to a
  relative path in cwd. Env-derived candidates are now only added when the
  variable is non-empty.
- **`OLLAMA_MODELS` global mutation** — `InferenceOllama.one_inference` and
  `.batch_inference` now save and restore (or remove) `OLLAMA_MODELS` around
  `ollama.pull` so concurrent instances with different `model_dir` values do
  not clobber each other.
- **Subprocess safety in `InferenceLlamacpp._mtmd`** — added null-byte
  validation on `system_message` and `prompt`, a `None` guard on `llm`, and
  a comment documenting why list-based invocation is safe without
  `shlex.escape()`.
- Multi-GPU input preparation: `.to(self._model.device)` raised
  `AttributeError` when `device_map="auto"` splits the model across GPUs.
  Fixed by using `next(self._model.parameters()).device` instead.

## [0.2.0] — 2026-05-11 (dev2 branch)

### Fixed
- Version mismatch across `pyproject.toml`, `urbanworm/__init__.py`, and
  `CITATION.cff`. `__version__` is now resolved at runtime via
  `importlib.metadata` so it stays in sync with the installed distribution.
- `GeoTaggedData.__init__` was using chained dict assignment, which aliased
  `self.svis`, `self.photos`, and `self.audios` to the same underlying dict.
  They are now independent.
- `_pack` (in `inference.Inference`) dropped the trailing group when
  consecutive locations stayed equal to the end of the input.
- `InferenceLlamacpp.one_inference` / `batch_inference` shadowed the `temp`
  (temperature) parameter with a temp-file path inside the URL/base64 loops.
- `closest()` raised `NameError` when a season was provided but didn't match
  one of the four hard-coded checks.
- `get_sound_from_location` raised `NameError` for the single-clip path
  (`flattened_slice_list` was only defined in the multi-clip branch).
- `_mtmd` (Ollama, multi-image, `multiImgInput=False`) now passes `img[i]`
  to per-image inference instead of the full image list.
- `sound_url_to_temp` no longer returns a path to a deleted file on download
  failure — it cleans up and re-raises.
- Bare `except:` clauses replaced with narrow exception classes; obviously-
  unsafe `timeout=999`/`9999` values reduced to `30`/`60` seconds.
- `download_to_dir` now raises `ValueError` when `to_dir` is missing instead
  of silently returning, and aligns sentinel paths so list lengths stay
  consistent on download failures.
- `construct_units` now raises `ValueError`/`TypeError` on bad input instead
  of printing and silently returning `None`.
- `getSV` honours `MAPILLARY_API_KEY` env var like the other source helpers.
- `InferenceOllama` `skip_errors=True` now actually suppresses validation
  errors and returns an empty `Response` instead of re-raising.

### Changed
- Replaced `requirements.txt` as the source of truth: dependencies and
  optional extras (`ollama`, `audio`, `llamacpp`, `dev`, `all`) now live in
  `pyproject.toml` `[project]`. Added missing transitive deps
  (`mercantile`, `pyproj`, `shapely`).
- `pd.concat`-in-loop replaced with single concat in `get_svi_from_locations`,
  `get_photo_from_location`, and `get_sound_from_location` (O(n²) → O(n)).
- `print(...)` calls in library code switched to a module-level
  `logging.getLogger("urbanworm")`.
- Replaced flake8 config with `[tool.ruff]`. Added `[tool.pytest.ini_options]`.
- CI split into a fast `unit` job (Ubuntu, py3.10/3.11/3.12) gated on every
  push and PR, plus a self-hosted `integration` job that runs on `main`.
- Internal helpers `_year_range` and `_parse_created` (defined inside
  `getSound`) consolidated to top-level `solr_year_range` and
  `parse_iso_created` in `urbanworm.utils.utils`.

### Added
- `tests/test_utils.py`, `tests/test_format.py`, `tests/test_dataset.py`,
  `tests/test_inference_helpers.py` — pure-logic unit tests.
- `urbanworm.utils.{geo,io,json_repair,timefilter,face,audio}` submodules
  re-exporting curated helpers from the catch-all `utils.utils` module.
- `urbanworm.sources.{mapillary,flickr,freesound}` submodules re-exporting
  `getSV`, `getPhoto`, `getSound` for a more discoverable namespace.
- **`InferenceUnsloth`** — new VLM backend mirroring `InferenceOllama`'s
  public surface but running locally via `unsloth.FastVisionModel`. Supports
  GPU `batch_size` for throughput, lazy import (no torch/unsloth pulled in
  unless the class is constructed), JSON-repair fallback, and
  `skip_errors` parity. Default checkpoint:
  `unsloth/Qwen3-VL-3B-Instruct`. Tested with Qwen3-VL-3B/8B,
  Gemma-3-4B-IT, Qwen2-VL-2B, Qwen2.5-VL-7B-bnb-4bit. Install with
  `pip install "urban-worm[unsloth]"`. Tests in `tests/test_unsloth.py`
  use mocks so they run on any CI without a GPU.
- **Aporee audio source** — new `getSoundAporee()` and
  `urbanworm.sources.aporee` module. Filters a Radio Aporee catalog
  (CSV path or in-memory DataFrame with `url`, `latitude`, `longitude`
  columns; optional `id`/`identifier`, `name`, `description`, `tags`,
  `created`, `duration_s`) by spatial proximity using the same
  semantics as the Freesound path. `getSound()` is now a dispatcher
  with `source: str = 'freesound'` (default) or `'aporee'`.
  `GeoTaggedData.get_sound_from_location` accepts matching `source=`,
  `catalog=`, and `probe_durations=` parameters; existing Freesound
  callers keep working unchanged. Output schema includes a
  `preview-hq-mp3` alias of `url` so `download_to_dir` and the slicing
  pipeline need no changes.
- **`probe_audio_duration(url)`** in `urbanworm.utils.utils`
  (re-exported from `urbanworm.utils.audio`). Downloads an mp3 to a
  tempfile and reads its length via pydub (with mutagen as a fallback).
  Used by the Aporee path when `slice_duration` is requested but the
  catalog has no `duration_s` column.
- **`enrich_aporee_catalog(catalog, out_path=None, min_duration=None,
  skip_existing=True, timeout=60)`** in `urbanworm.dataset`. One-shot
  helper that probes every URL in an Aporee catalog, populates
  `duration_s`, optionally drops rows shorter than `min_duration`, and
  optionally writes the result back to CSV.
- **`fetch_aporee_catalog(bbox, year, hour, season, southern, rows,
  verify_urls, out_path, enrich_durations, min_duration, timeout,
  page_size)`** in `urbanworm.dataset`. Pulls the geolocated Aporee
  catalog from Internet Archive's `radio-aporee-maps` collection via
  the IA Scrape API. Server-side bbox + year filters; client-side hour
  + hemisphere-aware season filters. Optional `verify_urls=True` looks
  up the exact mp3 filename per identifier; default is the fast
  `<id>.mp3` fallback. Output schema is compatible with
  :func:`getSoundAporee` so a fetched DataFrame can be passed directly.
- `getSoundAporee` now accepts the script-style column aliases
  ``lat`` / ``lon`` / ``capture_time`` (renamed internally to the
  canonical ``latitude`` / ``longitude`` / ``created``).
- 33 unit tests in `tests/test_aporee.py` (filtering, dispatcher,
  duration probing, enrichment, IA fetcher with mocked HTTP, alias
  acceptance).
- **`fov='auto'` for `getSV` / `get_svi_from_locations`** — sizes the
  perspective field-of-view per image so the building footprint at the
  query location is just framed (extent + 10% margin, clamped to
  `[fov_min, fov_max]`). Two new helpers in `urbanworm.utils.utils`
  (re-exported via `urbanworm.utils.geo`):
    - `auto_fov_from_polygon(camera_lon, camera_lat, polygon, ...)` —
      computes the angular extent of a `shapely` polygon as seen from
      the camera. The polygon is taken from each unit's `row.geometry`
      when `get_svi_from_locations` is called against building
      footprints loaded by `getBuildings()`.
    - `auto_fov_from_distance(distance_m, building_width_m=15, ...)` —
      heuristic fallback when no polygon is available (e.g. the user
      passed a bare coordinate to `getSV`).
  `getSV` accepts `fov: int | float | str` and a new
  `target_polygon=` parameter; `fov_margin`, `fov_min`, `fov_max`,
  `building_height` control the auto sizing. Requires
  `reoriented=True`.
- **`fov='auto'` is height-aware.** Both `auto_fov_from_polygon` and
  `auto_fov_from_distance` now take `building_height_m` (default 9 m,
  ~3 stories) and `aspect_ratio` (image_width / image_height) and
  return the wider of two requirements: horizontal extent of the
  footprint *or* horizontal FOV needed so the rendered image's
  derived vertical FOV (`vFOV = wFOV / aspect`) covers the building's
  height. Tall, narrow buildings now have their roofs framed instead
  of cropped. Set `building_height=0` to skip the height term.
  15 unit tests in `tests/test_auto_fov.py`.
- **GlobalBuildingAtlas (`gba`) building source** with per-building
  height. New `getGBABuildings(bbox, gba_path, ...)` in
  `urbanworm.utils.building` loads a local GBA file
  (GPKG / GeoJSON / GeoParquet — anything `geopandas.read_file`
  understands), filters by bbox + area, and normalizes the height
  column to `height_m` (recognises `height`, `h`, `bldg_height`,
  `building_height`, `z` as aliases). `GeoTaggedData.getBuildings`
  gains `source='gba'` (with a required `gba_path=`) and logs how
  many of the loaded buildings carry a height value.
- **Per-building height for `fov='auto'`.** When `self.units` has a
  `height_m` column, `get_svi_from_locations` now uses each row's
  actual height instead of the global `building_height` default.
  Falls back gracefully on NaN / missing values.
- **`source='globfp3d'`** in `getBuildings()` for the **3D-GloBFP**
  dataset (Che et al., ESSD 2024). Auto-fetches `world_grid.zip` +
  `data_links.txt` from Zenodo record `15487037`, intersects with the
  bbox, then downloads matching per-tile shapefiles from Figshare. New
  public helpers (canonical names, all in `urbanworm.utils.building`):
  `getGloBFP3DBuildings`, `parse_globfp3d_data_links`,
  `figshare_article_id`, `load_globfp3d_grid_manifest`,
  `load_globfp3d_data_links`, `download_globfp3d_tile`,
  `fetch_globfp3d_for_bbox`. Cached by default under
  `~/.cache/urbanworm/globfp3d`.
- **`source='gba'`** in `getBuildings()` is now the *real*
  **GlobalBuildingAtlas** dataset (Zhu et al., ESSD 2025) — a
  separate dataset hosted on HuggingFace + mediaTUM. New helpers:
  `getGBABuildings`, `load_true_gba_polygon_manifest`,
  `fetch_true_gba_for_bbox`. Auto-fetches polygon tiles from
  `zhu-xlab/GBA.LoD1` using `representative/lod1.geojson` as the
  manifest, reprojects from EPSG:3857 to EPSG:4326. Cached under
  `~/.cache/urbanworm/gba`. Per-row heights from GBA.Height
  (mediaTUM `m1837832`) are NOT yet joined — `include_heights=True`
  is currently a no-op stub; tracking issue.
- **Backwards-compat aliases** retained for the previous GBA-prefixed
  names that actually pointed at the 3D-GloBFP pipeline:
  `parse_gba_data_links`, `load_gba_grid_manifest`,
  `load_gba_data_links`, `download_gba_tile`, `fetch_gba_for_bbox`,
  `_default_gba_cache_dir`, `GBA_ZENODO_RECORD`/`GBA_GRID_URL`/etc.
  Old code continues to work; new code should use the
  `globfp3d`-prefixed names for clarity.
- 23 unit tests in `tests/test_gba.py` covering local-file loaders for
  both datasets, the new dispatcher in `getBuildings()` (validates all
  four sources), parser helpers, figshare-id extraction, filename
  matching, end-to-end fetch with mocked HTTP for both datasets.
- `.env.example` documenting environment variables for API keys.
- `CHANGELOG.md` (this file).

## [0.1.9]
Pre-existing release. See git history.
