# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — dev2 branch

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
- `.env.example` documenting environment variables for API keys.
- `CHANGELOG.md` (this file).

## [0.1.9]
Pre-existing release. See git history.
