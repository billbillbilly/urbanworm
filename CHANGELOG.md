# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
