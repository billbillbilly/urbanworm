"""JSONL-based checkpoint helpers for resumable collection and inference.

Both the collection stage (``GeoTaggedData.get_svi_from_locations`` etc.) and
the inference stage (``*.batch_inference``) can write a JSONL file as they
process each item.  On the next run the file is read back, already-completed
items are skipped, and processing continues where it left off.

Collection record schema (one line per location):
    {"loc_id": <any>, "ids": [...], "paths": [...], "data": [...],
     "slices": [...] | null, "metadata": [{...row...}, ...]}

Inference record schema (one line per image / prompt):
    {"idx": <int>, "responses": [{field: value, ...}, ...], "data": <str|list>}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("urbanworm")


# ---------------------------------------------------------------------------
# Compatibility shim for response2df
# ---------------------------------------------------------------------------

class _MockResponse:
    """Thin dict wrapper that satisfies both ``vars()`` and ``dict()`` so that
    checkpoint-restored records can pass through :func:`response2df` without
    modification.

    ``response2df`` calls:
      * ``vars(item).keys()``     — resolved via ``__dict__``
      * ``dict(item)``            — resolved via ``keys()`` + ``__getitem__``
    """

    def __init__(self, d: dict) -> None:
        self.__dict__.update(d)

    # Mapping protocol so dict(instance) works
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, k):
        return self.__dict__[k]

    def __repr__(self) -> str:
        return f"_MockResponse({self.__dict__!r})"


# ---------------------------------------------------------------------------
# Collection checkpoints
# ---------------------------------------------------------------------------

def load_collection_checkpoint(path: str | Path) -> tuple[set, list[dict]]:
    """Load a collection JSONL checkpoint.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        ``(done_ids, records)`` where *done_ids* is the set of ``loc_id``
        values already written and *records* is the raw list of dicts.
    """
    p = Path(path)
    if not p.exists():
        return set(), []

    done_ids: set = set()
    records: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done_ids.add(rec["loc_id"])
                records.append(rec)
            except Exception as exc:
                logger.debug(
                    "collection checkpoint: skipping malformed line %d (%s)", lineno, exc
                )

    logger.info(
        "collection checkpoint: loaded %d completed location(s) from %s",
        len(done_ids), path,
    )
    return done_ids, records


def append_collection_checkpoint(path: str | Path, record: dict) -> None:
    """Append one location record to a collection JSONL checkpoint.

    Creates parent directories automatically.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def restore_svis_from_checkpoint(records: list[dict]) -> tuple[dict, list]:
    """Rebuild ``self.svis`` and metadata frames from collection checkpoint records.

    Args:
        records: List of dicts loaded by :func:`load_collection_checkpoint`.

    Returns:
        ``(svis_dict, frames)`` where *svis_dict* mirrors the ``self.svis``
        payload and *frames* is a list of per-location DataFrames for
        rebuilding ``self.svi_metadata``.
    """
    import pandas as pd

    svis: dict = {"loc_id": [], "id": [], "data": [], "path": []}
    frames: list = []

    for rec in records:
        ids = rec.get("ids", [])
        paths = rec.get("paths", [])
        data = rec.get("data", [])

        # Pad shorter lists so lengths stay consistent
        n = max(len(ids), len(paths), len(data))
        ids = ids + [""] * (n - len(ids))
        paths = paths + [""] * (n - len(paths))
        data = data + [""] * (n - len(data))

        svis["loc_id"].extend([rec["loc_id"]] * n)
        svis["id"].extend(ids)
        svis["data"].extend(data)
        svis["path"].extend(paths)

        meta_rows = rec.get("metadata", [])
        if meta_rows:
            frames.append(pd.DataFrame(meta_rows))

    return svis, frames


def restore_photos_from_checkpoint(records: list[dict]) -> tuple[dict, list]:
    """Rebuild ``self.photos`` from collection checkpoint records."""
    import pandas as pd

    photos: dict = {"loc_id": [], "id": [], "data": [], "path": []}
    frames: list = []

    for rec in records:
        ids = rec.get("ids", [])
        data = rec.get("data", [])
        n = max(len(ids), len(data))
        ids = ids + [""] * (n - len(ids))
        data = data + [""] * (n - len(data))

        photos["loc_id"].extend([rec["loc_id"]] * n)
        photos["id"].extend(ids)
        photos["data"].extend(data)
        photos["path"].extend([""] * n)

        meta_rows = rec.get("metadata", [])
        if meta_rows:
            frames.append(pd.DataFrame(meta_rows))

    return photos, frames


def restore_audios_from_checkpoint(records: list[dict]) -> tuple[dict, list]:
    """Rebuild ``self.audios`` from collection checkpoint records."""
    import pandas as pd

    audios: dict = {"loc_id": [], "id": [], "data": [], "path": []}
    has_slices = any(rec.get("slices") for rec in records)
    if has_slices:
        audios["slice"] = []

    frames: list = []

    for rec in records:
        ids = rec.get("ids", [])
        data = rec.get("data", [])
        slices = rec.get("slices") or []
        n = max(len(ids), len(data))
        ids = ids + [""] * (n - len(ids))
        data = data + [""] * (n - len(data))

        audios["loc_id"].extend([rec["loc_id"]] * n)
        audios["id"].extend(ids)
        audios["data"].extend(data)
        audios["path"].extend([""] * n)
        if has_slices:
            audios["slice"].extend(slices + [None] * (n - len(slices)))  # type: ignore[index]

        meta_rows = rec.get("metadata", [])
        if meta_rows:
            frames.append(pd.DataFrame(meta_rows))

    return audios, frames


# ---------------------------------------------------------------------------
# Inference checkpoints
# ---------------------------------------------------------------------------

def load_inference_checkpoint(path: str | Path) -> list[dict]:
    """Load an inference JSONL checkpoint.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of per-image result records in original order.  Each record has
        at minimum ``{"idx": int, "responses": [...], "data": ...}``.
    """
    p = Path(path)
    if not p.exists():
        return []

    records: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as exc:
                logger.debug(
                    "inference checkpoint: skipping malformed line %d (%s)", lineno, exc
                )

    logger.info(
        "inference checkpoint: loaded %d completed item(s) from %s",
        len(records), path,
    )
    return records


def append_inference_checkpoint(path: str | Path, record: dict) -> None:
    """Append one image result record to an inference JSONL checkpoint."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def restore_ollama_results(records: list[dict]) -> dict:
    """Reconstruct an ``InferenceOllama``-compatible ``self.results`` dict
    from checkpoint records.

    Each record's ``responses`` list (plain dicts) is wrapped with
    :class:`_MockResponse` so that :func:`response2df` sees the same
    interface as live Pydantic objects.
    """
    responses = []
    data = []
    for rec in records:
        wrapped = [_MockResponse(d) for d in rec.get("responses", [])]
        responses.append(wrapped)
        data.append(rec["data"])
    return {"responses": responses, "data": data}


def restore_llamacpp_results(records: list[dict]) -> dict:
    """Reconstruct an ``InferenceLlamacpp``-compatible ``self.results`` dict
    from checkpoint records.

    LlamaCpp stores the raw ``extract_last_json`` dict directly, which
    ``InferenceLlamacpp.to_df`` can process without wrapping.
    """
    responses = []
    data = []
    for rec in records:
        responses.append(rec.get("responses"))
        data.append(rec["data"])
    return {"responses": responses, "data": data}
