"""IO helpers: image/audio downloads, base64, temp files, URL/path detection."""
from __future__ import annotations

from .checkpoint import (
    _MockResponse,
    append_collection_checkpoint,
    append_inference_checkpoint,
    load_collection_checkpoint,
    load_inference_checkpoint,
    restore_audios_from_checkpoint,
    restore_llamacpp_results,
    restore_ollama_results,
    restore_photos_from_checkpoint,
    restore_svis_from_checkpoint,
)
from .utils import (
    base64_to_image,
    base64img2temp,
    clip,
    download_freesound_preview,
    download_image_requests,
    encode_image_to_base64,
    is_base64,
    is_image_path,
    is_url,
    load_image_auto,
    save_base64,
    sound_url_to_temp,
    url2temp,
)

__all__ = [
    # checkpoint helpers
    "_MockResponse",
    "load_collection_checkpoint",
    "append_collection_checkpoint",
    "load_inference_checkpoint",
    "append_inference_checkpoint",
    "restore_svis_from_checkpoint",
    "restore_photos_from_checkpoint",
    "restore_audios_from_checkpoint",
    "restore_ollama_results",
    "restore_llamacpp_results",
    # original IO helpers
    "base64_to_image",
    "base64img2temp",
    "clip",
    "download_freesound_preview",
    "download_image_requests",
    "encode_image_to_base64",
    "is_base64",
    "is_image_path",
    "is_url",
    "load_image_auto",
    "save_base64",
    "sound_url_to_temp",
    "url2temp",
]
