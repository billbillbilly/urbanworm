"""Face detection / selfie filter (YuNet ONNX model)."""
from __future__ import annotations

from .utils import YuNet, is_selfie_photo

__all__ = ["YuNet", "is_selfie_photo"]
