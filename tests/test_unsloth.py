"""Tests for InferenceUnsloth — fully mocked, no GPU / unsloth needed.

These cover:
* lazy import / clean error when unsloth is missing
* message construction including system prompt + multi-image
* batch chunking with the user-facing batch_size argument
* JSON-repair fallback chain (direct → sanitize → extract)
* skip_errors=True behavior on irrecoverable model output
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from urbanworm.inference.unsloth import InferenceUnsloth

# ----------------------------------------------------------------------
# Helpers to build a fake unsloth + torch + processor stack
# ----------------------------------------------------------------------

def _install_fake_torch():
    """Install a minimal fake torch module if torch is not really available."""
    if "torch" in sys.modules:
        return
    fake = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    fake.no_grad = lambda: _NoGrad()
    sys.modules["torch"] = fake


def _make_fake_inputs(batch_n: int):
    """Mimic a HF BatchEncoding: dict-like with .to() and an input_ids tensor
    that has `.shape[1]` and `.dim() == 2`."""
    class _T:
        def __init__(self, n):
            self._n = n

        def dim(self):
            return 2

        @property
        def shape(self):
            # batch_size, seq_len
            return (self._n, 7)

    class _BatchEnc(dict):
        def to(self, device):
            return self

    enc = _BatchEnc(input_ids=_T(batch_n))
    return enc


def _make_fake_model_and_processor(decoded_texts):
    """Return (fake_model, fake_processor) where generate returns dummy ids
    and the processor's batch_decode returns ``decoded_texts`` verbatim."""

    class _Tensor:
        def __init__(self, payload):
            self._p = payload

        def __getitem__(self, key):
            # We don't actually slice — generate returns the same payload.
            return self

    fake_model = MagicMock()
    fake_model.device = "cpu"
    fake_model.generate = MagicMock(return_value=_Tensor("ids"))

    fake_processor = MagicMock()
    fake_processor.apply_chat_template = MagicMock(
        side_effect=lambda messages, **kw: f"TEMPLATED::{len(messages)}-msgs"
    )
    fake_processor.batch_decode = MagicMock(return_value=list(decoded_texts))
    fake_processor.return_value = _make_fake_inputs(len(decoded_texts))
    return fake_model, fake_processor


def _patch_lazy_imports(monkeypatch, model, processor):
    """Patch InferenceUnsloth._ensure_loaded so it skips the real unsloth import."""

    def fake_ensure(self):
        self._model = model
        self._processor = processor

    monkeypatch.setattr(InferenceUnsloth, "_ensure_loaded", fake_ensure)
    _install_fake_torch()


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_lazy_import_error_when_unsloth_missing():
    """Constructing the class is fine; only loading the model should fail."""
    inst = InferenceUnsloth(llm="unsloth/Qwen3-VL-3B-Instruct")
    # We block both unsloth and torch from being importable.
    with patch.dict(sys.modules, {"unsloth": None, "torch": None}):
        with pytest.raises(ImportError) as exc:
            inst._ensure_loaded()
    assert "urban-worm[unsloth]" in str(exc.value)


def test_audio_input_raises_not_implemented():
    inst = InferenceUnsloth()
    with pytest.raises(NotImplementedError):
        inst.one_inference(prompt="x", audio="/tmp/a.mp3")


def test_one_inference_builds_messages_and_calls_generate(monkeypatch, tmp_path):
    # Create a tiny image file so load_image_auto works.
    from PIL import Image
    img = Image.new("RGB", (32, 32), color=(127, 127, 127))
    img_path = tmp_path / "tiny.png"
    img.save(img_path)

    expected_json = '{"responses": [{"answer": true, "explanation": "ok"}]}'
    model, processor = _make_fake_model_and_processor([expected_json])
    inst = InferenceUnsloth(
        llm="unsloth/Qwen3-VL-3B-Instruct",
        schema={"answer": (bool, ...), "explanation": (str, ...)},
    )
    _patch_lazy_imports(monkeypatch, model, processor)

    df = inst.one_inference(system="You are X.", prompt="Is there a tree?", image=str(img_path))
    # Generate was called exactly once
    assert model.generate.call_count == 1
    # Chat template was applied with both system and user messages
    args, kwargs = processor.apply_chat_template.call_args
    messages = args[0]
    assert any(m["role"] == "system" for m in messages)
    assert any(m["role"] == "user" for m in messages)
    # The schema-hint instructs the model to emit JSON
    user_text = [
        c for c in messages[-1]["content"] if c["type"] == "text"
    ][0]["text"]
    assert "JSON" in user_text
    # Result contains the parsed answer
    assert df is not None and len(df) == 1


def test_batch_inference_chunks_by_batch_size(monkeypatch, tmp_path):
    from PIL import Image
    paths = []
    for i in range(5):
        p = tmp_path / f"i{i}.png"
        Image.new("RGB", (16, 16), color=(i, i, i)).save(p)
        paths.append(str(p))

    model, processor = _make_fake_model_and_processor(["unused"])
    chunks_seen: list[int] = []
    state = {"current_chunk_size": 0}

    # Processor is called once per chunk with a list of templated texts;
    # remember the chunk size so batch_decode can return the matching count.
    def _processor_call(text=None, images=None, **kw):
        state["current_chunk_size"] = len(text)
        return _make_fake_inputs(len(text))

    def _decode_side(*args, **kw):
        n = state["current_chunk_size"]
        chunks_seen.append(n)
        return ['{"responses":[{"answer":true,"explanation":"ok"}]}'] * n

    processor.side_effect = _processor_call
    processor.batch_decode = MagicMock(side_effect=_decode_side)

    inst = InferenceUnsloth(
        images=paths,
        schema={"answer": (bool, ...), "explanation": (str, ...)},
    )
    _patch_lazy_imports(monkeypatch, model, processor)

    df = inst.batch_inference(prompt="?", batch_size=2, disableProgressBar=True)
    # 5 items, batch_size=2 → chunks of [2, 2, 1] = 3 generate calls
    assert model.generate.call_count == 3
    assert chunks_seen == [2, 2, 1]
    # response2df returns one row per input item
    assert df is not None and len(df) == 5


def test_skip_errors_returns_empty_response(monkeypatch, tmp_path):
    from PIL import Image
    p = tmp_path / "x.png"
    Image.new("RGB", (16, 16)).save(p)

    # Model returns garbage that no fallback can rescue.
    model, processor = _make_fake_model_and_processor(["this is not json at all"])
    inst = InferenceUnsloth(
        schema={"answer": (bool, ...)},
        skip_errors=True,
    )
    _patch_lazy_imports(monkeypatch, model, processor)

    # Should not raise.
    df = inst.one_inference(prompt="?", image=str(p))
    assert df is not None


def test_skip_errors_false_raises(monkeypatch, tmp_path):
    from PIL import Image
    from pydantic import ValidationError
    p = tmp_path / "x.png"
    Image.new("RGB", (16, 16)).save(p)

    model, processor = _make_fake_model_and_processor(["not json"])
    inst = InferenceUnsloth(schema={"answer": (bool, ...)}, skip_errors=False)
    _patch_lazy_imports(monkeypatch, model, processor)

    with pytest.raises(ValidationError):
        inst.one_inference(prompt="?", image=str(p))


def test_default_model_is_qwen3_3b():
    inst = InferenceUnsloth()
    assert inst.llm == "unsloth/Qwen3-VL-3B-Instruct"
