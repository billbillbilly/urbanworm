"""Unsloth-backed multimodal inference for urban-worm.

Tested small VLM checkpoints (pass via ``llm=...``):

* ``unsloth/Qwen3-VL-3B-Instruct``        — fastest, lowest VRAM
* ``unsloth/Qwen3-VL-8B-Instruct``        — strongest 8B-class
* ``unsloth/gemma-3-4b-it``               — Gemma 3 multimodal, balanced
* ``unsloth/Qwen2-VL-2B-Instruct``        — older, very small
* ``unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`` — 4-bit weights

Any vision model that ``unsloth.FastVisionModel`` can load should work; the
checkpoints above are the ones the docstring's behavior was validated against.

Audio is not supported (Unsloth does not ship audio VLMs); the ``audio``
argument is accepted for API parity with :class:`InferenceOllama` but raises
``NotImplementedError`` if used.

This module imports ``unsloth`` lazily, so importing :mod:`urbanworm` does not
pull in torch / transformers / unsloth unless this class is actually
constructed. Install the optional extra::

    pip install "urban-worm[unsloth]"
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd
from tqdm import tqdm

from ..utils.checkpoint import (
    _MockResponse,
    append_inference_checkpoint,
    load_inference_checkpoint,
    restore_ollama_results,
)
from ..utils.utils import (
    extract_json_from_text,
    is_base64,
    is_url,
    load_image_auto,
    response2df,
    sanitize_json_text,
)
from .format import create_format
from .Inference import Inference

logger = logging.getLogger("urbanworm")


def _lazy_imports():
    """Import the heavy stack only when the class is actually instantiated."""
    try:
        import torch  # noqa: F401
        from unsloth import FastVisionModel
    except ImportError as e:
        raise ImportError(
            "InferenceUnsloth requires the optional 'unsloth' extra. "
            "Install with: pip install 'urban-worm[unsloth]'"
        ) from e
    return FastVisionModel


class InferenceUnsloth(Inference):
    """Vision-language model inference via Unsloth's ``FastVisionModel``.

    Args:
        llm: Unsloth-compatible model id or local path. Defaults to
            ``unsloth/Qwen3-VL-3B-Instruct``.
        load_in_4bit: Load weights in 4-bit (bitsandbytes). Big VRAM win,
            small quality cost. Default ``True``.
        max_seq_length: Maximum tokenized prompt+generation length passed to
            ``FastVisionModel.from_pretrained``. Default 4096.
        device: Override the device map. ``None`` lets Unsloth choose
            (typically ``"cuda"`` if available, else ``"mps"`` / ``"cpu"``).
        dtype: Override the compute dtype. ``None`` = auto.
        skip_errors: If ``True`` (default), schema-validation failures yield
            an empty ``Response(responses=[])`` so batch loops continue
            instead of crashing. Mirrors :class:`InferenceOllama`.
        **kwargs: Forwarded to :class:`Inference` (``image``, ``images``,
            ``audio``, ``audios``, ``geo_tagged_data``, ``schema``).
    """

    DEFAULT_MODEL = "unsloth/Qwen3-VL-3B-Instruct"

    def __init__(
        self,
        llm: str | None = None,
        load_in_4bit: bool = True,
        max_seq_length: int = 4096,
        device: str | None = None,
        dtype: Any = None,
        skip_errors: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.llm = llm or self.DEFAULT_MODEL
        self.load_in_4bit = load_in_4bit
        self.max_seq_length = max_seq_length
        self.device = device
        self.dtype = dtype
        self.skip_errors = skip_errors
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Lazy model load
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        FastVisionModel = _lazy_imports()
        logger.info("Loading Unsloth model %s (4bit=%s)", self.llm, self.load_in_4bit)
        self._model, self._processor = FastVisionModel.from_pretrained(
            self.llm,
            load_in_4bit=self.load_in_4bit,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            device_map=self.device,
        )
        FastVisionModel.for_inference(self._model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def one_inference(
        self,
        system: str = "",
        prompt: str = "",
        image: str | list | tuple = None,
        audio: str | list | tuple = None,
        temp: float = 0.0,
        top_k: int = 20,
        top_p: float = 0.8,
        max_new_tokens: int = 512,
    ) -> pd.DataFrame:
        """Run inference on a single image (or list of images for one prompt).

        Returns a one-row DataFrame in the same shape as
        :meth:`InferenceOllama.one_inference`.
        """
        if audio is not None:
            raise NotImplementedError(
                "Unsloth VLMs do not currently support audio input."
            )
        self._ensure_loaded()

        img = image if image is not None else self.img
        if img is None:
            raise ValueError("No image provided to one_inference().")
        imgs = [img] if isinstance(img, str) else list(img)
        # If a list, validate it's flat str list
        if not all(isinstance(i, str) for i in imgs):
            raise TypeError("`image` must be a path/url/base64 string or a flat list of those.")

        schema = create_format(self.schema)
        # Single-prompt call: one batch of one (or one batch of N images for
        # a multi-image-per-prompt scenario).
        responses = self._generate_batch(
            systems=[system],
            prompts=[prompt],
            images_per_prompt=[imgs],
            schema=schema,
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        dic = {"responses": [responses[0].responses], "data": [imgs]}
        try:
            return response2df(dic)
        except Exception as e:
            # Empty/malformed responses (e.g. skip_errors path) — return a
            # minimally-shaped frame instead of crashing.
            logger.warning("one_inference: response2df failed (%s); returning raw frame.", e)
            return pd.DataFrame({"responses": [responses[0].responses], "data": [imgs]})

    def batch_inference(
        self,
        system: str = "",
        prompt: str = "",
        temp: float = 0.0,
        top_k: int = 20,
        top_p: float = 0.8,
        max_new_tokens: int = 512,
        batch_size: int = 1,
        disableProgressBar: bool = False,
        checkpoint_path: str | None = None,
    ) -> pd.DataFrame:
        """Run inference over ``self.batch_images`` with optional GPU batching.

        Args:
            batch_size: Number of items to process per ``model.generate`` call.
                ``1`` (default) matches :class:`InferenceOllama` behavior.
                Larger values trade VRAM for throughput. Practical sweet spot
                for 7-8B VLMs on a 24GB GPU is ~4–8.
            checkpoint_path: Path to a JSONL file for resume-safe
                checkpointing.  Already-completed items are skipped on the
                next run.  Note: resume granularity is one ``batch_size``
                chunk — the last incomplete chunk is re-processed on resume.

        Returns:
            DataFrame, same shape as :meth:`InferenceOllama.batch_inference`.
        """
        self._ensure_loaded()

        imgs = self.batch_images if self.batch_images is not None else self.imgs
        if not imgs:
            raise ValueError("No images to run inference on.")

        # Normalize each item to a list of image strings (multi-image-per-
        # prompt is allowed; flat str becomes a 1-element list).
        items: list[list[str]] = [
            [it] if isinstance(it, str) else list(it) for it in imgs
        ]

        schema = create_format(self.schema)

        # ── resume from checkpoint ───────────────────────────────────────
        done_records = load_inference_checkpoint(checkpoint_path) if checkpoint_path else []
        bs = max(1, int(batch_size))
        # Align start to the nearest chunk boundary so we never replay a
        # half-finished chunk (at most bs-1 items are re-run on resume).
        start_idx = (len(done_records) // bs) * bs
        dic = restore_ollama_results(done_records[:start_idx])

        n = len(items)
        with tqdm(total=n - start_idx, desc="Processing", ncols=75, disable=disableProgressBar) as pbar:
            for start in range(start_idx, n, bs):
                chunk = items[start:start + bs]
                try:
                    chunk_resp = self._generate_batch(
                        systems=[system] * len(chunk),
                        prompts=[prompt] * len(chunk),
                        images_per_prompt=chunk,
                        schema=schema,
                        temp=temp,
                        top_k=top_k,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                    )
                    for j, r in enumerate(chunk_resp):
                        img_idx = start + j
                        responses = r.responses
                        dic["responses"].append(responses)
                        dic["data"].append(imgs[img_idx])

                        if checkpoint_path:
                            try:
                                responses_dump = [item.model_dump() for item in responses]
                            except Exception:
                                responses_dump = [dict(item) for item in responses] if responses else []
                            append_inference_checkpoint(checkpoint_path, {
                                "idx": img_idx,
                                "responses": responses_dump,
                                "data": imgs[img_idx] if isinstance(imgs[img_idx], str)
                                        else list(imgs[img_idx]),
                            })
                except Exception as e:
                    logger.warning(
                        "batch_inference: chunk [%d, %d) failed (%s); "
                        "filling stub responses.", start, start + len(chunk), e,
                    )
                    for k in range(len(chunk)):
                        img_idx = start + k
                        dic["responses"].append([])
                        dic["data"].append(imgs[img_idx])
                pbar.update(len(chunk))

        self.results = dic
        return self.to_df(output=True)

    def to_df(self, output: bool = True) -> pd.DataFrame | None:
        """Convert ``self.results`` into a DataFrame. Mirrors InferenceOllama."""
        if self.results is None:
            return None
        try:
            self.df = response2df(self.results)
        except Exception as e:
            logger.warning("to_df: response2df failed (%s); returning raw dict.", e)
            self.df = pd.DataFrame(self.results)
        return self.df if output else None

    # ------------------------------------------------------------------
    # Internal: batched generate
    # ------------------------------------------------------------------
    def _generate_batch(
        self,
        systems: Sequence[str],
        prompts: Sequence[str],
        images_per_prompt: Sequence[Sequence[str]],
        schema,
        temp: float,
        top_k: int,
        top_p: float,
        max_new_tokens: int,
    ):
        """Run one ``model.generate`` over ``len(prompts)`` items.

        ``images_per_prompt[i]`` is the list of image references attached to
        prompt ``i``. We resolve each reference (path/url/base64) into a PIL
        image via ``load_image_auto``, then feed everything to the processor.
        """
        import torch

        assert len(systems) == len(prompts) == len(images_per_prompt)

        # 1) Build chat-template messages and resolve images per item
        templated_texts: list[str] = []
        loaded_images: list[list] = []
        for system, prompt, img_refs in zip(systems, prompts, images_per_prompt, strict=True):
            content = [{"type": "image"} for _ in img_refs]
            content.append({"type": "text", "text": self._with_schema_hint(prompt, schema)})
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": content})

            text = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            templated_texts.append(text)
            loaded_images.append([load_image_auto(self._coerce_image(r)) for r in img_refs])

        # 2) Processor → padded tensor batch
        # The HF VLM processor accepts a list[list[PIL.Image]] and a list[str].
        # Some processors prefer flat list[PIL.Image] when each prompt has
        # exactly one image — handle that for the common case.
        processor_images: Any
        if all(len(x) == 1 for x in loaded_images):
            processor_images = [x[0] for x in loaded_images]
        else:
            processor_images = loaded_images

        inputs = self._processor(
            text=templated_texts,
            images=processor_images,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        # 3) Generate
        do_sample = bool(temp and temp > 0)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temp if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
        )
        # transformers complains if you pass None — drop them.
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # 4) Decode only the newly generated tokens
        input_lens = inputs["input_ids"].shape[1] if inputs["input_ids"].dim() == 2 else None
        if input_lens is not None:
            new_tokens = output_ids[:, input_lens:]
        else:
            new_tokens = output_ids
        decoded = self._processor.batch_decode(new_tokens, skip_special_tokens=True)

        # Defensive: if the processor returned a mismatched count, trim/pad
        # so downstream output cardinality always matches the input batch.
        n_in = len(prompts)
        if len(decoded) != n_in:
            logger.warning(
                "batch_decode returned %d items for a %d-item batch; aligning.",
                len(decoded), n_in,
            )
            if len(decoded) > n_in:
                decoded = decoded[:n_in]
            else:
                decoded = list(decoded) + [""] * (n_in - len(decoded))

        # 5) Parse each item against the schema
        return [self._parse_to_schema(text, schema) for text in decoded]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _with_schema_hint(prompt: str, schema) -> str:
        """Append a JSON-only instruction so the model emits parseable output."""
        try:
            schema_text = schema.model_json_schema()
        except Exception:
            return prompt
        import json
        return (
            f"{prompt}\n\n"
            "Respond with ONLY a JSON object matching this schema. "
            "Do not include any prose, markdown, or explanation outside the JSON.\n"
            f"Schema: {json.dumps(schema_text, separators=(',', ':'))}"
        )

    @staticmethod
    def _coerce_image(ref: str) -> str:
        """Pass-through for now — load_image_auto already handles all of
        url / path / base64 / data-uri."""
        if not isinstance(ref, str):
            raise TypeError(f"Image reference must be str, got {type(ref).__name__}")
        if not (is_url(ref) or is_base64(ref) or len(ref) > 0):
            raise ValueError("Empty image reference.")
        return ref

    def _parse_to_schema(self, raw_text: str, schema):
        """Three-stage validation: direct → sanitized → balanced-extracted.

        On exhaustion, behavior depends on ``self.skip_errors``.
        """
        try:
            return schema.model_validate_json(raw_text)
        except Exception as e:
            logger.debug("direct JSON validation failed: %s", e)

        repaired = sanitize_json_text(str(raw_text))
        try:
            return schema.model_validate_json(repaired)
        except Exception as e:
            logger.debug("validation after sanitize failed: %s", e)

        extracted = extract_json_from_text(repaired) or repaired
        try:
            return schema.model_validate_json(extracted)
        except Exception:
            if self.skip_errors:
                logger.warning(
                    "Could not validate model output against schema; "
                    "returning empty response."
                )
                return schema(responses=[])
            raise
