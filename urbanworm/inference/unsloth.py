"""Unsloth-backed multimodal inference for urban-worm.

Tested small VLM checkpoints (pass via ``llm=...``):

* ``unsloth/Qwen3-VL-4B-Instruct``            — recommended default, T4-friendly
* ``unsloth/Qwen3-VL-3B-Instruct``            — lowest VRAM
* ``unsloth/Qwen3-VL-8B-Instruct``            — strongest 8B-class
* ``unsloth/gemma-3-4b-it``                   — Gemma 3 multimodal, balanced
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
from .format import create_format, schema_dict
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


# ── Known error patterns that indicate a Torch compile / Accelerate conflict ─
_COMPILE_HOOK_PATTERNS = (
    "torch.compiler.disable",
    "AlignDevicesHook",
    "Skip calling",
    "UNSLOTH_COMPILE_DISABLE",
    "accumulated_recompile_limit",
)


def configure_runtime(disable_compile: bool = True) -> None:
    """Set environment variables that control Unsloth / Torch compilation.

    **Must be called before** :class:`InferenceUnsloth` is instantiated (and
    before ``import unsloth``), because the flags are read at import time.
    ``InferenceUnsloth.__init__`` calls this automatically when
    ``disable_compile=True`` (the default).

    Args:
        disable_compile: When ``True`` (default), sets
            ``UNSLOTH_COMPILE_DISABLE=1``, ``UNSLOTH_DISABLE_FAST_GENERATION=1``,
            and ``TORCH_COMPILE_DISABLE=1``.  Trades a small throughput loss for
            stability on long inference jobs — strongly recommended for runs
            over ~10 k samples or when Accelerate device hooks are in use.
    """
    import os
    if disable_compile:
        os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        logger.debug("Unsloth/Torch compilation disabled via environment variables.")


def clear_compile_cache() -> None:
    """Remove the Unsloth compiled-model cache from the system temp directory.

    Useful when a stale cache causes ``AlignDevicesHook`` / Torch-Dynamo
    recompile errors.  Safe to call before loading a model.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    candidates = {Path(tempfile.gettempdir()) / "unsloth_compiled_cache"}
    # Only add env-derived paths when the variable is actually set and
    # non-empty; Path("") resolves to cwd and would delete the wrong dir.
    for var in ("TEMP", "TMP"):
        val = os.environ.get(var, "")
        if val:
            candidates.add(Path(val) / "unsloth_compiled_cache")
    for p in candidates:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            logger.info("Removed Unsloth compile cache: %s", p)


def _log_runtime_versions() -> None:
    """Log key dependency versions at INFO level for debugging."""
    import importlib

    import torch
    logger.info("torch %s | CUDA %s | GPU: %s",
                torch.__version__,
                torch.version.cuda or "n/a",
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    for pkg in ("transformers", "accelerate", "unsloth"):
        try:
            mod = importlib.import_module(pkg)
            logger.info("%s %s", pkg, getattr(mod, "__version__", "unknown"))
        except ImportError:
            logger.info("%s: not installed", pkg)


def _classify_error(exc: Exception) -> str | None:
    """Return a hint string if ``exc`` matches a known recoverable pattern."""
    msg = str(exc)
    if any(p in msg for p in _COMPILE_HOOK_PATTERNS):
        return (
            "Torch compile + Accelerate hook conflict detected. "
            "Re-run with disable_compile=True (already the default) and call "
            "clear_compile_cache() before loading the model, or set "
            "UNSLOTH_COMPILE_DISABLE=1 in your environment."
        )
    if "out of memory" in msg.lower():
        return "CUDA out-of-memory. Reduce batch_size or use a smaller/more-quantised model."
    if "expected scalar type" in msg.lower() and ("bfloat16" in msg.lower() or "float16" in msg.lower()):
        return (
            "Dtype mismatch between image processor (float32) and model (bf16/fp16). "
            "This is handled automatically in _generate_batch; if it still occurs, "
            "try passing dtype=torch.bfloat16 explicitly to InferenceUnsloth."
        )
    return None


class InferenceUnsloth(Inference):
    """Vision-language model inference via Unsloth's ``FastVisionModel``.

    Args:
        llm: Unsloth-compatible model id or local path. Defaults to
            ``unsloth/Qwen3-VL-3B-Instruct``.
        load_in_4bit: Load weights in 4-bit (bitsandbytes). Big VRAM win,
            small quality cost. Default ``True``.
        max_seq_length: Maximum tokenized prompt+generation length passed to
            ``FastVisionModel.from_pretrained``. Default 4096.
        device: Override the device map string (e.g. ``"cuda:0"``,
            ``"auto"``).  ``None`` (default) auto-detects: uses
            ``device_map="auto"`` when multiple CUDA GPUs are present so the
            model is spread across all of them, otherwise falls back to a
            single GPU or CPU.
        max_memory: Per-device VRAM budget passed to ``from_pretrained`` as
            ``max_memory``.  ``None`` (default) auto-computes 90 % of each
            GPU's total capacity when multi-GPU is detected.  Example:
            ``{0: "10GiB", 1: "10GiB"}``.
        model_dir: Local directory where HuggingFace model weights are cached.
            Passed as ``cache_dir`` to ``FastVisionModel.from_pretrained``.
            ``None`` (default) uses the HuggingFace default
            (``~/.cache/huggingface/hub`` or the ``HF_HOME`` env var).
        dtype: Override the compute dtype. ``None`` = auto.
        disable_compile: Disable Unsloth / Torch auto-compilation before the
            model is loaded. Default ``True``.  Strongly recommended for
            production / large-scale inference jobs — prevents the
            ``AlignDevicesHook`` / Torch-Dynamo recompile crashes that occur
            when Accelerate device hooks conflict with compiled code.  Sets
            ``UNSLOTH_COMPILE_DISABLE=1``, ``UNSLOTH_DISABLE_FAST_GENERATION=1``,
            and ``TORCH_COMPILE_DISABLE=1`` via :func:`configure_runtime`.
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
        max_memory: dict | None = None,
        model_dir: str | None = None,
        dtype: Any = None,
        disable_compile: bool = True,
        skip_errors: bool = True,
        **kwargs,
    ) -> None:
        # Configure runtime env-vars before any unsloth/torch import happens.
        configure_runtime(disable_compile)
        super().__init__(**kwargs)
        self.llm = llm or self.DEFAULT_MODEL
        self.load_in_4bit = load_in_4bit
        self.max_seq_length = max_seq_length
        self.device = device
        self.max_memory = max_memory
        self.model_dir = model_dir
        self.dtype = dtype
        self.disable_compile = disable_compile
        self.skip_errors = skip_errors
        self._model = None
        self._processor = None
        self._model_dtype = None   # set by _ensure_loaded; cached to avoid repeated next(parameters())

    # ------------------------------------------------------------------
    # Lazy model load
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        FastVisionModel = _lazy_imports()
        import torch

        # ── resolve device_map ───────────────────────────────────────────
        device_map = self.device
        max_memory = self.max_memory

        if device_map is None:
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                # Spread the model across all visible GPUs.  Reserve 90 % of
                # each GPU's capacity so the activation memory has headroom.
                device_map = "auto"
                if max_memory is None:
                    # Use *free* VRAM (not total) so that memory already
                    # occupied by the CUDA runtime, driver, and any other
                    # processes is not double-counted.  Reserve 90 % of what
                    # is currently free to leave headroom for activations.
                    # mem_get_info returns (free_bytes, total_bytes).
                    max_memory = {
                        i: f"{int(torch.cuda.mem_get_info(i)[0] * 0.90 / (1024 ** 3))}GiB"
                        for i in range(n_gpus)
                    }
                logger.info(
                    "Detected %d GPUs; using device_map='auto' with max_memory=%s",
                    n_gpus, max_memory,
                )
            elif n_gpus == 1:
                device_map = "cuda:0"
            else:
                device_map = "cpu"

        _log_runtime_versions()
        logger.info("Loading Unsloth model %s (4bit=%s)", self.llm, self.load_in_4bit)
        load_kwargs: dict = dict(
            load_in_4bit=self.load_in_4bit,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            device_map=device_map,
        )
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        if self.model_dir is not None:
            load_kwargs["cache_dir"] = self.model_dir
            logger.info("Model cache directory: %s", self.model_dir)
        self._model, self._processor = FastVisionModel.from_pretrained(
            self.llm, **load_kwargs,
        )
        FastVisionModel.for_inference(self._model)
        # Cache the model's compute dtype so _generate_batch can cast image
        # tensors to match (processor always emits float32; BF16/FP16 models
        # raise "expected scalar type BFloat16 but found Float" otherwise).
        self._model_dtype = next(self._model.parameters()).dtype
        logger.info("Model compute dtype: %s", self._model_dtype)

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
        task_chunk_size: int | None = None,
        disableProgressBar: bool = False,
        checkpoint_path: str | None = None,
        failed_log_path: str | None = None,
    ) -> pd.DataFrame:
        """Run inference over ``self.batch_images`` with optional GPU batching.

        Args:
            batch_size: Number of items per ``model.generate`` call (the GPU
                batch).  Larger values trade VRAM for throughput; sweet spot
                for 7–8 B VLMs on a 24 GB GPU is ~4–8.
            task_chunk_size: Logical job partition size, independent of
                ``batch_size``.  When set, the dataset is split into segments
                of this size and progress is reported at the task-chunk level,
                making long runs (e.g. 144 k samples) easier to monitor.
                ``None`` (default) disables task-level chunking.
                Example: ``batch_size=4, task_chunk_size=1000`` → ~145 task
                chunks, each processed internally in batches of 4.
            checkpoint_path: Path to a JSONL file for resume-safe
                checkpointing.  Already-completed items are skipped on the
                next run.
            failed_log_path: Optional path to a CSV file where permanently
                failed sample indices and error messages are appended.  Lets
                you rerun only the failures later.

        Returns:
            DataFrame, same shape as :meth:`InferenceOllama.batch_inference`.
        """
        import csv

        import torch

        self._ensure_loaded()

        imgs = self.batch_images if self.batch_images is not None else self.imgs
        if not imgs:
            raise ValueError("No images to run inference on.")

        items: list[list[str]] = [
            [it] if isinstance(it, str) else list(it) for it in imgs
        ]
        schema = create_format(self.schema)
        bs = max(1, int(batch_size))
        n = len(items)

        # ── resume from checkpoint ───────────────────────────────────────
        done_records = load_inference_checkpoint(checkpoint_path) if checkpoint_path else []
        # Resume from exactly where we left off.  The old formula
        # ``(len // bs) * bs`` rounded down to the nearest batch boundary,
        # which caused the trailing partial batch to be re-processed on every
        # restart.  Since checkpoints are written per-item (not per-batch),
        # len(done_records) is always the exact number of completed items.
        start_idx = len(done_records)
        dic = restore_ollama_results(done_records)

        # ── task-chunk boundaries for progress reporting ─────────────────
        tcs = int(task_chunk_size) if task_chunk_size and task_chunk_size > 0 else None
        n_task_chunks = ((n - start_idx) + tcs - 1) // tcs if tcs else 1

        def _save_checkpoint(img_idx, responses):
            if not checkpoint_path:
                return
            try:
                responses_dump = [r.model_dump() for r in responses]
            except Exception:
                responses_dump = [dict(r) for r in responses] if responses else []
            append_inference_checkpoint(checkpoint_path, {
                "idx": img_idx,
                "responses": responses_dump,
                "data": imgs[img_idx] if isinstance(imgs[img_idx], str) else list(imgs[img_idx]),
            })

        def _log_failed(img_idx, error_msg):
            if not failed_log_path:
                return
            import os
            write_header = not os.path.exists(failed_log_path)
            with open(failed_log_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if write_header:
                    w.writerow(["idx", "data", "error"])
                w.writerow([img_idx,
                             imgs[img_idx] if isinstance(imgs[img_idx], str) else str(imgs[img_idx]),
                             error_msg[:500]])

        task_chunk_idx = 0
        with tqdm(total=n - start_idx, desc="Processing", ncols=80,
                  disable=disableProgressBar) as pbar:
            for start in range(start_idx, n, bs):
                # ── task-chunk boundary logging ──────────────────────────
                if tcs:
                    rel = start - start_idx
                    if rel % tcs == 0:
                        task_chunk_idx = rel // tcs
                        logger.info(
                            "Task chunk %d/%d: rows [%d, %d)",
                            task_chunk_idx + 1, n_task_chunks,
                            start, min(start + tcs, n),
                        )

                chunk = items[start:start + bs]
                responses_list = self._run_chunk_with_retry(
                    chunk=chunk,
                    start=start,
                    system=system,
                    prompt=prompt,
                    schema=schema,
                    temp=temp,
                    top_k=top_k,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )

                for k, (responses, err) in enumerate(responses_list):
                    img_idx = start + k
                    dic["responses"].append(responses)
                    dic["data"].append(imgs[img_idx])
                    if responses:
                        _save_checkpoint(img_idx, responses)
                    if err is not None:
                        _log_failed(img_idx, err)

                # Proactively free reserved-but-unused allocator pool.
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                pbar.update(len(chunk))

        self.results = dic
        return self.to_df(output=True)

    def _run_chunk_with_retry(
        self,
        chunk: list,
        start: int,
        system: str,
        prompt: str,
        schema,
        temp: float,
        top_k: int,
        top_p: float,
        max_new_tokens: int,
    ) -> list[tuple[list, str | None]]:
        """Run ``chunk`` through the model, retrying with halved batch size on failure.

        Returns a list of ``(responses, error_msg)`` tuples — one per item.
        ``error_msg`` is ``None`` on success and a string on permanent failure.
        Retry ladder: ``len(chunk)`` → ``len(chunk) // 2`` → … → ``1`` → stub.
        """
        import torch

        # results[i] = (responses, error) or None if not yet processed
        results: list[tuple[list, str | None] | None] = [None] * len(chunk)

        current_bs = len(chunk)
        while current_bs >= 1:
            pending = [i for i, r in enumerate(results) if r is None]
            if not pending:
                break

            for sub_start in range(0, len(pending), current_bs):
                sub_idx = pending[sub_start: sub_start + current_bs]
                sub_items = [chunk[i] for i in sub_idx]
                try:
                    sub_resp = self._generate_batch(
                        systems=[system] * len(sub_items),
                        prompts=[prompt] * len(sub_items),
                        images_per_prompt=sub_items,
                        schema=schema,
                        temp=temp,
                        top_k=top_k,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                    )
                    for list_pos, i in enumerate(sub_idx):
                        results[i] = (sub_resp[list_pos].responses, None)
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                except Exception as exc:
                    hint = _classify_error(exc)
                    if hint:
                        logger.warning("[urbanworm|HINT] %s", hint)
                    if current_bs > 1:
                        # Retry with a smaller sub-batch.  Log the failure so
                        # it is visible in the run log rather than silently
                        # discarded — previously these exceptions were swallowed
                        # with no diagnostic output at all.
                        logger.warning(
                            "_run_chunk_with_retry: batch of %d failed "
                            "(will retry at bs=%d): %s: %s",
                            current_bs, current_bs // 2,
                            type(exc).__name__, str(exc)[:200],
                        )
                    if current_bs == 1:
                        # Final retry exhausted.
                        if not self.skip_errors:
                            raise
                        err_str = str(exc)
                        for i in sub_idx:
                            abs_idx = start + i
                            logger.warning(
                                "batch_inference: item %d permanently failed (%s); "
                                "filling stub.", abs_idx, err_str[:120],
                            )
                            results[i] = ([], err_str)
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

            current_bs //= 2

        # Any remaining None slots (shouldn't happen, but be defensive)
        for i, r in enumerate(results):
            if r is None:
                results[i] = ([], "unknown — slot unfilled after retry ladder")

        return results  # type: ignore[return-value]

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
    def _apply_dtype_hooks_once(self) -> None:
        """
        Register forward pre-hooks on the vision encoder and every ViT block
        so that all floating-point inputs are cast to the model dtype before
        each forward pass.  Called once on the first _generate_batch
        invocation, after the model has been lazily loaded.

        Root cause: with device_map='auto' across multiple GPUs, accelerate
        splits ViT blocks between devices and moves tensors without re-casting
        their dtype.  The image processor emits float32 pixel_values; these
        reach bf16 layer norms inside the ViT and raise:
            RuntimeError: expected scalar type BFloat16 but found Float

        Fix: PyTorch's register_forward_pre_hook fires before accelerate's own
        device-dispatch hook (accelerate monkey-patches module.forward directly,
        so PyTorch's hook machinery runs first).  We cast every floating tensor
        to the model dtype before accelerate touches it.
        """
        import torch

        if getattr(self, "_dtype_hooks_applied", False):
            return

        # Guard: model must be live before we can inspect its dtype or walk its
        # modules.  _ensure_loaded() is always called before _generate_batch,
        # so this path is only reachable if someone calls _apply_dtype_hooks_once
        # directly without loading the model first.
        model = getattr(self, "_model", None)
        if model is None:
            return

        # Mark applied only after confirming the model is present, so a future
        # call after the model IS loaded will still apply the hooks correctly.
        self._dtype_hooks_applied = True

        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            return

        if model_dtype == getattr(torch, "float32", None):
            return  # image processor and model already agree; nothing to do

        def _cast_float_args(module, args):
            return tuple(
                a.to(model_dtype)
                if (
                    isinstance(a, torch.Tensor)
                    and a.is_floating_point()
                    and a.dtype != model_dtype
                )
                else a
                for a in args
            )

        # 1. Hook the top-level vision encoder (pixel_values entry point).
        vision_mod = None
        for _attr in ("visual", "vision_model", "vision_tower", "image_encoder"):
            _candidate = getattr(model, _attr, None)
            if _candidate is None:
                _inner = getattr(model, "model", None)
                _candidate = getattr(_inner, _attr, None) if _inner is not None else None
            if _candidate is not None:
                vision_mod = _candidate
                vision_mod.register_forward_pre_hook(_cast_float_args)
                break

        # 2. Hook every individual ViT transformer block so that inter-block
        #    hidden_states are cast when blocks are split across GPUs.
        #    Blocks are identified by: has norm1 + (attn or self_attn).
        _seen_ids: set = set()
        _search_roots = [model]
        if hasattr(model, "model"):
            _search_roots.append(model.model)
        if vision_mod is not None:
            _search_roots.append(vision_mod)

        for _root in _search_roots:
            for _name, _submod in _root.named_modules():
                if id(_submod) in _seen_ids:
                    continue
                if hasattr(_submod, "norm1") and (
                    hasattr(_submod, "attn") or hasattr(_submod, "self_attn")
                ):
                    _seen_ids.add(id(_submod))
                    _submod.register_forward_pre_hook(_cast_float_args)


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
        self._apply_dtype_hooks_once()
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

        # With device_map="auto" the model is split across several GPUs and
        # has no single .device attribute.  Move inputs to whichever device
        # holds the first parameter (= the embedding / input layer).
        first_device = next(self._model.parameters()).device
        inputs = self._processor(
            text=templated_texts,
            images=processor_images,
            return_tensors="pt",
            padding=True,
        ).to(first_device)

        # The processor always emits pixel_values (and related image tensors)
        # as float32, but BF16/FP16 models expect their native dtype.
        # Cast every floating-point tensor in the batch to the model dtype.
        # Integer tensors (input_ids, attention_mask, etc.) are left as-is.
        # Guard with hasattr so the block is safely skipped when torch is
        # mocked (e.g. in unit tests that stub out the heavy stack).
        _TensorCls = getattr(torch, "Tensor", None)
        if _TensorCls is not None:
            model_dtype = getattr(self, "_model_dtype", None) or next(self._model.parameters()).dtype
            for key in list(inputs.keys()):
                val = inputs[key]
                if isinstance(val, _TensorCls) and val.is_floating_point():
                    inputs[key] = val.to(model_dtype)

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
        """Append a JSON-only instruction so the model emits parseable output.

        Uses schema_dict(..., inline_refs=True) instead of model_json_schema()
        directly so that Pydantic's $defs/$ref indirection is flattened into a
        single, self-contained JSON object.  Small VLMs often ignore the wrapper
        structure when they encounter an unfamiliar $ref and return the inner
        fields directly, which fails schema validation.
        """
        try:
            schema_text = schema_dict(schema, inline_refs=True)
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

    @staticmethod
    def _close_truncated_json(text: str) -> str:
        """Close any unclosed JSON braces/brackets caused by max_new_tokens cutoff.

        Walks the text character-by-character, tracking open structures on a
        stack while skipping string contents (including escaped quotes).  Any
        unclosed ``{`` / ``[`` are closed in reverse order at the end.

        Example::

            '{"responses":[{"is_outdoor":true'
            → '{"responses":[{"is_outdoor":true}]}'
        """
        stack: list[str] = []
        in_string = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]" and stack and stack[-1] == ch:
                    stack.pop()
        return text + "".join(reversed(stack))

    def _parse_to_schema(self, raw_text: str, schema):
        """Four-stage validation: direct → sanitized → extracted → bracket-closed.

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
        except Exception as e:
            logger.debug("validation after extract failed: %s", e)

        # Stage 4: the output may have been cut off by max_new_tokens before
        # the closing braces/brackets were generated.  Close any open structures
        # and retry once more.
        closed = self._close_truncated_json(extracted)
        try:
            return schema.model_validate_json(closed)
        except Exception:
            if self.skip_errors:
                logger.warning(
                    "Could not validate model output against schema; "
                    "returning empty response."
                )
                return schema(responses=[])
            raise
