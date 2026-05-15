"""API-backed multimodal inference for urbanworm.

Wraps Anthropic Claude, OpenAI, and Google Gemini behind the same
``one_inference`` / ``batch_inference`` surface as the local backends.

Install the optional extra::

    pip install "urban-worm[api]"

or install individual provider SDKs as needed::

    pip install anthropic          # for provider="anthropic"
    pip install openai             # for provider="openai"
    pip install google-genai       # for provider="google"

Usage::

    from urbanworm import InferenceAPI

    schema = {"condition": (str, ...), "reason": (str, None)}

    inf = InferenceAPI(
        llm="claude-opus-4-6",
        provider="anthropic",
        api_key="sk-ant-...",   # or set ANTHROPIC_API_KEY env var
        geo_tagged_data=gtd,
        schema=schema,
    )
    df = inf.batch_inference(
        system="You are an urban-environment analyst.",
        prompt="Describe the building condition in the image.",
        checkpoint_path="./run/labels.jsonl",
    )

Tested models:

* Anthropic — ``claude-opus-4-6``, ``claude-sonnet-4-6``, ``claude-haiku-4-5-20251001``
* OpenAI    — ``gpt-4o``, ``gpt-4o-mini``
* Google    — ``gemini-2.0-flash``, ``gemini-1.5-pro``
"""
from __future__ import annotations

import base64
import json
import logging
import os
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
    encode_image_to_base64,
    is_image_path,
    is_url,
    response2df,
)
from .format import schema as _make_inner_schema, schema_dict
from .Inference import Inference

logger = logging.getLogger("urbanworm")

_PROVIDERS = ("anthropic", "openai", "google")

# ---------------------------------------------------------------------------
# Lazy provider imports
# ---------------------------------------------------------------------------

def _import_anthropic():
    try:
        import anthropic
        return anthropic
    except ImportError as e:
        raise ImportError(
            "provider='anthropic' requires the anthropic SDK. "
            "Install with: pip install anthropic"
        ) from e


def _import_openai():
    try:
        import openai
        return openai
    except ImportError as e:
        raise ImportError(
            "provider='openai' requires the openai SDK. "
            "Install with: pip install openai"
        ) from e


def _import_google():
    try:
        import google.genai as genai
        return genai
    except ImportError:
        pass
    try:
        import google.generativeai as genai  # type: ignore[no-redef]
        return genai
    except ImportError as e:
        raise ImportError(
            "provider='google' requires the google-genai SDK. "
            "Install with: pip install google-genai"
        ) from e


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _to_b64(image: str) -> str:
    """Return a base64 string.  Encodes a local file; passes b64 through."""
    if is_image_path(image):
        return encode_image_to_base64(image)
    # Already base64 (or something else) — return as-is
    return image


def _mime_from_path(path: str) -> str:
    ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class InferenceAPI(Inference):
    """Vision-language inference via hosted API providers.

    Args:
        llm: Model name for the chosen provider, e.g. ``"claude-opus-4-6"``,
            ``"gpt-4o"``, ``"gemini-2.0-flash"``.
        provider: One of ``"anthropic"``, ``"openai"``, or ``"google"``.
        api_key: API key.  If ``None``, each provider falls back to its
            standard environment variable (``ANTHROPIC_API_KEY``,
            ``OPENAI_API_KEY``, ``GOOGLE_API_KEY``).
        max_tokens: Maximum tokens to generate per call.  Default 1024.
        skip_errors: If ``True`` (default), API / parse errors per image are
            logged and that image gets an empty response instead of crashing
            the batch.
        **kwargs: Forwarded to :class:`~urbanworm.inference.Inference`
            (``image``, ``images``, ``geo_tagged_data``, ``schema``).
    """

    def __init__(
        self,
        llm: str,
        provider: str = "anthropic",
        api_key: str | None = None,
        max_tokens: int = 1024,
        skip_errors: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if provider not in _PROVIDERS:
            raise ValueError(
                f"provider must be one of {_PROVIDERS!r}; got {provider!r}"
            )
        self.llm = llm
        self.provider = provider
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.skip_errors = skip_errors

    # ------------------------------------------------------------------
    # Public API  (mirrors InferenceOllama)
    # ------------------------------------------------------------------

    def one_inference(
        self,
        system: str = "",
        prompt: str = "",
        image: str | list | tuple | None = None,
        audio: str | list | tuple | None = None,
        **_kwargs,
    ) -> pd.DataFrame:
        """Run inference on a single image.

        Args:
            system: System prompt.
            prompt: User prompt.
            image: Path, URL, or base64 string (or list of those for
                multi-image-per-prompt).
            audio: Accepted for API parity; raises ``NotImplementedError``.

        Returns:
            One-row DataFrame.
        """
        if audio is not None:
            raise NotImplementedError(
                "InferenceAPI does not support audio input."
            )

        img = image if image is not None else self.img
        if img is None:
            raise ValueError("No image provided to one_inference().")
        imgs = [img] if isinstance(img, str) else list(img)

        response_dict = self._call(system, prompt, imgs)
        dic = {
            "responses": [[_MockResponse(response_dict)]],
            "data": [imgs],
        }
        try:
            return response2df(dic)
        except Exception as e:
            logger.warning("one_inference: response2df failed (%s); returning raw.", e)
            return pd.DataFrame({"responses": [[response_dict]], "data": [imgs]})

    def batch_inference(
        self,
        system: str = "",
        prompt: str = "",
        disableProgressBar: bool = False,
        checkpoint_path: str | None = None,
        **_kwargs,
    ) -> pd.DataFrame:
        """Run inference over all collected images.

        Args:
            system: System prompt.
            prompt: User prompt.
            disableProgressBar: Suppress tqdm bar.
            checkpoint_path: Path to a JSONL file for resume-safe
                checkpointing.  On the next run items already in the file
                are skipped automatically.

        Returns:
            DataFrame — same shape as the other backends.
        """
        imgs = self.batch_images if self.batch_images is not None else self.imgs
        if not imgs:
            raise ValueError("No images to run inference on.")

        # ── resume from checkpoint ───────────────────────────────────────
        done_records = load_inference_checkpoint(checkpoint_path) if checkpoint_path else []
        start_idx = len(done_records)

        dic = restore_ollama_results(done_records)

        # ── process remaining images ─────────────────────────────────────
        for i in tqdm(
            range(start_idx, len(imgs)),
            desc="Processing",
            ncols=75,
            disable=disableProgressBar,
        ):
            img = imgs[i]
            img_list = [img] if isinstance(img, str) else list(img)
            try:
                response_dict = self._call(system, prompt, img_list)
                wrapped = [_MockResponse(response_dict)]
            except Exception as e:
                logger.warning(
                    "batch_inference: image %d failed (%s). Continuing.", i, e
                )
                if self.skip_errors:
                    response_dict = {}
                    wrapped = []
                else:
                    raise

            dic["responses"].append(wrapped)
            dic["data"].append(img)

            if checkpoint_path:
                append_inference_checkpoint(checkpoint_path, {
                    "idx": i,
                    "responses": [response_dict],
                    "data": img if isinstance(img, str) else list(img),
                })

        self.results = dic
        return self.to_df(output=True)

    def to_df(self, output: bool = True) -> pd.DataFrame | None:
        """Convert ``self.results`` into a DataFrame."""
        if self.results is None:
            return None
        try:
            self.df = response2df(self.results)
        except Exception as e:
            logger.warning("to_df: response2df failed (%s); returning raw frame.", e)
            self.df = pd.DataFrame(self.results)
        return self.df if output else None

    # ------------------------------------------------------------------
    # Internal: dispatch to provider
    # ------------------------------------------------------------------

    def _call(self, system: str, prompt: str, images: list[str]) -> dict:
        """Make one API call and return a validated response dict."""
        inner_model = _make_inner_schema(self.schema)
        field_schema = schema_dict(inner_model, inline_refs=True)

        if self.provider == "anthropic":
            return self._call_anthropic(system, prompt, images, field_schema)
        elif self.provider == "openai":
            return self._call_openai(system, prompt, images, field_schema)
        else:
            return self._call_google(system, prompt, images, field_schema)

    # ── Anthropic ────────────────────────────────────────────────────────

    def _call_anthropic(
        self, system: str, prompt: str, images: list[str], field_schema: dict
    ) -> dict:
        anthropic = _import_anthropic()
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)

        content: list[dict] = []
        for img in images:
            if is_url(img):
                content.append({
                    "type": "image",
                    "source": {"type": "url", "url": img},
                })
            else:
                b64 = _to_b64(img)
                mime = _mime_from_path(img) if is_image_path(img) else "image/png"
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                })
        content.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model=self.llm,
            max_tokens=self.max_tokens,
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": content}],
            tools=[{
                "name": "respond",
                "description": "Provide a structured response.",
                "input_schema": field_schema,
            }],
            tool_choice={"type": "tool", "name": "respond"},
        )
        # Extract tool-use block input
        for block in response.content:
            if block.type == "tool_use":
                return block.input  # already a dict

        raise ValueError("Anthropic response contained no tool_use block.")

    # ── OpenAI ───────────────────────────────────────────────────────────

    def _call_openai(
        self, system: str, prompt: str, images: list[str], field_schema: dict
    ) -> dict:
        openai = _import_openai()
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=api_key)

        user_content: list[dict] = []
        for img in images:
            if is_url(img):
                url = img
            else:
                b64 = _to_b64(img)
                mime = _mime_from_path(img) if is_image_path(img) else "image/png"
                url = f"data:{mime};base64,{b64}"
            user_content.append({"type": "image_url", "image_url": {"url": url}})
        user_content.append({"type": "text", "text": prompt})

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=self.llm,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": field_schema,
                    "strict": False,
                },
            },
            max_tokens=self.max_tokens,
        )
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)

    # ── Google ───────────────────────────────────────────────────────────

    def _call_google(
        self, system: str, prompt: str, images: list[str], field_schema: dict
    ) -> dict:
        genai = _import_google()
        api_key = self.api_key or os.getenv("GOOGLE_API_KEY")

        # Support both google-genai (newer) and google-generativeai (legacy)
        try:
            # google-genai >= 1.0 path
            client = genai.Client(api_key=api_key)
            parts: list[Any] = []
            for img in images:
                if is_url(img):
                    import requests as _req
                    img_bytes = _req.get(img, timeout=30).content
                    parts.append(genai.types.Part.from_bytes(
                        data=img_bytes, mime_type="image/jpeg"
                    ))
                else:
                    b64_bytes = base64.b64decode(_to_b64(img))
                    mime = _mime_from_path(img) if is_image_path(img) else "image/png"
                    parts.append(genai.types.Part.from_bytes(
                        data=b64_bytes, mime_type=mime
                    ))
            parts.append(prompt)

            cfg = genai.types.GenerateContentConfig(
                system_instruction=system or None,
                response_mime_type="application/json",
                response_schema=field_schema,
                max_output_tokens=self.max_tokens,
            )
            response = client.models.generate_content(
                model=self.llm,
                contents=parts,
                config=cfg,
            )
            return json.loads(response.text)

        except AttributeError:
            # Fallback: google-generativeai legacy path
            genai.configure(api_key=api_key)
            parts_legacy: list[Any] = []
            for img in images:
                if is_url(img):
                    import requests as _req
                    img_bytes = _req.get(img, timeout=30).content
                    parts_legacy.append({"mime_type": "image/jpeg", "data": img_bytes})
                else:
                    b64_bytes = base64.b64decode(_to_b64(img))
                    mime = _mime_from_path(img) if is_image_path(img) else "image/png"
                    parts_legacy.append({"mime_type": mime, "data": b64_bytes})
            parts_legacy.append(prompt)

            model = genai.GenerativeModel(
                model_name=self.llm,
                system_instruction=system or None,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    max_output_tokens=self.max_tokens,
                ),
            )
            response = model.generate_content(parts_legacy)
            return json.loads(response.text)
