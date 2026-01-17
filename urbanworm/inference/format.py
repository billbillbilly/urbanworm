from __future__ import annotations
import copy
import json
from typing import Any, Dict, Generic, List, Tuple, Type, TypeVar, Union, cast
from pydantic import BaseModel, ConfigDict, create_model

T = TypeVar("T", bound=BaseModel)
FieldSpec = Tuple[Any, Any]  # (type, default), e.g. (bool, ...)


class _ForbidExtraBase(BaseModel):
    """Base model for dynamically-created models."""
    model_config = ConfigDict(extra="forbid")


def schema(fields: Dict[str, FieldSpec], model_name: str = "QnA") -> Type[BaseModel]:
    """
    Create a customized Pydantic model with customized fields.

    Args:
        fields: dict of field definitions for create_model.
            Example:
                fields = {
                    "question": (str, ...),
                    "answer": (bool, ...),
                    "explanation": (str, None),
                }
        model_name: name for the generated model class.

    Returns:
        A Pydantic model class.
    """
    if not isinstance(fields, dict) or not fields:
        raise ValueError("`fields` must be a non-empty dict of {name: (type, default)}.")

    return create_model(model_name, __base__=_ForbidExtraBase, **fields)


class Response(BaseModel, Generic[T]):
    """Wrapper schema: {"responses": [ ... ]}"""
    model_config = ConfigDict(extra="forbid")
    responses: List[T]


def create_format(
    fields: Dict[str, FieldSpec],
    *,
    item_model_name: str = "QnA",
    wrapper_model_name: str | None = None,
) -> Type[BaseModel]:
    """
    Create a typed `Response[CustomQnA]` model using a dynamically defined schema.

    Args:
        fields: field definitions for the inner model.
        item_model_name: name of the inner model.
        wrapper_model_name: optional pretty name for the specialized wrapper class.

    Returns:
        A concrete Pydantic model class: Response[CustomQnA].
    """
    CustomQnA = schema(fields, model_name=item_model_name)
    Model = cast(Type[BaseModel], Response[CustomQnA])  # concrete generic specialization

    # Give the specialized model a stable readable name (optional)
    if wrapper_model_name is None:
        wrapper_model_name = f"Response_{item_model_name}"
    try:
        Model.__name__ = wrapper_model_name  # type: ignore[attr-defined]
    except Exception:
        pass

    return Model


def _inline_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inline #/$defs/* references to avoid $ref in the final schema.

    This is useful because some json-schema consumers (incl. some llama.cpp builds)
    don’t fully support $ref/$defs.

    Note: This handles the common Pydantic pattern where $ref nodes are dicts like:
        {"$ref": "#/$defs/QnA"}
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, stack: set[str]) -> Any:
        if isinstance(node, dict):
            # If it's a pure $ref node, inline it
            if set(node.keys()) == {"$ref"} and isinstance(node["$ref"], str):
                ref: str = node["$ref"]
                if ref.startswith("#/$defs/"):
                    name = ref.split("/")[-1]
                    if name in stack:
                        # cycle protection (rare for your use case)
                        return {}
                    target = defs.get(name)
                    if target is None:
                        return node
                    stack.add(name)
                    out = resolve(copy.deepcopy(target), stack)
                    stack.remove(name)
                    return out

            return {k: resolve(v, stack) for k, v in node.items() if k != "$defs"}

        if isinstance(node, list):
            return [resolve(x, stack) for x in node]

        return node

    out = resolve(copy.deepcopy(schema), set())
    if isinstance(out, dict):
        out.pop("$defs", None)
    return cast(Dict[str, Any], out)


def schema_dict(model: Type[BaseModel], *, inline_refs: bool = True) -> Dict[str, Any]:
    """
    Build JSON schema dict from a Pydantic model.

    Args:
        model: Pydantic model class.
        inline_refs: if True, inline $ref/$defs.

    Returns:
        JSON schema as a dict.
    """
    s = model.model_json_schema()
    return _inline_refs(s) if inline_refs else s


def schema_json(
    model: Type[BaseModel],
    *,
    inline_refs: bool = True,
    compact: bool = True,
) -> str:
    """
    JSON schema string (optionally inlined, optionally compact).

    Use this to write to a file and pass to llama.cpp with -jf/--json-schema-file.
    """
    s = schema_dict(model, inline_refs=inline_refs)
    if compact:
        return json.dumps(s, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(s, ensure_ascii=False, indent=2)