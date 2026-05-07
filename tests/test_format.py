"""Tests for the dynamic Pydantic schema builder."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from urbanworm.inference.format import (
    create_format,
    schema as build_item_schema,
    schema_dict,
    schema_json,
)


def test_create_format_validates_correct_payload():
    Wrapper = create_format({"answer": (bool, ...), "explanation": (str, ...)})
    out = Wrapper.model_validate_json(
        '{"responses": [{"answer": true, "explanation": "yes"}]}'
    )
    assert out.responses[0].answer is True
    assert out.responses[0].explanation == "yes"


def test_create_format_rejects_extra_fields():
    Wrapper = create_format({"answer": (bool, ...)})
    with pytest.raises(ValidationError):
        Wrapper.model_validate_json(
            '{"responses": [{"answer": true, "extra": "no"}]}'
        )


def test_create_format_empty_fields_raises():
    with pytest.raises(ValueError):
        build_item_schema({})


def test_schema_json_inlines_refs_by_default():
    Wrapper = create_format({"answer": (bool, ...)})
    js = schema_json(Wrapper)
    parsed = json.loads(js)
    # No $defs / $ref should remain after inlining
    assert "$defs" not in parsed
    text = json.dumps(parsed)
    assert "$ref" not in text


def test_schema_dict_inlines_nested_refs():
    Wrapper = create_format({"a": (int, ...)})
    s = schema_dict(Wrapper, inline_refs=True)
    # responses array items should be the inlined object schema
    item_schema = s["properties"]["responses"]["items"]
    assert item_schema.get("type") == "object"
    assert "a" in item_schema["properties"]
