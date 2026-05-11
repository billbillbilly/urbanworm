"""Smoke + behavior tests for GeoTaggedData (no network)."""
from __future__ import annotations

import pandas as pd
import pytest

from urbanworm.dataset import GeoTaggedData


def test_payload_dicts_are_independent():
    """Regression: __init__ used chained assignment and aliased the three dicts."""
    g = GeoTaggedData()
    g.svis["data"].append("svi-img")
    g.photos["data"].append("flickr-img")
    assert g.svis["data"] == ["svi-img"]
    assert g.photos["data"] == ["flickr-img"]
    assert g.audios["data"] == []  # untouched


def test_construct_units_from_nested_list():
    g = GeoTaggedData(locations=[[-83.235572, 42.348092], [-83.235154, 42.348806]])
    assert g.units is not None
    assert len(g.units) == 2
    assert "loc_id" in g.units.columns


def test_construct_units_from_dataframe():
    df = pd.DataFrame({"longitude": [-83.235572], "latitude": [42.348092]})
    g = GeoTaggedData(locations=df)
    assert g.units is not None
    assert len(g.units) == 1


def test_construct_units_invalid_list_raises():
    with pytest.raises(ValueError):
        GeoTaggedData(locations=[1, 2])  # not a nested list


def test_construct_units_invalid_dict_raises():
    with pytest.raises(ValueError):
        GeoTaggedData(locations={"x": [1], "y": [2]})


def test_construct_units_unsupported_type_raises():
    with pytest.raises(TypeError):
        GeoTaggedData(locations=42)
