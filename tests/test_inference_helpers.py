"""Tests for small helpers in urbanworm.inference."""
from __future__ import annotations

from urbanworm.inference.Inference import _pack


def test_pack_groups_consecutive_equal_locations():
    locations = [1, 1, 2, 2, 2, 3]
    data = ["a", "b", "c", "d", "e", "f"]
    assert _pack(locations, data) == [["a", "b"], ["c", "d", "e"], ["f"]]


def test_pack_includes_trailing_group():
    locations = [1, 1]
    data = ["a", "b"]
    # The previous implementation dropped the only/final group.
    assert _pack(locations, data) == [["a", "b"]]


def test_pack_handles_singletons():
    locations = [1, 2, 3]
    data = ["a", "b", "c"]
    assert _pack(locations, data) == [["a"], ["b"], ["c"]]


def test_pack_handles_empty():
    assert _pack([], []) == []
