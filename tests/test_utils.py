"""Pure-logic unit tests for urbanworm.utils.utils — no network or models."""
from __future__ import annotations

import math

import pytest

from urbanworm.utils import utils as U


# ---- json repair ---------------------------------------------------------

def test_sanitize_strips_code_fence_and_nbsp():
    text = "```json\n{\xa0\"a\": 1}\n```"
    assert U.sanitize_json_text(text) == '{ "a": 1}'


def test_sanitize_handles_none():
    assert U.sanitize_json_text(None) == ""


def test_extract_json_from_text_finds_balanced():
    text = "blah blah {\"a\": {\"b\": 2}} trailing"
    assert U.extract_json_from_text(text) == '{"a": {"b": 2}}'


def test_extract_json_from_text_returns_none_for_unbalanced():
    assert U.extract_json_from_text("no braces here") is None
    assert U.extract_json_from_text("{not closed") is None


def test_extract_last_json_picks_trailing_object():
    text = "preamble\n{\n\"a\": 1\n}"
    assert U.extract_last_json(text) == {"a": 1}


# ---- season / time-of-day helpers ----------------------------------------

@pytest.mark.parametrize(
    "season, expected",
    [
        ("spring", {3, 4, 5}),
        ("summer", {6, 7, 8}),
        ("fall", {9, 10, 11}),
        ("autumn", {9, 10, 11}),
        ("winter", {12, 1, 2}),
        (None, None),
    ],
)
def test_season_months(season, expected):
    assert U.season_months(season) == expected


def test_season_months_invalid():
    with pytest.raises(ValueError):
        U.season_months("octember")


@pytest.mark.parametrize(
    "tod, sample_hour, expected",
    [
        ("morning", 6, True),
        ("morning", 13, False),
        ("afternoon", 13, True),
        ("evening", 19, True),
        ("night", 23, True),
        ("night", 3, True),
        ("night", 12, False),
    ],
)
def test_tod_hours(tod, sample_hour, expected):
    hours = U.tod_hours(tod)
    assert (sample_hour in hours) is expected


def test_tod_hours_none():
    assert U.tod_hours(None) is None


def test_tod_hours_invalid():
    with pytest.raises(ValueError):
        U.tod_hours("brunch")


# ---- year_range ----------------------------------------------------------

def test_year_range_single_value():
    start, end = U.year_range([2021])
    assert start.startswith("2021-01-01") and end.startswith("2021-12-31")


def test_year_range_two_values_swapped():
    start, end = U.year_range((2024, 2020))
    assert start.startswith("2020") and end.startswith("2024")


def test_year_range_invalid():
    with pytest.raises(ValueError):
        U.year_range([])


# ---- haversine -----------------------------------------------------------

def test_haversine_zero_distance():
    assert U.haversine_m(40.0, -83.0, 40.0, -83.0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_pair():
    # ~111 km between (0,0) and (1,0)
    d = U.haversine_m(0.0, 0.0, 1.0, 0.0)
    assert 110_000 < d < 112_000


# ---- bbox containment ----------------------------------------------------

def test_is_coordinate_in_bbox_inside_and_edge():
    bbox = (-1, -1, 1, 1)
    assert U.is_coordinate_in_bbox(0, 0, bbox)
    assert U.is_coordinate_in_bbox(-1, -1, bbox)  # inclusive
    assert U.is_coordinate_in_bbox(1, 1, bbox)
    assert not U.is_coordinate_in_bbox(2, 0, bbox)


# ---- query_string --------------------------------------------------------

def test_query_string_handles_list_and_str():
    assert U.query_string(["bird", "song"]) == "bird song"
    assert U.query_string("traffic ") == "traffic"
    assert U.query_string(None) == ""


# ---- sliced_duration -----------------------------------------------------

def test_sliced_duration_full_clips():
    # 60s split into 20s clips -> 3 clips of [0,20000],[20000,40000],[40000,60000]
    out = U.sliced_duration(60, 20)
    assert out == [[0, 20000], [20000, 40000], [40000, 60000]]


def test_sliced_duration_max_number_caps():
    out = U.sliced_duration(60, 20, number=2)
    assert len(out) == 2
    assert out[0] == [0, 20000]


def test_sliced_duration_shorter_than_clip_returns_full_signal():
    # duration < clip_duration -> single clip covering whole signal (in ms)
    out = U.sliced_duration(5, 20)
    assert out == [[0, 5000]]


# ---- utm projection ------------------------------------------------------

def test_lonlat_to_utm_epsg_northern_zones():
    # Detroit-ish (~ -83, 42) -> UTM zone 17N -> 32617
    assert U.lonlat_to_utm_epsg(-83.0, 42.0) == 32617


def test_lonlat_to_utm_epsg_southern_zones():
    # Buenos Aires-ish (~ -58, -34) -> UTM zone 21S -> 32721
    assert U.lonlat_to_utm_epsg(-58.0, -34.0) == 32721


def test_calculate_bearing_north():
    # Same longitude, target north -> bearing 0
    b = U.calculate_bearing(40.0, -83.0, 41.0, -83.0)
    assert math.isclose(b, 0.0, abs_tol=0.5)


def test_calculate_bearing_east():
    # Same latitude, target east -> bearing ~90
    b = U.calculate_bearing(40.0, -83.0, 40.0, -82.0)
    assert 80 < b < 100
