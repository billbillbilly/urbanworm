"""Tests for the Aporee catalog-driven sound source.

These exercise the offline filtering logic — no network. The catalog is built
in-memory and passed as a DataFrame.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from urbanworm.dataset import (
    enrich_aporee_catalog,
    fetch_aporee_catalog,
    getSound,
    getSoundAporee,
)


# ----------------------------------------------------------------------
# A small synthetic Aporee catalog for tests.
# Distances are computed from query point (0, 0).
# ----------------------------------------------------------------------
def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # ~111 m north of origin (1/1000 degree of latitude ≈ 111 m)
            {"identifier": "near", "url": "https://aporee.example/near.mp3",
             "latitude": 0.001, "longitude": 0.0,
             "name": "Wind in tall grass", "tags": "wind,nature",
             "created": "2022-06-15T10:30:00", "duration_s": 45.0},
            # ~1.1 km north — outside the 200m radius but within wider radii
            {"identifier": "far", "url": "https://aporee.example/far.mp3",
             "latitude": 0.01, "longitude": 0.0,
             "name": "Distant traffic", "tags": "city,traffic",
             "created": "2020-01-10T22:00:00", "duration_s": 8.0},
            # 0 m — exactly at the query point
            {"identifier": "here", "url": "https://aporee.example/here.mp3",
             "latitude": 0.0, "longitude": 0.0,
             "name": "Birdsong at dawn", "tags": "birds,morning",
             "created": "2023-04-02T05:45:00", "duration_s": 120.0},
        ]
    )


# ----------------------------------------------------------------------
# Spatial filter
# ----------------------------------------------------------------------
def test_filters_by_distance():
    df = getSoundAporee(
        location=(0.0, 0.0), distance=200, catalog=_catalog(), max_return=10,
    )
    # 'here' (0 m) and 'near' (~111 m) both within 200 m; 'far' is not.
    assert set(df["identifier"]) == {"here", "near"}


def test_max_return_caps_results_after_sort_by_distance():
    df = getSoundAporee(
        location=(0.0, 0.0), distance=10_000, catalog=_catalog(), max_return=1,
    )
    assert len(df) == 1
    # The closest one wins
    assert df["identifier"].iloc[0] == "here"


def test_returns_empty_when_no_match():
    df = getSoundAporee(
        location=(0.0, 0.0), distance=1, catalog=_catalog(), max_return=5,
    )
    # 'here' is at 0 m so it does match — narrow further
    df = getSoundAporee(
        location=(50.0, 50.0), distance=10, catalog=_catalog(), max_return=5,
    )
    assert df is None or df.empty


# ----------------------------------------------------------------------
# Output schema parity with Freesound
# ----------------------------------------------------------------------
def test_output_has_freesound_compatible_columns():
    df = getSoundAporee(
        location=(0.0, 0.0), distance=1_000, catalog=_catalog(), max_return=5,
        loc_id=42,
    )
    # Required for the GeoTaggedData / download_to_dir pipeline
    for col in ("id", "url", "preview-hq-mp3", "loc_id", "distance_m"):
        assert col in df.columns, f"missing column: {col}"
    # preview-hq-mp3 aliases url
    assert (df["url"] == df["preview-hq-mp3"]).all()
    # loc_id propagated
    assert (df["loc_id"] == 42).all()


def test_id_falls_back_to_identifier_or_synthesizes():
    cat = _catalog().drop(columns=["identifier"])
    cat["url"] = ["a", "b", "c"]
    cat = pd.DataFrame({
        "url": ["a", "b"],
        "latitude": [0.0, 0.001],
        "longitude": [0.0, 0.0],
    })
    df = getSoundAporee(location=(0.0, 0.0), distance=1_000, catalog=cat, max_return=5)
    # Synthesized ids when no `id` and no `identifier` columns exist
    assert all(str(x).startswith("aporee_") for x in df["id"])


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def test_missing_required_columns_raises():
    bad = pd.DataFrame({"url": ["x"], "latitude": [0.0]})
    with pytest.raises(ValueError) as exc:
        getSoundAporee(location=(0.0, 0.0), catalog=bad)
    assert "longitude" in str(exc.value)


def test_missing_catalog_raises():
    with pytest.raises(ValueError) as exc:
        getSoundAporee(location=(0.0, 0.0))
    assert "catalog" in str(exc.value)


def test_wrong_catalog_type_raises():
    with pytest.raises(TypeError):
        getSoundAporee(location=(0.0, 0.0), catalog=12345)


# ----------------------------------------------------------------------
# Optional filters degrade gracefully
# ----------------------------------------------------------------------
def test_query_filter_matches_name_and_description():
    df = getSoundAporee(
        location=(0.0, 0.0), distance=10_000, catalog=_catalog(), max_return=5,
        query="bird",
    )
    assert set(df["identifier"]) == {"here"}


def test_tag_filter_matches_substring():
    df = getSoundAporee(
        location=(0.0, 0.0), distance=10_000, catalog=_catalog(), max_return=5,
        tag=["traffic"],
    )
    assert set(df["identifier"]) == {"far"}


def test_year_filter():
    df = getSoundAporee(
        location=(0.0, 0.0), distance=10_000, catalog=_catalog(), max_return=5,
        year=[2022, 2023],
    )
    assert set(df["identifier"]) == {"here", "near"}


def test_duration_max_only():
    df = getSoundAporee(
        location=(0.0, 0.0), distance=10_000, catalog=_catalog(), max_return=5,
        duration=10,
    )
    assert set(df["identifier"]) == {"far"}


def test_duration_range():
    df = getSoundAporee(
        location=(0.0, 0.0), distance=10_000, catalog=_catalog(), max_return=5,
        duration=(40, 100),
    )
    assert set(df["identifier"]) == {"near"}


# ----------------------------------------------------------------------
# slice column mirrors Freesound behavior
# ----------------------------------------------------------------------
def test_slice_column_added_when_slice_duration_set():
    df = getSoundAporee(
        location=(0.0, 0.0), distance=10_000, catalog=_catalog(), max_return=1,
        slice_duration=20, slice_max_num=2,
    )
    assert "slice" in df.columns
    # 'here' has duration_s=120 → 20s clips → at most 2 returned
    slices = df["slice"].iloc[0]
    assert isinstance(slices, list) and len(slices) == 2
    # Each slice is [start_ms, end_ms]
    assert all(len(s) == 2 for s in slices)


# ----------------------------------------------------------------------
# Dispatcher: getSound(source='aporee') routes correctly
# ----------------------------------------------------------------------
def test_getSound_dispatcher_routes_to_aporee():
    df = getSound(
        location=(0.0, 0.0), distance=1_000, source="aporee",
        catalog=_catalog(), max_return=2,
    )
    assert set(df["identifier"]) == {"here", "near"}


def test_getSound_dispatcher_unknown_source_raises():
    with pytest.raises(ValueError) as exc:
        getSound(location=(0.0, 0.0), source="spotify")
    assert "spotify" in str(exc.value) or "Unsupported" in str(exc.value)


# ----------------------------------------------------------------------
# Missing duration_s — graceful skip + on-demand probing
# ----------------------------------------------------------------------
def _catalog_no_duration() -> pd.DataFrame:
    """Same shape as _catalog() but without the duration_s column."""
    df = _catalog().drop(columns=["duration_s"])
    return df


def test_slice_without_duration_and_probe_disabled_skips_silently():
    """No `slice` column when probing is disabled and `duration_s` is absent."""
    df = getSoundAporee(
        location=(0.0, 0.0), distance=10_000, catalog=_catalog_no_duration(),
        max_return=5, slice_duration=20, probe_durations=False,
    )
    # Filter still works; just no slicing happens.
    assert df is not None and len(df) > 0
    assert "slice" not in df.columns


def test_slice_without_duration_triggers_probe_when_enabled():
    """When probe_durations=True, missing duration_s triggers a per-row probe."""
    cat = _catalog_no_duration()

    # Stub probe_audio_duration so the test stays offline.
    def fake_probe(url, timeout=60.0):
        return {"https://aporee.example/here.mp3": 60.0,
                "https://aporee.example/near.mp3": 30.0,
                "https://aporee.example/far.mp3": 5.0}.get(url)

    with patch("urbanworm.utils.utils.probe_audio_duration", side_effect=fake_probe):
        df = getSoundAporee(
            location=(0.0, 0.0), distance=10_000, catalog=cat, max_return=5,
            slice_duration=15, probe_durations=True,
        )
    assert "slice" in df.columns
    # The closest one ('here') has 60s duration → 4 clips of 15s.
    here_row = df[df["identifier"] == "here"].iloc[0]
    assert len(here_row["slice"]) == 4
    assert here_row["slice"][0] == [0, 15000]


def test_get_sound_from_location_handles_missing_slice_column():
    """The dataset accumulator must not crash when source returns no `slice`."""
    from urbanworm.dataset import GeoTaggedData
    cat = _catalog_no_duration()
    gtd = GeoTaggedData(locations=[[0.0, 0.0]])
    # slice_duration set, but probe_durations=False → getSoundAporee returns
    # rows without a `slice` column. The accumulator should fall through to
    # the un-sliced branch instead of raising KeyError.
    gtd.get_sound_from_location(
        distance=10_000, source="aporee", catalog=cat, max_return=2,
        slice_duration=20, probe_durations=False, silent=True,
    )
    # We should have collected ids/data without slice info.
    assert len(gtd.audios["id"]) > 0
    # Either no 'slice' key, or it stayed empty
    assert gtd.audios.get("slice", []) == []


# ----------------------------------------------------------------------
# enrich_aporee_catalog
# ----------------------------------------------------------------------
def test_enrich_aporee_catalog_adds_duration_column(tmp_path):
    cat = _catalog_no_duration()
    csv_path = tmp_path / "catalog.csv"
    cat.to_csv(csv_path, index=False)

    def fake_probe(url, timeout=60.0):
        return {"https://aporee.example/here.mp3": 120.0,
                "https://aporee.example/near.mp3": 45.0,
                "https://aporee.example/far.mp3": 8.0}.get(url)

    with patch("urbanworm.utils.utils.probe_audio_duration", side_effect=fake_probe):
        out = enrich_aporee_catalog(str(csv_path))
    assert "duration_s" in out.columns
    assert set(out["duration_s"].dropna().tolist()) == {120.0, 45.0, 8.0}


def test_enrich_aporee_catalog_writes_to_out_path(tmp_path):
    cat = _catalog_no_duration()

    def fake_probe(url, timeout=60.0):
        return 30.0

    out_csv = tmp_path / "enriched.csv"
    with patch("urbanworm.utils.utils.probe_audio_duration", side_effect=fake_probe):
        enrich_aporee_catalog(cat, out_path=str(out_csv))
    written = pd.read_csv(out_csv)
    assert "duration_s" in written.columns
    assert (written["duration_s"] == 30.0).all()


def test_enrich_aporee_catalog_drops_short_when_min_duration_set():
    cat = _catalog_no_duration()

    def fake_probe(url, timeout=60.0):
        return {"https://aporee.example/here.mp3": 60.0,
                "https://aporee.example/near.mp3": 5.0,
                "https://aporee.example/far.mp3": 3.0}.get(url)

    with patch("urbanworm.utils.utils.probe_audio_duration", side_effect=fake_probe):
        out = enrich_aporee_catalog(cat, min_duration=10.0)
    # Only the 60s recording survives the 10s floor.
    assert len(out) == 1
    assert out["identifier"].iloc[0] == "here"


def test_enrich_aporee_catalog_skip_existing_default():
    """skip_existing=True (default) leaves already-set duration_s alone."""
    cat = _catalog()  # has duration_s populated
    probe_called = []

    def fake_probe(url, timeout=60.0):
        probe_called.append(url)
        return 999.0

    with patch("urbanworm.utils.utils.probe_audio_duration", side_effect=fake_probe):
        out = enrich_aporee_catalog(cat)
    # Original durations preserved, no probe calls
    assert probe_called == []
    assert out["duration_s"].iloc[0] == 45.0  # 'near'


def test_enrich_aporee_catalog_force_reprobe():
    cat = _catalog()  # has duration_s populated
    with patch("urbanworm.utils.utils.probe_audio_duration", return_value=999.0):
        out = enrich_aporee_catalog(cat, skip_existing=False)
    assert (out["duration_s"] == 999.0).all()


def test_enrich_aporee_catalog_requires_url_column():
    bad = pd.DataFrame({"latitude": [0], "longitude": [0]})
    with pytest.raises(ValueError) as exc:
        enrich_aporee_catalog(bad)
    assert "url" in str(exc.value)


# ----------------------------------------------------------------------
# fetch_aporee_catalog
# ----------------------------------------------------------------------
def _ia_response(items, cursor=None):
    """Build a fake IA Scrape API response."""
    body = {"items": items}
    if cursor is not None:
        body["cursor"] = cursor
    return body


def _make_requests_get_mock(pages):
    """Return a callable suitable for ``patch('requests.get', side_effect=...)``.

    `pages` is a list of dicts; each call returns the next one wrapped in a
    fake Response object.
    """
    pages_iter = iter(pages)

    class _FakeResp:
        def __init__(self, payload):
            self._p = payload
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._p

    def _get(url, params=None, headers=None, timeout=None):
        return _FakeResp(next(pages_iter))

    return _get


def test_fetch_aporee_catalog_single_page():
    items = [
        {"identifier": "berlin_001", "title": "Spree at dawn", "date": "2021-06-15T05:30:00Z",
         "description": "river ambience", "subject": "river,water",
         "latitude": "52.52", "longitude": "13.40", "licenseurl": "cc-by"},
        {"identifier": "berlin_002", "title": "U-Bahn", "date": "2020-11-04T22:10:00Z",
         "description": "underground", "subject": ["transport", "city"],
         "latitude": "52.50", "longitude": "13.42", "licenseurl": "cc-by"},
    ]
    pages = [_ia_response(items)]  # only one page; no cursor → loop ends

    with patch("requests.get", side_effect=_make_requests_get_mock(pages)):
        df = fetch_aporee_catalog(rows=10)

    assert len(df) == 2
    # Schema sanity — both script-style and getSoundAporee-style names present
    for col in ("identifier", "id", "latitude", "longitude", "url",
                "capture_time", "created", "title", "name", "description",
                "tags", "licence", "duration_s", "year", "month", "hour",
                "season"):
        assert col in df.columns, f"missing column: {col}"

    # Subject list collapsed to comma string
    assert df.loc[df["identifier"] == "berlin_002", "tags"].iloc[0] == "transport,city"
    # url falls back to <id>.mp3 because verify_urls defaults to False
    assert df.loc[0, "url"].endswith("/berlin_001.mp3")
    # season derived from month + lat (Berlin north → June=summer)
    assert df.loc[df["identifier"] == "berlin_001", "season"].iloc[0] == "summer"


def test_fetch_aporee_catalog_paginates_with_cursor():
    page1 = _ia_response(
        [{"identifier": "a", "title": "", "date": "", "description": "",
          "subject": "", "latitude": "0", "longitude": "0", "licenseurl": ""}],
        cursor="CURSOR1",
    )
    page2 = _ia_response(
        [{"identifier": "b", "title": "", "date": "", "description": "",
          "subject": "", "latitude": "1", "longitude": "1", "licenseurl": ""}],
        # no cursor → loop ends
    )
    # Note: page_size=100 (min) and rows=2 will request page_n=2 each call.
    # The mock just returns whatever is next regardless of params.
    with patch("requests.get", side_effect=_make_requests_get_mock([page1, page2])):
        df = fetch_aporee_catalog(rows=2, page_size=100)
    assert list(df["identifier"]) == ["a", "b"]


def test_fetch_aporee_catalog_skips_rows_without_geo():
    items = [
        {"identifier": "good", "title": "", "date": "", "description": "",
         "subject": "", "latitude": "10", "longitude": "20", "licenseurl": ""},
        {"identifier": "no_geo", "title": "", "date": "", "description": "",
         "subject": "", "latitude": "", "longitude": "", "licenseurl": ""},
    ]
    with patch("requests.get", side_effect=_make_requests_get_mock([_ia_response(items)])):
        df = fetch_aporee_catalog(rows=10)
    assert list(df["identifier"]) == ["good"]


def test_fetch_aporee_catalog_hour_filter():
    items = [
        {"identifier": "morn", "title": "", "date": "2021-06-15T08:00:00Z",
         "description": "", "subject": "",
         "latitude": "0", "longitude": "0", "licenseurl": ""},
        {"identifier": "night", "title": "", "date": "2021-06-15T23:00:00Z",
         "description": "", "subject": "",
         "latitude": "0", "longitude": "0", "licenseurl": ""},
    ]
    with patch("requests.get", side_effect=_make_requests_get_mock([_ia_response(items)])):
        df = fetch_aporee_catalog(rows=10, hour=(6, 12))
    assert list(df["identifier"]) == ["morn"]


def test_fetch_aporee_catalog_season_filter_southern_hemisphere():
    items = [
        # December + southern lat → "summer" in southern hemisphere
        {"identifier": "buenos_aires", "title": "", "date": "2021-12-15T12:00:00Z",
         "description": "", "subject": "",
         "latitude": "-34.6", "longitude": "-58.4", "licenseurl": ""},
        # December + northern lat → "winter"
        {"identifier": "berlin", "title": "", "date": "2021-12-15T12:00:00Z",
         "description": "", "subject": "",
         "latitude": "52.5", "longitude": "13.4", "licenseurl": ""},
    ]
    with patch("requests.get", side_effect=_make_requests_get_mock([_ia_response(items)])):
        df = fetch_aporee_catalog(rows=10, season="summer")
    assert list(df["identifier"]) == ["buenos_aires"]


def test_fetch_aporee_catalog_writes_to_out_path(tmp_path):
    items = [
        {"identifier": "x", "title": "", "date": "2021-06-15T08:00:00Z",
         "description": "", "subject": "",
         "latitude": "0", "longitude": "0", "licenseurl": ""},
    ]
    out = tmp_path / "cat.csv"
    with patch("requests.get", side_effect=_make_requests_get_mock([_ia_response(items)])):
        fetch_aporee_catalog(rows=10, out_path=str(out))
    written = pd.read_csv(out)
    assert "identifier" in written.columns
    assert len(written) == 1


def test_fetched_catalog_works_with_getSoundAporee_directly():
    """End-to-end: fetched catalog → getSoundAporee should filter cleanly."""
    items = [
        {"identifier": "near", "title": "Wind", "date": "2021-06-15T08:00:00Z",
         "description": "", "subject": "wind",
         "latitude": "0.001", "longitude": "0.0", "licenseurl": ""},
        {"identifier": "far", "title": "Traffic", "date": "2020-11-01T22:00:00Z",
         "description": "", "subject": "city,traffic",
         "latitude": "0.05", "longitude": "0.0", "licenseurl": ""},
    ]
    with patch("requests.get", side_effect=_make_requests_get_mock([_ia_response(items)])):
        cat = fetch_aporee_catalog(rows=10)

    df = getSoundAporee(
        location=(0.0, 0.0), distance=200, catalog=cat, max_return=10,
    )
    # only 'near' is within 200 m
    assert list(df["identifier"]) == ["near"]


# ----------------------------------------------------------------------
# Column alias acceptance — script-style lat/lon/capture_time
# ----------------------------------------------------------------------
def test_getSoundAporee_accepts_lat_lon_capture_time_aliases():
    cat = pd.DataFrame([
        {"identifier": "a", "url": "https://aporee.example/a.mp3",
         "lat": 0.0, "lon": 0.0,
         "capture_time": "2021-06-15T08:00:00Z"},
        {"identifier": "b", "url": "https://aporee.example/b.mp3",
         "lat": 1.0, "lon": 0.0,
         "capture_time": "2020-11-01T22:00:00Z"},
    ])
    df = getSoundAporee(
        location=(0.0, 0.0), distance=10_000, catalog=cat, max_return=10,
    )
    # Should have renamed lat→latitude, lon→longitude internally
    assert "latitude" in df.columns and "longitude" in df.columns
    # And capture_time → created so year filter still works
    df2 = getSoundAporee(
        location=(0.0, 0.0), distance=10_000_000, catalog=cat, max_return=10,
        year=[2021],
    )
    assert list(df2["identifier"]) == ["a"]
