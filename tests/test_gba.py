"""Tests for the GlobalBuildingAtlas (gba) source.

Uses an in-memory synthetic catalog written to GPKG so the loader can be
exercised end-to-end without a real GBA file.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from urbanworm.dataset import GeoTaggedData
from urbanworm.utils.building import getGBABuildings


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_gba_geodf() -> gpd.GeoDataFrame:
    """Three test buildings in/around (0, 40)."""
    return gpd.GeoDataFrame(
        [
            {"id": 1, "height": 12.0, "geometry":
                Polygon([(0.000, 40.000), (0.0001, 40.000),
                         (0.0001, 40.0001), (0.000, 40.0001)])},
            {"id": 2, "height": 30.0, "geometry":
                Polygon([(0.0005, 40.0005), (0.0006, 40.0005),
                         (0.0006, 40.0006), (0.0005, 40.0006)])},
            # outside the bbox we'll query
            {"id": 3, "height": 8.0, "geometry":
                Polygon([(1.0, 40.0), (1.0001, 40.0),
                         (1.0001, 40.0001), (1.0, 40.0001)])},
        ],
        crs="EPSG:4326",
    )


def _write_gpkg(tmp_path) -> str:
    p = tmp_path / "synthetic.gpkg"
    _make_gba_geodf().to_file(p, driver="GPKG")
    return str(p)


# ----------------------------------------------------------------------
# getGBABuildings — direct
# ----------------------------------------------------------------------
def test_getGBABuildings_filters_by_bbox(tmp_path):
    p = _write_gpkg(tmp_path)
    out = getGBABuildings(bbox=(-0.01, 39.99, 0.01, 40.01), gba_path=p)
    assert out is not None and len(out) == 2  # third building outside


def test_getGBABuildings_normalizes_height_column(tmp_path):
    """The source uses 'height'; loader should rename to 'height_m'."""
    p = _write_gpkg(tmp_path)
    out = getGBABuildings(bbox=(-0.01, 39.99, 0.01, 40.01), gba_path=p)
    assert "height_m" in out.columns
    assert "height" not in out.columns
    assert set(out["height_m"]) == {12.0, 30.0}


def test_getGBABuildings_preserves_height_dtype(tmp_path):
    p = _write_gpkg(tmp_path)
    out = getGBABuildings(bbox=(-0.01, 39.99, 0.01, 40.01), gba_path=p)
    assert pd.api.types.is_numeric_dtype(out["height_m"])


def test_getGBABuildings_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        getGBABuildings(bbox=(0, 0, 1, 1), gba_path=str(tmp_path / "nope.gpkg"))


def test_getGBABuildings_bbox_with_no_match_returns_none(tmp_path):
    p = _write_gpkg(tmp_path)
    out = getGBABuildings(bbox=(50.0, 50.0, 51.0, 51.0), gba_path=p)
    assert out is None


def test_getGBABuildings_area_filter(tmp_path):
    """Both test buildings are tiny (~100 m² ish); a high min_area drops them."""
    p = _write_gpkg(tmp_path)
    out = getGBABuildings(
        bbox=(-0.01, 39.99, 0.01, 40.01),
        gba_path=p,
        min_area=1_000_000,  # 1 km² floor
    )
    assert out is None


# ----------------------------------------------------------------------
# GeoTaggedData.getBuildings(source='gba')
# ----------------------------------------------------------------------
def test_getBuildings_globfp3d_path_is_optional(monkeypatch):
    """source='globfp3d' without gba_path → auto-fetch from Zenodo."""
    called = {"n": 0}

    def fake_fetch(bbox, cache_dir=None, timeout=120.0):
        called["n"] += 1
        return None

    monkeypatch.setattr(
        "urbanworm.utils.building.fetch_globfp3d_for_bbox", fake_fetch,
    )
    gtd = GeoTaggedData()
    gtd.getBuildings(bbox=(0, 0, 1, 1), source="globfp3d")
    assert called["n"] == 1


def test_getBuildings_gba_path_is_optional_now(monkeypatch):
    """source='gba' (true GBA) without gba_path → auto-fetch from HuggingFace."""
    called = {"n": 0}

    def fake_fetch(bbox, cache_dir=None, timeout=120.0, include_heights=False):
        called["n"] += 1
        return None

    monkeypatch.setattr(
        "urbanworm.utils.building.fetch_true_gba_for_bbox", fake_fetch,
    )
    gtd = GeoTaggedData()
    gtd.getBuildings(bbox=(0, 0, 1, 1), source="gba")
    assert called["n"] == 1


def test_getBuildings_unsupported_source_lists_all_four():
    gtd = GeoTaggedData()
    with pytest.raises(ValueError) as exc:
        gtd.getBuildings(bbox=(0, 0, 1, 1), source="terraserver")
    msg = str(exc.value)
    # All four valid sources must appear in the error message
    for src in ("osm", "microsoft", "globfp3d", "gba"):
        assert src in msg


def test_getBuildings_globfp3d_populates_units_with_height(tmp_path):
    """source='globfp3d' with a local file populates units + height_m."""
    p = _write_gpkg(tmp_path)
    gtd = GeoTaggedData()
    gtd.getBuildings(
        bbox=(-0.01, 39.99, 0.01, 40.01),
        source="globfp3d",
        gba_path=p,
    )
    assert gtd.units is not None and len(gtd.units) == 2
    assert "height_m" in gtd.units.columns
    assert set(gtd.units["height_m"]) == {12.0, 30.0}


def test_getBuildings_gba_populates_units_with_height(tmp_path):
    """source='gba' with a local file populates units + height_m."""
    p = _write_gpkg(tmp_path)
    gtd = GeoTaggedData()
    gtd.getBuildings(
        bbox=(-0.01, 39.99, 0.01, 40.01),
        source="gba",
        gba_path=p,
    )
    assert gtd.units is not None and len(gtd.units) == 2
    assert "height_m" in gtd.units.columns
    assert set(gtd.units["height_m"]) == {12.0, 30.0}


# ----------------------------------------------------------------------
# get_svi_from_locations passes per-row height when available
# ----------------------------------------------------------------------
def test_get_svi_from_locations_uses_per_row_height(tmp_path, monkeypatch):
    """When self.units has height_m, getSV should be called with that
    row's value as building_height — not the global default."""
    p = _write_gpkg(tmp_path)
    gtd = GeoTaggedData()
    gtd.getBuildings(
        bbox=(-0.01, 39.99, 0.01, 40.01),
        source="gba",
        gba_path=p,
    )

    captured_heights: list[float] = []

    def fake_getSV(*args, **kwargs):
        captured_heights.append(kwargs.get("building_height"))
        return None, None  # treated as "no SVI found"

    monkeypatch.setattr("urbanworm.dataset.getSV", fake_getSV)
    gtd.get_svi_from_locations(
        key="dummy-token",
        fov="auto",
        building_height=9.0,  # global default (should be overridden)
        silent=True,
    )

    # Two rows, two getSV calls; heights from the catalog were used.
    assert sorted(captured_heights) == [12.0, 30.0]


def test_get_svi_from_locations_falls_back_to_default_when_no_height_column(
        tmp_path, monkeypatch):
    """Without a height_m column, the global default is used."""
    # Use OSM-style polygons (no height column at all)
    gdf = gpd.GeoDataFrame(
        [
            {"geometry": Polygon([(0.0, 40.0), (0.0001, 40.0),
                                  (0.0001, 40.0001), (0.0, 40.0001)])},
        ],
        crs="EPSG:4326",
    )
    gtd = GeoTaggedData()
    gtd.units = gdf

    captured: list[float] = []

    def fake_getSV(*args, **kwargs):
        captured.append(kwargs.get("building_height"))
        return None, None

    monkeypatch.setattr("urbanworm.dataset.getSV", fake_getSV)
    gtd.get_svi_from_locations(
        key="dummy", fov="auto", building_height=15.0, silent=True,
    )
    assert captured == [15.0]


def test_get_svi_from_locations_falls_back_when_row_height_is_nan(monkeypatch):
    """Mixed catalog: rows with NaN height_m should use the global default."""
    gdf = gpd.GeoDataFrame(
        [
            {"height_m": 25.0, "geometry":
                Polygon([(0.0, 40.0), (0.0001, 40.0),
                         (0.0001, 40.0001), (0.0, 40.0001)])},
            {"height_m": pd.NA, "geometry":
                Polygon([(0.001, 40.0), (0.0011, 40.0),
                         (0.0011, 40.0001), (0.001, 40.0001)])},
        ],
        crs="EPSG:4326",
    )
    gtd = GeoTaggedData()
    gtd.units = gdf

    captured: list[float] = []

    def fake_getSV(*args, **kwargs):
        captured.append(kwargs.get("building_height"))
        return None, None

    monkeypatch.setattr("urbanworm.dataset.getSV", fake_getSV)
    gtd.get_svi_from_locations(
        key="dummy", fov="auto", building_height=9.0, silent=True,
    )
    # First row uses 25.0, second falls back to 9.0
    assert captured == [25.0, 9.0]


# ----------------------------------------------------------------------
# parse_gba_data_links — pure-text parser, fully offline
# ----------------------------------------------------------------------
def test_parse_gba_data_links_simple_range():
    from urbanworm.utils.building import parse_gba_data_links
    text = (
        "1. 3D-GloBFP: ... (PART I, grid ID: 0-2)\n"
        "https://figshare.com/articles/dataset/3D-GloBFP_part1/100\n"
        "\n"
        "2. 3D-GloBFP: ... (PART II, grid ID: 3-3)\n"
        "https://figshare.com/articles/dataset/3D-GloBFP_part2/200\n"
    )
    out = parse_gba_data_links(text)
    assert set(out.keys()) == {0, 1, 2, 3}
    assert out[0].endswith("/100")
    assert out[3].endswith("/200")


def test_parse_gba_data_links_handles_swapped_range():
    from urbanworm.utils.building import parse_gba_data_links
    text = (
        "1. (grid ID: 5-3)\n"
        "https://example.org/x/100\n"
    )
    out = parse_gba_data_links(text)
    # Should have expanded 3..5
    assert set(out.keys()) == {3, 4, 5}


def test_parse_gba_data_links_skips_non_grid_lines():
    from urbanworm.utils.building import parse_gba_data_links
    text = (
        "preamble line that has no grid mention\n"
        "1. (grid ID: 10-11)\n"
        "https://example.org/x/777\n"
        "trailing prose\n"
    )
    out = parse_gba_data_links(text)
    assert set(out.keys()) == {10, 11}


# ----------------------------------------------------------------------
# figshare_article_id
# ----------------------------------------------------------------------
def test_figshare_article_id_extracted_from_collection_url():
    from urbanworm.utils.building import figshare_article_id
    url = "https://figshare.com/articles/dataset/3D-GloBFP_PART_grid_ID_0-400_/28879733"
    assert figshare_article_id(url) == "28879733"


def test_figshare_article_id_with_trailing_slash():
    from urbanworm.utils.building import figshare_article_id
    assert figshare_article_id("https://figshare.com/articles/dataset/foo/123/") == "123"


def test_figshare_article_id_invalid():
    from urbanworm.utils.building import figshare_article_id
    with pytest.raises(ValueError):
        figshare_article_id("not even a url")


# ----------------------------------------------------------------------
# _match_grid_id heuristics
# ----------------------------------------------------------------------
def test_match_grid_id_word_boundary():
    from urbanworm.utils.building import _match_grid_id
    assert _match_grid_id("0.zip", 0) is True
    assert _match_grid_id("tile_42.zip", 42) is True
    assert _match_grid_id("123.shp", 123) is True
    # A leading 1 in 100 should NOT match grid_id=0
    assert _match_grid_id("100.zip", 0) is False
    # Wrong extension
    assert _match_grid_id("0.txt", 0) is False
    # Unrelated number
    assert _match_grid_id("99.zip", 100) is False


# ----------------------------------------------------------------------
# fetch_gba_for_bbox — end-to-end with mocked HTTP + tile shp
# ----------------------------------------------------------------------
def test_fetch_gba_for_bbox_intersects_grid_and_loads_tile(tmp_path, monkeypatch):
    from urbanworm.utils import building as B

    # 1) Build a tiny "world grid" shapefile with a tile covering (0,40)
    grid_dir = tmp_path / "world_grid"
    grid_dir.mkdir()
    grid = gpd.GeoDataFrame(
        [
            {"fid": 7, "geometry":
                Polygon([(-1, 39), (1, 39), (1, 41), (-1, 41)])},
            {"fid": 99, "geometry":
                Polygon([(50, 0), (51, 0), (51, 1), (50, 1)])},  # far away
        ],
        crs="EPSG:4326",
    )
    grid.to_file(grid_dir / "world_grid.shp")

    # 2) Write a fake data_links.txt to cache_dir
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "data_links.txt").write_text(
        "1. (grid ID: 0-10)\n"
        "https://figshare.com/articles/dataset/foo/12345\n"
    )
    # Stash the unzipped grid at cache/world_grid; load_gba_grid_manifest
    # picks the .shp directly.
    (cache / "world_grid").mkdir()
    for ext in ("shp", "shx", "dbf", "prj", "cpg"):
        src = grid_dir / f"world_grid.{ext}"
        if src.exists():
            (cache / "world_grid" / f"world_grid.{ext}").write_bytes(src.read_bytes())
    # Mark the zip as present so the loader doesn't try to download it.
    (cache / "world_grid.zip").write_bytes(b"\x00")

    # 3) Build the per-tile .shp the loader will read after "downloading"
    tile_dir = cache / "tiles" / "7"
    tile_dir.mkdir(parents=True)
    tile_gdf = gpd.GeoDataFrame(
        [
            {"id": 1, "Height": 12.5, "geometry":
                Polygon([(0.0, 40.0), (0.001, 40.0),
                         (0.001, 40.001), (0.0, 40.001)])},
            {"id": 2, "Height": 33.0, "geometry":
                Polygon([(0.002, 40.0), (0.0021, 40.0),
                         (0.0021, 40.0001), (0.002, 40.0001)])},
        ],
        crs="EPSG:4326",
    )
    tile_gdf.to_file(tile_dir / "tile_7.shp")

    # 4) Patch the figshare API + tile-download functions so no real HTTP fires
    monkeypatch.setattr(
        B, "_figshare_list_files",
        lambda article_id, timeout=60.0: [],   # not used because tile is already cached
    )

    # 5) Run the orchestrator
    out = B.fetch_gba_for_bbox(
        bbox=(-0.01, 39.99, 0.01, 40.01),
        cache_dir=str(cache),
    )
    assert out is not None
    assert len(out) == 2
    # Height column normalization happens in getGBABuildings, not here —
    # but we can verify the source 'Height' came through.
    assert "Height" in out.columns or "height_m" in out.columns


def test_fetch_gba_for_bbox_returns_none_when_no_grids_match(tmp_path):
    from urbanworm.utils import building as B
    cache = tmp_path / "cache"
    cache.mkdir()
    grid_dir = cache / "world_grid"
    grid_dir.mkdir()
    grid = gpd.GeoDataFrame(
        [{"fid": 0, "geometry":
            Polygon([(50, 0), (51, 0), (51, 1), (50, 1)])}],
        crs="EPSG:4326",
    )
    grid.to_file(grid_dir / "world_grid.shp")
    (cache / "world_grid.zip").write_bytes(b"\x00")
    (cache / "data_links.txt").write_text(
        "1. (grid ID: 0-0)\nhttps://figshare.com/articles/dataset/foo/1\n"
    )
    out = B.fetch_gba_for_bbox(bbox=(-1, -1, 1, 1), cache_dir=str(cache))
    assert out is None
