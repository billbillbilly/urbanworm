"""Tests for the fov='auto' geometry helpers and getSV integration."""
from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

from urbanworm.utils.utils import (
    auto_fov_from_distance,
    auto_fov_from_polygon,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _square_polygon_at(centre_lon: float, centre_lat: float, half_m: float) -> Polygon:
    """Build a roughly axis-aligned square polygon ``half_m`` half-extent."""
    # 1 degree of latitude ≈ 111_320 m; 1 degree of longitude ≈ that * cos(lat)
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * math.cos(math.radians(centre_lat)))
    return Polygon([
        (centre_lon - dlon, centre_lat - dlat),
        (centre_lon + dlon, centre_lat - dlat),
        (centre_lon + dlon, centre_lat + dlat),
        (centre_lon - dlon, centre_lat + dlat),
    ])


# ----------------------------------------------------------------------
# auto_fov_from_distance — pure trig
# ----------------------------------------------------------------------
# These tests pin the *width-only* path; pass building_height_m=0 to
# disable the height term so the assertions stay deterministic.

def test_distance_fov_inverse_relationship():
    """Closer = wider FOV needed for the same building width."""
    near = auto_fov_from_distance(distance_m=20, building_width_m=15, margin=0.0,
                                  building_height_m=0)
    far = auto_fov_from_distance(distance_m=200, building_width_m=15, margin=0.0,
                                 building_height_m=0)
    assert near > far


def test_distance_fov_known_value_at_50m():
    """A 15 m wide building 50 m away → 2*atan(7.5/50) ≈ 17.06°."""
    fov = auto_fov_from_distance(
        distance_m=50, building_width_m=15, margin=0.0,
        min_fov=0.0, max_fov=180.0, building_height_m=0,
    )
    expected = math.degrees(2 * math.atan(7.5 / 50))
    assert fov == pytest.approx(expected, rel=1e-6)


def test_distance_fov_clamped_to_min():
    """Far distance should clamp to fov_min, not return a tiny value."""
    fov = auto_fov_from_distance(distance_m=10_000, building_width_m=15,
                                 min_fov=30.0, building_height_m=0)
    assert fov == 30.0


def test_distance_fov_clamped_to_max():
    """Standing inside the building → would compute >180°, clamp to fov_max."""
    fov = auto_fov_from_distance(distance_m=1, building_width_m=100,
                                 max_fov=120.0, building_height_m=0)
    assert fov == 120.0


def test_distance_fov_margin_applied():
    """Margin adds proportional padding to the raw FOV."""
    bare = auto_fov_from_distance(
        distance_m=50, building_width_m=15, margin=0.0,
        min_fov=0.0, max_fov=180.0, building_height_m=0,
    )
    padded = auto_fov_from_distance(
        distance_m=50, building_width_m=15, margin=0.20,
        min_fov=0.0, max_fov=180.0, building_height_m=0,
    )
    assert padded == pytest.approx(bare * 1.20, rel=1e-6)


# ----------------------------------------------------------------------
# building_height term — drives FOV when the height-derived requirement
# exceeds the width-derived one.
# ----------------------------------------------------------------------
def test_distance_fov_height_term_dominates_for_tall_narrow_building():
    """1m-wide skinny tower 30m tall, 50m away: height drives the FOV."""
    fov = auto_fov_from_distance(
        distance_m=50, building_width_m=1, building_height_m=30,
        margin=0.0, aspect_ratio=1.0,
        min_fov=0.0, max_fov=180.0,
    )
    # vertical extent ≈ 2*atan(15/50) ≈ 33.4°. With aspect=1.0,
    # horizontal_FOV needed = vertical_FOV * aspect = 33.4°.
    expected_v = math.degrees(2 * math.atan(15 / 50))
    assert fov == pytest.approx(expected_v, rel=1e-6)


def test_distance_fov_aspect_ratio_widens_height_term():
    """Wider image (larger aspect) needs more horizontal FOV for the same height."""
    fov_square = auto_fov_from_distance(
        distance_m=50, building_width_m=0, building_height_m=20,
        margin=0.0, aspect_ratio=1.0, min_fov=0.0, max_fov=180.0,
    )
    fov_wide = auto_fov_from_distance(
        distance_m=50, building_width_m=0, building_height_m=20,
        margin=0.0, aspect_ratio=1.78, min_fov=0.0, max_fov=180.0,  # 16:9
    )
    assert fov_wide > fov_square
    assert fov_wide == pytest.approx(fov_square * 1.78, rel=1e-6)


def test_distance_fov_default_height_is_9m():
    """Sanity: default building_height kicks in (>0). Not asserting an exact
    value, just that the default produces a non-trivial result."""
    fov = auto_fov_from_distance(distance_m=50, building_width_m=15,
                                 aspect_ratio=1.4, margin=0.0,
                                 min_fov=0.0, max_fov=180.0)
    # With width=15, distance=50 → width FOV ≈ 17.06°
    # With height=9, distance=50 → vert ≈ 10.3°, horizontal=10.3*1.4 ≈ 14.4°
    # Width term wins: ~17°.
    assert 16.0 < fov < 18.0


# ----------------------------------------------------------------------
# auto_fov_from_polygon
# ----------------------------------------------------------------------
def test_polygon_fov_close_building_is_wider_than_far_one():
    """Same building, closer camera → larger FOV."""
    poly = _square_polygon_at(0.0, 40.0, half_m=8)  # 16 m wide
    near_camera_lat = 40.0 + 25 / 111_320.0  # 25 m north
    far_camera_lat = 40.0 + 200 / 111_320.0  # 200 m north

    fov_near = auto_fov_from_polygon(
        camera_lon=0.0, camera_lat=near_camera_lat, polygon=poly,
        margin=0.0, min_fov=0.0, max_fov=180.0, building_height_m=0,
    )
    fov_far = auto_fov_from_polygon(
        camera_lon=0.0, camera_lat=far_camera_lat, polygon=poly,
        margin=0.0, min_fov=0.0, max_fov=180.0, building_height_m=0,
    )
    assert fov_near > fov_far


def test_polygon_fov_matches_near_corner_geometry():
    """Camera due north of a 16 m square building, 50 m to its centroid.

    With ``building_height_m=0`` the height term is skipped so we test the
    pure-footprint geometry. The widest corners of the polygon (as seen from
    the camera) are the *near* corners at (±8, +8) — i.e. 50−8=42 m away —
    giving ``2 * atan(8 / 42)`` ≈ 21.57°.
    """
    poly = _square_polygon_at(0.0, 40.0, half_m=8)
    camera_lat = 40.0 + 50 / 111_320.0
    fov = auto_fov_from_polygon(
        camera_lon=0.0, camera_lat=camera_lat, polygon=poly,
        margin=0.0, min_fov=0.0, max_fov=180.0, building_height_m=0,
    )
    expected = math.degrees(2 * math.atan(8 / (50 - 8)))
    assert fov == pytest.approx(expected, rel=0.05)


def test_polygon_fov_margin_increases_extent():
    poly = _square_polygon_at(0.0, 40.0, half_m=8)
    camera_lat = 40.0 + 50 / 111_320.0
    bare = auto_fov_from_polygon(0.0, camera_lat, poly, margin=0.0,
                                 min_fov=0.0, max_fov=180.0,
                                 building_height_m=0)
    padded = auto_fov_from_polygon(0.0, camera_lat, poly, margin=0.25,
                                   min_fov=0.0, max_fov=180.0,
                                   building_height_m=0)
    assert padded == pytest.approx(bare * 1.25, rel=1e-6)


def test_polygon_fov_clamped():
    poly = _square_polygon_at(0.0, 40.0, half_m=8)
    very_far_lat = 40.0 + 50_000 / 111_320.0
    fov = auto_fov_from_polygon(
        0.0, very_far_lat, poly, min_fov=20.0, max_fov=120.0,
        building_height_m=0,
    )
    assert fov == 20.0


def test_polygon_fov_height_term_dominates_for_tall_skinny_building():
    """A tiny footprint (1m square) but tall building → height drives FOV."""
    poly = _square_polygon_at(0.0, 40.0, half_m=0.5)  # 1m-wide tower
    camera_lat = 40.0 + 50 / 111_320.0
    fov = auto_fov_from_polygon(
        camera_lon=0.0, camera_lat=camera_lat, polygon=poly,
        margin=0.0, min_fov=0.0, max_fov=180.0,
        building_height_m=30, aspect_ratio=1.0,
    )
    # Width contribution: ~2*atan(0.5/49.5) ≈ 1.16°  (negligible)
    # Height contribution at 50 m centroid: 2*atan(15/50) ≈ 33.4°
    # FOV should be ~the height contribution.
    expected = math.degrees(2 * math.atan(15 / 50))
    assert fov == pytest.approx(expected, rel=0.05)


def test_polygon_fov_height_term_widens_with_aspect_ratio():
    """Same height/distance but wider image aspect → larger horizontal FOV."""
    poly = _square_polygon_at(0.0, 40.0, half_m=0.5)
    camera_lat = 40.0 + 50 / 111_320.0
    fov_sq = auto_fov_from_polygon(
        0.0, camera_lat, poly, margin=0.0, min_fov=0.0, max_fov=180.0,
        building_height_m=30, aspect_ratio=1.0,
    )
    fov_wide = auto_fov_from_polygon(
        0.0, camera_lat, poly, margin=0.0, min_fov=0.0, max_fov=180.0,
        building_height_m=30, aspect_ratio=1.78,
    )
    assert fov_wide > fov_sq
    assert fov_wide == pytest.approx(fov_sq * 1.78, rel=1e-6)


def test_polygon_fov_height_zero_disables_height_term():
    """building_height_m=0 means width-only behavior (back-compat)."""
    poly = _square_polygon_at(0.0, 40.0, half_m=0.5)
    camera_lat = 40.0 + 50 / 111_320.0
    fov_h0 = auto_fov_from_polygon(
        0.0, camera_lat, poly, margin=0.0, min_fov=0.0, max_fov=180.0,
        building_height_m=0,
    )
    fov_tall = auto_fov_from_polygon(
        0.0, camera_lat, poly, margin=0.0, min_fov=0.0, max_fov=180.0,
        building_height_m=30,
    )
    assert fov_tall > fov_h0


def test_polygon_fov_requires_exterior():
    with pytest.raises(TypeError):
        auto_fov_from_polygon(0.0, 0.0, polygon="not a polygon")


# ----------------------------------------------------------------------
# getSV: parameter validation
# ----------------------------------------------------------------------
def test_getsv_auto_requires_reoriented():
    """fov='auto' without reoriented=True must raise — there's no view to size."""
    from urbanworm.dataset import getSV
    with pytest.raises(ValueError) as exc:
        getSV(
            location=[0.0, 0.0],
            key="dummy-token",
            pano=True,
            reoriented=False,
            fov="auto",
        )
    assert "reoriented" in str(exc.value).lower()
