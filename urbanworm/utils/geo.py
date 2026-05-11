"""Geographic helpers: projections, bearings, bbox math, haversine.

This module re-exports a curated set of helpers from ``utils.py`` so callers
can ``from urbanworm.utils.geo import haversine_m`` instead of pulling from
the catch-all module. The originals stay where they are for backwards
compatibility.
"""
from __future__ import annotations

from .utils import (
    auto_fov_from_distance,
    auto_fov_from_polygon,
    calculate_bearing,
    degree2dis,
    dis2degree,
    haversine_m,
    is_coordinate_in_bbox,
    lonlat_to_utm_epsg,
    meters_to_degrees,
    projection,
)

__all__ = [
    "auto_fov_from_distance",
    "auto_fov_from_polygon",
    "calculate_bearing",
    "degree2dis",
    "dis2degree",
    "haversine_m",
    "is_coordinate_in_bbox",
    "lonlat_to_utm_epsg",
    "meters_to_degrees",
    "projection",
]
