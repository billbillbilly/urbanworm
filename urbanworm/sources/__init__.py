"""Per-API source modules. Each re-exports the corresponding free function
defined in :mod:`urbanworm.dataset`. Provided for a more discoverable layout:

    from urbanworm.sources.mapillary import getSV
    from urbanworm.sources.flickr import getPhoto
    from urbanworm.sources.freesound import getSound
    from urbanworm.sources.aporee import getSoundAporee
"""
from __future__ import annotations

from . import aporee, flickr, freesound, mapillary  # noqa: F401

__all__ = ["mapillary", "flickr", "freesound", "aporee"]
