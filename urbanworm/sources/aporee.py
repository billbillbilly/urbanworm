"""Radio Aporee audio source.

Re-exports the helpers that live in :mod:`urbanworm.dataset`:

* :func:`getSoundAporee` — filter a catalog by spatial proximity
* :func:`fetch_aporee_catalog` — fetch the catalog from Internet Archive
* :func:`enrich_aporee_catalog` — probe URLs for ``duration_s``
"""
from __future__ import annotations

from ..dataset import enrich_aporee_catalog, fetch_aporee_catalog, getSoundAporee

__all__ = ["getSoundAporee", "fetch_aporee_catalog", "enrich_aporee_catalog"]
