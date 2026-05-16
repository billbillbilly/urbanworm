# Dataset

The `GeoTaggedData` class is the central data container for the collection pipeline.
It fetches building footprints, retrieves geo-located media (street views, photos, audio),
and hands the results off to an inference backend.

## GeoTaggedData

::: urbanworm.dataset.GeoTaggedData

---

## Standalone helpers

These functions are also available at the top level (`from urbanworm import getSV, …`)
but are more commonly called through `GeoTaggedData`.

::: urbanworm.dataset.getSV

::: urbanworm.dataset.getPhoto

::: urbanworm.dataset.getSound
