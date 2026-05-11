from __future__ import annotations

import logging
import re
from pathlib import Path

import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Polygon

from .utils import *  # noqa: F401,F403

_GEOD = Geod(ellps="WGS84")
logger = logging.getLogger("urbanworm")


def getOSMbuildings(
    bbox: Union[tuple, list],
    min_area: Union[float, int] = 0,
    max_area: Optional[Union[float, int]] = None,
    timeout: int = 9999,
) -> Optional[gpd.GeoDataFrame]:
    """
    Get building footprints within a bounding box from OpenStreetMap using the Overpass API.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        min_area: minimum footprint area in square meters
        max_area: maximum footprint area in square meters (None = no upper limit)
        timeout: request timeout in seconds

    Returns:
        GeoDataFrame in EPSG:4326, or None if no buildings found.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    south, west, north, east = min_lat, min_lon, max_lat, max_lon

    url = "https://overpass-api.de/api/interpreter"

    # Correct Overpass QL settings syntax (single chain ending with ;)
    query = f"""
[out:json][timeout:{timeout}];
(
  way["building"]({south},{west},{north},{east});
  relation["building"]({south},{west},{north},{east});
);
out geom;
""".strip()

    headers = {
        "User-Agent": "urban-worm/1.0",
        "Accept": "application/json",
    }

    r = requests.post(url, data={"data": query}, headers=headers, timeout=timeout + 30)

    # If Overpass errors, it often returns HTML/text/XML -> show a helpful message
    if r.status_code != 200:
        raise RuntimeError(f"Overpass HTTP {r.status_code}. Head: {r.text[:300]}")

    if not r.text.strip():
        raise RuntimeError("Overpass returned an empty response body.")

    try:
        data = r.json()
    except Exception as e:
        ctype = r.headers.get("Content-Type")
        raise RuntimeError(
            f"Overpass did not return JSON (Content-Type={ctype}). Head: {r.text[:300]}"
        ) from e

    buildings = []
    for element in data.get("elements", []):
        geom = element.get("geometry")
        if not geom:
            continue

        coords = [(node["lon"], node["lat"]) for node in geom]
        if len(coords) < 3:
            continue

        # Close ring if needed
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue

        area_m2 = abs(_GEOD.geometry_area_perimeter(poly)[0])

        if area_m2 >= float(min_area) and (max_area is None or area_m2 <= float(max_area)):
            buildings.append(poly)

    if len(buildings) < 1:
        return None
    return gpd.GeoDataFrame(geometry=buildings, crs="EPSG:4326")


# get building footprints from open building footprints released by Bing Maps using a bbox
# Adopted code is originally from https://github.com/microsoft/GlobalMLBuildingFootprints.git
# Credits to contributors @GlobalMLBuildingFootprints.
def getGlobalMLBuilding(bbox: tuple | list, min_area: float | int = 0.0,
                        max_area: float | int = None) -> gpd.GeoDataFrame:
    """
    getGlobalMLBuilding

    Fetch building footprints from the Global ML Building dataset within a given bounding box.

    Args:
        bbox (tuple or list): Bounding box defined as (min_lon, min_lat, max_lon, max_lat).
        min_area (float or int): Minimum building footprint area in square meters. Defaults to 0.0.
        max_area (float or int, optional): Maximum building footprint area in square meters. Defaults to None (no upper limit).

    Returns:
        gpd.GeoDataFrame: Filtered building footprints within the bounding box.
    """
    import tempfile

    import mercantile
    from shapely import geometry
    from tqdm import tqdm

    def filter_area(data, minm=0, maxm=None):
        utm = data.estimate_utm_crs()
        data = data.to_crs(utm)
        data["footprint_area"] = data.geometry.area
        data = data[data["footprint_area"] >= float(minm)]
        if maxm is not None:
            data = data[data["footprint_area"] < float(maxm)]
        return data.to_crs(epsg=4326)

    min_lon, min_lat, max_lon, max_lat = bbox
    aoi_geom = {
        "coordinates": [
            [
                [min_lon, min_lat],
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat]
            ]
        ],
        "type": "Polygon"
    }
    aoi_shape = geometry.shape(aoi_geom)
    # Extract bounding box coordinates
    minx, miny, maxx, maxy = aoi_shape.bounds
    # get tiles intersect bbox
    quad_keys = set()
    for tile in list(mercantile.tiles(minx, miny, maxx, maxy, zooms=9)):
        quad_keys.add(mercantile.quadkey(tile))
    quad_keys = list(quad_keys)
    # Download the building footprints for each tile and crop with bbox
    df = pd.read_csv(
        "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv", dtype=str
    )

    idx = 0
    combined_gdf = gpd.GeoDataFrame()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download the GeoJSON files for each tile that intersects the input geometry
        tmp_fns = []
        for quad_key in tqdm(quad_keys):
            rows = df[df["QuadKey"] == quad_key]
            if rows.shape[0] == 1:
                url = rows.iloc[0]["Url"]

                df2 = pd.read_json(url, lines=True)
                df2["geometry"] = df2["geometry"].apply(geometry.shape)

                gdf = gpd.GeoDataFrame(df2, crs=4326)
                fn = os.path.join(tmpdir, f"{quad_key}.geojson")
                tmp_fns.append(fn)
                if not os.path.exists(fn):  # Skip if file already exists
                    gdf.to_file(fn, driver="GeoJSON")
            elif rows.shape[0] > 1:
                logger.warning(
                    "Multiple rows found for QuadKey %s; processing all entries.",
                    quad_key,
                )
                for _, row in rows.iterrows():
                    url = row["Url"]
                    df2 = pd.read_json(url, lines=True)
                    df2["geometry"] = df2["geometry"].apply(geometry.shape)
                    gdf = gpd.GeoDataFrame(df2, crs=4326)
                    fn = os.path.join(tmpdir, f"{quad_key}_{_}.geojson")
                    tmp_fns.append(fn)
                    if not os.path.exists(fn):  # Skip if file already exists
                        gdf.to_file(fn, driver="GeoJSON")
            else:
                raise ValueError(f"QuadKey not found in dataset: {quad_key}")
        # Merge the GeoJSON files into a single file
        for fn in tmp_fns:
            gdf = gpd.read_file(fn)  # Read each file into a GeoDataFrame
            gdf = gdf[gdf.geometry.within(aoi_shape)]  # Filter geometries within the AOI
            gdf['id'] = range(idx, idx + len(gdf))  # Update 'id' based on idx
            idx += len(gdf)
            combined_gdf = pd.concat([combined_gdf, gdf], ignore_index=True)

    combined_gdf = filter_area(combined_gdf, min_area, max_area)
    # Reproject back to WGS84
    combined_gdf = combined_gdf.to_crs(4326)
    return combined_gdf


# --------------------------------------------------------------------
# Building-height-bearing datasets — local-file loaders.
#
# Two distinct datasets are supported by gloBFPr (R package) and now this
# module:
#
#   * **3D-GloBFP** (Che et al. 2024, 2025) — Zenodo + Figshare, per-tile
#     shapefiles indexed by `world_grid.zip` + `data_links.txt`.
#     See `getGloBFP3DBuildings` / `fetch_globfp3d_for_bbox`.
#
#   * **GlobalBuildingAtlas (GBA)** (Zhu Lab @ TU Munich, ESSD 2025) —
#     HuggingFace + mediaTUM, per-tile GeoJSON indexed by
#     `representative/lod1.geojson`. See `getGBABuildings` /
#     `fetch_true_gba_for_bbox`.
# --------------------------------------------------------------------

# Common column names for building height across GBA / 3D-GloBFP / generic
# releases — whichever is found first wins, then we normalize to ``height_m``.
_HEIGHT_ALIASES = ("height_m", "height", "h", "bldg_height", "building_height", "z")


def _normalize_height_column(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """If a height-like column exists, rename it to ``height_m``."""
    for col in _HEIGHT_ALIASES:
        if col in gdf.columns:
            if col != "height_m":
                gdf = gdf.rename(columns={col: "height_m"})
            # Coerce to numeric so downstream code can math it.
            try:
                gdf["height_m"] = gpd.pd.to_numeric(gdf["height_m"], errors="coerce")  # type: ignore[attr-defined]
            except AttributeError:
                # geopandas re-exports pandas as gpd.pd in older versions; if
                # not available, fall through silently.
                pass
            return gdf
    return gdf


def getGloBFP3DBuildings(
        bbox: tuple | list,
        gba_path: str | Path | None = None,
        min_area: float | int = 0.0,
        max_area: float | int | None = None,
        cache_dir: str | Path | None = None,
        timeout: float = 120.0,
) -> gpd.GeoDataFrame | None:
    """Load **3D-GloBFP** (Che et al. 2024, 2025) footprints inside ``bbox``.

    *Not to be confused with the GlobalBuildingAtlas (GBA) dataset by the
    Zhu Lab — that's a separate dataset on HuggingFace + mediaTUM. See
    :func:`getGBABuildings` for GBA.*

    Two operating modes:

    1. **Auto-fetch from Zenodo + Figshare** (default — when ``gba_path`` is
       ``None``). Mirrors the workflow from the R package gloBFPr:

       - Download the global grid manifest (`world_grid.zip` from Zenodo
         record ``15487037``) and ``data_links.txt`` once into ``cache_dir``.
       - Spatially intersect the grid with ``bbox`` to find tile IDs.
       - Look up each tile's Figshare collection URL in ``data_links.txt``.
       - For each tile, query the Figshare API for the matching per-tile
         shapefile zip, download it, and extract.
       - Concat, clip to bbox, return EPSG:4326 GeoDataFrame.

    2. **Local file** (when ``gba_path`` is provided). Skips all network
       activity — useful for offline / pre-downloaded workflows. Accepts any
       vector format ``geopandas.read_file`` understands (GPKG, GeoJSON,
       Shapefile, Parquet).

    The output is normalized so the height column is always called
    ``height_m`` (recognises ``height``, ``h``, ``bldg_height``,
    ``building_height``, ``z`` as aliases).
    """
    import pandas as pd

    min_lon, min_lat, max_lon, max_lat = bbox

    if gba_path is None:
        # Auto-fetch path
        gdf = fetch_globfp3d_for_bbox(bbox, cache_dir=cache_dir, timeout=timeout)
        if gdf is None or gdf.empty:
            return None
    else:
        gba_path = Path(gba_path)
        if not gba_path.exists():
            raise FileNotFoundError(
                f"GBA file not found: {gba_path}. Either drop the path to "
                "auto-fetch from Zenodo, or download the GBA tile/region "
                "from https://github.com/zhu-xlab/GlobalBuildingAtlas first."
            )
        aoi_bbox = (min_lon, min_lat, max_lon, max_lat)
        try:
            gdf = gpd.read_file(gba_path, bbox=aoi_bbox)
        except TypeError:
            gdf = gpd.read_file(gba_path)
            if not gdf.empty:
                gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat]

    if gdf is None or gdf.empty:
        logger.warning("No GBA buildings found in bbox %s", bbox)
        return None

    if gdf.crs is None:
        logger.warning("GBA file has no CRS; assuming EPSG:4326.")
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    # Area filter using planimetric area in the source's local UTM zone.
    if min_area > 0 or max_area is not None:
        utm = gdf.estimate_utm_crs()
        meters = gdf.to_crs(utm)
        meters["footprint_area"] = meters.geometry.area
        if min_area > 0:
            meters = meters[meters["footprint_area"] >= float(min_area)]
        if max_area is not None:
            meters = meters[meters["footprint_area"] < float(max_area)]
        gdf = meters.to_crs(4326)
        if gdf.empty:
            logger.warning(
                "GBA buildings present but none match area filter "
                "[min=%s, max=%s] in bbox %s", min_area, max_area, bbox,
            )
            return None

    gdf = _normalize_height_column(gdf)
    # Keep the index clean
    gdf = gdf.reset_index(drop=True)

    # Coerce height_m to numeric if the rename happened above the pandas
    # fallback (separate from the GeoDataFrame .pd alias attempt).
    if "height_m" in gdf.columns:
        gdf["height_m"] = pd.to_numeric(gdf["height_m"], errors="coerce")

    n_with_height = (gdf["height_m"].notna().sum()
                     if "height_m" in gdf.columns else 0)
    logger.info(
        "GBA: loaded %d buildings (%d with height_m) from %s",
        len(gdf), n_with_height, gba_path,
    )
    return gdf


# --------------------------------------------------------------------
# 3D-GloBFP (Che et al. 2024, 2025) auto-fetch pipeline.
# Mirrors the workflow used by the R package gloBFPr:
#   1) Download world_grid.zip + data_links.txt from Zenodo
#   2) Spatially intersect the grid with the user's bbox
#   3) Look up each intersecting grid_id in data_links.txt to find its
#      figshare collection URL
#   4) Hit the figshare API to find the per-tile zip(s) and download them
#   5) Read the unzipped shapefile(s) and concat
# Everything is cached in `cache_dir` (default ~/.cache/urbanworm/globfp3d).
#
# NOTE: this is *not* the GlobalBuildingAtlas (GBA) by the Zhu Lab —
# GBA is hosted on HuggingFace + mediaTUM and is handled separately
# below. The two datasets are distinct; gloBFPr supports both.
# --------------------------------------------------------------------

GLOBFP3D_ZENODO_RECORD = "15487037"
GLOBFP3D_ZENODO_BASE = f"https://zenodo.org/records/{GLOBFP3D_ZENODO_RECORD}/files"
GLOBFP3D_GRID_URL = f"{GLOBFP3D_ZENODO_BASE}/world_grid.zip?download=1"
GLOBFP3D_LINKS_URL = f"{GLOBFP3D_ZENODO_BASE}/data_links.txt?download=1"
GLOBFP3D_FIGSHARE_API = "https://api.figshare.com/v2/articles"

# Backwards-compat aliases — kept so older code that imported the GBA-prefixed
# constants doesn't break. New code should use the GLOBFP3D_* names.
GBA_ZENODO_RECORD = GLOBFP3D_ZENODO_RECORD
GBA_ZENODO_BASE = GLOBFP3D_ZENODO_BASE
GBA_GRID_URL = GLOBFP3D_GRID_URL
GBA_LINKS_URL = GLOBFP3D_LINKS_URL
GBA_FIGSHARE_API = GLOBFP3D_FIGSHARE_API


def _default_globfp3d_cache_dir() -> Path:
    return Path.home() / ".cache" / "urbanworm" / "globfp3d"


# Backwards-compat alias
def _default_gba_cache_dir() -> Path:  # noqa: D401
    return _default_globfp3d_cache_dir()


def _stream_download(url: str, dest: Path, timeout: float = 120.0) -> None:
    """Stream-download ``url`` to ``dest`` (atomic rename via .partial)."""
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    with requests.get(url, stream=True, timeout=timeout,
                      headers={"User-Agent": "urban-worm/0.x (+gba fetch)"}) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.rename(dest)


def parse_globfp3d_data_links(text: str) -> dict[int, str]:
    """Parse the body of ``data_links.txt`` into ``{grid_id: collection_url}``.

    The file is structured as::

        N. 3D-GloBFP: ... (PART X, grid ID: A-B)
        https://figshare.com/articles/.../<article_id>

    We expand each ``A-B`` into the full integer range and map every ID to
    the URL line that follows.
    """
    out: dict[int, str] = {}
    pattern = re.compile(r"grid\s*ID:\s*(\d+)\s*-\s*(\d+)", re.IGNORECASE)
    lines = [ln.rstrip() for ln in text.splitlines()]
    i = 0
    while i < len(lines):
        m = pattern.search(lines[i])
        if not m:
            i += 1
            continue
        start, end = int(m.group(1)), int(m.group(2))
        if end < start:
            start, end = end, start
        # Find the next URL line within the next few lines
        url = ""
        for j in range(i + 1, min(i + 4, len(lines))):
            ln = lines[j].strip()
            if ln.startswith("http"):
                url = ln
                break
        for gid in range(start, end + 1):
            out[gid] = url
        i += 2
    return out


def figshare_article_id(url: str) -> str:
    """Extract the numeric figshare article id from a collection URL."""
    m = re.search(r"/(\d+)(?:/[^/]*)?/?$", url)
    if not m:
        raise ValueError(f"Cannot extract figshare article id from {url!r}")
    return m.group(1)


def _figshare_list_files(article_id: str, timeout: float = 60.0) -> list[dict]:
    """Return the file list for a figshare article via its public API."""
    import requests
    api = f"{GBA_FIGSHARE_API}/{article_id}"
    r = requests.get(
        api,
        timeout=timeout,
        headers={
            "Accept": "application/json",
            "User-Agent": "urban-worm/0.x (+gba fetch)",
        },
    )
    r.raise_for_status()
    return r.json().get("files", [])


def _match_grid_id(filename: str, grid_id: int) -> bool:
    """Filename-matching heuristics for ``<grid_id>.zip`` / ``<grid_id>.shp.zip``
    / ``tile_<grid_id>.zip``. Uses a word-boundary check so 0 doesn't match 100."""
    base = filename.lower()
    s = str(grid_id)
    return bool(
        re.search(rf"(?<![\d]){s}(?![\d])", base)
        and (base.endswith(".zip") or base.endswith(".shp"))
    )


def load_globfp3d_grid_manifest(cache_dir: Path | str | None = None) -> gpd.GeoDataFrame:
    """Download (once) and load the 3D-GloBFP global grid shapefile from Zenodo."""
    cache_dir = Path(cache_dir or _default_globfp3d_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)

    grid_zip = cache_dir / "world_grid.zip"
    if not grid_zip.exists():
        logger.info("3D-GloBFP: downloading grid manifest from %s", GLOBFP3D_GRID_URL)
        _stream_download(GLOBFP3D_GRID_URL, grid_zip)

    extract_dir = cache_dir / "world_grid"
    if not extract_dir.exists() or not list(extract_dir.rglob("*.shp")):
        import zipfile
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(grid_zip) as z:
            z.extractall(extract_dir)

    shp_files = list(extract_dir.rglob("*.shp"))
    if not shp_files:
        raise RuntimeError(f"world_grid.zip contains no .shp file (cache: {cache_dir})")

    grid = gpd.read_file(shp_files[0])
    if grid.crs is None:
        grid = grid.set_crs(4326)
    elif grid.crs.to_epsg() != 4326:
        grid = grid.to_crs(4326)
    return grid


def load_globfp3d_data_links(cache_dir: Path | str | None = None) -> dict[int, str]:
    """Download (once) and parse 3D-GloBFP data_links.txt from Zenodo."""
    cache_dir = Path(cache_dir or _default_globfp3d_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    links_file = cache_dir / "data_links.txt"
    if not links_file.exists():
        logger.info("3D-GloBFP: downloading data_links.txt from %s", GLOBFP3D_LINKS_URL)
        _stream_download(GLOBFP3D_LINKS_URL, links_file)
    return parse_globfp3d_data_links(links_file.read_text(encoding="utf-8", errors="replace"))


def download_globfp3d_tile(
        grid_id: int,
        collection_url: str,
        cache_dir: Path | str | None = None,
        timeout: float = 120.0,
) -> list[Path]:
    """Resolve ``grid_id`` within a Figshare collection and download the 3D-GloBFP tile.

    Returns the local paths to the unzipped shapefile(s).
    """
    import zipfile

    cache_dir = Path(cache_dir or _default_globfp3d_cache_dir())
    tile_dir = cache_dir / "tiles" / str(grid_id)
    existing = list(tile_dir.rglob("*.shp")) if tile_dir.exists() else []
    if existing:
        return existing

    article_id = figshare_article_id(collection_url)
    files = _figshare_list_files(article_id, timeout=timeout)

    matched = [f for f in files if _match_grid_id(f.get("name", ""), grid_id)]
    if not matched:
        raise FileNotFoundError(
            f"No file matching grid_id={grid_id} in figshare article {article_id}"
        )

    tile_dir.mkdir(parents=True, exist_ok=True)
    for f in matched:
        name = f["name"]
        url = f.get("download_url") or f.get("url") or ""
        if not url:
            continue
        dest = tile_dir / name
        if not dest.exists():
            logger.info("GBA: downloading tile %d (%s)", grid_id, name)
            _stream_download(url, dest, timeout=timeout)
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(dest) as z:
                    z.extractall(tile_dir)
            except zipfile.BadZipFile as e:
                logger.warning("GBA: %s is not a valid zip (%s)", dest, e)

    out = list(tile_dir.rglob("*.shp"))
    if not out:
        raise RuntimeError(f"No .shp extracted for grid_id={grid_id} in {tile_dir}")
    return out


def fetch_globfp3d_for_bbox(
        bbox: tuple | list,
        cache_dir: Path | str | None = None,
        timeout: float = 120.0,
) -> gpd.GeoDataFrame | None:
    """Auto-fetch 3D-GloBFP buildings for a bbox.

    1. Loads the world grid manifest (download once, cache).
    2. Spatially intersects the grid with ``bbox`` to find tile IDs.
    3. Looks up each intersecting tile in ``data_links.txt`` to find its
       Figshare collection and downloads the matching per-tile shapefile.
    4. Concats all tiles, clips to bbox, returns an EPSG:4326 GeoDataFrame
       with the height column normalized to ``height_m``.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
        cache_dir: Directory for caching downloads. Default
            ``~/.cache/urbanworm/gba``.
        timeout: Per-request HTTP timeout (seconds).

    Returns:
        GeoDataFrame or ``None`` if no buildings are found in the bbox.
    """
    import pandas as pd

    cache_dir = Path(cache_dir or _default_globfp3d_cache_dir())
    grid = load_globfp3d_grid_manifest(cache_dir)
    links = load_globfp3d_data_links(cache_dir)

    min_lon, min_lat, max_lon, max_lat = bbox
    intersect = grid.cx[min_lon:max_lon, min_lat:max_lat]
    if intersect.empty:
        logger.warning("3D-GloBFP: no grid tiles intersect bbox %s", bbox)
        return None

    # Identify the grid-id column (varies between releases).
    id_col = None
    for cand in ("fid", "FID", "id", "ID", "gid", "GID", "grid_id", "GRID_ID"):
        if cand in intersect.columns:
            id_col = cand
            break
    if id_col is None:
        intersect = intersect.reset_index().rename(columns={"index": "fid"})
        id_col = "fid"

    grid_ids = sorted(set(int(x) for x in intersect[id_col]))
    logger.info("3D-GloBFP: bbox intersects %d tile(s): %s", len(grid_ids), grid_ids)

    frames = []
    for gid in grid_ids:
        if gid not in links:
            logger.warning("3D-GloBFP: grid_id %d not listed in data_links.txt", gid)
            continue
        try:
            shp_paths = download_globfp3d_tile(gid, links[gid], cache_dir, timeout=timeout)
        except Exception as e:
            logger.warning("3D-GloBFP: failed to download tile %d (%s)", gid, e)
            continue
        for p in shp_paths:
            try:
                tile = gpd.read_file(p)
                if tile.crs is None:
                    tile = tile.set_crs(4326)
                elif tile.crs.to_epsg() != 4326:
                    tile = tile.to_crs(4326)
                frames.append(tile)
            except Exception as e:
                logger.warning("3D-GloBFP: failed to read %s (%s)", p, e)

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    out = out.cx[min_lon:max_lon, min_lat:max_lat]
    return out if not out.empty else None


# Backwards-compat aliases for earlier "gba"-prefixed function names.
# These previously referred to the 3D-GloBFP pipeline; the canonical names
# now use the `globfp3d_` prefix.
parse_gba_data_links = parse_globfp3d_data_links
load_gba_grid_manifest = load_globfp3d_grid_manifest
load_gba_data_links = load_globfp3d_data_links
download_gba_tile = download_globfp3d_tile
fetch_gba_for_bbox = fetch_globfp3d_for_bbox


# --------------------------------------------------------------------
# GlobalBuildingAtlas (Zhu Lab @ TU Munich) — the *real* GBA dataset.
# Hosted on HuggingFace (zhu-xlab/GBA.LoD1, zhu-xlab/GBA.ODbLPolygon)
# and mediaTUM (m1837832 for heights). The polygon manifest lives at
# `representative/lod1.geojson`. Tiles are EPSG:3857 GeoJSON.
#
# This pipeline:
#   1) Download `lod1.geojson` (polygon-tile manifest) from HuggingFace.
#   2) Spatially intersect with the user's bbox to find tile names.
#   3) Download each per-tile GeoJSON from the appropriate split.
#   4) Concat, reproject to EPSG:4326, return.
#
# Heights live in a separate dataset (mediaTUM `m1837832`) and need a
# separate download — exposed as a follow-up enrichment step.
# --------------------------------------------------------------------

GBA_HF_LOD1_REPO = "zhu-xlab/GBA.LoD1"
GBA_HF_ODBL_REPO = "zhu-xlab/GBA.ODbLPolygon"
GBA_HF_BASE = "https://huggingface.co/datasets"
GBA_LOD1_MANIFEST_URL = (
    f"{GBA_HF_BASE}/{GBA_HF_LOD1_REPO}/resolve/main/representative/lod1.geojson"
)
GBA_HEIGHT_ZIP_MANIFEST_URL = (
    f"{GBA_HF_BASE}/{GBA_HF_LOD1_REPO}/resolve/main/representative/height_zip.geojson"
)
GBA_HEIGHT_TIF_MANIFEST_URL = (
    f"{GBA_HF_BASE}/{GBA_HF_LOD1_REPO}/resolve/main/representative/height_tif.geojson"
)


def _default_true_gba_cache_dir() -> Path:
    return Path.home() / ".cache" / "urbanworm" / "gba"


def load_true_gba_polygon_manifest(
        cache_dir: Path | str | None = None,
        url: str = GBA_LOD1_MANIFEST_URL,
) -> gpd.GeoDataFrame:
    """Download (once) and load GBA's `lod1.geojson` polygon-tile manifest.

    The manifest is a GeoDataFrame whose rows describe per-tile polygon
    files — its columns vary across releases but typically include a tile
    identifier and the relative path on HuggingFace.
    """
    cache_dir = Path(cache_dir or _default_true_gba_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "lod1.geojson"
    if not dest.exists():
        logger.info("GBA: downloading polygon-tile manifest from %s", url)
        _stream_download(url, dest)
    return gpd.read_file(dest)


def fetch_true_gba_for_bbox(
        bbox: tuple | list,
        cache_dir: Path | str | None = None,
        timeout: float = 120.0,
        include_heights: bool = False,
) -> gpd.GeoDataFrame | None:
    """Auto-fetch GlobalBuildingAtlas (Zhu Lab) building polygons for ``bbox``.

    Mirrors the gloBFPr workflow for the real GBA dataset:

    1. Download ``lod1.geojson`` (polygon-tile manifest) from HuggingFace.
    2. Spatially intersect with ``bbox`` to find tile names.
    3. Download each per-tile GeoJSON (HuggingFace Polygon / ODbLPolygon
       splits depending on the manifest's source column).
    4. Concat, reproject to EPSG:4326, return.

    Args:
        bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in EPSG:4326.
        cache_dir: Default ``~/.cache/urbanworm/gba``.
        timeout: Per-request HTTP timeout.
        include_heights: If True, also fetch and join GBA.Height rasters
            from mediaTUM. **Currently a no-op stub** — the true GBA height
            join requires raster sampling per polygon centroid, which is
            best handled by the user with `rasterio.sample`. Tracking issue
            in CHANGELOG.

    Returns:
        GeoDataFrame in EPSG:4326 (geometry only — no per-row height yet
        unless ``include_heights=True`` and the heights step is wired up).
        ``None`` if no tiles intersect the bbox.

    Notes:
        Tile-file column in the manifest is auto-detected from common
        candidates (`tile_id`, `name`, `path`, `tile`, `id`). If your
        manifest uses a different column, post-rename it before calling.
    """
    import pandas as pd

    cache_dir = Path(cache_dir or _default_true_gba_cache_dir())
    manifest = load_true_gba_polygon_manifest(cache_dir, GBA_LOD1_MANIFEST_URL)

    # Manifest is in EPSG:4326 most of the time; reproject if not.
    if manifest.crs is None:
        manifest = manifest.set_crs(4326)
    elif manifest.crs.to_epsg() != 4326:
        manifest = manifest.to_crs(4326)

    min_lon, min_lat, max_lon, max_lat = bbox
    intersect = manifest.cx[min_lon:max_lon, min_lat:max_lat]
    if intersect.empty:
        logger.warning("GBA: no polygon tiles intersect bbox %s", bbox)
        return None

    # Detect the column that names the per-tile file or path.
    name_col = None
    for cand in ("path", "file", "filename", "tile_id", "tile", "name", "id"):
        if cand in intersect.columns:
            name_col = cand
            break
    if name_col is None:
        raise RuntimeError(
            "GBA polygon manifest has no recognizable tile-name column; "
            "got columns: " + ", ".join(intersect.columns)
        )

    tiles_dir = cache_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for _, row in intersect.iterrows():
        rel = str(row[name_col])
        # Build the full HF URL — manifest entries can be either a relative
        # path inside the repo, or already an absolute URL.
        if rel.startswith("http"):
            tile_url = rel
            tile_name = rel.rsplit("/", 1)[-1]
        else:
            tile_url = f"{GBA_HF_BASE}/{GBA_HF_LOD1_REPO}/resolve/main/{rel.lstrip('/')}"
            tile_name = rel.rsplit("/", 1)[-1] or rel
        dest = tiles_dir / tile_name
        if not dest.exists():
            try:
                logger.info("GBA: downloading tile %s", tile_name)
                _stream_download(tile_url, dest, timeout=timeout)
            except Exception as e:
                logger.warning("GBA: failed to download %s (%s)", tile_url, e)
                continue
        try:
            tile = gpd.read_file(dest)
            if tile.crs is None:
                # Per the GBA README, files are EPSG:3857
                tile = tile.set_crs(3857)
            if tile.crs.to_epsg() != 4326:
                tile = tile.to_crs(4326)
            frames.append(tile)
        except Exception as e:
            logger.warning("GBA: failed to read %s (%s)", dest, e)

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    out = out.cx[min_lon:max_lon, min_lat:max_lat]
    if out.empty:
        return None

    if include_heights:
        # TODO: wire up GBA.Height raster sampling from mediaTUM.
        # The dataset is split between height_zip.geojson (zip tiles) and
        # height_tif.geojson (TIF tiles); per-polygon height is then
        # obtained via raster sampling at each centroid (rasterio.sample).
        logger.warning(
            "GBA: include_heights=True is not yet implemented; returning "
            "polygons only. Track progress in CHANGELOG."
        )

    return _normalize_height_column(out)


def getGBABuildings(
        bbox: tuple | list,
        gba_path: str | Path | None = None,
        min_area: float | int = 0.0,
        max_area: float | int | None = None,
        cache_dir: str | Path | None = None,
        timeout: float = 120.0,
        include_heights: bool = False,
) -> gpd.GeoDataFrame | None:
    """Load **GlobalBuildingAtlas** (Zhu et al., ESSD 2025) building polygons.

    *Distinct from 3D-GloBFP — see :func:`getGloBFP3DBuildings` for that
    dataset.*

    Two operating modes:

    1. **Auto-fetch from HuggingFace** (default — when ``gba_path`` is
       ``None``). Mirrors the gloBFPr workflow for the real GBA dataset:

       - Downloads ``representative/lod1.geojson`` (the polygon-tile
         manifest) from HuggingFace ``zhu-xlab/GBA.LoD1`` once into
         ``cache_dir``.
       - Spatially intersects the manifest with ``bbox`` to find tile
         names.
       - Downloads each per-tile GeoJSON, reprojects from EPSG:3857
         (GBA's native CRS) to EPSG:4326, concats, clips to bbox.

    2. **Local file** (when ``gba_path`` is provided). Skips network
       activity. Accepts any vector format ``geopandas.read_file`` reads
       (GeoJSON, GPKG, Shapefile, Parquet).

    Args:
        bbox (tuple|list): ``(min_lon, min_lat, max_lon, max_lat)`` in
            EPSG:4326.
        gba_path (str|Path, optional): Path to a local GBA polygon file.
            If ``None``, auto-fetch from HuggingFace.
        min_area (float|int): Minimum footprint area in m² (default 0).
        max_area (float|int, optional): Maximum footprint area in m².
        cache_dir (str|Path, optional): Where to cache the auto-fetched
            manifest and per-tile downloads. Default
            ``~/.cache/urbanworm/gba``.
        timeout (float): Per-request HTTP timeout (seconds).
        include_heights (bool): If True, attempt to join GBA.Height
            rasters from mediaTUM. **Currently a no-op** (raster sampling
            isn't wired up yet; tracking issue in CHANGELOG).

    Returns:
        GeoDataFrame in EPSG:4326 with at least ``geometry`` and (when the
        source provides it) ``height_m`` columns. ``None`` if no buildings.

    Note:
        GBA polygons come without per-row height in their bare form.
        Heights are a separate dataset on mediaTUM (`m1837832`) — until
        ``include_heights`` is implemented, downstream code that wants
        per-building height should fall back to the global
        ``building_height`` default in ``get_svi_from_locations``.
    """
    import pandas as pd

    min_lon, min_lat, max_lon, max_lat = bbox

    if gba_path is None:
        gdf = fetch_true_gba_for_bbox(
            bbox, cache_dir=cache_dir, timeout=timeout,
            include_heights=include_heights,
        )
        if gdf is None or gdf.empty:
            return None
    else:
        gba_path = Path(gba_path)
        if not gba_path.exists():
            raise FileNotFoundError(
                f"GBA file not found: {gba_path}. Either drop the path to "
                "auto-fetch from HuggingFace, or download the GBA tile "
                "from https://huggingface.co/datasets/zhu-xlab/GBA.LoD1 "
                "first."
            )
        aoi_bbox = (min_lon, min_lat, max_lon, max_lat)
        try:
            gdf = gpd.read_file(gba_path, bbox=aoi_bbox)
        except TypeError:
            gdf = gpd.read_file(gba_path)
            if not gdf.empty:
                gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat]

    if gdf is None or gdf.empty:
        logger.warning("No GBA buildings found in bbox %s", bbox)
        return None

    if gdf.crs is None:
        # GBA's native CRS is EPSG:3857 per the README
        gdf = gdf.set_crs(3857).to_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    if min_area > 0 or max_area is not None:
        utm = gdf.estimate_utm_crs()
        meters = gdf.to_crs(utm)
        meters["footprint_area"] = meters.geometry.area
        if min_area > 0:
            meters = meters[meters["footprint_area"] >= float(min_area)]
        if max_area is not None:
            meters = meters[meters["footprint_area"] < float(max_area)]
        gdf = meters.to_crs(4326)
        if gdf.empty:
            logger.warning(
                "GBA buildings present but none match area filter "
                "[min=%s, max=%s] in bbox %s", min_area, max_area, bbox,
            )
            return None

    gdf = _normalize_height_column(gdf)
    gdf = gdf.reset_index(drop=True)

    if "height_m" in gdf.columns:
        gdf["height_m"] = pd.to_numeric(gdf["height_m"], errors="coerce")

    n_with_height = (
        int(gdf["height_m"].notna().sum()) if "height_m" in gdf.columns else 0
    )
    logger.info(
        "GBA: loaded %d buildings (%d with height_m) for bbox %s",
        len(gdf), n_with_height, bbox,
    )
    return gdf
