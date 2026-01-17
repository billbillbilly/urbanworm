from __future__ import annotations
import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Polygon
from .utils import *

_GEOD = Geod(ellps="WGS84")


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
    import mercantile
    from tqdm import tqdm
    import tempfile
    from shapely import geometry

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
                print(f"Warning: Multiple rows found for QuadKey: {quad_key}. Processing all entries.")
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