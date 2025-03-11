import geopandas as gpd
import pandas as pd
import numpy as np
from pyproj import Transformer
from pyproj import CRS
from shapely.geometry import Polygon
import math
import requests
import sys
from pano2pers import Equirectangular

# Load shapefile
def loadSHP(file):
    try:
        # Read shapefile
        gdf = gpd.read_file(file)
        # Ensure CRS is WGS84 for visualization
        gdf = gdf.to_crs("EPSG:4326")
        return gdf

    except Exception as e:
        print(f"Error reading or displaying Shapefile: {e}")
        return None

# Get street view images from Mapillary
def getSV(centroid, epsg, key, multi=False):
    bbox = projection(centroid, epsg)
    url = f"https://graph.mapillary.com/images?access_token={key}&fields=id,compass_angle,thumb_2048_url,geometry&bbox={bbox}&is_pano=true"
    # while not response or 'data' not in response:
    try:
        response = requests.get(url).json()
        # find the closest image
        response = closest(centroid, response, multi)

        svis = []
        for i in range(len(response)):
            # Extract Image ID, Compass Angle, image url, and coordinates
            img_heading = float(response.iloc[i,1])
            img_url = response.iloc[i,2]
            image_lon, image_lat = response.iloc[i,5]
            # calculate bearing to the house
            bearing_to_house = calculate_bearing(image_lat, image_lon, centroid.y, centroid.x)
            relative_heading = (bearing_to_house - img_heading) % 360
            # reframe image
            svi = Equirectangular(img_url=img_url)
            sv = svi.GetPerspective(80, relative_heading, 10, 300, 400, 128)
            svis.append(sv)
        return svis
    except:
        print("no street view image found")
        return None

# Reproject the point to the desired EPSG
def projection(self, centroid, epsg):
        x, y = self.degree2dis(centroid, epsg)
        # Get unit name (meters, degrees, etc.)
        crs = CRS.from_epsg(epsg)
        unit_name = crs.axis_info[0].unit_name
        # set search distance to 25 meters
        r = 50
        if unit_name == 'foot':
            r = 164.042
        elif unit_name == 'degree':
            print("Error: epsg must be projected system.")
            sys.exit(1)
        # set bbox
        x_min = x - r
        y_min = y - r
        x_max = x + r
        y_max = y + r
        # Convert to EPSG:4326 (Lat/Lon) 
        x_min, y_min = self.dis2degree(x_min, y_min, epsg)
        x_max, y_max = self.dis2degree(x_max, y_max, epsg)
        return f'{x_min},{y_min},{x_max},{y_max}'

# Convert distance to degree
def dis2degree(ptx, pty, epsg):
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    x, y = transformer.transform(ptx, pty)
    return x, y

# Convert degree to distance
def degree2dis(pt, epsg):
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(pt.x, pt.y)
    return x, y

# find the closest image to the house
def closest(centroid, response, multi=False):
    c = [centroid.x, centroid.y]
    res_df = pd.DataFrame(response['data'])
    res_df[['point','coordinates']] = pd.DataFrame(res_df.geometry.tolist(), index= res_df.index)
    res_df[['lon','lat']] = pd.DataFrame(res_df.coordinates.tolist(), index= res_df.index)
    id_array = np.array(res_df['id'])
    lon_array = np.array(res_df['lon'])
    lat_array = np.array(res_df['lat'])
    dis_array = (lon_array-c[0])*(lon_array-c[0]) + (lat_array-c[1])*(lat_array-c[1])
    if multi == True and len(dis_array) > 3:
        smallest_indices = np.argsort(dis_array)[:3]
        return res_df.loc[res_df['id'].isin(id_array[smallest_indices])]
    ind = np.where(dis_array == np.min(dis_array))[0]
    id = id_array[ind][0]
    return res_df.loc[res_df['id'] == id]

# filter images by time and seasons

# calculate bearing between two points
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1

    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360  # Normalize to 0-360

# get building footprints from OSM uing bbox
def getOSMbuildings(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox

    url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json]
    [timeout:900];
    (
        node["building"]({min_lat},{min_lon},{max_lat},{max_lon});
        way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
        relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom;
    """

    # payload = "data=" + requests.utils.quote(query)
    # response = requests.post(url, data=payload)
    response = requests.get(url, params={"data": query})

    
    data = response.json()

    # Extract building polygons
    buildings = []
    for element in data.get('elements', []):
        if 'tags' in element and 'building' in element['tags']:
            buildings.append(element.get('geometry'))
    # Extract building polygons
    # buildings = []
    # for element in data.get("elements", []):
    #     if "geometry" in element:
    #         coords = [(node["lon"], node["lat"]) for node in element["geometry"]]
    #         if len(coords) > 2:  # Ensure at least 3 points for a polygon
    #             buildings.append(Polygon(coords))

    # Convert to GeoDataFrame
    # gdf = gpd.GeoDataFrame(geometry=buildings, crs="EPSG:4326")
    return data

# The adapted function is from geosam and originally from https://github.com/gumblex/tms2geotiff. 
# Credits to Dr.Qiusheng Wu and the GitHub user @gumblex.
def tms_to_geotiff(
    output,
    bbox,
    zoom=None,
    resolution=None,
    source="OpenStreetMap",
    crs="EPSG:3857",
    to_cog=False,
    return_image=False,
    overwrite=False,
    quiet=False,
    **kwargs,
):
    """Download TMS tiles and convert them to a GeoTIFF. The source is adapted from https://github.com/gumblex/tms2geotiff.
        Credits to the GitHub user @gumblex.

    Args:
        output (str): The output GeoTIFF file.
        bbox (list): The bounding box [minx, miny, maxx, maxy], e.g., [-122.5216, 37.733, -122.3661, 37.8095]
        zoom (int, optional): The map zoom level. Defaults to None.
        resolution (float, optional): The resolution in meters. Defaults to None.
        source (str, optional): The tile source. It can be one of the following: "OPENSTREETMAP", "ROADMAP",
            "SATELLITE", "TERRAIN", "HYBRID", or an HTTP URL. Defaults to "OpenStreetMap".
        crs (str, optional): The output CRS. Defaults to "EPSG:3857".
        to_cog (bool, optional): Convert to Cloud Optimized GeoTIFF. Defaults to False.
        return_image (bool, optional): Return the image as PIL.Image. Defaults to False.
        overwrite (bool, optional): Overwrite the output file if it already exists. Defaults to False.
        quiet (bool, optional): Suppress output. Defaults to False.
        **kwargs: Additional arguments to pass to gdal.GetDriverByName("GTiff").Create().

    """

    import re
    import io
    import math
    import itertools
    import concurrent.futures

    from PIL import Image

    try:
        from osgeo import gdal, osr
    except ImportError:
        raise ImportError("GDAL is not installed. Install it with pip install GDAL")

    try:
        import httpx

        SESSION = httpx.Client()
    except ImportError:
        import requests

        SESSION = requests.Session()

    if not overwrite and os.path.exists(output):
        print(
            f"The output file {output} already exists. Use `overwrite=True` to overwrite it."
        )
        return

    xyz_tiles = {
        "OPENSTREETMAP": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "ROADMAP": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        "SATELLITE": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "TERRAIN": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        "HYBRID": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    }

    basemaps = get_basemaps()

    if isinstance(source, str):
        if source.upper() in xyz_tiles:
            source = xyz_tiles[source.upper()]
        elif source in basemaps:
            source = basemaps[source]
        elif source.startswith("http"):
            pass
    else:
        raise ValueError(
            'source must be one of "OpenStreetMap", "ROADMAP", "SATELLITE", "TERRAIN", "HYBRID", or a URL'
        )

    def resolution_to_zoom_level(resolution):
        """
        Convert map resolution in meters to zoom level for Web Mercator (EPSG:3857) tiles.
        """
        # Web Mercator tile size in meters at zoom level 0
        initial_resolution = 156543.03392804097

        # Calculate the zoom level
        zoom_level = math.log2(initial_resolution / resolution)

        return int(zoom_level)

    if isinstance(bbox, list) and len(bbox) == 4:
        west, south, east, north = bbox
    else:
        raise ValueError(
            "bbox must be a list of 4 coordinates in the format of [xmin, ymin, xmax, ymax]"
        )

    if zoom is None and resolution is None:
        raise ValueError("Either zoom or resolution must be provided")
    elif zoom is not None and resolution is not None:
        raise ValueError("Only one of zoom or resolution can be provided")

    if resolution is not None:
        zoom = resolution_to_zoom_level(resolution)

    EARTH_EQUATORIAL_RADIUS = 6378137.0

    Image.MAX_IMAGE_PIXELS = None

    gdal.UseExceptions()
    web_mercator = osr.SpatialReference()
    try:
        web_mercator.ImportFromEPSG(3857)
    except RuntimeError as e:
        # https://github.com/PDAL/PDAL/issues/2544#issuecomment-637995923
        if "PROJ" in str(e):
            pattern = r"/[\w/]+"
            match = re.search(pattern, str(e))
            if match:
                file_path = match.group(0)
                os.environ["PROJ_LIB"] = file_path
                os.environ["GDAL_DATA"] = file_path.replace("proj", "gdal")
                web_mercator.ImportFromEPSG(3857)

    WKT_3857 = web_mercator.ExportToWkt()

    def from4326_to3857(lat, lon):
        xtile = math.radians(lon) * EARTH_EQUATORIAL_RADIUS
        ytile = (
            math.log(math.tan(math.radians(45 + lat / 2.0))) * EARTH_EQUATORIAL_RADIUS
        )
        return (xtile, ytile)

    def deg2num(lat, lon, zoom):
        lat_r = math.radians(lat)
        n = 2**zoom
        xtile = (lon + 180) / 360 * n
        ytile = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
        return (xtile, ytile)

    def is_empty(im):
        extrema = im.getextrema()
        if len(extrema) >= 3:
            if len(extrema) > 3 and extrema[-1] == (0, 0):
                return True
            for ext in extrema[:3]:
                if ext != (0, 0):
                    return False
            return True
        else:
            return extrema[0] == (0, 0)

    def paste_tile(bigim, base_size, tile, corner_xy, bbox):
        if tile is None:
            return bigim
        im = Image.open(io.BytesIO(tile))
        mode = "RGB" if im.mode == "RGB" else "RGBA"
        size = im.size
        if bigim is None:
            base_size[0] = size[0]
            base_size[1] = size[1]
            newim = Image.new(
                mode, (size[0] * (bbox[2] - bbox[0]), size[1] * (bbox[3] - bbox[1]))
            )
        else:
            newim = bigim

        dx = abs(corner_xy[0] - bbox[0])
        dy = abs(corner_xy[1] - bbox[1])
        xy0 = (size[0] * dx, size[1] * dy)
        if mode == "RGB":
            newim.paste(im, xy0)
        else:
            if im.mode != mode:
                im = im.convert(mode)
            if not is_empty(im):
                newim.paste(im, xy0)
        im.close()
        return newim

    def finish_picture(bigim, base_size, bbox, x0, y0, x1, y1):
        xfrac = x0 - bbox[0]
        yfrac = y0 - bbox[1]
        x2 = round(base_size[0] * xfrac)
        y2 = round(base_size[1] * yfrac)
        imgw = round(base_size[0] * (x1 - x0))
        imgh = round(base_size[1] * (y1 - y0))
        retim = bigim.crop((x2, y2, x2 + imgw, y2 + imgh))
        if retim.mode == "RGBA" and retim.getextrema()[3] == (255, 255):
            retim = retim.convert("RGB")
        bigim.close()
        return retim

    def get_tile(url):
        retry = 3
        while 1:
            try:
                r = SESSION.get(url, timeout=60)
                break
            except Exception:
                retry -= 1
                if not retry:
                    raise
        if r.status_code == 404:
            return None
        elif not r.content:
            return None
        r.raise_for_status()
        return r.content

    def draw_tile(
        source, lat0, lon0, lat1, lon1, zoom, filename, quiet=False, **kwargs
    ):
        x0, y0 = deg2num(lat0, lon0, zoom)
        x1, y1 = deg2num(lat1, lon1, zoom)
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        corners = tuple(
            itertools.product(
                range(math.floor(x0), math.ceil(x1)),
                range(math.floor(y0), math.ceil(y1)),
            )
        )
        totalnum = len(corners)
        futures = []
        with concurrent.futures.ThreadPoolExecutor(5) as executor:
            for x, y in corners:
                futures.append(
                    executor.submit(get_tile, source.format(z=zoom, x=x, y=y))
                )
            bbox = (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))
            bigim = None
            base_size = [256, 256]
            for k, (fut, corner_xy) in enumerate(zip(futures, corners), 1):
                bigim = paste_tile(bigim, base_size, fut.result(), corner_xy, bbox)
                if not quiet:
                    print(
                        f"Downloaded image {str(k).zfill(len(str(totalnum)))}/{totalnum}"
                    )

        if not quiet:
            print("Saving GeoTIFF. Please wait...")
        img = finish_picture(bigim, base_size, bbox, x0, y0, x1, y1)
        imgbands = len(img.getbands())
        driver = gdal.GetDriverByName("GTiff")

        if "options" not in kwargs:
            kwargs["options"] = [
                "COMPRESS=DEFLATE",
                "PREDICTOR=2",
                "ZLEVEL=9",
                "TILED=YES",
            ]

        gtiff = driver.Create(
            filename,
            img.size[0],
            img.size[1],
            imgbands,
            gdal.GDT_Byte,
            **kwargs,
        )
        xp0, yp0 = from4326_to3857(lat0, lon0)
        xp1, yp1 = from4326_to3857(lat1, lon1)
        pwidth = abs(xp1 - xp0) / img.size[0]
        pheight = abs(yp1 - yp0) / img.size[1]
        gtiff.SetGeoTransform((min(xp0, xp1), pwidth, 0, max(yp0, yp1), 0, -pheight))
        gtiff.SetProjection(WKT_3857)
        for band in range(imgbands):
            array = np.array(img.getdata(band), dtype="u8")
            array = array.reshape((img.size[1], img.size[0]))
            band = gtiff.GetRasterBand(band + 1)
            band.WriteArray(array)
        gtiff.FlushCache()

        if not quiet:
            print(f"Image saved to {filename}")
        return img

    try:
        image = draw_tile(
            source, south, west, north, east, zoom, output, quiet, **kwargs
        )
        if return_image:
            return image
        if crs.upper() != "EPSG:3857":
            reproject(output, output, crs, to_cog=to_cog)
        elif to_cog:
            image_to_cog(output, output)
    except Exception as e:
        raise Exception(e)