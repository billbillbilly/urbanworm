import geopandas as gpd
import pandas as pd
import numpy as np
from pyproj import Transformer
from pyproj import CRS
from shapely.geometry import Polygon
import math
import requests
import sys
import os
from .pano2pers import Equirectangular
import base64
import cv2
import matplotlib.pyplot as plt
import re
from urlsigner import sign_url

def is_base64(s):
    """Checks if a string is base64 encoded."""
    import io
    from PIL import Image
    try:
        # Decode Base64
        decoded_data = base64.b64decode(s, validate=True)
        # Verify it's an image
        image = Image.open(io.BytesIO(decoded_data))
        image.verify()
        return True
    except Exception:
        return False

def is_image_path(s):
    """Checks if a string is a valid path and if a file exists at that path, and if it is an image."""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
    return os.path.isfile(s) and s.lower().endswith(image_extensions)

def detect_input_type(input_string):
    """Detects if the input string is an image path or base64 encoded."""
    if is_image_path(input_string):
        return "image_path"
    elif is_base64(input_string):
        return "base64"
    else:
        return "unknown"

def encode_image_to_base64(image_path):
    """Encodes an image file to a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Load shapefile
def loadSHP(file):
    """
    Import shp file
    
    Returns:
        GeoDataframe: A GeoDataframe converted from a .shp file
    """
    try:
        # Read shapefile
        gdf = gpd.read_file(file)
        # Ensure CRS is WGS84
        gdf = gdf.to_crs("EPSG:4326")
        return gdf

    except Exception as e:
        print(f"Error reading or displaying Shapefile: {e}")
        return None

# offset polygon by distance
def meters_to_degrees(meters, latitude):
    """Convert meters to degrees dynamically based on latitude."""
    # Approximate adjustment
    meters_per_degree = 111320 * (1 - 0.000022 * abs(latitude))
    return meters / meters_per_degree

def getGSV(lon, lat, key:str, secret:str, multi:bool=False,
           fov:int=80, heading:int=None, pitch:int=5,
           height:int=300, width:int=400) -> list[str]:
    """
    getGSV

    Retrieve the closest street view image(s) near a coordinate using the Google Street View API.

    Args:
        lon (float): Longitude of the location.
        lat (float): Latitude of the location.
        key (str): Google API access token.
        multi (bool, optional): Whether to return multiple SVIs (default is False).
        fov (int, optional): Field of view in degrees for the perspective image. Defaults to 80.
        heading (int, optional): Camera heading in degrees. If None, it will be computed based on the house orientation.
        pitch (int, optional): Camera pitch angle. Defaults to 10.
        height (int, optional): Height in pixels of the returned image. Defaults to 300.
        width (int, optional): Width in pixels of the returned image. Defaults to 400.

    Returns:
        list[str]: A list of images in base64 format
    """
    
    try:
        y, x, date, status = request_metadata(lat, lon, key, secret)
        if status != 'OK':
            return []
        if heading is None:
            # calculate bearing to the house
            bearing_to_house = calculate_bearing(lat, lon, y, x)
            heading = (bearing_to_house + 180) % 360
        # Get image URL
        img_url = f"https://maps.googleapis.com/maps/api/streetview?size={width}x{height}&fov={fov}&heading={heading}&pitch={pitch}&location={y},{x}&key={key}"
        resp = Equirectangular.read_url2img(img_url)
        # Convert to base64
        _, buffer = cv2.imencode('.png', resp)
        sv = base64.b64encode(buffer).decode('utf-8')
        return [sv]
    except Exception as e:
        print(f"Error in getGSV: {e}")
        return []
    
def request_metadata(lat, lon, api_key, secret):
    get_meta = lambda x, y, key: f"https://maps.googleapis.com/maps/api/streetview/metadata?size=400x300&location={y},{x}&key={key}"
    
    # request
    meta_url = get_meta(lon, lat, api_key)
    meta_request = sign_url(meta_url, secret)
    response = requests.get(meta_request)
    response = response.json()
    status = response['status']
    if status == 'OK':
        lat = response['location']['lat']
        lon = response['location']['lng']
        date = response['date']
        return lat, lon, date, status
    else:
        return None, None, None, status

# Get street view images from Mapillary
def getMSV(centroid, epsg:int, key:str, multi:bool=False, 
          fov:int=80, heading:int=None, pitch:int=10, 
          height:int=300, width:int=400) -> list[str]:
    """
    getMSV

    Retrieve the closest street view image(s) near a coordinate using the Mapillary API.

    Args:
        centroid: The coordinates (geometry.centroid of GeoDataFrame)
        epsg (int): EPSG code for projecting the coordinates.
        key (str): Mapillary API access token.
        multi (bool, optional): Whether to return multiple SVIs (default is False).
        fov (int, optional): Field of view in degrees for the perspective image. Defaults to 80.
        heading (int, optional): Camera heading in degrees. If None, it will be computed based on the house orientation.
        pitch (int, optional): Camera pitch angle. Defaults to 10.
        height (int, optional): Height in pixels of the returned image. Defaults to 300.
        width (int, optional): Width in pixels of the returned image. Defaults to 400.
    
    Returns:
        list[str]: A list of images in base64 format
    """
    bbox = projection(centroid, epsg)
    
    url = f"https://graph.mapillary.com/images?access_token={key}&fields=id,compass_angle,thumb_2048_url,geometry&bbox={bbox}&is_pano=true"
    svis = []
    try:
        response = retry_request(url)
        response = response.json()
        # find the closest image
        response = closest(centroid, response, multi)

        for i in range(len(response)):
            # Extract Image ID, Compass Angle, image url, and coordinates
            img_heading = float(response.iloc[i,1])
            img_url = response.iloc[i,2]
            image_lon, image_lat = response.iloc[i,5]
            if heading is None:
                # calculate bearing to the house
                bearing_to_house = calculate_bearing(image_lat, image_lon, centroid.y, centroid.x)
                relative_heading = (bearing_to_house - img_heading) % 360
            else:
                relative_heading = heading
            # reframe image
            svi = Equirectangular(img_url=img_url)
            sv = svi.GetPerspective(fov, relative_heading, pitch, height, width, 128)
            svis.append(sv)
        return svis
    except Exception as e:
        print(f"Error in getSV: {e}")
        return []

# Reproject the point to the desired EPSG
def projection(centroid, epsg):
    x, y = degree2dis(centroid, epsg)
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
    x_min, y_min = dis2degree(x_min, y_min, epsg)
    x_max, y_max = dis2degree(x_max, y_max, epsg)
    return f'{x_min},{y_min},{x_max},{y_max}'

def retry_request(url, retries=3):
    response = None
    for _ in range(retries):
        # Check for rate limit or server error
        try:
            response = requests.get(url)
            # If the response status code is in the list, wait and retry
            if response.status_code != 200:
                continue
            else:
                return response
        except:
            pass
    return response

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
def getOSMbuildings(bbox:tuple|list, min_area:float|int=0, max_area:float|int=None) -> gpd.GeoDataFrame | None:
    """
    getOSMbuildings

    Get building footprints within a bounding box from OpenStreetMap using the Overpass API.

    Args:
        bbox (tuple or list): A bounding box in the form (min_lon, min_lat, max_lon, max_lat).
        min_area (float or int): Minimum footprint area in square meters. Defaults to 0.
        max_area (float or int, optional): Maximum footprint area in square meters. If None, no upper limit is applied.

    Returns:
        gpd.GeoDataFrame or None: A GeoDataFrame of building footprints if any are found; otherwise, None.
    """
    # Extract bounding box coordinates
    min_lon, min_lat, max_lon, max_lat = bbox

    url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [bbox:{max_lat},{max_lon},{min_lat},{min_lon}]
    [out:json]
    [timeout:900];
    (
        way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
        relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom;
    """

    payload = "data=" + requests.utils.quote(query)
    response = requests.post(url, data=payload)
    data = response.json()

    buildings = []
    for element in data.get("elements", []):
        if "geometry" in element:
            coords = [(node["lon"], node["lat"]) for node in element["geometry"]]
            if len(coords) > 2:  
                polygon = Polygon(coords)
                # Approx. conversion to square meters
                area_m2 = polygon.area * (111320 ** 2)  
                # Filter buildings by area
                if area_m2 >= min_area and (max_area is None or area_m2 <= max_area):
                    buildings.append(polygon)

    if len(buildings) == 0:
        return None
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=buildings, crs="EPSG:4326")
    return gdf

# get building footprints from open building footprints released by Bing Maps using a bbox
# Adopted code is originally from https://github.com/microsoft/GlobalMLBuildingFootprints.git
# Credits to contributors @GlobalMLBuildingFootprints.
def getGlobalMLBuilding(bbox:tuple | list, epsg:int=None, min_area:float|int=0.0, max_area:float|int=None) -> gpd.GeoDataFrame:
    """
    getGlobalMLBuilding
    
    Fetch building footprints from the Global ML Building dataset within a given bounding box.

    Args:
        bbox (tuple or list): Bounding box defined as (min_lon, min_lat, max_lon, max_lat).
        epsg (int, optional): EPSG code for coordinate transformation. Required if min_area > 0 or max_area is specified.
        min_area (float or int): Minimum building footprint area in square meters. Defaults to 0.0.
        max_area (float or int, optional): Maximum building footprint area in square meters. Defaults to None (no upper limit).

    Returns:
        gpd.GeoDataFrame: Filtered building footprints within the bounding box.
    """
    import mercantile
    from tqdm import tqdm
    import tempfile
    from shapely import geometry

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
                if not os.path.exists(fn): # Skip if file already exists
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
            combined_gdf = pd.concat([combined_gdf,gdf],ignore_index=True)
    
    # # Reproject to a UTM CRS for accurate area measurement
    # utm_crs = combined_gdf.estimate_utm_crs()  
    # # Compute area and filter buildings by area
    # combined_gdf = combined_gdf.to_crs(utm_crs)
    # combined_gdf["area_"] = combined_gdf.geometry.area
    # combined_gdf = combined_gdf[combined_gdf["area_"] >= min_area]  # Filter min area
    # if max_area:
    #     combined_gdf = combined_gdf[combined_gdf["area_"] <= max_area]  # Filter max area

    combined_gdf = filterBF(combined_gdf, epsg, min_area, max_area)
    # Reproject back to WGS84
    combined_gdf = combined_gdf.to_crs('EPSG:4326')
    return combined_gdf

def filterBF(data, epsg, minm, maxm):
    # Reproject to a CRS for accurate area measurement
    gdf_proj = data.units.to_crs(epsg)
    # Compute area and filter buildings by area
    gdf_proj['footprint_area'] = gdf_proj.geometry.area
    gdf_proj = gdf_proj[gdf_proj["footprint_area"] >= minm]  # Filter min area
    if maxm:
        gdf_proj = gdf_proj[gdf_proj["footprint_area"] <= maxm]  # Filter max area
    return gdf_proj

def response2gdf(qna_dict):
    """
    Extracts filds from QnA objects as a single dictionary.
    """

    import pandas as pd
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Point
    # import math

    # def findQgroup(num, groupSzie, totalSize):
    #     if groupSzie == 1:
    #         return 1
    #     else:
    #         return math.ceil((num/totalSize)/(groupSzie/totalSize))
        
    def renameKey(qna_list, t):
        return [{f'{t}_{key}{i+1}': qna_list[i][key] for key in qna_list[i]} for i in range(len(qna_list))]

    def extract_qna(qna, tag, fs):
        if 'QnA' not in str(type(qna[0][0])) and tag == 'street_view':
            out = [extract_qna(qna[i], tag, fs) for i in range(len(qna))]
            return out
        else:
            question_num = len(qna[0])
            fs_ = [fs for i in range(question_num)]
            fs_ = list(np.concatenate(fs_))
            # size = len(fs_)
            dic = {}
            fields = []
            for i in range(len(qna)):
                qna_ = [dict(q) for q in qna[i]]
                qna_ = renameKey(qna_, tag)
                qna_ = {k: v for d_ in qna_ for k, v in d_.items()}
                if i == 0:
                    # dic = {f'{tag}_{fs_[idx]}{findQgroup(idx+1,question_num,size)}': [] for idx in range(size)}
                    dic = {key:[] for key in qna_}
                    fields = list(dic.keys())
                for field_i in range(len(fields)):
                    try:
                        dic[fields[field_i]] += [qna_[fields[field_i]]]
                    except:
                        pass
            return dic
    
    fields, qna_top, qna_street = None, None, None
    df_top, df_street = None, None

    if "top_view" not in qna_dict and "street_view" not in qna_dict:
        raise ValueError("no response in the input data.")
    
    if "top_view" in qna_dict:
        qna_top = qna_dict['top_view']
    if "street_view" in qna_dict:
        qna_street = qna_dict['street_view']
    # if "top_view" in qna_dict and "street_view" in qna_dict:
    #     qna_top = qna_dict['top_view']
    #     qna_street = qna_dict['street_view']
    
    # Create dictionary for GeoDataFrame
    geo_data = {
        "geometry": [Point(lon, lat) for lon, lat in zip(qna_dict["lon"], qna_dict["lat"])]
    }
    geo_df = gpd.GeoDataFrame(geo_data, crs="EPSG:4326")

    if qna_top:
        fields = list(vars(qna_dict["top_view"][0][0]).keys())
        df_top = pd.DataFrame(extract_qna(qna_top, 'top_view', fields))
    if qna_street:
        try:
            fields = list(vars(qna_dict["street_view"][0][0]).keys())
        except: 
            fields = list(vars(qna_dict["street_view"][0][0][0]).keys())
        df_street = pd.DataFrame(extract_qna(qna_street, 'street_view', fields))
    
    if df_top is not None and df_street is not None:
        df_temp = pd.concat([df_top, df_street], axis=1)
        return pd.concat([geo_df, df_temp], axis=1)
    elif df_top is not None:
        return pd.concat([geo_df, df_top], axis=1)
    elif df_street is not None:
        return pd.concat([geo_df, df_top], axis=1)

def plot_base64_image(img_base64:str):
    """Decodes a Base64 image and plots it using Matplotlib."""

    import matplotlib.pyplot as plt

    # Decode Base64 to bytes
    img_data = base64.b64decode(img_base64)
    
    # Convert to NumPy array
    np_arr = np.frombuffer(img_data, np.uint8)
    
    # Decode image from memory
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB (Matplotlib expects RGB format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot the image
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()

# chat with model to analyze/summarize results
def chatpd(messages:list,
           model:str) -> list:
    import ollama

    # chat with model
    final_reply = ""
    res = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature":0.2,
            "top_k":0.8,
            "top_p":0.8
        },
        stream=True
    )
    for chunk in res:
        final_reply += chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)
    messages.append({
        "role": "assistant",
        "content": final_reply
    })
    return messages

#----------------------- To Do -----------------------

# function to plot/visualize results (images + questions + answer + explanation + ...)

#-----------------------------------------------------

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
    quiet=True,
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

# The function is from geosam. Credits to Dr.Qiusheng Wu.
def get_basemaps(free_only=True):
    """Returns a dictionary of xyz basemaps.

    Args:
        free_only (bool, optional): Whether to return only free xyz tile services that do not require an access token. Defaults to True.

    Returns:
        dict: A dictionary of xyz basemaps.
    """

    basemaps = {}
    xyz_dict = get_xyz_dict(free_only=free_only)
    for item in xyz_dict:
        name = xyz_dict[item].name
        url = xyz_dict[item].build_url()
        basemaps[name] = url

    return basemaps

# The function is from geosam. Credits to Dr.Qiusheng Wu.
def get_xyz_dict(free_only=True):
    """Returns a dictionary of xyz services.

    Args:
        free_only (bool, optional): Whether to return only free xyz tile services that do not require an access token. Defaults to True.

    Returns:
        dict: A dictionary of xyz services.
    """
    import collections
    import xyzservices.providers as xyz

    def _unpack_sub_parameters(var, param):
        temp = var
        for sub_param in param.split("."):
            temp = getattr(temp, sub_param)
        return temp

    xyz_dict = {}
    for item in xyz.values():
        try:
            name = item["name"]
            tile = _unpack_sub_parameters(xyz, name)
            if _unpack_sub_parameters(xyz, name).requires_token():
                if free_only:
                    pass
                else:
                    xyz_dict[name] = tile
            else:
                xyz_dict[name] = tile

        except Exception:
            for sub_item in item:
                name = item[sub_item]["name"]
                tile = _unpack_sub_parameters(xyz, name)
                if _unpack_sub_parameters(xyz, name).requires_token():
                    if free_only:
                        pass
                    else:
                        xyz_dict[name] = tile
                else:
                    xyz_dict[name] = tile

    xyz_dict = collections.OrderedDict(sorted(xyz_dict.items()))
    return xyz_dict

# The function is from geosam. Credits to Dr.Qiusheng Wu.
def reproject(
    image, output, dst_crs="EPSG:4326", resampling="nearest", to_cog=True, **kwargs
):
    """Reprojects an image.

    Args:
        image (str): The input image filepath.
        output (str): The output image filepath.
        dst_crs (str, optional): The destination CRS. Defaults to "EPSG:4326".
        resampling (Resampling, optional): The resampling method. Defaults to "nearest".
        to_cog (bool, optional): Whether to convert the output image to a Cloud Optimized GeoTIFF. Defaults to True.
        **kwargs: Additional keyword arguments to pass to rasterio.open.

    """
    import rasterio as rio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    if isinstance(resampling, str):
        resampling = getattr(Resampling, resampling)

    image = os.path.abspath(image)
    output = os.path.abspath(output)

    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))

    with rio.open(image, **kwargs) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rio.open(output, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    **kwargs,
                )

    if to_cog:
        image_to_cog(output, output)

# The function is from geosam. Credits to Dr.Qiusheng Wu.
def image_to_cog(source, dst_path=None, profile="deflate", **kwargs):
    """Converts an image to a COG file.

    Args:
        source (str): A dataset path, URL or rasterio.io.DatasetReader object.
        dst_path (str, optional): An output dataset path or or PathLike object. Defaults to None.
        profile (str, optional): COG profile. More at https://cogeotiff.github.io/rio-cogeo/profile. Defaults to "deflate".

    Raises:
        ImportError: If rio-cogeo is not installed.
        FileNotFoundError: If the source file could not be found.
    """
    try:
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles

    except ImportError:
        raise ImportError(
            "The rio-cogeo package is not installed. Please install it with `pip install rio-cogeo` or `conda install rio-cogeo -c conda-forge`."
        )

    if not source.startswith("http"):
        source = check_file_path(source)

        if not os.path.exists(source):
            raise FileNotFoundError("The provided input file could not be found.")

    if dst_path is None:
        if not source.startswith("http"):
            dst_path = os.path.splitext(source)[0] + "_cog.tif"
        else:
            dst_path = temp_file_path(extension=".tif")

    dst_path = check_file_path(dst_path)

    dst_profile = cog_profiles.get(profile)
    cog_translate(source, dst_path, dst_profile, **kwargs)

# The function is from geosam. Credits to Dr.Qiusheng Wu.
def check_file_path(file_path, make_dirs=True):
    """Gets the absolute file path.

    Args:
        file_path (str): The path to the file.
        make_dirs (bool, optional): Whether to create the directory if it does not exist. Defaults to True.

    Raises:
        FileNotFoundError: If the directory could not be found.
        TypeError: If the input directory path is not a string.

    Returns:
        str: The absolute path to the file.
    """
    if isinstance(file_path, str):
        if file_path.startswith("~"):
            file_path = os.path.expanduser(file_path)
        else:
            file_path = os.path.abspath(file_path)

        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir) and make_dirs:
            os.makedirs(file_dir)

        return file_path

    else:
        raise TypeError("The provided file path must be a string.")
    
# The function is from geosam. Credits to Dr.Qiusheng Wu.
def temp_file_path(extension):
    """Returns a temporary file path.

    Args:
        extension (str): The file extension.

    Returns:
        str: The temporary file path.
    """

    import tempfile
    import uuid

    if not extension.startswith("."):
        extension = "." + extension
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{extension}")

    return file_path