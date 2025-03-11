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