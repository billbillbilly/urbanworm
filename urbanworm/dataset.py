from geopandas import GeoDataFrame
from .utils.building import *
from .utils.pano2pers import Equirectangular
from .utils.utils import projection, retry_request, closest, calculate_bearing
import pandas as pd
from tqdm.auto import tqdm
import os

class GeoTaggedData:
    def __init__(self,
                 locations: list|tuple|dict|pd.DataFrame=None,
                 units: GeoDataFrame=None):
        '''
        Args:
            locations (list|tuple|dict|Dataframe): A list of coordinates (longitude/x and latitude/y) or a dictionary keyed by longitude and latitude or a dataframe with columns "longitude" and "latitude".
            units (GeoDataFrame): The path to the shapefile or geojson file, or GeoDataFrame.

        Examples:
            # retrieve street view with building footprints (OSM)
            gtd = GeoTaggedData()
            gtd.getBuildingFootprints(bbox=(-83.235572,42.348092,-83.235154,42.348806))
            gtd.get_svi_from_locations(key="your Mapillary token")

            # locations - a nested list of coordinates
            gtd = GeoTaggedData(location=[[-83.235572,42.348092],[-83.235154,42.348806]])
            # locations - a dataframe with columns "longitude" and "latitude"
            df = pd.Dataframe({"longitude":[-83.235572, -83.235154], "latitude":[42.348092, 42.348806]})
            gtd = GeoTaggedData(locations=df)
        '''

        self.images = None
        self.locations = locations
        self.units = units
        if locations is not None and units is None:
            self.construct_units()

        self.svis = self.photos = self.audios = {
            'loc_id': [],
            'id': [],
            'data': [],
            'path':[],
        }

        self.svi_metadata = None
        self.photo_metadata = None
        self.audio_metadata = None
        self.plot = None

    def construct_units(self):
        if isinstance(self.locations, list):
            if isinstance(self.locations[0], list):
                coor = {
                    'x': [],
                    'y': []
                }
                for location in self.locations:
                    coor['x'].append(location[0])
                    coor['y'].append(location[1])
                df = pd.DataFrame(coor)
                geometry = gpd.points_from_xy(df['x'], df['y'])
                id_df = pd.DataFrame({'loc_id':[i for i in range(len(df))]})
                self.units = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            else:
                print("coordinates should be stored in a nested list")
                return None
        elif isinstance(self.locations, dict):
            if 'longitude' in self.locations and 'latitude' in self.locations:
                geometry = gpd.points_from_xy(self.locations['longitude'], self.locations['latitude'])
                id_df = pd.DataFrame({'loc_id': [i for i in range(len(self.locations['longitude']))]})
            else:
                print("the dictionary of coordinates should be keyed by longitude and latitude")
                return None
        elif isinstance(self.locations, pd.DataFrame):
            if 'longitude' in self.locations.columns and 'latitude' in self.locations.columns:
                geometry = gpd.points_from_xy(self.locations['longitude'], self.locations['latitude'])
                id_df = pd.DataFrame({'loc_id': [i for i in range(len(self.locations['longitude']))]})
            else:
                print("the dataframe of coordinates should include columns of longitude and latitude")
                return None
        else:
            return None
        self.units = gpd.GeoDataFrame(id_df, geometry=geometry, crs="EPSG:4326")
        return None

    def getBuildings(self,
                     bbox: list | tuple = None,
                     source: str = 'osm',
                     min_area: float | int = 0,
                     max_area: float | int = None,
                     random_sample: int = None)-> None:
        '''
            Extract buildings from OpenStreetMap using the bbox.

            Args:
                bbox (list or tuple): The bounding box.
                source (str): The source of the buildings. ['osm', 'microsoft']
                min_area (float or int): The minimum area.
                max_area (float or int): The maximum area.
                random_sample (int): The number of random samples.
        '''

        if source not in ['osm', 'microsoft']:
            raise Exception(f'{source} is not supported')

        if source == 'osm':
            buildings = getOSMbuildings(bbox, min_area, max_area)
        elif source == 'microsoft':
            buildings = getGlobalMLBuilding(bbox, min_area, max_area)
        if buildings is None or buildings.empty:
            if source == 'osm':
                print("No buildings found in the bounding box. Please check https://overpass-turbo.eu/ for areas with buildings.")
                return None
            if source == 'microsoft':
                print("No buildings found in the bounding box. Please check https://github.com/microsoft/GlobalMLBuildingFootprints for areas with buildings.")
                return None
        if random_sample is not None:
            buildings = buildings.sample(random_sample)
        self.units = buildings.to_crs(4326)
        print(f"{len(buildings)} buildings found in the bounding box.")
        return None

    def get_svi_from_locations(self,
                               id_column:str=None,
                               distance:int = 50,
                               key: str = None,
                               pano: bool = True, reoriented: bool = True,
                               multi_num: int = 1, interval: int = 1,
                               fov: int = 80, heading: int = None, pitch: int = 5,
                               height: int = 500, width: int = 700,
                               year: list | tuple = None, season: str = None, time_of_day: str = 'day',
                               silent: bool = True):
        """
            get_svi_from_locations

            Retrieve the closest street view image(s) near each coordinate using the Mapillary API.
            The street view image will be reoriented to look at the coordinate when `reoriented = True`.

            Args:
                id_column (str, optional): The name of column that has unique identifier (or something similar) for each location.
                distance (int): The max distance in meters between the centroid and the street view
                key (str): Mapillary API access token.
                pano (bool): Whether to search for pano street view images only. (Default is True)
                reoriented (bool): Whether to reorient and crop street view images. (Default is True)
                multi_num (int): The number of multiple SVIs (Default is 1).
                interval (int): The interval in meters between each SVI (Default is 1).
                fov (int): Field of view in degrees for the perspective image. (Defaults is 80).
                heading (int): Camera heading in degrees. If None, it will be computed based on the house orientation.
                pitch (int): Camera pitch angle. (Default is 10).
                height (int): Height in pixels of the returned image. (Default is 480).
                width (int): Width in pixels of the returned image. (Default is 640).
                year (list[str], optional): Year of data (start year, end year).
                season (str, optional): Season of data. One of ["spring","summer","fall","autumn","winter"]
                time_of_day (str, optional): Time of data. One of ["day","night"] (Default is 'day')
                silent (bool): If True, do not show error traceback (Default is True).
            """

        self.svis = {
            'loc_id': [],
            'id': [],
            'data': [],
            'path': [],
        }
        self.svi_metadata = None

        if id_column is None:
            id_column = 'loc_id'
            if id_column not in self.units.columns:
                self.units[id_column] = [i for i in range(len(self.units))]
        res_df = None
        skip_count = 0
        for index, row in tqdm(self.units.iterrows(), total=len(self.units)):
            loc_id = row[id_column]
            try:
                svis, output_df = getSV([row.geometry.centroid.x, row.geometry.centroid.y],
                                        loc_id,
                                        distance,
                                        key,
                                        pano,
                                        reoriented,
                                        multi_num,
                                        interval,
                                        fov, heading, pitch,
                                        height,
                                        width,
                                        year,
                                        season,
                                        time_of_day,
                                        silent = silent
                                        )
                if svis is None:
                    skip_count += 1
                    continue

                self.svis['data'] += svis
                self.svis['loc_id'] += output_df['loc_id'].tolist()
                self.svis['id'] += output_df['id'].tolist()

                if res_df is None:
                    res_df = output_df
                else:
                    res_df = pd.concat([res_df, output_df])
            except Exception as e:
                if not silent: print(f'skipping {[row.geometry.centroid.x, row.geometry.centroid.y]}: {e}')
                skip_count += 1
                continue
        self.svi_metadata = res_df
        if skip_count > 0:
            print(f'Collect data for {len(self.units) - skip_count} locations and skipped {skip_count} locations due to no data found.')
        return None

    def get_photo_from_location(self,
                                id_column:str=None,
                                distance: int = 50,
                                key: str = None,
                                query: str | list[str] = None,
                                tag: str | list[str] = None,
                                max_return: int = 1,
                                year: list | tuple = None,
                                season: str = None,
                                time_of_day: str = None,
                                exclude_personal_photo: bool = True,
                                exclude_from_location:int = None,
                                silent = True,
                                ):
        '''
            get_photo_from_location

            Retrieve geotagged photos from Flickr

            Args:
                id_column: (str, optional): The name of column that has unique identifier (or something similar) for each location.
                distance (int): Search radius in meters (converted to km; Flickr radius max is 32 km).
                key (str): Flickr API key. If None, reads env var FLICKR_API_KEY.
                query (str, optional): Query string to search for.
                tag (str | list[str]): Tag string or list of tags (comma-separated). Acts as a "limiting agent" for geo queries.
                max_return (int): Number of photos to return (after filters).
                year: [Y] or (Y,) or (Y1, Y2) inclusive. Filters by taken date range.
                season (str): One of {"spring","summer","fall","autumn","winter"} (post-filter by taken month).
                time_of_day (str): One of {"morning","afternoon","evening","night"} (post-filter by taken hour).
                exclude_personal_photo (bool): If True, exclude personal photo from locations. (Default is True)
                exclude_from_location (int, optional): Drop retrieved data with a distance from the given location.
                silent (bool): If True, do not show error traceback (Default is True).
        '''

        from .utils.pano2pers import read_url2img
        from importlib.resources import files, as_file

        self.photos = {
            'loc_id': [],
            'id': [],
            'data': [],
            'path': [],
        }
        self.photo_metadata = None

        if id_column is None:
            id_column = 'loc_id'
            if id_column not in self.units.columns:
                self.units[id_column] = [i for i in range(len(self.units))]
        res_df = None
        skip_count = 0
        for index, row in tqdm(self.units.iterrows(), total=len(self.units)):
            loc_id = row['loc_id']
            try:
                output_df = getPhoto([row.geometry.centroid.x, row.geometry.centroid.y],
                                     loc_id,
                                     distance,
                                     key,
                                     query,
                                     tag,
                                     max_return,
                                     year,
                                     season,
                                     time_of_day,
                                     exclude_from_location,
                                     output_df=True)
                if exclude_personal_photo:
                    model_res = files("urbanworm.models") / "face_detection_yunet_2023mar.onnx"
                    drop_list = []
                    for ind, r in output_df.iterrows():
                        with as_file(model_res) as model_path:
                            is_selfie = is_selfie_photo(model_path, r['url'])
                            if is_selfie:
                                drop_list += [ind]
                    if len(drop_list) > 0:
                        output_df.drop(drop_list, axis=0, inplace=True)
                        if len(output_df) == 0:
                            continue

                self.photos['loc_id'] += output_df['loc_id'].tolist()
                self.photos['data'] += output_df['url'].tolist()
                self.photos['id'] += output_df['id'].tolist()
                if res_df is None:
                    res_df = output_df
                else:
                    res_df = pd.concat([res_df, output_df])
            except Exception as e:
                if not silent: print(e)
                skip_count += 1
                continue
        self.photo_metadata = res_df
        if skip_count > 0:
            print(f'Collect data for {len(self.units) - skip_count} locations and skipped {skip_count} locations due to no data found.')
        return None

    def get_sound_from_location(self,
                                id_column: str = None,
                                distance: int = 50,
                                key: str = None,
                                query: str | list[str] = None,
                                tag: str | list[str] = None,
                                max_return: int = 1,
                                year: list | tuple = None,
                                season: str = None,
                                time_of_day: str = None,
                                duration: int = None,
                                exclude_from_location: int = None,
                                slice_duration: int = None,
                                slice_max_num: int = None,
                                silent: bool = True
                                ):

        '''
            get_sound_from_location

            Retrieve geotagged sound recordings from Freesound

            Args:
                id_column (str, optional): The name of column that has unique identifier (or something similar) for each location.
                distance (int): radius in meters (converted to km for Freesound geofilt).
                key (str): Freesound API key. If None, reads env var FREESOUND_API_KEY.
                query (str, optional): Query string to search for.
                tag (str | list[str]): tag string or list of tags (used as filters).
                max_return (int): number of sounds to return (after post-filters).
                year (int | list): [Y] or (Y,) or (Y1, Y2) inclusive (filters by upload date "created").
                season (str): one of {"spring","summer","fall","autumn","winter"} (post-filter by created month).
                time_of_day (str): one of {"morning","afternoon","evening","night"} (post-filter by created hour).
                duration (int | list[int] | tuple[int]): maximum duration in seconds (<= duration). If you want a range, pass a tuple/list (min,max).
                exclude_from_location (int, optional): Drop retrieved data with a distance from the given location.
                slice_duration (int, optional): Split the original sound signal into clips with the given duration.
                slice_max_num (int, optional): Maximum number of clips sliced from the original sound signal.
                silent (bool): If True, do not show error traceback (Default is True).
        '''

        self.audios = {
            'loc_id': [],
            'id': [],
            'data': [],
            'path': [],
        }
        self.audio_metadata = None

        if slice_duration is not None:
            self.audios['slice'] = []

        if id_column is None:
            id_column = 'loc_id'
            if id_column not in self.units.columns:
                self.units[id_column] = [i for i in range(len(self.units))]
        res_df = None
        skip_count = 0
        for index, row in tqdm(self.units.iterrows(), total=len(self.units)):
            loc_id = row['loc_id']
            try:
                output_df = getSound([row.geometry.centroid.x, row.geometry.centroid.y],
                                     loc_id,
                                     distance,
                                     key,
                                     query,
                                     tag,
                                     max_return,
                                     year,
                                     season,
                                     time_of_day,
                                     duration,
                                     exclude_from_location,
                                     slice_duration,
                                     slice_max_num,
                                     output_df = True)

                if slice_duration is not None:
                    slice_list = output_df['slice'].tolist()
                    loc_id_list = output_df['loc_id'].tolist()
                    data_list = output_df['preview-hq-mp3'].tolist()
                    id_list = output_df['id'].tolist()

                    slice_num = 1
                    if isinstance(slice_list[0][0], list):
                        slice_num = len(slice_list[0])
                        flattened_slice_list = [item for sublist in slice_list for item in sublist]
                    if slice_num > 1:
                        loc_id_list_ = []
                        data_list_ = []
                        id_list_ = []
                        for item in loc_id_list:
                            loc_id_list_.extend([item] * slice_num)
                        for item in data_list:
                            data_list_.extend([item] * slice_num)
                        for item in id_list:
                            id_list_.extend([item] * slice_num)
                        self.audios['loc_id'] += loc_id_list_
                        self.audios['data'] += data_list_
                        self.audios['id'] += id_list_
                        self.audios['slice'] += flattened_slice_list
                    else:
                        self.audios['loc_id'] += loc_id_list
                        self.audios['data'] += data_list
                        self.audios['id'] += id_list
                        self.audios['slice'] += flattened_slice_list
                else:
                    self.audios['loc_id'] += output_df['loc_id'].tolist()
                    self.audios['data'] += output_df['preview-hq-mp3'].tolist()
                    self.audios['id'] += output_df['id'].tolist()

                if res_df is None:
                    res_df = output_df
                else:
                    res_df = pd.concat([res_df, output_df])
            except Exception as e:
                if not silent: print(e)
                skip_count += 1
                continue
        self.audio_metadata = res_df
        if skip_count > 0:
            print(f'Collect data for {len(self.units) - skip_count} locations and skipped {skip_count} locations due to no data found.')
        return None

    def download_to_dir(self, data:str = None, to_dir:str = None, prefix: str = None)-> None:
        '''
            download_to_dir

            Download retrieved data to a directory.

            Args:
                data (str): Type of data to download: ['svi', 'audio', 'photo'].
                to_dir (str): the directory to save the downloaded data.
                prefix (str, optional):  The prefix to add to the output filename.
        '''
        if data not in ['svi', 'audio', 'photo']:
            raise ValueError('Invalid data type provided. It has to be one of ["svi", "audio", "photo"].')
        if to_dir is not None:
            if not os.path.exists(to_dir):
                print("The directory doesn't exist.")
                print("The directory is created now.")
                out_dir = Path(to_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
        else:
            print("You need to specify a directory to download.")
            return None
        if data == 'svi':
            if len(self.svis['id']) == 0:
                return None
            self.svis['path'] = []
            for i in tqdm(range(len(self.svis['data'])), total=len(self.svis['data'])):
                loc_id = self.svis['loc_id'][i]
                img_id = self.svis['id'][i]
                path = f'{to_dir}/{prefix}_{loc_id}' if prefix is not None else f'./{to_dir}/{loc_id}'
                p = path + f'_{img_id}.png'
                try:
                    if is_base64(self.svis['data'][i]):
                        save_base64(self.svis['data'][i], p)
                    else:
                        download_image_requests(self.svis['data'][i], p)
                except:
                    self.svis['path'] += [" "]
                    continue
                self.svis['path'] += [p]
        elif data == 'audio':
            if len(self.audios['id']) == 0:
                return None
            self.audios['path'] = []
            if 'slice' in self.audios:
                for i in tqdm(range(len(self.audios['data'])), total=len(self.audios['data'])):
                    loc_id = self.audios['loc_id'][i]
                    audio_id = self.audios['id'][i]
                    slices = self.audios['slice'][i]
                    path = f'{to_dir}/{prefix}_{loc_id}' if prefix is not None else f'./{to_dir}/{loc_id}'
                    start = slices[0]
                    end = slices[1]
                    p = path + f'_{audio_id}_clip_{start}_{end}.mp3'
                    try:
                        clip(self.audios['data'][i], start, end, p)
                    except:
                        continue
                    self.audios['path'] += [p]
            else:
                for i in tqdm(range(len(self.audios['data'])), total=len(self.audios['data'])):
                    loc_id = self.audios['loc_id'][i]
                    audio_id = self.audios['id'][i]
                    path = f'{to_dir}/{prefix}_{loc_id}' if prefix is not None else f'./{to_dir}/{loc_id}'
                    p = path + f'_{audio_id}.mp3'
                    try:
                        download_freesound_preview(self.audios['data'][i], p)
                    except:
                        self.audios['path'] += [" "]
                        continue
                    self.audios['path'] += [p]
        elif data == 'photo':
            if len(self.photos['id']) == 0:
                return None
            self.photos['path'] = []
            for i in tqdm(range(len(self.photos['data'])), total=len(self.photos['data'])):
                loc_id = self.photos['loc_id'][i]
                photo_id = self.photos['id'][i]
                path = f'{to_dir}/{prefix}_{loc_id}' if prefix is not None else f'./{to_dir}/{loc_id}'
                p = path + f'_{photo_id}.png'
                try:
                    download_image_requests(self.photos['data'][i], p)
                except:
                    self.photos['path'] += [" "]
                self.photos['path'] += [p]
        return None

    def set_images(self, img_type: str):
        '''
            set_images

            Set retrieved street view images or Flickr photos as images dataset

            Args:
                img_type (str): 'photo' or 'svi'
        '''
        if img_type == 'svi':
            self.images = self.svis
        elif img_type == 'photo':
            self.images = self.photos
        return None

    def plot_data(self, data:str = None, export_gdf: bool = False) -> None:
        '''

        Args:
            data (str): Type of data to download: ['svi', 'audio', 'photo'].
            export_gdf (bool): Export gpd.GeoDataFrame.
        '''
        if data is None:
            return None

        if data == 'svi':
            temp = self.svi_metadata
            geometry = gpd.points_from_xy(temp['image_lon'], temp['image_lat'])
            temp['detail'] = temp.apply(
                lambda row: f'<a href="{row["url"]}">View image details</a>',
                axis=1
            )
            gdf = gpd.GeoDataFrame(temp, geometry=geometry, crs="EPSG:4326")
            popup = ["id", "captured_at", "detail"]
        elif data == 'photo':
            temp = self.photo_metadata
            geometry = gpd.points_from_xy(temp['longitude'], temp['latitude'])
            temp['detail'] = temp.apply(
                lambda row: f'<a href="{row["url"]}">View photo details</a>',
                axis=1
            )
            gdf = gpd.GeoDataFrame(temp, geometry=geometry, crs="EPSG:4326")
            popup = ["id", "datetaken", "detail"]
        elif data == 'audio':
            temp = self.audio_metadata
            geometry = gpd.points_from_xy(temp['longitude'], temp['latitude'])
            temp['detail'] = temp.apply(
                lambda row: f'<a href="{row["url"]}">Listen to the sound</a>',
                axis=1
            )
            gdf = gpd.GeoDataFrame(self.audio_metadata, geometry=geometry, crs="EPSG:4326")
            popup = ["id", "created_dt", "detail"]
        else:
            raise ValueError('Invalid data type provided. It has to be one of ["svi", "audio", "photo"].')

        self.plot = gdf.explore(
            popup=popup,
            color="red",
            marker_kwds=dict(radius=5, fill=True),
            tiles="CartoDB positron",
            name="map",
        )
        return gdf if export_gdf else self.plot


# Get street view images from Mapillary
def getSV(location: list|tuple,
          loc_id: int | str = None,
          distance:int = 50,
          key: str = None,
          pano: bool = False,
          reoriented: bool = False,
          multi_num: int = 1,
          interval: int = 1,
          fov: int = 80, heading: int = None, pitch: int = 5,
          height: int = 500, width: int = 700,
          year: list | tuple = None,
          season: str = None,
          time_of_day: str = None,
          output_df: bool = True,
          silent: bool = False) -> pd.DataFrame | list | None:
    """
        getSV

        Retrieve the closest street view image(s) near a coordinate using the Mapillary API.
        The street view image will be reoriented to look at the coordinate.

        Args:
            location: coordinates (longitude/x and latitude/y)
            loc_id (int|str, optional): The id of the location
            distance (int): The max distance in meters between the centroid and the street view
            key (str): Mapillary API access token.
            pano (bool): Whether to search for pano street view images only. (Default is True)
            reoriented (bool): Whether to reorient and crop street view images. (Default is True)
            multi_num (int): The number of multiple SVIs (Default is 1).
            interval (int): The interval in meters between each SVI (Default is 1).
            fov (int): Field of view in degrees for the perspective image. Defaults to 80.
            heading (int): Camera heading in degrees. If None, it will be computed based on the location orientation.
            pitch (int): Camera pitch angle. (Default is 10).
            height (int): Height in pixels of the returned image. (Default is 480).
            width (int): Width in pixels of the returned image. (Default is 640).
            year (list[str], optional): Year of data (start year, end year).
            season (str, optional): Season of data.
            time_of_day (str, optional): Time of data.
            output_df (bool, optional): Whether to return a dataframe containing only the closest. (Default is True)
            silent (bool, optional): Whether to silence output (Default is False).

        Returns:
            list[str]: A list of images in base64 format
            DataFrame: A dataframe containing metadata about the closest street view images.
    """

    bbox = projection(location, r=distance)
    url = f"https://graph.mapillary.com/images?access_token={key}&fields=id,compass_angle,thumb_original_url,captured_at,geometry,sequence&bbox={bbox}"
    # 2048 -> original to get higher resolution
    if pano:
        url += "&is_pano=true"
    if pano == False and reoriented == True:
        reoriented = False

    svis = []
    svi_df = {
        "id": [],
        "sequence": [],
        "captured_at": [],
        "compass_angle": [],
        "image_lon": [],
        "image_lat": [],
        'url': [],
        'loc_id': []
    }
    if loc_id is None:
        del svi_df['loc_id']

    try:
        response = retry_request(url)
        if response is None:
            if not silent: print(f'skip location: {location} due to no data found')
            if output_df:
                return None, None
            return None
        response = response.json()
        # find the closest image
        response = closest(location, response, multi_num, interval, year, season, time_of_day, key)
        if response is None:
            if not silent: print(f'skip location: {location} due to no data found')
            if output_df:
                return None, None
            return None

        for index, row in response.iterrows():
            # Extract Image ID, Compass Angle, image url, and coordinates
            img_heading = float(row['compass_angle'])
            img_url = row['thumb_original_url']
            image_lon, image_lat = row['coordinates']
            if heading is None:
                # calculate bearing to the house
                bearing_to_house = calculate_bearing(image_lat, image_lon, location[1], location[0])
                relative_heading = (bearing_to_house - img_heading) % 360
            else:
                relative_heading = heading
            # reframe image
            if reoriented:
                svi = Equirectangular(img_url=img_url)
                sv = svi.GetPerspective(fov, relative_heading, pitch, height, width, 128)
                svis.append(sv)
            else:
                svis.append(img_url)

            if output_df:
                svi_df['id'].append(row['id'])
                svi_df['sequence'].append(row['sequence'])
                svi_df['captured_at'].append(f'{row["year"]}-{row["month"]}-{row["day"]}-{row["hour"]}')
                svi_df['image_lon'].append(image_lon)
                svi_df['image_lat'].append(image_lat)
                svi_df['compass_angle'].append(img_heading)
                svi_df['url'].append(img_url)
                if 'loc_id' in svi_df:
                    svi_df['loc_id'].append(loc_id)
        if output_df:
            return svis, pd.DataFrame(svi_df)
        else:
            return svis
    except Exception as e:
        if not silent: print(f'skip location: {location} due to {e}')
        if output_df:
            return None, None
        return None


from .utils.utils import season_months,tod_hours,year_range
def getPhoto(
        location: list | tuple,
        loc_id: int | str = None,
        distance: int = 50,
        key: str = None,
        query: str | list[str] = None,
        tag: str | list[str] = None,
        max_return: int = 1,
        year: list | tuple = None,
        season: str = None,
        time_of_day: str = None,
        exclude_from_location:int = None,
        output_df: bool = True
):
    """
        getPhoto

        Fetch public Flickr photos with geotags near a location (or within a Flickr place).

        Args:
            location (list|tuple): (lon, lat) required. Coordinates of location (longitude, latitude) for searching for geotagged photos
            loc_id (int | str): The id of the location.
            distance (int): Search radius in meters (converted to km; Flickr radius max is 32 km).
            key (str): Flickr API key. If None, reads env var FLICKR_API_KEY.
            query (str | list[str]): Query parameters to pass to Flickr API (free text search).
            tag: Tag string or list of tags (comma-separated). Acts as a "limiting agent" for geo queries.
            max_return: Number of photos to return (after filters).
            year (str | tuple): [Y] or (Y,) or (Y1, Y2) inclusive. Filters by taken date range.
            season (str): One of {"spring","summer","fall","autumn","winter"} (post-filter by taken month).
            time_of_day (str): One of {"morning","afternoon","evening","night"} (post-filter by taken hour).
            exclude_from_location (int, optional): drop retrieved photos within a distance (in meter) from the given location. (Default is None)
            output_df (bool): If True, return a pandas.DataFrame; otherwise return dict (if max_return==1)
                       or list[dict].

        Returns:
            dict | list[dict] | pandas.DataFrame
    """

    import os
    import requests
    from datetime import datetime, timedelta, timezone

    if exclude_from_location is not None:
        drop_area = projection(location, r=distance)

    # -------------------------
    # Validate inputs
    # -------------------------
    if max_return is None or int(max_return) < 1:
        raise ValueError("max_return must be >= 1.")
    max_return = int(max_return)

    api_key = key or os.getenv("FLICKR_API_KEY")
    if not api_key:
        raise ValueError("Missing Flickr API key. Pass key=... or set env var FLICKR_API_KEY.")

    lon, lat = location
    months = season_months(season)
    hours = tod_hours(time_of_day)
    y_range = year_range(year)

    # Radius in km (Flickr max 32km) :contentReference[oaicite:3]{index=3}
    radius_km = max(float(distance) / 1000.0, 0.01)
    radius_km = min(radius_km, 32.0)

    # Geo queries need a "limiting agent"; tags or min/max dates qualify. :contentReference[oaicite:4]{index=4}
    # If user provided none, default to last 365 days so results aren’t silently limited to ~12 hours.
    now_utc = datetime.now(timezone.utc)
    default_min_upload_date = int((now_utc - timedelta(days=365)).timestamp())

    # -------------------------
    # Build Flickr request
    # -------------------------
    endpoint = "https://api.flickr.com/services/rest/"

    extras = ",".join(
        [
            "description",
            "license",
            "date_upload",
            "date_taken",
            "owner_name",
            "geo",
            "tags",
            "views",
            "media",
            "url_sq",
            "url_t",
            "url_s",
            "url_q",
            "url_m",
            "url_n",
            "url_z",
            "url_c",
            "url_l",
            "url_o",
        ]
    )

    params = {
        "method": "flickr.photos.search",
        "api_key": api_key,
        "format": "json",
        "nojsoncallback": 1,
        "extras": extras,
        "safe_search": 1, # safe only for un-authed calls
        "media": "photos",
        "has_geo": 1,
        "content_types": 0, # photos
        "sort": "relevance",
        "lat": lat,
        "lon": lon,
        "radius": radius_km,
        "radius_units": "km"
    }

    if query:
        q = query_string(query)
        if q:
            params["text"] = q

    # tags
    if tag:
        if isinstance(tag, (list, tuple)):
            tags = ",".join([str(t).strip() for t in tag if str(t).strip()])
            params["tags"] = tags
            params["tag_mode"] = "all"
        else:
            params["tags"] = str(tag).strip()

    # date range (taken) if specified
    if y_range is not None:
        params["min_taken_date"], params["max_taken_date"] = y_range
    else:
        # If no explicit limiting agent, set min_upload_date (acts as limiting agent for geo queries). :contentReference[oaicite:7]{index=7}
        if not tag and season is None and time_of_day is None:
            params["min_upload_date"] = default_min_upload_date

    # -------------------------
    # Fetch + post-filter
    # -------------------------
    session = requests.Session()

    # Geo/bbox queries only return up to 250/page. :contentReference[oaicite:8]{index=8}
    per_page = min(250, max(50, max_return * 20))
    params["per_page"] = per_page

    results = []
    seen = set()

    max_pages = 150
    for page in range(1, max_pages + 1):
        params["page"] = page
        r = session.get(endpoint, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if data.get("stat") != "ok":
            msg = data.get("message") or data.get("error") or str(data)
            raise RuntimeError(f"Flickr API error: {msg}")

        photos = (data.get("photos") or {}).get("photo") or []
        if not photos:
            break

        for p in photos:
            if exclude_from_location is not None:
                if is_coordinate_in_bbox(p["longitude"], p["latitude"], drop_area):
                    continue
            pid = p.get("id")
            if not pid or pid in seen:
                continue
            seen.add(pid)

            taken_dt = parse_taken(p)
            if months and taken_dt and taken_dt.month not in months:
                continue
            if hours and taken_dt and taken_dt.hour not in hours:
                continue

            s_lat = float(p["latitude"]) if "latitude" in p and p["latitude"] not in (None, "") else None
            s_lon = float(p["longitude"]) if "longitude" in p and p["longitude"] not in (None, "") else None

            url = best_url(p)
            out = {
                "loc_id": '',
                "id": pid,
                "title": p.get("title"),
                "owner": p.get("owner"),
                # "ownername": p.get("ownername"),
                "datetaken": p.get("datetaken") or p.get("date_taken"),
                "latitude": s_lat,
                "longitude": s_lon,
                # "accuracy": int(p["accuracy"]) if "accuracy" in p and str(p["accuracy"]).isdigit() else None,
                "distance_m": haversine_m(lat, lon, s_lat, s_lon) if (s_lat is not None and s_lon is not None) else None,
                "tags": p.get("tags"),
                "description": p.get("description"),
                "views": int(p["views"]) if "views" in p and str(p["views"]).isdigit() else None,
                "license": p.get("license"),
                "url": url,
                # "page_url": f"https://www.flickr.com/photos/{p.get('owner')}/{pid}",
            }

            if loc_id is not None:
                out["loc_id"] = loc_id
            else:
                del out["loc_id"]

            results.append(out)

            # if len(results) >= max_return:
            #     break

        if len(results) >= max_return:
            break

    if output_df:
        import pandas as pd
        df = pd.DataFrame(results)
        df = df.sort_values(by='distance_m', ascending=True)
        return df.head(max_return)

    if max_return == 1:
        return results[0] if results else None
    return results


def getSound(
        location: list | tuple,
        loc_id: int | str = None,
        distance: int = 50,
        key: str = None,
        query: str | list[str] | None = None,
        tag: str | list[str] = None,
        max_return: int = 1,
        year: list | tuple = None,
        season: str = None,
        time_of_day: str = None,
        duration: int = 300,
        exclude_from_location:int = None,
        slice_duration:int = None,
        slice_max_num:int = None,
        output_df: bool = True,
) -> pd.DataFrame:

    """
        getSound

        Fetch geotagged Freesound audio near a point, using Freesound API v2 search + geospatial filter.

        Notes:
        - Uses token authentication (API key) via Authorization header.
        - Returns preview URLs (mp3/ogg). Downloading original audio requires OAuth2.

        Args:
            location: (lon, lat) required.
            loc_id (int | str, optional): .
            distance (int): radius in meters (converted to km for Freesound geofilt).
            key (str): Freesound API key. If None, reads env var FREESOUND_API_KEY.
            query (str, optional): Freesound API query (e.g., 'traffic', '"bird song" -crow').
            tag: tag string or list of tags (used as filters).
            max_return: number of sounds to return (after post-filters).
            year: [Y] or (Y,) or (Y1, Y2) inclusive (filters by upload date "created").
            season (str): one of {"spring","summer","fall","autumn","winter"} (post-filter by created month).
            time_of_day (str): one of {"morning","afternoon","evening","night"} (post-filter by created hour).
            duration (int): maximum duration in seconds (<= duration). If you want a range, pass a tuple/list (min,max). (Default is 300)
            exclude_from_location (int, optional): Drop retrieved photos within a distance (in meter) from the given location.
            slice_duration (int, optional): Split the original sound signal into clips with the given duration.
            slice_max_num (int, optional): Maximum number of clips sliced from the original sound signal.
            output_df (bool): if True, return a pandas.DataFrame.

        Returns:
            dict | list[dict] | pandas.DataFrame
    """
    import os
    import requests
    from datetime import datetime

    if exclude_from_location is not None:
        drop_area = projection(location, r=distance)

    # -------------------------
    # Helpers
    # -------------------------
    def _parse_created(s):
        if not s:
            return None
        # Examples look like "2014-04-16T20:07:11.145" (no timezone).
        # Try a couple variants.
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
        # last resort: fromisoformat (py3.11+ handles many cases)
        try:
            return datetime.fromisoformat(s.replace("Z", ""))
        except Exception:
            return None

    def _year_range(y, with_z: bool=False):
        if y is None:
            return None
        if not isinstance(y, (list, tuple)) or len(y) == 0:
            raise ValueError("year must be a list/tuple like [2020] or (2020, 2022).")
        if len(y) == 1:
            y1 = y2 = int(y[0])
        else:
            y1, y2 = int(y[0]), int(y[1])
            if y2 < y1:
                y1, y2 = y2, y1

        # Use standard Solr-like ISO range; we will retry without 'Z' if needed.
        z = "Z" if with_z else ""
        start = f"{y1:04d}-01-01T00:00:00{z}"
        end = f"{y2:04d}-12-31T23:59:59{z}"
        return start, end

    # -------------------------
    # Validate inputs
    # -------------------------
    if max_return is None or int(max_return) < 1:
        raise ValueError("max_return must be >= 1.")
    max_return = int(max_return)

    api_key = key or os.getenv("FREESOUND_API_KEY")
    if not api_key:
        raise ValueError("Missing Freesound API key. Pass key=... or set env var FREESOUND_API_KEY.")

    lon, lat = location

    # meters -> km for geofilt d=<km>
    radius_km = max(float(distance) / 1000.0, 0.01)

    months = season_months(season)
    hours = tod_hours(time_of_day)

    # duration: allow int (max seconds) or tuple/list (min,max)
    dur_filter = None
    if duration is not None:
        if isinstance(duration, (list, tuple)) and len(duration) == 2:
            dmin = float(duration[0])
            dmax = float(duration[1])
            if dmax < dmin:
                dmin, dmax = dmax, dmin
            dur_filter = f"duration:[{dmin} TO {dmax}]"
        else:
            dmax = float(duration)
            dur_filter = f"duration:[0 TO {dmax}]"

    # -------------------------
    # Build request
    # -------------------------
    endpoint = "https://freesound.org/apiv2/search/"
    headers = {"Authorization": f"Token {api_key}"}  # token auth

    # Request useful fields, including previews (mp3/ogg URLs) and geotag.
    fields = ",".join(
        [
            "id",
            "name",
            "username",
            "license",
            "created",
            "duration",
            "geotag",
            "tags",
            "previews",
            "url",
            "num_downloads",
            "avg_rating",
            "description"
        ]
    )

    # Base filter parts
    filter_parts = []
    filter_parts.append("is_geotagged:1")
    filter_parts.append(f"{{!geofilt sfield=geotag pt={lat},{lon} d={radius_km}}}")

    # tag filters
    if tag:
        if isinstance(tag, (list, tuple)):
            for t in tag:
                t = str(t).strip()
                if t:
                    filter_parts.append(f"tag:{t}")
        else:
            t = str(tag).strip()
            if t:
                filter_parts.append(f"tag:{t}")

    if dur_filter:
        filter_parts.append(dur_filter)

    # year filter (created range): try with Z, retry without if API complains
    created_range_z = _year_range(year, with_z=True)
    created_range_noz = _year_range(year, with_z=False)

    qstr = query_string(query)

    def _do_request(created_range):
        fp = list(filter_parts)
        if created_range is not None:
            start, end = created_range
            fp.append(f"created:[{start} TO {end}]")
        params = {
            "query": qstr,  # empty query allowed
            "filter": " ".join(fp),
            "fields": fields,
            "page": 1,
            "page_size": min(150, max(50, max_return * 25)),
            "sort": "score",
        }
        return params

    session = requests.Session()

    # -------------------------
    # Fetch + post-filter
    # -------------------------
    results = []
    seen = set()
    max_pages = 150

    # First attempt (with Z)
    params = _do_request(created_range_z)

    for attempt in (1, 2):
        try:
            for page in range(1, max_pages + 1):
                params["page"] = page
                r = session.get(endpoint, params=params, headers=headers, timeout=999)

                if r.status_code == 400 and attempt == 1 and year is not None:
                    # likely date format issue; retry without Z
                    raise ValueError("Date filter rejected; retrying without 'Z'.")
                if r.status_code == 404:
                    break
                r.raise_for_status()
                data = r.json()

                page_results = data.get("results") or []
                if not page_results:
                    break

                for s in page_results:
                    sid = s.get("id")
                    if sid is None or sid in seen:
                        continue
                    seen.add(sid)

                    created_dt = _parse_created(s.get("created"))
                    if months and created_dt and created_dt.month not in months:
                        continue
                    if hours and created_dt and created_dt.hour not in hours:
                        continue

                    # Parse geotag "lat lon"
                    s_lat = s_lon = None
                    if s.get("geotag"):
                        parts = str(s["geotag"]).split()
                        if len(parts) == 2:
                            try:
                                s_lat = float(parts[0])
                                s_lon = float(parts[1])
                                if exclude_from_location is not None:
                                    if is_coordinate_in_bbox(s_lon, s_lat, drop_area):
                                        continue
                            except Exception:
                                pass

                    out = {
                        "loc_id": '',
                        "id": sid,
                        "name": s.get("name"),
                        "username": s.get("username"),
                        "license": s.get("license"),
                        "created": s.get("created"),
                        "duration": s.get("duration"),
                        "tags": s.get("tags"),
                        "geotag": s.get("geotag"),
                        "latitude": s_lat,
                        "longitude": s_lon,
                        "distance_m": haversine_m(lat, lon, s_lat, s_lon) if (s_lat is not None and s_lon is not None) else None,
                        "previews": s.get("previews"),
                        "url": s.get("url"),
                        "page_url": f"https://freesound.org/people/{s.get('username')}/sounds/{sid}/" if s.get("username") and sid else None,
                        "description": s.get("description"),
                        "num_downloads": s.get("num_downloads"),
                        "avg_rating": s.get("avg_rating"),
                        "slice": []
                    }
                    if loc_id is not None:
                        out["loc_id"] = loc_id
                    else:
                        del out["loc_id"]
                    if slice_duration is None:
                        del out["slice"]
                    else:
                        out["slice"] = sliced_duration(int(out["duration"]), slice_duration, slice_max_num)
                    results.append(out)
                    # if len(results) >= max_return:
                    #     break
                if not data.get("next"):
                    break
                if len(results) >= max_return:
                    break

            break  # success, don’t do second attempt
        except ValueError:
            # Retry without Z in created range
            if attempt == 1 and year is not None:
                params = _do_request(created_range_noz)
                continue
            raise

    # -------------------------
    # Return shape
    # -------------------------
    if output_df:
        import pandas as pd
        df = pd.DataFrame(results)
        df = df.sort_values(by='distance_m', ascending=True)
        previews_df = df['previews'].apply(pd.Series)
        previews_df.columns = [f'{col}' for col in previews_df.columns]
        df = pd.concat([df.drop('previews', axis=1), previews_df], axis=1)
        return df.head(max_return)

    if max_return == 1:
        return results[0] if results else None
    return results
