from __future__ import annotations

import logging
import os

import pandas as pd
from geopandas import GeoDataFrame
from tqdm.auto import tqdm

from .utils.building import *  # noqa: F401,F403  (also re-exports gpd, Path, etc.)
from .utils.pano2pers import Equirectangular
from .utils.utils import (
    calculate_bearing,
    closest,
    projection,
    retry_request,
)

logger = logging.getLogger("urbanworm")

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

        # NOTE: each must be its own dict literal — chained assignment would
        # alias all three names to the SAME underlying dict object.
        def _empty_payload():
            return {'loc_id': [], 'id': [], 'data': [], 'path': []}

        self.svis = _empty_payload()
        self.photos = _empty_payload()
        self.audios = _empty_payload()

        self.svi_metadata = None
        self.photo_metadata = None
        self.audio_metadata = None
        self.plot = None

    def construct_units(self):
        if isinstance(self.locations, list):
            if not self.locations or not isinstance(self.locations[0], (list, tuple)):
                raise ValueError(
                    "locations as a list must be a nested list of "
                    "[longitude, latitude] pairs."
                )
            xs = [loc[0] for loc in self.locations]
            ys = [loc[1] for loc in self.locations]
            geometry = gpd.points_from_xy(xs, ys)
            id_df = pd.DataFrame({'loc_id': list(range(len(self.locations)))})
        elif isinstance(self.locations, dict):
            if 'longitude' in self.locations and 'latitude' in self.locations:
                geometry = gpd.points_from_xy(self.locations['longitude'], self.locations['latitude'])
                id_df = pd.DataFrame({'loc_id': list(range(len(self.locations['longitude'])))})
            else:
                raise ValueError(
                    "locations as a dict must be keyed by 'longitude' and 'latitude'."
                )
        elif isinstance(self.locations, pd.DataFrame):
            if 'longitude' in self.locations.columns and 'latitude' in self.locations.columns:
                geometry = gpd.points_from_xy(self.locations['longitude'], self.locations['latitude'])
                id_df = pd.DataFrame({'loc_id': list(range(len(self.locations['longitude'])))})
            else:
                raise ValueError(
                    "locations as a DataFrame must include columns 'longitude' and 'latitude'."
                )
        else:
            raise TypeError(
                f"Unsupported locations type: {type(self.locations).__name__}. "
                "Use list[list], dict, or pandas.DataFrame."
            )
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
            raise ValueError(f'Unsupported building source {source!r}; '
                             f'choose from "osm" or "microsoft".')

        if source == 'osm':
            buildings = getOSMbuildings(bbox, min_area, max_area)
        else:  # 'microsoft'
            buildings = getGlobalMLBuilding(bbox, min_area, max_area)
        if buildings is None or buildings.empty:
            if source == 'osm':
                logger.warning(
                    "No buildings found in the bounding box. "
                    "Check https://overpass-turbo.eu/ for areas with buildings."
                )
            else:
                logger.warning(
                    "No buildings found in the bounding box. "
                    "Check https://github.com/microsoft/GlobalMLBuildingFootprints "
                    "for areas with buildings."
                )
            return None
        if random_sample is not None:
            buildings = buildings.sample(random_sample)
        self.units = buildings.to_crs(4326)
        logger.info("%d buildings found in the bounding box.", len(buildings))
        return None

    def get_svi_from_locations(self,
                               id_column:str=None,
                               distance:int = 50,
                               key: str = None,
                               pano: bool = True, reoriented: bool = True,
                               multi_num: int = 1, interval: int = 1,
                               fov: int | float | str = 80, heading: int = None, pitch: int = 5,
                               height: int = 500, width: int = 700,
                               year: list | tuple = None, season: str = None, time_of_day: str = 'day',
                               fov_margin: float = 0.10,
                               fov_min: float = 30.0,
                               fov_max: float = 120.0,
                               building_height: float = 9.0,
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
                fov (int | float | str): Field of view in degrees (default 80). Pass
                    ``'auto'`` (with ``reoriented=True``) to size the FOV per image
                    so the building footprint at each location is just framed.
                    The polygon used is each unit's ``row.geometry`` from
                    ``self.units`` — i.e. the building footprint loaded by
                    ``getBuildings()``. Falls back to a distance-based heuristic
                    if a unit's geometry is a point.
                heading (int): Camera heading in degrees. If None, it will be computed based on the house orientation.
                pitch (int): Camera pitch angle. (Default is 10).
                height (int): Height in pixels of the returned image. (Default is 480).
                width (int): Width in pixels of the returned image. (Default is 640).
                year (list[str], optional): Year of data (start year, end year).
                season (str, optional): Season of data. One of ["spring","summer","fall","autumn","winter"]
                time_of_day (str, optional): Time of data. One of ["day","night"] (Default is 'day')
                fov_margin (float): When ``fov='auto'``, fractional padding added to the
                    auto-computed FOV (0.10 = +10%). Default 0.10.
                fov_min (float): Lower clamp for ``fov='auto'`` (degrees). Default 30°.
                fov_max (float): Upper clamp for ``fov='auto'`` (degrees). Default 120°.
                building_height (float): Assumed building height in meters used by
                    ``fov='auto'`` (default 9 m, ~3 stories). The auto path
                    returns the wider of the horizontal extent (footprint) and
                    the vertical extent (height projected through the image's
                    aspect ratio) so a tall building's roof isn't cropped. Set
                    to 0 to skip the height term.
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
        # Resolve API key once with env var fallback
        resolved_key = key or os.getenv("MAPILLARY_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Missing Mapillary access token. Pass key=... or set env var MAPILLARY_API_KEY."
            )
        # Accumulate per-location frames and concat once for O(n) instead of O(n^2)
        frames: list[pd.DataFrame] = []
        skip_count = 0
        for _index, row in tqdm(self.units.iterrows(), total=len(self.units)):
            loc_id = row[id_column]
            try:
                # Pass the unit's polygon to enable fov='auto' framing.
                # Points (no `.exterior`) become None, so getSV will fall
                # back to its distance-based heuristic.
                target_poly = getattr(row.geometry, "exterior", None)
                target_poly = row.geometry if target_poly is not None else None
                svis, output_df = getSV(
                    [row.geometry.centroid.x, row.geometry.centroid.y],
                    loc_id=loc_id,
                    distance=distance,
                    key=resolved_key,
                    pano=pano,
                    reoriented=reoriented,
                    multi_num=multi_num,
                    interval=interval,
                    fov=fov, heading=heading, pitch=pitch,
                    height=height, width=width,
                    year=year, season=season, time_of_day=time_of_day,
                    target_polygon=target_poly,
                    fov_margin=fov_margin, fov_min=fov_min, fov_max=fov_max,
                    building_height=building_height,
                    silent=silent,
                )
                if svis is None:
                    skip_count += 1
                    continue

                self.svis['data'] += svis
                self.svis['loc_id'] += output_df['loc_id'].tolist()
                self.svis['id'] += output_df['id'].tolist()

                frames.append(output_df)
            except Exception as e:
                if not silent:
                    logger.warning(
                        'skipping %s: %s',
                        [row.geometry.centroid.x, row.geometry.centroid.y], e,
                    )
                skip_count += 1
                continue
        self.svi_metadata = pd.concat(frames, ignore_index=True) if frames else None
        if skip_count > 0:
            logger.info(
                'Collected data for %d locations; skipped %d (no data found).',
                len(self.units) - skip_count, skip_count,
            )
        return None

    def get_photo_from_location(self,
                                id_column:str=None,
                                distance: int = 50,
                                key: str = None,
                                query: str | list[str] = None,
                                geo_context: int = None,
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
                geo_context (int, optional): Specify whether a geotagged photo was taken indoors or outdoors. 0: Not defined; 1: Indoors; 2: Outdoors. (Default is None)
                tag (str | list[str]): Tag string or list of tags (comma-separated). Acts as a "limiting agent" for geo queries.
                max_return (int): Number of photos to return (after filters).
                year: [Y] or (Y,) or (Y1, Y2) inclusive. Filters by taken date range.
                season (str): One of {"spring","summer","fall","autumn","winter"} (post-filter by taken month).
                time_of_day (str): One of {"morning","afternoon","evening","night"} (post-filter by taken hour).
                exclude_personal_photo (bool): If True, exclude personal photo from locations. (Default is True)
                exclude_from_location (int, optional): Drop retrieved data with a distance from the given location.
                silent (bool): If True, do not show error traceback (Default is True).
        '''

        from importlib.resources import as_file, files

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
                self.units[id_column] = list(range(len(self.units)))
        frames: list[pd.DataFrame] = []
        skip_count = 0
        for _index, row in tqdm(self.units.iterrows(), total=len(self.units)):
            loc_id = row[id_column]
            try:
                output_df = getPhoto([row.geometry.centroid.x, row.geometry.centroid.y],
                                     loc_id,
                                     distance,
                                     key,
                                     query,
                                     geo_context,
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
                frames.append(output_df)
            except Exception as e:
                if not silent:
                    logger.warning("photo fetch error: %s", e)
                skip_count += 1
                continue
        self.photo_metadata = pd.concat(frames, ignore_index=True) if frames else None
        if skip_count > 0:
            logger.info(
                'Collected data for %d locations; skipped %d (no data found).',
                len(self.units) - skip_count, skip_count,
            )
        return None

    def get_sound_from_location(self,
                                id_column: str = None,
                                distance: int = 50,
                                source: str = 'freesound',
                                key: str = None,
                                catalog: str | pd.DataFrame = None,
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
                                probe_durations: bool = True,
                                silent: bool = True
                                ):

        '''
            get_sound_from_location

            Retrieve geotagged sound recordings from Freesound (default) or
            from a Radio Aporee catalog you provide as a CSV / DataFrame.

            Args:
                id_column (str, optional): The name of column that has unique identifier (or something similar) for each location.
                distance (int): radius in meters (converted to km for Freesound geofilt).
                source (str): one of {"freesound", "aporee"} (Default is "freesound").
                key (str): Freesound API key. Required only when source="freesound".
                    If None, reads env var FREESOUND_API_KEY.
                catalog (str | pandas.DataFrame): Required only when source="aporee".
                    Path to a CSV or an in-memory DataFrame containing at minimum the columns
                    ``url``, ``latitude``, ``longitude``. Optional columns recognised:
                    ``id``/``identifier``, ``name``/``title``, ``description``, ``tags``,
                    ``created`` (ISO timestamp), ``duration_s``.
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
                probe_durations (bool): Aporee-only. When ``slice_duration`` is set
                    but the catalog has no ``duration_s`` column, probe each
                    selected URL once to learn its length. Set False to skip
                    slicing instead. Default True.
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
                self.units[id_column] = list(range(len(self.units)))
        frames: list[pd.DataFrame] = []
        skip_count = 0
        for _index, row in tqdm(self.units.iterrows(), total=len(self.units)):
            loc_id = row[id_column]
            try:
                output_df = getSound([row.geometry.centroid.x, row.geometry.centroid.y],
                                     loc_id=loc_id,
                                     distance=distance,
                                     source=source,
                                     key=key,
                                     catalog=catalog,
                                     query=query,
                                     tag=tag,
                                     max_return=max_return,
                                     year=year,
                                     season=season,
                                     time_of_day=time_of_day,
                                     duration=duration,
                                     exclude_from_location=exclude_from_location,
                                     slice_duration=slice_duration,
                                     slice_max_num=slice_max_num,
                                     probe_durations=probe_durations,
                                     output_df=True)

                # `slice` may be missing if the source couldn't compute it
                # (e.g. Aporee catalog with no duration_s and probe_durations
                # disabled). Fall back to the un-sliced path in that case.
                if slice_duration is not None and 'slice' in output_df.columns:
                    slice_list = output_df['slice'].tolist()
                    loc_id_list = output_df['loc_id'].tolist()
                    data_list = output_df['preview-hq-mp3'].tolist()
                    id_list = output_df['id'].tolist()

                    # `slice_list[i]` is always a list of [start_ms, end_ms] pairs
                    # (one per generated clip). Flatten and replicate metadata
                    # to match the per-clip cardinality.
                    flattened_slice_list = [
                        item for sublist in slice_list for item in sublist
                    ]
                    repeated_loc, repeated_data, repeated_id = [], [], []
                    for sublist, lid, d, sid in zip(
                            slice_list, loc_id_list, data_list, id_list, strict=False):
                        n = len(sublist)
                        repeated_loc.extend([lid] * n)
                        repeated_data.extend([d] * n)
                        repeated_id.extend([sid] * n)
                    self.audios['loc_id'] += repeated_loc
                    self.audios['data'] += repeated_data
                    self.audios['id'] += repeated_id
                    self.audios['slice'] += flattened_slice_list
                else:
                    self.audios['loc_id'] += output_df['loc_id'].tolist()
                    self.audios['data'] += output_df['preview-hq-mp3'].tolist()
                    self.audios['id'] += output_df['id'].tolist()

                frames.append(output_df)
            except Exception as e:
                if not silent:
                    logger.warning("sound fetch error: %s", e)
                skip_count += 1
                continue
        self.audio_metadata = pd.concat(frames, ignore_index=True) if frames else None
        if skip_count > 0:
            logger.info(
                'Collected data for %d locations; skipped %d (no data found).',
                len(self.units) - skip_count, skip_count,
            )
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
        if to_dir is None:
            raise ValueError("to_dir must be provided.")
        if not os.path.exists(to_dir):
            logger.info("Directory %s does not exist; creating.", to_dir)
            Path(to_dir).mkdir(parents=True, exist_ok=True)
        if data == 'svi':
            if len(self.svis['id']) == 0:
                return None
            self.svis['path'] = []
            for i in tqdm(range(len(self.svis['data'])), total=len(self.svis['data'])):
                loc_id = self.svis['loc_id'][i]
                img_id = self.svis['id'][i]
                path = f'{to_dir}/{prefix}_{loc_id}' if prefix is not None else f'./{to_dir}/{loc_id}'
                p = path + f'_{img_id}.png'
                if not os.path.exists(p):
                    try:
                        if is_base64(self.svis['data'][i]):
                            save_base64(self.svis['data'][i], p)
                        else:
                            download_image_requests(self.svis['data'][i], p)
                    except Exception:
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
                    if not os.path.exists(p):
                        try:
                            clip(self.audios['data'][i], start, end, p)
                        except Exception:
                            self.audios['path'] += [" "]
                            continue
                    self.audios['path'] += [p]
            else:
                for i in tqdm(range(len(self.audios['data'])), total=len(self.audios['data'])):
                    loc_id = self.audios['loc_id'][i]
                    audio_id = self.audios['id'][i]
                    path = f'{to_dir}/{prefix}_{loc_id}' if prefix is not None else f'./{to_dir}/{loc_id}'
                    p = path + f'_{audio_id}.mp3'
                    if not os.path.exists(p):
                        try:
                            download_freesound_preview(self.audios['data'][i], p)
                        except Exception:
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
                if not os.path.exists(p):
                    try:
                        download_image_requests(self.photos['data'][i], p)
                    except Exception:
                        # download failed: align list lengths with sentinel
                        self.photos['path'] += [" "]
                        continue
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
          fov: int | float | str = 80, heading: int = None, pitch: int = 5,
          height: int = 500, width: int = 700,
          year: list | tuple = None,
          season: str = None,
          time_of_day: str = None,
          target_polygon=None,
          fov_margin: float = 0.10,
          fov_min: float = 30.0,
          fov_max: float = 120.0,
          building_height: float = 9.0,
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
            fov (int | float | str): Field of view in degrees for the perspective image
                (default 80). Pass ``'auto'`` together with ``reoriented=True`` to
                size the FOV per image so the target building is just framed —
                see ``target_polygon`` / ``fov_margin`` / ``fov_min`` / ``fov_max``.
                When ``target_polygon`` is None, ``'auto'`` falls back to a
                distance-based heuristic (assumes ~15 m wide building).
            heading (int): Camera heading in degrees. If None, it will be computed based on the location orientation.
            pitch (int): Camera pitch angle. (Default is 10).
            height (int): Height in pixels of the returned image. (Default is 480).
            width (int): Width in pixels of the returned image. (Default is 640).
            year (list[str], optional): Year of data (start year, end year).
            season (str, optional): Season of data.
            time_of_day (str, optional): Time of data.
            target_polygon (shapely.geometry.Polygon, optional): Building footprint
                used by ``fov='auto'`` to compute the angular extent of the target.
                Coordinates are assumed to be ``(lon, lat)`` in WGS84.
            fov_margin (float): Fractional padding added to the auto-computed
                FOV (0.10 = +10%). Default 0.10.
            fov_min (float): Lower clamp for ``fov='auto'`` (degrees). Default 30°.
            fov_max (float): Upper clamp for ``fov='auto'`` (degrees). Default 120°.
            building_height (float): Assumed building height in meters used by
                ``fov='auto'`` (default 9 m, ~3 stories). The auto path returns
                the wider of the horizontal extent (footprint) and the vertical
                extent (height projected through the image's aspect ratio) so a
                tall building's roof isn't cropped. Set to 0 to skip the
                height term.
            output_df (bool, optional): Whether to return a dataframe containing only the closest. (Default is True)
            silent (bool, optional): Whether to silence output (Default is False).

        Returns:
            list[str]: A list of images in base64 format
            DataFrame: A dataframe containing metadata about the closest street view images.
    """
    # Resolve auto-fov mode upfront so per-image computation can branch.
    auto_fov = isinstance(fov, str) and fov.strip().lower() == "auto"
    if auto_fov and not reoriented:
        raise ValueError("fov='auto' requires reoriented=True (we need a directed view).")

    api_key = key or os.getenv("MAPILLARY_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Mapillary access token. Pass key=... or set env var MAPILLARY_API_KEY."
        )
    bbox = projection(location, r=distance)
    url = f"https://graph.mapillary.com/images?access_token={api_key}&fields=id,computed_compass_angle,thumb_original_url,captured_at,computed_geometry,sequence&bbox={bbox}"
    # 2048 -> original to get higher resolution
    if pano:
        url += "&is_pano=true"
    if not pano and reoriented:
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
            if not silent:
                logger.warning('skip location %s: no data found', location)
            if output_df:
                return None, None
            return None
        response = response.json()
        # find the closest image
        response = closest(location, response, multi_num, interval, year, season, time_of_day, api_key)
        if response is None:
            if not silent:
                logger.warning('skip location %s: no data found', location)
            if output_df:
                return None, None
            return None

        for _index, row in response.iterrows():
            # Extract Image ID, Compass Angle, image url, and coordinates
            img_heading = float(row['computed_compass_angle'])
            img_url = row['thumb_original_url']

            if 'computed_geometry.coordinates' in row.index:
                image_lon, image_lat = row['computed_geometry.coordinates']
            elif 'coordinates' in row.index:
                image_lon, image_lat = row['coordinates']
            else:
                coor_columns = [col for col in row.index if 'coordinates' in col]
                image_lon, image_lat = row[coor_columns[0]]

            if heading is None:
                # calculate bearing to the house
                bearing_to_house = calculate_bearing(image_lat, image_lon, location[1], location[0])
                relative_heading = (bearing_to_house - img_heading) % 360
            else:
                relative_heading = heading

            # Resolve effective FOV per image — supports the literal 'auto'.
            if auto_fov:
                from .utils.utils import (
                    auto_fov_from_distance,
                    auto_fov_from_polygon,
                    haversine_m,
                )
                aspect = float(width) / float(height)
                if target_polygon is not None:
                    abs_bearing = (img_heading + relative_heading) % 360
                    effective_fov = auto_fov_from_polygon(
                        camera_lon=image_lon, camera_lat=image_lat,
                        polygon=target_polygon,
                        view_bearing_deg=abs_bearing,
                        margin=fov_margin, min_fov=fov_min, max_fov=fov_max,
                        building_height_m=building_height,
                        aspect_ratio=aspect,
                    )
                else:
                    dist_m = haversine_m(image_lat, image_lon, location[1], location[0])
                    effective_fov = auto_fov_from_distance(
                        distance_m=dist_m,
                        margin=fov_margin, min_fov=fov_min, max_fov=fov_max,
                        building_height_m=building_height,
                        aspect_ratio=aspect,
                    )
            else:
                effective_fov = fov

            # reframe image
            if reoriented:
                svi = Equirectangular(img_url=img_url)
                sv = svi.GetPerspective(effective_fov, relative_heading, pitch, height, width, 128)
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
        if not silent:
            logger.warning('skip location %s: %s', location, e)
        if output_df:
            return None, None
        return None


from .utils.utils import season_months, tod_hours, year_range


def getPhoto(
        location: list | tuple,
        loc_id: int | str = None,
        distance: int = 50,
        key: str = None,
        query: str | list[str] = None,
        geo_context: int = None,
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
            geo_context (int): Specify whether a geotagged photo was taken indoors or outdoors. 0: Not defined; 1: Indoors; 2: Outdoors. (Default is None)
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
    from datetime import datetime, timedelta, timezone

    import requests

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

    if geo_context:
        params["geo_context"] = geo_context

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
        source: str = 'freesound',
        key: str = None,
        catalog: str | pd.DataFrame = None,
        query: str | list[str] | None = None,
        tag: str | list[str] = None,
        max_return: int = 1,
        year: list | tuple = None,
        season: str = None,
        time_of_day: str = None,
        duration: int = 300,
        exclude_from_location: int = None,
        slice_duration: int = None,
        slice_max_num: int = None,
        probe_durations: bool = True,
        output_df: bool = True,
) -> pd.DataFrame | dict | list | None:
    """Dispatch to the per-source helpers.

    Args:
        source (str): one of {"freesound", "aporee"}. Default "freesound".
        catalog: required when source="aporee" — see :func:`getSoundAporee`.
        probe_durations: Aporee-only. See :func:`getSoundAporee`.

    All other arguments are forwarded; ``key`` is only used by Freesound,
    ``catalog`` and ``probe_durations`` only by Aporee.
    """
    src = (source or 'freesound').lower()
    if src == 'freesound':
        return _getSoundFreesound(
            location=location, loc_id=loc_id, distance=distance, key=key,
            query=query, tag=tag, max_return=max_return, year=year,
            season=season, time_of_day=time_of_day, duration=duration,
            exclude_from_location=exclude_from_location,
            slice_duration=slice_duration, slice_max_num=slice_max_num,
            output_df=output_df,
        )
    elif src == 'aporee':
        return getSoundAporee(
            location=location, loc_id=loc_id, distance=distance,
            catalog=catalog, query=query, tag=tag, max_return=max_return,
            year=year, season=season, time_of_day=time_of_day,
            duration=duration, exclude_from_location=exclude_from_location,
            slice_duration=slice_duration, slice_max_num=slice_max_num,
            probe_durations=probe_durations,
            output_df=output_df,
        )
    else:
        raise ValueError(
            f"Unsupported sound source {source!r}; choose 'freesound' or 'aporee'."
        )


def _getSoundFreesound(
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
        exclude_from_location: int = None,
        slice_duration: int = None,
        slice_max_num: int = None,
        output_df: bool = True,
) -> pd.DataFrame:

    """
        _getSoundFreesound (internal — call via :func:`getSound` with source='freesound')

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

    from .utils.utils import parse_iso_created as _parse_created
    from .utils.utils import solr_year_range as _year_range

    if exclude_from_location is not None:
        drop_area = projection(location, r=distance)

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
                r = session.get(endpoint, params=params, headers=headers, timeout=60)

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


# --------------------------------------------------------------------
# Aporee source
# --------------------------------------------------------------------
def getSoundAporee(
        location: list | tuple,
        loc_id: int | str = None,
        distance: int = 50,
        catalog: str | pd.DataFrame = None,
        query: str | list[str] | None = None,
        tag: str | list[str] = None,
        max_return: int = 1,
        year: list | tuple = None,
        season: str = None,
        time_of_day: str = None,
        duration: int | list | tuple = None,
        exclude_from_location: int = None,
        slice_duration: int = None,
        slice_max_num: int = None,
        probe_durations: bool = True,
        output_df: bool = True,
) -> pd.DataFrame | dict | list | None:
    """Filter a Radio Aporee catalog (CSV or DataFrame) by spatial proximity.

    Aporee (radio aporee ::: maps) does not expose a public geo-query API the
    way Freesound does, so this helper takes a pre-built catalog of geotagged
    Aporee URLs and filters it with the same semantics as
    :func:`_getSoundFreesound`. The resulting DataFrame uses the same column
    names so the downstream ``GeoTaggedData`` / ``download_to_dir`` pipeline
    needs no changes.

    Args:
        location (list | tuple): (lon, lat) of the query point.
        loc_id (int | str, optional): Identifier for the query location.
        distance (int): Search radius in meters.
        catalog (str | pandas.DataFrame): Path to a CSV file or an in-memory
            DataFrame. Required columns: ``url``, ``latitude``, ``longitude``.
            Optional columns recognised by the filters: ``id``/``identifier``,
            ``name``/``title``, ``description``, ``tags``, ``created`` (ISO
            timestamp), ``duration_s``.
        query (str | list[str], optional): Substring(s) matched against
            ``name``/``title`` and ``description`` (case-insensitive). Skipped
            silently if neither column is present.
        tag (str | list[str], optional): Substring(s) matched against ``tags``
            (case-insensitive). Skipped if column is absent.
        max_return (int): Number of nearest sounds to return.
        year, season, time_of_day: Same semantics as :func:`getSound`. Applied
            against the ``created`` column if present.
        duration (int | list[int] | tuple[int]): Filter on ``duration_s`` if
            present. Pass an int for max-only or (min, max) for a range.
        exclude_from_location (int, optional): Drop rows inside this radius
            (m) around the query point — useful for "what's nearby but not
            *at* this exact spot".
        slice_duration (int, optional): Pre-compute clip windows on top of
            the chosen recording's ``duration_s`` (mirrors Freesound path).
        slice_max_num (int, optional): Cap on number of clips per recording.
        probe_durations (bool): If True (default) and ``slice_duration`` is
            requested but the catalog has no ``duration_s`` column, fetch
            each selected recording once with
            :func:`urbanworm.utils.utils.probe_audio_duration` to learn its
            length so slice windows can be computed. Set False to skip
            slicing instead (faster; no per-row download).
        output_df (bool): If True (default) return a ``pandas.DataFrame``.

    Returns:
        ``pandas.DataFrame``, ``dict``, ``list[dict]``, or ``None`` if the
        filtered catalog is empty.
    """
    import os

    from .utils.utils import (
        haversine_m,
        is_coordinate_in_bbox,
        parse_iso_created,
        probe_audio_duration,
        season_months,
        sliced_duration,
        tod_hours,
    )

    # -------------------------
    # Validate inputs
    # -------------------------
    if max_return is None or int(max_return) < 1:
        raise ValueError("max_return must be >= 1.")
    max_return = int(max_return)

    if catalog is None:
        env_path = os.getenv("APOREE_CATALOG")
        if env_path:
            catalog = env_path
        else:
            raise ValueError(
                "source='aporee' requires a catalog (CSV path or DataFrame). "
                "Pass catalog=... or set APOREE_CATALOG env var."
            )

    if isinstance(catalog, str):
        df = pd.read_csv(catalog)
    elif isinstance(catalog, pd.DataFrame):
        df = catalog.copy()
    else:
        raise TypeError(
            "catalog must be a CSV path (str) or a pandas.DataFrame; "
            f"got {type(catalog).__name__}."
        )

    # Accept the alternate column names produced by fetch_aporee_catalog()
    # (which mirrors archive.org's `lat` / `lon` / `date` field names).
    _aliases = {"lat": "latitude", "lon": "longitude", "capture_time": "created"}
    for src, dst in _aliases.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    required = {"url", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Aporee catalog is missing required columns: {sorted(missing)}. "
            "At minimum it needs 'url', 'latitude', 'longitude' "
            "(or 'lat'/'lon' which will be renamed)."
        )

    if df.empty:
        return None if not output_df else pd.DataFrame()

    lon, lat = location

    # Coerce coords to float and drop rows that aren't usable.
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "url"]).copy()
    if df.empty:
        return None if not output_df else pd.DataFrame()

    # -------------------------
    # Spatial filter
    # -------------------------
    df["distance_m"] = df.apply(
        lambda r: haversine_m(lat, lon, float(r["latitude"]), float(r["longitude"])),
        axis=1,
    )
    df = df[df["distance_m"] <= float(distance)]

    if exclude_from_location is not None and not df.empty:
        drop_area = projection(location, r=exclude_from_location)
        mask = df.apply(
            lambda r: not is_coordinate_in_bbox(
                float(r["longitude"]), float(r["latitude"]), drop_area
            ),
            axis=1,
        )
        df = df[mask]

    # -------------------------
    # Text / tag filters
    # -------------------------
    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return [str(t).strip().lower() for t in x if str(t).strip()]
        return [str(x).strip().lower()]

    qterms = _as_list(query)
    tterms = _as_list(tag)

    if qterms:
        text_cols = [c for c in ("name", "title", "description") if c in df.columns]
        if text_cols:
            haystack = df[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
            df = df[haystack.apply(lambda s: all(q in s for q in qterms))]

    if tterms and "tags" in df.columns:
        tag_haystack = df["tags"].astype(str).str.lower()
        df = df[tag_haystack.apply(lambda s: all(t in s for t in tterms))]

    # -------------------------
    # Time filters (only if `created` column is present)
    # -------------------------
    if "created" in df.columns and (year is not None or season or time_of_day):
        parsed = df["created"].apply(parse_iso_created)
        if year is not None:
            ys = year if isinstance(year, (list, tuple)) else [year]
            y1 = int(ys[0])
            y2 = int(ys[-1])
            if y2 < y1:
                y1, y2 = y2, y1
            df = df[parsed.apply(lambda dt: dt is not None and y1 <= dt.year <= y2)]
            parsed = parsed[df.index]
        if season:
            months = season_months(season)
            df = df[parsed.apply(lambda dt: dt is not None and dt.month in months)]
            parsed = parsed[df.index]
        if time_of_day:
            hours = tod_hours(time_of_day)
            df = df[parsed.apply(lambda dt: dt is not None and dt.hour in hours)]

    # -------------------------
    # Duration filter (only if `duration_s` column is present)
    # -------------------------
    if duration is not None and "duration_s" in df.columns:
        ds = pd.to_numeric(df["duration_s"], errors="coerce")
        if isinstance(duration, (list, tuple)) and len(duration) == 2:
            dmin, dmax = float(duration[0]), float(duration[1])
            if dmax < dmin:
                dmin, dmax = dmax, dmin
            df = df[(ds >= dmin) & (ds <= dmax)]
        else:
            df = df[ds <= float(duration)]

    if df.empty:
        return None if not output_df else pd.DataFrame()

    # -------------------------
    # Normalize output schema to match Freesound path
    # -------------------------
    df = df.sort_values(by="distance_m", ascending=True).head(max_return).reset_index(drop=True)

    # `id` column: prefer existing, then `identifier`, else fall back to row index.
    if "id" not in df.columns:
        if "identifier" in df.columns:
            df["id"] = df["identifier"]
        else:
            df["id"] = [f"aporee_{i}" for i in range(len(df))]

    # Alias `url` as `preview-hq-mp3` so downstream ``download_to_dir`` works
    # without any branching.
    df["preview-hq-mp3"] = df["url"]

    if loc_id is not None:
        df["loc_id"] = loc_id
    elif "loc_id" not in df.columns:
        df["loc_id"] = ""

    # Optional slice column to mirror Freesound behavior. Aporee catalogs
    # often lack a `duration_s` column because that metadata isn't on the
    # site — probe each selected URL on-demand if requested, or skip
    # slicing with a clear warning.
    if slice_duration is not None:
        if "duration_s" not in df.columns:
            if probe_durations:
                logger.info(
                    "Aporee catalog has no 'duration_s' column; probing %d "
                    "selected recordings to determine clip windows. "
                    "(Pass probe_durations=False to skip.)",
                    len(df),
                )
                # Wrap in a lambda so pandas doesn't see attributes like
                # `.keys()` on a callable (e.g. when probe_audio_duration is
                # patched with a MagicMock in tests) and mistakenly take the
                # dict-like apply codepath.
                df["duration_s"] = df["url"].apply(lambda u: probe_audio_duration(u))
            else:
                logger.warning(
                    "Aporee catalog has no 'duration_s' column and "
                    "probe_durations=False; skipping slice generation. "
                    "Run urbanworm.dataset.enrich_aporee_catalog() once to "
                    "permanently add duration_s to your CSV."
                )

        if "duration_s" in df.columns:
            df["slice"] = df["duration_s"].apply(
                lambda d: sliced_duration(int(d), slice_duration, slice_max_num)
                if pd.notna(d) and float(d) > 0 else [[0, 0]]
            )

    if output_df:
        return df

    records = df.to_dict(orient="records")
    if max_return == 1:
        return records[0] if records else None
    return records


def enrich_aporee_catalog(
        catalog: str | pd.DataFrame,
        out_path: str | None = None,
        min_duration: float | None = None,
        skip_existing: bool = True,
        timeout: float = 60.0,
) -> pd.DataFrame:
    """Add a ``duration_s`` column to an Aporee catalog by probing each URL.

    Aporee URLs don't carry duration metadata, so this helper downloads each
    file once, reads its length with pydub (or mutagen as a fallback), and
    annotates the catalog. Optionally drops rows shorter than
    ``min_duration``.

    Use this once after building / updating your catalog so that subsequent
    :func:`getSoundAporee` calls with ``slice_duration`` can compute clip
    windows without paying the per-row probe cost every time.

    Args:
        catalog (str | pandas.DataFrame): CSV path or in-memory DataFrame.
            Must have a ``url`` column.
        out_path (str, optional): If provided, writes the enriched DataFrame
            back to this CSV path.
        min_duration (float, optional): Drop rows shorter than this many
            seconds (after probing). ``None`` keeps all rows.
        skip_existing (bool): If ``True`` (default) and ``duration_s`` is
            already populated for a row, leave it alone. Set ``False`` to
            re-probe every row.
        timeout (float): Per-URL request timeout (seconds).

    Returns:
        The enriched ``pandas.DataFrame``.
    """
    from .utils.utils import probe_audio_duration

    if isinstance(catalog, str):
        df = pd.read_csv(catalog)
    elif isinstance(catalog, pd.DataFrame):
        df = catalog.copy()
    else:
        raise TypeError(
            "catalog must be a CSV path (str) or a pandas.DataFrame; "
            f"got {type(catalog).__name__}."
        )

    if "url" not in df.columns:
        raise ValueError("Aporee catalog must have a 'url' column.")
    if "duration_s" not in df.columns:
        df["duration_s"] = pd.NA

    needs_probe = df.index if not skip_existing else df.index[df["duration_s"].isna()]
    logger.info(
        "enrich_aporee_catalog: probing %d / %d rows", len(needs_probe), len(df),
    )

    for i in tqdm(needs_probe, desc="probing", ncols=75):
        url = df.at[i, "url"]
        if not isinstance(url, str) or not url.startswith("http"):
            continue
        d = probe_audio_duration(url, timeout=timeout)
        if d is not None:
            df.at[i, "duration_s"] = round(float(d), 2)

    if min_duration is not None:
        before = len(df)
        df = df[
            df["duration_s"].notna()
            & (pd.to_numeric(df["duration_s"], errors="coerce") >= float(min_duration))
        ].reset_index(drop=True)
        logger.info(
            "enrich_aporee_catalog: dropped %d rows shorter than %ss",
            before - len(df), min_duration,
        )

    if out_path is not None:
        df.to_csv(out_path, index=False)
        logger.info("enrich_aporee_catalog: wrote %d rows to %s", len(df), out_path)

    return df


# --------------------------------------------------------------------
# fetch_aporee_catalog — pulls geolocated metadata from Internet Archive
# (radio aporee ::: maps live in the `radio-aporee-maps` IA collection).
# --------------------------------------------------------------------
_IA_SCRAPE = "https://archive.org/services/search/v1/scrape"
_IA_METADATA = "https://archive.org/metadata"
_IA_DOWNLOAD = "https://archive.org/download"
_APOREE_COLLECTION = "radio-aporee-maps"

_IA_FIELDS = [
    "identifier",
    "title",
    "date",
    "subject",
    "description",
    "latitude",
    "longitude",
    "licenseurl",
]


def _season_for(month: int, lat: float | None, southern: bool = False) -> str:
    """Hemisphere-aware season label. Mirrors the helper in scratch_aporee.py."""
    if month is None:
        return ""
    if southern or (lat is not None and lat < 0):
        month = ((month - 1 + 6) % 12) + 1
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def _ia_verify_mp3_url(identifier: str, timeout: float = 30.0) -> str:
    """Look up the actual MP3 filename via IA's metadata API.

    Falls back to ``<identifier>.mp3`` if the lookup fails.
    """
    import requests

    base = f"{_IA_DOWNLOAD}/{identifier}"
    try:
        r = requests.get(
            f"{_IA_METADATA}/{identifier}/files",
            timeout=timeout,
            headers={"User-Agent": "urban-worm/0.x (+aporee fetcher)"},
        )
        r.raise_for_status()
        files = r.json().get("result", [])
        # Prefer originals over derivatives
        mp3s = [
            f["name"] for f in files
            if f.get("name", "").endswith(".mp3") and f.get("source") != "derivative"
        ]
        if not mp3s:
            mp3s = [f["name"] for f in files if f.get("name", "").endswith(".mp3")]
        if mp3s:
            return f"{base}/{mp3s[0]}"
    except Exception as e:
        logger.debug("IA mp3 lookup failed for %s: %s", identifier, e)
    return f"{base}/{identifier}.mp3"


def fetch_aporee_catalog(
        bbox: tuple[float, float, float, float] | list | None = None,
        year: int | tuple[int, int] | list | None = None,
        hour: int | tuple[int, int] | list | None = None,
        season: str | list[str] | None = None,
        southern: bool = False,
        rows: int = 0,
        verify_urls: bool = False,
        out_path: str | None = None,
        enrich_durations: bool = False,
        min_duration: float | None = None,
        timeout: float = 60.0,
        page_size: int = 500,
) -> pd.DataFrame:
    """Fetch the Aporee sound-map catalog from Internet Archive.

    All Aporee field recordings are mirrored on archive.org under the
    ``radio-aporee-maps`` collection. This helper queries IA's Scrape API
    with optional server-side ``bbox`` / ``year`` filters and applies
    ``hour`` / ``season`` filters client-side, then returns a DataFrame in
    the schema :func:`getSoundAporee` expects.

    Args:
        bbox: ``(lat_min, lon_min, lat_max, lon_max)`` to filter server-side.
            Pass ``None`` for the whole world.
        year: Single year (``2021``) or inclusive range (``(2018, 2022)``).
            Filtered server-side via IA's ``date`` field.
        hour: UTC hour or inclusive range (``(9, 17)`` or ``(22, 4)`` for
            midnight-wrap). Applied client-side against ``capture_time``.
        season: One of ``"spring" | "summer" | "autumn"/"fall" | "winter"``,
            or a list. Hemisphere is auto-detected from each row's latitude;
            pass ``southern=True`` to force southern interpretation.
        southern (bool): Force southern-hemisphere season interpretation.
        rows (int): Maximum number of records to fetch. ``0`` means all.
        verify_urls (bool): If True, query IA's metadata API for each
            identifier to find the exact mp3 filename. Slow but accurate.
            Default False uses the ``<identifier>.mp3`` fallback (works
            for the vast majority of items).
        out_path (str, optional): If provided, write the resulting DataFrame
            to this CSV path.
        enrich_durations (bool): If True, also probe each fetched URL for
            its duration via :func:`enrich_aporee_catalog` (slow — one
            request per row).
        min_duration (float, optional): When ``enrich_durations=True``,
            drop rows shorter than this many seconds.
        timeout (float): Per-request HTTP timeout (seconds).
        page_size (int): Records per Scrape-API page (min 100).

    Returns:
        ``pandas.DataFrame`` with columns:
        ``identifier, id, latitude, longitude, url, capture_time, created,
        year, month, hour, season, title, name, description, tags, licence,
        duration_s``. ``id`` aliases ``identifier`` and ``name`` aliases
        ``title`` for compatibility with :func:`getSoundAporee`'s filters.
    """
    import requests

    # Build query
    query = f"collection:{_APOREE_COLLECTION}"
    whole_world = (-90.0, -180.0, 90.0, 180.0)
    bbox_t = tuple(bbox) if bbox is not None else whole_world
    if bbox_t != whole_world:
        lat_min, lon_min, lat_max, lon_max = bbox_t
        query += (
            f" AND lat:[{lat_min:g} TO {lat_max:g}]"
            f" AND lon:[{lon_min:g} TO {lon_max:g}]"
        )
    if year is not None:
        if isinstance(year, (list, tuple)):
            y1, y2 = int(year[0]), int(year[-1])
            if y2 < y1:
                y1, y2 = y2, y1
        else:
            y1 = y2 = int(year)
        query += f" AND date:[{y1}-01-01T00:00:00Z TO {y2}-12-31T23:59:59Z]"

    # Normalize hour filter
    hour_range: tuple[int, int] | None = None
    if hour is not None:
        if isinstance(hour, (list, tuple)):
            hour_range = (int(hour[0]), int(hour[-1]))
        else:
            hour_range = (int(hour), int(hour))

    # Normalize season filter to a set of months
    season_set: set[int] | None = None
    if season is not None:
        from .utils.utils import season_months as _sm
        names = season if isinstance(season, (list, tuple)) else [season]
        season_set = set()
        for s in names:
            season_set |= _sm(s)

    logger.info("fetch_aporee_catalog: query=%s", query)
    headers = {"User-Agent": "urban-worm/0.x (+aporee fetcher)"}
    page_size = max(100, int(page_size))

    items: list[dict] = []
    cursor: str | None = None
    fetched = 0
    skip_no_geo = skip_hour = skip_season = 0

    pbar = tqdm(desc="fetch aporee", unit="rec", disable=False)
    while True:
        if rows and fetched >= rows:
            break
        page_n = page_size if not rows else min(page_size, rows - fetched)
        params = {
            "q": query,
            "fields": ",".join(_IA_FIELDS),
            "count": max(100, page_n),
        }
        if cursor:
            params["cursor"] = cursor

        r = requests.get(_IA_SCRAPE, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if "items" not in data:
            raise RuntimeError(f"IA scrape API error: {data}")

        docs = data["items"]
        next_cursor = data.get("cursor")
        if not docs:
            break

        for doc in docs:
            try:
                lat_v = float(doc.get("latitude") or "")
                lon_v = float(doc.get("longitude") or "")
            except (ValueError, TypeError):
                skip_no_geo += 1
                continue

            ident = doc.get("identifier", "")
            title = doc.get("title", "")
            ctime = (doc.get("date") or "").strip()
            description = doc.get("description", "")
            licence = doc.get("licenseurl", "")
            subject = doc.get("subject", "")
            # `subject` may come back as a list — collapse to comma-string
            if isinstance(subject, list):
                subject = ",".join(str(s) for s in subject)

            # Client-side hour filter
            if hour_range is not None:
                hh = _ia_extract_hour(ctime)
                if hh is None:
                    skip_hour += 1
                    continue
                h_start, h_end = hour_range
                if h_start <= h_end:
                    matched = h_start <= hh <= h_end
                else:
                    matched = hh >= h_start or hh <= h_end
                if not matched:
                    skip_hour += 1
                    continue

            # Client-side season filter
            if season_set is not None:
                mm = _ia_extract_month(ctime)
                if mm is None:
                    skip_season += 1
                    continue
                row_month = mm
                if southern or lat_v < 0:
                    row_month = ((mm - 1 + 6) % 12) + 1
                if row_month not in season_set:
                    skip_season += 1
                    continue

            url = (
                _ia_verify_mp3_url(ident, timeout=timeout)
                if verify_urls
                else f"{_IA_DOWNLOAD}/{ident}/{ident}.mp3"
            )

            items.append({
                "identifier": ident,
                "id": ident,                        # alias for getSoundAporee
                "latitude": lat_v,
                "longitude": lon_v,
                "url": url,
                "capture_time": ctime,              # script's column name
                "created": ctime,                   # getSoundAporee filter name
                "title": title,
                "name": title,                      # alias for getSoundAporee.query
                "description": description,
                "tags": subject,                    # IA's `subject` -> our tags
                "licence": licence,
                "duration_s": None,
            })
            fetched += 1
            pbar.update(1)
            if rows and fetched >= rows:
                break

        # Cursor is the source of truth for "more pages available" — IA's
        # scrape API can return a partial page mid-stream, so don't bail
        # out just because len(docs) < page_n.
        if not next_cursor:
            break
        cursor = next_cursor

    pbar.close()
    logger.info(
        "fetch_aporee_catalog: kept %d, skipped no_geo=%d hour=%d season=%d",
        len(items), skip_no_geo, skip_hour, skip_season,
    )

    df = pd.DataFrame(items)
    if df.empty:
        if out_path:
            df.to_csv(out_path, index=False)
        return df

    # Enrich with derived time columns (year/month/hour/season) for
    # downstream convenience. ``parse_iso_created`` handles missing
    # fractional-seconds gracefully.
    from .utils.utils import parse_iso_created
    parsed = df["capture_time"].apply(parse_iso_created)
    df["year"] = parsed.apply(lambda d: d.year if d is not None else None)
    df["month"] = parsed.apply(lambda d: d.month if d is not None else None)
    df["hour"] = parsed.apply(lambda d: d.hour if d is not None else None)
    df["season"] = df.apply(
        lambda r: _season_for(r["month"], r["latitude"], southern) if r["month"] else "",
        axis=1,
    )

    if enrich_durations:
        df = enrich_aporee_catalog(df, min_duration=min_duration, timeout=timeout)

    if out_path:
        df.to_csv(out_path, index=False)
        logger.info("fetch_aporee_catalog: wrote %d rows to %s", len(df), out_path)

    return df


def _ia_extract_hour(s: str) -> int | None:
    """Pull the UTC hour out of an IA ``date`` field, or None."""
    from datetime import datetime
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt).hour
        except ValueError:
            continue
    return None


def _ia_extract_month(s: str) -> int | None:
    """Pull the month out of an IA ``date`` field, or None."""
    from datetime import datetime
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(s.strip(), fmt).month
        except ValueError:
            continue
    return None
