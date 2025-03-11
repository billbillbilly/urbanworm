__version__ = '0.0.1'

import ollama
from pydantic import BaseModel
import rasterio
from rasterio.mask import mask
from samgeo import tms_to_geotiff
import tempfile
import os
from typing import List
from utils import loadSHP
from utils import getSV

class QnA(BaseModel):
    question: str
    answer: str
    explanation: str

class Response(BaseModel):
    responses: List[QnA] = []

class UrbanDataSet:
    '''
    Dataset class for urban imagery inference using LLM.

    Args:
        image (str): The path to the image.
        images (list): The list of image paths.
        units (str): The path to the shapefile.
        format (Response): The response format.
        mapillary_key (str): The Mapillary API key.
        random_sample (int): The number of random samples.
    '''
    def __init__(self, image=None, images=None, units=None, format=None, mapillary_key=None, random_sample=None):
        self.img = image
        self.imgs = images
        if random_sample != None and units != None:
            self.parcels = loadSHP(units).sample(random_sample)
        elif random_sample == None and units != None:
            self.parcels = loadSHP(units)
        else:
            self.parcels = units
        if format == None:
            self.format = Response()
        else:
            self.format = format
        self.mapillary_key = mapillary_key
    
    def oneImgChat(self, system=None, prompt=None, 
                   temp=0.0, top_k=0.8, top_p=0.8):
        r = self.LLM_chat(system=system, prompt=prompt, img=[self.img], 
                             temp=temp, top_k=top_k, top_p=top_p)
        return r.responses[0]
    
    def loopImgChat(self, system=None, prompt=None, 
                    temp=0.0, top_k=0.8, top_p=0.8):
        res = []
        for img in self.imgs:
            r = self.LLM_chat(system=system, prompt=prompt, img=[img], 
                              temp=temp, top_k=top_k, top_p=top_p)
            res += [r.responses[0]]
        return res
            
    def loopUnitChat(self, system=None, prompt=None, 
                     temp=0.0, top_k=0.8, top_p=0.8, 
                     type='top', epsg=None, multi=False):
        '''
        chat with LLM model for each unit in the shapefile.

        Args:
            type (str): The type of image to process.
            epsg (int): The EPSG code.
            multi (bool): The multi flag.
        '''
        dic = {
            "lon": [],
            "lat": [],
        }
        for i in range(len(self.parcels)):
            # Get the extent of one polygon from the filtered GeoDataFrame
            polygon = self.parcels.geometry.iloc[i]
            centroid = polygon.centroid
            minx, miny, maxx, maxy = polygon.bounds
            bbox = [minx, miny, maxx, maxy]

            dic['lon'].append(self.parcels.centroid.x.iloc[i])
            dic['lat'].append(self.parcels.centroid.y.iloc[i])

            if type == 'top' or type == 'both':
                # Download data using tms_to_geotiff
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
                    image = temp_file.name  # Get the temp file path
                tms_to_geotiff(output=image, bbox=bbox, zoom=22, 
                            source="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", 
                            overwrite=True)
                # Clip the image with the polygon
                with rasterio.open(image) as src:
                    # Reproject the polygon back to match raster CRS
                    polygon = self.parcels.to_crs(src.crs).geometry.iloc[i]
                    out_image, out_transform = mask(src, [polygon], crop=True)
                    out_meta = src.meta.copy()

                out_meta.update({
                    "driver": "JPEG",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "count": 3 #Ensure RGB (3 bands)
                })

                # Create a temporary file for the clipped JPEG
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_jpg:
                    clipped_image = temp_jpg.name
                with rasterio.open(clipped_image, "w", **out_meta) as dest:
                    dest.write(out_image)
                # clean up temp file
                os.remove(image)
                # process areal image
                top_res = self.LLM_chat(system=system, 
                                    prompt=prompt, 
                                    img=[clipped_image], 
                                    temp=temp, 
                                    top_k=top_k, 
                                    top_p=top_p)
                dic['roof_condition'].append(top_res)
                # clean up temp file
                os.remove(clipped_image)

            # process street view image
            if  (type == 'street' or type == 'both') and epsg != None and self.mapillary_key != None:
                input_svis = [getSV(centroid, epsg, self.mapillary_key, multi=multi)]
                if None not in input_svis:
                    res = self.LLM_chat(system=system, 
                                        prompt=prompt, 
                                        img=input_svis, 
                                        temp=temp, 
                                        top_k=top_k, 
                                        top_p=top_p)
                    dic['streetview_conditions'].append(res)
        return dic
    
    def LLM_chat(self, system=None, prompt=None, img=None, temp=None, top_k=None, top_p=None):
        '''
        This function is used to chat with the LLM model with a list of images.

        Args:
            img (list): The list of image paths.
        '''

        if prompt != None and img != None:
            if len(img) == 1:
                return self.chat(system, prompt, img[0], temp, top_k, top_p)
            elif len(img) == 3:
                res = []
                system = f'You are analyzing aerial or street view images. For street view, you should just foucus on the building and yard in the middle. {system}'
                for i in range(len(img)):
                    r = self.chat(system, prompt, img[i], temp, top_k, top_p)
                    res += [r.responses]
                return res

    def chat(self, system=None, prompt=None, img=None, temp=None, top_k=None, top_p=None):
        '''
        This function is used to chat with the LLM model.'

        Args:
            system (str): The system message.
            prompt (str): The user message.
            img (str): The image path.
            temp (float): The temperature value.
            top_k (float): The top_k value.
            top_p (float): The top_p value.
        '''
        res = ollama.chat(
            model='llama3.2-vision',
            format=self.format.model_json_schema(),
            messages=[
                {
                    'role': 'system',
                    'content': system
                },
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [img]
                }
            ],
            options={
                "temperature":temp,
                "top_k":top_k,
                "top_p":top_p
            }
        )
        return self.format.model_validate_json(res.message.content)
    
    # def getSV(self, centroid, epsg, key, multi=False):
    #     bbox = self.projection(centroid, epsg)
    #     url = f"https://graph.mapillary.com/images?access_token={key}&fields=id,compass_angle,thumb_2048_url,geometry&bbox={bbox}&is_pano=true"
    #     # while not response or 'data' not in response:
    #     try:
    #         response = requests.get(url).json()
    #         # find the closest image
    #         response = self.closest(centroid, response, multi)

    #         svis = []
    #         for i in range(len(response)):
    #             # Extract Image ID, Compass Angle, image url, and coordinates
    #             img_heading = float(response.iloc[i,1])
    #             img_url = response.iloc[i,2]
    #             image_lon, image_lat = response.iloc[i,5]
    #             # calculate bearing to the house
    #             bearing_to_house = self.calculate_bearing(image_lat, image_lon, centroid.y, centroid.x)
    #             relative_heading = (bearing_to_house - img_heading) % 360
    #             # reframe image
    #             svi = Equirectangular(img_url=img_url)
    #             sv = svi.GetPerspective(80, relative_heading, 10, 300, 400, 128)
    #             svis.append(sv)
    #         return svis
    #     except:
    #         print("no street view image found")
    #         return None
    
    # def projection(self, centroid, epsg):
    #     x, y = self.degree2dis(centroid, epsg)
    #     # Get unit name (meters, degrees, etc.)
    #     crs = CRS.from_epsg(epsg)
    #     unit_name = crs.axis_info[0].unit_name
    #     # set search distance to 25 meters
    #     r = 50
    #     if unit_name == 'foot':
    #         r = 164.042
    #     elif unit_name == 'degree':
    #         print("Error: epsg must be projected system.")
    #         sys.exit(1)
    #     # set bbox
    #     x_min = x - r
    #     y_min = y - r
    #     x_max = x + r
    #     y_max = y + r
    #     # Convert to EPSG:4326 (Lat/Lon) 
    #     x_min, y_min = self.dis2degree(x_min, y_min, epsg)
    #     x_max, y_max = self.dis2degree(x_max, y_max, epsg)
    #     return f'{x_min},{y_min},{x_max},{y_max}'

    # def dis2degree(self, ptx, pty, epsg):
    #     transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    #     x, y = transformer.transform(ptx, pty)
    #     return x, y
    
    # def degree2dis(self, pt, epsg):
    #     transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    #     x, y = transformer.transform(pt.x, pt.y)
    #     return x, y
    
    # def closest(self, centroid, response, multi=False):
    #     c = [centroid.x, centroid.y]
    #     res_df = pd.DataFrame(response['data'])
    #     res_df[['point','coordinates']] = pd.DataFrame(res_df.geometry.tolist(), index= res_df.index)
    #     res_df[['lon','lat']] = pd.DataFrame(res_df.coordinates.tolist(), index= res_df.index)
    #     id_array = np.array(res_df['id'])
    #     lon_array = np.array(res_df['lon'])
    #     lat_array = np.array(res_df['lat'])
    #     dis_array = (lon_array-c[0])*(lon_array-c[0]) + (lat_array-c[1])*(lat_array-c[1])
    #     if multi == True and len(dis_array) > 3:
    #         smallest_indices = np.argsort(dis_array)[:3]
    #         return res_df.loc[res_df['id'].isin(id_array[smallest_indices])]
    #     ind = np.where(dis_array == np.min(dis_array))[0]
    #     id = id_array[ind][0]
    #     return res_df.loc[res_df['id'] == id]
    
    # # filter images by time and seasons
    
    # def calculate_bearing(self, lat1, lon1, lat2, lon2):
    #     lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    #     delta_lon = lon2 - lon1

    #     x = math.sin(delta_lon) * math.cos(lat2)
    #     y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))

    #     bearing = math.degrees(math.atan2(x, y))
    #     return (bearing + 360) % 360  # Normalize to 0-360