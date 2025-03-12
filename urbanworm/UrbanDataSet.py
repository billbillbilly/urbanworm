__version__ = '0.0.1'

import ollama
from pydantic import BaseModel
import rasterio
from rasterio.mask import mask
import tempfile
import os
from typing import List
from utils import loadSHP
from utils import meters_to_degrees
from utils import getSV
from utils import tms_to_geotiff
from utils import getOSMbuildings

class QnA(BaseModel):
    question: str
    answer: str
    explanation: str

class Response(BaseModel):
    responses: List[QnA] = []

class UrbanDataSet:
    '''
    Dataset class for urban imagery inference using MLLM.

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
            self.units = loadSHP(units).sample(random_sample)
        elif random_sample == None and units != None:
            self.units = loadSHP(units)
        else:
            self.units = units
        if format == None:
            self.format = Response()
        else:
            self.format = format
        self.mapillary_key = mapillary_key

    def bbox2osmBuildings(self, bbox, min_area=0, max_area=None, random_sample=None):
        '''
        This function is used to extract buildings from OpenStreetMap using the bbox.

        Args:
            bbox (list): The bounding box.
            min_area (int): The minimum area.
            max_area (int): The maximum area.
            random_sample (int): The number of random samples.
        '''
        buildings = getOSMbuildings(bbox, min_area, max_area)
        if buildings is None or buildings.empty:
            return "No buildings found in the bounding box. Please check https://overpass-turbo.eu/ for areas with buildings."
        if random_sample != None:
            buildings = buildings.sample(random_sample)
        self.units = buildings
        return f"{len(buildings)} buildings found in the bounding box."
    
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
            
    def loopUnitChat(self, system=None, prompt:dict=None, 
                     temp:float=0.0, top_k:float=0.8, top_p:float=0.8, 
                     type:str='top', epsg:int=None, multi:bool=False):
        '''
        chat with MLLM model for each unit in the shapefile.
        example prompt:
        prompt = {
            'top': ''
                Is there any damage on the roof?
            '',
            'street': ''
                Is the wall missing or damaged?
                Is the yard maintained well?
            ''
        }

        Args:
            system (optinal): The system message.
            prompt (dict): The prompt message for either top or street view or both.
            type (str): The type of image to process.
            epsg (int): The EPSG code (required when type='street' or type='both').
            multi (bool): The multi flag for multiple street view images for one unit.
        '''

        if type == 'top' and 'top' not in prompt:
            return "Please provide prompt for top view images when type='top'"
        if type == 'street' and 'street' not in prompt:
            return "Please provide prompt for street view images when type='street'"
        if type == 'both' and 'top' not in prompt and 'street' not in prompt:
            return "Please provide prompt for both top and street view images when type='both'"

        dic = {
            "lon": [],
            "lat": [],
        }

        for i in range(len(self.units)):
            # Get the extent of one polygon from the filtered GeoDataFrame
            polygon = self.units.geometry.iloc[i]
            centroid = polygon.centroid
            
            dic['lon'].append(centroid.x)
            dic['lat'].append(centroid.y)

            if type == 'top' or type == 'both':
                # Convert meters to degrees dynamically based on latitude
                # Approximate adjustment (5 meters)
                degree_offset = meters_to_degrees(5, centroid.y)  # Convert 5m to degrees
                polygon = polygon.buffer(degree_offset)
                # Compute bounding box
                minx, miny, maxx, maxy = polygon.bounds
                bbox = [minx, miny, maxx, maxy]

                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
                    image = temp_file.name
                # Download data using tms_to_geotiff
                tms_to_geotiff(output=image, bbox=bbox, zoom=22, 
                               source="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", 
                               overwrite=True)
                # Clip the image with the polygon
                with rasterio.open(image) as src:
                    # Reproject the polygon back to match raster CRS
                    polygon = self.units.to_crs(src.crs).geometry.iloc[i]
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
                # process aerial image
                top_res = self.LLM_chat(system=system, 
                                    prompt=f'Please note that you should answer questions for the aerial image. {prompt["top"]}', 
                                    img=[clipped_image], 
                                    temp=temp, 
                                    top_k=top_k, 
                                    top_p=top_p)
                # initialize the list
                if i == 0:
                    dic['top_view'] = []
                dic['top_view'].append(top_res.responses)
                # clean up temp file
                os.remove(clipped_image)

            # process street view image
            if  (type == 'street' or type == 'both') and epsg != None and self.mapillary_key != None:
                input_svis = getSV(centroid, epsg, self.mapillary_key, multi=multi)
                if None not in input_svis:
                    res = self.LLM_chat(system=system, 
                                        prompt=f'Please note that you should answer questions for the street view image. {prompt["street"]}', 
                                        img=input_svis, 
                                        temp=temp, 
                                        top_k=top_k, 
                                        top_p=top_p)
                    # initialize the list
                    if i == 0:
                        dic['street_view'] = []
                    dic['street_view'].append(res.responses)
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

