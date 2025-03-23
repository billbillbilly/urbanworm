__version__ = '0.0.1'

import ollama
from pydantic import BaseModel
import rasterio
import geopandas as gpd
from rasterio.mask import mask
import tempfile
import os
from typing import Union
from typing import List
from .utils import *

class QnA(BaseModel):
    question: str
    answer: str
    explanation: str

class Response(BaseModel):
    responses: List[QnA] = []

class UrbanDataSet:
    '''
    Dataset class for urban imagery inference using MLLM.
    '''
    def __init__(self, image=None, images:list=None, units:str=None, format=None, mapillary_key=None, random_sample:int=None):
        '''
        Add data or api key

        Args:
            image (str): The path to the image.
            images (list): The list of image paths.
            units (str): The path to the shapefile.
            format (Response): The response format.
            mapillary_key (str): The Mapillary API key.
            random_sample (int): The number of random samples.
        '''

        if image != None and detect_input_type(image) == 'image_path':
            self.img = encode_image_to_base64(image)
        else:
            self.img = image

        if images != None and detect_input_type(images[0]) == 'image_path':
            self.imgs = [encode_image_to_base64(im) for im in images]
        else:
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

        self.results = None

        self.preload_model()

    def preload_model(self):
        """
        Ensures that the required Ollama model is available.
        If not, it automatically pulls the model.
        """

        import ollama

        model_name = "llama3.2-vision"
        try:
            ollama.pull(model_name)

        except Exception as e:
            print(f"Warning: Ollama is not installed or failed to check models: {e}")
            print("Please install Ollama client: https://github.com/ollama/ollama/tree/main")
            raise RuntimeError("Ollama not available. Install it before running.")

    def bbox2Buildings(self, bbox, source='osm', min_area=0, max_area=None, random_sample=None) -> str:
        '''
        Extract buildings from OpenStreetMap using the bbox.

        Args:
            bbox (list): The bounding box.
            source (str): The source of the buildings. ['osm', 'being']
            min_area (int): The minimum area.
            max_area (int): The maximum area.
            random_sample (int): The number of random samples.

        Returns:
            str: The number of buildings found in the bounding box
        '''
        if source == 'osm':
            buildings = getOSMbuildings(bbox, min_area, max_area)
        elif source == 'being':
            buildings = getGlobalMLBuilding(bbox, min_area, max_area)
        if buildings is None or buildings.empty:
            if source == 'osm':
                return "No buildings found in the bounding box. Please check https://overpass-turbo.eu/ for areas with buildings."
            if source == 'being':
                return "No buildings found in the bounding box. Please check https://github.com/microsoft/GlobalMLBuildingFootprints for areas with buildings."
        if random_sample != None:
            buildings = buildings.sample(random_sample)
        self.units = buildings
        return f"{len(buildings)} buildings found in the bounding box."
    
    def oneImgChat(self, system=None, prompt=None, 
                   temp=0.0, top_k=0.8, top_p=0.8, 
                   saveImg:bool=True) -> dict:
        
        '''
        Chat with MLLM model with one image.

        Args:
            system (optinal): The system message.
            prompt (str): The prompt message.
            img (str): The image path.
            temp (float): The temperature value.
            top_k (float): The top_k value.
            top_p (float): The top_p value.
            saveImg (bool): The saveImg for save each image in base64 format in the output.

        Returns:
            dict: A dictionary includes questions/messages, responses/answers, and image base64 (if required) 
        '''

        print("Inference starts ...")
        r = self.LLM_chat(system=system, prompt=prompt, img=[self.img], 
                          temp=temp, top_k=top_k, top_p=top_p)
        r = dict(r.responses[0])
        if saveImg:
            r['img'] = self.img
        return r
    
    def loopImgChat(self, system=None, prompt=None, 
                    temp=0.0, top_k=0.8, top_p=0.8,
                    saveImg:bool=True, progressBar:bool=False) -> list:
        '''
        Chat with MLLM model for each image.

        Args:
            system (optinal): The system message.
            prompt (str): The prompt message.
            temp (float): The temperature value.
            top_k (float): The top_k value.
            top_p (float): The top_p value.
            saveImg (bool): The saveImg for save each image in base64 format in the output.
            progressBar (bool): The progress bar for showing the progress of data analysis over the units

        Returns:
            list A list of dictionaries. Each dict includes questions/messages, responses/answers, and image base64 (if required)
        '''

        from tqdm import tqdm

        res = []
        for i in tqdm(range(len(self.imgs)), desc="Processing...", ncols=75, disable=progressBar):
        # for i in range(len(self.imgs)):
            img = self.imgs[i]
            r = self.LLM_chat(system=system, prompt=prompt, img=[img], 
                              temp=temp, top_k=top_k, top_p=top_p)
            r = dict(r.responses[0])
            if saveImg:
                im = {'img': img}
                res += [{**r, **im}]
            else:
                res += [r]
        return res
            
    def loopUnitChat(self, system=None, prompt:dict=None, 
                     temp:float=0.0, top_k:float=0.8, top_p:float=0.8, 
                     type:str='top', epsg:int=None, multi:bool=False, 
                     sv_fov:int=80, sv_pitch:int=10, sv_size:tuple=(300,400),
                     saveImg:bool=True, progressBar:bool=False) -> dict:
        """
        Chat with the MLLM model for each spatial unit in the shapefile.

        This function loops through all units (e.g., buildings or parcels) in `self.units`, 
        generates top and/or street view images, and prompts a language model 
        with custom messages. It stores results in `self.results`.

        When finished, your self.results object looks like this:
        ```python
        {
            'from_loopUnitChat': {
                'lon': [...],
                'lat': [...],
                'top_view': [[QnA, QnA, ...], ...],      # Optional
                'street_view': [[QnA, QnA, ...], ...],   # Optional
            },
            'base64_imgs': {
                'top_view_base64': [...],      # Optional
                'street_view_base64': [...],   # Optional
            }
        }
        ```

        Example prompt:
            prompt = {
                "top": "
                    Is there any damage on the roof?
                ",
                "street": "
                    Is the wall missing or damaged? 
                    Is the yard maintained well?
                "
            }

        Args:
            system (str, optional): System message to guide the LLM behavior.
            prompt (dict): Dictionary containing the prompts for 'top' and/or 'street' views.
            temp (float, optional): Temperature for generation randomness. Defaults to 0.0.
            top_k (float, optional): Top-k sampling parameter. Defaults to 0.8.
            top_p (float, optional): Top-p sampling parameter. Defaults to 0.8.
            type (str, optional): Which image type(s) to use: "top", "street", or "both". Defaults to "top".
            epsg (int, optional): EPSG code for coordinate transformation. Required if type includes "street".
            multi (bool, optional): Whether to return multiple SVIs per unit. Defaults to False.
            sv_fov (int, optional): Field of view for street view. Defaults to 80.
            sv_pitch (int, optional): Pitch angle for street view. Defaults to 10.
            sv_size (tuple, optional): Size (height, width) for street view images. Defaults to (300, 400).
            saveImg (bool, optional): Whether to save images (as base64 strings) in output. Defaults to True.
            progressBar (bool, optional): Whether to show progress bar. Defaults to False.

        Returns:
            dict: A dictionary containing prompts, responses, and (optionally) image data for each unit.
        """

        from tqdm import tqdm

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

        top_view_imgs = {'top_view_base64':[]}
        street_view_imgs = {'street_view_base64':[]}

        for i in tqdm(range(len(self.units)), desc="Processing...", ncols=75, disable=progressBar):
        # for i in range(len(self.units)):
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
                               source="SATELLITE", 
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

                # convert image into base64
                clipped_image_base64 = encode_image_to_base64(clipped_image)
                top_view_imgs['top_view_base64'] += [clipped_image_base64]

                # process aerial image
                top_res = self.LLM_chat(system=system, 
                                    prompt=prompt["top"], 
                                    img=[clipped_image], 
                                    temp=temp, 
                                    top_k=top_k, 
                                    top_p=top_p)
                # initialize the list
                if i == 0:
                    dic['top_view'] = []
                if saveImg:
                    dic['top_view'].append(top_res.responses)
                
                # clean up temp file
                os.remove(clipped_image)

            # process street view image
            if (type == 'street' or type == 'both') and epsg != None and self.mapillary_key != None:
                input_svis = getSV(centroid, epsg, self.mapillary_key, multi=multi, 
                                   fov=sv_fov, pitch=sv_pitch, height=sv_size[0], width=sv_size[1])
                if None not in input_svis:
                    # save imgs
                    if saveImg:
                        street_view_imgs['street_view_base64'] += [input_svis]
                    # inference
                    res = self.LLM_chat(system=system, 
                                        prompt=prompt["street"], 
                                        img=input_svis, 
                                        temp=temp, 
                                        top_k=top_k, 
                                        top_p=top_p)
                    # initialize the list
                    if i == 0:
                        dic['street_view'] = []
                    if multi:
                        dic['street_view'] += [res]
                    else:
                        dic['street_view'] += [res.responses]

        self.results = {'from_loopUnitChat':dic, 'base64_imgs':{**top_view_imgs, **street_view_imgs}}
        return dic
    
    def to_gdf(self) -> gpd.GeoDataFrame | str:
        """
        Convert the output from an MLLM response (from .loopUnitChat) into a GeoDataFrame.

        This method extracts coordinates, questions, responses, and base64-encoded input images
        from the stored `self.results` object, and formats them into a structured GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing spatial responses and associated metadata.
            str: An error message if `.loopUnitChat()` has not been run or if the format is unsupported.
        """

        import pandas as pd
        import copy

        if self.results is not None:
            if 'from_loopUnitChat' in self.results:
                res_df = response2gdf(self.results['from_loopUnitChat'])
                img_dic = copy.deepcopy(self.results['base64_imgs'])
                if img_dic['top_view_base64'] != [] or img_dic['street_view_base64'] != []:
                    if img_dic['top_view_base64'] == []:
                        img_dic.pop("top_view_base64")
                    if img_dic['street_view_base64'] == []:
                        img_dic.pop("street_view_base64")
                    imgs_df = pd.DataFrame(img_dic)
                    return pd.concat([res_df, imgs_df], axis=1)
                else:
                    return res_df
            else:
                return "This method can only support the output of '.loopUnitChat()' method"
        else:
            return "This method can only be called after running the '.loopUnitChat()' method"
    
    def LLM_chat(self, system=None, prompt=None, img=None, temp=None, top_k=None, top_p=None) -> Union["Response", list["QnA"]]:
        '''
        Chat with the LLM model with a list of images.\
        
        Depending on the number of images provided, the method will:
        - Return a single Response object if only one image is provided.
        - Return a list of QnA objects if multiple images are provided (e.g., aerial and street views).

        Args:
            system (str): The system message guiding the LLM.
            prompt (str): The user prompt to the LLM.
            img (list[str]): A list of image paths (1 or 3 recommended).
            temp (float, optional): Temperature parameter for response randomness.
            top_k (float, optional): Top-K sampling filter.
            top_p (float, optional): Top-P (nucleus) sampling filter.

        Returns:
            Union[Response, list[QnA]]: A Response object if a single reply is generated,
            or a list of QnA objects for multi-turn/image-question responses.
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

    def chat(self, system=None, prompt=None, img=None, temp=None, top_k=None, top_p=None) -> Response:
        '''
        Chat with the LLM model using a system message, prompt, and optional image.

        Args:
            system (str): The system-level instruction for the model.
            prompt (str): The user message or question.
            img (str): Path to a single image to be sent to the model.
            temp (float, optional): Sampling temperature for generation (higher = more random).
            top_k (float, optional): Top-k sampling parameter.
            top_p (float, optional): Top-p (nucleus) sampling parameter.

        Returns:
            Response: Parsed response from the LLM, returned as a `Response` object.
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
    
    def plotBase64(self, img):
        '''
        plot a single base64 image
        '''
        plot_base64_image(img)