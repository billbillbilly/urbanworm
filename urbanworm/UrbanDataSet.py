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
    Dataset class for urban imagery inference using MLLMs.
    '''

    def __init__(self, image=None, images: list = None, units: str | gpd.GeoDataFrame = None,
                 format: Response = None, mapillary_key: str = None, random_sample: int = None):
        '''
        Add data or api key

        Args:
            image (str): The path to the image.
            images (list): The list of image paths.
            units (str or GeoDataFrame): The path to the shapefile or geojson file, or GeoDataFrame.
            format (Response): The response format.
            mapillary_key (str): The Mapillary API key.
            random_sample (int): The number of random samples.
        '''

        if image is not None and detect_input_type(image) == 'image_path':
            self.img = encode_image_to_base64(image)
        else:
            self.img = image

        if images is not None and detect_input_type(images[0]) == 'image_path':
            self.imgs = images
            self.base64Imgs = [encode_image_to_base64(im) for im in images]
        else:
            self.imgs = images

        if random_sample is not None and units is not None:
            self.units = self.__checkUnitsInputType(units)
            self.units = self.units.sample(random_sample)
        elif random_sample == None and units is not None:
            self.units = self.__checkUnitsInputType(units)
        else:
            self.units = units

        if format is None:
            self.format = Response()
        else:
            self.format = format

        self.mapillary_key = mapillary_key

        self.results, self.geo_df, self.df = None, None, None
        self.messageHistory = []

    def __checkUnitsInputType(self, input: str | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        match input:
            case str():
                if ".shp" in input.lower() or ".geojson" in input.lower():
                    return loadSHP(input)
                else:
                    raise ("Wrong type for units input!")
            case gpd.GeoDataFrame():
                return input
            case _:
                raise ("Wrong type for units input!")

    def __checkModel(self, model: str) -> None:
        '''
        Check if the model is available.

        Args:
            model (str): The model name.
        '''

        if model not in ['granite3.2-vision',
                         'llama3.2-vision',
                         'gemma3',
                         'gemma3:1b',
                         'gemma3:12b',
                         'gemma3:27b',
                         'minicpm-v',
                         'mistral-small3.1']:
            raise Exception(f'{model} is not supported')

    def preload_model(self, model_name: str):
        """
        Ensures that the required Ollama model is available.
        If not, it automatically pulls the model.

        Args:
            model_name (str): model name
        """
        import ollama

        try:
            ollama.pull(model_name)

        except Exception as e:
            print(f"Warning: Ollama is not installed or failed to check models: {e}")
            print("Please install Ollama client: https://github.com/ollama/ollama/tree/main")
            raise RuntimeError("Ollama not available. Install it before running.")

    def bbox2Buildings(self, bbox: list | tuple, source: str = 'osm', epsg: int = None,
                       min_area: float | int = 0, max_area: float | int = None,
                       random_sample: int = None) -> str:
        '''
        Extract buildings from OpenStreetMap using the bbox.

        Args:
            bbox (list or tuple): The bounding box.
            source (str): The source of the buildings. ['osm', 'bing']
            epsg (int, optional): EPSG code for coordinate transformation. Required if source='bing' and (min_area > 0 or max_area) is specified.
            min_area (float or int): The minimum area.
            max_area (float or int): The maximum area.
            random_sample (int): The number of random samples.

        Returns:
            str: The number of buildings found in the bounding box
        '''

        if source not in ['osm', 'bing']:
            raise Exception(f'{source} is not supported')

        if source == 'osm':
            buildings = getOSMbuildings(bbox, min_area, max_area)
        elif source == 'bing':
            if epsg is None:
                raise "Please specify epsg"
            buildings = getGlobalMLBuilding(bbox, epsg, min_area, max_area)
        if buildings is None or buildings.empty:
            if source == 'osm':
                return "No buildings found in the bounding box. Please check https://overpass-turbo.eu/ for areas with buildings."
            if source == 'bing':
                return "No buildings found in the bounding box. Please check https://github.com/microsoft/GlobalMLBuildingFootprints for areas with buildings."
        if random_sample is not None:
            buildings = buildings.sample(random_sample)
        self.units = buildings
        return f"{len(buildings)} buildings found in the bounding box."

    def oneImgChat(self, model: str = 'gemma3:12b', system: str = None, prompt: str = None,
                   temp: float = 0.0, top_k: float = 1.0, top_p: float = 0.8,
                   saveImg: bool = True) -> dict:

        '''
        Chat with MLLM model with one image.

        Args:
            model (str): Model name. Defaults to "gemma3:12b". ['granite3.2-vision', 'llama3.2-vision', 'gemma3', 'gemma3:1b', 'gemma3:12b', 'minicpm-v', 'mistral-small3.1']
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

        self.__checkModel(model)
        self.preload_model(model)

        print("Inference starts ...")
        r = self.LLM_chat(model=model, system=system, prompt=prompt, img=[self.img],
                          temp=temp, top_k=top_k, top_p=top_p)
        r = dict(r.responses[0])
        if saveImg:
            r['img'] = self.img
        return r

    def loopImgChat(self, model: str = 'gemma3:12b', system: str = None, prompt: str = None,
                    temp: float = 0.0, top_k: float = 1.0, top_p: float = 0.8, saveImg: bool = False,
                    output_df: bool = False, disableProgressBar: bool = False) -> dict:
        '''
        Chat with MLLM model for each image.

        Args:
            model (str): Model name. Defaults to "gemma3:12b". ['granite3.2-vision', 'llama3.2-vision', 'gemma3', 'gemma3:1b', 'gemma3:12b', 'minicpm-v', 'mistral-small3.1']
            system (str, optinal): The system message.
            prompt (str): The prompt message.
            temp (float): The temperature value.
            top_k (float): The top_k value.
            top_p (float): The top_p value.
            saveImg (bool): The saveImg for saving each image in base64 format in the output.
            output_df (bool): The output_df for saving the result in a pandas DataFrame. Defaults to False.
            disableProgressBar (bool): The progress bar for showing the progress of data analysis over the units

        Returns:
            list A list of dictionaries. Each dict includes questions/messages, responses/answers, and image base64 (if required)
        '''

        self.__checkModel(model)
        self.preload_model(model)

        from tqdm import tqdm

        dic = {'responses': [], 'img': []}
        for i in tqdm(range(len(self.imgs)), desc="Processing...", ncols=75, disable=disableProgressBar):
            img = self.base64Imgs[i]
            r = self.LLM_chat(model=model, system=system, prompt=prompt, img=[img],
                              temp=temp, top_k=top_k, top_p=top_p)
            r = r.responses
            if saveImg:
                if i == 0:
                    dic['imgBase64'] = []
                dic['imgBase64'] += [img]
            dic['responses'] += [r]
            dic['img'] += [self.imgs[i]]
        self.results = {'from_loopImgChat': dic}
        if output_df:
            return self.to_df(output=True)
        return dic

    def loopUnitChat(self, model: str = 'gemma3:12b', system: str = None, prompt: dict = None,
                     temp: float = 0.0, top_k: float = 1.0, top_p: float = 0.8,
                     type: str = 'top', epsg: int = None, multi: bool = False,
                     sv_fov: int = 80, sv_pitch: int = 10, sv_size: list | tuple = (300, 400),
                     year: list | tuple = None, season: str = None, time_of_day: str = None,
                     saveImg: bool = True, output_gdf: bool = False, disableProgressBar: bool = False) -> dict:
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
                'top_view': [[QnA, QnA, ...], ...],     
                'street_view': [[QnA, QnA, ...], ...],   
            },
            'base64_imgs': {
                'top_view_base64': [...],      
                'street_view_base64': [...], 
            }
        }
        ```

        Example prompt:
        ```python
        prompt = {
            "top": "
                Is there any damage on the roof?
            ",
            "street": "
                Is the wall missing or damaged? 
                Is the yard maintained well?
            "
        }
        ```

        Args:
            model (str): Model name. Defaults to "gemma3:12b". ['granite3.2-vision', 'llama3.2-vision', 'gemma3', 'gemma3:1b', 'gemma3:12b', 'gemma3:27b', 'minicpm-v', 'mistral-small3.1]
            system (str, optional): System message to guide the LLM behavior.
            prompt (dict): Dictionary containing the prompts for 'top' and/or 'street' views.
            temp (float, optional): Temperature for generation randomness. Defaults to 0.0.
            top_k (float, optional): Top-k sampling parameter. Defaults to 1.0.
            top_p (float, optional): Top-p sampling parameter. Defaults to 0.8.
            type (str, optional): Which image type(s) to use: "top", "street", or "both". Defaults to "top".
            epsg (int, optional): EPSG code for coordinate transformation. Required if type includes "street".
            multi (bool, optional): Whether to return multiple SVIs per unit. Defaults to False.
            sv_fov (int, optional): Field of view for street view. Defaults to 80.
            sv_pitch (int, optional): Pitch angle for street view. Defaults to 10.
            sv_size (list, tuple, optional): Size (height, width) for street view images. Defaults to (300, 400).
            year (list or tuple): The year ranges (e.g., (2018,2023)).
            season (str): 'spring', 'summer', 'fall', 'winter'.
            time_of_day (str): 'day' or 'night'.
            saveImg (bool, optional): Whether to save images (as base64 strings) in output. Defaults to True.
            output_gdf (bool, optional): Whether to return results as a GeoDataFrame. Defaults to False.
            disableProgressBar (bool, optional): Whether to show progress bar. Defaults to False.

        Returns:
            dict: A dictionary containing prompts, responses, and (optionally) image data for each unit.
        """

        self.__checkModel(model)
        self.preload_model(model)

        from tqdm import tqdm

        if type == 'top' and 'top' not in prompt:
            print("Please provide prompt for top view images when type='top'")
        if type == 'street' and 'street' not in prompt:
            print("Please provide prompt for street view images when type='street'")
        if type == 'both' and 'top' not in prompt and 'street' not in prompt:
            print("Please provide prompt for both top and street view images when type='both'")
        if (type == 'both' or type == 'street') and self.mapillary_key is None:
            print("API key is missing. The program will process with type='top'")

        dic = {
            "lon": [],
            "lat": [],
        }

        top_view_imgs = {'top_view_base64': []}
        street_view_imgs = {'street_view_base64': []}

        for i in tqdm(range(len(self.units)), desc="Processing...", ncols=75, disable=disableProgressBar):
            # Get the extent of one polygon from the filtered GeoDataFrame
            polygon = self.units.geometry.iloc[i]
            centroid = polygon.centroid

            dic['lon'].append(centroid.x)
            dic['lat'].append(centroid.y)

            # process street view image
            if (type == 'street' or type == 'both') and epsg != None and self.mapillary_key != None:
                input_svis = getSV(centroid, epsg, self.mapillary_key, multi=multi,
                                   fov=sv_fov, pitch=sv_pitch, height=sv_size[0], width=sv_size[1],
                                   year=year, season=season, time_of_day=time_of_day)

                if len(input_svis) != 0:
                    # save imgs
                    if saveImg:
                        street_view_imgs['street_view_base64'] += [input_svis]
                    # inference
                    res = self.LLM_chat(model=model,
                                        system=system,
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
                else:
                    dic['lon'].pop()
                    dic['lat'].pop()
                    continue

            # process aerial image
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
                    "count": 3
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
                top_res = self.LLM_chat(model=model,
                                        system=system,
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

        self.results = {'from_loopUnitChat': dic, 'base64_imgs': {**top_view_imgs, **street_view_imgs}}
        # reset message history
        if self.messageHistory != []:
            self.messageHistory = []
            print('Reset message history.')
        if output_gdf:
            return self.to_gdf(output=True)
        return dic

    def to_df(self, output: bool = True) -> pd.DataFrame | str:
        """
        Convert the output from an MLLM reponse (from .loopImgChat) into a DataFrame.

        Args:
            output (bool): Whether to return a DataFrame. Defaults to True.
        Returns:
            pd.DataFrame: A DataFrame containing responses and associated metadata.
            str: An error message if `.loopImgChat()` has not been run or if the format is unsupported.
        """

        if self.results is not None:
            if 'from_loopImgChat' in self.results:
                self.df = response2df(self.results['from_loopImgChat'])
                if output:
                    return self.df
            else:
                print("This method can only support the output of 'self.loopImgChat()' method")

    def to_gdf(self, output: bool = True) -> gpd.GeoDataFrame | str:
        """
        Convert the output from an MLLM response (from .loopUnitChat) into a GeoDataFrame.

        This method extracts coordinates, questions, responses, and base64-encoded input images
        from the stored `self.results` object, and formats them into a structured GeoDataFrame.

        Args:
            output (bool): Whether to return a GeoDataFrame. Defaults to True.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing spatial responses and associated metadata.
            str: An error message if `.loopUnitChat()` has not been run or if the format is unsupported.
        """

        import geopandas as gpd
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
                    self.geo_df = gpd.GeoDataFrame(pd.concat([res_df, imgs_df], axis=1), geometry="geometry")
                else:
                    self.geo_df = gpd.GeoDataFrame(res_df, geometry="geometry")
                if output:
                    return self.geo_df
            else:
                print("This method can only support the output of 'self.loopUnitChat()' method")
        else:
            print("This method can only be called after running the 'self.loopUnitChat()' method")

    def LLM_chat(self, model: str = 'gemma3:12b', system: str = None, prompt: str = None,
                 img: list[str] = None, temp: float = None, top_k: float = None, top_p: float = None) -> Union[
        "Response", list["QnA"]]:
        '''
        Chat with the LLM model with a list of images.
        
        Depending on the number of images provided, the method will:
        - Return a single Response object if only one image is provided.
        - Return a list of QnA objects if multiple images are provided (e.g., aerial and street views).

        Args:
            model (str): Model name.
            system (str): The system message guiding the LLM.
            prompt (str): The user prompt to the LLM.
            img (list[str]): A list of image paths.
            temp (float, optional): Temperature parameter for response randomness.
            top_k (float, optional): Top-K sampling filter.
            top_p (float, optional): Top-P (nucleus) sampling filter.

        Returns:
            Union[Response, list[QnA]]: A Response object if a single reply is generated,
            or a list of QnA objects for multi-turn/image-question responses.
        '''

        if prompt is not None and img is not None:
            if len(img) == 1:
                return self.chat(model, system, prompt, img[0], temp, top_k, top_p)
            elif len(img) == 3:
                res = []
                system = f'You are analyzing aerial or street view images. For street view, you should just focus on the building and yard in the middle. {system}'
                for i in range(len(img)):
                    r = self.chat(model, system, prompt, img[i], temp, top_k, top_p)
                    res += [r.responses]
                return res
        else:
            raise Exception("Prompt or image(s) is missing.")

    def chat(self, model: str = 'gemma3:12b', system: str = None, prompt: str = None,
             img=None, temp=None, top_k: float = None, top_p: float = None) -> Response:
        '''
        Chat with the LLM model using a system message, prompt, and optional image.

        Args:
            model (str): Model name. Defaults to "gemma3:12b". ['granite3.2-vision', 'llama3.2-vision', 'gemma3', 'gemma3:1b', 'gemma3:12b', 'minicpm-v', 'mistral-small3.1']
            system (str): The system-level instruction for the model.
            prompt (str): The user message or question.
            img (str): Path to a single image to be sent to the model.
            temp (float, optional): Sampling temperature for generation (higher = more random).
            top_k (float, optional): Top-k sampling parameter.
            top_p (float, optional): Top-p (nucleus) sampling parameter.

        Returns:
            Response: Parsed response from the LLM, returned as a `Response` object.
        '''
        if top_k > 100.0:
            top_k = 100.0
        elif top_k <= 0:
            top_k = 1.0

        if top_p > 1.0:
            top_p = 1.0
        elif top_p <= 0:
            top_p = 0

        res = ollama.chat(
            model=model,
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
                "temperature": temp,
                "top_k": top_k,
                "top_p": top_p
            }
        )
        return self.format.model_validate_json(res.message.content)

    def __summarize_geo_df(self, max_rows: int = 2) -> tuple[str, list[dict]]:
        """
        Summarize key characteristics of self.geo_df for LLM context.

        Args:
            max_rows (int): Number of sample rows to return.

        Returns:
            tuple[str, list]: (summary string, example row list)
        """
        import pandas as pd

        if self.geo_df is None or self.geo_df.empty:
            return "The dataset is empty.", []

        df = self.geo_df.copy()
        summary = []

        # Columns to exclude from summary (usually large/unnecessary for LLM)
        exclude_cols = ['geometry', 'top_view_base64', 'street_view_base64']
        non_geom_cols = [col for col in df.columns if col not in exclude_cols]

        # Basic dataset stats
        summary.append(f"- Number of spatial units: {len(df)}")

        # Bounding box
        bounds = df.total_bounds  # [minx, miny, maxx, maxy]
        summary.append(
            f"- Bounding box: lon [{bounds[0]:.4f}, {bounds[2]:.4f}], "
            f"lat [{bounds[1]:.4f}, {bounds[3]:.4f}]"
        )

        summary.append(f"- Number of data fields (excluding geometry and large fields): {len(non_geom_cols)}")
        summary.append(f"- Field names: {', '.join(non_geom_cols)}")

        # Sample rows
        example_rows = df[non_geom_cols].head(max_rows).to_dict(orient='records')
        for idx, row in enumerate(example_rows):
            summary.append(f"  Sample {idx + 1}: {row}")

        # Adaptive statistics for answer columns
        answer_cols = [col for col in df.columns if 'answer' in col.lower()]
        for col in answer_cols:
            if col in df.columns:
                series = df[col]
                col_type = pd.api.types.infer_dtype(series, skipna=True)

                summary.append(f"- Field '{col}' type: {col_type}")

                if pd.api.types.is_numeric_dtype(series):
                    summary.append(
                        f"  Value range: min={series.min():.2f}, max={series.max():.2f}, mean={series.mean():.2f}")
                elif pd.api.types.is_string_dtype(series) or pd.api.types.is_bool_dtype(series):
                    counts = series.astype(str).str.lower().value_counts()
                    formatted = ', '.join([f"{k}: {v}" for k, v in counts.items()])
                    summary.append(f"  Value distribution: {formatted}")
                else:
                    unique_vals = series.dropna().unique().tolist()
                    summary.append(f"  Unique values: {unique_vals[:5]}")

        # Q/A field pairing
        q_cols = [col for col in df.columns if 'question' in col.lower()]
        a_cols = [col for col in df.columns if 'answer' in col.lower()]
        qa_pairs = list(zip(q_cols, a_cols))
        if qa_pairs:
            summary.append("- Example Q&A Pairs:")
            for q, a in qa_pairs:
                if q in df.columns and a in df.columns:
                    q_sample = str(df[q].iloc[0])
                    a_sample = str(df[a].iloc[0])
                    summary.append(f"    * Q: '{q_sample}' → A: '{a_sample}'")

        return "\n".join(summary), example_rows

    def dataAnalyst(self,
                    prompt: str,
                    system: str = 'You are a spatial data analyst.',
                    model: str = 'gemma3') -> None:
        """
        Conversational spatial data analysis using a language model, with context-aware initialization.

        Args:
            prompt (str): User query related to spatial analysis.
            system (str): Base system prompt for the assistant.
            model (str): LLM model name to use.

        Returns:
            None
        """
        import copy

        self.preload_model(model)

        if self.messageHistory == []:
            if self.geo_df is None:
                print("Start to convert results to GeoDataFrame ...")
                self.to_gdf(output=False)

            # Clean up columns not relevant for reasoning
            data = copy.deepcopy(self.geo_df)
            for col in ['top_view_base64', 'street_view_base64']:
                if col in data.columns:
                    data.pop(col)

            # Generate natural language summary and samples
            summary_str, _ = self.__summarize_geo_df()

            user_prompt = f"""
            Please analyze and summarize the main patterns found in the answer columns of this dataset.
            Consider the value types (e.g., numeric or categorical), and also consider the relationship between question and answer fields when interpreting the values.

            Dataset summary:
            {summary_str}

            Use the information above to complete the analysis.
            """

            self.messageHistory += [
                {
                    'role': "system",
                    'content': system
                },
                {
                    'role': 'user',
                    'content': user_prompt.strip(),
                }
            ]

        conversations = chatpd(self.messageHistory, model)
        self.messageHistory = conversations

    def plotBase64(self, img: str):
        '''
        plot a single base64 image

        Args:
            img (str): image base64 string
        '''
        plot_base64_image(img)

    def export(self, out_type: str, file_name: str) -> None:
        '''
        Exports the result to a specified spatial data format.

        This method saves the spatial data stored in `self.geo_df` to a file in the specified format.
        If the GeoDataFrame is not yet initialized, it will attempt to convert the results first.

        Args:
            out_type (str): The output file format. 
                            Options include: 'geojson': Exports the data as a GeoJSON file;
                                            'shapefile' : Exports the data as an ESRI Shapefile.
                                            'geopackage': Exports the data as a GeoPackage (GPKG).

            file_name (str): The path and file name where the data will be saved. 
                            For shapefiles, provide a `.shp` file path.
                            For GeoJSON, use `.geojson`.
                            For GeoPackage, use `.gpkg`.
        Returns: 
            None
        '''
        if self.geo_df is None:
            print("Start to convert results to GeoDataFrame ...")
            self.to_gdf(output=False)
        if out_type == 'geojson':
            self.geo_df.to_file(file_name, driver='GeoJSON')
        elif out_type == 'shapefile':
            self.geo_df.to_file(out_type)
        elif out_type == 'seopackage':
            self.geo_df.to_file(file_name, layer='data', driver="GPKG")

    def plot_gdf(self, figsize=(12, 10), summary_func=None, show_table: bool = True):
        """
        Visualize all Q&A pairs from geo_df as separate maps with optional answer tables.

        - Automatically adjusts color scheme based on answer data type:
            * Numeric answers → gradient cmap (viridis)
            * Categorical answers (string/bool) → color-coded groups (case-insensitive)

        Args:
            figsize (tuple): Figure size.
            summary_func (callable): Function to reduce list-type fields (e.g., lambda x: x[0]).
            show_table (bool): Whether to include an answer table.
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        from pandas.plotting import table

        if self.geo_df is None:
            print("GeoDataFrame not available. Run .to_gdf() first.")
            return

        gdf = self.geo_df.to_crs(epsg=4326).copy().reset_index(drop=True)
        gdf["PointID"] = gdf.index + 1
        gdf_units = self.units.to_crs(epsg=4326) if self.units is not None else None

        q_cols = [col for col in gdf.columns if 'question' in col.lower()]
        a_cols = [col for col in gdf.columns if 'answer' in col.lower()]
        q_a_pairs = list(zip(q_cols, a_cols))

        if not q_a_pairs:
            print("No question/answer pairs found.")
            return

        for question_col, answer_col in q_a_pairs:
            df_plot = gdf.copy()

            # Reduce list answers if needed
            if summary_func and df_plot[answer_col].apply(lambda x: isinstance(x, list)).any():
                df_plot[answer_col] = df_plot[answer_col].apply(summary_func)

            answer_data = df_plot[answer_col]
            is_numeric = pd.api.types.is_numeric_dtype(answer_data)

            if is_numeric:
                color_kwargs = {'column': answer_col, 'cmap': 'viridis', 'legend': True}
            else:
                # Normalize to lowercase and use as color group
                df_plot["_answer_norm"] = answer_data.astype(str).str.lower()
                categories = df_plot["_answer_norm"].unique()
                cmap = plt.get_cmap('tab10')
                category_colors = {cat: cmap(i) for i, cat in enumerate(categories)}
                df_plot["_color"] = df_plot["_answer_norm"].map(category_colors)
                color_kwargs = {'color': df_plot["_color"]}

            # Figure and layout
            if show_table:
                fig, (ax_map, ax_table) = plt.subplots(1, 2, figsize=(figsize[0] * 1.6, figsize[1]))
            else:
                fig, ax_map = plt.subplots(figsize=figsize)

            if gdf_units is not None:
                gdf_units.plot(ax=ax_map, facecolor='#f0f0f0', edgecolor='black', linewidth=1)

            df_plot.plot(ax=ax_map, markersize=60, edgecolor='black', **color_kwargs)

            # Annotate point IDs
            for _, row in df_plot.iterrows():
                ax_map.annotate(str(row["PointID"]),
                                xy=(row.geometry.x, row.geometry.y),
                                xytext=(3, 3),
                                textcoords="offset points",
                                fontsize=9,
                                color='black')

            # Show legend for categorical values
            if not is_numeric:
                import matplotlib.patches as mpatches
                legend_handles = [mpatches.Patch(color=category_colors[cat], label=cat) for cat in categories]
                ax_map.legend(handles=legend_handles, title="Answer", loc='upper right', frameon=True)

            # Title and labels
            question_text = df_plot[question_col].iloc[0] if question_col in df_plot else "Question"
            ax_map.set_title(question_text, fontsize=14)
            ax_map.set_xlabel("Longitude", fontsize=12)
            ax_map.set_ylabel("Latitude", fontsize=12)
            ax_map.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
            ax_map.set_aspect('equal')

            # Answer table
            if show_table:
                ax_table.axis("off")
                table_df = df_plot[["PointID", answer_col]].copy()
                table_df.columns = ["ID", "Answer"]
                tbl = table(ax_table, table_df, loc="upper center", colWidths=[0.15, 0.3])
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(10)
                tbl.scale(1, 1.2)

            plt.tight_layout()
            plt.show()

