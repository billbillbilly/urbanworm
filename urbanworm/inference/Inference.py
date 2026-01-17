from __future__ import annotations
import logging
from ..dataset import GeoTaggedData

def _pack(locations, dataset):
    packed_data = []
    pack = []
    for i in range(len(locations)):
        if i == 0:
            pack += [dataset[i]]
        else:
            if locations[i] == locations[i - 1]:
                pack += [dataset[i]]
            else:
                packed_data += [pack]
                pack = [dataset[i]]
    return packed_data


class Inference:
    def __init__(self,
                 image: str|list|tuple = None,
                 images: list|tuple = None,
                 audio: str|list|tuple = None,
                 audios: list|tuple = None,
                 geo_tagged_data: GeoTaggedData = None,
                 schema: dict = None):
        '''
            Args:
                image (str | list | tuple): The image path.
                images (list | tuple): A list of image paths.
                audio (str | list | tuple): The audio path.
                audios (list | tuple): A list of audio paths.
                geo_tagged_data (GeoTaggedData): Data constructor.
                schema (dict): The response format.
        '''

        self.batch_images = self.batch_audios = self.batch_audios_slice = None
        self.img = image
        self.imgs = images
        self.audio = audio
        self.audios = audios
        self.geo_tagged_data = geo_tagged_data
        self.extract_from_geo_tagged_data()
        self.schema, self.results, self.df = None, None, None

        if schema is None:
            self.schema = {"questions": (str, ...),
                           "answer": (str, ...)}
        else:
            self.schema = schema

        self.logger = logging.getLogger("urbanworm")

    def extract_from_geo_tagged_data(self):
        if self.geo_tagged_data is not None:
            if self.geo_tagged_data.images is not None:
                if len(self.geo_tagged_data.images['path']) > 0:
                    self.batch_images = self.geo_tagged_data.images['path']
                else:
                    self.batch_images = self.geo_tagged_data.images['data']

            if len(self.geo_tagged_data.audios['path']) > 0:
                self.batch_audios = self.geo_tagged_data.audios['path']
            else:
                self.batch_audios = self.geo_tagged_data.audios['data']
            if 'slice' in self.geo_tagged_data.audios:
                self.batch_audios_slice = self.geo_tagged_data.audios['slice']
        else:
            self.batch_images = self.imgs
            self.batch_audios = self.audios

    def pack_by_location(self):
        if self.geo_tagged_data is not None:
            if self.batch_images is not None:
                locations = self.geo_tagged_data.images['loc_id']
                self.batch_images = _pack(locations, self.batch_images)
        if self.geo_tagged_data is not None:
            if self.batch_audios is not None:
                locations = self.geo_tagged_data.audios['loc_id']
                self.batch_audios = _pack(locations, self.batch_audios)