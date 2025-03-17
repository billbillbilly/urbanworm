# Urban-Worm

## Introduction
Urban-Worm is a Python library that integrates remote sensing imagery, street view data, and multimodal model to assess urban units with precision. Using Llama 3.2 vision and a specialized API for data collection, Urban-Worm is designed to support the automation of the evaluation for urban environments, including roof integrity, structural condition, landscape quality, and urban perception.

<picture>
  <img alt="workflow" src="docs/images/urabn_worm_diagram" width="100%">
</picture>

## Features
- download building footprints from OSM and global building released by Bing map 
- search and clip aerial and street view images using urban units such as parcel and building footprint data
- stream street view images via APIs 
- automatically calibrate the oritation of panorama street view 

## Installation
Please make sure [Ollama](https://ollama.com/) is installed before installing urban-worm
For Linux, users can install ollama:
```sh
curl -fsSL https://ollama.com/install.sh | sh
```

#### MacOS
For mac users, it may be nacessary to install gdal:
```sh
 brew install gdal
```
If the installation method above does not work, try to install with conda if you have:
```sh
 conda install -c conda-forge gdal
```

The package urabnworm can be installed with `pip`:
```sh
pip install urbanworm 
```

## Usage
#### single-image inference
```python
from urbanworm import UrbanDataSet

data = UrbanDataSet(image = '../docs/data/test1.jpg')
system = '''
    Given a top view image, you are going to roughly estimate house conditions. Your answer should be based only on your observation. 
    The format of your response must include question, answer (yes or no), explaination (within 50 words)
'''
prompt = '''
    Is there any damage on the roof?
'''
res = data.oneImgChat(system=system, prompt=prompt)
```

#### multiple (aerial + street view) images inference using OSM data
```python
bbox = (-83.235572,42.348092,-83.235154,42.348806)
data = UrbanDataSet()
data.bbox2osmBuildings(bbox)

system = '''
    Given a top view image or street view images, you are going to roughly estimate house conditions. 
    Your answer should be based only on your observation. 
    The format of your response must include question, answer (yes or no), explaination (within 50 words) for each question.
'''

prompt = {
    'top': '''
        Is there any damage on the roof?
    ''',
    'street': '''
        Is the wall missing or damaged?
        Is the yard maintained well?
    '''
}

# add the Mapillary key
data.mapillary_key = 'MLY|......'
# inspect both the aerial and street view images (with type='both')
res = data.loopUnitChat(system=system, prompt=prompt, type='both', epsg=2253)
```



## Reference
Wu et al., (2023). samgeo: A Python package for segmenting geospatial data with the Segment Anything Model (SAM). Journal of Open Source Software, 8(89), 5663, https://doi.org/10.21105/joss.05663

Structured outputs https://ollama.com/blog/structured-outputs