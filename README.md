[![image](https://img.shields.io/pypi/v/urban-worm.svg)](https://pypi.python.org/pypi/urban-worm)
[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/billbillbilly/urbanworm/blob/main/docs/example_osm.ipynb)

<picture>
  <img alt="logo" src="docs/images/urabn_worm_logo.jpg" width="100%">
</picture>

# Urban-Worm

## Introduction
Urban-Worm is a Python library that integrates remote sensing imagery, street view data, and multimodal model to assess urban units. Using APIs for data collection and Llama 3.2 vision for inference, Urban-Worm is designed to support the automation of the evaluation for urban environments, including roof integrity, structural condition, landscape quality, and urban perception.

- Free software: MIT license
- [Documentation](https://billbillbilly.github.io/urbanworm/)

<picture>
  <img alt="workflow" src="docs/images/urabn_worm_diagram.jpg" width="100%">
</picture>

## Features
- run Llama 3.2 vision locally with local datasets and remain information privacy
- download building footprints from OSM and global building released by Bing map 
- search and clip aerial and street view images (via APIs) based on urban units such as parcel and building footprint data
- automatically calibrate the oritation of panorama street view and the extent of aerial image

## Installation
Please make sure [Ollama](https://ollama.com/) is installed before installing urban-worm

For Linux, users can also install ollama by running in the terminal:
```sh
curl -fsSL https://ollama.com/install.sh | sh
```
For MacOS, users can also install ollama using `brew`:
```sh
brew install ollama
```
To install `brew`, run in the terminal:
```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Windows users should directly install the [Ollama client](https://ollama.com/)

#### install gdal first
For macOS, Linux, and Windows users, `gdal` may need to be installed at very begining using `conda`. Please download and install [Anaconda](https://www.anaconda.com/download/success) to use `conda`.

If the installation method above does not work, try to install with `conda`:
```sh
 conda install -c conda-forge gdal
```

Mac users may install `gdal` (if the installation method below does not work, try to install with conda):
```sh
 brew install gdal
```

The package urabnworm can be installed with `pip`:
```sh
pip install urban-worm 
```

## Usage
#### single-image inference
```python
from urbanworm import UrbanDataSet

data = UrbanDataSet(image = '../docs/data/test1.jpg')
system = '''
    Given a top view image, you are going to roughly estimate house conditions. Your answer should be based only on your observation. 
    The format of your response must include question, answer (yes or no), explanation (within 50 words)
'''
prompt = '''
    Is there any damage on the roof?
'''
data.oneImgChat(system=system, prompt=prompt)
# output:
# {'question': 'Is there any damage on the roof?',
#  'answer': 'no',
#  'explanation': 'No visible signs of damage or wear on the roof',
#  'img': '/9j/4AAQSkZ...'}
```

#### multiple (aerial & street view) images inference using OSM data
```python
bbox = (-83.235572,42.348092,-83.235154,42.348806)
data = UrbanDataSet()
data.bbox2Buildings(bbox)

system = '''
    Given a top view image or street view images, you are going to roughly estimate house conditions. 
    Your answer should be based only on your observation. 
    The format of your response must include question, answer (yes or no), explanation (within 50 words) for each question.
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
# use both the aerial and street view images (with type='both')
data.loopUnitChat(system=system, prompt=prompt, type='both', epsg=2253)
# convert results into GeoDataframe
data.to_gdf()
```

More examples can be found [here](docs/example.ipynb).

## To do
The next version will have functionalities using Google APIs:
- search for units with address
- find historical images

An agent webUI is in the incoming development plan for providing interactive operation and data visualization. 

## Legal Notice
This repository and its content are provided for educational purposes only. By using the information and code provided, users acknowledge that they are using the APIs and models at their own risk and agree to comply with any applicable laws and regulations. Users who intend to download a large number of image tiles from any basemap are advised to contact the basemap provider to obtain permission before doing so. Unauthorized use of the basemap or any of its components may be a violation of copyright laws or other applicable laws and regulations.

## Acknowledgements
The package is heavily built on Ollama client, Ollama-python, and Llama 3.2 Vision. Credit goes to the developers of these projects.
- [ollama](https://github.com/ollama/ollama)
- [ollama-python](https://github.com/ollama/ollama-python)
- [structured outputs](https://ollama.com/blog/structured-outputs)
- [llama 3.2 vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)

The functionality about sourcing and processing GIS data (satellite & street view imagery) is built on the following open projects. Credit goes to the developers of these projects.
- [tms2geotiff](https://github.com/gumblex/tms2geotiff)
- [GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints)
- [Mapillary API](https://www.mapillary.com/developer/api-documentation)

The development of this package is supported and inspired by the city of Detroit.
