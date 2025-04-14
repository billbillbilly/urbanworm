[![image](https://img.shields.io/pypi/v/urban-worm.svg)](https://pypi.python.org/pypi/urban-worm)
[![PyPI Downloads](https://static.pepy.tech/badge/urban-worm)](https://pepy.tech/project/urban-worm)
[![PyPI Downloads](https://static.pepy.tech/badge/urban-worm/week)](https://pepy.tech/projects/urban-worm)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://billbillbilly.github.io/urbanworm/)
[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/billbillbilly/urbanworm/blob/main/docs/example_osm.ipynb)

<picture>
  <img alt="logo" src="docs/images/urabn_worm_logo.png" width="100%">
</picture>

# Urban-Worm

## Introduction
Urban-Worm is a Python library that integrates remote sensing imagery, street view data, and vision-language models (VLMs) to assess urban units. Using APIs for data collection and VLMs for inference, Urban-Worm is designed to support the automation of the evaluation for urban environments, including roof integrity, structural condition, landscape quality, and urban perception.

- Free software: MIT license
- Website/Documentation: [https://land-info-lab.github.io/urbanworm/](https://land-info-lab.github.io/urbanworm/)

<picture>
  <img alt="workflow" src="docs/images/urabn_worm_diagram.png" width="100%">
</picture>

## Features
- Run VLMs locally with local datasets and ensure information privacy
- Download building footprints from OSM and global building data released by Bing Maps, with options to filter building footprints by area
- Search and clip aerial and street view images (via APIs) based on urban units such as parcel and building footprint data
- Automatically calibrate the orientation of the panorama street view and the extent of the aerial image
- Visualize results on maps and in tables
- Interact with LLMs through a streaming chat interface to analyze and interpret results

## Installation
#### install Ollama client
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

#### install GDAL first
For macOS, Linux, and Windows users, `gdal` may need to be installed at very begining using `conda`. Please download and install [Anaconda](https://www.anaconda.com/download/success) to use `conda`.

If the installation method above does not work, try to install with `conda`:
```sh
 conda install -c conda-forge gdal
```

Mac users may install `gdal` (if the installation method below does not work, try to install with conda):
```sh
 brew install gdal
```

#### install the package
The package urabnworm can be installed with `pip`:
```sh
pip install urban-worm 
```

To install the development version from this repo:
``` sh
pip install -e git+https://github.com/billbillbilly/urbanworm.git#egg=urban-worm
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
- [ ] One-shot learning in each chat method to help the model get familiar with the questions and expected answers 
- [ ] Multiple images inference for pairwise comparison and more
- [x] Basic plot method in UrbanDataSet class
- [x] Improve the method dataAnalyst in UrbanDataSet class by adding functionality of feeding a more meaningful introduction of data to LLMs
- [ ] A web UI providing interactive operation and data visualization 

The next version (v0.2.0) will have:
- [ ] agent-based city walk simulation
- [ ] search for a unit with an address (using Google APIs)
- [ ] find historical images (using Google APIs)

## Legal Notice
This repository and its content are provided for educational purposes only. By using the information and code provided, users acknowledge that they are using the APIs and models at their own risk and agree to comply with any applicable laws and regulations. Users who intend to download a large number of image tiles from any basemap are advised to contact the basemap provider to obtain permission before doing so. Unauthorized use of the basemap or any of its components may be a violation of copyright laws or other applicable laws and regulations.

## Acknowledgements
The package is heavily built on the Ollama client and Ollama-python. Credit goes to the developers of these projects.
- [ollama](https://github.com/ollama/ollama)
- [ollama-python](https://github.com/ollama/ollama-python)

The functionality about sourcing and processing GIS data (satellite & street view imagery) and 360-degree street view image processing is built on the following open projects. Credit goes to the developers of these projects.
- [tms2geotiff](https://github.com/gumblex/tms2geotiff)
- [GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints)
- [Mapillary API](https://www.mapillary.com/developer/api-documentation)
- [Equirec2Perspec](https://github.com/fuenwang/Equirec2Perspec)

The development of this package is supported and inspired by the city of Detroit.
