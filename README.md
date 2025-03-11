# Urban Worm

## Introduction
Urban-Worm is a Python library that integrates remote sensing imagery, street view data, and multimodal model to assess urban units with precision. Using Llama 3.2 vision and a specialized API for data collection, Urban-Worm is designed to support the automation of the evaluation for urban environments, including roof integrity, structural condition, landscape quality, and urban perception.

## features
- search and clip aerial and street view images using urban units such as parcel and building footprint data
- stream street view images via API 
- automatically calibrate the oritation of panorama street view 

## Installation
Please make sure [Ollama](https://ollama.com/) is installed before installing urban-worm

```sh
conda create -n urbanworm python==3.10
conda activate urbanworm
pip install -r requirements.txt 
```

## Usage


## Reference
Wu et al., (2023). samgeo: A Python package for segmenting geospatial data with the Segment Anything Model (SAM). Journal of Open Source Software, 8(89), 5663, https://doi.org/10.21105/joss.05663

https://ollama.com/blog/structured-outputs