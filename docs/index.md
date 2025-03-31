# Welcome to Urban-Worm

<picture>
  <img alt="logo" src="https://github.com/billbillbilly/urbanworm/blob/main/docs/images/urabn_worm_logo.png?raw=true" width="100%">
</picture>

[![image](https://img.shields.io/pypi/v/urban-worm.svg)](https://pypi.python.org/pypi/urban-worm)

A python package for studying urban environment imagery with Llama vison model

## Introduction
Urban-Worm is a Python library that integrates remote sensing imagery, street view data, and multimodal model to assess urban units. Using APIs for data collection and vision-language models for inference, Urban-Worm is designed to support the automation of the evaluation for urban environments, including roof integrity, structural condition, landscape quality, and urban perception.

<picture>
  <img alt="logo" src="https://github.com/billbillbilly/urbanworm/blob/main/docs/images/urabn_worm_diagram.png?raw=true" width="100%">
</picture>

## Features
- run vision-language models locally with local datasets and remain information privacy
- download building footprints from OSM and global building released by Bing map 
- search and clip aerial and street view images (via APIs) based on urban units such as parcel and building footprint data
- automatically calibrate the oritation of panorama street view and the extent of aerial image
- stream chat with LLMs to analyze results

## Acknowledgements
The package is heavily built on Ollama client and Ollama-python. Credit goes to the developers of these projects.

- [ollama](https://github.com/ollama/ollama)
- [ollama-python](https://github.com/ollama/ollama-python)
- [structured outputs](https://ollama.com/blog/structured-outputs)

The functionality about sourcing and processing GIS data (satellite & street view imagery) and 360-degree street view image processing is built on the following open projects. Credit goes to the developers of these projects.

- [tms2geotiff](https://github.com/gumblex/tms2geotiff)
- [GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints)
- [Mapillary API](https://www.mapillary.com/developer/api-documentation)
- [Equirec2Perspec](https://github.com/fuenwang/Equirec2Perspec)

The development of this package is supported and inspired by the city of Detroit.