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
- Run vision-language models locally using custom datasets while maintaining data privacy
- Download building footprints from OpenStreetMap (OSM) and global building datasets released by Bing Maps
- Search for (via APIs) and clip aerial or street view imagery, based on urban units such as parcels or building footprints
- Automatically calibrate the orientation of panoramic street views and define the spatial extent of aerial imagery
- Interact with LLMs through a streaming chat interface to analyze and interpret results

## Computing Resources
Large language models (LLMs) and vision-language models (VLMs) typically require substantial RAM and can be quite slow when running solely on a CPU. While macOS devices with Apple Silicon can accelerate computation using Metal, performance is still not on par with high-end NVIDIA GPUs, which are strongly recommended for optimal speed and efficiency. This package is built on Ollama, which provides fast performance—especially for local use with quantized models. However, the number of parameters in a model directly affects VRAM requirements. More powerful models need more VRAM to run entirely on the GPU. For instance, the Llama3.2-Vision model with 11 billion parameters requires approximately 10 GB of GPU memory. Additionally, image resolution plays a crucial role in inference time—particularly for high-resolution inputs like satellite imagery or street views. The higher the resolution, the longer a VLM may take to process the image.

## Legal Notice
This repository and its content are provided for educational purposes only. By using the information and code provided, users acknowledge that they are using the APIs and models at their own risk and agree to comply with any applicable laws and regulations. Users who intend to download a large number of image tiles from any basemap are advised to contact the basemap provider to obtain permission before doing so. Unauthorized use of the basemap or any of its components may be a violation of copyright laws or other applicable laws and regulations.

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