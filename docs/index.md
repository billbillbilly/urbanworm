# Welcome to Urban-Worm

<picture>
  <img alt="logo" src="https://github.com/billbillbilly/urbanworm/blob/main/docs/images/urabn_worm_logo.png?raw=true" width="100%">
</picture>

[![image](https://img.shields.io/pypi/v/urban-worm.svg)](https://pypi.python.org/pypi/urban-worm)
[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/billbillbilly/urbanworm/blob/main/docs/example_colab.ipynb)

## Introduction
Urban-**WORM** (**W**orkflow **O**f **R**eproducible **M**ultimodal Inference) is a user-friendly high-level interface that 
is designed for adding rich and meaningful captions for crowdsourced data with geotags using multimodal models (MMs). 
Urban-WORM can support the batched analysis of images and sounds for investigating urban environments at scales. 
The investigation may cover topics about building conditions, street appearance, people's activities, etc.

<picture>
  <img alt="logo" src="https://github.com/billbillbilly/urbanworm/blob/main/docs/images/urabn_worm_diagram.png?raw=true" width="100%">
</picture>

## Features
- Collect geotagged data (Mapillary street views, Flickr photos, and Freesound audios) via APIs 
within the proximity of building footprints (or other POIs)
- Calibrate the orientation of the panorama street views to look at given locations
- Filter out personal photo using face detection
- Divide sound recording to multiple clips with given duration
- Support (batched) multiple data input with multimodal models

## Computing Resources
MMs typically require substantial RAM and can be quite slow when running solely on a CPU. While macOS devices with Apple 
Silicon can accelerate computation using Metal, performance is still not on par with high-end NVIDIA GPUs, which are 
strongly recommended for optimal speed and efficiency. This package is built on Ollama, which provides fast performance—especially 
for local use with quantized models. However, the number of parameters in a model directly affects VRAM requirements. 
More powerful models need more VRAM to run entirely on the GPU. For instance, the Llama3.2-Vision model with 11 billion 
parameters requires approximately 10 GB of GPU memory. Additionally, image resolution plays a crucial role in inference 
time—particularly for high-resolution inputs like satellite imagery or street views. The higher the resolution, the longer 
a VLM may take to process the image.

## Legal Notice
This repository and its content are provided for educational purposes only. By using the information and code provided, 
users acknowledge that they are using the APIs and models at their own risk and agree to comply with any applicable laws 
and regulations. 

## Acknowledgements
The package is heavily built on llamacpp and Ollama. Credit goes to the developers of these projects.

- [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master)
- [ollama](https://github.com/ollama/ollama)
- [ollama-python](https://github.com/ollama/ollama-python)


The functionality about sourcing and processing GIS data and image processing is built on the following open projects. 
Credit goes to the developers of these projects.

- [GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints)
- [Equirec2Perspec](https://github.com/fuenwang/Equirec2Perspec)
- [Mapillary API](https://www.mapillary.com/developer/api-documentation)
- [Flickr API](https://www.flickr.com/services/api/)
- [Freesound API](https://freesound.org/apiv2/apply)

The development of this package is supported and inspired by the city of Detroit.
