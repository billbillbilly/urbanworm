[![image](https://img.shields.io/pypi/v/urban-worm.svg)](https://pypi.python.org/pypi/urban-worm)
[![PyPI Downloads](https://static.pepy.tech/badge/urban-worm)](https://pepy.tech/project/urban-worm)
[![PyPI Downloads](https://static.pepy.tech/badge/urban-worm/week)](https://pepy.tech/projects/urban-worm)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://billbillbilly.github.io/urbanworm/)
[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/billbillbilly/urbanworm/blob/main/docs/example_colab.ipynb)

<picture>
  <img alt="logo" src="docs/images/urabn_worm_logo.png" width="100%">
</picture>

# Urban-WORM

## Introduction
Urban-**WORM** (**W**orkflow **O**f **R**eproducible **M**ultimodal Inference) is a user-friendly high-level interface that 
is designed for adding rich and meaningful captions for crowdsourced data with geotags using multimodal models. 
Urban-WORM can support the batched analysis of images and sounds for investigating urban environments at scales. 
The investigation may cover topics about building conditions, street appearance, people's activities, etc.

- Free software: MIT license
- Website/Documentation: [https://digital-landscapes.github.io/urbanworm/](https://land-info-lab.github.io/urbanworm/)

<picture>
  <img alt="workflow" src="docs/images/urabn_worm_diagram.png" width="90%">
</picture>

## Features
- Collect geotagged data (Mapillary street views, Flickr photos, and Freesound audios) via APIs 
within the proximity of building footprints (or other POIs)
- Calibrate the orientation of the panorama street views to look at given locations
- Filter out personal photo using face detection
- Divide sound recording to multiple clips with given duration
- Support (batched) multiple data input with multimodal models

## Installation

### 1 install the package
The package `urban-worm` can be installed with `pip`:
```sh
pip install urban-worm
```

### 2 Inference with llama.cpp
To run more pre-quantized models with vision capabilities, please install pre-built version of llama.cpp:
``` sh
# Windows
winget install llama.cpp

# Mac and Linux
brew install llama.cpp
```
More information about the installation 
[here](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)

More GGUF models can be found at the Hugging Face pages 
[here](https://huggingface.co/collections/ggml-org/multimodal-ggufs-68244e01ff1f39e5bebeeedc) and [here](https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending&search=gguf)

### 3 Inference with Ollama client

Please make sure [Ollama](https://ollama.com/) is installed before using urban-worm if you plan to rely on Ollama

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

To install the development version from this repo:
``` sh
pip install -e git+https://github.com/billbillbilly/urbanworm.git#egg=urban-worm
```

## Usage

```python
from urbanworm.inference.llama import InferenceOllama

data = InferenceOllama(image = 'docs/data/img_1.jpg')
system = '''
    Your answer should be based only on your observation. 
    The format of your response must include answer (yes/True or no/False), explanation (within 50 words)
'''
prompt = '''
    Is there a tree?
'''

data.llm = "hf.co/ggml-org/InternVL3-8B-Instruct-GGUF:Q8_0"
data.schema = {
    "answer": (bool, ...),
    "explanation": (str, ...)
}
data.one_inference(system=system, prompt=prompt)
```
More examples can be found [here](docs/1_basic_inference.ipynb).

## To do
v0.1.x:
- [x] A module for collecting social media data (Flickr and Freesound)
- [x] A method for inferencing sound recordings

v0.2.x:
- [ ] A web UI providing interactive operation and data visualization

## Legal Notice
This repository and its content are provided for educational and research purposes only. By using the information and 
code provided, users acknowledge that they are using the APIs and models at their own risk and agree to comply with any 
applicable laws and regulations. 

## Acknowledgements
The package is heavily built on llama.cpp and Ollama. Credit goes to the developers of these projects.
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
