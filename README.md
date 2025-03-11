# Urban-Worm

## Introduction
Urban-Worm is a Python library that integrates remote sensing imagery, street view data, and multimodal model to assess urban units with precision. Using Llama 3.2 vision and a specialized API for data collection, Urban-Worm is designed to support the automation of the evaluation for urban environments, including roof integrity, structural condition, landscape quality, and urban perception.

## Features
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
```python
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

## Reference
Wu et al., (2023). samgeo: A Python package for segmenting geospatial data with the Segment Anything Model (SAM). Journal of Open Source Software, 8(89), 5663, https://doi.org/10.21105/joss.05663

Structured outputs https://ollama.com/blog/structured-outputs