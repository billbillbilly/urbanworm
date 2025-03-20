
To use urban-worm in a project:
```python
from urbanworm import UrbanDataSet
```

## single-image
```python
# load a local image
data = UrbanDataSet(image = './docs/data/test1.jpg')
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

## multiple images using OSM data and Mapillary API
Get building footprints as units and collect satellite and street view images based on each unit. Finally, chat with MLLM model for each unit based on collected images.

To get a token/key to access data via mapillary api, please [create an acount and apply](https://www.mapillary.com/dashboard/developers) on Mapillary

```python
bbox = (-83.235572,42.348092,-83.235154,42.348806)
data = UrbanDataSet()
data.bbox2Buildings(bbox, source='osm')

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