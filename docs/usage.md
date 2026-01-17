
To use urban-worm in a project:

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

data.schema = {
    "answer": (bool, ...),
    "explanation": (str, ...)
}

data.one_inference(system=system, prompt=prompt)
```