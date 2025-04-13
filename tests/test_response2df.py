# tests/test_response2df.py
from urbanworm.utils import response2df

class Dummy:
    def __init__(self, question, answer, explanation):
        self.question = question
        self.answer = answer
        self.explanation = explanation

image_paths = [
    'docs/data/test1.jpg',
    'docs/data/test2.jpg',
    'docs/data/test3.jpg'
]

qna_dict = {
    'responses': [
        [Dummy("Is roof damaged?", True, "Looks okay.")],
        [Dummy("Is roof damaged?", False, "Roof appears damaged.")],
        [Dummy("Is roof damaged?", True, "No visible damage.")]
    ],
    'img': image_paths,
    'imgBase64': ['b64_1', 'b64_2', 'b64_3']  # 这里用 dummy base64 数据
}

df = response2df(qna_dict)
print(df)
