# tests/test_response2df.py

from urbanworm.utils import response2df, response2gdf
from shapely.geometry import Point

class Dummy:
    def __init__(self, question, answer, explanation):
        self.question = question
        self.answer = answer
        self.explanation = explanation

def main():
    print("=== Test: response2df ===")

    image_paths = [
        'docs/data/test1.jpg',
        'docs/data/test2.jpg',
        'docs/data/test3.jpg'
    ]

    qna_dict = {
        'responses': [
            [Dummy("Is roof damaged?", "yes", "Looks okay.")],
            [Dummy("Is roof damaged?", "no", "Roof appears damaged.")],
            [Dummy("Is roof damaged?", "yes", "No visible damage.")]
        ],
        'img': image_paths,
        'imgBase64': ['b64_1', 'b64_2', 'b64_3']
    }

    df = response2df(qna_dict)
    print(df)

    print("\n=== Test: response2gdf ===")

    geo_qna_dict = {
        'top_view': qna_dict['responses'],
        'lon': [-82.9253, -82.9254, -82.9255],
        'lat': [42.4418, 42.4419, 42.4420]
    }

    gdf = response2gdf(geo_qna_dict)
    print(gdf)

if __name__ == "__main__":
    main()
