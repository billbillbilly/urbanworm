import os
import base64
import pandas as pd
from shapely.geometry import Point
from urbanworm.utils import getSV
from urbanworm.utils import calculate_bearing

# Mapillary API Key
MAPILLARY_KEY = ""

CSV_PATH = r"C:\Users\L1ght\OneDrive\Umich-School\Xiaohao-Project\Urban-Worm\ground_truth\groundtruth_1_labeled.csv"

TEST_INDEX = 251
NUM_TRIALS = 5

OUTPUT_DIR = "testGetSV"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_getsv_consistency():
    df = pd.read_csv(CSV_PATH)
    lon, lat = eval(df.loc[TEST_INDEX, "geometry_coordinates"])  # [lon, lat]
    print(f"\n=== Testing getSV for index {TEST_INDEX} at lon: {lon}, lat: {lat} ===")
    centroid = Point(lon, lat)

    for i in range(NUM_TRIALS):
        print(f"\nTrial {i + 1}/{NUM_TRIALS}")
        try:
            images = getSV(
                centroid=centroid,
                epsg=3857,
                key=MAPILLARY_KEY,
                multi=True,
                heading=None,
                pitch=10,
                fov=80,
                width=400,
                height=300
            )
            if not images:
                print("No image returned.")
            else:
                print(f"Returned {len(images)} image(s).")
                for j, img in enumerate(images):
                    out_path = os.path.join(
                        OUTPUT_DIR,
                        f"debug_sv_index{TEST_INDEX}_trial{i + 1}_img{j + 1}.jpg"
                    )
                    with open(out_path, "wb") as f:
                        f.write(base64.b64decode(img))
                    print(f"Saved image {j + 1} to: {out_path}")
        except Exception as e:
            print(f"Error on trial {i + 1}: {e}")

if __name__ == "__main__":
    test_getsv_consistency()
