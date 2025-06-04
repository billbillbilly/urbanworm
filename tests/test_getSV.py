import os
import base64
import pandas as pd
from shapely.geometry import Point
from urbanworm.utils import getSV
from urbanworm.utils import calculate_bearing

# Mapillary API Key
MAPILLARY_KEY = ""

CSV_PATH = r"C:\Users\L1ght\OneDrive\Umich-School\Xiaohao-Project\Urban-Worm\ground_truth\groundtruth_1_labeled.csv"

TEST_INDEX = 155
NUM_TRIALS = 1

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
                multi=True
            )
            if not images:
                print("No image returned.")
            else:
                print(f"Returned {len(images)} image(s).")
                for j, img in enumerate(images):
                    out_path = os.path.join(
                        OUTPUT_DIR,
                        f"original_sv_index{TEST_INDEX}_trial{i + 1}_img{j + 1}.jpg"
                    )
                    with open(out_path, "wb") as f:
                        f.write(base64.b64decode(img))
                    print(f"Saved image {j + 1} to: {out_path}")
        except Exception as e:
            print(f"Error on trial {i + 1}: {e}")

def test_getsv_focus_on_house():
    df = pd.read_csv(CSV_PATH)
    lon, lat = eval(df.loc[TEST_INDEX, "geometry_coordinates"])
    centroid = Point(lon, lat)

    print(f"\n=== Testing focused getSV at index {TEST_INDEX} ===")
    try:
        images = getSV(
            centroid=centroid,
            epsg=3857,
            key=MAPILLARY_KEY,
            multi=True,
            heading=None,
            pitch=5,
            fov=45,
            width=640,
            height=480
        )
        if not images:
            print("No image returned.")
        else:
            print(f"Returned {len(images)} focused image(s).")
            for j, img in enumerate(images):
                out_path = os.path.join(
                    OUTPUT_DIR,
                    f"focused_sv_index{TEST_INDEX}_img{j + 1}.jpg"
                )
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(img))
                print(f"Saved focused image {j + 1} to: {out_path}")
    except Exception as e:
        print(f"Error in focused getSV test: {e}")

def test_getsv_pitch():
    df = pd.read_csv(CSV_PATH)
    lon, lat = eval(df.loc[TEST_INDEX, "geometry_coordinates"])
    centroid = Point(lon, lat)

    print(f"\n=== Testing focused getSV at index {TEST_INDEX} ===")
    try:
        images = getSV(
            centroid=centroid,
            epsg=3857,
            key=MAPILLARY_KEY,
            multi=True,
            heading=None,
            pitch=5,
            fov=30,
            width=640,
            height=480
        )
        if not images:
            print("No image returned.")
        else:
            print(f"Returned {len(images)} focused image(s).")
            for j, img in enumerate(images):
                out_path = os.path.join(
                    OUTPUT_DIR,
                    f"pitch_sv_index{TEST_INDEX}_img{j + 1}.jpg"
                )
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(img))
                print(f"Saved focused image {j + 1} to: {out_path}")
    except Exception as e:
        print(f"Error in focused getSV test: {e}")

if __name__ == "__main__":
    # test_getsv_consistency()
    test_getsv_focus_on_house()
    test_getsv_pitch()
