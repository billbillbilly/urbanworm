import os
import base64
from shapely.geometry import Point
from urbanworm.utils import getSV

# Mapillary API Key
MAPILLARY_KEY = ""

lon, lat = -83.19973659858849, 42.374458946845095
centroid = Point(lon, lat)

OUTPUT_DIR = "testGetSV"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_getsv_focus_on_house():
    print(f"\n=== Testing focused getSV at fixed coordinates ===")
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
                    f"focused_sv_fixed_img{j + 1}.jpg"
                )
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(img))
                print(f"Saved focused image {j + 1} to: {out_path}")
    except Exception as e:
        print(f"Error in focused getSV test: {e}")

def test_getsv_pitch():
    print(f"\n=== Testing getSV with pitch at fixed coordinates ===")
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
            print(f"Returned {len(images)} pitch image(s).")
            for j, img in enumerate(images):
                out_path = os.path.join(
                    OUTPUT_DIR,
                    f"pitch_sv_fixed_img{j + 1}.jpg"
                )
                with open(out_path, "wb") as f:
                    f.write(base64.b64decode(img))
                print(f"Saved pitch image {j + 1} to: {out_path}")
    except Exception as e:
        print(f"Error in getSV pitch test: {e}")

if __name__ == "__main__":
    test_getsv_focus_on_house()
    test_getsv_pitch()
