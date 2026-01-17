from __future__ import annotations
import io
import urllib
from urllib.parse import urlparse
import pandas as pd
import numpy as np
from pyproj import Transformer
import math
import requests
import os
import base64
import cv2
from datetime import datetime
import tempfile
import json

def is_url(url:str) -> bool:
    try:
        result = urlparse(url)
        # Check if both scheme and network location exist
        return all([result.scheme, result.netloc])
    except:
        return False

def is_base64(s):
    """Checks if a string is base64 encoded."""
    import io
    from PIL import Image
    try:
        # Decode Base64
        decoded_data = base64.b64decode(s, validate=True)
        # Verify it's an image
        image = Image.open(io.BytesIO(decoded_data))
        image.verify()
        return True
    except:
        return False


def is_image_path(s):
    """Checks if a string is a valid path and if a file exists at that path, and if it is an image."""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
    return os.path.isfile(s) and s.lower().endswith(image_extensions)


def detect_input_type(input_string):
    """Detects if the input string is an image path or base64 encoded."""
    if is_image_path(input_string):
        return "image_path"
    elif is_base64(input_string):
        return "base64"
    else:
        return "unknown"


def encode_image_to_base64(image_path):
    """Encodes an image file to a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# -------------------- street view utils ---------------------
# offset polygon by distance
def meters_to_degrees(meters, latitude):
    """Convert meters to degrees dynamically based on latitude."""
    # Approximate adjustment
    meters_per_degree = 111320 * (1 - 0.000022 * abs(latitude))
    return meters / meters_per_degree


# # Generate bbox based on a centroid and a radius
def projection(centroid, r):
    x, y, utm_epsg = degree2dis(centroid)
    # set search distance to 50 meters
    # set bbox
    x_min = x - r
    y_min = y - r
    x_max = x + r
    y_max = y + r
    # Convert to EPSG:4326 (Lat/Lon)
    x_min, y_min = dis2degree(x_min, y_min, utm_epsg)
    x_max, y_max = dis2degree(x_max, y_max, utm_epsg)
    return f'{x_min},{y_min},{x_max},{y_max}'

def retry_request(url, retries=3):
    response = None
    for _ in range(retries):
        # Check for rate limit or server error
        try:
            response = requests.get(url)
            # If the response status code is in the list, wait and retry
            if response.status_code != 200:
                continue
            else:
                return response
        except:
            pass
    return response

# --- UTM -> degrees (WGS84) ---
def dis2degree(ptx, pty, utm_epsg):
    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(ptx, pty)
    return lon, lat

# Convert degree to distance
def degree2dis(pt):
    ptx, pty = pt
    utm_epsg = lonlat_to_utm_epsg(ptx, pty)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    x, y = transformer.transform(ptx, pty)
    return x, y, utm_epsg

def lonlat_to_utm_epsg(lon: float, lat: float) -> int:
    """
    Compute the UTM EPSG code (WGS84) from longitude / latitude.
    """
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone  # UTM zone in northern hemisphere
    else:
        return 32700 + zone  # UTM zone in southern hemisphere


# find the closest image to the house
def closest(location=None,
            response=None,
            multi_num=None,
            interval=None,
            year=None,
            season=None,
            time_of_day=None,
            key=None):
    res_df = pd.DataFrame(response['data'])
    if res_df.empty:
        return None
    # extract info
    res_df = _extract_info(res_df)
    # filter by time: year/season/time of day
    year_start, year_end, season_, day_start, day_end = get_capture_time_range(year, season, time_of_day)

    try:
        if year_start is not None:
            res_df = res_df[(res_df['year'] >= year_start) & (res_df['year'] <= year_end)]
        if season_ is not None:
            season_start, season_end = season_[season.lower()]
            res_df1 = res_df[(res_df['month'] == season_start[0]) & (res_df['day'] >= season_start[1])]
            res_df2 = res_df[(res_df['month'] == season_end[0]) & (res_df['day'] <= season_end[1])]
            if 'winter' in season_:
                res_df3 = res_df[(res_df['month'] <= 2)]

            if 'summer' in season_:
                res_df3 = res_df[(res_df['month'] > 6) & (res_df['month'] < 9)]

            if 'fall' in season_:
                res_df3 = res_df[(res_df['month'] > 9) & (res_df['month'] < 12)]

            if 'spring' in season_:
                res_df3 = res_df[(res_df['month'] > 3) & (res_df['month'] < 6)]

            res_df = pd.concat([res_df1, res_df2])
            res_df = pd.concat([res_df, res_df3])
        if day_start is not None:
            res_df = res_df[(res_df['hour'] >= day_start) & (res_df['hour'] <= day_end)]

        if len(res_df) == 0:
            return None
    except Exception as e:
        print(f"Error in filtering street views by time: {e}")
        return None

    # when year is None, select the street views captured in the latest year
    if year is None:
        # sort by year
        res_df_ = res_df[res_df['year'] == res_df['year'].max()]
        if multi_num is not None:
            yearList = sorted(list(set(res_df.sort_values(by='year')['year'].to_list())), reverse=True)
            count = 1
            while len(res_df_) < 3:
                res_df_ = res_df[res_df['year'] >= yearList[count]]
                count += 1
        res_df = res_df_

    # sort by distance
    id_array = np.array(res_df['id'])
    lon_array = np.array(res_df['lon'])
    lat_array = np.array(res_df['lat'])
    dis_array = (lon_array - location[0]) * (lon_array - location[0]) + (lat_array - location[1]) * (lat_array - location[1])
    # find the closest one
    ind = np.where(dis_array == np.min(dis_array))[0]
    id_ = id_array[ind][0]
    closest_df = res_df.loc[res_df['id'] == id_]
    # get the id and sequence id of the closest image
    sq = get_sequence(closest_df['sequence'].iloc[0], key)

    if multi_num is not None and len(sq) >= multi_num > 1:
        id1, id2 = None, None
        multi_num = min(int(multi_num), 3)

        index_of_closest = sq.index(id_)
        n = len(sq)

        # within valid range: we can take both sides (or at least left side)
        if (index_of_closest - interval) >= 0 and (index_of_closest + interval) < n:
            id1 = sq[index_of_closest - interval]
            if multi_num > 2:
                id2 = sq[index_of_closest + interval]

        # too close to the end: take from the left
        elif (index_of_closest + interval) >= n:
            # shrink interval until indices are valid (but never below 1)
            if multi_num > 2:
                while interval > 1 and (index_of_closest - 2 * interval) < 0:
                    interval -= 1
            else:
                while interval > 1 and (index_of_closest - interval) < 0:
                    interval -= 1

            # now assign (guard again just in case)
            if (index_of_closest - interval) >= 0:
                id1 = sq[index_of_closest - interval]
            if multi_num > 2 and (index_of_closest - 2 * interval) >= 0:
                id2 = sq[index_of_closest - 2 * interval]

        # at (or near) the beginning: take from the right
        elif index_of_closest <= 0:
            if multi_num > 2:
                while interval > 1 and (index_of_closest + 2 * interval) >= n:
                    interval -= 1
            else:
                while interval > 1 and (index_of_closest + interval) >= n:
                    interval -= 1

            if (index_of_closest + interval) < n:
                id1 = sq[index_of_closest + interval]
            if multi_num > 2 and (index_of_closest + 2 * interval) < n:
                id2 = sq[index_of_closest + 2 * interval]

        closest_0 = get_svi_from_id(id_, key)
        closest_0 = _extract_info(closest_0, with_geometry=False)
        closest_1 = get_svi_from_id(id1, key)
        closest_1 = _extract_info(closest_1, with_geometry=False)

        if id2 is not None:
            closest_2 = get_svi_from_id(id2, key)
            closest_2 = _extract_info(closest_2, with_geometry=False)
            temp = pd.concat([closest_1, closest_2])
            out = pd.concat([closest_0, temp])
            out = out.rename(columns={'geometry.coordinates': 'coordinates'})
            return out
        else:
            out = pd.concat([closest_0, closest_1])
            return out

    else:
        if multi_num is not None and len(dis_array) > multi_num:
            if multi_num > 3: multi_num = 3
            smallest_indices = np.argsort(dis_array)[:multi_num]
            return res_df.loc[res_df['id'].isin(id_array[smallest_indices])]
        return closest_df


# filter images by time and seasons
def mapillary_timestamp_to_datetime(timestamp_ms):
    timestamp_sec = timestamp_ms / 1000.0
    dt_object = datetime.fromtimestamp(timestamp_sec)
    return dt_object.year, dt_object.month, dt_object.day, dt_object.hour

def get_capture_time_range(year: list | tuple = None, season: str = None, time_of_day: str = None):
    """
    Generate start and end time for filtering images by season and time of day.
    """
    year_start, year_end = None, None
    season_ = None
    day_start, day_end = None, None

    if year is not None:
        year_start = year[0]
        year_end = year[1]

    if season is not None:
        seasons = {
            "spring": ((3, 20), (6, 20)),
            "summer": ((6, 21), (9, 21)),
            "fall": ((9, 22), (12, 22)),
            "winter": ((12, 21), (3, 19))
        }
        season_ = {season.lower(): seasons[season.lower()]}

    if time_of_day is not None:
        if time_of_day.lower() == "day":
            day_start = 6
            day_end = 18
        elif time_of_day.lower() == "night":
            day_start = 18
            day_end = 6
        else:
            raise ValueError("time_of_day must be 'day' or 'night'")

    return year_start, year_end, season_, day_start, day_end

def get_sequence(sequence_id, key=None):
    url = f"https://graph.mapillary.com/image_ids?access_token={key}&sequence_id={sequence_id}"
    response = retry_request(url)
    response = response.json()
    return pd.DataFrame.from_dict(response['data'])['id'].tolist()

def get_svi_from_id(id, key):
    url = f"https://graph.mapillary.com/{id}?access_token={key}&fields=id,compass_angle,thumb_original_url,captured_at,geometry,sequence"
    response = retry_request(url)
    response = response.json()
    return pd.json_normalize(response)

def _extract_info(raw, with_geometry=True):
    if with_geometry:
        # extract coordinates
        raw[['point', 'coordinates']] = pd.DataFrame(raw.geometry.tolist(), index=raw.index)
        raw[['lon', 'lat']] = pd.DataFrame(raw.coordinates.tolist(), index=raw.index)

    # extract capture time
    raw['captured'] = raw['captured_at'].apply(mapillary_timestamp_to_datetime)
    raw[['year', 'month', 'day', 'hour']] = pd.DataFrame(raw.captured.to_list(), index=raw.index)
    return raw

# calculate bearing between two points
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1

    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360  # Normalize to 0-360

def save_base64(b64, fn = None):
    b64 = base64.b64decode(b64)
    with open(fn, "wb") as f:
        f.write(b64)

def download_image_requests(image_url, save_path):
    """
    Downloads an image from a URL using the requests library and saves it locally.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(image_url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open the file in write-binary mode ('wb') and write the content
            with open(save_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


# ------------------ ollama utils -------------------
def response2df(qna_dict):
    """
    Extracts filds from QnA objects as a single dictionary and convert it into a Dataframe.
    """
    import pandas as pd
    import numpy as np

    def renameKey(qna_list):
        return [{f'{key}{i + 1}': qna_list[i][key] for key in qna_list[i]} for i in range(len(qna_list))]

    def extract_qna(qna, fs):
        question_num = len(qna[0])
        fs_ = [fs for i in range(question_num)]
        fs_ = list(np.concatenate(fs_))
        dic = {}
        fields = []
        for i in range(len(qna)):
            qna_ = [dict(q) for q in qna[i]]
            qna_ = renameKey(qna_)
            qna_ = {k: v for d_ in qna_ for k, v in d_.items()}
            if i == 0:
                dic = {key: [] for key in qna_}
                fields = list(dic.keys())
            for field_i in range(len(fields)):
                try:
                    dic[fields[field_i]] += [qna_[fields[field_i]]]
                except:
                    pass
        return dic

    qna_ = qna_dict['responses']
    img_ = qna_dict['data']

    fields = list(vars(qna_[0][0]).keys())
    df_ = pd.DataFrame(extract_qna(qna_, fields))
    df_['data'] = [img_[i] for i in range(len(img_))]
    return df_

def base64_to_image(img_base64: str):
    """Converts a Base64-encoded string to an RGB image (NumPy array)."""
    img_data = base64.b64decode(img_base64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ---------------- llamacpp utils-------------------
def extract_last_json(text: str):
    split = ["\n{\n", "\n{", "{"]
    retry = 0
    while text.rfind(split[retry]) == -1:
        retry += 1
        if retry >= 3:
            return None
    start = text.rfind(split[retry])
    candidate = text[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

def responses_to_wide_all_columns(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    row = {}
    for col in df.columns:
        vals = df[col].tolist()
        for i, v in enumerate(vals, start=1):
            row[f"{col}_{i}"] = v
    # stable column order: by row index then column
    cols = [f"{col}_{i}" for i in range(1, n+1) for col in df.columns]
    return pd.DataFrame([row])[cols]

from typing import Sequence
def pick_best_gguf(files: Sequence[str], prefer: Sequence[str]) -> str:
    # Main model: endswith .gguf and NOT mmproj
    candidates = [f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
    if not candidates:
        raise FileNotFoundError("No model .gguf found in repo (excluding mmproj).")

    # Prefer quant strings in filename
    for p in prefer:
        for f in candidates:
            if p.lower() in f.lower():
                return f

    # Fallback: shortest name (often the "default"), else first
    candidates_sorted = sorted(candidates, key=lambda x: (len(x), x))
    return candidates_sorted[0]


def pick_best_mmproj(files: Sequence[str], prefer: Sequence[str]) -> str:
    # mmproj is typically named with "mmproj" and endswith .bin or .gguf
    candidates = [
        f for f in files
        if "mmproj" in f.lower() and (f.lower().endswith(".bin") or f.lower().endswith(".gguf"))
    ]
    if not candidates:
        raise FileNotFoundError("No mmproj file found in repo (expected name containing 'mmproj').")

    # Prefer hints if provided (e.g., f16, q8_0) but keep simple
    for p in prefer:
        for f in candidates:
            if p.lower() in f.lower():
                return f

    candidates_sorted = sorted(candidates, key=lambda x: (len(x), x))
    return candidates_sorted[0]

# def get_mmproj(llm):
#     model_dir = Path(llm)
#     candidates = list(model_dir.parent.glob("*mmproj*.gguf"))
#     candidates = sorted(candidates)
#     if len(candidates) > 0:
#         return str(candidates[0])
#     else:
#         print("No mmproj found")
#         sys.exit(0)

def base64img2temp(s: str) -> str:
    try:
        raw = base64.b64decode(s, validate=True)
    except Exception as e:
        raise ValueError("Invalid base64 image string") from e
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("OpenCV could not decode the provided base64 into an image")
    fd, tmp_path = tempfile.mkstemp(prefix="urban_worm_", suffix=".png")
    os.close(fd)
    try:
        ok = cv2.imwrite(tmp_path, img)  # PNG supports alpha if present
        if not ok:
            raise IOError("cv2.imwrite failed")
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return tmp_path

from .pano2pers import read_url2img
def url2temp(url: str) -> str:
    img = read_url2img(url)
    fd, tmp_path = tempfile.mkstemp(prefix="urban_worm_", suffix=".jpg")
    os.close(fd)
    try:
        ok = cv2.imwrite(tmp_path, img)
        if not ok:
            raise IOError("cv2.imwrite failed")
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return tmp_path


# --------------- image utils ----------------
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from PIL import Image


_DATA_URI_RE = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)

def load_image_auto(
    src: Union[str, Path, bytes, bytearray, BytesIO, Any],
    *,
    convert: Optional[str] = "RGB",   # e.g., "RGB"
    timeout: float = 20.0,
    user_agent: str = "Mozilla/5.0",
) -> Image.Image:
    """
    Load an image into a PIL Image from:
      - Local path (str/Path)
      - URL (http/https)
      - Base64 string (raw base64 or data URI: data:image/...;base64,...)
      - bytes/bytearray
      - file-like object (anything with .read())

    Args:
        src: The image source.
        convert: If provided, convert the image mode (e.g., "RGB").
        timeout: Timeout for URL fetching.
        user_agent: User-Agent header for URL fetching.

    Returns:
        PIL.Image.Image

    Raises:
        ValueError: If src cannot be detected/decoded.
        FileNotFoundError: If a local path is provided but doesn't exist.
    """
    # 1) Already bytes-like
    if isinstance(src, (bytes, bytearray)):
        img = Image.open(BytesIO(src))
        img = img.convert(convert) if convert else img
        img.format = "png"
        return img

    # 2) Path-like
    if isinstance(src, Path):
        p = src.expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Image path not found: {p}")
        img = Image.open(p)
        img = img.convert(convert) if convert else img
        img.format = "png"
        return img

    # 3) File-like
    if hasattr(src, "read") and callable(getattr(src, "read")):
        data = src.read()
        if isinstance(data, str):
            data = data.encode("utf-8", errors="ignore")
        img = Image.open(BytesIO(data))
        img = img.convert(convert) if convert else img
        img.format = "png"
        return img

    # 4) String input: path vs url vs base64
    if isinstance(src, str):
        s = src.strip()

        # 4a) Local path (try first)
        # handles absolute/relative paths
        p = Path(s).expanduser()
        if p.exists() and p.is_file():
            img = Image.open(p)
            img = img.convert(convert) if convert else img
            img.format = "png"
            return img

        # 4b) URL
        parsed = urlparse(s)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            req = Request(s, headers={"User-Agent": user_agent})
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            img = Image.open(BytesIO(data))
            img = img.convert(convert) if convert else img
            img.format = "png"
            return img

        # 4c) data URI base64
        if _DATA_URI_RE.match(s):
            b64_part = s.split(",", 1)[1]
            data = base64.b64decode(b64_part, validate=False)
            img = Image.open(BytesIO(data))
            img = img.convert(convert) if convert else img
            img.format = "png"
            return img

        # 4d) raw base64 (best-effort)
        # Remove whitespace/newlines; try strict decode first
        compact = re.sub(r"\s+", "", s)
        try:
            data = base64.b64decode(compact, validate=True)
            img = Image.open(BytesIO(data))
            img = img.convert(convert) if convert else img
            img.format = "png"
            return img
        except Exception:
            pass

        # If it looks like a Windows path but doesn't exist, raise FileNotFoundError for clarity
        if any(sep in s for sep in (os.sep, "/", "\\")) and (s.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"))):
            raise FileNotFoundError(f"Image path not found: {s}")

        raise ValueError("Could not detect image source as a valid path, URL, or base64 string.")

    raise ValueError(f"Unsupported src type: {type(src)}")

# ------------ flickr utils -------------
def is_coordinate_in_bbox(x, y, bbox):
    """
    Check if a point (x, y) is within a bounding box (xmin, ymin, xmax, ymax).
    Boundaries are inclusive (>=, <=).
    """
    xmin, ymin, xmax, ymax = bbox
    if (xmin <= x <= xmax) and (ymin <= y <= ymax):
        return True
    else:
        return False

def best_url(p):
    # Prefer largest available
    for k in ("url_o", "url_l", "url_c", "url_z", "url_m", "url_n", "url_q", "url_s", "url_t", "url_sq"):
        if k in p and p[k]:
            return p[k]
    # Fallback: construct a medium-ish URL if possible
    if p.get("server") and p.get("id") and p.get("secret"):
        # size suffix omitted -> typically medium
        return f"https://live.staticflickr.com/{p['server']}/{p['id']}_{p['secret']}.jpg"
    return None

# detect face
def is_selfie_photo(model_path, img_url: str):
    model = YuNet(modelPath=model_path)
    img = read_url2img(img_url)
    H, W = img.shape[:2]
    model.setInputSize([W, H])
    results = model.infer(img)
    faces = results.shape[0]
    return faces > 0

class YuNet:
    def __init__(self,
                 modelPath, inputSize=[320, 320],
                 confThreshold=0.6, nmsThreshold=0.3,
                 topK=5000, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize) # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        # Forward
        faces = self._model.detect(image)
        return np.empty(shape=(0, 5)) if faces[1] is None else faces[1]

# ------------- freesound utils -----------
def query_string(q):
    if q is None:
        return ""
    if isinstance(q, (list, tuple)):
        q = " ".join([str(x).strip() for x in q if str(x).strip()])
    else:
        q = str(q).strip()
    return q

def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def season_months(s):
    if s is None:
        return None
    s = s.strip().lower()
    if s == "spring":
        return {3, 4, 5}
    if s == "summer":
        return {6, 7, 8}
    if s in {"fall", "autumn"}:
        return {9, 10, 11}
    if s == "winter":
        return {12, 1, 2}
    raise ValueError("season must be one of: spring, summer, fall, autumn, winter.")


def tod_hours(t):
    if t is None:
        return None
    t = t.strip().lower()
    if t == "morning":
        return set(range(5, 12))  # 05:00–11:59
    if t == "afternoon":
        return set(range(12, 17))  # 12:00–16:59
    if t == "evening":
        return set(range(17, 21))  # 17:00–20:59
    if t == "night":
        return set(list(range(21, 24)) + list(range(0, 5)))  # 21:00–04:59
    raise ValueError("time_of_day must be one of: morning, afternoon, evening, night.")


def year_range(y):
    if y is None:
        return None
    if not isinstance(y, (list, tuple)) or len(y) == 0:
        raise ValueError("year must be a list/tuple like [2020] or (2020, 2022).")
    if len(y) == 1:
        y1 = y2 = int(y[0])
    else:
        y1, y2 = int(y[0]), int(y[1])
        if y2 < y1:
            y1, y2 = y2, y1
    min_dt = f"{y1:04d}-01-01 00:00:00"
    max_dt = f"{y2:04d}-12-31 23:59:59"
    return min_dt, max_dt

def parse_taken(p):
    taken = p.get("datetaken") or p.get("date_taken")
    if not taken:
        return None
    # Flickr typically returns "YYYY-MM-DD HH:MM:SS"
    try:
        return datetime.strptime(taken, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def download_freesound_preview(preview_url: str, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(preview_url, stream=True, timeout=999) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return None

def sliced_duration(duration, clip_duration, number=None):
    num = duration // clip_duration
    duration = duration * 1000
    clip_duration = clip_duration * 1000
    if num >= 1:
        start = 0
        end = clip_duration
        _num = num
        if number is not None:
            if num > number:
                _num = number
        return [[start+i*clip_duration, end+i*clip_duration] for i in range(_num)]
    else:
        return [[0, duration]]

from pydub import AudioSegment
def clip(url=None, start_ms=None, end_ms=None, output_file_path=None):
    try:
        # Download the audio data using requests
        res = requests.get(url)
        # Use BytesIO to treat the downloaded content as a file in memory
        audio_data = BytesIO(res.content)
        # Load the audio file
        audio = AudioSegment.from_file(audio_data, format="mp3")
        # Extract the desired segment
        clipped_audio = audio[start_ms:end_ms]
        # Export the clipped audio to a new file
        clipped_audio.export(output_file_path, format="mp3")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

def sound_url_to_temp(url, slice: list|tuple = None):
    fd, tmp_path = tempfile.mkstemp(prefix="urban_worm_", suffix=".mp3")
    os.close(fd)
    try:
        res = requests.get(url)
        audio_data = BytesIO(res.content)
        audio = AudioSegment.from_file(audio_data, format="mp3")
        if slice is not None:
            audio = audio[slice[0]:slice[1]]
        audio.export(tmp_path, format="mp3")
    except:
        os.remove(tmp_path)
    return tmp_path

# --- Added by patch: robust JSON cleanup for model outputs ---
import re
from typing import Optional

def sanitize_json_text(text: str) -> str:
    """
    Make LLM JSON-ish output more parseable:
    - strip code fences ```json ... ```
    - replace non-breaking spaces and odd whitespace with regular spaces
    - trim obvious junk before/after
    """
    if text is None:
        return ""
    # Remove code fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    # Replace NBSP and similar
    text = text.replace("\xa0", " ").replace("\u00A0", " ")
    # Collapse weird whitespace
    text = re.sub(r"[ \t\u200b\u200c\u200d]+", " ", text)
    # Strip leading/trailing non-json chars
    return text.strip()

def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract the first *balanced* top-level JSON object {...} from text.
    Returns None if nothing balanced is found.
    """
    if not text:
        return None
    start = text.find("{")
    best = None
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    # quick sanity check: looks like JSON
                    if candidate.count("{") == candidate.count("}"):
                        best = candidate
                        return best
        # not balanced; try next open brace
        start = text.find("{", start+1)
    return best
# --- End patch helpers ---
