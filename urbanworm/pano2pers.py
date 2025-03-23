'''
The source code is adapted from https://github.com/fuenwang/Equirec2Perspec.git. Credit to the author @fuenwang.
'''

import cv2
import numpy as np
from urllib.request import urlopen
import base64

# Equirectangular to Perspective
class Equirectangular:
    '''
    Covert paronoma to perspective
    '''

    def __init__(self, img_path:str=None, img_url:str=None):
        '''
        Add image

        Args:
            img_path (str): Image path
            img_url (str): Image URL
        '''
        if img_path != None:
            self._img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        elif img_url != None:
            self._img = self.read_url2img(img_url)
        [self._height, self._width, _] = self._img.shape
    
    def read_url2img(self, url:str) -> np.ndarray:
        '''
        Read image from a URL

        Args:
            url (str): Image URL

        Returns: 
            np.ndarray: The image as a NumPy array.
        '''
        resp = urlopen(url, timeout=100)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def GetPerspective(self, FOV:float, THETA:float, PHI:float, height:int, width:int, RADIUS:int = 128) -> str:
        """
        Convert an equirectangular panorama image to a perspective view.

        This function computes the perspective projection of a 360° panorama image 
        based on field of view and view angles, returning the perspective as a 
        base64-encoded PNG image (useful for web/LLM APIs).

        Args:
            FOV (float): Field of view in degrees.
            THETA (float): Horizontal viewing angle (left/right), in degrees.
            PHI (float): Vertical viewing angle (up/down), in degrees.
            height (int): Height of the output image.
            width (int): Width of the output image.
            RADIUS (int, optional): Projection sphere radius. Defaults to 128.

        Returns:
            str: A base64-encoded PNG string representing the perspective view.
        """

        # THETA is left/right angle, PHI is up/down angle, both in degree
        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        # xyz = np.zeros([height, width, 3], np.float)
        xyz = np.zeros([height, width, 3], np.float32)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]
        
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        # lon = np.zeros([height * width], np.float)
        lon = np.zeros([height * width], np.float32)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool_)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool_)
        
        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy
    
        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        # Convert for Ollama
        _, buffer = cv2.imencode('.png', persp)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
        