import argparse
import numpy as np
import cv2 as cv
import urllib.request

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """

    ### START CODE HERE ###
    response = urllib.request.urlopen(url)
    image_data = response.read()

    image_bytes = np.frombuffer(image_data, dtype=np.uint8)
    image = cv.imdecode(image_bytes, flags=kwargs.get("flags", cv.IMREAD_COLOR))
    ### END CODE HERE ###
    
    return image

load_image_from_url()