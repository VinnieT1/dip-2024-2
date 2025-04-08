# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    matched_img = np.zeros_like(source_img)

    for channel in range(3):
        src = source_img[..., channel].ravel()
        ref = reference_img[..., channel].ravel()

        src_hist, _ = np.histogram(src, bins=256, range=(0, 255), density=True)
        ref_hist, _ = np.histogram(ref, bins=256, range=(0, 255), density=True)

        src_cdf = np.cumsum(src_hist)
        ref_cdf = np.cumsum(ref_hist)

        lookup_table = np.zeros(256, dtype=np.uint8)
        ref_value = 0
        for src_value in range(256):
            while ref_value < 255 and ref_cdf[ref_value] < src_cdf[src_value]:
                ref_value += 1
            lookup_table[src_value] = ref_value

        matched_channel = lookup_table[source_img[..., channel]]
        matched_img[..., channel] = matched_channel

    return matched_img