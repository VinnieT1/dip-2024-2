# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def translate_img(img: np.ndarray) -> np.ndarray:
    translated_img = np.zeros_like(img)
    translated_img[1:, 1:] = img[:-1, :-1]

    return translated_img

def stretch_img(img: np.ndarray) -> np.ndarray:
    img_rows, img_cols = img.shape
    new_cols = int(img_cols * 1.5)
    stretched_img = np.zeros((img_rows, new_cols), dtype=img.dtype)

    for i in range(img_rows):
        for j in range(new_cols):
            original_col = int(j / 1.5)
            stretched_img[i, j] = img[i, original_col]
    
    return stretched_img

def distort_img(img: np.ndarray) -> np.ndarray:
    img_rows, img_cols = img.shape
    center_x, center_y = img_cols / 2, img_rows / 2
    k = 0.1 
    distorted_img = np.zeros_like(img)
    
    for i in range(img_rows):
        for j in range(img_cols):
            x = (j - center_x) / center_x
            y = (i - center_y) / center_y
            r = np.sqrt(x**2 + y**2)
            factor = 1 + k * r**2
            src_x = int(center_x + (x / factor) * center_x)
            src_y = int(center_y + (y / factor) * center_y)
            
            if 0 <= src_x < img_cols and 0 <= src_y < img_rows:
                distorted_img[i, j] = img[src_y, src_x]
    
    return distorted_img

def apply_geometric_transformations(img: np.ndarray) -> dict:
    return {
        'translated': translate_img(img),
        'rotated': np.rot90(img, -1),
        'stretched': stretch_img(img),
        'mirrored': np.fliplr(img),
        'distorted': distort_img(img),
    }