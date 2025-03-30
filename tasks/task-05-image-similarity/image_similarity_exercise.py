# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def mse(i1: np.ndarray, i2: np.ndarray) -> float:
    return np.mean((i1 - i2) ** 2)

def psnr(mean_sqr_error: float) -> float:
    return float('inf') if mean_sqr_error == 0 else -10 * np.log10(mean_sqr_error)

def ssim(i1: np.ndarray, i2: np.ndarray) -> float:
    sample_mean1 = np.mean(i1)
    sample_mean2 = np.mean(i2)

    covariance_matrix = np.cov(i1.flatten(), i2.flatten())

    sample_variance1 = covariance_matrix[0, 0]
    sample_variance2 = covariance_matrix[1, 1]
    covariance = covariance_matrix[0, 1]

    k1 = 0.01
    k2 = 0.03
    L = 1.0

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    numerator = (2 * sample_mean1 * sample_mean2 + c1) * (2 * covariance + c2)
    denominator = (sample_mean1 ** 2 + sample_mean2 ** 2 + c1) * (sample_variance1 + sample_variance2 + c2)
    
    return numerator / denominator

def npcc(i1: np.ndarray, i2: np.ndarray) -> float:
    mean1 = np.mean(i1)
    mean2 = np.mean(i2)

    numerator = np.sum((i1 - mean1) * (i2 - mean2))
    denominator = np.sqrt(np.sum((i1 - mean1) ** 2) * np.sum((i2 - mean2) ** 2))
    
    return numerator / denominator

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    mse_value = mse(i1, i2)
    psnr_value = psnr(mse_value)
    ssim_value = ssim(i1, i2)
    npcc_value = npcc(i1, i2)

    return {
        "mse": mse_value,
        "psnr": psnr_value,
        "ssim": ssim_value,
        "npcc": npcc_value,
    }
