import cv2
import numpy as np

def compute_noise(image):

    blur=cv2.GaussianBlur(
        image,
        (5,5),
        0
    )

    noise=image - blur

    noise=cv2.normalize(
        noise,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return noise.astype(np.uint8)

# this file will be used to Blur the image and then subtract the image and blured image to get high frequency image (noise)
# Gaussian Blur is used and kernel of size 5 by 5 is used