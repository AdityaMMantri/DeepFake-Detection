import cv2
import numpy as np


def compute_fft(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # compute FFT
    fft = np.fft.fft2(gray)

    # shift low frequency to center
    fft_shift = np.fft.fftshift(fft)

    # magnitude spectrum
    magnitude = np.log(np.abs(fft_shift) + 1)

    # normalize to 0-255
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    magnitude = magnitude.astype(np.uint8)

    # convert to 3-channel image
    magnitude = cv2.cvtColor(
        magnitude,
        cv2.COLOR_GRAY2RGB
    )

    return magnitude

# this file will be used to compute the FFT image from the RGB image