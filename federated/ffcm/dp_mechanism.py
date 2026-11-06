
import numpy as np
def add_gaussian_noise(arr, sigma):
    return arr + np.random.normal(0, sigma, arr.shape)
