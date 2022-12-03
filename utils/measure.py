from math import log10, sqrt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import torch
  
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def SSIM(real, output): 
    mse_const = mean_squared_error(real, output)
    ssim_const = ssim(real, output,
                    data_range=output.max() - output.min())

def Correl(real, output):
    