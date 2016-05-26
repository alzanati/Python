"""
@purpose: test face recognition
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image




if __name__ == '__main__':
    x = np.array([[90, 60, 90],
                  [90, 90, 30],
                  [60, 60, 60],
                  [60, 60, 90],
                  [30, 30, 30]], np.float32)
    print('buit-in\n', np.cov(x))
    y = zero_mean_data(x)
    z = covariance(y)
    print(z)