"""
@purpose: test face recognition
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def zero_mean_data(data):
    columns = data.shape[1]
    zero_data = []
    print(columns)
    for i in range(columns):
        tmp = data[:, i]
        zero_data.append(tmp - tmp.mean())
    return np.array(zero_data).T

    #   calculate covariance matrix
def covariance(data):
    divsor = len(data) - 1
    tData = data.T
    covMatrix = np.dot(tData, data)
    covMatrix = np.divide(covMatrix, divsor)
    return covMatrix


if __name__ == '__main__':
    x = np.array([[90, 60, 50, 34],
                  [90, 90, 60, 34],
                  [60, 60, 60, 64],
                  [60, 60, 90, 45],
                  [30, 30, 30, 45]], np.float16)
    print('buit-in\n', np.cov(x.T))
    y = zero_mean_data(x)
    z = covariance(y)
    print('my\n', z)