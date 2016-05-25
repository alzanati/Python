"""
@purpose: implement a face recognition using pca
"""

import cv2
import numpy as np
from matplotlib import pylab as plt
from scipy import signal as sg


class FaceRecognition:
    def __init__(self):
        self.smileFeatureVector = []  # list to hold smild image vectors
        self.sadFeatureVector = []  # list to hold sad image vectors
        self.smilePaths = []
        self.sadPaths = []
        self.imageLocations = []
        self.dataPaths = []
        self.absPaths = []
        self.absPath = 'front_images_part1.csv'
        self.seekLength = 0

        self.load_data_from_csv()

    def load_data_from_csv(self):
        csvFile = open(self.absPath, 'r')
        self.seekLength = self.get_seek_length(absPath=self.absPath)
        for i in range(self.seekLength):
            self.dataPaths.append(csvFile.readline())
        self.absPaths = self.get_abs_paths()

        for i in range(len(self.absPaths) - 1):
            tmpStr = self.absPaths[i]
            tmpChar = tmpStr[-5]
            if tmpChar == 'b':
                self.smilePaths.append(tmpStr)
            elif tmpChar == 'a':
                self.sadPaths.append(tmpStr)

        # read images to vectors
        for i in range(len(self.smilePaths)):
            img = cv2.imread(self.smilePaths[i])
            img = img.reshape(-1, 1)
            self.smileFeatureVector.append(img)

        for i in range(len(self.sadPaths)):
            img = cv2.imread(self.sadPaths[i])
            img = img.reshape(-1, 1)
            self.sadFeatureVector.append(img)

    def get_abs_paths(self):
        for i in range(len(self.dataPaths)):
            strTmp = self.dataPaths[i]
            strTmp = strTmp[:-2]
            self.absPaths.append(strTmp)
        return self.absPaths

    def get_seek_length(self, absPath):
        csvFile = open(absPath, 'r')
        seekLength = 0
        while csvFile.readline():
            seekLength += 1
        return seekLength

    # calculate covariance matrix
    @staticmethod
    def covariance(data):
        mean = data.mean()  # mean of data
        zero_mean_data = data - mean  # zero mean data
        data_transposed = zero_mean_data.T  # transpose data
        cov_matrix = np.multiply(zero_mean_data, data_transposed)  # matrix multiplication X * X'
        cov_matrix /= (data.shape[0] - 1)  # cov / n - 1

        for i in range(len(cov_matrix[0])):
            print(cov_matrix[i])

        return cov_matrix

    # calculate pca coefficients with covariance matrix (150 * 150)
        covMatrix = FaceRecognition.covariance(data)

    # now we have max 150 eiginvector
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

    # project data on vector to get weight matrix choose only 50 vector
        adjusted_data = data.T * eig_vecs

    # get new vector and compare with old vectors with ssd
        new_vector = eig_vecs * new_data.T

    # get normalized cross correlation
        e = new_vector - adjusted_data

    # get ROC curve


if __name__ == '__main__':
    # run algorithm
    f = FaceRecognition()
    x = np.array([[24, 24, -6, -6, -36], [0, 30, 0, 0, -30], [30, -30, 0, 30, -30]])
    y = np.ma.cov(x)
    print(y)
