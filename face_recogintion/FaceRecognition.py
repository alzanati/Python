"""
@purpose: implement a face recognition using pca
"""

import cv2
import numpy as np
from matplotlib import pylab as plt
from scipy import signal as sg


class FaceRecognition:
    def __init__(self, csvFile):
        self.smileFeatureVector = []    # list to hold smild image vectors
        self.sadFeatureVector = []      # list to hold sad image vectors
        self.smilePaths = []            # smile faces paths
        self.sadPaths = []              # sad faces paths
        self.dataPaths = []             #
        self.absPaths = []              # absolute paths without ";"
        self.absPath = csvFile          # absolute path for data sets
        self.seekLength = 0             # data set count
        self.testImagesCount = 49       # to get 76 smile and 76 sad

        self.load_data_from_csv()
        self.run()

    def load_data_from_csv(self):

        #   load all data sets paths from csv file
        csvFile = open(self.absPath, 'r')
        self.seekLength = self.get_seek_length(absPath=self.absPath)
        for i in range(self.seekLength):
            self.dataPaths.append(csvFile.readline())
        self.absPaths = self.get_abs_paths()

        #   divide data sets into smile and sad according to data set "b" represent smile
        for i in range(len(self.absPaths) - self.testImagesCount):
            tmpStr = self.absPaths[i]
            tmpChar = tmpStr[-5]
            if tmpChar == 'b':
                self.smilePaths.append(tmpStr)
            elif tmpChar == 'a':
                self.sadPaths.append(tmpStr)

        #   read smile images and convert them to (m x n) x 1 vector, and add it to list
        for i in range(len(self.smilePaths)):
            img = cv2.imread(self.smilePaths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(-1, 1)
            self.smileFeatureVector.append(img)
        self.smileFeatureVector = np.asarray(self.smileFeatureVector)
        self.smileFeatureVector = self.smileFeatureVector.T[0]
        print('smiles: \n', self.smileFeatureVector.shape)


        #   read sad images and convert them to (m x n) x 1 vector, and add it to list
        for i in range(len(self.sadPaths)):
            img = cv2.imread(self.sadPaths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(-1, 1)
            self.sadFeatureVector.append(img)
        self.sadFeatureVector = np.asarray(self.sadFeatureVector)
        self.sadFeatureVector = self.sadFeatureVector.T[0]
        print('sads: \n', self.sadFeatureVector.shape)

        self.absPaths.clear()
        self.dataPaths.clear()
        self.sadPaths.clear()
        self.smilePaths.clear()

    #   get absolute path of images by parsing it to remove ";"
    def get_abs_paths(self):
        for i in range(len(self.dataPaths)):
            strTmp = self.dataPaths[i]
            strTmp = strTmp[:-2]
            self.absPaths.append(strTmp)
        return self.absPaths

    #   get length of data sets ( count of images )
    def get_seek_length(self, absPath):
        csvFile = open(absPath, 'r')
        seekLength = 0
        while csvFile.readline():
            seekLength += 1
        return seekLength

    #   calculate zero mean data
    @staticmethod
    def zero_mean_data(data):
        n = data.shape[1]
        zero_data = []
        print(n)
        for i in range(n):
            tmp = data[:, i]
            zero_data.append(tmp - tmp.mean())
        return np.array(zero_data).T

    #   calculate covariance matrix
    @staticmethod
    def covariance(data):
        divsor = len(data) - 1
        tData = data.T
        covMatrix = np.dot(tData, data)
        covMatrix = np.divide(covMatrix, divsor)
        return covMatrix

    def run(self):
        # calculate cov matrix
        self.sadCovMatrix = self.zero_mean_data(self.sadFeatureVector)
        self.smilCovMatrix = self.zero_mean_data(self.smileFeatureVector)
        self.sadCovMatrix = self.covariance(self.sadCovMatrix)
        self.smilCovMatrix = self.covariance(self.smilCovMatrix)
        print(self.sadCovMatrix.shape)
        print(self.smilCovMatrix.shape)

        # now we have max 75 eiginvector for each category
        self.eigValsSad, self.eigVecsSad = np.linalg.eig(self.sadCovMatrix)
        self.weight = (self.eigValsSad / self.eigValsSad.max()) * 100
        print(self.eigValsSad[self.weight > 1.3])
        self.weight = self.weight > 1.3         # get boolean indexing to access vectors
        print(self.weight)

        self.eigValsSmile, self.eigVecsSmile = np.linalg.eig(self.smilCovMatrix)
        self.weight = (self.eigValsSmile / self.eigValsSmile.max()) * 100
        print(self.eigValsSmile[self.weight > 1.3])
        self.weight = self.weight > 1.3         # get boolean indexing to access vectors
        print(self.weight)
    #
    # #   project data on vector to get weight matrix choose only 50 vector
    #     adjusted_data = data.T * eig_vecs
    #
    # #   get new vector and compare with old vectors with ssd
    #     new_vector = eig_vecs * new_data.T
    #
    # #   get normalized cross correlation
    #     e = new_vector - adjusted_data

    #   get ROC curve


if __name__ == '__main__':
    # run algorithm
    f = FaceRecognition('front_images_part1.csv')
    x = np.array([[24, 24, -6, -6, -36],
                  [0, 30, 0, 0, -30],
                  [30, -30, 0, 30, -30]])

    # cv2.namedWindow("sd", cv2.WINDOW_NORMAL)
    # cv2.imshow("sd", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(np.ma.cov(x.T))
    # z = FaceRecognition.zero_mean_data(x)
    # z = FaceRecognition.covariance(z)
    # print(z)
