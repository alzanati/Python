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
        self.dataPaths = []
        self.absPaths = []                  # absolute paths without ";"
        self.normalizedSad = np.array(0)    # store zero mean data
        self.normalizedSmile = np.array(0)  # store zero mean data
        self.absPath = csvFile              # absolute path for data sets
        self.seekLength = 0         # data set count
        self.testImagesCount = 49   # to get 76 smile and 76 sad
        self.sadList = []
        self.smileList = []

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
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(-1, 1)
            self.smileFeatureVector.append(img)
        self.smileFeatureVector = np.asarray(self.smileFeatureVector)
        self.smileFeatureVector = self.smileFeatureVector.T[0]
        print('smiles feature vector: \n', self.smileFeatureVector.shape)

        #   read sad images and convert them to (m x n) x 1 vector, and add it to list
        for i in range(len(self.sadPaths)):
            img = cv2.imread(self.sadPaths[i])
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(-1, 1)
            self.sadFeatureVector.append(img)
        self.sadFeatureVector = np.asarray(self.sadFeatureVector)
        self.sadFeatureVector = self.sadFeatureVector.T[0]
        print('sads feature vector: \n', self.sadFeatureVector.shape)

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

    #   get mean image of category
    def get_mean_image(self, data):
        n = data.shape[1]
        tmpMean = []
        meanImage = []
        for i in range(n):
            tmpData = data[:, i]
            tmpMean.append(tmpData.mean())
        tmpMean = np.array(tmpMean).mean()
        for i in range (260 * 360 * 3):
            meanImage.append(tmpMean)
        return np.array(meanImage, np.int32)

    def run(self):

        #   get mean image of each category
        self.sadMeanImage = self.get_mean_image(self.sadFeatureVector)
        self.smileMeanImage = self.get_mean_image(self.smileFeatureVector)

        # calculate cov matrix
        self.normalizedSad = self.zero_mean_data(self.sadFeatureVector)
        self.normalizedSmile = self.zero_mean_data(self.smileFeatureVector)
        print('normalized smile shape: ', self.normalizedSmile.shape)
        print('normalized sad shape: ', self.normalizedSad.shape)

        self.sadCovMatrix = self.covariance(self.normalizedSad)
        self.smilCovMatrix = self.covariance(self.normalizedSmile)
        print('sad cov matrix shape: ', self.sadCovMatrix.shape)
        print('smile cov matrix shape: ', self.smilCovMatrix.shape)

        # now we have max 75 eiginvector for each category
        self.eigValsSad, self.eigVecsSad = np.linalg.eig(self.sadCovMatrix)
        self.eigVecsSad = self.eigVecsSad / np.linalg.norm(self.eigVecsSad)
        self.weight = (self.eigValsSad / self.eigValsSad.sum()) * 100

        self.weight = self.weight > 0.55
        self.eigVecsSad = self.eigVecsSad[self.weight]
        print('sad eig vectors: ', self.eigVecsSad.shape)

        self.eigValsSmile, self.eigVecsSmile = np.linalg.eig(self.smilCovMatrix)
        self.eigVecsSmile = self.eigVecsSmile / np.linalg.norm(self.eigVecsSmile)
        self.weight = (self.eigValsSmile / self.eigValsSmile.sum()) * 100

        self.weight = self.weight > 0.64
        self.eigVecsSmile = self.eigVecsSmile[self.weight]
        print('smile eig vectors: ', self.eigVecsSmile.shape)

        #   project data on vector to get weight matrix of sads and smiles to get eigin faces
        self.sadProjectionMatrix = np.dot(self.eigVecsSad, self.normalizedSad.T)
        print('projection matrix of sads: ', self.sadProjectionMatrix.shape)

        self.smileProjectionMatrix = np.dot(self.eigVecsSmile, self.normalizedSmile.T)
        print('projection matrix of smiles: ', self.smileProjectionMatrix.shape)

        self.sadWeightMatrix = np.dot(self.normalizedSad.T, self.sadProjectionMatrix.T)
        print('sad weight matrix: ', self.sadWeightMatrix.shape)

        self.smileWeightMatrix = np.dot(self.normalizedSmile.T, self.smileProjectionMatrix.T)
        print('smile weight matrix: ', self.smileWeightMatrix.shape)

        #   get new vector and compare with old vectors with ssd
        #   by multiply with each eigin face after substract mean image
        newImage = cv2.imread('/home/mohamed/workspace/Python/dataSets/front_images_part1/17b.jpg')
        # newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        newImage = newImage.reshape(-1, 1)

        newImage_sad = newImage - self.sadMeanImage.reshape(-1, 1)
        newImage_smile = newImage - self.smileMeanImage.reshape(-1, 1)

        projectedSadImage = np.dot(self.sadProjectionMatrix, newImage_sad)
        projectedSmileImage = np.dot(self.smileProjectionMatrix, newImage_smile)
        print('sad_image projected matrix: ', projectedSadImage.shape)
        print('smile_image projected matrix: ', projectedSmileImage.shape)

        for i in range(self.smileWeightMatrix.shape[0]):
            sd = self.smileWeightMatrix[i].reshape(-1, 1)
            projectedSmileImage = projectedSmileImage.reshape(-1, 1)
            sd = (sd - projectedSmileImage)**2
            sd = np.sqrt(sd.sum())
            self.smileList.append(sd)

        for i in range(self.sadWeightMatrix.shape[0]):
            sd = self.sadWeightMatrix[i].reshape(-1, 1)
            projectedSadImage = projectedSadImage.reshape(-1, 1)
            sd = (sd - projectedSadImage) ** 2
            sd = np.sqrt(sd.sum())
            self.sadList.append(sd)

        ssdb = np.asarray(self.smileList)
        ssdb = ssdb.sum()

        ssda = np.asarray(self.sadList)
        ssda = ssda.sum()

        if ssdb < ssda:
            print('smile')
        elif ssdb > ssda:
            print('sad')

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
