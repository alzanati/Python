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
        self.testImagesCount = 50   # to get 76 smile and 76 sad
        self.sadList = []
        self.smileList = []

        self.load_data_from_csv()
        self.__run()

    def load_data_from_csv(self):
        #   load all data sets paths from csv file
        csvFile = open(self.absPath, 'r')
        self.seekLength = self.get_seek_length(absPath=self.absPath)
        for i in range(self.seekLength):
            self.dataPaths.append(csvFile.readline())
        self.absPaths = self.get_abs_paths()
        self.absPaths = sorted(self.absPaths)
        print(self.absPaths)

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
            self.width = img.shape[0]
            self.height = img.shape[1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(-1, 1)
            self.smileFeatureVector.append(img)
        self.smileFeatureVector = np.asarray(self.smileFeatureVector)
        self.smileFeatureVector = self.smileFeatureVector.T[0]
        print('smiles feature vector: \n', self.smileFeatureVector.shape)

        #   read sad images and convert them to (m x n) x 1 vector, and add it to list
        for i in range(len(self.sadPaths)):
            img = cv2.imread(self.sadPaths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        for i in range (self.width * self.height):
            meanImage.append(tmpMean)
        return np.array(meanImage, np.int32)

    def __calculate_cov(self):
        #   get mean image of each category
        self.sadMeanImage = self.get_mean_image(self.sadFeatureVector)
        self.smileMeanImage = self.get_mean_image(self.smileFeatureVector)

        # calculate cov matrix
        self.normalizedSad = self.sadFeatureVector - self.sadMeanImage.reshape(-1, 1)
        self.normalizedSmile = self.smileFeatureVector - self.smileMeanImage.reshape(-1, 1)

        print('normalized smile shape: ', self.normalizedSmile.shape)
        print('normalized sad shape: ', self.normalizedSad.shape)

        self.sadCovMatrix = self.covariance(self.normalizedSad)
        self.smilCovMatrix = self.covariance(self.normalizedSmile)
        print('sad cov matrix shape: ', self.sadCovMatrix.shape)
        print('smile cov matrix shape: ', self.smilCovMatrix.shape)

    def __calculate_eignvectors(self):
        # now we have max 75 eiginvector for each category
        self.eigValsSad, self.eigVecsSad = np.linalg.eig(self.sadCovMatrix)
        self.eigVecsSadNorm = []
        for i in range(len(self.eigVecsSad)):
            tmp = self.eigVecsSad[i] / np.linalg.norm(self.eigVecsSad[i])
            self.eigVecsSadNorm.append(tmp)
        self.eigVecsSad = np.asarray(self.eigVecsSadNorm)
        self.weight = (self.eigValsSad / self.eigValsSad.sum()) * 100

        self.weight = self.weight > 1
        self.eigVecsSad = self.eigVecsSad[self.weight]
        print('sad eig vectors: ', self.eigVecsSad.shape)

        self.eigValsSmile, self.eigVecsSmile = np.linalg.eig(self.smilCovMatrix)
        self.eigVecsSmileNorm = []
        for i in range(len(self.eigVecsSad)):
            tmp = self.eigVecsSmile[i] / np.linalg.norm(self.eigVecsSmile[i])
            self.eigVecsSmileNorm.append(tmp)
        self.eigVecsSmile = np.asarray(self.eigVecsSmileNorm)
        self.weight = (self.eigValsSmile / self.eigValsSmile.sum()) * 100

        self.weight = self.weight > 1
        self.eigVecsSmile = self.eigVecsSmile[self.weight]
        print('smile eig vectors: ', self.eigVecsSmile.shape)

    def __calculate_eigfaces(self):
        #   project data on vector to get weight matrix of sads and smiles to get eigin faces
        self.sadProjectionMatrix = np.dot(self.eigVecsSad, self.normalizedSad.T)
        print('projection matrix of sads(eign_faces): ', self.sadProjectionMatrix.shape)

        self.smileProjectionMatrix = np.dot(self.eigVecsSmile, self.normalizedSmile.T)
        print('projection matrix of smiles(eign_faces): ', self.smileProjectionMatrix.shape)

    def __calculate_kth_coefficient(self):
        #   project data on vector to get weight matrix of sads and smiles to get eigin faces
        self.sadProjectionMatrix = np.dot(self.eigVecsSad, self.normalizedSad.T)
        print('projection matrix of sads(eign_faces): ', self.sadProjectionMatrix.shape)

        self.smileProjectionMatrix = np.dot(self.eigVecsSmile, self.normalizedSmile.T)
        print('projection matrix of smiles(eign_faces): ', self.smileProjectionMatrix.shape)

        self.sadWeightMatrix = np.dot(self.normalizedSad.T, self.sadProjectionMatrix.T)
        print('sad weight matrix: ', self.sadWeightMatrix.shape)

        self.smileWeightMatrix = np.dot(self.normalizedSmile.T, self.smileProjectionMatrix.T)
        print('smile weight matrix: ', self.smileWeightMatrix.shape)

    def __run(self):

        self.__calculate_cov()

        self.__calculate_eignvectors()

        self.__calculate_kth_coefficient()



        #   get ROC curve

    def test(self, testImag):
        self.testImage = cv2.imread(testImag)
        self.testImage = cv2.cvtColor(self.testImage, cv2.COLOR_BGR2GRAY)
        self.testImage = self.testImage.reshape(-1, 1)

        newImage_sad = self.testImage - self.sadMeanImage.reshape(-1, 1)
        newImage_smile = self.testImage - self.smileMeanImage.reshape(-1, 1)

        self.projectedSadImage = np.dot(self.sadProjectionMatrix, newImage_sad)
        self.projectedSmileImage = np.dot(self.smileProjectionMatrix, newImage_smile)
        print('sad_image projected matrix: ', self.projectedSadImage.shape)
        print('smile_image projected matrix: ', self.projectedSmileImage.shape)

        self.__compare_show(self.projectedSmileImage, self.projectedSadImage)

    def __compare_show(self, projectedSmileImage, projectedSadImage):
        for i in range(self.smileWeightMatrix.shape[0]):
            sd = self.smileWeightMatrix[i].reshape(-1, 1)
            projectedSmileImage = projectedSmileImage.reshape(-1, 1)
            sd = (sd - projectedSmileImage) ** 2
            sd = np.sqrt(sd.sum())
            self.smileList.append(sd)

        for i in range(self.sadWeightMatrix.shape[0]):
            sd = self.sadWeightMatrix[i].reshape(-1, 1)
            projectedSadImage = projectedSadImage.reshape(-1, 1)
            sd = (sd - projectedSadImage) ** 2
            sd = np.sqrt(sd.sum())
            self.sadList.append(sd)

        ssdb = np.asarray(self.smileList)
        ssdb = ssdb.sum() / ssdb.max()

        ssda = np.asarray(self.sadList)
        ssda = ssda.sum() / ssda.max()

        if ssdb < ssda:
            img = cv2.imread('smilee.jpg')
            cv2.namedWindow('smile', cv2.WINDOW_NORMAL)
            cv2.imshow('smile', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif ssdb > ssda:
            img = cv2.imread('sadd.jpg')
            cv2.namedWindow('sad', cv2.WINDOW_NORMAL)
            cv2.imshow('sad', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # run algorithm
    f = FaceRecognition('front_images_part1.csv')
    f.test('/home/mohamed/workspace/Python/dataSets/front_images_part1/86b.jpg')
