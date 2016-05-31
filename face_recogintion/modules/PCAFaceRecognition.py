"""
@purpose: implement a face recognition using pca
"""

import cv2
import numpy as np

from modules.AbstractRecogntion import FacePCA


class FaceRecognition(FacePCA):
    def __init__(self, csvFile, count):
        self.absPath = csvFile
        self.featureVector = []  # list to hold smild image vectors
        self.imageCount = count
        self.dataPaths = []
        self.trainingList = []
        self.testList = []
        self.ecuDis = []
        self.indecies = np.array(0)
        self.index = 0
        self.truePositive = 0
        self.falsePositive = 0
        self.TPR = []
        self.FPR = []

        self.load_data_from_csv()
        self.__run()
        self.test()

    def load_data_from_csv(self):
        #   load all data sets paths from csv file
        csvFile = open(self.absPath, 'r')
        self.seekLength = self.imageCount
        for i in range(self.seekLength):
            self.dataPaths.append(csvFile.readline())
        self.absPaths = self.get_abs_paths(self.dataPaths)

        #   get width & height
        img = cv2.imread(self.absPaths[i])
        self.width = img.shape[0]
        self.height = img.shape[1]

        #   divide data sets into training(2) and test (1)
        for i in range(self.imageCount):
            tmpStr = self.absPaths[i]
            tmpChar = tmpStr[-6:-4]
            if str(tmpChar) == str(11):
                self.trainingList.append(tmpStr)
            elif str(tmpChar) == str(12):
                self.trainingList.append(tmpStr)
            elif str(tmpChar) == str(13):
                self.testList.append(tmpStr)

        # read training images and convert them to (m x n) x 1 vector, and add it to list
        for i in range(len(self.trainingList)):
            img = cv2.imread(self.trainingList[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape(-1, 1)
            self.featureVector.append(img)
        self.featureVector = np.asarray(self.featureVector)
        self.featureVector = self.featureVector.T[0]  # column vector
        print('feature vector: \n', self.featureVector.shape)

        self.absPaths.clear()
        self.dataPaths.clear()

    def __calculate_cov(self):
        #   get mean image of each category
        self.meanImage = self.get_mean_image(self.featureVector, self.width, self.height)

        # calculate cov matrix
        self.normalizedData = np.array(self.featureVector) - self.meanImage.reshape(-1, 1)
        print('normalized data shape: ', self.normalizedData.shape)

        self.covMatrix = self.covariance(self.normalizedData)
        print('cov matrix shape: ', self.covMatrix.shape)

    def __calculate_eignvectors(self):
        # now we have max 200 eiginvector for each category
        self.eigVals, self.eigVecs = np.linalg.eig(self.covMatrix)
        self.eigVecsNorm = []

        for i in range(len(self.eigVecs)):
            tmp = self.eigVecs[i] / np.linalg.norm(self.eigVecs[i])
            self.eigVecsNorm.append(tmp)

        self.eigVecs = np.asarray(self.eigVecsNorm)
        self.weight = (self.eigVals / self.eigVals.sum()) * 100

        self.weight = self.weight > 0.5
        self.eigVecs = self.eigVecs[self.weight]
        print('eig vectors shape: ', self.eigVecs.shape)

    def __calculate_eigfaces(self):
        #   project data on vector to get weight matrix of sads and smiles to get eigin faces
        self.projectionMatrix = np.dot(self.eigVecs, self.normalizedData.T)
        self.projectionMatrix /= self.projectionMatrix.max()
        print('projection matrix(eignfaces): ', self.projectionMatrix.shape)

    def __calculate_kth_coefficient(self):
        #   project data on vector to get weight matrix of sads and smiles to get eigin faces
        self.weightMatrix = np.dot(self.normalizedData.T, self.projectionMatrix.T)
        self.weightMatrix /= self.weightMatrix.max()
        print('data weight matrix: ', self.weightMatrix.shape)

    def __run(self):
        #   run algorithm
        self.__calculate_cov()

        self.__calculate_eignvectors()

        self.__calculate_eigfaces()

        self.__calculate_kth_coefficient()

    def test(self):
        th = 0.6
        for path in self.testList:
            th -= 0.09
            testImage = cv2.imread(path)
            testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
            testImage = testImage.reshape(-1, 1)
            newImage = testImage - self.meanImage.reshape(-1, 1)
            projectedImage = np.dot(self.projectionMatrix, newImage)
            projectedImage /= projectedImage.max()

            tmpDis = self.__compare_show(projectedImage, th)    # [200]
            for i in range(20):
                if i == self.index:
                    if tmpDis[self.index] == 1:
                        self.truePositive += 1
                    if tmpDis[self.index+1] == 1:
                        self.truePositive += 1
                else:
                    if tmpDis[i] == 1:
                        self.falsePositive += 1
            self.index += 2

            print(self.truePositive)

            truePositiveRate = self.truePositive / (self.truePositive + self.falsePositive)
            falsePositiveRate = self.falsePositive / (self.truePositive + self.falsePositive)
            self.TPR.append(truePositiveRate)
            self.FPR.append(falsePositiveRate)
            tmpDis.clear()

        print(self.TPR)
        print(self.FPR)

    def __compare_show(self, projectedImage, th):
        #   get ssd of weight matrix and new image
        ssdb = self.get_ssd(self.weightMatrix.shape[0], self.weightMatrix,
                            projectedImage, th)

        return ssdb


if __name__ == '__main__':
    # run algorithm
    f = FaceRecognition('originalimages_part1.csv', 1400)
