"""
@purpose: Implement abstract class for recognition
@author: Mohamed Hosny Ahmed
"""

import numpy as np


class FacePCA(object):
    #   get absolute path of images by parsing it to remove ";"
    @staticmethod
    def get_abs_paths(dataPaths):
        absPaths = []
        for i in range(len(dataPaths)):
            strTmp = dataPaths[i]
            strTmp = strTmp[:-2]
            absPaths.append(strTmp)
        return absPaths

    #   get length of data sets ( count of images )
    @staticmethod
    def get_seek_length(absPath):
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
    @staticmethod
    def get_mean_image(data, width, height):
        n = data.shape[1]
        tmpMean = []
        meanImage = []
        for i in range(n):
            tmpData = data[:, i]
            tmpMean.append(tmpData.mean())
        tmpMean = np.array(tmpMean).mean()
        for i in range(width * height):
            meanImage.append(tmpMean)
        return np.array(meanImage, np.int32)

    @staticmethod
    def get_ssd(imageCounts, wieghtMatrix, projectedImage):
        ssdList = []
        for i in range(imageCounts):
            sd = wieghtMatrix[i].reshape(-1, 1)
            projectedImage = projectedImage.reshape(-1, 1)
            ssd = np.sum((sd - projectedImage)**2)
            ssdList.append(np.sqrt(ssd))
        return np.asarray(ssdList)

    @staticmethod
    def sorted_nicely(l):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

