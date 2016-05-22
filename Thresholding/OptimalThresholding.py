"""
@purpose: implement optimal threshold
@author: Mohamed Hosny Ahmed
@date: 1 / 5 / 2016
"""

import numpy as np


class OptimalThreshold:

    # initialization to classes
    def __init__(self, image):
        self.img = image
        self.h, self.w = self.img.shape
        self.backGround = np.zeros(0)
        self.object = np.zeros(0)
        self.optimalThreshold = 0
        self.run_algorithm()  # run algorithm

    # get 4 corners as initail background
    def get_init_back_ground(self):
        corner1 = self.img[0][0]
        corner2 = self.img[0][self.w - 1]
        corner3 = self.img[self.h - 1][0]
        corner4 = self.img[self.h - 1][self.w - 1]
        intMatrix = np.array([[corner1, corner2], [corner3, corner4]], np.int32)
        return intMatrix.sum() / np.size(intMatrix)

    # get rest of pixels as initail object
    def get_init_object(self):
        tmpImg = []
        for i in range(self.h):
            for j in range(self.w):
                if i == 0 and j == 0:
                    continue
                elif i == self.h-1 and j == 0:
                    continue
                elif i == 0 and j == self.w-1:
                    continue
                elif i == self.h-1 and j == self.w-1:
                    continue
                else:
                    tmpImg.append(self.img[i][j])
        return self.get_mean_list(tmpImg)

    # return optimal thresholding value use to draw new image
    def get_optimal_threshold(self):
        return self.optimalThreshold

    # get object back ground pixels as a list represent whole pixles
    def get_object_backGround(self, thr):

        tmpBackGround = []
        tmpObject = []

        for i in range(self.h):
            for j in range(self.w):
                pixelValue = self.img[i][j]

                if pixelValue > thr:
                    tmpObject.append(pixelValue)
                else:
                    tmpBackGround.append(pixelValue)

        return tmpObject, tmpBackGround

    # start algorithm until old threshold = new threshold
    def run_algorithm(self):
        ba = self.get_init_back_ground()
        ob = self.get_init_object()
        initThre = self.get_mean2(ba, ob)

        while True:
            obj, back = self.get_object_backGround(initThre)
            finalThre = self.get_mean2(self.get_mean_list(obj),
                                       self.get_mean_list(back))
            if initThre == finalThre:
                self.optimalThreshold = initThre
                break
            else:
                initThre = finalThre

    @staticmethod
    def get_mean_list(lists):
        return sum(lists) / len(lists)

    @staticmethod
    def get_vector_mean(vector):
        return sum(vector) / len(vector)

    @staticmethod
    def get_mean2(num1, num2):
        return int((num1 + num2) / 2)

    # general method to draw image with threshold
    @staticmethod
    def get_binary_image(image, threshold):
        h, w = image.shape
        for i in range(h):
            for j in range(w):
                if image[i][j] > threshold:
                    image[i][j] = 255
                else:
                    image[i][j] = 0

        return image
