"""
Purpose: Detect Feature using Harris operator
Author: Mohamed Hosny Ahmed
Date: 10 / 4 / 2016
"""

import cv2
import numpy as np
import myopencv
import cvarthmetics


class HarrisDetector:

    def __init__(self, image):
        self.img = image
        self.rows, self.cols, self.channels = img.shape

        # initialize data
        self.tmpImg = np.zeros((self.rows, self.cols), np.float32)
        self.Ix2 = np.zeros((self.rows, self.cols), np.float32)
        self.Iy2 = np.zeros((self.rows, self.cols), np.float32)
        self.Ixy2 = np.zeros((self.rows, self.cols), np.float32)
        self.H = np.zeros((2, 2), np.float32)
        self.myCV = 0
        self.R = 0
        self.flag = 1
        self.calc_harris_operator()

    def calc_harris_operator(self):
        # check if we are in this method
        print ('>> calc_harris_operator')

        # 1- Compute gradient at x, y
        self.myCV = myopencv.cvUtiliy(self.img)
        self.Ix2 = self.myCV.calc_changes('x', 3)
        self.Iy2 = self.myCV.calc_changes('y', 3)

        # 2- Compute Ix^2, Iy^2, Ixy^2
        self.Ix2 = np.multiply(self.Ix2, self.Ix2)
        self.Iy2 = np.multiply(self.Iy2, self.Iy2)
        self.Ixy2 = np.multiply(self.Ix2, self.Iy2)

        # 3- Sum all pixels by bluring image by one to get summation with window 3x3
        self.Ix2 = self.myCV.summ_pixel_window(self.Ix2, 3)
        self.Iy2 = self.myCV.summ_pixel_window(self.Iy2, 3)
        self.Ixy2 = self.myCV.summ_pixel_window(self.Ixy2, 3)

        # 4- Define H(x,y) matrix
        for row in range(0, self.rows):
            for col in range(0, self.cols):
                self.H[0][0] = self.Ix2[row][col]
                self.H[0][1] = self.Ixy2[row][col]
                self.H[1][0] = self.Ixy2[row][col]
                self.H[1][1] = self.Iy2[row][col]

                # 5- Compute R = det(H(wx,y)) - K*Trace(H(x,y))^2
                self.R = self.harris_response(self.H)
                if self.R < -10000:
                    self.tmpImg[row, col] = float(self.R)
                else:
                    self.tmpImg[row, col] = 0.0

        self.tmpImg = np.array(self.tmpImg, np.float32)
        self.tmpImg = cvarthmetics.Arthmetics.get_local_maxima(self.tmpImg)

        return self.tmpImg

    # get harris response
    def harris_response(self, H):
        # check if we are in this method
        if self.flag == 1:
            print('>> harris_response')
            self.flag = 0

        trace = cv2.trace(H)
        response = cv2.determinant(H) - 0.04 * np.power(trace[0], 2)
        return response

    # return harris index's
    def get_harris(self):
        # check if we are in this method
        print('>> get_harris')
        x = self.tmpImg * self.img
        return self.tmpImg * self.img


if __name__ == '__main__':

    x=0
    img = cv2.imread('/home/prof/Work_Space/CV/Assignment_3_CV/grad.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    harris = HarrisDetector(image=gray)
    c = harris.get_harris()

    # print c.max()
    xx = c > 0.004*c.max()
    img[xx] = [0, 0, 255]

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
