"""
@purpose: test face recognition
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

if __name__ == '__main__':
    im = cv2.imread("/home/mohamed/Pictures/hwa_feh_eih.png")
    cv2.namedWindow(winname='image',flags=cv2.WINDOW_AUTOSIZE)
    cv2.imshow(winname='image', mat=im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
