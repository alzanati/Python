"""
@purpose: uiInterface to thresholding class
@author: Mohamed Hosny Ahmed
@date: 11 / 5 / 2016
"""

import sys
import cv2
from PyQt4 import QtCore, QtGui
import kmeans_test
import kmeans_test


class UiThreshold(QtGui.QWidget):
    def __init__(self):
        super(UiThreshold, self).__init__()

        self.path = None
        self.im = None

        self.threshold = None
        self.optimal = None
        self.pathLabel = None
        self.pathEdit = None
        self.but = None
        self.hBox = None

        self.initUi()
        # self.initAlgorithm()

    def initUi(self):
        self.pathLabel = QtGui.QLabel('Enter image path')
        self.pathEdit = QtGui.QLineEdit()
        self.but = QtGui.QPushButton('Run')
        self.hBox = QtGui.QHBoxLayout()

        self.hBox.addWidget(self.pathLabel)
        self.hBox.addWidget(self.pathEdit)
        self.hBox.addWidget(self.but)
        self.setLayout(self.hBox)

        self.but.clicked.connect(self.butClicked)

        self.setGeometry(300, 150, 400, 100)
        self.setWindowTitle('Optimal')
        self.show()

    def butClicked(self):
        self.path = self.pathEdit.text()
        self.pathEdit.setText(' ')
        self.initAlgorithm(self.path)

    def initAlgorithm(self, path):
        self.im = cv2.imread(path)
        self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
        self.kmean = kmeans_test.kmeans(self.im, k=3, iterations=10)
        self.kmean.run_algorithm()

def main():
    app = QtGui.QApplication(sys.argv)
    ui = UiThreshold()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()