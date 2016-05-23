"""
@author: realpython.com
@purpose: Implement a face detection algorithm with ui interface
@modifier: Mohamed Hosny Ahemd
"""

import cv2
import sys
from PyQt4 import QtCore, QtGui
import gtk

class CascadeClassifier(QtGui.QWidget):
    def __init__(self):
        super(CascadeClassifier, self).__init__()
        self.imagePath = " "
        self.cascadePath = " "
        self.init_ui()

    def init_ui(self):
        #   put all ui tools
        self.imageLabel = QtGui.QLabel('image path: ')
        self.cascadeLabel = QtGui.QLabel('cascade path: ')
        self.imageEditPath = QtGui.QLineEdit()
        self.cascadeEditPath = QtGui.QLineEdit()
        self.detectButton = QtGui.QPushButton('Detect Face')
        self.clearButton = QtGui.QPushButton('Clear')

        #   create layouts
        self.hBox1 = QtGui.QHBoxLayout()
        self.hBox2 = QtGui.QHBoxLayout()
        self.vBox = QtGui.QVBoxLayout()
        self.gBox = QtGui.QGridLayout()

        #   put ui tools in layout
        self.hBox1.addWidget(self.imageLabel)
        self.hBox1.addWidget(self.imageEditPath)
        self.hBox2.addWidget(self.cascadeLabel)
        self.hBox2.addWidget(self.cascadeEditPath)

        self.vBox.addLayout(self.hBox1)
        self.vBox.addLayout(self.hBox2)
        self.gBox.addLayout(self.vBox, 1, 1)
        self.gBox.addWidget(self.detectButton)
        self.gBox.addWidget(self.clearButton)

        #   add to screen
        self.setLayout(self.gBox)

        #   connect button to slot
        self.detectButton.clicked.connect(self.run_algorithm)
        self.clearButton.clicked.connect(self.clearPaths)

        #   finish window
        self.setGeometry(300, 150, 500, 200)
        self.setWindowTitle('cascade classifier')
        self.show()

    def clearPaths(self):
        self.imageEditPath.clear()
        self.cascadeEditPath.clear()

    def run_algorithm(self):
        self.imagePath = self.imageEditPath.text()
        self.cascadePath = self.cascadeEditPath.text()

        #   create classifier
        faceCascade = cv2.CascadeClassifier(self.cascadePath)
        image = cv2.imread(self.imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CvFeatureParams_HAAR
        )
        print("Found {0} faces!".format(len(faces)))

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # show image
        cv2.namedWindow(winname="Faces found", flags=cv2.WINDOW_NORMAL)
        cv2.imshow("Faces found", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    app = QtGui.QApplication(sys.argv)
    cs = CascadeClassifier()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
