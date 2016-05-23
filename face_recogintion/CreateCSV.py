"""
@author : opencv.org
@purpose : implement a an algorithm to create csv indexing of files with ui interface
@modifier : Mohamed Hosny Ahmed
"""

import sys
import os.path
from PyQt4 import QtCore, QtGui

class CreateCSV(QtGui.QWidget):
    def __init__(self):
        super(CreateCSV, self).__init__()

        self.path = ''
        self.endPath = ''
        self.dataPaths = []
        self.count = 0
        self.initUi()

    def initUi(self):
        self.pathLabel = QtGui.QLabel('Enter dataset path')
        self.endLabel = QtGui.QLabel('Enter csv target path')
        self.pathEdit = QtGui.QLineEdit()
        self.endEdit = QtGui.QLineEdit()
        self.but = QtGui.QPushButton('Run')

        self.hBox1 = QtGui.QHBoxLayout()
        self.hBox2 = QtGui.QHBoxLayout()
        self.vBox = QtGui.QVBoxLayout()
        self.gBox = QtGui.QGridLayout()

        self.hBox1.addWidget(self.pathLabel)
        self.hBox1.addWidget(self.pathEdit)
        self.vBox.addLayout(self.hBox1)

        self.hBox2.addWidget(self.endLabel)
        self.hBox2.addWidget(self.endEdit)
        self.vBox.addLayout(self.hBox2)

        self.gBox.addLayout(self.vBox, 1, 1)
        self.gBox.addWidget(self.but)
        self.setLayout(self.gBox)

        self.but.clicked.connect(self.butClicked)

        self.setGeometry(300, 150, 500, 100)
        self.setWindowTitle('Optimal')
        self.show()

    def butClicked(self):
        self.path = self.pathEdit.text()
        self.endPath = self.endEdit.text()
        self.pathEdit.clear()
        self.endEdit.clear()
        self.run()

    def run(self):
        SEPARATOR = ";"
        DATASEPRATOR = "$$"
        label = 0

        for root, dirs, files in os.walk(self.path):
            for directory in dirs:
                subject_path = os.path.join(root, directory)
                for filename in os.listdir(subject_path):
                    absPath = subject_path + "/" + filename
                    current_path = absPath + SEPARATOR + str(label)
                    self.dataPaths.append(current_path)
                    self.count += 1
                self.dataPaths.append(DATASEPRATOR + directory + " count : " + str(self.count))

                parsedFiles = '\n'.join(self.dataPaths)
                textFile = open(self.endPath + directory + ".csv", 'w')
                textFile.write(parsedFiles)
                textFile.close()

                label += 1

    def get_images(self):
        return self.dataPaths


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ui = CreateCSV()
    sys.exit(app.exec())
