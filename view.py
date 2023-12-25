from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ui_hw2 import Ui_MainWindow, GraffitiBoard

class View(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.graffiti_board = GraffitiBoard()
        self.graffiti_board.setGeometry(QtCore.QRect(290, 30, 351, 271))
        self.graffiti_board.setParent(self.ui.MNIST_Classifier_using_VGG19_groupBox)


# class View(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.ui = MainWindow()
#         # self.ui.setupUi(self)