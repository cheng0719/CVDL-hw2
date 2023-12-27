from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import view, model, static
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class Controller:
    def __init__(self):
        self.view = view.View()
        self.model = model.Model()
        self.static = static.Static()

        # Load Image
        self.view.ui.Load_Image_pushButton.clicked.connect(self.load_image)
        # Load Video
        self.view.ui.Load_Video_pushButton.clicked.connect(self.load_video)

        # 1. Background Subtraction
        self.view.ui.Background_Subtraction_pushButton.clicked.connect(self.background_subtraction)

        # 2. Optical Flow
        # Preprocessing
        self.view.ui.Preprocessing_pushButton.clicked.connect(self.preprocessing)
        # Video Tracking
        self.view.ui.Video_tracking_pushButton.clicked.connect(self.video_tracking)

        # 3. PCA
        # Dimension Reduction
        self.view.ui.Dimension_Reduction_pushButton.clicked.connect(self.dimension_reduction)

        # 4. MNIST Classifier Using VGG19
        # Show Model Structure
        self.view.ui.Show_Model_Structure_VGG19_pushButton.clicked.connect(self.show_model_structure)
        # Show Accuracy and Loss
        self.view.ui.Show_Accuracy_and_Loss_pushButton.clicked.connect(self.show_accuracy_and_loss)
        # Predict
        self.view.ui.Predict_pushButton.clicked.connect(self.predict)
        # Reset
        self.view.ui.Reset_pushButton.clicked.connect(self.reset)

        # 5. ResNet50
        # Load Image
        self.view.ui.Load_Image_ResNet50_pushButton.clicked.connect(self.load_image_resnet50)
        # Show Images
        self.view.ui.Show_Images_pushButton.clicked.connect(self.show_images)
        # Show Model Structure
        self.view.ui.Show_Model_Structure_ResNet50_pushButton.clicked.connect(self.show_model_structure_resnet50)
        # Show Comparison
        self.view.ui.Show_Comparison_pushButton.clicked.connect(self.show_comparison)
        # Inference
        self.view.ui.Inference_pushButton.clicked.connect(self.inference)

        self.view.show()

    def load_image(self):
        filePath, filterType = QtWidgets.QFileDialog.getOpenFileName()
        self.static.imgPath = filePath

    def load_video(self):
        filePath, filterType = QtWidgets.QFileDialog.getOpenFileName()
        self.static.vidPath = filePath

    def background_subtraction(self):
        if self.static.vidPath == '':
            QMessageBox.warning(self.view, 'Warning', 'Please load video first!')
            return
        else:
            self.model.background_subtraction(self.static.vidPath)

    def preprocessing(self):
        if self.static.vidPath == '':
            QMessageBox.warning(self.view, 'Warning', 'Please load video first!')
            return
        else:
            self.model.preprocessing(self.static.vidPath)

    def video_tracking(self):
        if self.static.vidPath == '':
            QMessageBox.warning(self.view, 'Warning', 'Please load video first!')
            return
        else:
            self.model.video_tracking(self.static.vidPath)

    def dimension_reduction(self):
        if self.static.imgPath == '':
            QMessageBox.warning(self.view, 'Warning', 'Please load image first!')
            return
        else:
            self.model.dimension_reduction(self.static.imgPath)

    def show_model_structure(self):
        self.model.show_model_structure()

    def show_accuracy_and_loss(self):
        self.model.show_accuracy_and_loss()

    def predict(self):
        # Get the image data from the graffiti board
        pil_image = self.view.graffiti_board.get_image_data()

        # Show the image using Pillow
        # pil_image.show()

        output, probabilities = self.model.predict(pil_image)

        # Display output in the text box
        output_label_vgg = QtWidgets.QLabel(str(output), self.view.ui.MNIST_Classifier_using_VGG19_groupBox)
        output_label_vgg.setGeometry(QtCore.QRect(145, 230, 151, 31))
        output_label_vgg.setObjectName("output_label_vgg")
        output_label_vgg.show()

        # Get class labels (0 to 9)
        class_labels = list(range(10))

        # Plot the probabilities
        plt.bar(class_labels, probabilities.cpu().numpy())
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Probability of each class')
        plt.xticks(class_labels)
        plt.show()

    def reset(self):
        self.view.graffiti_board.clearBoard()

        # Find and remove the existing output_label_vgg
        existing_output_label_vgg = self.view.ui.MNIST_Classifier_using_VGG19_groupBox.findChild(QtWidgets.QLabel, "output_label_vgg")
        if existing_output_label_vgg:
            existing_output_label_vgg.deleteLater()

    def load_image_resnet50(self):
        filePath, filterType = QtWidgets.QFileDialog.getOpenFileName()
        self.static.imgResnetPath = filePath

        # display the image which path is self.static.imgResnetPath in the self.view.ui.ResNet50_graphicsView
        self.view.ui.ResNet50_graphicsView.scene = QtWidgets.QGraphicsScene()
        self.view.ui.ResNet50_graphicsView.scene.addPixmap(QtGui.QPixmap(self.static.imgResnetPath))
        # resize the image to fit the graphics view
        self.view.ui.ResNet50_graphicsView.fitInView(self.view.ui.ResNet50_graphicsView.scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        # set the scene
        self.view.ui.ResNet50_graphicsView.setScene(self.view.ui.ResNet50_graphicsView.scene)

        # Modify the content of output_label_resnet50
        self.view.ui.output_label_resnet50.setText('Prediction : ')
        # Modify the position of output_label_resnet50
        self.view.ui.output_label_resnet50.setGeometry(QtCore.QRect(440, 320, 200, 30))
        # Show the modified output_label_resnet50
        self.view.ui.output_label_resnet50.show()

    def show_images(self):
        self.model.show_images()

    def show_model_structure_resnet50(self):
        self.model.show_model_structure_resnet50()

    def show_comparison(self):
        self.model.show_comparison()

    def inference(self):
        output = self.model.inference(self.static.imgResnetPath)

        # Modify the content of output_label_resnet50
        self.view.ui.output_label_resnet50.setText('Prediction : ' + output)
        # Modify the position of output_label_resnet50
        self.view.ui.output_label_resnet50.setGeometry(QtCore.QRect(425, 320, 200, 30))
        # Show the modified output_label_resnet50
        self.view.ui.output_label_resnet50.show()


