from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import view, model, static

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
        pass

    def show_model_structure(self):
        pass

    def show_accuracy_and_loss(self):
        pass

    def predict(self):
        pass

    def reset(self):
        pass

    def load_image_resnet50(self):
        pass

    def show_images(self):
        pass

    def show_model_structure_resnet50(self):
        pass

    def show_comparison(self):
        pass

    def inference(self):
        pass

