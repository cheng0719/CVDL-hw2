import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torchsummary import summary
from PIL import Image

# Define a VGG19 model with batch normalization
class VGG19BN(nn.Module):
    def __init__(self):
        super(VGG19BN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10) # 10 means 10 classification classes
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Model:
    def __init__(self):
        pass

    def background_subtraction(self, vidPath):
        cap = cv2.VideoCapture(vidPath)
        ret, frame = cap.read()
        if not ret:
            return
        
        # Create background subtractor
        history = 500
        dist2Threshold = 400
        subtractor = cv2.createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows=True)
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Blur frame
            blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
            
            # Get background mask
            mask = subtractor.apply(blurred_frame)

            # Generate Frame (R) with only moving object by cv2.bitwise_and
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Show the frame with the background mask
            cv2.imshow('frame', frame)
            cv2.imshow('mask', mask)
            cv2.imshow('result', result)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        

    def preprocessing(self, vidPath):
        cap = cv2.VideoCapture(vidPath)
        ret, frame = cap.read()
        if not ret:
            return
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Adjust parameters for goodFeaturesToTrack
        maxCorners = 1
        qualityLevel = 0.3
        minDistance = 7
        blockSize = 7
        
        # Detect corners using goodFeaturesToTrack
        corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance, blockSize)
        
        if corners is not None:
            # Get the coordinates of the corner
            x, y = corners[0][0]
            
            # Draw a red cross mark at the corner point, set the length of the line to 20 pixels, and the line thickness to 4 pixels
            cv2.line(frame, (int(x)-10, int(y)), (int(x)+10, int(y)), (0, 0, 255), 4)
            cv2.line(frame, (int(x), int(y)-10), (int(x), int(y)+10), (0, 0, 255), 4)
        
        # Show the frame with the cross mark
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 960, 540)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def video_tracking(self, vidPath):
        cap = cv2.VideoCapture(vidPath)
        ret, prev_frame = cap.read()
        if not ret:
            return
        
        # Convert previous frame to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Adjust parameters for goodFeaturesToTrack
        maxCorners = 1
        qualityLevel = 0.3
        minDistance = 7
        blockSize = 7
        
        # Detect corners using goodFeaturesToTrack
        prev_corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners, qualityLevel, minDistance, blockSize)
        
        # Create an empty mask image
        mask = np.zeros_like(prev_frame)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert current frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow using cv2.calcOpticalFlowPyrLK
            next_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_corners, None)
            
            # Select good points
            good_new = next_corners[status == 1]
            good_old = prev_corners[status == 1]
            
            # Draw trajectory lines
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 100, 255), 2)
                frame = cv2.line(frame, (int(a)-10, int(b)), (int(a)+10, int(b)), (0, 0, 255), 4)
                frame = cv2.line(frame, (int(a), int(b)-10), (int(a), int(b)+10), (0, 0, 255), 4)
                # frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 100, 255), -1)
            
            # Overlay trajectory lines on the frame
            output = cv2.add(frame, mask)
            
            # Show the frame with the trajectory lines
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 960, 540)
            cv2.imshow('frame', output)
            
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break
            
            # Update previous frame and corners
            prev_gray = gray.copy()
            prev_corners = good_new.reshape(-1, 1, 2)
        
        cap.release()
        cv2.destroyAllWindows()

    def dimension_reduction(self, imgPath):
        # Step 1: Convert RGB image to gray scale image
        img = cv2.imread(imgPath)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Normalize gray scale image
        normalized_img = gray_img / 255.0
        
        # Step 3: Use PCA for dimension reduction
        w, h = gray_img.shape
        min_dim = min(w, h)
        mse_threshold = 3.0
        n = 1
        
        while True:
            pca = PCA(n_components=n)
            reduced_img = pca.inverse_transform(pca.fit_transform(normalized_img.reshape(-1, min_dim)))
            
            # Step 4 : Use MSE(Mean Square Error) to compute reconstruction error
            mse = np.mean(((normalized_img - reduced_img.reshape(w, h))) ** 2 * 255.0 * 255.0)

            print("n: {}, MSE: {}\n".format(n, mse))
            if mse <= mse_threshold or n >= min_dim:
                break
            
            n += 1
        
        print("Minimum n value:", n)
        
        # Step 5: Plot the gray scale image and the reconstruction image
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(normalized_img, cmap='gray')
        axs[0].set_title("Gray Scale Image")
        axs[0].axis('off')
        
        axs[1].imshow(reduced_img.reshape(w, h), cmap='gray')
        axs[1].set_title("Reconstruction Image (n={})".format(n))
        axs[1].axis('off')
        
        plt.show()
    
    def show_model_structure(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vgg = models.vgg19_bn().to(device)

        summary(vgg, (3, 224, 224))
    
    def show_accuracy_and_loss(self):
        pass

    def predict(self, img):
        # Define the transform for the input image
        transform = transforms.Compose([
            # transforms.Resize((224, 224)),  # Resize to VGG19 input size   # WRONG!!!!
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Load the trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg19_bn_model = VGG19BN().to(device)
        vgg19_bn_model.load_state_dict(torch.load('./models/vgg19_bn_mnist_state_dict.pt'))
        vgg19_bn_model.eval()

        input_img_gray = img.convert('L')
        input_tensor = transform(input_img_gray)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output = vgg19_bn_model(input_batch)

        # Get the predicted class
        _, predicted_class = torch.max(output, 1)
        print(f"Predicted class: {predicted_class.item()}")
    
    def show_images(self):
        # Load 2 image using PIL, 1 is the image from ./inference_dataset/cat/1.jpg, the other is the image from ./inference_dataset/dog/1.jpg
        cat = Image.open('./inference_dataset/cat/1.jpg')
        dog = Image.open('./inference_dataset/dog/1.jpg')
        
        # Convert image to RGB
        cat = cat.convert('RGB')
        dog = dog.convert('RGB')
        
        # Convert image to tensor
        transform = transforms.Compose([
            # Resize image to 224x224
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        cat_transformed = transform(cat)
        dog_transformed = transform(dog)
        
        # Show 2 image in 1 window using matplotlib.pyplot.imshow()
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(cat_transformed.permute(1, 2, 0))
        axs[0].set_title("Cat")
        axs[0].axis('off')

        axs[1].imshow(dog_transformed.permute(1, 2, 0))
        axs[1].set_title("Dog")
        axs[1].axis('off')

        plt.show()
    
    def show_model_structure_resnet50(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet50 = models.resnet50().to(device)

        summary(resnet50, (3, 224, 224))
        
        
        