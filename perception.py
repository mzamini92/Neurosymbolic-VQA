import numpy as np
import pandas as pd
import os
import dlib
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms as tfms
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

import torch
import torch.nn as nn
from tqdm import tqdm
import dlib


import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_position_to_draw(text, point, font_face, font_scale, thickness):
    """Gives the coordinates to draw centered"""

    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)


def detect_shape(contour):
    """Returns the shape (e.g. 'triangle', 'square') from the contour"""

    detected_shape = '-----'

    # Calculate perimeter of the contour:
    perimeter = cv2.arcLength(contour, True)

    # Get a contour approximation:
    contour_approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

    # Check if the number of vertices is 3. In this case, the contour is a triangle
    if len(contour_approx) == 3:
        detected_shape = 'triangle'

    # Check if the number of vertices is 4. In this case, the contour is a square/rectangle
    elif len(contour_approx) == 4:

        # We calculate the aspect ration from the bounding rect:
        x, y, width, height = cv2.boundingRect(contour_approx)
        aspect_ratio = float(width) / height

        # A square has an aspect ratio close to 1 (comparison chaining is used):
        if 0.90 < aspect_ratio < 1.10:
            detected_shape = "square"
        else:
            detected_shape = "rectangle"

    # Check if the number of vertices is 5. In this case, the contour is a pentagon
    elif len(contour_approx) == 5:
        detected_shape = "pentagon"

    # Check if the number of vertices is 6. In this case, the contour is a hexagon
    elif len(contour_approx) == 6:
        detected_shape = "hexagon"
    elif len(contour_approx) == 8:
        detected_shape = "star"

    # The shape as more than 6 vertices. In this example, we assume that is a circle
    else:
        detected_shape = "circle"
    # return the name of the shape and the found vertices
    return detected_shape, contour_approx


def array_to_tuple(arr):
    """Converts array to tuple"""

    return tuple(arr.reshape(1, -1)[0])


def draw_contour_points(img, cnts, color):
    """Draw all points from a list of contours"""

    for cnt in cnts:
        print(cnt.shape)
        squeeze = np.squeeze(cnt)
        print(squeeze.shape)

        for p in squeeze:
            pp = array_to_tuple(p)
            cv2.circle(img, pp, 10, color, -1)

    return img


def draw_contour_outline(img, cnts, color, thickness=1):
    """Draws contours outlines of each contour"""

    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)


import matplotlib.pyplot as plt

def show_img_with_matplotlib(color_img, title):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    # Create a new figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img_RGB)

    # Set the title and turn off axis labels
    ax.set_title(title)
    ax.axis('off')

    # Show the plot
    plt.show()



class ObjectDetector:
    def __init__(self, options=None):
        # Initialize object detector options
        self.options = options
        if self.options is None:
            self.options = dlib.simple_object_detector_training_options()

    def fit(self, train_data, filename='detector.svm', visualize=False):
        '''Trains an object detector (HOG + SVM) and saves the model'''
        # Separate the images and bounding boxes in different lists.
        images = [val[0] for val in train_data.values()]
        bounding_boxes = [val[1] for val in train_data.values()]

        # Train the model
        detector = dlib.train_simple_object_detector(images, bounding_boxes, self.options)

        # Visualize HOG
        if visualize:
            win = dlib.image_window()
            win.set_image(detector)
            dlib.hit_enter_to_continue()

        # Check results
        results = dlib.test_simple_object_detector(images, bounding_boxes, detector)
        print(f'Training Results: {results}')

        # Save model
        detector.save(filename)
        print(f'Saved the model to {filename}')

    def predict(self, test_data):
        '''Tests an object detector (HOG + SVM) on test data'''
        # Separate the images and bounding boxes in different lists.
        images = [val[0] for val in test_data.values()]
        bounding_boxes = [val[1] for val in test_data.values()]

        # Load the trained detector
        detector = dlib.simple_object_detector('detector.svm')

        # Test the detector on test data
        results = dlib.test_simple_object_detector(images, bounding_boxes, detector)
        print(f'Test Results: {results}')

        # Draw and annotate the predicted bounding boxes on the images
        for i in range(len(images)):
            image = images[i]
            bboxes = bounding_boxes[i]

            # Convert image to RGB (assuming it's in BGR format)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect objects using the detector
            detected_bboxes = detector(image_rgb)


            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply cv2.threshold() to get a binary image:
            ret, thresh = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY)
            # Find contours using the thresholded image:
            # Note: cv2.findContours() has been changed to return only the contours and the hierarchy
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            
            # Draw and annotate predicted bounding boxes on the image
            for bbox in detected_bboxes:
                x, y, w, h = bbox.left(), bbox.top(), bbox.width(), bbox.height()
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, 'Obj', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display the image with bounding boxes
            #cv2.imshow(f'Test Image {i+1}', image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #color_img = cv2.imread(image)
            title = 'Image Title'
            show_img_with_matplotlib(image, title)

            
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

import wandb



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # For progress bar during training

class ShapeClassifier(nn.Module):
    '''Simple CNN based Image Classifier for Shapes (circle | rectangle)'''
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 36, 3, stride=2, padding=1),
                                  nn.BatchNorm2d(36),
                                  nn.Conv2d(36, 36, 3, stride=2, padding=1),
                                  nn.BatchNorm2d(36),
                                  nn.Conv2d(36, 36, 3, stride=2, padding=1),
                                  nn.BatchNorm2d(36))
        self.fc = nn.Linear(900, 8)
    
    def forward(self, x):
        '''Forward Pass'''
        # batch_size (N)
        N = x.size()[0]
        # Extract features with CNN
        x = self.conv(x)
        # Classifier head
        x = self.fc(x.reshape(N, -1))
        
        #return x
        return F.softmax(x, dim=1)  # Apply softmax activation to get probabilities for each shape

    def calculate_fc_input_size(self):
        # Use a dummy tensor to calculate the size after the convolutional layers
        input_tensor = torch.randn(1, 1, 36, 36)  # Adjust the size according to your input image size
        output_tensor = self.conv(input_tensor)
        output_tensor = F.adaptive_avg_pool2d(output_tensor, (1, 1))
        fc_input_size = output_tensor.view(output_tensor.size(0), -1).size(1)
        return fc_input_size
    
    def train_classifier(self, train_loader, test_loader, lr=0.001, epochs=10, filename='classifier.pth', device=None):
        '''Train the shape classifier'''
        # Automatically set device if not provided
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize lists to store loss and accuracy values for train and test data
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        # Initialize wandb
        wandb.init(project="NSAI")

        # Mount to device
        self.to(device)

        # Create optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()  # Adjusted for multi-class classification

        self.train()
        # Start Training
        for epoch in range(epochs):
            pbar = tqdm(total=len(train_loader), desc='Epoch {}'.format(epoch+1))
            losses = []

            # Initialize variables for tracking loss and correct predictions
            epoch_loss = 0.0
            correct_predictions = 0

            for i, (image, label) in enumerate(train_loader):
                # Mount to device
                image, label = image.to(device).float(), label.to(device)

                # Forward prop
                out = self(image)

                # Loss
                loss = criterion(out, label)  # Adjusted for multi-class classification

                # Update total loss
                epoch_loss += loss.item() * image.size(0)

                # Update correct predictions
                _, predicted_labels = torch.max(out, 1)
                correct_predictions += (predicted_labels == label).sum().item()


                # Backprop and Optimization
                #optimizer.zero_grad()  # Clear gradients
                loss.backward()
                optimizer.step()

                # Verbose
                losses.append(loss.item())
                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})

            # Calculate epoch-level loss and accuracy
            epoch_loss /= len(train_loader.dataset)
            epoch_accuracy = correct_predictions / len(train_loader.dataset)

            # Append loss and accuracy values to the lists
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

            # Log metrics to wandb for each epoch
            wandb.log({"Train Loss": epoch_loss, "Train Accuracy": epoch_accuracy})

            print(f'Epoch {epoch+1}: Mean Loss = {sum(losses)/len(losses)}')
            pbar.close()

        # Validation on the test set after all epochs
        self.eval()
        correct_test = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device).float(), labels.to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)

                _, predicted = outputs.max(1)
                correct_test += predicted.eq(labels).sum().item()

                test_loss += loss.item() * images.size(0)
                

            # Calculate test accuracy and loss after all epochs
            test_accuracy = correct_test / len(test_loader.dataset)
            test_loss /= len(test_loader.dataset)

            test_accuracies.append(test_accuracy)
            test_losses.append(test_loss)

            # Log metrics to wandb for test set
            wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy})
            print(f'Test Loss = {test_loss:.4f} Test Accuracy = {test_accuracy:.2f}')

        # Save model after all epochs
        model_path = os.path.join(filename)
        torch.save(self.state_dict(), model_path)

        # Plot loss and accuracy for train and test data
#         plt.figure(figsize=(10, 4))

#         plt.subplot(1, 2, 1)
#         plt.plot(range(1, epochs + 1), train_losses, label='Train')
#         plt.plot(range(1, epochs + 1), test_losses, label='Test')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.title('Training and Test Loss')
#         plt.legend()

#         plt.subplot(1, 2, 2)
#         plt.plot(range(1, epochs + 1), train_accuracies, label='Train')
#         plt.plot(range(1, epochs + 1), test_accuracies, label='Test')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.title('Training and Test Accuracy')
#         plt.legend()

#         plt.tight_layout()
#         plt.show()

shapes = ['triangle', 'circle', 'hexagon', 'square', 'pentagon', 'rectangle', 'star', 'octagon']
        

class Binarize(object):
    def __init__(self):
        '''Converts Grayscale to Binary (except white every other color is zeroed)'''
        pass
    
    def __call__(self, img_tensor):
        '''
        Args:
            img_tensor (tensor): 0-1 scaled tensor with 1 channel
        Returns:
            tensor
        '''
        return (img_tensor > 0.95).float()

class PerceptionPipe():
    '''
    Full Perception Pipeline i.e.
    detector -> attribute extraction -> structural scene representation
    '''
    def __init__(self, detector_file, classifer_file, device='cpu'):
        # Object detector
        self.detector = dlib.simple_object_detector(detector_file)
        
        # Shape Classifier
        self.classifier = ShapeClassifier().to(device)

        self.classifier.load_state_dict(torch.load(classifer_file))
        self.device = device
        
        self.colors = np.array([[0,0,255], [0,255,0], [255,0,0], 
                               [0,156,255], [128,128,128], [0,255,255]])
        
        self.idx2color = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange', 4: 'gray', 5: 'yellow'}
        self.preproc = tfms.Compose([tfms.Grayscale(),
                                     tfms.Resize((40, 40)),
                                     tfms.ToTensor()])
    
    
    def detect(self, img):
        '''Detects and Returns Objects and its centers'''
        # Detect
        detections = self.detector(img)
        objects = []

        print('detectionsdetectionsdetections',detections)
        for detection in detections:
            # Get the bbox coords
            x1, y1 = int(detection.left()), int(detection.top())
            x2, y2 = int(detection.right()), int(detection.bottom())
            
            # Clip negative values to zero
            x1, y1, x2, y2 = np.array([x1, y1, x2, y2]).clip(min=0).tolist()
            

            # Find the center
            center = (int((x1+x2)/2), int((y1+y2)/2))

            # Crop the individual object
            obj = img[y1:y2, x1:x2]

            objects.append((obj, center))
            

        return objects
    


    def extract_attributes(self, x_img, prob=0.5, debug=False):
        '''Returns the shape and color of a given object'''
        image = Image.fromarray(cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB))
        img = self.preproc(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.classifier(img).squeeze()
            print(out,'OUTTT')
            if debug:
                print(out)

        # Get the predicted shape class index with the highest probability
        
        shape_id = torch.argmax(out).item()
        shape = shapes[shape_id]  # Assuming you have shape labels like 'shape_0', 'shape_1', ...

        # Extract Color
        center_pixel = (x_img[20, 20, :]).astype('int')
        color_id = cosine_similarity(center_pixel.reshape(1, -1), self.colors).argmax()
        color = self.idx2color[color_id]

        return shape, color
    
    def scene_repr(self, img, prob=0.5, debug=False):
        '''Returns a structured scene representation as a dataframe'''
        # Perform object detection and get the objects
        objects = self.detect(img)
        
        # Init Scene representation
        scene_df = pd.DataFrame(columns=['shape', 'color', 'position'])
        
        for obj, center in objects:
            shape, color = self.extract_attributes(obj, prob, debug)
            scene_df = scene_df.append({'shape': shape, 
                                        'color': color, 
                                        'position': center}, ignore_index=True)
        
        return scene_df
