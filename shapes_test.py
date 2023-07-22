import os
import math
import random
import cv2
import numpy as np

# Define the shapes
shapes = {
['red', 'green', 'blue', 'orange', 'gray', 'yellow']
#rgb=["#0000ff","#00ff00","#ff0000","#999999","#00ffff","#333399"]

    "rectangle": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "circle": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "triangle": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "square": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "pentagon": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "hexagon": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "octagon": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "star": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"]
}



# Create a directory to store the shape images
output_dir = "shapes_data_test"
os.makedirs(output_dir, exist_ok=True)

# Generate the images for each shape
for shape, colors in shapes.items():
    shape_dir = os.path.join(output_dir, shape)
    os.makedirs(shape_dir, exist_ok=True)

    for i in range(1, 21):  # Generate 100 images of each shape
        # Create a new image with white background
        image = np.zeros((36, 36, 3), np.uint8) * 255

        # Calculate the center coordinates of the shape
        center = (18, 18)

        # Get a random color for the shape
        color = random.choice(colors)
        color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

        if shape == "circle":
            # Calculate the radius based on the image size
            radius = int(min(image.shape[:2]) // 2.2)
            # Draw the circle filled with the random color
            cv2.circle(image, center, radius, color_rgb, -1)
        elif shape == "rectangle":
            # Define the rectangle dimensions
            width = 18
            height = 12
            # Calculate the coordinates of the top-left and bottom-right corners
            top_left = (center[0] - width + 2, center[1] - height + 2)
            bottom_right = (center[0] + width- 2, center[1] + height - 2)
            # Draw the rectangle filled with the random color
            cv2.rectangle(image, top_left, bottom_right, color_rgb, -1)
        elif shape == "triangle":
            # Calculate the coordinates of the triangle vertices
            vertices = np.array([(center[0], center[1] - 16),
                                 (center[0] - 16, center[1] + 16),
                                 (center[0] + 16, center[1] + 16)], np.int32)
            # Draw the triangle filled with the random color
            cv2.fillPoly(image, [vertices], color_rgb)
        elif shape == "square":
            # Define the square dimensions
            size = 18
            # Calculate the coordinates of the top-left and bottom-right corners
            top_left = (center[0] - size + 3, center[1] - size + 3)
            bottom_right = (center[0] + size - 3, center[1] + size - 3)
            # Draw the square filled with the random color
            cv2.rectangle(image, top_left, bottom_right, color_rgb, -1)
        elif shape == "pentagon":
            size = 15
            # Calculate the coordinates of the five pentagon vertices
            points = np.array([(center[0], center[1] - size),
                              (center[0] - size, center[1] - int(size / 2)),
                              (center[0] - int(size / 2), center[1] + size),
                              (center[0] + int(size / 2), center[1] + size),
                              (center[0] + size, center[1] - int(size / 2))])
            # Draw the pentagon filled with the random color
            cv2.fillPoly(image,  [points], color_rgb)
        elif shape == "hexagon":
            size = 15
            # Calculate the coordinates of the six hexagon vertices
            points = np.array([(center[0] - size, center[1]),
                              (center[0] - int(size / 2), center[1] - int(size * 0.86)),
                              (center[0] + int(size / 2), center[1] - int(size * 0.86)),
                              (center[0] + size, center[1]),
                              (center[0] + int(size / 2), center[1] + int(size * 0.86)),
                              (center[0] - int(size / 2), center[1] + int(size * 0.86))])
            # Draw the hexagon filled with the random color
            cv2.fillPoly(image, [points], color_rgb)
        elif shape == "octagon":
            size = 15
            # Calculate the coordinates of the eight octagon vertices
            points = np.array([(center[0] - size, center[1]),
                              (center[0] - int(size / 1.9), center[1] - int(size / 1.9)),
                              (center[0] + int(size / 1.9), center[1] - int(size / 1.9)),
                              (center[0] + size, center[1]),
                              (center[0] + size, center[1] + int(size / 1.9)),
                              (center[0] + int(size / 1.9), center[1] + size),
                              (center[0] - int(size / 1.9), center[1] + size),
                              (center[0] - size, center[1] + int(size / 1.9))])

            # Draw the octagon filled with the random color
            cv2.fillPoly(image, [points], color_rgb)
        elif shape == "star":
            # Define the size of the star
            size = 15

            # Calculate the coordinates of the five star vertices
            points = np.array([(center[0], center[1] - size),
                              (center[0] - int(size * 0.224), center[1] - int(size * 0.224)),
                              (center[0] - size, center[1]),
                              (center[0] - int(size * 0.224), center[1] + int(size * 0.224)),
                              (center[0], center[1] + size),
                              (center[0] + int(size * 0.224), center[1] + int(size * 0.224)),
                              (center[0] + size, center[1]),
                              (center[0] + int(size * 0.224), center[1] - int(size * 0.224))])
            # Draw the star filled with the random color
            cv2.fillPoly(image, [points], color_rgb)

        # Apply random rotation
        #rotation_angle = random.randint(0, 360)
        #rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        #rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        # Apply random blur
        blur_radius = random.uniform(0, 1)
        blurred_image = cv2.GaussianBlur(image, (5, 5), blur_radius)

        # Apply random contrast
        #contrast_factor = random.uniform(0.5, 1.5)
        #enhanced_image = cv2.convertScaleAbs(blurred_image, alpha=contrast_factor, beta=0)

        # Save the image as a JPG file
        image_path = os.path.join(shape_dir, f"{shape}_{i}.jpg")
        cv2.imwrite(image_path, blurred_image)

