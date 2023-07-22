import os
import math
import random
import cv2
import numpy as np

# Define the shapes
shapes = {
    "rectangle": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "circle": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "triangle": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "square": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "pentagon": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "hexagon": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "octagon": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"],
    "star": ["#0000ff", "#00FF00", "#FF0000", "#0099ff", "#999999","#00ffff"]
}
size = 15
size_2 = 20
size_3=10
def center_generate(objects):
    '''Generate centers of objects'''
    while True:
        pas = True
        center = np.random.randint(15, 21, 2)  # Generate random centers between 15 and 20        
        if len(objects) > 0:
            for name, c, shape, _ in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center




# Create a directory to store the shape images
output_dir = "shapes_data_test"
os.makedirs(output_dir, exist_ok=True)
img_size = 36
# Generate the images for each shape
for shape, colors in shapes.items():

    objects = [] 
    shape_dir = os.path.join(output_dir, shape)
    os.makedirs(shape_dir, exist_ok=True)

    for i in range(1, 100):  # Generate 100 images of each shape
        # Create a new image with white background
        image = np.ones((36, 36, 3), np.uint8) * 255

        # Calculate the center coordinates of the shape
        center = center_generate(objects)
        #center = [18, 18]

        # Get a random color for the shape
        color = random.choice(colors)
        color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

        if shape == "circle":
            # Calculate the radius based on the image size
            radius = int(min(image.shape[:2]) // 3)
            # Draw the circle filled with the random color
            cv2.circle(image, center, radius, color_rgb, -1)
        elif shape == "rectangle":
            # Define the rectangle dimensions
            size = random.randint(5, 10)
            size_2 = random.randint(10, 15)
            # Calculate the coordinates of the top-left and bottom-right corners
            top_left = (center[0]-size, center[1]-(size_2))
            bottom_right = (center[0]+size, center[1]+(size_2))
            # Draw the rectangle filled with the random color
            cv2.rectangle(image, top_left, bottom_right, color_rgb, -1)
        elif shape == "triangle":
            # Generate random sizes for the triangle
            size = random.randint(10, 15)
            size_2 = random.randint(10, 20)

            # Calculate the coordinates of the triangle vertices based on the random sizes
            vertices = np.array([(center[0], center[1] + size_2),
                                 (center[0] - size // 2, center[1]),
                                 (center[0] + size // 2, center[1])])

            # Draw the triangle filled with the random color on the background
            cv2.fillPoly(image, [vertices], color_rgb)
        elif shape == "square":
            # Define the square dimensions
            size = random.randint(10, 15)
            # Calculate the coordinates of the top-left and bottom-right corners
            top_left = (center[0]-size, center[1]-size)
            bottom_right = (center[0]+size, center[1]+size)
            # Draw the square filled with the random color
            cv2.rectangle(image, top_left, bottom_right, color_rgb, -1)
        elif shape == "pentagon":
            size = random.randint(10, 15)
            # Calculate the coordinates of the five pentagon vertices
            points = np.array([(center[0], center[1] - size),
                              (center[0] - size, center[1] - int(size / 2)),
                              (center[0] - int(size / 2), center[1] + size),
                              (center[0] + int(size / 2), center[1] + size),
                              (center[0] + size, center[1] - int(size / 2))])
            # Draw the pentagon filled with the random color
            cv2.fillPoly(image,  [points], color_rgb)
        elif shape == "hexagon":
            size = random.randint(10, 15)
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
            size = random.randint(10, 15)
            # Calculate the coordinates of the eight octagon vertices
            points = np.array([(center[0] - size, center[1]),
                              (center[0] - int(size / 2), center[1] - int(size / 2)),
                              (center[0] + int(size / 2), center[1] - int(size / 2)),
                              (center[0] + size, center[1]),
                              (center[0] + size, center[1] + int(size / 2)),
                              (center[0] + int(size / 2), center[1] + size),
                              (center[0] - int(size / 2), center[1] + size),
                              (center[0] - size, center[1] + int(size / 2))])

            # Draw the octagon filled with the random color
            cv2.fillPoly(image, [points], color_rgb)
        elif shape == "star":
            # Generate random sizes for the star
            size = random.randint(10, 15)

            # Calculate the coordinates of the five star vertices based on the random size
            points = np.array([(center[0], center[1] - size),
                               (center[0] - int(size * 0.224), center[1] - int(size * 0.224)),
                               (center[0] - size, center[1]),
                               (center[0] - int(size * 0.224), center[1] + int(size * 0.224)),
                               (center[0], center[1] + size),
                               (center[0] + int(size * 0.224), center[1] + int(size * 0.224)),
                               (center[0] + size, center[1]),
                               (center[0] + int(size * 0.224), center[1] - int(size * 0.224))])

            # Draw the star filled with the random color on the background
            cv2.fillPoly(image, [points], color_rgb)

        # Apply random rotation
        centers = (18, 18)
        #rotation_angle = random.randint(0, 360)
        #rotation_matrix = cv2.getRotationMatrix2D(centers, rotation_angle, 1.0)
        #rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        # Apply random rotation
        #rotation_angle = random.randint(0, 360)
        #rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        #rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        # Apply random blur
        blur_radius = random.uniform(0, 1)
        blurred_image = cv2.GaussianBlur(image, (5, 5), blur_radius)

        # Apply random contrast
        contrast_factor = random.uniform(0.5, 1.5)
        enhanced_image = cv2.convertScaleAbs(blurred_image, alpha=contrast_factor, beta=0)

        # Save the image as a JPG file
        image_path = os.path.join(shape_dir, f"{shape}_{i}.png")
        cv2.imwrite(image_path, image)

