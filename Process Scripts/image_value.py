import cv2
import os
import numpy as np

# Define the folder path containing the images
folder_path = '/Users/youngj/Local/Project/2022_Rolling_Bearing_Fault_Diagnosis/CWT'

# Create an empty dictionary to store the entropy values of each image
entropy_dict = {}

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image (assuming all image files have extension .jpg)
    if filename.endswith('.png'):
        # Load the grayscale image
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)

        # Compute the histogram of the image
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])

        # Normalize the histogram to obtain a probability distribution function
        pdf = hist / np.sum(hist)

        # Calculate the entropy of the image
        entropy = -np.sum(pdf * np.log2(pdf + 1e-10))  # Adding a small value to avoid log(0))

        # Store the entropy value in the dictionary
        entropy_dict[filename] = entropy

# Print the entropy values for each image
for filename, entropy in entropy_dict.items():
    print('Image:', filename, ', entropy:', entropy, 'bits per pixel')