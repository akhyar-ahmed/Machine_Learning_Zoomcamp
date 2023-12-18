import traceback
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np

from PIL import Image
from skimage import feature
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


# Function to get image size
def get_image_size(image_path):
    # Open an image file
    img = Image.open(image_path)
    # Get image size
    width, height = img.size
    print(f"The image width is {width} and the image height is {height}\n")
    return width, height


# Function to plot class distribution
def plot_class_distribution(directory):
    class_counts = [len(os.listdir(os.path.join(directory, class_folder))) for class_folder in os.listdir(directory)]
    class_labels = os.listdir(directory)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_labels, y = class_counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    # Save the figure
    try:
        plt.savefig('plots/class_distribution.png', bbox_inches = 'tight')
        print("Class distribution figure saved successfully!\n")
    except Exception as e:
        print("Class distribution figure can't be saved!\n")
        print(e)


# Function to display color distribution
def create_color_distribution(img):
    colors = ('b', 'g', 'r')
    
    plt.figure(figsize = (10, 5))
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color = color)
    
    plt.title('Color Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    # Save the figure
    try:
        plt.savefig('plots/color_distribution_of_an_image.png', bbox_inches = 'tight')
        print("Color distribution figure saved successfully!\n")
    except Exception as e:
        print("Color distribution figure can't be saved!\n")
        print(e)


# Function to display texture patterns
def create_texture_patterns(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = feature.canny(gray, sigma=3)
    
    plt.figure(figsize = (10, 5))
    plt.imshow(edges, cmap = 'gray')
    plt.title('Texture Patterns (Canny Edge Detection)')
    plt.axis('off')
    # Save the figure
    try:
        plt.savefig('plots/texture_patterns.png', bbox_inches = 'tight')
        print("Texture patterns image figure saved successfully!\n")
    except Exception as e:
        print("Texture patterns figure can't be saved!\n")
        print(e)


# Function to create color distribution and texture patterns for a sample image
def do_content_analysis(sample_image_path, height, width):        
    sample_img = cv2.imread(sample_image_path)

    create_color_distribution(sample_img)
    create_texture_patterns(sample_img)

    # Data augmentation exploration
    augmentation_datagen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest'
    )

    # Generate augmented images for a sample image
    img = image.load_img(sample_image_path, target_size=(height, width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    augmented_images = augmentation_datagen.flow(x, batch_size=1)

    num_images = len(augmented_images)
    num_images = max(num_images, 4)

    plt.figure(figsize=(12, 4))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        img = np.squeeze(augmented_images[0])
        plt.imshow(image.array_to_img(img))
        plt.axis('off')

    plt.suptitle('Data Augmentation Exploration')
    # Save the figure
    try:
        plt.savefig('plots/data_augmentation_exploration.png', bbox_inches = 'tight')
        print("Augmentation exploration figure saved successfully!\n")
    except Exception as e:
        print("Augmentation exploration figure can't be saved!\n")
        print("Exception occurred:\n", e)
        print("Here's the full traceback:")
        traceback.print_exc()