import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import exposure

def resize_image(image_path, target_size=(224, 224)):
    """Resize image to target size"""
    image = load_img(image_path, target_size=target_size)
    return img_to_array(image)

def normalize_image(image):
    """Normalize pixel values to [0,1]"""
    return image.astype('float32') / 255.0

def normalize_staining(image):
    """Normalize H&E staining"""
    # Separate channels
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Normalize a and b channels
    a = exposure.rescale_intensity(a, out_range=(0, 255))
    b = exposure.rescale_intensity(b, out_range=(0, 255))
    
    # Merge channels
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def create_augmentation():
    """Create augmentation pipeline specific for histopathological images"""
    return ImageDataGenerator(
        rotation_range=90,  # Full rotation for microscopy images
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  # Important for microscopy images
        fill_mode='reflect',
        brightness_range=[0.9, 1.1]  # Slight brightness variation
    )

def preprocess_images(images, augment=False):
    """Main preprocessing pipeline"""
    processed = []
    for image in images:
        # Apply staining normalization
        image = normalize_staining(image)
        # Normalize pixel values
        image = normalize_image(image)
        processed.append(image)
    
    processed = np.array(processed)
    
    if augment:
        augmentor = create_augmentation()
        return augmentor.flow(processed, shuffle=False)
    
    return processed