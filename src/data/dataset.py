import os
import logging
import requests
import tarfile
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self, data_dir, img_size=(224, 224), magnification='100X', 
                 use_subtypes=False, batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.magnification = magnification
        self.use_subtypes = use_subtypes
        self.batch_size = batch_size
        self.dataset_url = "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
        
        self.classes = {
            'benign': ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma'],
            'malignant': ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
        }

    def download_dataset(self):
        """Download and extract BreakHis dataset"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        archive_path = os.path.join(self.data_dir, "breakhis.tar.gz")
        
        if not os.path.exists(archive_path):
            logger.info("Downloading BreakHis dataset...")
            response = requests.get(self.dataset_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(archive_path, 'wb') as f, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc="Downloading"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

        logger.info("Extracting files...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(self.data_dir)
        os.remove(archive_path)

    def create_dataset(self, images, labels):
        """Create TensorFlow dataset with batching"""
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(buffer_size=len(images))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def load_data(self):
        """Load and split dataset with memory-efficient batching"""
        if not os.path.exists(os.path.join(self.data_dir, 'BreaKHis_v1')):
            self.download_dataset()

        images = []
        labels = []
        
        for class_name, subtypes in self.classes.items():
            class_idx = 0 if class_name == 'benign' else 1
            
            for subtype in subtypes:
                base_path = os.path.join(
                    self.data_dir, 'BreaKHis_v1', 'histology_slides',
                    'breast', class_name, 'SOB', subtype
                )
                
                if not os.path.exists(base_path):
                    continue
                
                logger.info(f"Loading {class_name} {subtype} images...")
                for patient in os.listdir(base_path):
                    mag_path = os.path.join(base_path, patient, self.magnification)
                    if not os.path.exists(mag_path):
                        continue
                        
                    for img_name in os.listdir(mag_path):
                        if img_name.endswith('.png'):
                            try:
                                img_path = os.path.join(mag_path, img_name)
                                img = Image.open(img_path).convert('RGB')
                                img = img.resize(self.img_size)
                                img_array = np.array(img) / 255.0
                                images.append(img_array)
                                labels.append(class_idx)
                            except Exception as e:
                                logger.error(f"Error loading {img_path}: {e}")

        if not images:
            raise ValueError(f"No images found for magnification {self.magnification}")

        X = np.array(images)
        y = np.array(labels)
        logger.info(f"Loaded {len(X)} images. Class distribution: {np.bincount(y)}")
        
        # Split and create TensorFlow datasets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                         random_state=42)
        
        train_dataset = self.create_dataset(X_train, y_train)
        val_dataset = self.create_dataset(X_val, y_val)
        
        return train_dataset, val_dataset