import os
import logging
import gc
import tensorflow as tf
from data.dataset import Dataset
from data.preprocessing import preprocess_images
from models.cnn import CNN
from models.train import train_model
from utils.evaluation import evaluate_model
from utils.visualization import plot_training_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Enable memory growth for GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    breakhis_dir = os.path.join(data_dir, 'BreaKHis_v1', 'histology_slides', 'breast')
    
    try:
        # Initialize dataset with memory-efficient loading
        dataset = Dataset(
            data_dir=breakhis_dir,
            batch_size=32,  # Process images in batches
            prefetch_size=4  # Prefetch next batches
        )
        
        # Load and preprocess in batches
        train_generator, val_generator = dataset.load_data()
        
        # Initialize model
        model = CNN()
        
        # Train model using generators
        history = train_model(
            model=model,
            train_data=train_generator,
            val_data=val_generator,
            epochs=50
        )
        
        # Evaluate and visualize
        evaluate_model(model, val_generator)
        plot_training_history(history)
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        # Cleanup
        gc.collect()
        raise
    
    finally:
        # Cleanup
        gc.collect()

if __name__ == "__main__":
    main()