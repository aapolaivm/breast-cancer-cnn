import os
from data.dataset import Dataset
from data.preprocessing import preprocess_images
from models.cnn import CNN
from models.train import train_model
from utils.evaluation import evaluate_model
from utils.visualization import plot_training_history

def main():
    # Load and preprocess the dataset
    dataset = Dataset()
    train_images, val_images, train_labels, val_labels = dataset.load_data()
    train_images, val_images = preprocess_images(train_images, val_images)

    # Initialize and train the CNN model
    model = CNN()
    history = train_model(model, train_images, train_labels, val_images, val_labels)

    # Evaluate the model
    evaluate_model(model, val_images, val_labels)

    # Visualize the training history
    plot_training_history(history)

if __name__ == "__main__":
    main()