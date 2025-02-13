class Dataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = []
        self.labels = []

    def load_data(self):
        # Load images and labels from the data directory
        pass

    def split_data(self, train_size=0.8):
        # Split the dataset into training and validation sets
        pass

    def get_train_data(self):
        # Return training data
        pass

    def get_val_data(self):
        # Return validation data
        pass