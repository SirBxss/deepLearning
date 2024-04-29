import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from sklearn import utils


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle_data=False):
        # Initialize instance variables
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle_data = shuffle_data

        # Class dictionary mapping integer labels to class names
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        # Load labels from JSON file
        with open(label_path, 'r') as f:
            self.labels = json.load(f)

        # Initialize lists for image file paths and labels
        self.image_files = []
        self.image_labels = []

        # Load image file paths and labels
        for file_name, label in self.labels.items():
            image_path = os.path.join(file_path, f"{file_name}.npy")
            self.image_files.append(image_path)
            self.image_labels.append(label)

        # Initialize the current index and epoch
        self.current_index = 0
        self.current_epoch_count = 0

        # Shuffle data if requested
        if shuffle_data:
            self.shuffle_data_set()

    def shuffle_data_set(self):
        # Shuffle the data set (image files and labels)
        self.image_files, self.image_labels = utils.shuffle(self.image_files, self.image_labels)

    def next(self):
        # Check if we've reached the end of the dataset
        if self.current_index >= len(self.image_files):
            # Start a new epoch
            self.current_epoch_count += 1

            # Reset the index
            self.current_index = 0

            # Shuffle the data set if requested
            if self.shuffle_data:
                self.shuffle_data_set()

        # Determine the starting and ending index for the batch
        start_index = self.current_index
        end_index = start_index + self.batch_size

        # Handle wrapping around the data set for the batch
        if end_index > len(self.image_files):
            # Wrap around and create a new epoch
            # Splitting the batch into two parts
            batch_files = self.image_files[start_index:] + self.image_files[:end_index - len(self.image_files)]
            batch_labels = self.image_labels[start_index:] + self.image_labels[:end_index - len(self.image_files)]
        else:
            # Get the batch files and labels
            batch_files = self.image_files[start_index:end_index]
            batch_labels = self.image_labels[start_index:end_index]

        # Update the current index
        self.current_index = end_index if end_index <= len(self.image_files) else end_index - len(self.image_files)

        # Load and process the images in the batch
        images = []
        for file in batch_files:
            # Load the image
            image = np.load(file)

            # Resize the image if necessary
            if image.shape[:2] != tuple(self.image_size[:2]):
                image = transform.resize(image, self.image_size[:2], mode='reflect')

            # Apply mirroring and rotation if requested
            image = self.augment(image)

            # Append the processed image
            images.append(image)

        # Convert images list to numpy array
        images = np.array(images)

        # Return the batch of images and labels
        return images, np.array(batch_labels)

    def augment(self, img):
        if self.mirroring and np.random.rand() < 0.5:
            img = np.fliplr(img)  # Mirror the image horizontally

        if self.rotation:
            # Randomly rotate the image by 90, 180, or 270 degrees
            rotation_choice = np.random.choice([0, 1, 2, 3])
            img = np.rot90(img, k=rotation_choice)

        return img

    def current_epoch(self):
        # Return the current epoch count
        return self.current_epoch_count

    def class_name(self, label):
        # Return the class name corresponding to an integer label
        return self.class_dict[label]

    def show(self):
        # Generate a batch using next()
        images, labels = self.next()

        # Determine the number of rows and columns for the plot grid
        num_rows = int(np.sqrt(len(images)))
        num_cols = int(np.ceil(len(images) / num_rows))

        # Create a plot grid
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))

        # Plot each image in the batch
        for idx, (image, label) in enumerate(zip(images, labels)):
            row = idx // num_cols
            col = idx % num_cols
            ax = axs[row, col]

            # Display the image
            ax.imshow(image)

            # Set the title using the class name
            ax.set_title(self.class_name(label))

            # Hide axis for better visualization
            ax.axis('off')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
