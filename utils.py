"""
Utility functions for rotation angle prediction.
"""
from torch.utils.data import Dataset
from torchvision import transforms
import random


class RotationDataset(Dataset):
    """
       Custom RotationDataset class which inherits from the PyTorch Dataset for CIFAR-10 dataset loading.
       """
    def __init__(self, image_pairs, angle_classes):
        self.image_pairs = image_pairs  # List of (image1, image2) pairs
        self.angle_classes = angle_classes  # List of corresponding rotation angles

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image1, image2 = self.image_pairs[idx]
        angle_class = self.angle_classes[idx]

        return image1, image2, angle_class


# Discretize the rotation angles into 360 classes.
def discretize_angle(angle):
    return int(round(angle)) % 360  # Convert angle to an integer between 0 and 359 i.e. into 360 classes.


# Generate synthetic image pairs and rotation angles for training and validation, for classification approach.
def generate_synthetic_pairs_classification(dataset, num_samples_per_image=1, augmentation=None,
                                            use_augmentation=False):
    image_pairs = []
    angles = []
    for i in range(num_samples_per_image):
        for image, _ in dataset:
            # Apply data augmentation to the TRAIN base image before rotation. This can improve model generalization.
            if augmentation is not None and use_augmentation and i > 0:  # Leave the first sample without augmentation.
                image = augmentation(image)

            # Randomly sample a rotation angle between 0 and 359 degrees i.e. for 360 classes (one class for each
            # degree)
            angle = random.uniform(0, 359)  # Returns a random number between, and included, 0 and 359, for a total of
            # 360 classes.

            # Rotate the image
            rotated_image = transforms.functional.rotate(image, angle)

            # Append the image pair and angle
            image_pairs.append((image, rotated_image))
            angles.append(angle)
    return image_pairs, angles


# Generate synthetic image pairs and rotation angles for training and validation, for regression approach.
def generate_synthetic_pairs_regression(dataset, num_samples_per_image=1, augmentation=None, use_augmentation=False):
    image_pairs = []
    angles = []
    for i in range(num_samples_per_image):
        for image, _ in dataset:
            # Apply data augmentation to the TRAIN base image before rotation. This can improve model generalization.
            if augmentation is not None and use_augmentation and i > 0:  # Leave the first sample without augmentation.
                image = augmentation(image)

            # Randomly sample a rotation angle between 0 and 360 degrees, included 0 and 360.
            angle = random.uniform(0, 360)

            # Rotate the image
            rotated_image = transforms.functional.rotate(image, angle)

            # Append the image pair and angle
            image_pairs.append((image, rotated_image))
            angles.append(angle)
    return image_pairs, angles
