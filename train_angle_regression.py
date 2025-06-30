import argparse
import copy
from loguru import logger
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from model.models import RotationAnglePredictorCustomNet, RotationAnglePredictorResNet, \
    RotationAnglePredictorTransformer, get_model_info
from utils.utils import RotationDataset, generate_synthetic_pairs_regression


def set_random_seed(seed):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Parameters:
        seed (int): The seed value to use.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    Main function to run rotation angle prediction based on regression approach.
    """
    parser = argparse.ArgumentParser(
        description='Rotation angle prediction based on regression approach.')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=50,  # 32, 100
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate for training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers to use in data loading')
    parser.add_argument('--input_size', type=int, default=32,  # 32, 224
                        help='Number of input size for training and validation dataset: 32 for custom or 224 for '
                             'others such as ResNet and vision transformer.')
    parser.add_argument('--network_type', type=str, default='custom',
                        help='Network type to use: custom, resnet or transformer.')
    parser.add_argument('--num_samples_per_image', type=int, default=10,
                        help='Number of image pairs to generate per base image.')
    parser.add_argument('--use_augmentation', default=True, action="store_true",
                        help='Use data augmentation of training data (if num_samples_per_image > 1).')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Args: {}".format(args))

    # For reproducibility
    set_random_seed(42)

    # Data preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),  # Resize images to match the network input size
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR10 mean and std
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),  # Resize images to match the network input size
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR10 mean and std
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std
    ])

    # Augmentation transforms. Apply data augmentation to the TRAIN base image before rotation. This can improve model
    # generalization.
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # This degrades performance.
        transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0)),
    ])

    # CIFAR10 dataset
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)

    logger.info("\tGenerating synthetic pairs has started .... ")
    train_image_pairs, train_angles = generate_synthetic_pairs_regression(train_dataset, args.num_samples_per_image,
                                                                          augmentation, args.use_augmentation)
    val_image_pairs, val_angles = generate_synthetic_pairs_regression(val_dataset, args.num_samples_per_image)
    logger.info("\tGenerating synthetic pairs is finished.")

    # Convert angles to numpy arrays
    train_angles = np.array(train_angles, dtype=np.float32)
    val_angles = np.array(val_angles, dtype=np.float32)

    # Create training and validation datasets
    train_rotation_dataset = RotationDataset(train_image_pairs, train_angles)
    val_rotation_dataset = RotationDataset(val_image_pairs, val_angles)

    # Create dataloaders
    train_dataloader = DataLoader(train_rotation_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  shuffle=True)
    val_dataloader = DataLoader(val_rotation_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                shuffle=False)

    # Initialize the model
    logger.info("\tInitializing model...")
    if args.network_type == 'custom':
        model = RotationAnglePredictorCustomNet(out_features=1).to(device)  # Regression
    elif args.network_type == 'resnet':  # Requires more computation resource to train
        model = RotationAnglePredictorResNet(out_features=1).to(device)  # Regression
    elif args.network_type == 'transformer':  # Requires more computation resource to train
        model = RotationAnglePredictorTransformer(out_features=1).to(device)  # Regression
    else:
        raise ValueError("Error: Unsupported network type:" + args.network_type)

    logger.info("Model Summary: {}".format(get_model_info(model, (args.input_size, args.input_size), device)))
    logger.info("\tModel is initialized.")

    # Define loss function and optimizer
    criterion = nn.MSELoss()   # Mean Square Error Loss for regression
    # optimizer = optim.SGD(model.parameters(), lr=0.001)  # SGD optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)  # AdamW optimizer (good for transformer).

    # Define a learning rate scheduler
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)  # Reduce LR by 0.1 every 15 epochs.
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular')

    # Training loop
    tolerance = 5  # Tolerance for accuracy evaluation (in degrees)
    accum_train_loss = []
    accum_train_mae = []
    accum_train_accuracy = []
    accum_val_loss = []
    accum_val_mae = []  # Mean Absolute Error (MAE)
    accum_val_accuracy = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = 10e6
    best_epoch = 0
    logger.info("\tTraining has started .... ")
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_correct = 0
        train_total = 0

        # Training phase
        for image1, image2, angle in train_dataloader:
            image1, image2, angle = image1.to(device), image2.to(device), angle.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(image1, image2)  # Give original and rotated images as input

            # Compute loss
            loss = criterion(outputs.squeeze(), angle)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Compute MAE
            train_mae += torch.abs(outputs.squeeze() - angle).sum().item()

            # Compute accuracy within tolerance
            train_total += angle.size(0)
            train_correct += (torch.abs(outputs.squeeze() - angle) <= tolerance).sum().item()

        accum_train_loss.append(train_loss / len(train_dataloader))
        train_mae /= len(train_dataloader)
        train_accuracy = 100 * train_correct / train_total
        accum_train_mae.append(train_mae)
        accum_train_accuracy.append(train_accuracy)

        # Print training loss
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {train_loss / len(train_dataloader):.4f}, "
              f"Training MAE: {train_mae:.4f}, Training Accuracy (within {tolerance}°): {train_accuracy:.2f}%, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for image1, image2, angle in val_dataloader:
                image1, image2, angle = image1.to(device), image2.to(device), angle.to(device)

                # Forward pass
                outputs = model(image1, image2)  # Give original and rotated images as input

                # Compute loss
                val_loss += criterion(outputs.squeeze(), angle).item()

                # Compute MAE
                val_mae += torch.abs(outputs.squeeze() - angle).sum().item()

                # Compute accuracy within tolerance
                val_total += angle.size(0)
                val_correct += (torch.abs(outputs.squeeze() - angle) <= tolerance).sum().item()

        # Update the learning rate
        scheduler.step()
        # scheduler.step(val_loss)  # For ReduceLROnPlateau

        # Print validation loss, MAE, and accuracy
        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataloader)
        val_accuracy = 100 * val_correct / val_total
        accum_val_loss.append(val_loss)
        accum_val_mae.append(val_mae)
        accum_val_accuracy.append(val_accuracy)

        # Deep copy the model to keep the best model weights
        if val_mae <= best_mae:
            best_mae = val_mae
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}, "
              f"Validation Accuracy (within {tolerance}°): {val_accuracy:.2f}%")

    # Load best model weights and then save the model
    model.load_state_dict(best_model_wts)
    save_model_name = 'rotation_angle_regressor_' + args.network_type + '.pth'
    torch.save(model.state_dict(), save_model_name)
    print(f"Best model is saved at {best_epoch + 1} epoch, with validation (test) mean absolute error (MAE) of "
          f"{best_mae}.")

    logger.info("\tTraining is finished.")

    # Plot training and validation losses, train and val MAE, and train and val accuracies
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))
    ax1.plot(accum_train_loss)
    ax1.plot(accum_val_loss)
    ax1.legend(['train-loss', 'val-loss'])
    ax2.plot(accum_train_mae)
    ax2.plot(accum_val_mae)
    ax2.legend(['train-mae', 'val-mae'])
    ax3.plot(accum_train_accuracy)
    ax3.plot(accum_val_accuracy)
    ax3.legend(['train-accuracy', 'val-accuracy'])
    plt.show()


# Execute from the interpreter
if __name__ == "__main__":
    main()
