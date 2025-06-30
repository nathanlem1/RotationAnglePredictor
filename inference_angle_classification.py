import argparse
from loguru import logger
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from model.models import RotationAnglePredictorCustomNet, RotationAnglePredictorResNet, \
    RotationAnglePredictorTransformer, get_model_info
from utils.utils import discretize_angle, generate_synthetic_pairs_classification


def main():
    """
    Main function to run rotation angle prediction based on classification approach.
    """
    parser = argparse.ArgumentParser(
        description='Rotation angle prediction based on classification approach.')
    parser.add_argument('--input_size', type=int, default=32,  # 32, 224
                        help='Number of input size for training and validation dataset: 32 for custom or 224 for '
                             'others such as ResNet and vision transformers.')
    parser.add_argument('--network_type', type=str, default='custom',
                        help='Network type to use: custom, resnet or transformer.')
    parser.add_argument('--model_name', type=str, default='./pretrained/rotation_angle_classifier_custom_46_epoch.pth',
                        help='The name of the trained model (ckpt) for inference or evaluation.')
    parser.add_argument('--test_index', type=int, default=60,
                        help='Index of the test image: 0 to 9999 (CIFAR10 test data has 10000 images).')
    parser.add_argument('--image1', type=str, default=None, help='Original image')
    parser.add_argument('--image2', type=str, default=None, help='Rotated image')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Args: {}".format(args))

    # Data preprocessing
    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),  # Resize images to match the network input size
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR10 mean and std
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std
    ])

    # Initialize the model
    logger.info("\tInitializing model...")
    if args.network_type == 'custom':
        model = RotationAnglePredictorCustomNet(out_features=360).to(device)  # Classification
    elif args.network_type == 'resnet':
        model = RotationAnglePredictorResNet(out_features=360).to(device)  # Classification
    elif args.network_type == 'transformer':
        model = RotationAnglePredictorTransformer(out_features=360).to(device)  # Classification
    else:
        raise ValueError("Error: Unsupported network type:" + args.network_type)

    logger.info("Model Summary: {}".format(get_model_info(model, (args.input_size, args.input_size), device)))
    logger.info("\tModel is initialized.")

    # Load the trained model
    logger.info("\tLoading model...")
    model.load_state_dict(torch.load(args.model_name))
    model.eval()
    logger.info("\tModel is loaded.")

    if args.image1 is not None and args.image2 is not None:
        # Read two images as input
        original_image_read = Image.open(args.image1).convert('RGB')
        original_image = val_transform(original_image_read).unsqueeze(0).to(device)
        rotated_image_read = Image.open(args.image2).convert('RGB')
        rotated_image = val_transform(rotated_image_read).unsqueeze(0).to(device)
    else:
        # CIFAR10 dataset
        val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)

        logger.info("\tGenerating synthetic pairs has started .... ")
        val_image_pairs, val_angles = generate_synthetic_pairs_classification(val_dataset)
        logger.info("\tGenerating synthetic pairs is finished.")

        # Discretize the angles
        val_angle_classes = [discretize_angle(angle) for angle in val_angles]

        # Prepare input images
        original_image = val_image_pairs[args.test_index][0].unsqueeze(0).to(device)
        rotated_image = val_image_pairs[args.test_index][1].unsqueeze(0).to(device)
        true_label = val_angle_classes[args.test_index]
        print(f"True Rotation Angle: {true_label} degrees")

    # Predict the rotation angle
    with torch.no_grad():
        outputs = model(original_image, rotated_image)
        predicted_class = torch.argmax(outputs, dim=1).item()
        print(f"Predicted Rotation Angle: {predicted_class} degrees")

    # Correct the rotated image using the predicted rotation angle.
    corrected_image = transforms.functional.rotate(rotated_image, -predicted_class)  # Negative input predicted angle

    # Display the original image (image 1) and rotated image (image 2), and then the corrected image using the
    # predicted angle
    original_img = original_image.squeeze() / 2 + 0.5  # unnormalize
    original_img_np = original_img.cpu().numpy()
    rotated_img = rotated_image.squeeze() / 2 + 0.5  # unnormalize
    rotated_img_np = rotated_img.cpu().numpy()
    corrected_img = corrected_image.squeeze() / 2 + 0.5  # unnormalize
    corrected_img_np = corrected_img.cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))
    ax1.imshow(np.transpose(original_img_np, (1, 2, 0)))
    ax1.axis('off')  # Hide the axis labels
    ax1.set_title("Original image")

    ax2.imshow(np.transpose(rotated_img_np, (1, 2, 0)))
    ax2.axis('off')  # Hide the axis labels
    ax2.set_title("Rotated image")

    ax3.imshow(np.transpose(corrected_img_np, (1, 2, 0)))
    ax3.axis('off')  # Hide the axis labels
    degree = chr(176)  # create from known unicode point
    title_name = "Corrected image by predicted angle of " + str(predicted_class) + degree
    ax3.set_title(title_name)

    plt.show()


# Execute from the interpreter
if __name__ == "__main__":
    main()
