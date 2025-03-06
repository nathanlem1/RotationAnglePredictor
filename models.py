"""
We design the rotation angle prediction model using different deep learning architectures. We concatenate the input
images along the channel dimension, e.g. two 3x32x32 images lead to 6x32x32.
"""

from copy import deepcopy
from thop import profile
import torch
import torch.nn as nn

from torchvision import models


def get_model_info(model, tsize, device):
    """
    Define a function to get model info

    Parameters:
        model: Model to retrieve information about it.
        tsize: Test image input size (tuple).
        device: Device to be used (cuda or cpu).
    Return:
        Model information such as number of parameters, GFLOPs and MACs.
    """

    img = torch.zeros((1, 3, tsize[0], tsize[1]), device=device)
    macs, params = profile(deepcopy(model.to(device)), inputs=(img, img), verbose=False)
    params /= 1e6  # Number of parameters (in millions)
    macs /= 1e9   # MACs (Multiply-ACcumulate operations)
    flops = macs * 2  # Gflops - Giga FLOPs (Floating Point OPerations). Each MAC counts as two FLOPs.
    info = "Params: {:.4f}M, GFLOPs: {:.4f}, GMACs: {:.4f}".format(params, flops, macs)
    return info


class RotationAnglePredictorCustomNet(nn.Module):
    def __init__(self, out_features):
        """
        Define a custom neural network model for rotation angle prediction.

        Parameters:
            out_features (int): 1 for regression (single value) or 360 for classification into 360 classes (one for each
        degree).
         """
        super(RotationAnglePredictorCustomNet, self).__init__()
        # Input has 6 channels (3 for each image)
        self.conv1 = nn.Conv2d(6, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)  # Add dropout for regularization
        self.fc1 = nn.Linear(256 * 2 * 2, 512)  # Assuming input size is 32x32
        self.fc2 = nn.Linear(512, out_features)  # Output: 360 classes for classification or single value for regression

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # Concatenate the two images along the channel dimension

        x = self.bn1(self.relu(self.conv1(x)))
        x = self.maxpool(x)

        x = self.bn2(self.relu(self.conv2(x)))
        x = self.maxpool(x)

        x = self.bn3(self.relu(self.conv3(x)))
        x = self.maxpool(x)

        x = self.bn4(self.relu(self.conv4(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class RotationAnglePredictorResNet(nn.Module):
    def __init__(self, out_features):
        """
        Adapt ResNet model for rotation angle prediction.

        Parameters:
            out_features (int): 1 for regression (single value) or 360 for classification into 360 classes (one for each
        degree).
        """
        super(RotationAnglePredictorResNet, self).__init__()

        # Load pre-trained ResNet (e.g. ResNet18, ResNet50, etc.)
        # self.backbone = models.resnet18(pretrained=True)
        self.backbone = models.resnet50(pretrained=True)

        # Modify the first convolutional layer to accept 6 channels (two concatenated images along the channel
        # dimension)
        original_first_conv = self.backbone.conv1

        # Create a new first convolutional layer with 6 input channels.
        new_first_conv = nn.Conv2d(
            in_channels=6,  # 6 input channels (3 for each image)
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=original_first_conv.bias is not None
        )

        # Initialize the weights for the new convolutional layer. Copy the weights from the original first convolutional
        # layer for the first 3 channels.
        with torch.no_grad():
            new_first_conv.weight[:, :3] = original_first_conv.weight
            # Initialize the remaining 3 channels with the same weights (or random initialization)
            new_first_conv.weight[:, 3:] = original_first_conv.weight.clone()  # copy the same weight

        # Replace the first convolutional layer in ResNet
        self.backbone.conv1 = new_first_conv

        # Modify the final fully connected layer for 360-way classification or regression.
        num_features = self.backbone.fc.in_features  # Get the number of input features for the final layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),  # Add an intermediate layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout for regularization
            nn.Linear(512, out_features)  # Output: 360 classes for classification or single value for regression.
        )

    def forward(self, image1, image2):

        # Concatenate the two images along the channel dimension
        combined_input = torch.cat((image1, image2), dim=1)

        angle = self.backbone(combined_input)

        return angle


class RotationAnglePredictorTransformer(nn.Module):
    def __init__(self, out_features):
        """
        Adapt vision transformer model for rotation angle prediction.

        Parameters:
            out_features (int): 1 for regression (single value) or 360 for classification into 360 classes (one for each
        degree).
        """
        super(RotationAnglePredictorTransformer, self).__init__()

        # Load a pre-trained Vision Transformer (e.g. Swin-Tiny, ViT-Base, etc.)
        # self.backbone = models.vit_b_16(pretrained=True)  # Standard vision transformer (slower)
        self.backbone = models.swin_t(pretrained=True)   # Swin transformer

        # Modify the patch embedding layer to accept 6 channels
        # original_patch_embed = self.backbone.conv_proj  # ViT
        original_patch_embed = self.backbone.features[0][0]  # Swin

        # Create a new patch embedding layer with 6 input channels.
        new_patch_embed = nn.Conv2d(
            in_channels=6,  # 6 input channels (3 for each image)
            out_channels=original_patch_embed.out_channels,
            kernel_size=original_patch_embed.kernel_size,
            stride=original_patch_embed.stride,
            padding=original_patch_embed.padding,
        )

        # Initialize the weights for the new patch embedding layer. Copy the weights from the original patch embedding
        # layer for the first 3 channels.
        with torch.no_grad():
            new_patch_embed.weight[:, :3] = original_patch_embed.weight
            # Initialize the remaining 3 channels with the same weights (or random initialization)
            new_patch_embed.weight[:, 3:] = original_patch_embed.weight.clone()  # copy the same weight

        # Replace the patch embedding layer in the ViT / Swin
        # self.backbone.conv_proj = new_patch_embed  # ViT
        self.backbone.features[0][0] = new_patch_embed  # Swin

        # Modify the final head for 360-way classification or regression.
        # num_features = self.backbone.heads.head.in_features  # ViT
        num_features = self.backbone.head.in_features   # Swin
        self.backbone.head = nn.Sequential(
            nn.Linear(num_features, 512),  # Intermediate layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(512, out_features)  # Output: 360 classes for classification or single value for regression.
        )

    def forward(self, image1, image2):

        # Concatenate the two images along the channel dimension
        combined_input = torch.cat((image1, image2), dim=1)

        angle = self.backbone(combined_input)

        return angle
