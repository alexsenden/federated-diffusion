import torch

import torchvision.transforms as transforms

from .crop_center_transform import CenterCropToSquare

metrics_gt_transform = transforms.Compose(
    [
        CenterCropToSquare(),
        transforms.Resize((256, 256)),
        transforms.Resize((299, 299)),  # Inception expects 299x299 images
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
    ]
)

metrics_generated_transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),  # Inception expects 299x299 images
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
    ]
)

tsne_gt_transform = transforms.Compose(
    [
        CenterCropToSquare(),
        transforms.Resize((256, 256)),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Inception expects values between [-1, 1]
    ]
)

tsne_generated_transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Inception expects values between [-1, 1]
    ]
)
