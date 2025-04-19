import torch
from PIL import Image


class CenterCropToSquare:
    def __call__(self, img):
        if not isinstance(img, (Image.Image, torch.Tensor)):
            raise TypeError("Input should be a PIL Image or torch Tensor.")

        # Get image size
        if isinstance(img, Image.Image):
            width, height = img.size
        else:
            _, height, width = img.shape

        if height < width:
            size = height
            left = (width - size) // 2
            top = 0
        elif width < height:
            size = width
            left = 0
            top = (height - size) // 2
        else:
            # Already square
            return img

        right = left + size
        bottom = top + size

        if isinstance(img, Image.Image):
            return img.crop((left, top, right, bottom))
        else:
            return img[:, top:bottom, left:right]
