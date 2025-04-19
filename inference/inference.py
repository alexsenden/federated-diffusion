import argparse
import os
import torch

from diffusers import StableDiffusionPipeline

MODELS_DIR = "../flwr/central-aggregator/model"
NUM_SAMPLES = 200

plant_classes = [
    "Carpetweed",
    "Crabgrass",
    "Eclipta",
    "Goosegrass",
    "Morningglory",
    "Nutsedge",
    "Palmer Amaranth",
    "Prickly Sida",
    "Purslane",
    "Ragweed",
    "Sicklepod",
    "Spotted Spurge",
    "Spurred Anoda",
    "Swinecress",
    "Waterhemp",
]


def parse_args():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--modelPath",
        type=str,
        required=True,
        help="The name of the model to be used",
    )
    parser.add_argument(
        "--outputDir",
        type=str,
        required=True,
        help="The directory for the output images",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Starting {args.modelPath}")

    # Ensure outputDir exists
    os.makedirs(args.outputDir, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Load finetuned unet weights
    pipe.unet.load_state_dict(
        torch.load(args.modelPath, weights_only=True), strict=True
    )
    pipe.unet.eval()
    pipe.to("cuda", dtype=torch.float16)

    # Generate NUM_SAMPLES images
    for i in range(NUM_SAMPLES):
        # Iterate through classes
        for plant_class in plant_classes:
            # Ensure outputDir/class exists
            os.makedirs(f"{args.outputDir}/{plant_class}", exist_ok=True)
            # Generate and save the image
            image = pipe(
                f"A photo of a {plant_class} plant",
                num_inference_steps=30,
                guidance_scale=7.5,
                height=256,
                width=256,
            ).images[0]
            print(f"{args.outputDir}/{plant_class}/{plant_class}_{i}.png")
            image.save(f"{args.outputDir}/{plant_class}/{plant_class}_{i}.png")


main()
