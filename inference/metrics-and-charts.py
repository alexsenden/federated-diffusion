import argparse
import os

from torchvision import datasets

from evaluator.inception_transforms import (
    metrics_generated_transform,
    metrics_gt_transform,
    tsne_generated_transform,
    tsne_gt_transform,
)
from evaluator.generative_metrics import get_generative_metrics
from evaluator.tsne import get_tsne


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate generative metrics and t-SNE chart for a diffusion model."
    )
    parser.add_argument(
        "--genPath",
        type=str,
        required=True,
        help="Path to the generated images",
    )
    parser.add_argument(
        "--gtPath",
        type=str,
        required=True,
        help="Path to the ground truth images",
    )
    parser.add_argument(
        "--trialName",
        type=str,
        required=True,
        help="Name of the trial",
    )

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()

    output_dir = f"results/{args.trialName}"
    os.makedirs(output_dir, exist_ok=True)

    get_tsne(
        datasets.ImageFolder(root=args.gtPath, transform=tsne_gt_transform),
        datasets.ImageFolder(root=args.genPath, transform=tsne_generated_transform),
        output_dir,
        args.trialName,
    )
    get_generative_metrics(
        datasets.ImageFolder(root=args.gtPath, transform=metrics_gt_transform),
        datasets.ImageFolder(root=args.genPath, transform=metrics_generated_transform),
        output_dir,
    )


main()
