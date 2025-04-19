import torch
import matplotlib.pyplot as plt
import numpy as np

from torchvision.models import inception_v3
from sklearn.manifold import TSNE

from .image_classes import plant_classes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inception = inception_v3(pretrained=True, transform_input=False)
inception.fc = torch.nn.Identity()  # remove classification layer to get 2048-d latent
inception.eval()
inception.to(device)

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
]


def get_color(label):
    if "Carpetweed" in label:
        return colors[0]
    elif "Crabgrass" in label:
        return colors[1]
    elif "Eclipta" in label:
        return colors[2]
    elif "Goosegrass" in label:
        return colors[3]
    elif "Morningglory" in label:
        return colors[4]
    elif "Nutsedge" in label:
        return colors[5]
    elif "Palmer Amaranth" in label:
        return colors[6]
    elif "Prickly Sida" in label:
        return colors[7]
    elif "Purslane" in label:
        return colors[8]
    elif "Ragweed" in label:
        return colors[9]
    elif "Sicklepod" in label:
        return colors[10]
    elif "Spotted Spurge" in label:
        return colors[11]
    elif "Spurred Anoda" in label:
        return colors[12]
    elif "Swinecress" in label:
        return colors[13]
    elif "Waterhemp" in label:
        return colors[14]
    else:
        return "white"


def extract_features(images, label):
    features = []
    labels = []
    for img in images:
        with torch.no_grad():
            feat = inception(img.to(device).unsqueeze(0)).squeeze().cpu().numpy()
        features.append(feat)
        labels.append(label)
    return features, labels


def generate_plot(labels, features_2d, output_dir, trial_name):
    plt.figure(figsize=(10, 8))
    for label in set(labels):
        color = get_color(label)
        idxs = [i for i, l in enumerate(labels) if l == label]
        if "Real" in label:
            plt.scatter(
                [features_2d[i][0] for i in idxs],
                [features_2d[i][1] for i in idxs],
                label=label,
                marker="o",
                color=color,
                alpha=0.6,
            )
        else:
            plt.scatter(
                [features_2d[i][0] for i in idxs],
                [features_2d[i][1] for i in idxs],
                label=label,
                color=color,
                marker="x",
            )

    plt.title(f"{trial_name} - t-SNE of InceptionV3 Latent Features")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/t-sne.png")


# Images must already be transformed to 299x299 and normalized
def get_tsne(gt, generated, output_dir, trial_name):
    features = []
    labels = []

    for plant_class in plant_classes:
        print(f"Processing {plant_class}")

        generated_class_index = generated.class_to_idx[plant_class]
        generated_class_images = [
            img for img, label in generated if label == generated_class_index
        ]

        gt_class_index = gt.class_to_idx[plant_class]
        gt_class_images = [img for img, label in gt if label == gt_class_index]

        generated_label = f"Synthetic {plant_class}"
        gt_label = f"Real {plant_class}"

        generated_features, generated_labels = extract_features(
            generated_class_images, generated_label
        )
        gt_features, gt_labels = extract_features(gt_class_images, gt_label)

        features.extend(generated_features)
        labels.extend(generated_labels)

        features.extend(gt_features)
        labels.extend(gt_labels)

    print(f"Computing TSNE...")
    tsne = TSNE(n_components=2, random_state=42)
    features = np.stack(features)
    print(f"Shape stacked: {features.shape}")
    # features = features.reshape(features.shape[0], -1)
    # print(f"Shape flattened: {features.shape}")
    features_2d = tsne.fit_transform(features)

    print(f"Generating plot...")
    generate_plot(labels=labels, features_2d=features_2d, output_dir=output_dir, trial_name=trial_name)
