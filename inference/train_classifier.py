import argparse
import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--trainingPath",
        type=str,
        required=True,
        help="Path to the training images",
    )
    parser.add_argument(
        "--outputDir",
        type=str,
        required=True,
        help="Path to save the output to",
    )

    args = parser.parse_args()
    print(args)

    return args


args = parse_args()

os.makedirs(args.outputDir, exist_ok=True)

# --- Config ---
num_classes = 105
batch_size = 32
num_epochs = 5000
lr = 0.00001
patience = 3  # Early stopping patience
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = f"{args.outputDir}/resnet50_classifier.pth"

# --- Transforms ---
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)

# --- Datasets and loaders ---
full_dataset = datasets.ImageFolder(
    args.trainingPath, transform=transform
)
test_dataset = datasets.ImageFolder("test", transform=transform)

train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
datasets_ = {
    "train": train_dataset,
    "val": val_dataset,
    "test": test_dataset,
}

loaders = {
    phase: DataLoader(
        datasets_[phase], batch_size=batch_size, shuffle=(phase == "train")
    )
    for phase in ["train", "val", "test"]
}

# --- Model ---
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

# --- Training loop with early stopping ---
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    for phase in ["train", "val"]:
        model.train() if phase == "train" else model.eval()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in loaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(datasets_[phase])
        epoch_acc = running_corrects.double() / len(datasets_[phase])
        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if phase == "val":
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, checkpoint_path)
                print("📌 Best model updated.")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"⚠️  No improvement for {epochs_without_improvement} epoch(s).")
                if epochs_without_improvement >= patience:
                    print("⏹️ Early stopping triggered.")
                    model.load_state_dict(best_model_wts)
                    break
    else:
        continue
    break

print(f"\n✅ Training complete. Best val accuracy: {best_acc:.4f}")

# Test the model
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in loaders["test"]:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"📊 Test Accuracy: {test_acc:.4f}")

# Calculate precision, recall, and F1 score
precision = precision_score(all_labels, all_preds, average="weighted", zero_division=1)
recall = recall_score(all_labels, all_preds, average="weighted", zero_division=1)
f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=1)

print(f"⚡ Precision: {precision:.4f}")
print(f"⚡ Recall: {recall:.4f}")
print(f"⚡ F1 Score: {f1:.4f}")

with open(f"{args.outputDir}/resnet50_classifier.csv", "w") as file:
    file.write("accuracy,precision,recall,f1\n")
    file.write(f"{test_acc:.4f},{precision:.4f},{recall:.4f},{f1:.4f}\n")
