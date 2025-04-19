import torch

from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer

from .diffusion_model import MODEL_NAME

DATASET_NAME = "alexsenden/cottonweedid15_partitioned"

fds = None


def load_partition(dataset, partition_id):
    return dataset.filter(lambda datum: str(datum["partition"]) == str(partition_id))


def prep_data(partition_id):
    global fds
    if fds is None:
        dataset = load_dataset(
            DATASET_NAME,
        )["train"]

    partition = load_partition(dataset, partition_id)

    # Divide data: 98% train, 2% test
    partition_train_test = partition.train_test_split(test_size=0.02, seed=42)

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = []
        for caption in examples["text"]:
            if isinstance(caption, str):
                captions.append(caption)
            else:
                raise ValueError(
                    f'Caption column "text" should contain either strings or lists of strings.'
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                256,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    partition_train_test["train"] = partition_train_test["train"].with_transform(
        preprocess
    )
    partition_train_test["test"] = partition_train_test["test"].with_transform(
        preprocess
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    trainloader = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=1,
        collate_fn=collate_fn,
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=1, collate_fn=collate_fn
    )

    return trainloader, testloader
