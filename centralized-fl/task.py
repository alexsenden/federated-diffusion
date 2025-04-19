"""central-aggregator: A Flower / HuggingFace app."""

import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F
import transformers
from datasets.utils.logging import disable_progress_bar
from flwr.common import Context, logger
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer

from central_aggregator.net import unwrap_net
from central_aggregator.utils import log_mem_info

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
disable_progress_bar()
transformers.logging.set_verbosity_error()

fds = None  # Cache FederatedDataset


def load_data(
    context: Context, partition_id: int, num_partitions: int, model_name: str
):
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = NaturalIdPartitioner(partition_by="partition")
        fds = FederatedDataset(
            dataset="alexsenden/cottonweedid15_partitioned",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)

    # Divide data: 98% train, 2% test
    partition_train_test = partition.train_test_split(test_size=0.02, seed=42)

    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")

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
                context.run_config["resolution"],
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomCrop(context.run_config["resolution"]),
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


def train(net, trainloader, device, context):
    # logger.log(20, "Beginning training")

    unet, text_encoder, vae, noise_scheduler = unwrap_net(net)

    # Initialize the optimizer
    optimizer = AdamW(unet.parameters(), lr=context.run_config["learning-rate"])

    # Freeze the weights of the VAE and text encoder, set unet to train
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    num_epochs = (context.run_config["local-steps"] // len(trainloader)) + 1
    local_step = 0

    for epoch in range(num_epochs):
        for batch in trainloader:
            local_step += 1

            # if local_step % 10 == 0:
            #     logger.log(20, f"Training step {local_step}")

            # Convert images to latent space
            latents = vae.encode(
                batch["pixel_values"].to(device, torch.float16)
            ).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents).to(device)

            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(
                batch["input_ids"].to(device), return_dict=False
            )[0]

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            # Predict the noise residual and compute loss
            model_pred = unet(
                noisy_latents.to(device, torch.float32),
                timesteps,
                encoder_hidden_states.to(torch.float32),
                return_dict=False,
            )[0]
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Backpropagate
            loss.backward()
            if local_step % context.run_config["gradient-accumulation-steps"] == 0:
                # log_mem_info(device)
                optimizer.step()
                optimizer.zero_grad()

            if local_step >= context.run_config["local-steps"]:
                break


def test(net, testloader, device, context):
    unet, text_encoder, vae, noise_scheduler = unwrap_net(net)
    unet.eval()

    total_loss = 0
    with torch.no_grad():
        for batch in testloader:
            # Convert images to latent space
            latents = vae.encode(
                batch["pixel_values"].to(device, torch.float16)
            ).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents).to(device)

            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(
                batch["input_ids"].to(device), return_dict=False
            )[0]

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            # Predict the noise residual and compute loss
            model_pred = unet(
                noisy_latents.to(device, torch.float32),
                timesteps,
                encoder_hidden_states.to(torch.float32),
                return_dict=False,
            )[0]
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            total_loss += loss.item()

    return total_loss / len(testloader.dataset)


def get_weights(net):
    unet, text_encoder, vae, noise_scheduler = unwrap_net(net)
    return [val.cpu().numpy() for _, val in unet.state_dict().items()]


def set_weights(net, parameters):
    unet, text_encoder, vae, noise_scheduler = unwrap_net(net)
    params_dict = zip(unet.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    unet.load_state_dict(state_dict, strict=True)
