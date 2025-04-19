import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import transformers

from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import CLIPTextModel, CLIPTokenizer

transformers.logging.set_verbosity_error()

LR = 0.000001
LOCAL_STEPS = 20000
MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATASET_NAME = "alexsenden/cottonweedid15_partitioned"

GRADIENT_ACCUMULATION_STEPS = 4

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device}")


def prep_data():
    dataset = load_dataset(
        DATASET_NAME,
    )["train"]

    # Divide data: 98% train, 2% test
    partition_train_test = dataset.train_test_split(test_size=0.02, seed=42)

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
        if "image" in examples:
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
        sampler=oversampling_sampler(partition_train_test),
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=2,
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=1, collate_fn=collate_fn
    )

    return trainloader, testloader


def oversampling_sampler(dataset):
    counts = np.bincount(dataset["train"]["label"])
    weights = 1.0 / counts
    sample_weights = weights[dataset["train"]["label"]]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(dataset["train"]), replacement=True
    )


def unwrap_net(net):
    return net["unet"], net["text_encoder"], net["vae"], net["noise_scheduler"]


def wrap_net(unet, text_encoder, vae, noise_scheduler):
    return {
        "unet": unet,
        "text_encoder": text_encoder,
        "vae": vae,
        "noise_scheduler": noise_scheduler,
    }


def init_net():
    # Load the pretrained components of the diffusion model
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")

    # Move text_encoder and VAE to GPU and cast to float16 - these are only used for inference
    text_encoder.to(device, dtype=torch.float16)
    vae.to(device, dtype=torch.float16)

    # Move unet to GPU
    unet.to(device)

    unet.enable_gradient_checkpointing()

    return wrap_net(unet, text_encoder, vae, noise_scheduler)


def train(trainloader, model):
    unet, text_encoder, vae, noise_scheduler = unwrap_net(model)

    # Initialize the optimizer
    optimizer = AdamW(unet.parameters(), lr=LR)

    # Freeze the weights of the VAE and text encoder, set unet to train
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    num_epochs = (LOCAL_STEPS // len(trainloader)) + 1
    local_step = 0

    total_loss = 0

    for epoch in range(num_epochs):
        for batch in trainloader:
            local_step += 1

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
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
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

            if local_step % 100 == 0:
                print(f"Trained step {local_step} at {time.time()}")
                print(f"Training loss: {total_loss / 100}\n")
                total_loss = 0

            # Backpropagate
            loss.backward()
            if local_step % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            if local_step >= LOCAL_STEPS:
                break


def test(testloader, model):
    unet, text_encoder, vae, noise_scheduler = unwrap_net(model)
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
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
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

    print(f"Validation loss: {total_loss / len(testloader.dataset)}")
    
def save_model(model):
    os.makedirs("model", exist_ok=True)
    unet, text_encoder, vae, noise_scheduler = unwrap_net(model)
    torch.save(
        unet.state_dict(),
        f"model/centralized_model.pth",
    )


def main():
    trainloader, testloader = prep_data()
    model = init_net()
    train(trainloader, model)
    test(testloader, model)
    save_model(model)


main()
