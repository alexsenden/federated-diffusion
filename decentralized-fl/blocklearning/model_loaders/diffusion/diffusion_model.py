from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from torch.optim import AdamW

from blocklearning.model_loaders.model import Model
from .net import unwrap_net

transformers.logging.set_verbosity_error()

LR = 0.000001
LOCAL_STEPS = 100
MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"

GRADIENT_ACCUMULATION_STEPS = 4


def net_get_weights(net):
    unet, text_encoder, vae, noise_scheduler = unwrap_net(net)
    return [val.cpu().numpy() for _, val in unet.state_dict().items()]


class DiffusionModel(Model):
    def __init__(self, model, partition):
        self.model = model
        self.layers, self.count = self.__get_layer_info(model)
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Device used: {self.device}")

    def __get_layer_info(self, model):
        layers = []
        total = 0
        for layer in net_get_weights(model):
            shape = layer.shape
            weights = np.prod(shape)
            total += weights
            layers.append((shape, weights))
        return layers, total
    
    def to_device_and_dtype(self):
        unet, text_encoder, vae, noise_scheduler = unwrap_net(self.model)
        
        # Move text_encoder and VAE to GPU and cast to float16 - these are only used for inference
        text_encoder.to(self.device, dtype=torch.float16)
        vae.to(self.device, dtype=torch.float16)

        # Move unet to GPU
        unet.to(self.device)
        

    def train(self, trainloader):
        unet, text_encoder, vae, noise_scheduler = unwrap_net(self.model)

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
                    batch["pixel_values"].to(self.device, torch.float16)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents).to(self.device)

                # Sample a random timestep for each image
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=self.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(self.device), return_dict=False
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)

                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents.to(self.device, torch.float32),
                    timesteps,
                    encoder_hidden_states.to(torch.float32),
                    return_dict=False,
                )[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                total_loss += loss.item()

                # Backpropagate
                loss.backward()
                if local_step % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if local_step >= LOCAL_STEPS:
                    break

        return total_loss / LOCAL_STEPS

    def test(self, testloader):
        unet, text_encoder, vae, noise_scheduler = unwrap_net(self.model)
        unet.eval()

        total_loss = 0
        with torch.no_grad():
            for batch in testloader:
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(self.device, torch.float16)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents).to(self.device)

                # Sample a random timestep for each image
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=self.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(self.device), return_dict=False
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)

                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents.to(self.device, torch.float32),
                    timesteps,
                    encoder_hidden_states.to(torch.float32),
                    return_dict=False,
                )[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                total_loss += loss.item()

        return total_loss / len(testloader.dataset)

    def get_unet(self):
        unet, text_encoder, vae, noise_scheduler = unwrap_net(self.model)
        return unet

    def get_weights(self):
        unet, text_encoder, vae, noise_scheduler = unwrap_net(self.model)
        return [val.cpu().numpy() for _, val in unet.state_dict().items()]

    def set_weights(self, parameters):
        unet, text_encoder, vae, noise_scheduler = unwrap_net(self.model)
        params_dict = zip(unet.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        unet.load_state_dict(state_dict, strict=True)
