from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel

MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"    

noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")

noise_scheduler.save_pretrained("../sd1.5/scheduler")
text_encoder.save_pretrained("../sd1.5/text_encoder")
vae.save_pretrained("../sd1.5/vae")
unet.save_pretrained("../sd1.5/unet")
