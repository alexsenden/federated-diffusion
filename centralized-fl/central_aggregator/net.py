from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from flwr.common import Context
from transformers import CLIPTextModel

def unwrap_net(net):
    return net["unet"], net["text_encoder"], net["vae"], net["noise_scheduler"]

def wrap_net(unet, text_encoder, vae, noise_scheduler):
    return {"unet": unet, "text_encoder": text_encoder, "vae": vae, "noise_scheduler": noise_scheduler}

def init_net(context: Context):
    # Load the pretrained components of the diffusion model
    noise_scheduler = DDPMScheduler.from_pretrained(context.run_config["model-name"], subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        context.run_config["model-name"], subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        context.run_config["model-name"], subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        context.run_config["model-name"], subfolder="unet"
    )
    
    unet.enable_gradient_checkpointing()
    
    return wrap_net(unet, text_encoder, vae, noise_scheduler)
