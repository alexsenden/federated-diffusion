import torch
import numpy as np
import flwr as fl
import os

from diffusers import UNet2DConditionModel
from flwr.common import Context, logger
from collections import OrderedDict

dummy_unet = None


def create_save_unet(context: Context):
    global dummy_unet
    dummy_unet = UNet2DConditionModel.from_pretrained(
        context.run_config["model-name"], subfolder="unet"
    )


def save_unet(
    strategy_name: str, server_round: int, aggregated_parameters: fl.common.Parameters
) -> None:
    if aggregated_parameters is not None:
        logger.log(
            20,
            f"Saving round {server_round} aggregated_parameters to model/{strategy_name}/{strategy_name}_round_{server_round}.pth",
        )

        # Convert `Parameters` to `list[np.ndarray]`
        aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
            aggregated_parameters
        )

        # Convert `list[np.ndarray]` to PyTorch `state_dict`
        params_dict = zip(dummy_unet.state_dict().keys(), aggregated_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        dummy_unet.load_state_dict(state_dict, strict=True)

        # Save the model to disk
        os.makedirs(f"model/{strategy_name}", exist_ok=True)
        torch.save(
            dummy_unet.state_dict(),
            f"model/{strategy_name}/{strategy_name}_round_{server_round}.pth",
        )
