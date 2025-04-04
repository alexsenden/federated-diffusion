import os
import tempfile
import torch

from collections import OrderedDict
from diffusers import UNet2DConditionModel


MAX_WEIGHTS_PER_NODE = 2


class IpfsWeightsLoader:
    def __init__(self, ipfs_api="/ip4/127.0.0.1/tcp/5001"):
        self.ipfs_api = ipfs_api
        self.stored_weights = []

    def load(self, id):
        with tempfile.TemporaryDirectory() as tempdir:
            weights_path = os.path.join(tempdir, "weights.pth")
            os.system(f"ipfs get --api {self.ipfs_api} -o {weights_path} {id}")
            state_dict = torch.load(weights_path)
            return [val.cpu().numpy() for _, val in state_dict.items()]

    def store(self, weights):
        if len(self.stored_weights) >= MAX_WEIGHTS_PER_NODE:
            os.system(f"ipfs --api {self.ipfs_api} pin rm {self.stored_weights.pop(0)}")
            os.system(f"ipfs --api {self.ipfs_api} repo gc")

        dummy_unet = UNet2DConditionModel.from_pretrained("/model/unet")
        params_dict = zip(dummy_unet.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        with tempfile.TemporaryDirectory() as tempdir:
            weights_path = os.path.join(tempdir, "weights.pth")
            torch.save(state_dict, weights_path)

            weights_id = (
                os.popen(f"ipfs add --api {self.ipfs_api} -q {weights_path}")
                .read()
                .strip()
                .split("\n")
                .pop()
            )
            self.stored_weights.append(weights_id)
            
            return weights_id
