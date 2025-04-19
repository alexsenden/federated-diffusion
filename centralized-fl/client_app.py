"""central-aggregator: A Flower / HuggingFace app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, logger

from central_aggregator.task import get_weights, load_data, set_weights, test, train
from central_aggregator.net import init_net, unwrap_net
from central_aggregator.utils import log_mem_info


def print_available_devices():
    logger.log(20, "Available devices:")
    for i in range(torch.cuda.device_count()):
        logger.log(20, torch.cuda.get_device_properties(i).name)


# Flower client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, context):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.context = context
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # print_available_devices()

        # Unwrap net
        unet, text_encoder, vae, noise_scheduler = unwrap_net(net)

        # Move text_encoder and VAE to GPU and cast to float16 - these are only used for inference
        text_encoder.to(self.device, dtype=torch.float16)
        vae.to(self.device, dtype=torch.float16)

        # Move unet to GPU
        unet.to(self.device)

        # log_mem_info(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader, device=self.device, context=self.context)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss = test(self.net, self.testloader, self.device, context=self.context)
        return float(loss), len(self.testloader), {}


def client_fn(context: Context):
    # Get this client's dataset partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = context.run_config["model-name"]
    trainloader, valloader = load_data(
        context, partition_id, num_partitions, model_name
    )

    # Load model
    net = init_net(context)

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, context).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
