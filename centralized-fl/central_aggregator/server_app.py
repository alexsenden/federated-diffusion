"""central-aggregator: A Flower / HuggingFace app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from central_aggregator.task import get_weights
from central_aggregator.net import init_net
from central_aggregator.strategy.get_strategy import get_strategy, get_num_server_rounds
from central_aggregator.strategy.save_unet import create_save_unet


def server_fn(context: Context):
    create_save_unet(context)

    # Initialize global model
    net = init_net(context)

    weights = get_weights(net)
    initial_parameters = ndarrays_to_parameters(weights)

    # Define strategy
    strategy = get_strategy(context, initial_parameters)
    num_rounds = get_num_server_rounds(context)

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
