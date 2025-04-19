from flwr.server.strategy import (
    FedAvg,
    DifferentialPrivacyServerSideAdaptiveClipping,
    FaultTolerantFedAvg,
    FedAdagrad,
    FedAdam,
    FedYogi,
    FedAvgM,
    Krum,
    QFedAvg,
)
from flwr.common import Context

from central_aggregator.strategy.save_strategy import SaveModelStrategy


def get_strategy(context: Context, initial_parameters):
    """Return the strategy with the given name."""

    strategy_name = context.run_config["strategy"]
    base_args = {
        "initial_parameters": initial_parameters,
        "fraction_fit": context.run_config["fraction-fit"],
    }

    if strategy_name == "FedAvg":
        return SaveModelStrategy(FedAvg(**base_args))
    elif strategy_name == "QFedAvg":  # Broken Implementation
        return SaveModelStrategy(QFedAvg(**base_args))
    elif strategy_name == "FaultTolerantFedAvg":
        return SaveModelStrategy(FaultTolerantFedAvg(**base_args))
    elif strategy_name == "FedAdagrad":
        return SaveModelStrategy(FedAdagrad(**base_args))
    elif strategy_name == "FedAdam":
        return SaveModelStrategy(FedAdam(**base_args))
    elif strategy_name == "FedYogi":
        return SaveModelStrategy(FedYogi(**base_args))
    elif strategy_name == "FedAvgM":
        return SaveModelStrategy(FedAvgM(**base_args))
    elif strategy_name == "FedMedian":
        return SaveModelStrategy(FedAvgM(**base_args))
    elif strategy_name == "Krum":
        return SaveModelStrategy(Krum(**base_args))
    elif strategy_name == "MultiKrum":
        return SaveModelStrategy(
            Krum(
                **base_args,
                num_clients_to_keep=2,
            )
        )
    elif strategy_name == "DifferentialPrivacyServerSideAdaptiveClipping":
        return SaveModelStrategy(
            DifferentialPrivacyServerSideAdaptiveClipping(
                strategy=FedAvg(**base_args),
                # clipped_count_stddev=0.2,
                num_sampled_clients=context.run_config["dp-num-sampled-clients"],
                noise_multiplier=context.run_config["dp-noise-multiplier"],
            )
        )

    raise NotImplementedError(f"Unknown strategy: {strategy_name}")


def get_num_server_rounds(context: Context):
    strategy_name = context.run_config["strategy"]

    if strategy_name == "Krum":
        return context.run_config["krum-num-server-rounds"]
    elif strategy_name == "MultiKrum":
        return context.run_config["multikrum-num-server-rounds"]

    return context.run_config["num-server-rounds"]
