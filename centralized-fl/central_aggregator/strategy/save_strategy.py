from flwr.common import Parameters, Scalar
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, FitRes, EvaluateIns, EvaluateRes
from typing import Dict, List, Optional, Tuple, Union

from central_aggregator.strategy.save_unet import save_unet

SAVE_EACH_N_ROUNDS = 10


class SaveModelStrategy(Strategy):
    def __init__(self, strategy: Strategy) -> None:
        super().__init__()
        self.strategy = strategy

    def __repr__(self) -> str:
        return self.strategy.__repr__()

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics = self.strategy.aggregate_fit(
            server_round, results, failures
        )

        if server_round % SAVE_EACH_N_ROUNDS == 0:
            strategy_name = self.strategy.__repr__()[
                : self.strategy.__repr__().find("(")
            ]
            save_unet(strategy_name, server_round, aggregated_parameters)

        return aggregated_parameters, aggregated_metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.strategy.evaluate(server_round, parameters)

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        return self.strategy.num_fit_clients(num_available_clients)

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        return self.strategy.num_evaluation_clients(num_available_clients)
