import json
import time
import torch

from .utilities import float_to_int
from .model_loaders.diffusion.diffusion_model import LOCAL_STEPS

SAVE_DIR = '/writable'


class Trainer:
    def __init__(self, contract, weights_loader, model, data, logger=None, priv=None, partition=1):
        print(f"Initializing trainer {partition}")
        self.logger = logger
        self.priv = priv
        self.weights_loader = weights_loader
        self.contract = contract
        (self.trainloader, self.testloader) = data
        self.model = model
        self.partition = partition
        self.__register()
        self.last_round = -1

    def train(self):
        (round, weights_id) = self.contract.get_training_round()
        
        # Prevent trainer from training multiple times per round
        if round <= self.last_round:
            return
        self.last_round = round

        if self.logger is not None:
            self.logger.info(
                json.dumps(
                    {
                        "event": "start",
                        "round": round,
                        "weights": weights_id,
                        "ts": time.time_ns(),
                    }
                )
            )

        if weights_id != "":
            weights = self.weights_loader.load(weights_id)
            self.model.set_weights(weights)
            
            # if (round % 10 == 1 or round == 2) and self.partition == 0:
            #     torch.save(self.model.get_unet().state_dict(), f"{SAVE_DIR}/model_round_{round}.pth")

        if self.logger is not None:
            self.logger.info(
                json.dumps(
                    {"event": "train_start", "round": round, "ts": time.time_ns()}
                )
            )

        print(f"Trainer {self.partition} starting training for round {round}")
        trainingAccuracy = float_to_int((1 / self.model.train(self.trainloader)) * 10000)
        
        print(f"Trainer {self.partition} starting validation for round {round}")
        validationAccuracy = float_to_int((1 / self.model.test(self.testloader)) * 10000)

        if self.logger is not None:
            self.logger.info(
                json.dumps({"event": "train_end", "round": round, "ts": time.time_ns()})
            )

        weights = self.model.get_weights()

        weights_id = self.weights_loader.store(weights)

        submission = {
            "trainingAccuracy": trainingAccuracy,
            "testingAccuracy": validationAccuracy,
            "trainingDataPoints": LOCAL_STEPS,
            "weights": weights_id,
        }
        self.contract.submit_submission(submission)

        if self.logger is not None:
            self.logger.info(
                json.dumps(
                    {
                        "event": "end",
                        "round": round,
                        "weights": weights_id,
                        "ts": time.time_ns(),
                        "submission": submission,
                    }
                )
            )

    # Private utilities
    def __register(self):
        print(f"__register -ing trainer {self.partition}")
        if self.logger is not None:
            self.logger.info(
                json.dumps({"event": "checking_registration", "ts": time.time_ns()})
            )

        self.contract.register_as_trainer()

        if self.logger is not None:
            self.logger.info(
                json.dumps({"event": "registration_checked", "ts": time.time_ns()})
            )
