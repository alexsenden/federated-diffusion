# Hack path so we can load 'blocklearning'
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
import logging
import time
import random
import json
import utilities as utilities
import blocklearning

# import blocklearning.model_loaders as model_loaders
# import blocklearning.weights_loaders as weights_loaders
# import blocklearning.utilities as butilities

NUM_TRAINERS_PER_ROUND = 4

# Setup Log
log_file = "../../blocklearning-results/results/CURRENT/logs/manager.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)


while os.path.exists(log_file):
    log_file = log_file.replace(".log", "_1.log")
    print(f"a log already exists, saving to {log_file}")


logging.basicConfig(
    level="INFO",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename=log_file, encoding="utf-8", mode="a+"),
    ],
)
log = logging.getLogger("manager")


def get_owner_account(data_dir):
    accounts = utilities.read_json(os.path.join(data_dir, "accounts.json"))
    account_address = list(accounts["owner"].keys())[0]
    account_password = accounts["owner"][account_address]
    return account_address, account_password


@click.command()
@click.option(
    "--provider", default="http://127.0.0.1:8545", help="web3 API HTTP provider"
)
@click.option(
    "--abi", default="../build/contracts/NoScore.json", help="contract abi file"
)
@click.option("--contract", required=True, help="contract address")
@click.option("--scoring", default="none", help="scoring method")
@click.option("--trainers", default="random", help="clients selection method")
@click.option(
    "--data-dir", default=utilities.default_datadir, help="ethereum data directory path"
)
@click.option(
    "--val",
    default="./datasets/mnist/5/owner_val.npz",
    help="validation data .npz file",
)
@click.option("--rounds", default=1, type=click.INT, help="number of rounds")
def main(provider, abi, contract, scoring, trainers, data_dir, val, rounds):
    print("Preparing to start training")
    account_address, account_password = get_owner_account(data_dir)
    print(f"Account retrieved address: {account_address}")
    contract = blocklearning.Contract(
        log, provider, abi, account_address, account_password, contract
    )
    print(f"Contract instatiated")
    # weights_loader = weights_loaders.IpfsWeightsLoader()
    # model_loader = model_loaders.IpfsModelLoader(contract, weights_loader)
    # model = model_loader.load()
    # x_val, y_val = butilities.numpy_load(val)

    def choose_participants():
        print("Choosing participants")
        all_trainers = contract.get_trainers()
        all_aggregators = contract.get_aggregators()

        round_trainers = None
        round_aggregators = all_aggregators
        round_scorers = None
        
        print(f"All trainers: {all_trainers}")
        print(f"Number of trainers: {len(all_trainers)}")
        print(f"Number of trainers per round: {NUM_TRAINERS_PER_ROUND}")

        if trainers == "random":
            round_trainers = random.sample(all_trainers, NUM_TRAINERS_PER_ROUND)
        elif trainers == "fcfs":
            round_trainers = random.randint(len(all_trainers) // 2, len(all_trainers))
        elif trainers == "all":
            round_trainers = all_trainers

        if scoring == "multi-krum":
            round_scorers = round_aggregators
        elif scoring == "blockflow" or scoring == "marginal-gain":
            round_scorers = round_trainers

        print("Participants chosen")

        return round_trainers, round_aggregators, round_scorers

    # def eval_model(weights):
    #   log.info(json.dumps({ 'event': 'eval_start', 'ts': time.time_ns(), 'round': round }))
    #   model.set_weights(weights_loader.load(weights))
    #   metrics = model.evaluate(x_val, y_val)
    #   accuracy = metrics['sparse_categorical_accuracy']
    #   log.info(json.dumps({ 'event': 'eval_end', 'ts': time.time_ns(), 'round': round }))
    #   return accuracy

    for i in range(0, rounds):
        round_trainers, round_aggregators, round_scorers = choose_participants()
        print(
            f"********************************\nRound {i} - Trainers: {round_trainers}, Aggregators: {round_aggregators}\n********************************\n"
        )
        log.info(
            json.dumps(
                {
                    "event": "start",
                    "trainers": round_trainers,
                    "aggregators": round_aggregators,
                    "scorers": round_scorers,
                    "ts": time.time_ns(),
                }
            )
        )

        if trainers == "fcfs":
            contract.start_round(round_trainers, round_aggregators)
        elif scoring != "none":
            contract.start_round(round_trainers, round_aggregators, round_scorers)
        else:
            contract.start_round(round_trainers, round_aggregators)

        round = contract.get_round()

        while (
            contract.get_round_phase()
            != blocklearning.RoundPhase.WAITING_FOR_TERMINATION
        ):
            time.sleep(0.25)

        contract.terminate_round()
        weights = contract.get_weights(round)
        # accuracy = eval_model(weights)

        log.info(
            json.dumps(
                {
                    "event": "end",
                    "ts": time.time_ns(),
                    "round": round,
                    "weights": weights,
                }
            )
        )


main()
