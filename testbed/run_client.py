import web3
import time
import click
import requests
import blocklearning
import blocklearning.scorers as scorers
import blocklearning.model_loaders as model_loaders
import blocklearning.weights_loaders as weights_loaders
import blocklearning.utilities as utilities
from blocklearning.contract import RoundPhase


@click.command()
@click.option(
    "--provider", default="http://127.0.0.1:8545", help="web3 API HTTP provider"
)
@click.option("--ipfs", default="/ip4/127.0.0.1/tcp/5001", help="IPFS API provider")
@click.option(
    "--abi", default="./build/contracts/NoScore.json", help="contract abi file"
)
@click.option(
    "--account", help="ethereum account to use for this computing server", required=True
)
@click.option("--passphrase", help="passphrase to unlock account", required=True)
@click.option("--contract", help="contract address", required=True)
@click.option("--log", default="/writable/log.log", help="logging file")
@click.option("--scoring", default=None, help="scoring method")
@click.option("--partition", default=None, help="dataset partition number of this node")
def main(provider, ipfs, abi, account, passphrase, contract, log, scoring, partition):
    log = utilities.setup_logger(log, "client")
    weights_loader = weights_loaders.IpfsWeightsLoader(ipfs)

    # Load Training and Testing Data
    trainloader, testloader = model_loaders.diffusion.load_data.prep_data(partition)

    # Get Contract and Register as Trainer
    print(f"Contract addr: {contract}")
    contract = blocklearning.Contract(log, provider, abi, account, passphrase, contract)

    print(f"Creating model loader - {partition}")
    # Load Model
    model_loader = model_loaders.IpfsModelLoader(
        contract, weights_loader, ipfs_api=ipfs
    )
    print(f"Model loader created - {partition}")
    model = model_loader.load()
    print(f"Model loaded - {partition}")

    trainer = blocklearning.Trainer(
        contract, weights_loader, model, (trainloader, testloader), logger=log, partition=partition
    )

    # Setup the scorer for the clients. Only Marginal Gain and BlockFlow run on the client
    # device and use the client's testing dataset as the validation dataset for the scores.
    scorer = None
    if scoring == "marginal-gain":
        scorer = scorers.MarginalGainScorer(
            log, contract, model, weights_loader, testloader
        )
    elif scoring == "blockflow":
        scorer = scorers.AccuracyScorer(
            log, contract, model, weights_loader, testloader
        )
    if scorer is not None:
        scorer = blocklearning.Scorer(contract, scorer=scorer, logger=log)

    while True:
        try:
            phase = contract.get_round_phase()
            if phase == RoundPhase.WAITING_FOR_UPDATES:
                trainer.train()
            elif phase == RoundPhase.WAITING_FOR_SCORES and scorer is not None:
                scorer.score()
        except web3.exceptions.ContractLogicError as err:
            print(err, flush=True)
        except requests.exceptions.ReadTimeout as err:
            print(err, flush=True)

        time.sleep(0.5)


main()
