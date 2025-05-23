import os
import click
import shutil

from web3 import Web3

import utilities.processes as proc
import utilities.geth as geth
import utilities.json_utils as ujson
import utilities.singularity as singularity
from utilities.unlock_account import unlock_account


@click.group()
def cli():
    pass


@cli.command()
@click.option("-d", "--dir", default="..", help="project base directory")
def build_images(dir):
    # os.system("docker pull ethereum/client-go:v1.10.16")

    # Only using pow for this project
    # bases = {
    # "poa": "ethereum/client-go:v1.10.16",
    # "pow": "ethereum/client-go:v1.10.16",
    # "qbft": "quorumengineering/quorum:22.4.0",
    # }

    # node_dockerfile = os.path.join(dir, "testbed/docker/Dockerfile.node")
    # bfl_dockerfile = os.path.join(dir, "testbed/docker/Dockerfile.bfl")

    node_dockerfile = os.path.join(dir, "singularity/build_eth_node_image.sh")
    bfl_dockerfile = os.path.join(dir, "singularity/build_py_node_image.sh")

    os.system(f"bash {node_dockerfile}")
    os.system(f"bash {bfl_dockerfile}")

    # for consensus in bases:
    #     os.system(
    #         f"docker build -f {node_dockerfile} -t geth-node-{consensus} --build-arg GENESIS=genesis_{consensus}.json --build-arg BASE={bases[consensus]} {dir}"
    #     )

    # os.system(f"docker build -f {bfl_dockerfile} -t bfl-node {dir}")


@cli.command()
@click.option(
    "-d", "--dir", default="../experiment/results/CURRENT/logs", help="logs directory"
)
def collect_logs(dir):
    output = proc.run_and_output(
        'docker ps -a --format "{{ .ID }} {{ .Names }}"'
    ).strip()
    output = list(map(lambda x: x.split(" "), output.split("\n")))
    os.makedirs(dir, exist_ok=True)

    for [id, name] in output:
        if name.startswith("bfl-ml-client-") or name.startswith("bfl-ml-server-"):
            [type, number] = name.replace("bfl-ml-", "").split("-")
            src = f"{id}:/root/log.log"
            dst = os.path.join(dir, f"{type}_{number}.log")
            proc.run_and_output(f"docker cp {src} {dst}")
            print(id, type, number)


@cli.command()
@click.option("-g", "--genesis", default="./ethereum", help="genesis base path")
@click.option(
    "-b",
    "--balance",
    default="1000000000000000000000",
    help="balance to add to each account",
)
@click.option(
    "-d",
    "--data-dir",
    default=geth.default_datadir,
    help="ethereum data directory path",
)
def update_genesis(genesis, balance, data_dir):
    for consensus in ["poa", "pow", "qbft"]:
        src = os.path.join(genesis, f"genesis_{consensus}.json")
        dst = os.path.join(data_dir, f"genesis_{consensus}.json")
        geth.update_genesis(src, dst, consensus, balance, data_dir)


@cli.command()
@click.argument("miners", type=click.INT)
@click.argument("trainers", type=click.INT)
@click.option(
    "-d",
    "--data-dir",
    default=geth.default_datadir,
    help="ethereum data directory path",
)
@click.option(
    "-pl",
    "--password-length",
    type=click.INT,
    default=geth.default_password_length,
    help="password length",
)
def generate_accounts(miners, trainers, data_dir, password_length):
    print(f'Generating accounts to {data_dir}')
    shutil.rmtree(data_dir, ignore_errors=True)
    print(f'Datadir removed')
    geth.generate_accounts(
        miners, trainers, data_dir=data_dir, password_length=password_length
    )
    print(f'generate_accounts - done')


@cli.command()
@click.option(
    "-d",
    "--data-dir",
    default=geth.default_datadir,
    help="ethereum data directory path",
)
@click.option(
    "-p", "--provider", default="http://127.0.0.1:8545", help="ethereum API provider"
)
@click.option(
    "-t", "--trufflepath", default="~/bin/.npm-packages/node_modules/.bin/truffle", help="path to truffle executable (can replace with truffle if on path)"
)
def deploy_contract(data_dir, provider, trufflepath):
    accounts = ujson.read_json(os.path.join(data_dir, "accounts.json"))
    account_address = list(accounts["owner"].keys())[0]
    account_password = accounts["owner"][account_address]

    web3 = Web3(Web3.HTTPProvider(provider))

    address = Web3.to_checksum_address(account_address)

    unlock_account(address, account_password)

    os.system(f"cd .. && {trufflepath} migrate --reset")


@cli.command()
@click.argument("num_miners", type=click.INT)
def connect_peers(num_miners):
    # docker.connect_all(docker.get_enodes(network))
    singularity.connect_all(singularity.get_enodes(num_miners))


cli()
