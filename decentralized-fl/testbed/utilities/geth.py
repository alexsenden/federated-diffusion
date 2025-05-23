import os
import rlp
import string
import random
import tempfile
from eth_account import Account
from .json_utils import *
from .processes import *

default_datadir = "./ethereum/datadir"
default_password_length = 32


def generate_qbft_extra_data(validators):
    vanity = bytes.fromhex(
        "0000000000000000000000000000000000000000000000000000000000000000"
    )
    validators = map(lambda x: bytes.fromhex(x[2:]), list(validators))
    extra_data = "0x" + rlp.encode([vanity, list(validators), [], 0, []]).hex()
    return extra_data


def update_genesis(src, dst, consensus, balance, data_dir):
    genesis = read_json(src)
    genesis["alloc"] = {}
    accounts = read_json(os.path.join(data_dir, "accounts.json"))

    for account in accounts["miners"]:
        genesis["alloc"][account] = {"balance": balance}

    for account in accounts["trainers"]:
        genesis["alloc"][account] = {"balance": balance}

    for account in accounts["owner"]:
        genesis["alloc"][account] = {"balance": balance}

    if consensus == "poa":
        extra_data = (
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        )
        for account in list(accounts["miners"]):
            extra_data += account[2:]
        extra_data += "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        genesis["extraData"] = extra_data

    if consensus == "qbft":
        validators = read_json(os.path.join(data_dir, "miners.json"))
        genesis["extraData"] = generate_qbft_extra_data(validators)

    save_json(dst, genesis)


def get_random_string(length):
    characters = string.ascii_letters + string.digits
    result_str = "".join(random.choice(characters) for i in range(length))
    return result_str


def generate_account(password, data_dir):
    print("Generating individual account")
    with tempfile.NamedTemporaryFile() as fp:
        print(fp.name)
        fp.write(password.encode())
        fp.seek(0)

        out = run_and_output(
            f"geth --datadir {data_dir} account new --password {fp.name}"
        )
        
        print(f"Generated - {out}")
        return out.split("Public address of the key:")[1].split("\n")[0].strip()


def generate_keys(nodekey, host, data_dir):
    nodekey_file = os.path.join(data_dir, "geth", nodekey)
    run_and_output(f"bootnode -genkey {nodekey_file}")

    with open(nodekey_file, "r") as f:
        privkey = f.read()
        account = Account.from_key(privkey)

    pubkey = run_and_output(f"bootnode -nodekey {nodekey_file} -writeaddress").strip()
    enode = f"enode://{pubkey}@{host}:30303"
    address = account.address
    return address, enode


def generate_accounts(miners, clients, data_dir, password_length):
    print(f'Generating accounts')
    accounts = {"owner": {}, "miners": {}, "trainers": {}}

    static_nodes = []
    miner_addresses = []

    password = get_random_string(password_length)
    address = generate_account(password, data_dir)
    accounts["owner"][address] = password

    os.makedirs(f"{data_dir}/geth/")

    _, enode = generate_keys("nodekey_owner", "eth-owner", data_dir)
    static_nodes.append(enode)

    for i in range(0, miners):
        print(f'Generating miner account {i}')
        password = get_random_string(password_length)
        address = generate_account(password, data_dir)
        accounts["miners"][address] = password

        address, enode = generate_keys(f"nodekey_{i}", f"eth-miner-{i}", data_dir)
        # static_nodes.append(enode)
        miner_addresses.append(address)

    for i in range(0, clients):
        print(f'Generating client account {i}')
        password = get_random_string(password_length)
        address = generate_account(password, data_dir)
        accounts["trainers"][address] = password

    static_nodes[0] = static_nodes[0].replace("eth-owner", "127.0.0.1")

    print(f'Preparing to save')
    save_json(os.path.join(data_dir, "accounts.json"), accounts)
    save_json(os.path.join(data_dir, "miners.json"), miner_addresses)
    save_json(os.path.join(data_dir, "geth", "static-nodes.json"), static_nodes)
