# BlockLearning Framework

This code is forked from [hacdias/blocklearning](https://github.com/hacdias/blocklearning) at [TU Eindhoven](https://tue.nl/).
Please additionally cite that work when using this.

### Structure

- [`blocklearning/`](./blocklearning/) contains the modular Python library.
- [`contracts/`](./contracts/) contains Solidity smart contracts.
- [`migrations/`](./migrations/) contains smart contract migrations necessary to deploy the smart contracts using [Truffle](https://trufflesuite.com/).
- [`testbed/`](./testbed/) is the code necessary to run the framework in an experimental setup using ~~Docker~~ Singularity.
- [`singularity/`](./singularity/) contains scripts and container definitions to allow this project to run in Singularity containers (for HPC contexts).

### Running

Ensure the following dependencies are installed:

- [`go-ethereum`](https://github.com/ethereum/go-ethereum)
- [`Singularity`](https://github.com/sylabs/singularity)
- Truffle (Node module, tested with Node 18), `npm install -g truffle`
- Python Dependencies `pip install -r ../requirements.txt`

Then, run the `run-decentralized-fl.sh` script

```
bash run-decentralized-fl.sh
```