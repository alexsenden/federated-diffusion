set -x

# Set to 1 to build singularity images
BUILD_IMAGES=1

# Config
export IPFS_API="127.0.0.1:5001"
export MINERS=8
export CLIENTS=8
export SERVERS=8
export ABI="NoScore"
export SCORING="none"

export NETWORK_ID=`jq -r '.config.chainId' testbed/ethereum/datadir/genesis_pow.json`

# Remove existing files and stop any lingering processes
rm -rf fs/*
singularity instance stop --all

cd testbed

if [ $BUILD_IMAGES -eq 1 ]
then
    # # Generate the eth accounts
    # python3 toolkit.py generate-accounts 8 8

    # # Update the genesis block and get the network ID
    # python3 toolkit.py update-genesis
    # export NETWORK_ID=`jq -r '.config.chainId' ethereum/datadir/genesis_pow.json`

    # Build the singularity images
    python3 toolkit.py build-images
fi