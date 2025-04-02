set -x

# Set to 1 to build singularity images
BUILD_IMAGES=0

# Change these variables based on execution env
export HF_CACHE="/home/alexsenden/.cache/huggingface"

# Config
export IPFS_API="127.0.0.1:5001"
export MINERS=8
export CLIENTS=8
export SERVERS=8
export ABI="NoScore"
export SCORING="none"

export NETWORK_ID=`jq -r '.config.chainId' testbed/ethereum/datadir/genesis_pow.json`

cd testbed

if [ $BUILD_IMAGES -eq 1 ]
then
    # Generate the eth accounts
    python3 toolkit.py generate-accounts 8 8

    # Update the genesis block and get the network ID
    python3 toolkit.py update-genesis
    export NETWORK_ID=`jq -r '.config.chainId' ethereum/datadir/genesis_pow.json`

    Build the singularity images
    python3 toolkit.py build-images
fi

# Run eth nodes
cd ../singularity
bash run_eth_nodes.sh
cd ../testbed

# Connect all nodes
python3 toolkit.py connect-peers $MINERS

# # Deploy contract and store addr in $CONTRACT
export CONTRACT=`python3 toolkit.py deploy-contract | sed -n 's/.*contract address: *\([0-9a-fA-Fx]*\).*/\1/p' | awk 'NR==2'`
echo $CONTRACT

# # Run the clients and servers
# cd ../singularity
# bash run_py_nodes.sh