# NUM_SERVERS = 8
# NUM_CLIENTS = 8

# Env
# IPFS_API=/dns/host.docker.internal/tcp/5001
# CONTRACT=$CONTRACT
# SCORING=$SCORING
# MINERS=$MINERS

# volumes:
# - './datasets/$DATASET/$CLIENTS:/root/dataset'
# - ../build/contracts/$ABI.json:/root/abi.json

set -x

for ((i=0;i<SERVERS;i++)); do
    bash run_server_instance.sh $i
done

for ((i=0;i<CLIENTS;i++)); do
    bash run_client_instance.sh $i
done