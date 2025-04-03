set -x

index=$1

mkdir -p ../fs/server/$index

singularity instance start -B ../build/contracts/NoScore.json:/abi.json,$HF_CACHE:/hf-cache,../fs/server/$index:/writable py_node.sif py-server-$index

# Run IPFS on server 0
if [ $index -eq 0 ]
then
    singularity exec --env IPFS_PATH=/writable/ipfs instance://py-server-$index bash -c "ipfs init; ipfs daemon" &
fi

singularity exec --env HF_HOME=/hf-cache,INDEX=$index instance://py-server-$index bash /run_server.sh &