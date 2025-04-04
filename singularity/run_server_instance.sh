set -x

index=$1

mkdir -p ../fs/server/$index

singularity instance start -B ../build/contracts/NoScore.json:/abi.json,$HF_CACHE:/hf-cache,../fs/server/$index:/writable,../sd1.5:/model py_node.sif py-server-$index

# Run IPFS on server 0, silence output
if [ $index -eq 0 ]
then
    singularity exec --env IPFS_PATH=/writable/ipfs instance://py-server-$index bash -c "ipfs init; ipfs config Datastore.StorageMax 150GB; ipfs daemon > /dev/null 2>&1" &
fi

singularity exec --env HF_HOME=/hf-cache,INDEX=$index instance://py-server-$index bash /run_server.sh &