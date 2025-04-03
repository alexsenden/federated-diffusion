set -x

index=$1

mkdir -p ../fs/client/$index

singularity instance start -B ../build/contracts/NoScore.json:/abi.json,$HF_CACHE:/hf-cache,../fs/client/$index:/writable py_node.sif py-client-$index
singularity exec --env HF_HOME=/hf-cache,INDEX=$index instance://py-client-$index bash /run_client.sh &