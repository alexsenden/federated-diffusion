set -x

index=$1

singularity instance start -B ../build/contracts/NoScore.json:/abi.json,$HF_CACHE:/hf-cache py_node.sif py-server-$index
singularity exec instance://py-server-$index export HF_CACHE=/hf-cache
singularity exec instance://py-server-$index bash /run_server.sh &