set -x

index=$1

singularity instance start -B ../build/contracts/NoScore.json:/abi.json,$HF_CACHE:/hf-cache my_image.sif py_node.sif py-client-$index
singularity exec instance://py-client-$index export HF_CACHE=/hf-cache
singularity exec instance://py-client-$index bash /run_client.sh &