set -x

index=$1

mkdir -p ../fs/client/$index

singularity instance start --nv -B ../build/contracts/NoScore.json:/abi.json,$HF_CACHE:/hf-cache,../fs/client/$index:/writable,../sd1.5:/model py_node.sif py-client-$index
singularity exec --nv --env HF_HOME=/hf-cache,INDEX=$index,CUDA_VISIBLE_DEVICES=$index instance://py-client-$index bash /run_client.sh &