set -x

cd ../singularity
singularity build --fakeroot --force eth_node.sif eth_node.def