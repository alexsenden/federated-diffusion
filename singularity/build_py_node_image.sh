set -x

cd ../singularity
singularity build --fakeroot --force py_node.sif py_node.def