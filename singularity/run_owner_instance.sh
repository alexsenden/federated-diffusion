set -x

outpath=../fs/eth/owner

mkdir -p ../fs/eth/owner/.ethereum/geth

cp -r "../testbed/ethereum/datadir/keystore" $outpath/.ethereum/keystore
cp "../testbed/ethereum/datadir/geth/static-nodes.json" $outpath/.ethereum/geth/static-nodes.json
cp -r "../testbed/ethereum/datadir/geth" $outpath/.ethereum/_geth
cp "../testbed/ethereum/datadir/geth/nodekey_owner" $outpath/.ethereum/geth/nodekey

singularity instance start -B ../fs/eth/owner:/writable eth_node.sif eth-owner
singularity exec instance://eth-owner geth --datadir=/writable/.ethereum init /genesis.json && rm -f /writable/.ethereum/geth/nodekey
singularity exec instance://eth-owner bash /run_owner.sh --allow-insecure-unlock --http --http.addr="0.0.0.0" --http.api="eth,web3,net,admin,personal" --http.corsdomain="*" --netrestrict="127.0.0.1/27" &

#singularity exec instance://eth-owner bash /run_owner.sh --allow-insecure-unlock --http --http.addr="0.0.0.0" --http.api="eth,web3,net,admin,personal" --http.corsdomain="*" --netrestrict="172.16.254.0/20" 