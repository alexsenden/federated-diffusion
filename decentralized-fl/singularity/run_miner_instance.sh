set -x

index=$1

outpath=../fs/eth/$index

mkdir -p $outpath/.ethereum/geth

cp -r "../testbed/ethereum/datadir/keystore" $outpath/.ethereum/keystore
cp "../testbed/ethereum/datadir/geth/static-nodes.json" $outpath/.ethereum/geth/static-nodes.json
cp -r "../testbed/ethereum/datadir/geth" $outpath/.ethereum/_geth
cp "../testbed/ethereum/datadir/geth/nodekey_$index" $outpath/.ethereum/geth/nodekey

singularity instance start -B $outpath:/writable eth_node.sif eth-miner-$index
singularity exec instance://eth-miner-$index geth --datadir=/writable/.ethereum init /genesis.json && rm -f /writable/.ethereum/geth/nodekey
singularity exec --env INDEX=$index instance://eth-miner-$index bash /run_miner.sh --mine --miner.threads=1 --allow-insecure-unlock --http --http.addr="0.0.0.0" --http.port="880$index" --http.api="eth,web3,net,admin,personal" --http.corsdomain="*" --netrestrict="127.0.0.1/27" --port="3033$index" > ../fs/eth/$index/log.out &


# singularity exec --env INDEX=$index instance://eth-miner-$index bash /run_miner.sh --mine --miner.threads=1 --allow-insecure-unlock --http --http.addr="0.0.0.0" --http.api="eth,web3,net,admin,personal" --http.corsdomain="*" --netrestrict="172.16.254.0/20" &
