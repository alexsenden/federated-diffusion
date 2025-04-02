#!/bin/sh

set -x

IP=$(ifconfig eth0 | grep 'inet' | awk '{print $2}' | sed 's/addr://')
INDEX=$(dig -x $IP +short | sed 's/[^0-9]*//g')
ACCOUNT=$(jq -r '.trainers' accounts.json | jq 'keys_unsorted' | jq -r "nth($((INDEX-1)))")
PASSWORD=$(jq -r '.trainers' accounts.json | jq -r ".[\"$ACCOUNT\"]")

PRVINDEX=$((INDEX % MINERS))
if [ "$PRVINDEX" -eq "0" ]; then
  PRVINDEX=$MINERS
fi
PROVIDER=$(dig eth-miner-$PRVINDEX +short)

python run_client.py \
  --provider "http://$PROVIDER:8545" \
  --abi /abi.json \
  --ipfs $IPFS_API \
  --account $ACCOUNT \
  --passphrase $PASSWORD \
  --contract $CONTRACT \
  --log /log.log \
  --train /dataset/train/$((INDEX-1)).npz \
  --test /dataset/test/$((INDEX-1)).npz \
  --scoring $SCORING

