#!/bin/sh

set -x

echo $INDEX

ACCOUNT=$(jq -r '.miners' /accounts.json | jq 'keys_unsorted' | jq -r "nth($INDEX)")
PASSWORD=$(jq -r '.miners' /accounts.json | jq -r ".[\"$ACCOUNT\"]")

# cp /.ethereum/_geth/nodekey_${INDEX} /.ethereum/geth/nodekey
echo $PASSWORD > /writable/password.txt

geth --networkid=${NETWORK_ID} --miner.etherbase=${ACCOUNT} --unlock=${ACCOUNT} --syncmode=full --password=/writable/password.txt --datadir=/writable/.ethereum "$@"
