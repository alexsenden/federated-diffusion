#!/bin/sh

set -x

echo $INDEX

ACCOUNT=$(jq -r '.trainers' /accounts.json | jq 'keys_unsorted' | jq -r "nth($INDEX)")
PASSWORD=$(jq -r '.trainers' /accounts.json | jq -r ".[\"$ACCOUNT\"]")
PROVIDER=127.0.0.1:880$INDEX

python /run_client.py \
    --provider="http://$PROVIDER" \
    --abi=/abi.json \
    --ipfs=$IPFS_API \
    --account=$ACCOUNT \
    --passphrase=$PASSWORD \
    --contract=$CONTRACT \
    --log=/writable/log.log \
    --scoring=$SCORING \
    --partition=$INDEX &

