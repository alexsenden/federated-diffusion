#!/bin/sh

set -x

# cp /.ethereum/_geth/nodekey_owner /.ethereum/geth/nodekey
geth --networkid=${NETWORK_ID} --datadir=/writable/.ethereum "$@"
