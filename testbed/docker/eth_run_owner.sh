#!/bin/sh

set -x

# cp /.ethereum/_geth/nodekey_owner /.ethereum/geth/nodekey
geth --verbosity=1 --networkid=${NETWORK_ID} --datadir=/writable/.ethereum "$@"
