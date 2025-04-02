

cd testbed

python3 toolkit.py generate-accounts 8 8

python3 toolkit.py update-genesis

python3 toolkit.py build-images

docker network create \
    --driver=bridge \
    --subnet=172.16.254.0/20 \
    bflnet

CONSENSUS=qbft MINERS=8 docker compose -f blockchain.yml -p bfl up

python3 toolkit.py connect-peers <network>

python3 toolkit.py deploy-contract

CONTRACT=0x8C3CBC8C31e5171C19d8a26af55E0db284Ae9b4B \
    DATASET=mnist MINERS=10 AGGREGATORS=10 SCORERS=0 TRAINERS=25 \
    SCORING="none" ABI=NoScore \
    docker compose -f ml.yml -p bfl-ml up