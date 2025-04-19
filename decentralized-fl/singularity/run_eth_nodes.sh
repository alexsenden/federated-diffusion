set -x

bash run_owner_instance.sh

for ((i=0;i<MINERS;i++)); do
    bash run_miner_instance.sh $i
done