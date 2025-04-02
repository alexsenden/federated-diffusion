# export CONTRACT=`

regex="'NoScore'[\s\S]*?contract\saddress:\s+(.*)"
regex2="s/\'NoScore\'[\s\S]*?contract/\1/"
CONTRACT=`cat testbed/out.txt | sed -n 's/.*contract address: *\([0-9a-fA-Fx]*\).*/\1/p' | awk 'NR==2'` # Pipe into something that grabs the contract addr
echo $res
# echo ${BASH_REMATCH[1]}