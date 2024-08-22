#!/bin/bash
mkdir -p ./logs
API_SERVER=0

echo $API_SERVER
date

for i in {51..100}
do
  timeout 72000s bash -c "
    echo \"Combat $i...\"
    python run_werewolf.py \
      --current-game-number $i \
      --message-window 15 \
      --answer-topk 3 \
      --retri-question-number 3 \
      --exps-retrieval-threshold 0.80 \
      --similar-exps-threshold 0.01 \
      --max-tokens 512 \
      --temperature 0.3 \
      --use-api-server $API_SERVER \
      --version 1.4 \
      --environment-config ./examples/werewolf-8-3.json \
      --role-config ./config/1.json \
      > ./logs7/$i.con 2>&1
    echo \"Combat $i...OK!\"
  " &
done

wait

