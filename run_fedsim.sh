#!/usr/bin/env bash
python3  -u main.py --dataset=$1 --optimizer='fedsim'  \
            --learning_rate=0.01 --num_rounds=200 --clients_per_round=$4 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --model='mclr' \
            --drop_percent=$2 \
            --num_groups=$3 \
            --ex_name=$5 \
            --seed=0