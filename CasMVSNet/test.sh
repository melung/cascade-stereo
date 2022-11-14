#!/usr/bin/env bash
TESTPATH="data/"
TESTLIST="lists/our_list.txt"
CKPT_FILE=$1
python test.py --dataset=general_eval --batch_size=10 --num_view=5  --testpath=$TESTPATH  --testlist=$TESTLIST --loadckpt $CKPT_FILE ${@:2}
