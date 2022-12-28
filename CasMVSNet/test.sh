
TESTPATH="data/"
TESTLIST="lists/our_list.txt"

python test.py --dataset=general_eval --batch_size=15 --num_view=5  --testpath=$TESTPATH  --testlist=$TESTLIST --filter_method=gipuma
