
TESTPATH="data/"
TESTLIST="lists/our_list.txt"

python test_near_realtime.py --dataset=mdi_melungl --batch_size=6 --num_view=5  --testpath=$TESTPATH  --testlist=$TESTLIST --filter_method=gipuma
