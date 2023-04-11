TESTLIST="lists/our_list.txt"
python test_near_realtime.py --dataset=mdi_melungl --batch_size=6 --num_view=4 --testlist=$TESTLIST --filter_method=gipuma
