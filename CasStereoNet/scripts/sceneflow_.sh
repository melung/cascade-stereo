#!/usr/bin/env bash

export PYTHONWARNINGS="ignore"

save_path="./outputs/sceneflow_pretrained"
ckpt_path="./checkpoints/pretrained_sceneflow" #/casgwcnet.ckpt"
add_args="--ndisps "48,24"  --disp_inter_r "4,1"   --dlossw "0.5,1.0,2.0"  --batch_size 2 --eval_freq 3  --model gwcnet-c"
if [ ! -d $save_path ];then
    mkdir -p $save_path
fi

DATAPATH="../data/sceneflow/Sampler/"


#CDUA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=$1 main.py --dataset sceneflow \
#    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
#    --test_datapath $DATAPATH --test_dataset sceneflow \
#    --epochs 16 --lrepochs "10,12,14,16:2" \
#    --crop_width 512  --crop_height 256 --test_crop_width 960  --test_crop_height 512 --using_ns --ns_size 3 \
#    --model gwcnet-c --logdir ${ckpt_path}  ${add_args} | tee -a  ${save_path}/log.txt

CDUA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test_sanitycheck.txt \
    --test_datapath $DATAPATH --test_dataset sceneflow --mode test\
    --epochs 16 --lrepochs "10,12,14,16:2" \
    --crop_width 512  --crop_height 256 --test_crop_width 960  --test_crop_height 512 --using_ns --ns_size 3 \
    --model gwcnet-c --logdir ${ckpt_path}  ${add_args} | tee -a  ${save_path}/log.txt