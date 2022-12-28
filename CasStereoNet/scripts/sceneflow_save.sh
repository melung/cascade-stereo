#!/usr/bin/env bash
#set -x
export PYTHONWARNINGS="ignore"

save_path="./outputs/sceneflow_pretrained"
ckpt_path="./checkpoints/pretrained_sceneflow/casgwcnet.ckpt"
add_args="--ndisps "48,24" --disp_inter_r "4,1"   --dlossw "0.5,1.0,2.0" --model gwcnet"
if [ ! -d $save_path ];then
    mkdir -p $save_path
fi

DATAPATH="../data/sceneflow/Sampler/"
CDUA_VISIBLE_DEVICES="0" python save_disp.py --test_dataset sceneflow  --test_datapath $DATAPATH \
               --testlist ./filenames/sceneflow_test_sanitycheck.txt \
               --model gwcnet-c  --loadckpt ${ckpt_path}   \
               --logdir ${save_path} \
               --test_crop_width 960  --test_crop_height 512 \
               --ndisp "48,24" --disp_inter_r "4,1" --dlossw "0.5,2.0"  --using_ns --ns_size 3 \
               ${add_args} | tee -a  $save_path/log.txt
