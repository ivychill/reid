#!/usr/bin/env bash

case $1 in

    trick)
        python test.py \
            --gpu_ids 2 \
            --name warm5_s1_b8_lr2_p0.5 \
            --test_dir ../dataset/match/pytorch \
            --batchsize 32 \
            --which_epoch 59
        ;;

    pcb)
        python test.py \
            --gpu_ids 2 \
            --name PCB \
            --PCB \
            --test_dir ../dataset/match/pytorch \
            --batchsize 32 \
            --which_epoch 59
        ;;

    pcb_ms)
        python test.py \
            --gpu_ids 2 \
            --name pcb_ms \
            --PCB \
            --test_dir ../dataset/match/pytorch \
            --batchsize 32 \
            --which_epoch 59 \
            --ms 1,0.9
        ;;

    pcb_trick)
        python test.py \
            --gpu_ids 1 \
            --name pcb_trick \
            --PCB \
            --test_dir ../dataset/match/pytorch \
            --batchsize 256 \
            --which_epoch 119 \
            --ms 1,0.9
        ;;

    pcb_rpp)
        python test.py \
            --gpu_ids 1 \
            --name pcb_trick \
            --PCB densenet
            --stage full
            --test_dir ../dataset/match/pytorch \
            --batchsize 256 \
            --which_epoch 119 \
            --ms 1,0.9
        ;;

    rerank)
        python rerank_output.py \
            --name pcb_trick \
            --data_dir ../dataset/match/pytorch
        ;;

    *)
        echo "wrong argument"
		exit 1
    ;;
esac
exit 0

#../dataset/market1501/Market-1501-v15.09.15/pytorch