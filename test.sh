#!/usr/bin/env bash

case $1 in

    trick)
        python test.py \
            --gpu_ids 0 \
            --name warm5_s1_b8_lr2_p0.5 \
            --test_dir ../dataset/match/pytorch \
            --batchsize 32 \
            --which_epoch 59
        ;;

    pcb)
        python main_test.py \
            --gpu_ids 0 \
            --PCB densenet \
            --stage pcb \
            --data_dir ../dataset/match/pytorch \
            --model_dir ./model/base \
            --result_dir ./result/base \
            --batchsize 256 \
            --which_epoch last \
            --scales 1,0.9
        ;;

    rpp)
        python main_test.py \
            --gpu_ids 0 \
            --PCB densenet \
            --stage full \
            --RPP \
            --data_dir ../dataset/match/pytorch \
            --model_dir ./model/base \
            --result_dir ./result/base \
            --batchsize 256 \
            --which_epoch last \
            --ms 1,0.9
        ;;

    rerank)
        python rerank_output.py \
            --model_dir ./model/base \
            --result_dir ./result/base \
            --data_dir ../dataset/match/pytorch
        ;;

    *)
        echo "wrong argument"
		exit 1
    ;;
esac
exit 0

#../dataset/market1501/Market-1501-v15.09.15/pytorch
