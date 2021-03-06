#!/usr/bin/env bash

case $1 in

    trick)
        python train.py \
            --gpu_ids 0 \
            --train_all \
            --name trick \
            --warm_epoch 5 \
            --stride 1 \
            --erasing_p 0.5 \
            --batchsize 8 \
            --lr 0.02 \
            --data_dir ../dataset/match/pytorch
        ;;

    main)
        python main_train_arc.py \
            --gpu_ids 0 \
            --train_all \
            --model_dir ./model/base \
            --PCB densenet \
            --warm_epoch 5 \
            --stride 1 \
            --erasing_p 0 \
            --batchsize 64 \
            --lr 0.02 \
            --data_dir ../dataset/match/pytorch
        ;;

    rpp)
        python rpp.py \
            --gpu_ids 0 \
            --train_all \
            --model_dir ./model/base \
            --PCB densenet \
            --stage pcb \
            --warm_epoch 0 \
            --erasing_p 0 \
            --batchsize 64 \
            --lr 0.02 \
            --which_epoch 119 \
            --data_dir ../dataset/match/pytorch
        ;;

    *)
        echo "wrong argument"
		exit 1
    ;;
esac
exit 0

#../dataset/market1501/pytorch