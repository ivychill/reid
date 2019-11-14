#!/usr/bin/env bash

case $1 in

    res)
        python train.py \
            --gpu_ids 0 \
            --name ft_ResNet50 \
            --train_all \
            --batchsize 32  \
            --data_dir ../dataset/match/pytorch
        ;;

    trick)
        python train.py \
            --gpu_ids 1 \
            --train_all \
            --name trick \
            --warm_epoch 5 \
            --stride 1 \
            --erasing_p 0.5 \
            --batchsize 8 \
            --lr 0.02 \
            --data_dir ../dataset/match/pytorch
        ;;

    pcb)
        python train.py \
            --gpu_ids 2 \
            --name pcb \
            --PCB \
            --train_all \
            --lr 0.02 \
            --data_dir ../dataset/match/pytorch
        ;;

    pcb_trick)
        python train.py \
            --gpu_ids 1 \
            --train_all \
            --name lr0.1 \
            --PCB \
            --warm_epoch 5 \
            --stride 1 \
            --erasing_p 0.5 \
            --lr 0.02 \
            --data_dir ../dataset/match/pytorch
        ;;

    pcb_rpp)
        python main_train.py \
            --gpu_ids 2 \
            --train_all \
            --model_dir ./model/pcb_rpp \
            --PCB densenet \
            --RPP \
            --warm_epoch 5 \
            --stride 1 \
            --erasing_p 0.5 \
            --lr 0.02 \
            --data_dir ../dataset/match/pytorch
        ;;

    rpp)
        python rpp.py \
            --gpu_ids 2 \
            --train_all \
            --model_dir ./model/pcb_rpp \
            --PCB densenet \
            --warm_epoch 5 \
            --erasing_p 0.5 \
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

#../dataset/market1501/Market-1501-v15.09.15/pytorch