# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
# from model import ft_net, ft_net_dense, ft_net_NAS, PCB
from model import ft_net, ft_net_dense, ft_net_NAS
from model import PCB_dense as PCB
from random_erasing import RandomErasing
import yaml
import math
import random
import numpy as np
from shutil import copyfile
from log import *
from util import *


def train_model(model, criterion, optimizer, scheduler, log_file, stage, num_epochs=25):
    since = time.time()

    #best_model_wts = model.state_dict()
    #best_acc = 0.0
    last_model_wts = model.state_dict()
    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                if opt.PCB == 'none':
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                else:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) + sm(part[2]) + sm(part[3]) + sm(part[4]) + sm(part[5])
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part-1):
                        loss += criterion(part[i+1], labels)

                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    if opt.fp16: # we use optimier to backward loss
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()

                # statistics
                version = torch.__version__
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * inputs.size(0)
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * inputs.size(0)
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            log_file.write('{} epoch : {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc) + '\n')
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)            
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch%10 == 9:
                    save_network(opt, model, epoch, stage)
                draw_curve(epoch)

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(opt, model, 'last', stage)
    return model


######################################################################
# PCB train
# ------------------
# Step1 : train the PCB model
# According to original paper, we set the difference learning rate for difference layers.
def pcb_train(model, criterion, log_file, stage, num_epoch):
    ignored_params = list(map(id, model.classifiers.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        # {'params': base_params, 'lr': 0.01},
        # {'params': model.classifiers.parameters(), 'lr': 0.1},
        {'params': base_params, 'lr': 0.1 * opt.lr},
        {'params': model.classifiers.parameters(), 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        log_file, stage, num_epochs=num_epoch)
    return model


######################################################################
# RPP train
# ------------------
# Setp 2&3: train the rpp layers
# According to original paper, we set the learning rate at 0.01 for rpp layers.
def rpp_train(model, criterion, log_file, stage, num_epoch):
    # ignored_params = list(map(id, get_net(opt, model).avgpool.parameters()))
    # base_params = filter(lambda p: id(p) not in ignored_params, get_net(opt, model).parameters())
    # optimizer_ft = optim.SGD([
    #     {'params': base_params, 'lr': 0.00},
    #     {'params': get_net(opt, model).avgpool.parameters(), 'lr': 0.01},
    # ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    # optimizer_ft = optim.SGD(model.avgpool.parameters(), lr=0.01,
    #                           weight_decay=5e-4, momentum=0.9, nesterov=True)
    optimizer_ft = optim.SGD(model.avgpool.parameters(), lr=0.1 * opt.lr,
                              weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 100 epochs (never use)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        log_file, stage, num_epochs=num_epoch)
    return model

######################################################################
# full train
# ------------------
# Step 4: train the whole net
# According to original paper, we set the difference learning rate for the whole net
def full_train(model, criterion, log_file, stage, num_epoch):
    ignored_params = list(map(id, model.classifiers.parameters()))
    ignored_params += list(map(id, model.avgpool.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        # {'params': base_params, 'lr': 0.001},
        # {'params': model.classifiers.parameters(), 'lr': 0.01},
        # {'params': model.avgpool.parameters(), 'lr': 0.01},
        {'params': base_params, 'lr': 0.01 * opt.lr},
        {'params': model.classifiers.parameters(), 'lr': 0.1 * opt.lr},
        {'params': model.avgpool.parameters(), 'lr': 0.1 * opt.lr},
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 100 epochs (never use)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        log_file, stage, num_epochs=num_epoch)
    return model

######################################################################
# Draw Curve
#---------------------------
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join(opt.model_dir, 'train.jpg'))


if __name__ == '__main__':
    try:
        from apex.fp16_utils import *
        from apex import amp, optimizers
    except ImportError: # will be 3.x series
        logger.warning('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

    #### Options ####
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids',default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='training dir path')
    parser.add_argument('--model_dir',default='./model', type=str, help='output model dir')
    parser.add_argument('--log_dir',default='./logs/train', type=str, help='log dir')
    parser.add_argument('--train_all', action='store_true', help='use all training data' )
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
    parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    parser.add_argument('--PCB', default='none', choices=['none', 'resnet', 'densenet'], help='use PCB')
    parser.add_argument('--RPP', action='store_true', help='use RPP')
    parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
    opt = parser.parse_args()

    #### log ####
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(opt.log_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    set_logger(logger, log_dir)

    ####  setSeed ####
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #### gpu ####
    str_ids = opt.gpu_ids.split(',')
    opt.gids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            opt.gids.append(gid)
    if len(opt.gids)>0:
        torch.cuda.set_device(opt.gids[0])
    use_gpu = torch.cuda.is_available()

    #### Load Data ####
    transform_train_list = [
            #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
            transforms.Resize((256,128), interpolation=3),
            transforms.Pad(10),
            transforms.RandomCrop((256,128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    transform_val_list = [
            transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    if opt.PCB != 'none':
        transform_train_list = [
            transforms.Resize((384,192), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        transform_val_list = [
            transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

    if opt.erasing_p>0:
        transform_train_list = transform_train_list + [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]
    if opt.color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

    logger.info('transform {0}'.format(transform_train_list))
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    train_all = ''
    if opt.train_all:
        train_all = '_all'

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(opt.data_dir, 'train' + train_all), data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(opt.data_dir, 'val'), data_transforms['val'])
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
        shuffle=True, num_workers=8, pin_memory=True) for x in ['train', 'val']} # 8 workers may work faster

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    since = time.time()
    inputs, classes = next(iter(dataloaders['train']))
    logger.debug('dataloaders cost time {0} s'.format(time.time()-since))

    #### loss and metric statistics ####
    y_loss = {} # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")

    #### model ####
    if opt.use_dense:
        model = ft_net_dense(len(class_names), opt.droprate)
    elif opt.use_NAS:
        model = ft_net_NAS(len(class_names), opt.droprate)
    else:
        model = ft_net(len(class_names), opt.droprate, opt.stride)
    if opt.PCB != 'none':
        model = PCB(len(class_names))
    opt.nclasses = len(class_names)
    # logger.debug(str(model))

    #### save args and model ####
    if not os.path.isdir(opt.model_dir):
        os.mkdir(opt.model_dir)
    with open(os.path.join(opt.model_dir, 'opts.yaml'),'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)
    log_file = open(os.path.join(opt.model_dir, 'train.log'), 'w')
    copyfile('./main_train.py', os.path.join(opt.model_dir, 'main_train.py'))
    copyfile('./model.py', os.path.join(opt.model_dir, 'model.py'))

    #### start train ####
    # step1: PCB training #
    stage = 'pcb'
    model = PCB(len(class_names))
    if use_gpu:
        model = model.cuda()

    opt.warm_epoch = 5
    criterion = nn.CrossEntropyLoss()
    model = pcb_train(model, criterion, log_file, stage, 120)

    # step2&3: RPP training #
    if opt.RPP:
        stage = 'rpp'
        model = model.convert_to_rpp()
        if use_gpu:
            model = model.cuda()
        opt.warm_epoch = 0
        model = rpp_train(model, criterion, log_file, stage, 10)

        # step4: whole net training #
        stage = 'full'
        opt.warm_epoch = 0
        full_train(model, criterion, log_file, stage, 20)

    log_file.close()