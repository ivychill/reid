import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from log import logger


def load_network(opt, network):
    save_path = os.path.join(opt.model_dir, opt.stage, 'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

def save_network(opt, network, epoch_label, stage):
    save_filename = 'net_%s.pth'% epoch_label
    save_sub_dir = os.path.join(opt.model_dir, stage)
    if not os.path.isdir(save_sub_dir):
        os.mkdir(save_sub_dir)
    save_path = os.path.join(save_sub_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(opt.gids[0])

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

# used by test
# the model should be PCB_test instead of PCB
def extract_feature(opt, model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        logger.debug('extract_feature {0}'.format(count))
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if opt.PCB == 'resnet':
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts
        elif opt.PCB == 'densenet':
            ff = torch.FloatTensor(n,1024,6).zero_().cuda() # we have six parts

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in opt.ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)  # [256, 1024, 6], [batch_size, fc, part]
                ff += outputs               # [256, 1024, 6], [batch_size, fc, part]
        # norm feature
        if opt.PCB != 'none':
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)    # [256, 6144], [batch_size, fc*part]
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)   # [256, 6144], [batch_size, fc*part]
    return features

# used by train
# the model should be PCB instead of PCB_test
def extract_feature_and_label(opt, model, dataloaders):
    features = torch.FloatTensor()
    labels = []
    count = 0
    for data in dataloaders:
        img, label = data
        labels.extend(label)

        n, c, h, w = img.size()
        count += n
        # logger.debug('extract_feature {0}'.format(count))
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if opt.PCB == 'resnet':
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts
        elif opt.PCB == 'densenet':
            ff = torch.FloatTensor(n,1024,6).zero_().cuda() # we have six parts

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in opt.ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                _, feats = model(input_img)  # [256, 1024, 6], [batch_size, fc, part]
                ff += feats               # [256, 1024, 6], [batch_size, fc, part]
        # norm feature
        if opt.PCB != 'none':
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)    # [256, 6144], [batch_size, fc*part]
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)   # [256, 6144], [batch_size, fc*part]
    return features, np.asarray(labels)