# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
from datetime import datetime
import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
import numpy as np
# import torchvision
from torchvision import datasets, models, transforms
# import time
import os
import scipy.io
import yaml
import math
import random
import numpy as np
# from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
from model import ft_net, ft_net_dense, ft_net_NAS
from model import PCB_dense_arc as PCB
from model import PCB_dense_arc_test as PCB_test
import output
from log import *
from util import *

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    logger.warning('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

#### option ####
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--data_dir',default='../dataset/match/pytorch',type=str, help='./test_data')
parser.add_argument('--model_dir', default='./model/pcb_rpp', type=str, help='save model path')
parser.add_argument('--log_dir', default='./logs/test', type=str, help='log dir')
parser.add_argument('--result_dir', default='./result/pcb_rpp', type=str, help='save result dir')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', default='none', choices=['none', 'resnet', 'densenet'], help='use PCB')
parser.add_argument('--RPP', action='store_true', help='use RPP', default=False)
parser.add_argument('--stage', default='pcb', type=str, help='save model path')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--scales',default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
opt = parser.parse_args()
config_path = os.path.join(opt.model_dir, 'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16'] 
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 751 

#### log ####
subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
log_dir = os.path.join(os.path.expanduser(opt.log_dir), subdir)
if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
    os.makedirs(log_dir)
set_logger(logger, log_dir)

#### seed ####
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#### gpu ####
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
use_gpu = torch.cuda.is_available()

#### multi scale ####
logger.info('We use the scale: %s'%opt.scales)
str_ms = opt.scales.split(',')
opt.ms = []
for s in str_ms:
    s_f = float(s)
    opt.ms.append(math.sqrt(s_f))

#### data ####
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.PCB != 'none':
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(opt.data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(opt.data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes

if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses)
else:
    model_structure = ft_net(opt.nclasses, stride = opt.stride)

if opt.PCB != 'none':
    model_structure = PCB(opt.nclasses)

if opt.RPP:
    model_structure = model_structure.convert_to_rpp()
model = load_network(opt, model_structure)

# Remove the final fc layer and classifier layer
if opt.PCB != 'none':
    model = PCB_test(model)
else:
    model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(opt, model, dataloaders['gallery'])
    query_feature = extract_feature(opt, model,dataloaders['query'])
    if opt.multi:
        mquery_feature = extract_feature(opt, model, dataloaders['multi-query'])
    
# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'query_f':query_feature.numpy()}
if not os.path.isdir(opt.result_dir):
    os.mkdir(opt.result_dir)
scipy.io.savemat(os.path.join(opt.result_dir, 'feature.mat'),result)

if opt.multi:
    result = {'mquery_f':mquery_feature.numpy()}
    scipy.io.savemat(os.path.join(opt.result_dir, 'multi_query.mat'),result)

output.gen_submission(image_datasets, query_feature, gallery_feature, opt.result_dir)