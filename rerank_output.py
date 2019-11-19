import scipy.io
import torch
from torchvision import datasets, transforms
import random
import numpy as np
import time
import os
import argparse
from datetime import datetime
import yaml
import json
from  re_ranking import re_ranking
from log import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',default='../dataset/match/pytorch',type=str, help='./test_data')
parser.add_argument('--model_dir', default='./model/pcb_rpp', type=str, help='save model path')
parser.add_argument('--log_dir', default='./logs/test', type=str, help='log dir')
parser.add_argument('--result_dir', default='./result/pcb_rpp', type=str, help='save result dir')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
opt = parser.parse_args()
config_path = os.path.join(opt.model_dir,'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
opt.PCB = config['PCB']

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

data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


result = scipy.io.loadmat(os.path.join(opt.result_dir, 'feature.mat'))
query_feature = result['query_f']
gallery_feature = result['gallery_f']

#re-ranking
print('calculate initial distance')
q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
q_q_dist = np.dot(query_feature, np.transpose(query_feature))
g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
since = time.time()
reranked_dist_m = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=7, k2=3, lambda_value=0.85)
time_elapsed = time.time() - since
print('Reranking complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(opt.data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(opt.data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}
res = {}
for index_query in range(len(reranked_dist_m)):
    index_gallery = np.argsort(reranked_dist_m[index_query])  # from small to large
    # index_gallery = index_gallery[::-1]
    query_path, _ = image_datasets['query'].imgs[index_query]
    query_name = query_path.split('/')[-1]
    responses = []

    for i in range(200):
        img_path, _ = image_datasets['gallery'].imgs[index_gallery[i]]
        img_name = img_path.split('/')[-1]
        responses.append(img_name)

    res[query_name] = responses

if not os.path.isdir(opt.result_dir):
    os.mkdir(opt.result_dir)
with open(os.path.join(opt.result_dir, 'result_rerank.json'), 'w') as f:
    json.dump(res, f)