import os
from shutil import copyfile
import numpy as np


#---------------------------------------
#train_all
def prepare_train_all():
    train_path = download_path + '/train_set/train'
    train_save_path = download_path + '/pytorch/train_all'
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)

    train_list_file = download_path+ "/train_set/train_list.txt"
    fd = open(train_list_file)
    line = fd.readline()
    while line:
        ID = line.split(' ')[-1][:-1]
        src_path = train_path + '/' + line.split('/')[-1].split(' ')[0]
        dst_path = train_save_path + '/' + ID
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + line.split('/')[-1].split(' ')[0])

        line = fd.readline()

#---------------------------------------
#train_val
def prepare_train_val():
    train_path = download_path + '/train_set/train'
    train_save_path = download_path + '/pytorch/train'
    val_save_path = download_path + '/pytorch/val'
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
        os.mkdir(val_save_path)

    train_list_file = download_path+ "/train_set/train_list.txt"
    fd = open(train_list_file)
    line = fd.readline()
    while line:
        ID = line.split(' ')[-1][:-1]
        src_path = train_path + '/' + line.split('/')[-1].split(' ')[0]
        dst_path = train_save_path + '/' + ID
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = val_save_path + '/' + ID  # first image is used as val image
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + line.split('/')[-1].split(' ')[0])

        line = fd.readline()

#-----------------------------------------
#query
def prepare_query():
    query_path = download_path + '/test_set/query_a'
    query_save_path = download_path + '/pytorch/query'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    query_list_file = download_path+ "/test_set/query_a_list.txt"
    fd = open(query_list_file)
    line = fd.readline()
    while line:
        ID = line.split(' ')[-1][:-1]
        src_path = query_path + '/' + line.split('/')[-1].split(' ')[0]
        dst_path = query_save_path + '/' + ID
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + line.split('/')[-1].split(' ')[0])

        line = fd.readline()

#-----------------------------------------
#gallery
def prepare_gallery():
    gallery_path = download_path + '/test_set/gallery_a'
    gallery_save_path = download_path + '/pytorch/gallery'
    if not os.path.isdir(gallery_save_path):
        os.mkdir(gallery_save_path)

    for root, dirs, files in os.walk(gallery_path, topdown=True):
        for name in files:
            src_path = gallery_path + '/' + name
            dst_path = gallery_save_path
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)


if __name__ == "__main__":
    # You only need to change this line to your dataset download path
    download_path = '../dataset/match'

    if not os.path.isdir(download_path):
        print('please change the download_path')
        exit(1)

    save_path = download_path + '/pytorch'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    seed = 0
    np.random.seed(seed)