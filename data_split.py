import numpy as np
import os,sys
import shutil


def writeDataToDir(data_dic,img_source_dir,img_tar_dir,type_):
    dataList = dataDicToDdataList(data_dic)
    count = 0
    os.makedirs(img_tar_dir,exist_ok=True)
    save_txt = os.path.join(img_tar_dir,type_+'.txt')
    with open(save_txt,'w') as f:
        if os.path.exists(os.path.join(img_tar_dir,type_)):
            shutil.rmtree(os.path.join(img_tar_dir,type_))
        for data in dataList:
            img_pth_r, ind = data
            _, img_name = os.path.split(img_pth_r)
            msg = "{}/{} {}\n".format(type_,img_name,ind)
            f.write(msg)
            count +=1
            img_pth_source = os.path.join(img_source_dir,img_pth_r)
            img_pth_dir = os.path.join(img_tar_dir,type_,str(ind))
            os.makedirs(img_pth_dir,exist_ok=True)
            shutil.copy(img_pth_source,os.path.join(img_pth_dir,img_name))

    print('All items number:',count)
    print('finish write:',save_txt)

def getDataFromTxt(train_all_txt):
    # 从txt中读取数据，放到dic之中
    data_all = {}
    with open(train_all_txt, 'r') as f:
        data_raw = f.readlines()
        for line_i in data_raw:
            img_name, pid = line_i.split(' ')
            if int(pid) in data_all:
                data_all[int(pid)].append(img_name)
            else:
                data_all[int(pid)] = [img_name]
    print('All data len:',len(data_raw))
    printDataDicInfo(data_all)
    return data_all

def printDataDicInfo(data_all):
    info = {}
    for key in data_all.keys():
        data_len = len(data_all[key])
        if data_len in info:
            info[data_len] += 1
        else:
            info[data_len] = 1
    print('######### data info ##############')
    keys = list(info.keys())
    keys.sort()
    for key in keys:
        print('{} persons has {} pics'.format(info[key], key))


def splitQueryDataGallayrData(data_all_dic,skip=1,max_len=300,ratios=0.15,remain_skip=False):
    # 从包含所有数据的字典{pid:[img_pth1,img_pth2,..],pid2:[img_pth1,img_pth2..]}的数据中
    # 分离出，如果数据量小于skip则不抽取，否则至少抽取一个。
    gallary_data = []
    query_data = []
    remain_data = []
    all_inds = data_all_dic.keys()
    for ind in all_inds:
        paths = data_all_dic[ind]
        if len(paths) <= skip:
            # 不处理，直接丢掉
            if remain_skip:
                for img_pth in paths:
                    remain_data.append((img_pth, ind))
            continue
        if len(paths)>max_len:
            # 随机选择300个目标
            paths = np.random.choice(paths, int(max_len), replace=False)
        rand_num = 1 if int(len(paths)*ratios)<=1 else int(len(paths)*ratios)
        rand_inds = np.random.randint(0, len(paths), rand_num)
        for i, img_pth in enumerate(paths):
            if i in rand_inds:
                query_data.append((img_pth, ind))
            else:
                gallary_data.append((img_pth, ind))
    print('query data len:{},remain data len:{}'.format(len(query_data),len(remain_data)))
    return query_data,gallary_data,remain_data

def selectDataFromDataDic(inds,data_dic):
    data_select,data_last = {},{}
    for ind in data_dic.keys():
        if ind in inds:
            data_select[ind] = data_dic[ind]
        else:
            data_last[ind] = data_dic[ind]
    return data_select,data_last

def dataDicToDdataList(data_dic,skip=0):
    data_list = []
    for ind in data_dic.keys():
        paths = data_dic[ind]
        if len(paths) <= skip:
            continue
        else:
            for img_pth in paths:
                data_list.append((img_pth, ind))
    return data_list

def splitTrainValid(data_all_dic,ratio,skip=1,max_len=300):
    # 从包含所有数据的字典{pid:[img_pth1,img_pth2,..],pid2:[img_pth1,img_pth2..]}的数据中
    # 分离出，如果数据量小于skip则不抽取，否则至少抽取一个。
    train_data = {}
    valid_data = {}

    all_inds = data_all_dic.keys()
    data_inds = list(all_inds)
    valid_inds = np.random.choice(data_inds, int(len(data_inds) * ratio), replace=False)
    for ind in all_inds:
        paths = data_all_dic[ind]
        if len(paths) <= skip:
            # 不处理，直接丢掉
            train_data[ind] = data_all_dic[ind]
            continue
        if len(paths)>max_len:
            paths = np.random.choice(paths, int(max_len), replace=False)

        if ind in valid_inds:
            valid_data[ind] = paths
        else:
            train_data[ind] = paths
    printDataDicInfo(valid_data)
    printDataDicInfo(train_data)
    print('train data len:{},valid data len:{}'.format(len(train_data),len(valid_data)))
    return train_data,valid_data

def splitQueryGallayr(valid_dir,ratio,save_dir):
    # 从验证集文件夹中筛选部分数据做querry，gallery
    query_dir = os.path.join(save_dir, 'query')
    gallary_dir = os.path.join(save_dir, 'query')
    if os.path.exists(query_dir):
        shutil.rmtree(query_dir)
    if os.path.exists(gallary_dir):
        shutil.rmtree(gallary_dir)

    img_dirs = [os.path.join(valid_dir,d) for d in os.listdir(valid_dir)]
    for img_dir in img_dirs:
        img_pths = [os.path.join(img_dir,pth) for pth in os.listdir(img_dir) if pth.endswith(('jpg','png','JPG','PNG'))]
        if len(img_pths)<=1:
            print('error :',img_pths)
        rand_num = 1 if int(len(img_pths) * ratio) <= 1 else int(len(img_pths) * ratio)
        query_inds_sel = np.random.randint(0, len(img_pths), rand_num)
        for ind in range(len(img_pths)):
            if ind in query_inds_sel:
                type_ = 'query'
            else:
                type_ = 'gallary'
            path_info = img_pths[ind].split('/')
            pid_str ,img_name = path_info[-2:]
            os.makedirs(os.path.join(save_dir,type_,pid_str),exist_ok=True)
            img_tar_dir_ = os.path.join(save_dir, type_, pid_str, img_name)
            shutil.copy(img_pths[ind],img_tar_dir_)





if __name__ == "__main__":
    # 训练与验证分开
    # 先划分训练集与测试集，测试集中不能含有少于N个的目标，集合目标最多M个
    # 从验证集中抽出一部分目标分别做query与gallery
    # np.random.seed(4)
    np.random.seed(6)

    train_all_txt = '/usr/zll/person_reid/data/sz_reid/sz_reid_round1/train_list.txt' # 原始的数据txt
    img_source_dir = '/usr/zll/person_reid/data/sz_reid/sz_reid_round1' # 图片路径
    img_tar_dir = '/usr/zll/person_reid/data/sz_reid_xz2'  # 划分后图片存放路径


    ratio = 0.1 # 验证集比例
    data_all_dic = getDataFromTxt(train_all_txt)
    # skip表示低于多少张不会做验证集，max_len表示过多的图片，只选max_len张。
    train_data_dic, valid_data_dic = splitTrainValid(data_all_dic, ratio, skip=1, max_len=200)
    writeDataToDir(train_data_dic,img_source_dir,img_tar_dir,type_='train')
    writeDataToDir(valid_data_dic,img_source_dir,img_tar_dir,type_='valid')

    # 将验证集划分为query 和gallery
    valid_dir = '/usr/zll/person_reid/data/sz_reid_xz/valid'
    splitQueryGallayr(valid_dir, 0.2, img_tar_dir)
    print('finish!')




