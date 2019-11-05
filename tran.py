#coding:utf8
import os;

# download_path = '/opt/Preid/train'
file = open("traindata.txt")
# line = file.readline()
# last_ID = line.split(' ')[-1].split('\n')[0]
count = 0
allID = []
while 1:
    line = file.readline()
    ID = line.split(':')[-1]

    if not line:
        break
    id = int(ID)
    allID.append(id)
filename = 'data.txt'
with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        # f.write("I am now studying in NJTECH.\n")
    for i in range(0, 10000):
      a_count = allID.count(i)
      if a_count >0:
        str = "%d:%d\n" % (i, a_count)
        f.write(str)
'''
def rename():
        i=0
        path="/opt/Preid/train/pytorch/train_all"
        filelist=os.listdir(path)#该文件夹下所有的文件（包括文件夹）
        for dirs in filelist:
           # file_length =len(os.listdir(path + '/' + dirs))
           for files in os.listdir(path+'/'+dirs):#遍历所有文件
              s = str(dirs) +'_'
              Olddir=os.path.join(path,dirs,files);#原来的文件路径                
              if os.path.isdir(Olddir):#如果是文件夹则跳过
                    continue;
              filename=os.path.splitext(files)[0];#文件名
              filetype=os.path.splitext(files)[1];#文件扩展名
              Newdir=os.path.join(path,dirs,s+filename+filetype);#新的文件路径
              os.rename(Olddir,Newdir)#重命名
# rename()
download_path = '/opt/Preid/train'
file = open(download_path+ "/train_list.txt")
line = file.readline()
# last_ID = line.split(' ')[-1].split('\n')[0]
count = 0
allID = []
while 1:
    line = file.readline()
    ID = line.split(' ')[-1].split('\n')[0]
    id = int(ID)
    allID.append(id)
    count = count+1
    if count == 20428:
        # ID = line.split(' ')[-1].split('\n')[0]
        break

    # Olddir = download_path + '/train' + '/' + line.split('/')[-1].split(' ')[0]
    # Newdir = download_path + '/train' +'/' + ID + '_' + line.split('/')[-1].split(' ')[0]
    # os.rename(Olddir, Newdir)

    # if not line:
    #     break

filename = 'traindata.txt'
with open(filename,'w') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    # f.write("I am now studying in NJTECH.\n")
    for i in range(0,4768):
        a_count = allID.count(i)
        str = "%d:%d\n"%(i,a_count)
        f.write(str)
        # print(a_count)
'''