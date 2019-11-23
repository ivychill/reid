import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
'''
class ClassBlock(nn.Module):
    # def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
    def __init__(self, input_dim, class_num, droprate=0, relu=False, bnorm=True, num_bottleneck=512, linear=True):
        super(ClassBlock, self).__init__()
        add_block = []

        # add_block += [nn.Conv2d(input_dim, num_bottleneck, kernel_size=1, bias=False)]
        # add_block += [nn.BatchNorm2d(num_bottleneck)]
        # if relu:
        #     #add_block += [nn.LeakyReLU(0.1)]
        #     add_block += [nn.ReLU(inplace=True)]
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x
'''

class ClassBlock(nn.Module):
    # def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
    def __init__(self, input_dim, class_num, droprate=0, relu=False, bnorm=True, num_bottleneck=512, linear=True):
        super(ClassBlock, self).__init__()

        self.bottleneck = nn.BatchNorm1d(input_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(input_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):

        feat = self.bottleneck(x)
        cls_score = self.classifier(feat)

        return cls_score






# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(1024, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the NAS-based Model
class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super().__init__()
        model_name = 'nasnetalarge'
        # pip install pretrainedmodels
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 4032
        self.classifier = ClassBlock(4032, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the RPP layers
class RPP(nn.Module):
    def __init__(self):
        super(RPP, self).__init__()
        self.part = 6
        add_block = []
        add_block += [nn.Conv2d(1024, 6, kernel_size=1, bias=False)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        norm_block = []
        norm_block += [nn.BatchNorm2d(1024)]
        norm_block += [nn.ReLU(inplace=True)]
        # norm_block += [nn.LeakyReLU(0.1, inplace=True)]
        norm_block = nn.Sequential(*norm_block)
        norm_block.apply(weights_init_kaiming)

        self.add_block = add_block
        self.norm_block = norm_block
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        w = self.add_block(x)
        p = self.softmax(w)
        y = []
        for i in range(self.part):
            p_i = p[:, i, :, :]
            p_i = torch.unsqueeze(p_i, 1)
            y_i = torch.mul(x, p_i)
            y_i = self.norm_block(y_i)
            y_i = self.avgpool(y_i)
            y.append(y_i)

        f = torch.cat(y, 2)
        return f


# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()

        self.part = 6
        # resnet50
        resnet = models.resnet50(pretrained=True)
        # remove the final downsample
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)

        # define 6 classifiers
        self.classifiers = nn.ModuleList()
        for i in range(self.part):
            self.classifiers.append(ClassBlock(2048, class_num, True, 256))

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x[:, :, i, :]
            part[i] = torch.unsqueeze(part[i], 3)
            # print part[i].shape
            predict[i] = self.classifiers[i](part[i])

        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y

    def convert_to_rpp(self):
        self.avgpool = RPP()
        return self


class PCB_test(nn.Module):
    def __init__(self, model, featrue_H=False):
        super(PCB_test, self).__init__()
        self.part = 6
        self.featrue_H = featrue_H
        self.backbone = model.backbone
        self.avgpool = model.avgpool
        self.classifiers = nn.ModuleList()
        for i in range(self.part):
            self.classifiers.append(model.classifiers[i].add_block)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)

        if self.featrue_H:
            part = {}
            predict = {}
            # get six part feature batchsize*2048*6
            for i in range(self.part):
                part[i] = x[:, :, i, :]
                part[i] = torch.unsqueeze(part[i], 3)
                predict[i] = self.classifiers[i](part[i])

            y = []
            for i in range(self.part):
                y.append(predict[i])
            x = torch.cat(y, 2)
        f = x.view(x.size(0), x.size(1), x.size(2))
        return f

class PCB_dense(nn.Module):
    def __init__(self, class_num):
        super(PCB_dense, self).__init__()

        self.part = 6  # We cut the pool5 to 6 parts
        model_ft = models.densenet121(pretrained=True)
        # TODO: stride 1
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(1024*self.part, 128)
        # define 6 classifiers
        self.classifiers = nn.ModuleList()
        for i in range(self.part):
            self.classifiers.append(ClassBlock(1024, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.features(x)
        x = self.avgpool(x)
        x = self.dropout(x)     # torch.Size([32, 1024, 6, 1]),
        feats = x.view(x.size(0), x.size(1), x.size(2))  # torch.Size([256, 1024, 6]), [batch_size, fc, part]
        z = feats.view(feats.size(0), -1)   # torch.Size([64, 6144])
        # bn_feats = self.linear(z)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, i])     # torch.Size([32, 1024])
            # print part[i].shape
            predict[i] = self.classifiers[i](part[i])   # torch.Size([32, 4768]), [batch_size, class_num]

        y = []
        for i in range(self.part):
            y.append(predict[i])

        return y, z
        # return y, feats, bn_feats

    def convert_to_rpp(self):
        self.avgpool = RPP()
        return self

class PCB_dense_test(nn.Module):
    def __init__(self,model):
        super(PCB_dense_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))

    def forward(self, x):
        x = self.model.features(x)
        x = self.avgpool(x)     # torch.Size([256, 1024, 6, 1])
        y = x.view(x.size(0),x.size(1),x.size(2))   # torch.Size([256, 1024, 6]), [batch_size, fc, part]
        return y

# debug model structure
if __name__ == '__main__':
    net = PCB(751)
    net = net.convert_to_rpp()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 7, 7))
    output = net(input)
    # print(output[0].shape)
