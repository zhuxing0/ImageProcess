import os
import time
import math
from tqdm import *
import random
import numpy as np
import shutil
import cv2
import matplotlib.pyplot as plt
import pdb

import torch.optim as optim
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision

import warnings
warnings.filterwarnings("ignore")

from dataset import Dataset_2023

def categoryFromOutput(output):
    guess = torch.argmax(output, dim=1)
    return guess.item()

# 定义随机种子固定的函数
def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 全局训练函数
def train(name, net, lr, device, n_epoch = 100, list_categories=[]):
    if not os.path.exists(os.path.join('./checkpoints/', name)):
        os.makedirs(os.path.join('./checkpoints/', name))
    
    # 加载训练集和验证集
    train_dataset = Dataset_2023(mode='train', list_categories=list_categories)
    train_loader = data.DataLoader(train_dataset, batch_size=16, pin_memory=True, shuffle=True, num_workers=8, drop_last=True) # num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2
    test_dataset = Dataset_2023(mode='val', list_categories=list_categories)
    test_loader = data.DataLoader(test_dataset, batch_size=1, pin_memory=True, shuffle=True, num_workers=8, drop_last=True) # num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=.00001, eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, n_epoch*len(train_loader),
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    
    print("=========================开始{}训练===========================".format(name))

    # Keep track of losses for plotting
    n_categories = len(list_categories)
    all_losses = [[],[]]
    all_lr = []
    all_acc = [[] for _ in range(n_categories)]

    # 定义模型保存和验证集测试频率
    save_epoch = 10
    test_epoch = 1

    for e in range(n_epoch):
        print('epoch:{} training begin!'.format(e))
        train_loss = 0
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            category, category_tensor, img_tensor = [x.to(device) for x in data_blob]
            # print(category, category_tensor, img_tensor)

            output = net(img_tensor)  # 推理

            loss = criterion(output, category_tensor)  # 计算损失
            loss.backward() # 进行反向传播, 计算梯度
            optimizer.step() # 更新模型参数
            train_loss += loss

            all_lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
            scheduler.step()
        all_losses[0].append(train_loss/len(train_loader))
        print('epoch {} training finish! train loss: {}'.format(e, train_loss/len(train_loader)))

        # 保存模型
        if e % save_epoch == save_epoch - 1:
            save_path = os.path.join('./checkpoints/', name) + '/epoch_{}.pth'.format(e+1)
            torch.save(net, save_path)

        # 训练过程中测试
        if e % test_epoch == test_epoch - 1:
            print('epoch:{} testing begin!'.format(e))
            category_right = [0 for _ in range(n_categories)]
            category_num = [0 for _ in range(n_categories)]
            category_acc = [0 for _ in range(n_categories)]

            net.eval()
            test_loss = 0
            with torch.no_grad():
                for i_batch, (_, *data_blob) in enumerate(tqdm(test_loader)):
                    
                    category, category_tensor, img_tensor = [x.to(device) for x in data_blob]

                    output = net(img_tensor)  # 推理
                    loss = criterion(output, category_tensor)  # 计算损失
                    test_loss += loss
                    guess = categoryFromOutput(output)  # 根据输出概率匹配预测标签

                    # 此处可以自行补充，如计算混淆矩阵等。
                    if guess == category:
                        category_right[int(category)] += 1
                    category_num[int(category)] += 1
            
            print('  test loss: {}'.format(test_loss/len(test_loader)))
            all_losses[1].append(test_loss/len(test_loader))

            num_all = 0
            num_right = 0
            for i in range(n_categories):
                category_acc[i] = category_right[i] / category_num[i]
                num_all += category_num[i]
                num_right += category_right[i]
                all_acc[i].append(category_acc[i])
                print('    {} acc: {}/{} = {}%'.format(list_categories[i], category_right[i], category_num[i], round(100*category_acc[i], 2)))
            print('  all acc: {}/{} = {}%'.format(num_right, num_all, round(100*num_right/num_all, 2)))
            net.train()

    return all_losses, all_lr, all_acc

if __name__ == '__main__':

    # 调用函数，设置随机种子为23
    get_random_seed(23)

    # 选择使用 cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #  torch.device("cpu")
    print("Using {}".format(device))

    # 定义分类任务的种类
    list_categories = ["bird", "cat", 'cow', 'dog', 'elephant', 'giraffe', 'horse', 'sheep', 'zebra']
    n_categories = len(list_categories)

    # 创建Resnet50模型，修改模型的分类头
    Resnet50 = torchvision.models.resnet50(pretrained=False)
    num_features=Resnet50.fc.in_features
    Resnet50.fc=nn.Sequential(nn.Linear(num_features, n_categories),
                            nn.LogSoftmax(dim=1))
    Resnet50 = Resnet50.to(device)

    # 记录每个模型的损失值列表
    nets_loss = []
    nets_lr = []
    nets_acc = []

    # 模型名称、模型,、学习率、训练代数(列表)
    nets_name_list = ['ResNet-50']
    nets_list = [Resnet50]
    lr_list = [0.01]
    epoch_list = [20]

    # 训练模型
    for i in range(len(nets_list)):
        net_loss, net_lr, net_acc = train(nets_name_list[i], nets_list[i], lr_list[i], device, n_epoch = epoch_list[i], list_categories = list_categories)
        nets_loss.append(net_loss)
        nets_lr.append(net_lr)
        nets_acc.append(net_acc)

    if not os.path.exists('./result/'):
        os.makedirs('./result/')
    for idx, name in enumerate(nets_name_list):
        # 绘制损失值折线图
        plt.figure()
        for i in range(len(nets_loss[idx])):
            plot_list = [_.cpu().detach().numpy() for _ in nets_loss[idx][i]]
            plt.plot(plot_list)
        plt.legend(['train_loss', 'val_loss'])
        plt.savefig('./result/' + name + '_lr_'+str(lr_list[idx]) + '_epoch_'+str(epoch_list[idx]) + '_loss.jpg')
        # 绘制准确率折线图
        plt.figure()
        for i in range(len(nets_acc[idx])):
            plt.plot(nets_acc[idx][i])
        plt.legend(list_categories)
        plt.savefig('./result/' + name + '_lr_'+str(lr_list[idx]) + '_epoch_'+str(epoch_list[idx]) + '_acc.jpg')
        # 绘制学习率折线图
        plt.figure()
        plt.plot(nets_lr[idx])
        plt.legend(name+'_lr')
        plt.savefig('./result/' + name + '_lr_'+str(lr_list[idx]) + '_epoch_'+str(epoch_list[idx]) + '_lr.jpg')