import os
import time
import math
from tqdm import *
import random
import torch
import torch.nn as nn
import numpy as np
import torchvision
import warnings
warnings.filterwarnings("ignore")

from dataset import *

def categoryFromOutput(output):
    guess = torch.argmax(output, dim=1)
    return guess.item()

def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(net, device, list_categories=[]):
    with torch.no_grad():

        net = net.to(device)
        net.eval()

        n_categories = len(list_categories)
        category_right = [0 for _ in range(n_categories)]
        category_num = [0 for _ in range(n_categories)]
        category_acc = [0 for _ in range(n_categories)]

        test_dataset = Dataset_2023(mode='test', list_categories=list_categories)
        test_loader = data.DataLoader(test_dataset, batch_size=1, pin_memory=True, shuffle=True, num_workers=8, drop_last=True)

        t0 = time.time()
        for i_batch, (_, *data_blob) in enumerate(tqdm(test_loader)):
            
            category, _, img_tensor = [x.to(device) for x in data_blob]

            output = net(img_tensor)

            guess = categoryFromOutput(output)

            if guess == category:
                category_right[int(category)] += 1
            category_num[int(category)] += 1
        t1 = time.time()

        num_all = 0
        num_right = 0
        for i in range(n_categories):
            category_acc[i] = category_right[i] / category_num[i]
            num_all += category_num[i]
            num_right += category_right[i]
            print('    {} acc: {}/{} = {}%'.format(list_categories[i], category_right[i], category_num[i], round(100*category_acc[i], 2)))
        print('  all acc: {}/{} = {}%'.format(num_right, num_all, round(100*num_right/num_all, 2)))
        print('  FPS: {}'.format(len(test_loader)/(t1-t0)))

if __name__ == '__main__':

    get_random_seed(23)

    # cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {}".format(device))

    list_categories = ['bird', 'cat', 'cow', 'dog', 'elephant', 'giraffe', 'horse', 'sheep', 'zebra']
    n_categories = len(list_categories)

    # --------------------------------------------------------------可修改区域起始处-----------------------------------------------------------------------#
    # 此处可自行修改，加载你的模型(命名为net), 你也可以在model.py中定义你的模型并在此处调用（如果需要）
    from model_submit import *
    net = torch.load('./epoch_20.pth')

    # --------------------------------------------------------------可修改区域截止处-------------------------------------------------------------------------#

    test(net, device, list_categories = list_categories)