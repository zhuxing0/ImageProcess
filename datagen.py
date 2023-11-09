'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop

import xml.etree.ElementTree as ET
import pdb

class ListDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size
        self.root_annotations = root.replace('JPEGImages', 'Annotations')

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()

        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            fname = line.strip()
            self.fnames.append(fname)

            box = []
            label = []
            xml_path = os.path.join(self.root_annotations, fname+'.xml')
            tree = ET.parse(xml_path)
            childs = tree.getroot()
            elements = childs.findall('object')
            for element in elements:
                xmin = element.find('bndbox').find('xmin').text
                ymin = element.find('bndbox').find('ymin').text
                xmax = element.find('bndbox').find('xmax').text
                ymax = element.find('bndbox').find('ymax').text
                c = 0
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname+'.jpg'))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size,size))

        img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


def test():
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    dataset = ListDataset(root='/C/Users/26470/Desktop/2023数图课件/VOC2007/JPEGImages',
                          list_file='./data/val.txt', train=False, transform=transform, input_size=600)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for images, loc_targets, cls_targets in dataloader:
        print(images.size())
        print(loc_targets.size())
        print(cls_targets.size())
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'a.jpg')
        break

# test()
