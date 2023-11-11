import torch.utils.data as data
import os
import torch
import cv2

class Dataset_2023(data.Dataset):
    def __init__(self, aug_params=None, root='dataset/train/animal', mode='train', list_categories=[]):
        self.root = root.replace('train', mode)
        self.list_categories = list_categories
        self.aug_params = aug_params
        self.data_list = []
        for category in list_categories:
            img_list = os.listdir(os.path.join(self.root, category))
            for img_file in img_list:
                self.data_list.append((category, os.path.join(category, img_file)))

    def __getitem__(self, index):
        category, img_file = self.data_list[index]

        category_id = self.list_categories.index(category)
        category_tensor = torch.zeros(len(self.list_categories))
        category_tensor[category_id] = 1
        
        img = cv2.imread(os.path.join(self.root, img_file))
        img = cv2.resize(img, (256, 256))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        img_tensor = (2 * (img_tensor / 255.0) - 1.0).contiguous()

        return category_id, category_id, category_tensor, img_tensor
    
    def __len__(self):
        return len(self.data_list)
