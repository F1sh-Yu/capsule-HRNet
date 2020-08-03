from torch.utils.data import Dataset,BatchSampler
from PIL import Image
import numpy as np 
# from scipy import ndimage
import os

kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])


class BuildDataset(Dataset):
    def __init__(self, txt_path, datatxt, transform=None):
        super(BuildDataset, self).__init__()
        fh = open(os.path.join(txt_path,datatxt), 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# class BuildDatasetSRM(Dataset):
#     def __init__(self, datatxt, txt_path, transform=None):
#         super(BuildDatasetSRM, self).__init__()
#         fh = open(os.path.join(txt_path,datatxt), 'r')
#         imgs = []
#         for line in fh:
#             line = line.rstrip()
#             words = line.split()
#             imgs.append((words[0], int(words[1])))
#         self.imgs = imgs
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.imgs)
#
#     def __getitem__(self, index):
#         fn, label = self.imgs[index]
#         img = Image.open(fn).convert("RGB")
#         img = np.array(img)
#         r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
#         r1 = ndimage.convolve(r, kernel)
#         g1 = ndimage.convolve(g, kernel)
#         b1 = ndimage.convolve(b, kernel)
#         img1 = np.dstack((r1,g1,b1))
#         img2 = Image.fromarray(img1)
#         if self.transform:
#             img2 = self.transform(img2)
#         return img2, label
