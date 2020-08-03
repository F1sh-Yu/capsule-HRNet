import numpy as np
from model_big import VggExtractor
import torch
from cls_hrnet import *
import model_big
# x = np.zeros((1,3,224,224))
# x = torch.tensor(x,dtype=torch.float32)
# ext = get_cls_net()

capnet = model_big.CapsuleNet(2,-1)
# x = vgg_ext(x)
# x=capnet(x)
# print(len(x))
torch.load(map_location='cpu')
total_num = sum(p.numel() for p in capnet.parameters())
print(total_num)