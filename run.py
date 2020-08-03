import sys
import random
import torch.backends.cudnn as cudnn
from torch.optim import Adam
import torch.utils.data
import torchvision.transforms as transforms
import argparse
import model_big
from pre_data import BuildDataset
from utils import train, val
from cls_hrnet import *

sys.setrecursionlimit(15000)

parser = argparse.ArgumentParser()
parser.add_argument('--txtPath', default='databases/faceforensicspp', help='path to txt file')
parser.add_argument('--train_set', default='Deepfakes', help='train set')
parser.add_argument('--val_set', default='Deepfakes', help='validation set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=32, help='batch size')
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='checkpoints/binary_faceforensicspp', help='folder to output model checkpoints')
parser.add_argument('--disable_random', action='store_true', default=False,
                    help='disable randomness for routing matrix')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout percentage')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

opt.random = not opt.disable_random

if __name__ == "__main__":

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.gpu_id >= 0:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    # 如果为resume模式，则从上次训练的结果开始继续，否则新建一个训练文件
    if opt.resume > 0:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'a')
    else:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'w')

    # ext = get_cls_net("cls_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml","hrnet_w18_small_model_v2.pth")
    capnet = model_big.CapsuleNet(2, opt.gpu_id)
    capsule_loss = model_big.CapsuleLoss(opt.gpu_id)

    optimizer = Adam(capnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # 读取上次训练的结果
    if opt.resume > 0:
        capnet.load_state_dict(torch.load(os.path.join(opt.outf, 'capsule_' + str(opt.resume) + '.pt')))
        capnet.train(mode=True)
        optimizer.load_state_dict(torch.load(os.path.join(opt.outf, 'optim_' + str(opt.resume) + '.pt')))

        if opt.gpu_id >= 0:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(opt.gpu_id)

    # 网络加载到GPU
    if opt.gpu_id >= 0:
        capnet.cuda(opt.gpu_id)
        # ext.cuda(opt.gpu_id)
        capsule_loss.cuda(opt.gpu_id)

    # 缩放图片大小并转换为Tensor格式
    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_train = BuildDataset('/HDD2/yjc/capsule+HRNet/', opt.train_set, transform_fwd)
    assert dataset_train
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=False,
                                                   num_workers=int(opt.workers))

    dataset_val = BuildDataset('/HDD2/yjc/capsule+HRNet/', opt.val_set , transform_fwd)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize, shuffle=False,
                                                 num_workers=int(opt.workers))

    for epoch in range(opt.resume + 1, opt.niter + 1):

        acc_train, loss_train = train(dataloader_train,capnet,optimizer,capsule_loss,opt.gpu_id,opt.random,opt.dropout)

        # 保存训练状态
        torch.save(capnet.state_dict(), os.path.join(opt.outf, 'capsule_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optim_%d.pt' % epoch))

        acc_test, loss_test = val(dataloader_val, capnet, capsule_loss, opt.gpu_id)

        print('[Epoch %d] Train loss: %.4f   acc: %.2f | Test loss: %.4f  acc: %.2f'
              % (epoch, loss_train, acc_train * 100, loss_test, acc_test * 100))

        text_writer.write('%d,%.4f,%.2f,%.4f,%.2f\n'
                          % (epoch, loss_train, acc_train * 100, loss_test, acc_test * 100))

        text_writer.flush()
        capnet.train(mode=True)

    text_writer.close()
