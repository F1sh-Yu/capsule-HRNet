from sklearn import metrics
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np


def train(dataloader_train, net, optimizer, loss, gpu_id, random, dropout):
    count = 0
    loss_train = 0

    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)

    for img_data, labels_data in tqdm(dataloader_train):

        # 标签转换为array格式
        img_label = labels_data.numpy().astype(np.float)
        # 梯度清空
        optimizer.zero_grad()

        if gpu_id >= 0:
            img_data = img_data.cuda(gpu_id)
            labels_data = labels_data.cuda(gpu_id)

        # 运行网络得到结果
        input_v = Variable(img_data)
        # x = vgg_ext(input_v)
        classes, class_ = net(input_v, random=random, dropout=dropout)

        # 得到loss值
        loss_dis = loss(classes, Variable(labels_data, requires_grad=False))
        loss_dis_data = loss_dis.item()

        # 反向传播 优化
        loss_dis.backward()
        optimizer.step()

        # 网络输出转化为1/0的预测值
        output_dis = class_.data.cpu().numpy()
        output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

        for i in range(output_dis.shape[0]):
            if output_dis[i, 1] >= output_dis[i, 0]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        # 保存预测值
        tol_label = np.concatenate((tol_label, img_label))
        tol_pred = np.concatenate((tol_pred, output_pred))

        loss_train += loss_dis_data
        count += 1
        # if count == 2000: break

    acc_train = metrics.accuracy_score(tol_label, tol_pred)
    loss_train /= count

    return acc_train, loss_train


def val(dataloader_val, net, loss, gpu_id):
    loss_test = 0

    net.eval()

    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)

    count = 0

    for img_data, labels_data in dataloader_val:

        img_label = labels_data.numpy().astype(np.float)

        if gpu_id >= 0:
            img_data = img_data.cuda(gpu_id)
            labels_data = labels_data.cuda(gpu_id)

        input_v = Variable(img_data)

        # x = vgg_ext(input_v)
        classes, class_ = net(input_v, random=False)

        loss_dis = loss(classes, Variable(labels_data, requires_grad=False))
        loss_dis_data = loss_dis.item()
        output_dis = class_.data.cpu().numpy()

        output_pred = np.zeros((output_dis.shape[0]), dtype=np.float)

        for i in range(output_dis.shape[0]):
            if output_dis[i, 1] >= output_dis[i, 0]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        tol_label = np.concatenate((tol_label, img_label))
        tol_pred = np.concatenate((tol_pred, output_pred))

        loss_test += loss_dis_data
        count += 1
        # if count == 600:break

    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    loss_test /= count

    return acc_test, loss_test
