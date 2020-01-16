import argparse
import collections
import os
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler
from retinanet import model
from retinanet.dataloader import  CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

Batch_Size = 16
Num_Works = 4
LR = 3e-4
val_epoch_num = 1
Model_Save_Path= '../model_save/baseline_res50/'

print('CUDA available: {}'.format(torch.cuda.is_available()))
if not os.path.exists(Model_Save_Path):
    os.makedirs(Model_Save_Path)

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.',default='csv')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)',default='../data/annotations_train_a.csv')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)',default='../data/class_a.csv')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)',default='../data/annotations_val_a.csv')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=300)

    parser = parser.parse_args(args)

    # Create the data loaders

    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=Batch_Size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=Num_Works, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=False)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = torch.cuda.is_available()

    ## 载入之前训练的模型，恢复参数继续训练
    trained_model_state_dict = torch.load('../model_save/a/baseline_res50_retinanet_13.pth')
    trained_model_state_dict = {k.replace('module.',''):v for k,v in trained_model_state_dict.items()}
    retinanet.load_state_dict(trained_model_state_dict)
    del trained_model_state_dict

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=LR)

    after_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3,patience=5, verbose=True)
    scheduler = GradualWarmupScheduler(optimizer,multiplier=5,total_epoch=8,after_scheduler= after_scheduler)
    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            # print('image_shape',data['img'].shape)
            try:
                optimizer.zero_grad()

                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))
                if epoch_num < 2:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
            # print(type(np.mean(epoch_loss)))
        print('Evaluating dataset')
        if epoch_num % val_epoch_num == 0:
            AP = csv_eval.evaluate(dataset_val, retinanet)
            mAP = np.array([AP[i][0] for i in range(len(AP))])
            print('epoch-----------',epoch_num,'---Val_mAP---',mAP.mean())


        ## 在训练数据集上进行验证，验证的次数为val上的1/4
        if epoch_num % (val_epoch_num*4) == 0:
            AP = csv_eval.evaluate(dataset_train, retinanet)
            mAP = np.array([AP[i][0] for i in range(len(AP))])
            print('epoch-----------',epoch_num,'---Train_mAP---',mAP.mean())

        scheduler.step(metrics=np.mean(epoch_loss))
        torch.save(retinanet.state_dict(), '{}_retinanet_{}.pth'.format(Model_Save_Path, epoch_num))

if __name__ == '__main__':
    main()
