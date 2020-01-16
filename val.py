import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval
import numpy as np
assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_val', help='Path to COCO directory',default='../data/annotations_val_a.csv')
    parser.add_argument('--model_path', help='Path to model', type=str,default='./model_save')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)',default='../data/class_a.csv')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)

    parser = parser.parse_args(args)

    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=False)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_val.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    # retinanet.load_state_dict(torch.load(parser.model_path))


    retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    AP = csv_eval.evaluate(dataset_val, retinanet,iou_threshold=0.5,score_threshold=0.05,max_detections=5)
    mAP = np.array([AP[i][0] for i in range(len(AP))])
    print('mAP',mAP.mean())


if __name__ == '__main__':
    main()
