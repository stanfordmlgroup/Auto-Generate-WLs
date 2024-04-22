import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

import archs
from dataset import Dataset
from metrics import iou_score, dice_coef
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--dataset', default=None,
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', args.dataset, 'images', '*' + args.img_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    val_img_ids = img_ids
    
    mask_ids = glob(os.path.join('inputs', args.dataset, 'masks/0', '*' + args.mask_ext))
    mask_ids = [os.path.splitext(os.path.basename(p))[0] for p in mask_ids]
    
    not_in_masks = list(set(img_ids) - set(mask_ids))
    
    for not_mask in not_in_masks:
        img_path = os.path.join('inputs', args.dataset, 'images', not_mask + args.img_ext)
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        mask = 0 * np.ones((height, width), dtype=np.uint8)
        mask_path = os.path.join('inputs', args.dataset, 'masks/0/', not_mask + args.mask_ext)
        cv2.imwrite(mask_path, mask)
        


    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', args.dataset, 'images'),
        mask_dir=os.path.join('inputs', args.dataset, 'masks'),
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()
    avg_meter_dice = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], args.dataset, str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            avg_meter.update(iou, input.size(0))
            avg_meter_dice.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            
            

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], args.dataset, str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)
    print('DICE: %.4f' % avg_meter_dice.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
