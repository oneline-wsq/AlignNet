import os
os.environ["CUDA_VISIBLE_DEVICES"]="4,7"
import numpy as np
import sys
import logging
from time import time
from tensorboardX import SummaryWriter
import argparse

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from loss import SimpleLoss,DiscriminativeLoss
from data.dataset_2 import semantic_dataset
from data.const import NUM_CLASSES
from model import get_model
from evaluation.iou import get_batch_iou
from evaluate import onehot_encoding
from torchvision.utils import save_image


def write_log(writer, ious, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)

    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)


def train(args):
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "results.log"),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))


    
    train_loader = semantic_dataset(args.bsz, args.nworkers,args.laneInput_dir, \
                                    args.laneGT_dir, args.egopose_dir, \
                                    args.resolution,args.patch_h,args.patch_w)
    
    
    model = get_model(args.model, args.inC, args.outC, args.instance_seg, args.embedding_dim, args.pretrained,args.UNetBackbone)

    # gpus=[0,1] # 用两张卡训练
    model.cuda()
    # model = nn.DataParallel(model,device_ids=gpus, output_device=gpus[0])


    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = StepLR(opt, 10, 0.1)
    writer = SummaryWriter(logdir=args.logdir)

    loss_fn = SimpleLoss(args.pos_weight).cuda()
    embedded_loss_fn = DiscriminativeLoss(args.embedding_dim, args.delta_v, args.delta_d).cuda()

    model.train()
    counter = 0
    last_idx = len(train_loader) - 1
    for epoch in range(args.nepochs):
        for batchi, (input_mask, semantic_gt, instance_gt,token ,egopose) in enumerate(train_loader):
            t0 = time()
            opt.zero_grad()

            # 保存真值图片
            # for idx in range(len(token)):
            #     if token[idx]+'.jpg' not in os.listdir('imgs/train_gt'):
            #         save_image(semantic_gt[idx][-3:].float(),f'imgs/train_gt/{token[idx]}.jpg')



            semantic, embedding = model(input_mask.cuda())
            # semantic.shape: [4,4,1000,1000] , 第一个4为batchsize
            
            semantic_gt = semantic_gt.cuda().float() # [4,4,200,400], 第一个4为batch size, 第二个4为4个类别（含背景）
            instance_gt = instance_gt.cuda()
            
            seg_loss = loss_fn(semantic, semantic_gt) 
            
            if args.instance_seg:
                var_loss, dist_loss, reg_loss= embedded_loss_fn(embedding,instance_gt)
            else:
                var_loss=0
                dist_loss=0
                reg_loss=0
            
            final_loss = seg_loss * args.scale_seg + var_loss * args.scale_var + dist_loss * args.scale_dist
            final_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                intersects, union = get_batch_iou(onehot_encoding(semantic),semantic_gt)
                iou = intersects / (union + 1e-7)
                # save_image(onehot_encoding(semantic)[0][-3:],f'imgs/train_middle/{token[0]}.jpg')
                logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}]    "
                            f"Time: {t1-t0:>7.4f}    "
                            f"Loss: {final_loss.item():>7.4f}    "
                            f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")
                # 这里打印了除背景外的所有类别的iou

                writer.add_scalar('train/step_time', t1 - t0, counter)
                writer.add_scalar('train/seg_loss', seg_loss, counter)
                writer.add_scalar('train/var_loss', var_loss, counter)
                writer.add_scalar('train/dist_loss', dist_loss, counter)
                writer.add_scalar('train/reg_loss', reg_loss, counter)
                writer.add_scalar('train/final_loss', final_loss, counter)


        model_name = os.path.join(args.logdir, f"model{epoch}.pt")
        torch.save(model.state_dict(), model_name)
        logger.info(f"{model_name} saved")
        model.train()

        sched.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi frame alignment training.')
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs5')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='dataset')
    parser.add_argument('--laneInput_dir', type=str, default='dataset/pred_multiframes_box')
    parser.add_argument('--laneGT_dir', type=str, default='dataset/gt_multiframes_box')
    parser.add_argument('--egopose_dir', type=str, default='dataset/egopose.txt')
    # parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='BevEncode') 
    parser.add_argument("--pretrained", type=bool, default=True) 
    parser.add_argument("--UNetBackbone", type=str, default='vgg') 
    
    # training config
    parser.add_argument("--nepochs", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--resolution", type=float,default=0.05)
    parser.add_argument("--patch_h", type=float,default=50)
    parser.add_argument("--patch_w", type=float,default=50)
    parser.add_argument("--inC", type=int,default=3)
    parser.add_argument("--outC", type=int,default=4) # 包含背景类
    
    # embedding config
    parser.add_argument('--instance_seg', action='store_true', default=True)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)
    
    # loss config
    parser.add_argument("--scale_seg", type=float, default=0.5)
    parser.add_argument("--scale_var", type=float, default=0.5)
    parser.add_argument("--scale_dist", type=float, default=0.5)
    parser.add_argument("--scale_direction", type=float, default=0.2)

    args = parser.parse_args()
    train(args)
