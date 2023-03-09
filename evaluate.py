import argparse
import tqdm

import torch
from torchvision.utils import save_image
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4,1"

from data.dataset_2 import semantic_dataset
from data.const import NUM_CLASSES
from model import get_model
import matplotlib.pyplot as plt
from evaluation.iou import get_batch_iou


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)  # dim=1代表通道
    # 相当于按照通道，找每个像素点对应的最大的那个通道数
    one_hot = logits.new_full(logits.shape, 0) # 新建一个one_hot, 形状为logits.shape
    one_hot.scatter_(dim, max_idx, 1) # 
    return one_hot

def eval(args):
    
    model = get_model(args.model, args.inC, args.outC, args.instance_seg, args.embedding_dim, args.pretrained,args.UNetBackbone)
    weights = torch.load(args.modelf)
    weights_dict={}
    for k,v in weights.items():
        new_k = k.replace('module.','') if 'module' in k else k
        weights_dict[new_k]=v
    model.load_state_dict(weights_dict,strict=False)
    model.cuda()
    model.eval()
    
    val_loader = semantic_dataset(args.bsz, args.nworkers,args.test_laneInput_dir, \
                                args.test_laneGT_dir, args.egopose_dir, \
                            args.resolution,args.patch_h,args.patch_w)

    n=0
    total_intersects=0
    total_union=0
    with torch.no_grad():
        for input_mask, semantic_gt, instance_gt, token, egopose in tqdm.tqdm(val_loader):
            semantic,_= model(input_mask.cuda(non_blocking=True)) 
            semantic_onehot=onehot_encoding(semantic)
            semantic_gt = semantic_gt.cuda(non_blocking=True).float()
            # 计算iou
            intersets, union =get_batch_iou(semantic_onehot,semantic_gt)
            total_intersects += intersets
            total_union += union
            print('end')
            # # 归一化到0, 1范围
            # # 保存图片
            # output=semantic[0].cpu()[:,:,1:] # 转成(1000,1000,3)
            save_image(semantic_onehot[0][-3:],f'imgs/results5_instance/{token[0]}_out.jpg')
            save_image(input_mask[0],f'imgs/results5_instance/{token[0]}_input.jpg')
            save_image(semantic_gt[0][1:],f'imgs/results5_instance/{token[0]}_gt.jpg')
            n+=1
            
    print('IOU: ',total_intersects / (total_union +1e-7))
    
    return semantic,semantic_gt

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Multi frame alignment eval.')
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs5')

    # data config
    parser.add_argument('--dataroot', type=str, default='dataset')
    parser.add_argument('--laneInput_dir', type=str, default='dataset/pred_multiframes_box')
    parser.add_argument('--laneGT_dir', type=str, default='dataset/gt_multiframes_box')
    parser.add_argument('--egopose_dir', type=str, default='dataset/egopose.txt')
   
    parser.add_argument('--test_laneInput_dir', type=str, default='dataset/test_pred_box')
    parser.add_argument('--test_laneGT_dir', type=str, default='dataset/test_gt_box')

    
    # model config
    parser.add_argument("--model", type=str, default='BevEncode') 
    parser.add_argument("--pretrained", type=bool, default=True) 
    parser.add_argument("--UNetBackbone", type=str, default='vgg') 
    
    # training config
    parser.add_argument("--nepochs", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default='./runs5/model99.pt')
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

    args = parser.parse_args()
    
    semantic,semantic_gt=eval(args)
    