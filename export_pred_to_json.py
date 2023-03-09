import argparse
import mmcv
import tqdm
import torch

from data.dataset_2 import semantic_dataset
from data.const import NUM_CLASSES
from model import get_model
from postprocess.vectorize import vectorize
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def turnToWorld(args, egopose, coordinate):
    resoluton=args.resolution
    patch_x, patch_y=-egopose[0][0], -egopose[0][1] # 车的坐标
        
    trans_x = -patch_x + args.patch_w / 2.0
    trans_y = -patch_y + args.patch_h / 2.0
    
    xs=coordinate[:,0]
    ys=coordinate[:,1]
    
    wxs=[]
    wys=[]

    for x,y in zip(xs,ys):
        wx=x*resoluton - trans_x
        wy=y*resoluton - trans_y
        wxy=[float(wx[0]),float(wy[0])]
        wxs.append(float(wx[0]))
        wys.append(float(wy[0]))
    new_coords=[wxs,wys]
        
    return new_coords
    
    

def export_to_json(model, val_loader,  args):
    submission = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_external": False,
            "vector": True,
        },
        "results": {}
    }

    model.eval()
    with torch.no_grad():
        for batchi, (input_mask, semantic_gt, instance_gt, token , egopose) in enumerate(val_loader):
            torch.cuda.empty_cache()
            segmentation, _ = model(input_mask.cuda()) 
            for si in range(segmentation.shape[0]): # [bsz,4,200,400]
                coords, confidences, line_types = vectorize(segmentation[si])
                vectors = []
                
                for coord, confidence, line_type in zip(coords, confidences, line_types):
                    worldCoord=turnToWorld(args, egopose, coord)
                    vector = {'pts': worldCoord, 'pts_num': len(coord), "type": line_type, "confidence_level": confidence}
                    vectors.append(vector)
                # rec = val_loader.dataset.samples[batchi * val_loader.batch_size + si]
                submission['results'][batchi * val_loader.batch_size + si] = vectors
                print(batchi * val_loader.batch_size + si)
    
    mmcv.dump(submission, args.output)


def main(args):
    model = get_model(args.model, args.inC, args.outC, args.instance_seg, args.embedding_dim, args.pretrained,args.UNetBackbone)
    weights = torch.load(args.modelf)
    weights_dict={}
    for k,v in weights.items():
        new_k = k.replace('module.','') if 'module' in k else k
        weights_dict[new_k]=v
    model.load_state_dict(weights_dict,strict=True)
    model.cuda()
    # model=torch.nn.DataParallel(model, device_ids=[0,1]) # 这里的device_ids为新标号后的序号，不是物理意义上的序号
    model.eval()
    
    val_loader = semantic_dataset(args.bsz, args.nworkers,args.test_laneInput_dir, \
                                args.test_laneGT_dir, args.egopose_dir, \
                            args.resolution,args.patch_h,args.patch_w,isshuffle=False)
    
    export_to_json(model, val_loader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi frame alignment eval.')
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs3_withoutInstance')

    # data config
    parser.add_argument('--dataroot', type=str, default='dataset')
    parser.add_argument('--laneInput_dir', type=str, default='dataset/pred_multiframes_box')
    parser.add_argument('--laneGT_dir', type=str, default='dataset/gt_multiframes_box')
    parser.add_argument('--egopose_dir', type=str, default='dataset/egopose.txt')
   
    parser.add_argument('--test_laneInput_dir', type=str, default='dataset/vectorize_pred_box')
    parser.add_argument('--test_laneGT_dir', type=str, default='dataset/vectorize_gt_box')

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
    parser.add_argument('--modelf', type=str, default='runs3_withoutInstance/model99.pt')

    # data config
    parser.add_argument("--resolution", type=float,default=0.05)
    parser.add_argument("--patch_h", type=float,default=50)
    parser.add_argument("--patch_w", type=float,default=50)
    parser.add_argument("--inC", type=int,default=3)
    parser.add_argument("--outC", type=int,default=4) # 包含背景类

    # embedding config
    parser.add_argument('--instance_seg', action='store_true',default=False)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # output
    parser.add_argument("--output", type=str, default='output_vectorize.json')

    args = parser.parse_args()
    
    main(args)
