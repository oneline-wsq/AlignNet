from torch.utils.data import Dataset
import os
import json
from utils.preprocess_utils import get_proj_mat3
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely import affinity
import cv2
import torch
from data.rasterize import mask_for_lines
from data.rasterize import preprocess_map
from data.const import NUM_CLASSES
import copy

from torchvision.utils import save_image


class LaneData(Dataset):
    def __init__(self, laneInput_dir, laneGT_dir,egopose_dir,resolution=0.05,patch_h=50,patch_w=50):
        super(LaneData, self).__init__()
        # 初始化
        self.laneInput_dir=laneInput_dir # 输入的车道线的地址
        self.laneGT_dir=laneGT_dir # 对应真值的地址
        self.laneInput=sorted(os.listdir(self.laneInput_dir))
        
        # 读取ego pose
        self.ego2gs=get_proj_mat3(egopose_dir)
        self.resolution=resolution
        self.patch_h=patch_h  # 50m
        self.patch_w=patch_w  # 50m
        # 计算出画布大小
        self.canvas_h=int(self.patch_h/self.resolution)
        self.canvas_w=int(self.patch_w/self.resolution)
        
        self.thickness=2
        self.angle_class=36
        

    def readJsons(self,index):
        # 获得对应token的gt的json文件的路径
        input_json=os.path.join(self.laneInput_dir, self.laneInput[index])
        gt_json=os.path.join(self.laneGT_dir, self.laneInput[index])
        
        with open(input_json,'r') as f:
            input=json.load(f)
        with open(gt_json,'r') as f:
            gt=json.load(f)
        return input, gt
    
    
    def turnOneType(self,vectors):
        new_vectors=copy.deepcopy(vectors)
        for v in new_vectors:
            v['type']=0
        return new_vectors
        
    
    def get_semantic_map(self, vectors, egopose, gt=False):
        patch_x, patch_y=-egopose[0][0], -egopose[0][1] # 车的坐标
        
        """获得instance_masks"""
        # 将所有vectors的type都转成一类
        vectors=self.turnOneType(vectors)
        instance_masks= preprocess_map(vectors, (patch_x, patch_y),(self.patch_h,self.patch_w), (self.canvas_h,self.canvas_w), NUM_CLASSES, self.thickness, self.angle_class)
        
        semantic_masks = instance_masks != 0
        semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks]) # 在前面加上背景的通道
        instance_masks = instance_masks.sum(0) # 所有类别加起来
        # forward_oh_masks = label_onehot_encoding(forward_masks, self.angle_class+1)
        # backward_oh_masks = label_onehot_encoding(backward_masks, self.angle_class+1)
        # direction_masks = forward_oh_masks + backward_oh_masks
        # direction_masks = direction_masks / direction_masks.sum(0)
        
        # 将所有semantic_masks不为0的变为1
        map_mask = np.zeros([self.canvas_h, self.canvas_w,1],np.uint8)
        map_mask[instance_masks!=0]=255
        if gt:
            # 如果是gt, 则再生成一个通道的label
            background=~(map_mask[:,:,0])
            map_mask=np.stack((background,map_mask[:,:,0]),axis=-1) # (1000,1000,2)
        map_mask=torch.from_numpy(map_mask).permute(2,0,1)/255 # 归一化 [0,1], 才能与PIL读取的数据一致,坐标轴转换 (2,1000,1000)
            
        return map_mask, instance_masks#, forward_masks, backward_masks, direction_masks

        
    
    def __getitem__(self, index):
        token=self.laneInput[index].split('.')[0]
        input, gt = self.readJsons(index=index) # 读入json文件
        egopose=self.ego2gs[token]
        # 画图，转成图片
        
        input_mask,_=self.get_semantic_map(input,egopose) # torch.tensor: [1,1000,1000]
        gt_mask,instance_masks=self.get_semantic_map(gt,egopose,gt=True) # [2,1000,1000] ,第0维为背景
        # print('end')
        return input_mask, gt_mask, instance_masks, token, egopose

    def __len__(self):
        return len(self.laneInput) 
    
def semantic_dataset(bsz,nworkers,laneInput_dir, laneGT_dir,egopose_dir,resolution=0.05,patch_h=50,patch_w=50):
    train_dataset = LaneData(laneInput_dir, laneGT_dir,egopose_dir,resolution=0.05,patch_h=50,patch_w=50)
    # val_dataset = LaneData(laneInput_dir, laneGT_dir,egopose_dir,resolution=0.05,patch_h=50,patch_w=50)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    return train_loader