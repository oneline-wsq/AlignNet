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
        

    def readJsons(self,index):
        # 获得对应token的gt的json文件的路径
        input_json=os.path.join(self.laneInput_dir, self.laneInput[index])
        gt_json=os.path.join(self.laneGT_dir, self.laneInput[index])
        
        with open(input_json,'r') as f:
            input=json.load(f)
        with open(gt_json,'r') as f:
            gt=json.load(f)
        return input, gt

    def lines_to_img(self,lines,egopose,gt=False):
        # 目前是以ego pose为中心,50*50m, 按照 resolution m/像素 来转成图片
        # 转成的图片的宽和高就是：w=50/resolution, h=50/resolution
        # 如果 resolution=0.05, 则 w=1000, h=1000
        patch_x, patch_y=-egopose[0][0], -egopose[0][1] # 车的坐标
        scale_height = self.canvas_h / self.patch_h
        scale_weight = self.canvas_w / self.patch_w
        
        trans_x = -patch_x + self.patch_w / 2.0
        trans_y = -patch_y + self.patch_h / 2.0
        
        # for line in lines:
        # map_mask = np.zeros([self.canvas_h,self.canvas_w,3],np.uint8)
        map_mask = np.zeros([self.canvas_h, self.canvas_w,1],np.uint8) # 将它保存为黑白图像
        color_map={'0':(255,0,0),'1':(0,255,0),'2':(0,0,255)}
        for line in lines:
            # 这里的line为字典数据
            pts,line_type=line['pts'],line['type']
            if len(pts) < 2:
                continue
            # 将pts数据转为line
            new_line = LineString(pts)
            new_line = affinity.affine_transform(new_line,[1.0,0.0,0.0,1.0,trans_x,trans_y])
            new_line = affinity.scale(new_line,xfact=scale_weight,yfact=scale_height,origin=(0, 0))
            # 将线画上去
            coords=np.asarray(list(new_line.coords), np.int32)
            coords=coords.reshape((-1,2))
            # cv2.polylines(map_mask,[coords],False,color=color_map[str(line_type)],thickness=2)
            cv2.polylines(map_mask,[coords],False,color=(255,255,255),thickness=2) # 存储为白线
            
            # map_mask=cv2.cvtColor(map_mask,cv2.COLOR_BGR2RGB) # 转成RGB
            
        if gt:
            # 如果是gt, 则再生成一个通道的label
            # background=~(map_mask[:,:,0] + map_mask[:,:,1] + map_mask[:,:,2])
            background=~(map_mask[:,:,0])
            map_mask=np.stack((background,map_mask[:,:,0]),axis=-1) # (1000,1000,2)
        map_mask=torch.from_numpy(map_mask).permute(2,0,1)/255 # 归一化 [0,1], 才能与PIL读取的数据一致,坐标轴转换 (2,1000,1000)
        return map_mask


    def __getitem__(self, index):
        token=self.laneInput[index].split('.')[0]
        input, gt = self.readJsons(index=index) # 读入json文件
        egopose=self.ego2gs[token]
        # 画图，转成图片
        input_mask=self.lines_to_img(input,egopose) # torch.tensor: [1,1000,1000]
        gt_mask=self.lines_to_img(gt,egopose,gt=True) # [2,1000,1000] ,第0维为背景
        return input_mask, gt_mask, token, egopose

    def __len__(self):
        return len(self.laneInput) 
    
def semantic_dataset(bsz,nworkers,laneInput_dir, laneGT_dir,egopose_dir,resolution=0.05,patch_h=50,patch_w=50):
    train_dataset = LaneData(laneInput_dir, laneGT_dir,egopose_dir,resolution=0.05,patch_h=50,patch_w=50)
    # val_dataset = LaneData(laneInput_dir, laneGT_dir,egopose_dir,resolution=0.05,patch_h=50,patch_w=50)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    return train_loader