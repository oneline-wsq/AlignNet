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
        # print(self.laneInput[index])
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
    
    def Two2Divider(self,instance_masks):
        # idx=torch.nonzero(instance_masks[0,:,:].float() + instance_masks[2,:,:].float())
        divider_mask=instance_masks[0,:,:].bool()
        boundary_mask=instance_masks[2,:,:].bool()
        mask=divider_mask & boundary_mask
        idx=torch.where(mask==True)
        
        if len(idx[0])==0:
            return instance_masks
        
        # print(len(idx[0]))
        new_boundary=instance_masks[2,:,:]
        new_boundary[idx]=0
        instance_masks[2,:,:]=new_boundary
        return instance_masks
    
    def reGenerateIM(self,sm):
        # 分通道分别计算
        idx = 1 # idx从1开始
        new_IM=[]

        # 通过opencv找到联通关系
        for i in range(NUM_CLASSES):
            num_objects, labels=cv2.connectedComponents(sm[i].numpy().astype(np.uint8))
            newmask=np.zeros((labels.shape[0],labels.shape[1]),np.uint8)
            for cnum in range(1,num_objects):
                newmask[labels==cnum]=idx
                idx+=1
            new_IM.append(newmask) 
        # print('idx: ',idx-1)
        new_IM=np.stack(new_IM)
        
        return torch.tensor(new_IM)
            
    def plotInstance(self,new_IM):
        # 找到instance mask中的最大值
        IM=new_IM.sum(0).numpy() # 相加，转成np array数组
        m=np.max(IM)
        # print('IM中的最大值：',m)
        output=np.zeros((IM.shape[0],IM.shape[1],3),np.uint8)
        colors=[(255,0,0),(255,127,30),(255,255,30),(0,255,0),(0,0,255),(255,0,213)]
        for i in range(1,m+1):
            mask=IM==i
            print('\n',m)
            output[:,:,0][mask] = np.random.randint(0,255) # colors[(i-1)%m][0]
            output[:,:,1][mask] = np.random.randint(0,255)
            output[:,:,2][mask] = np.random.randint(0,255)
        cv2.imwrite(f'instance_mask_{self.token}.jpg',output)
        cv2.destroyAllWindows()
   

    def get_semantic_map(self, vectors, egopose, gt=False):
        patch_x, patch_y=-egopose[0][0], -egopose[0][1] # 车的坐标
        
        """获得instance_masks"""
        # 将所有vectors的type都转成一类
        # vectors=self.turnOneType(vectors)

        instance_masks= preprocess_map(vectors, (patch_x, patch_y),(self.patch_h,self.patch_w), (self.canvas_h,self.canvas_w), NUM_CLASSES, self.thickness, self.angle_class)
        

        semantic_masks = instance_masks != 0
        
        if gt:
            semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks]) # 在前面加上背景的通道

        semantic_masks=semantic_masks.float()
        
        # if gt:
        #     sm=semantic_masks[-3:]
        #     save_image(sm,'sm.jpg')
        #     sm_divider=semantic_masks[-3]
        #     save_image(sm_divider,'sm_divider.jpg')
        #     sm_boundary=semantic_masks[-1]
        #     save_image(sm_boundary,'sm_boundary.jpg')
            # print('end')

        # 根据联通关系，重新生成instance_masks
        if gt:
            instance_masks=self.reGenerateIM(semantic_masks[-3:])
            # self.plotInstance(instance_masks)
        instance_masks = instance_masks.sum(0) # 所有类别加起来
        

        return semantic_masks, instance_masks#, forward_masks, backward_masks, direction_masks

        
    
    def __getitem__(self, index):
        self.token=self.laneInput[index].split('.')[0]
        # print(token)
        input, gt = self.readJsons(index=index) # 读入json文件
        egopose=self.ego2gs[self.token]
        # 画图，转成图片
        
        input_mask,_=self.get_semantic_map(input,egopose) # torch.tensor: [1,1000,1000]
        gt_mask,instance_masks=self.get_semantic_map(gt,egopose,gt=True) # [2,1000,1000] ,第0维为背景
        # print('end')
        return input_mask, gt_mask, instance_masks, self.token, egopose

    def __len__(self):
        return len(self.laneInput) 
    
def semantic_dataset(bsz,nworkers,laneInput_dir, laneGT_dir,egopose_dir,resolution=0.05,patch_h=50,patch_w=50,isshuffle=True):
    train_dataset = LaneData(laneInput_dir, laneGT_dir,egopose_dir,resolution=0.05,patch_h=50,patch_w=50)
    # val_dataset = LaneData(laneInput_dir, laneGT_dir,egopose_dir,resolution=0.05,patch_h=50,patch_w=50)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=isshuffle, num_workers=nworkers, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    return train_loader