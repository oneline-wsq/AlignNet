import numpy as np
import torch
import torch.nn as nn
from postprocess.cluster import LaneNetPostProcessor
from postprocess.connect import sort_points_by_dist, connect_by_direction
import cv2

def onehot_encoding(logits, dim=0):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

def plotmask(mask):
    label=np.unique(mask)
    
    sub = np.zeros(shape=[mask.shape[0], mask.shape[1], 3], dtype=np.uint8)
    for i in range(1,label.shape[0]):
        idx=np.where(mask==i)
        c1=np.random.randint(0,255)
        c2=np.random.randint(0,255)
        c3=np.random.randint(0,255)
        sub[idx][0]=c1
        sub[idx][1]=c2
        sub[idx][2]=c3
    cv2.imwrite('mask.jpg',sub)

        

        

def vectorize(segmentation,embedding=None):
    # 返回图上的坐标
    segmentation = segmentation.softmax(0) # [4,200,400]
    oh_pred = onehot_encoding(segmentation).cpu().numpy() # onehot的预测
    
    max_pool_1 = nn.MaxPool2d((1, 5), padding=(0, 2), stride=1) # kernel_size=(1,5)
    avg_pool_1 = nn.AvgPool2d((9, 5), padding=(4, 2), stride=1)
    max_pool_2 = nn.MaxPool2d((5, 1), padding=(2, 0), stride=1)
    avg_pool_2 = nn.AvgPool2d((5, 9), padding=(2, 4), stride=1)
    post_processor=LaneNetPostProcessor(dbscan_eps=1.5, postprocess_min_samples=10)
    
    confidences = [] 
    line_types = [] 
    simplified_coords = []
    for i in range(1, oh_pred.shape[0]):  # shape[0]为 通道数=4, oh_pred:[4,200,400]
        single_mask = oh_pred[i].astype('uint8') # 第i类的mask (200,400)
        single_class_inst_mask, single_class_inst_coords = post_processor.postprocess(single_mask)
        
        # plotmask(single_class_inst_mask)
        num_inst = len(single_class_inst_coords)
        
        prob = segmentation[i] 
        prob[single_class_inst_mask == 0] = 0
        
        nms_mask_1 = ((max_pool_1(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
        avg_mask_1 = avg_pool_1(prob.unsqueeze(0))[0].cpu().numpy()
        nms_mask_2 = ((max_pool_2(prob.unsqueeze(0))[0] - prob) < 0.0001).cpu().numpy()
        avg_mask_2 = avg_pool_2(prob.unsqueeze(0))[0].cpu().numpy()
        vertical_mask = avg_mask_1 > avg_mask_2
        horizontal_mask = ~vertical_mask
        nms_mask = (vertical_mask & nms_mask_1) | (horizontal_mask & nms_mask_2)
        
        for j in range(1, num_inst + 1): # 遍历所有的类
            full_idx = np.where((single_class_inst_mask == j))
            full_lane_coord = np.vstack((full_idx[1], full_idx[0])).transpose()
            confidence = prob[single_class_inst_mask == j].mean().item() # 置信度的计算

            idx = np.where(nms_mask & (single_class_inst_mask == j))
            if len(idx[0]) == 0:
                continue
            lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

            # range_0 = np.max(full_lane_coord[:, 0]) - np.min(full_lane_coord[:, 0])
            # range_1 = np.max(full_lane_coord[:, 1]) - np.min(full_lane_coord[:, 1])
            
            # if range_0 > range_1:
            #     lane_coordinate = sorted(lane_coordinate, key=lambda x: x[0])
            # else:
            #     lane_coordinate = sorted(lane_coordinate, key=lambda x: x[1])

            lane_coordinate = np.stack(lane_coordinate)
            # lane_coordinate = sort_points_by_dist(lane_coordinate)
            lane_coordinate = lane_coordinate.astype('int32')
            # lane_coordinate = connect_by_direction(lane_coordinate, direction, step=7, per_deg=360 / angle_class)

            simplified_coords.append(lane_coordinate)
            confidences.append(confidence)
            line_types.append(i-1)  # 第 i 类
            
        
    return simplified_coords, confidences, line_types