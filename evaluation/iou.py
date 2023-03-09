import torch


def get_batch_iou(pred_map, gt_map):
    intersects = []
    unions = []
    with torch.no_grad():
        pred_map = pred_map.bool() # 转成bool类型
        gt_map = gt_map.bool()

        for i in range(pred_map.shape[1]): # 按照通道数遍历
            pred = pred_map[:, i] 
            tgt = gt_map[:, i]
            intersect = (pred & tgt).sum().float() # 计算相交 - Intersect 
            union = (pred | tgt).sum().float() # 计算合并 - union
            intersects.append(intersect)  # 按照通道数（不同类别) append
            unions.append(union)
    return torch.tensor(intersects), torch.tensor(unions)
