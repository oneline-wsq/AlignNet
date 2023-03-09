#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""

import cv2
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneFeat(object):
    """

    """
    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self, dbscan_eps=0.35, postprocess_min_samples=200):
        """

        """
        self.dbscan_eps = dbscan_eps
        self.postprocess_min_samples = postprocess_min_samples

    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        from sklearn.cluster import MeanShift

        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.postprocess_min_samples)
        # db = MeanShift()
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err:
            # print(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        # cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            # 'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255)
        lane_embedding_feats = instance_seg_ret[idx]

        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result=None):
        """

        :param binary_seg_result: (1000, 1000)
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1]], dtype=np.int) #(200,400)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates'] # 坐标

        if db_labels is None:
            return None, None

        lane_coords = []

        # for index in range(1,2): 
        #     # 一共两类
        #     idx = np.nonzero(binary_seg_result)
            
        #     pix_coord_idx = tuple((coord[:, 1], coord[:, 0]))
        #     mask[pix_coord_idx] = index
        #     lane_coords.append(coord)

        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx=np.where(db_labels==label)
            pix_coord_idx = tuple((coord[idx][:,1], coord[idx][:,0]))
            mask[pix_coord_idx] = label+1
            lane_coords.append(coord[idx]) # 将所有属于label的放在一起

        return mask, lane_coords

def detectLines(instance_mask):
    visible_img = np.expand_dims(instance_mask,-1).repeat(3,axis=-1)
    # 当前连通域的mask, 找到该mask上的所有直线
    minLineLength=5
    maxLineGap=100
    lines=cv2.HoughLinesP(instance_mask,1,np.pi/180,100,minLineLength,maxLineGap)
    
    cv2.imwrite('instance_mask.jpg',instance_mask)
    slopes=[]
    intercepts=[]
    lineParams=[]
    if not isinstance(lines, type(None)):
        for line in lines:
            for x1,y1, x2, y2 in line:
                color1=np.random.randint(0,255)
                color2=np.random.randint(0,255)
                color3=np.random.randint(0,255)
                cv2.line(visible_img,(x1,y1),(x2,y2),(color1,color2,color3),2)
                # 计算斜率和截距
                slope, intercept=np.polyfit([x1,x2],[y1,y2],1)
                slopes.append(slope)
                intercepts.append(intercept)
                lineParams.append([slope,intercept])
        
        # 判断是否slope是否接近 
        newSlopes=[slopes[0]] 
        newLineParams=[lineParams[0]]
        for si in range(1,len(slopes)):
            s=[slopes[si] for j in range(si)] 
            ic=[intercepts[si] for j in range(si)]
            beforS=slopes[:si]
            beforeI=intercepts[:si]
            # 计算两两之间的距离
            dis=[abs(ss-bs) for ss, bs in zip(s,beforS)]
            intercept_dis=[abs(cc-bc) for cc, bc in zip(ic,beforeI)]
            shouldIN=True
            if abs(slopes[si])<1:
                for id in range(len(dis)):
                    dd=dis[id]
                    if dd<0.05 and intercept_dis[id]<30:
                        shouldIN=False
            else:
                # 大于1，说明有可能是竖的线
                for dd in dis:
                    if dd<3:
                        shouldIN=False

            if shouldIN:
                newSlopes.append(slopes[si])
                newLineParams.append(lineParams[si])
    else:
        newSlopes=[]
        newLineParams=[]

    cv2.imwrite('straight_lines.jpg',visible_img)

    # 判断直线的斜率，如果斜率近似，

    return newSlopes, newLineParams


def findInstance(morphological_ret, labels):
    lane_coords=[] # 找到所有的
    mask = np.zeros(shape=[morphological_ret.shape[0], morphological_ret.shape[1]], dtype=np.int)
    
    # 找到labels对应的索引
    uniqueLabel=np.unique(labels)
    labelIDX=1
    for i in uniqueLabel:
        if i==0:
            continue
    
        idx=np.where(labels==i)
        lane_coordinate = np.vstack((idx[1],idx[0])).transpose()

        """ 直线检测，对于每个联通域，进行直线检测"""
        instance_mask = np.zeros(shape=[morphological_ret.shape[0], morphological_ret.shape[1]], dtype=np.uint8)
        # 将label==i的位置置为255
        instance_mask[idx]=255
        slops, LineParams=detectLines(instance_mask)
    
        """按照检测到的直线数量对点进行划分"""
        line_nums=len(slops)

        if line_nums<=1:
            # 说明只有一条或者一条都没检测出来（曲线）
            
            mask[idx]=labelIDX
            labelIDX+=1
            lane_coords.append(lane_coordinate)
        else:
            # 有条直线，就新建几个list
            subline=[[] for _ in range(line_nums)]

            for coord in lane_coordinate:
                x=coord[0]
                y=coord[1]
                dis=[abs(y-(LineParams[pi][0]*x+LineParams[pi][1])) for pi in range(line_nums)]
                min_idx=dis.index(min(dis))
                subline[min_idx].append(coord)  

            new_subline=[np.asarray(subline[ss]) for ss in range(line_nums)]
            for ss in range(line_nums):
                s_idx=new_subline[ss]
                for s_coord in s_idx:
                    mask[s_coord[1],s_coord[0]]=labelIDX
                labelIDX+=1
            
            lane_coords+=new_subline
        assert labelIDX-1==len(lane_coords)  
    return mask, lane_coords


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self, dbscan_eps=0.35, postprocess_min_samples=200):
        """

        :param ipm_remap_file_path: ipm generate file path
        """

        self._cluster = _LaneNetCluster(dbscan_eps, postprocess_min_samples)

    def postprocess(self, binary_seg_result, instance_seg_result=None, min_area_threshold=100):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area
        # morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)
        morphological_ret=binary_seg_result
        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        labels = connect_components_analysis_ret[1] # 每一个联通区域的mask,背景为0
        stats = connect_components_analysis_ret[2] # 联通区域外接矩形对应的参数
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold: # 面积/像素个数
                idx = np.where(labels == index)
                morphological_ret[idx] = 0
        # 上述步骤主要是将较小的区域去除掉
        # apply embedding features cluster
        # mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
        #     binary_seg_result=morphological_ret,
        #     instance_seg_result=instance_seg_result
        # )
        
        # 直接在得到的图像上通过连通域划分instance
        mask_image, lane_coords=findInstance(morphological_ret,labels)

        return mask_image, lane_coords
