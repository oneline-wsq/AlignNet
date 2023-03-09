import os
from sklearn.cluster import DBSCAN
import json
import numpy as np
import matplotlib.pyplot as plt

colors_plt = ['#FF0000', '#00FF00', '#0000FF']
laneType_dict={'divider':0,'ped_crossing':1,'boundary':2}
# 红，绿，蓝
#     'divider': 0,
#     'ped_crossing': 1,
#     'boundary': 2,
#     'others': -1

class clusterLine():
    def __init__(self, json_name):
        # 读入json文件
        with open(json_name,'r',encoding='utf8') as fp:
            self.json_data=json.load(fp)
    
        self.token=json_name.split('/')[-1].split('.')[0]
    def getEachType(self):
        # 获得每一种lane的所有坐标点
        eachtype={'divider':[],'boundary':[],'ped_crossing':[]}
        for line in self.json_data:
            pts=line['pts']
            pts_num=line['pts_num']
            line_type=line['type']
            if line_type==0:
                # divider
                eachtype['divider']+=pts
            elif line_type==1:
                # ped_crossing
                eachtype['ped_crossing']+=pts
            elif line_type==2:
                # boundary
                eachtype['boundary']+=pts
        return eachtype

    def clusterEachType(self, eachtype, eps=0.5, min_samples=10):
        # 首先将每一类分出来
        lines=[]
        for t, points in eachtype.items():
            # 对每一类别单独进行聚类
            pts=np.array(points)
            if not len(pts):
                continue
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
            db_labels=clustering.labels_
            unique_labels=np.unique(db_labels)
            num_clusters=len(unique_labels)
            # 一个聚类的label当成一条新的线
            for index, label in enumerate(unique_labels.tolist()):
                # 新建一个dict用来存储聚类后的线
                line_dict={}
                if label == -1:
                    continue
                idx=np.where(db_labels==label)
                sub_pts=pts[idx]
                line_dict['pts']=sub_pts.tolist()
                line_dict['pts_num']=len(sub_pts)
                line_dict['type']=laneType_dict[t]
                lines.append(line_dict) 
        return lines
                

    def plotEachType(self,eachtype,save_path='imgs/gt_differentType'):
        fig = plt.figure(figsize=(10, 5))
        ax=fig.add_subplot(1,1,1)

        for k,v in eachtype.items():
            pts=v
            x=np.array([p[0] for p in pts])
            y=np.array([p[1] for p in pts])
            ax.scatter(x, y, color=colors_plt[laneType_dict[k]], linewidth=1, alpha=0.5)
        img_name=os.path.join(save_path,f'{self.token}.png')
        fig.savefig(img_name)
    
    def plotLines(self, lines, save_path='imgs/gt_differentType'):
        fig = plt.figure(figsize=(10, 5))
        ax=fig.add_subplot(1,1,1)

        for line in lines:
            pts=line['pts']
            pts_num=line['pts_num']
            line_type=line['type']

            x=np.array([p[0] for p in pts])
            y=np.array([p[1] for p in pts])
            # if line_type==2:
            ax.plot(x, y, color=colors_plt[line_type], linewidth=1, alpha=0.5, markersize=1)
        img_name=os.path.join(save_path,f'{self.token}.png')
        fig.savefig(img_name)

if __name__=='__main__':
    json_name='dataset/gt_multiframes_box/1644541391216617.json'
    CL=clusterLine(json_name)
    eachtype=CL.getEachType()
    # CL.plotEachType(eachtype)
    lines=CL.clusterEachType(eachtype,eps=0.2,min_samples=5)
    CL.plotLines(lines)
    print('end')