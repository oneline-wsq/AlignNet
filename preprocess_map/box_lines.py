# 框出一定范围内的所有的线
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
import json
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, box
from utils.preprocess_utils import sample_pts_from_line, get_proj_mat3
import matplotlib.pyplot as plt
import numpy as np


colors_plt = ['#7DB213', '#33A6C3', '#F35B1A']
# 0-绿色，1-蓝色，2-橙色

if __name__=='__main__':
    # 读取每个ego pose对应的多帧数据
    json_rootdir='dataset/gt_multiframes'
    json_names=sorted(os.listdir(json_rootdir))

    # 读取对应的ego pose
    ego_dir='dataset/egopose.txt'
    
    ego2gs = get_proj_mat3(img_txt=ego_dir)

    n=0
    for j in json_names:
        n+=1
        token=j.split('.')[0]

        print(n, '-', token)

        # 读取json文件
        with open(f"{json_rootdir}/{j}", 'r') as f:
            vectors = json.load(f)
        
        # 找到对应的ego pose
        ego=ego2gs[token]
        trans=(-np.array(ego[0])).tolist() # 是否要加正负
        
        # 以ego pose为中心，左右距离25m，画一个矩形
        dis=25 # 单位m
        
        rect=box(trans[0]-dis,trans[1]-dis,trans[0]+dis,trans[1]+dis)

        # 遍历每一条line, 转成 lineString 数据类型
        frame_lists=[]
        for vector in vectors:
            pts=vector['pts']
            if len(pts) < 2:
                continue
            line=LineString(pts)
            new_line=line.intersection(rect)
            if not new_line.is_empty:
                # 重新采样，保存新的lines
                new_line_dict={}
                pix_coords0, pts_num = sample_pts_from_line(new_line)
                new_line_dict['pts']=pix_coords0.tolist() # 取一个负数
                new_line_dict['pts_num']=pts_num
                new_line_dict['type']=vector['type']
                frame_lists.append(new_line_dict) 
        
        # 将新的线保存到json文件中
        json_path=f'dataset/gt_multiframes_box/{token}.json'
        json_file = open(json_path,mode='w')
        json.dump(frame_lists,json_file,indent=4)

        # 画图看一下
        fig = plt.figure(figsize=(10, 5))
        ax=fig.add_subplot(1,1,1)

        # 把ego pose在图中画出来
        ax.plot(trans[0],trans[1],color='r',marker='*',markersize=10)

        for i in range(len(frame_lists)):
            # 读取里面的每一条线
            pts=frame_lists[i]['pts']
            line_type=frame_lists[i]['type']
            x=np.array([p[0] for p in pts])
            y=np.array([p[1] for p in pts])

            ax.plot(x, y, color=colors_plt[line_type], linewidth=1, alpha=1, markersize=1)

        fig.savefig(f'imgs/gt_multiframe_box/{token}.png')  
        plt.clf()
        plt.close()


        

        