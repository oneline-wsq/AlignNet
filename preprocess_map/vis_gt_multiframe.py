# 处理并生成训练数据
# 将前后多少帧的数据叠加起来

import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)


from utils.preprocess_utils import get_proj_mat3,sample_pts_from_line
colors_plt = ['#7DB213', '#33A6C3', '#F35B1A']


def getAllPoints(idx,gt_file,frame_nums,ego2gs):
    """
    获得前后frame_nums帧上的所有的点
    :param: frame_nums
    :return:
    """
    now_token=gt_file[idx].split('.')[0]
    left_frame_idx=max(0,idx-frame_nums)
    right_frame_idx=min(idx+frame_nums,len(gt_file))

    add_frames=gt_file[left_frame_idx:right_frame_idx] # 将这些帧的数据叠加起来

    add_list=[] # 用来存储所有的线-世界坐标系
    for f_name in add_frames:
        # 读该frame对应的ego pose
        frame_token=f_name.split('.')[0]
        ego2global=ego2gs[frame_token]
        trans, angle= ego2global[0], ego2global[1]
        patch_x, patch_y =trans[0],trans[1]
        angle=angle

        # 读对应的json文件
        with open(f"dataset/gt/{f_name}", 'r') as f:
            data = json.load(f)
        
        # 保存对应的frame帧的数据
        single_fram_data=[]

        for i in range(len(data)):
            # 读取里面的每一条线
            new_line_dict={}
            pts=data[i]['pts']
            line_type=data[i]['type']
            if len(pts) < 2:
                    continue
                    # pts.append([0, 0])
            
            # 转成世界坐标系
            points = LineString(pts)  # 一条线上的所有点
            new_line = affinity.affine_transform(points, [1.0, 0.0, 0.0, 1.0, patch_x, patch_y])  # 将该线转到车身坐标系，平移

            new_line = affinity.rotate(new_line, angle, origin=(patch_x, patch_y),
                                           use_radians=False)  # 将该线转到车身坐标系，旋转
            pix_coords0, pts_num = sample_pts_from_line(new_line)  # 在线上重新取点

            # 存储为新的线
            new_line_dict['pts']=(-pix_coords0).tolist() # 取一个负数
            new_line_dict['pts_num']=pts_num
            new_line_dict['type']=line_type
            single_fram_data.append(new_line_dict)
        

        # # 画转换后的线
        # fig = plt.figure(figsize=(10, 5))
        # ax=fig.add_subplot(1,1,1)

        # # 把ego pose在图中画出来
        # # print("x: ", patch_x, "y: ", patch_y)
        # ax.plot(-patch_x,-patch_y,color='r',marker='*',markersize=10)

        # for i in range(len(single_fram_data)):
        #     # 读取里面的每一条线
        #     pts=single_fram_data[i]['pts']
        #     line_type=single_fram_data[i]['type']
        #     x=np.array([p[0] for p in pts])
        #     y=np.array([p[1] for p in pts])

        #     ax.plot(x[:-1], y[:-1], color=colors_plt[line_type], linewidth=1, alpha=1, markersize=1)
        # fig.savefig(f'imgs/gts_single_world/{frame_token}.png')  

        # 加到add_list中
        add_list+=single_fram_data # 将单帧的加到上面
    return add_list



if __name__=='__main__':
    img_txt = r"dataset/egopose.txt"
    ego2gs = get_proj_mat3(img_txt=img_txt)
    n=0
    root_path = r"imgs/gts_multiframe"

    gts=sorted(os.listdir('dataset/gt'))

    # egopose_dict={}
    
    for idx, gt_file in enumerate(gts): 
        if n<800:
            n+=1
            continue
        token=gt_file.split('.')[0]
        if token!=gts[idx].split('.')[0]:
             print('error!')
             break
        add_list=getAllPoints(idx,gts,frame_nums=50,ego2gs=ego2gs)

        # 将add_list中的数据存到对应json文件中
        json_path=f'dataset/gt_multiframes/{token}.json'
        json_file = open(json_path,mode='w')
        json.dump(add_list,json_file,indent=4)
        print(f'{n}-{token}')

        
        # 得到该token的ego pose
        ego2global=ego2gs[token]
        trans, angle= ego2global[0], ego2global[1]
        patch_x, patch_y =trans[0],trans[1]
        # egopose_dict[token]=ego2global

        # # """画图"""
        # fig = plt.figure(figsize=(10, 5))
        # ax=fig.add_subplot(1,1,1)

        # # 把ego pose在图中画出来
        # print("x: ", -patch_x, "y: ", -patch_y)
        # ax.plot(-patch_x,-patch_y,color='r',marker='*',markersize=10)

        # for i in range(len(add_list)):
        #     # 读取里面的每一条线
        #     pts=add_list[i]['pts']
        #     line_type=add_list[i]['type']
        #     x=np.array([p[0] for p in pts])
        #     y=np.array([p[1] for p in pts])

        #     ax.plot(x, y, color=colors_plt[line_type], linewidth=1, alpha=1, markersize=1)

        # fig.savefig(f'{root_path}/{token}.png')  
        # plt.clf()
        # plt.close()

        
        n+=1

    # 保存所有token对应的ego pose 
    # json_path=f'dataset/gt_multiframes_egopose.json'
    # json_file = open(json_path,mode='w')
    # json.dump(egopose_dict,json_file,indent=4)
    
    print('end')

        



    
