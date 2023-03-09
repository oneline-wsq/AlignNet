# 处理并生成训练数据
# 将前后多少帧的数据叠加起来

import os
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
from utils.preprocess_utils import get_proj_mat3
colors_plt = ['#7DB213', '#33A6C3', '#33A6C3']

if __name__=='__main__':
    img_txt = r"dataset/egopose.txt"
    ego2gs = get_proj_mat3(img_txt=img_txt)
    n=0
    root_path = r"imgs/vis"

    gts=os.listdir('dataset/gt')


    for gt_file in gts: 
        with open(f"dataset/gt/{gt_file}", 'r') as f:
            data = json.load(f)

            token=gt_file.split('.')[0]

            fig = plt.figure(figsize=(10, 5))
            ax=fig.add_subplot(1,1,1)
            for i in range(len(data)):
                # 读取里面的每一条线
                pts=data[i]['pts']
                line_type=data[i]['type']
                x=np.array([p[0] for p in pts])
                y=np.array([p[1] for p in pts])

                ax.plot(x[:-1], y[:-1], color=colors_plt[line_type], linewidth=1, linestyle='dotted', alpha=1,
                     markersize=1)
            fig.savefig(f'imgs/gts/{token}.png')  
            plt.clf()
    # fig.savefig('imgs/gts/all.png')




    
